import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="Détecteur de Fractures ResNet", page_icon="🦴")

# --- 1. CHARGEMENT DU MODÈLE RESNET18 ---
@st.cache_resource
def load_trained_model():
    # On crée l'architecture ResNet18 identique à ton notebook
    model = models.resnet18(weights=None) # Pas besoin des poids ImageNet
    num_ftrs = model.fc.in_features
    # On remplace la dernière couche pour tes 2 classes
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Chargement des poids sauvegardés
    state_dict = torch.load('fracture_cnn.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model = load_trained_model()
    class_names = ['Fracturé', 'Non Fracturé']
except Exception as e:
    st.error(f"Erreur de compatibilité : {e}")
    st.info("Note : Le fichier .pth détecté appartient à un ResNet18.")
    st.stop()

# --- 2. INTERFACE STREAMLIT ---
st.title("🦴 Analyse de Radio (ResNet18)")

uploaded_file = st.file_uploader("Chargez une radio...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image à analyser", use_container_width=True)
    
    # Prétraitement
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)

    # Prédiction
    with st.spinner('Analyse par ResNet18...'):
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prob, predicted = torch.max(probabilities, 1)
            
            label = class_names[predicted.item()]
            confiance = prob.item() * 100

    if label == 'Fracturé':
        st.error(f"**Résultat : {label} ({confiance:.2f}%)**")
    else:
        st.success(f"**Résultat : {label} ({confiance:.2f}%)**")
