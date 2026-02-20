import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Détecteur de Fractures", page_icon="🦴")

# --- 1. DÉFINITION DE L'ARCHITECTURE (Doit être identique au notebook) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.classifier(x)
        return x

# --- 2. CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_trained_model():
    model = SimpleCNN()
    # Charger sur CPU pour la compatibilité cloud/locale
    state_dict = torch.load('bone_fracture_cnn.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model = load_trained_model()
    class_names = ['Fracturé', 'Non Fracturé'] # Ordre alphabétique des dossiers
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# --- 3. INTERFACE UTILISATEUR ---
st.title("🦴 Analyse de Radiographies Osseuses")
st.write("Système d'aide au diagnostic par Deep Learning.")

uploaded_file = st.file_uploader("Chargez une image de radio (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Affichage de l'image
    image