#!/usr/bin/env python3
"""
Script de chargement automatique du modèle robuste avec scaler
Généré automatiquement - Plus besoin de normalisation manuelle !
"""

import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class RobustGapPredictor(nn.Module):
    """Modèle robuste pour prédiction du gap."""
    
    def __init__(self, input_size=600, dropout_rate=0.2):
        super(RobustGapPredictor, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

def load_complete_model():
    """
    Charge le modèle complet avec son scaler.
    
    Returns:
        tuple: (model, scaler, metadata)
    """
    # Charger le modèle
    model = RobustGapPredictor(input_size=600)
    checkpoint = torch.load('robust_gap_model_with_scaler.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Charger le scaler
    with open('input_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Charger les métadonnées
    import json
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return model, scaler, metadata

def predict_gap(intensity_profile):
    """
    Prédit le gap pour un profil d'intensité donné.
    AUCUNE NORMALISATION MANUELLE NÉCESSAIRE !
    
    Args:
        intensity_profile (np.array): Profil d'intensité brut (jusqu'à 1000 points)
        
    Returns:
        float: Gap prédit en µm
    """
    model, scaler, metadata = load_complete_model()
    
    # Préprocessing automatique
    if len(intensity_profile) > 600:
        intensity_profile = intensity_profile[:600]
    elif len(intensity_profile) < 600:
        intensity_profile = np.pad(intensity_profile, (0, 600 - len(intensity_profile)), 'constant')
    
    # Normalisation AUTOMATIQUE avec le scaler d'entraînement
    profile_scaled = scaler.transform(intensity_profile.reshape(1, -1))
    
    # Prédiction
    with torch.no_grad():
        input_tensor = torch.FloatTensor(profile_scaled)
        gap_pred = model(input_tensor).item()
    
    return gap_pred

if __name__ == "__main__":
    print("🧪 Test du modèle chargé...")
    model, scaler, metadata = load_complete_model()
    print(f"✅ Modèle chargé avec succès !")
    print(f"   Source: {metadata['source_model']}")
    print(f"   Scaler: {metadata['scaler_type']}")
    print(f"   Plus besoin de normalisation manuelle !")
