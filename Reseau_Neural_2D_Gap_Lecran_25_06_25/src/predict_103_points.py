#!/usr/bin/env python3
"""
Prédiction avec données de 103 points

Auteur: Oussama GUELFAA
Date: 25/06/2025

Ce script adapte les données de 103 points pour les utiliser avec le modèle
entraîné sur 600 points. Plusieurs stratégies d'adaptation sont proposées.
"""

import numpy as np
import torch
import joblib
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import argparse

# Importer le modèle
from kfold_validation.Train_KFold import ImprovedDualParameterNet

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAdapter:
    """Classe pour adapter les données de 103 points vers 600 points."""
    
    def __init__(self):
        self.original_length = 103
        self.target_length = 600
        
    def method_1_interpolation_linear(self, data_103):
        """Méthode 1: Interpolation linéaire simple."""
        x_original = np.linspace(0, 1, self.original_length)
        x_target = np.linspace(0, 1, self.target_length)
        
        f_linear = interpolate.interp1d(x_original, data_103, kind='linear')
        data_600 = f_linear(x_target)
        
        return data_600
    
    def method_2_interpolation_cubic(self, data_103):
        """Méthode 2: Interpolation cubique (plus lisse)."""
        x_original = np.linspace(0, 1, self.original_length)
        x_target = np.linspace(0, 1, self.target_length)
        
        f_cubic = interpolate.interp1d(x_original, data_103, kind='cubic')
        data_600 = f_cubic(x_target)
        
        return data_600
    
    def method_3_spline_smoothing(self, data_103, smoothing_factor=0.1):
        """Méthode 3: Spline avec lissage."""
        x_original = np.linspace(0, 1, self.original_length)
        x_target = np.linspace(0, 1, self.target_length)
        
        tck = interpolate.splrep(x_original, data_103, s=smoothing_factor)
        data_600 = interpolate.splev(x_target, tck)
        
        return data_600
    
    def method_4_padding_interpolation(self, data_103):
        """Méthode 4: Padding + interpolation hybride."""
        x_original = np.linspace(0, 1, self.original_length)
        x_intermediate = np.linspace(0, 1, 300)
        
        f_cubic = interpolate.interp1d(x_original, data_103, kind='cubic')
        data_300 = f_cubic(x_intermediate)
        
        padding_needed = self.target_length - 300
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        
        data_600 = np.pad(data_300, (pad_left, pad_right), mode='edge')
        
        return data_600
    
    def method_5_fourier_reconstruction(self, data_103):
        """Méthode 5: Reconstruction par transformée de Fourier."""
        fft_data = np.fft.fft(data_103)
        
        fft_padded = np.zeros(self.target_length, dtype=complex)
        
        n_copy = min(len(fft_data) // 2, self.target_length // 2)
        fft_padded[:n_copy] = fft_data[:n_copy]
        fft_padded[-n_copy:] = fft_data[-n_copy:]
        
        data_600 = np.real(np.fft.ifft(fft_padded))
        data_600 = data_600 * (self.target_length / self.original_length)
        
        return data_600
    
    def compare_methods(self, data_103):
        """Compare toutes les méthodes d'adaptation."""
        methods = {
            'linear': self.method_1_interpolation_linear(data_103),
            'cubic': self.method_2_interpolation_cubic(data_103),
            'spline': self.method_3_spline_smoothing(data_103),
            'padding': self.method_4_padding_interpolation(data_103),
            'fourier': self.method_5_fourier_reconstruction(data_103)
        }
        
        return methods

def load_kfold_model():
    """Charge le modèle K-Fold entraîné."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImprovedDualParameterNet(input_size=600).to(device)
    checkpoint = torch.load("../models/dual_parameter_model_kfold.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    input_scaler = joblib.load("../models/input_scaler_kfold.pkl")
    gap_scaler = joblib.load("../models/gap_scaler_kfold.pkl")
    L_ecran_scaler = joblib.load("../models/L_ecran_scaler_kfold.pkl")
    
    logger.info(f"✅ Modèle K-Fold chargé")
    
    return model, input_scaler, gap_scaler, L_ecran_scaler, device

def predict_with_adapted_data(model, input_scaler, gap_scaler, L_ecran_scaler, device, data_600):
    """Fait une prédiction avec des données adaptées à 600 points."""
    data_filtered = gaussian_filter1d(data_600, sigma=0.5)
    data_scaled = input_scaler.transform(data_filtered.reshape(1, -1))
    
    with torch.no_grad():
        data_tensor = torch.FloatTensor(data_scaled).to(device)
        prediction_scaled = model(data_tensor)
        
        gap_pred = gap_scaler.inverse_transform(
            prediction_scaled[0, 0].cpu().numpy().reshape(-1, 1)
        )[0, 0]
        L_ecran_pred = L_ecran_scaler.inverse_transform(
            prediction_scaled[0, 1].cpu().numpy().reshape(-1, 1)
        )[0, 0]
    
    return gap_pred, L_ecran_pred

def main():
    """Fonction principale pour tester avec 103 points."""
    parser = argparse.ArgumentParser(description='Prédiction avec données de 103 points')
    parser.add_argument('--data', type=str, help='Données séparées par des virgules ou fichier')
    parser.add_argument('--method', type=str, default='cubic', 
                       choices=['linear', 'cubic', 'spline', 'padding', 'fourier', 'all'],
                       help='Méthode d\'adaptation à utiliser')
    parser.add_argument('--visualize', action='store_true', help='Créer des visualisations')
    
    args = parser.parse_args()
    
    logger.info("🚀 PRÉDICTION AVEC DONNÉES DE 103 POINTS")
    logger.info("="*50)
    
    # Charger le modèle
    model, input_scaler, gap_scaler, L_ecran_scaler, device = load_kfold_model()
    
    # Créer l'adaptateur
    adapter = DataAdapter()
    
    # Données d'exemple si aucune fournie
    if not args.data:
        logger.info("📊 Génération de données d'exemple (103 points)")
        x = np.linspace(0, 10, 103)
        data_103 = 1.0 + 0.3 * np.sin(2 * np.pi * x) + 0.1 * np.sin(6 * np.pi * x) + 0.05 * np.random.randn(103)
        data_103 = np.abs(data_103)
    else:
        if ',' in args.data:
            data_103 = np.array([float(x.strip()) for x in args.data.split(',')])
        else:
            data_103 = np.loadtxt(args.data)
        
        if len(data_103) != 103:
            logger.warning(f"⚠️ Données ont {len(data_103)} points, ajustement à 103")
            if len(data_103) > 103:
                data_103 = data_103[:103]
            else:
                data_103 = np.pad(data_103, (0, 103 - len(data_103)), 'edge')
    
    logger.info(f"✅ Données de 103 points chargées")
    logger.info(f"   Min: {np.min(data_103):.3f}, Max: {np.max(data_103):.3f}")
    logger.info(f"   Moyenne: {np.mean(data_103):.3f}, Écart-type: {np.std(data_103):.3f}")
    
    # Adapter les données selon la méthode choisie
    if args.method == 'all':
        adapted_methods = adapter.compare_methods(data_103)
        
        logger.info(f"\n🔄 TEST DE TOUTES LES MÉTHODES:")
        
        for method_name, data_600 in adapted_methods.items():
            gap_pred, L_ecran_pred = predict_with_adapted_data(
                model, input_scaler, gap_scaler, L_ecran_scaler, device, data_600
            )
            
            logger.info(f"\n📋 Méthode: {method_name.upper()}")
            logger.info(f"   Gap prédit: {gap_pred:.4f} µm")
            logger.info(f"   L_écran prédit: {L_ecran_pred:.1f} µm")
    
    else:
        # Utiliser une méthode spécifique
        if args.method == 'linear':
            data_600 = adapter.method_1_interpolation_linear(data_103)
        elif args.method == 'cubic':
            data_600 = adapter.method_2_interpolation_cubic(data_103)
        elif args.method == 'spline':
            data_600 = adapter.method_3_spline_smoothing(data_103)
        elif args.method == 'padding':
            data_600 = adapter.method_4_padding_interpolation(data_103)
        elif args.method == 'fourier':
            data_600 = adapter.method_5_fourier_reconstruction(data_103)
        else:
            raise ValueError(f"Méthode inconnue: {args.method}")
        
        gap_pred, L_ecran_pred = predict_with_adapted_data(
            model, input_scaler, gap_scaler, L_ecran_scaler, device, data_600
        )
        
        logger.info(f"\n🎯 PRÉDICTION (Méthode: {args.method.upper()})")
        logger.info(f"   Gap prédit: {gap_pred:.4f} µm")
        logger.info(f"   L_écran prédit: {L_ecran_pred:.1f} µm")
    
    logger.info(f"\n✅ PRÉDICTION TERMINÉE")

if __name__ == "__main__":
    main()
