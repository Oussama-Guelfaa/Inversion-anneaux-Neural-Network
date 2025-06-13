#!/usr/bin/env python3
"""
Démonstration du Modèle Dual Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Script de démonstration pour tester le modèle entraîné
sur de nouveaux échantillons.
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from dual_parameter_model import DualParameterPredictor, DualParameterMetrics
from data_loader import DualDataLoader

def load_trained_model(model_path="models/dual_parameter_model.pth"):
    """
    Charge le modèle entraîné.
    
    Args:
        model_path (str): Chemin vers le modèle sauvegardé
    
    Returns:
        tuple: (model, config, training_info)
    """
    print(f"🔄 Chargement du modèle depuis {model_path}...")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    # Charger le checkpoint (PyTorch 2.6 compatibility)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Recréer le modèle
    model = DualParameterPredictor(input_size=600)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    config = checkpoint['config']
    training_info = checkpoint.get('training_info', {})
    test_metrics = checkpoint.get('test_metrics', {})
    
    print(f"✅ Modèle chargé avec succès !")
    print(f"   Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Epochs d'entraînement: {training_info.get('final_epoch', 'N/A')}")
    print(f"   Performance test:")
    print(f"     Gap R²: {test_metrics.get('gap_r2', 'N/A'):.4f}")
    print(f"     L_ecran R²: {test_metrics.get('L_ecran_r2', 'N/A'):.4f}")
    print(f"     Combined R²: {test_metrics.get('combined_r2', 'N/A'):.4f}")
    
    return model, config, training_info, test_metrics

def predict_sample(model, profile, data_loader):
    """
    Prédit les paramètres pour un échantillon.
    
    Args:
        model: Modèle entraîné
        profile (np.array): Profil d'intensité [600]
        data_loader: DataLoader pour normalisation
    
    Returns:
        tuple: (gap_pred, L_ecran_pred)
    """
    # Normaliser l'entrée
    profile_scaled = data_loader.input_scaler.transform(profile.reshape(1, -1))
    
    # Prédiction
    with torch.no_grad():
        input_tensor = torch.FloatTensor(profile_scaled)
        prediction_scaled = model(input_tensor).numpy()
    
    # Dénormaliser la sortie
    prediction_original = data_loader.inverse_transform_predictions(prediction_scaled)
    
    gap_pred = prediction_original[0, 0]
    L_ecran_pred = prediction_original[0, 1]
    
    return gap_pred, L_ecran_pred

def demo_random_samples(model, data_loader, n_samples=5):
    """
    Démonstration avec échantillons aléatoires du dataset.
    
    Args:
        model: Modèle entraîné
        data_loader: DataLoader configuré
        n_samples (int): Nombre d'échantillons à tester
    """
    print(f"\n🎯 DÉMONSTRATION - {n_samples} ÉCHANTILLONS ALÉATOIRES")
    print("="*60)
    
    # Charger quelques échantillons du dataset
    config = {
        'data_processing': {
            'augmentation': {'enable': False},  # Pas d'augmentation pour la démo
            'data_splits': {'train': 0.8, 'validation': 0.1, 'test': 0.1},
            'normalization': {'target_scaling': {'separate_scaling': True}}
        },
        'training': {'batch_size': 32}
    }
    
    pipeline_result = data_loader.get_complete_pipeline(config)
    X_test = pipeline_result['raw_data'][2]  # Données de test non normalisées
    y_test = pipeline_result['raw_data'][5]  # Labels de test
    
    # Sélectionner des échantillons aléatoires
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    print(f"{'Index':<6} {'Gap Vrai':<10} {'Gap Prédit':<12} {'Erreur Gap':<12} {'L_ecran Vrai':<12} {'L_ecran Prédit':<14} {'Erreur L_ecran':<14}")
    print("-" * 90)
    
    total_gap_error = 0
    total_L_ecran_error = 0
    
    for i, idx in enumerate(indices):
        # Données vraies
        profile = X_test[idx]
        true_gap = y_test[idx, 0]
        true_L_ecran = y_test[idx, 1]
        
        # Prédiction
        pred_gap, pred_L_ecran = predict_sample(model, profile, data_loader)
        
        # Erreurs
        gap_error = abs(pred_gap - true_gap)
        L_ecran_error = abs(pred_L_ecran - true_L_ecran)
        
        total_gap_error += gap_error
        total_L_ecran_error += L_ecran_error
        
        print(f"{idx:<6} {true_gap:<10.4f} {pred_gap:<12.4f} {gap_error:<12.4f} {true_L_ecran:<12.2f} {pred_L_ecran:<14.2f} {L_ecran_error:<14.4f}")
    
    # Statistiques
    avg_gap_error = total_gap_error / n_samples
    avg_L_ecran_error = total_L_ecran_error / n_samples
    
    print("-" * 90)
    print(f"📊 STATISTIQUES:")
    print(f"   Erreur moyenne Gap: {avg_gap_error:.4f} µm")
    print(f"   Erreur moyenne L_ecran: {avg_L_ecran_error:.4f} µm")
    print(f"   Tolérance Gap: ±0.01 µm")
    print(f"   Tolérance L_ecran: ±0.1 µm")
    
    gap_within_tolerance = sum(1 for i in indices if abs(predict_sample(model, X_test[i], data_loader)[0] - y_test[i, 0]) <= 0.01)
    L_ecran_within_tolerance = sum(1 for i in indices if abs(predict_sample(model, X_test[i], data_loader)[1] - y_test[i, 1]) <= 0.1)
    
    print(f"   Gap dans tolérance: {gap_within_tolerance}/{n_samples} ({gap_within_tolerance/n_samples:.1%})")
    print(f"   L_ecran dans tolérance: {L_ecran_within_tolerance}/{n_samples} ({L_ecran_within_tolerance/n_samples:.1%})")

def demo_custom_prediction():
    """
    Démonstration avec prédiction personnalisée.
    """
    print(f"\n🔧 PRÉDICTION PERSONNALISÉE")
    print("="*40)
    print(f"Pour utiliser le modèle sur vos propres données:")
    print(f"")
    print(f"```python")
    print(f"# Charger le modèle")
    print(f"model, config, _, _ = load_trained_model()")
    print(f"")
    print(f"# Préparer le data loader")
    print(f"data_loader = DualDataLoader()")
    print(f"# ... configurer data_loader avec vos données ...")
    print(f"")
    print(f"# Prédire")
    print(f"gap, L_ecran = predict_sample(model, votre_profil, data_loader)")
    print(f"print(f'Gap prédit: {{gap:.4f}} µm')")
    print(f"print(f'L_ecran prédit: {{L_ecran:.2f}} µm')")
    print(f"```")

def main():
    """
    Fonction principale de démonstration.
    """
    print("🎯 DÉMONSTRATION MODÈLE DUAL GAP + L_ECRAN")
    print("="*50)
    print(f"Auteur: Oussama GUELFAA")
    print(f"Date: 06-01-2025")
    print("="*50)
    
    try:
        # 1. Charger le modèle entraîné
        model, config, training_info, test_metrics = load_trained_model()
        
        # 2. Préparer le data loader
        data_loader = DualDataLoader()
        
        # 3. Démonstration avec échantillons aléatoires
        demo_random_samples(model, data_loader, n_samples=10)
        
        # 4. Guide pour utilisation personnalisée
        demo_custom_prediction()
        
        print(f"\n🎉 Démonstration terminée avec succès !")
        print(f"📁 Modèle disponible: models/dual_parameter_model.pth")
        print(f"📊 Résultats complets: results/complete_results.json")
        print(f"📈 Visualisations: plots/")
        
    except Exception as e:
        print(f"❌ Erreur dans la démonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
