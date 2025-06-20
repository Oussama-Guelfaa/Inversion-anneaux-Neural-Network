#!/usr/bin/env python3
"""
Test de Prédiction Amélioré - Modèle Dual Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 19 - 06 - 2025

Script pour tester les prédictions du modèle entraîné
utilisant la même approche que demo.py pour des résultats optimaux.
"""

import sys
import os
sys.path.append('../src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd

from dual_parameter_model import DualParameterPredictor
from data_loader import DualDataLoader

def predict_sample(model, profile, data_loader):
    """
    Prédit les paramètres pour un échantillon (identique à demo.py).
    
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

def load_model_and_data():
    """
    Charge le modèle entraîné et les données de test (approche demo.py).
    
    Returns:
        tuple: (model, X_test, y_test, data_loader)
    """
    print("🔄 Chargement du modèle et des données...")
    
    # Charger le modèle
    model_path = "../models/dual_parameter_model.pth"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = DualParameterPredictor(input_size=600)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Modèle chargé: {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # Charger les données avec la même approche que demo.py
    data_loader = DualDataLoader()
    
    # Configuration modifiée pour avoir ~2400 échantillons de test
    # Avec 17,080 échantillons total: 70/16/14 ≈ 11,956/2,733/2,391 échantillons
    config = {
        'data_processing': {
            'augmentation': {'enable': False},  # Pas d'augmentation pour le test
            'data_splits': {'train': 0.70, 'validation': 0.16, 'test': 0.14},  # Split modifié pour ~2400 test
            'normalization': {'target_scaling': {'separate_scaling': True}}
        },
        'training': {'batch_size': 32}
    }
    
    pipeline_result = data_loader.get_complete_pipeline(config)
    
    # Récupérer les données de test non normalisées (comme dans demo.py)
    X_test = pipeline_result['raw_data'][2]  # Données de test non normalisées
    y_test = pipeline_result['raw_data'][5]  # Labels de test
    
    print(f"✅ Données chargées: {len(X_test)} échantillons de test")
    
    return model, X_test, y_test, data_loader

def evaluate_model_comprehensive(model, X_test, y_test, data_loader, n_samples=None):
    """
    Évaluation complète du modèle sur les données de test.
    
    Args:
        model: Modèle entraîné
        X_test: Données de test
        y_test: Labels de test
        data_loader: DataLoader configuré
        n_samples: Nombre d'échantillons à tester (None = tous)
    
    Returns:
        dict: Résultats détaillés
    """
    print(f"\n🎯 ÉVALUATION COMPLÈTE DU MODÈLE")
    print("="*50)
    
    # Sélectionner les échantillons
    if n_samples is None or n_samples > len(X_test):
        n_samples = len(X_test)
        indices = np.arange(len(X_test))
    else:
        indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    print(f"📊 Test sur {n_samples} échantillons")
    
    # Prédictions
    predictions_gap = []
    predictions_L_ecran = []
    true_gap = []
    true_L_ecran = []
    
    print("🔄 Calcul des prédictions...")
    for i, idx in enumerate(indices):
        if i % 100 == 0:
            print(f"   Progression: {i}/{n_samples}")
        
        profile = X_test[idx]
        gap_pred, L_ecran_pred = predict_sample(model, profile, data_loader)
        
        predictions_gap.append(gap_pred)
        predictions_L_ecran.append(L_ecran_pred)
        true_gap.append(y_test[idx, 0])
        true_L_ecran.append(y_test[idx, 1])
    
    # Conversion en arrays
    predictions_gap = np.array(predictions_gap)
    predictions_L_ecran = np.array(predictions_L_ecran)
    true_gap = np.array(true_gap)
    true_L_ecran = np.array(true_L_ecran)
    
    # Calcul des métriques
    gap_r2 = r2_score(true_gap, predictions_gap)
    L_ecran_r2 = r2_score(true_L_ecran, predictions_L_ecran)
    combined_r2 = (gap_r2 + L_ecran_r2) / 2
    
    gap_mae = mean_absolute_error(true_gap, predictions_gap)
    L_ecran_mae = mean_absolute_error(true_L_ecran, predictions_L_ecran)
    
    gap_rmse = np.sqrt(mean_squared_error(true_gap, predictions_gap))
    L_ecran_rmse = np.sqrt(mean_squared_error(true_L_ecran, predictions_L_ecran))
    
    # Analyse de tolérance
    gap_tolerance = 0.01  # µm
    L_ecran_tolerance = 0.1  # µm
    
    gap_within_tolerance = np.sum(np.abs(predictions_gap - true_gap) <= gap_tolerance)
    L_ecran_within_tolerance = np.sum(np.abs(predictions_L_ecran - true_L_ecran) <= L_ecran_tolerance)
    
    gap_accuracy = gap_within_tolerance / n_samples
    L_ecran_accuracy = L_ecran_within_tolerance / n_samples
    
    # Affichage des résultats
    print(f"\n📊 RÉSULTATS DE L'ÉVALUATION")
    print("="*40)
    print(f"🎯 Métriques R²:")
    print(f"   Gap R²: {gap_r2:.4f}")
    print(f"   L_ecran R²: {L_ecran_r2:.4f}")
    print(f"   Combined R²: {combined_r2:.4f}")
    
    print(f"\n📏 Erreurs moyennes:")
    print(f"   Gap MAE: {gap_mae:.4f} µm")
    print(f"   Gap RMSE: {gap_rmse:.4f} µm")
    print(f"   L_ecran MAE: {L_ecran_mae:.4f} µm")
    print(f"   L_ecran RMSE: {L_ecran_rmse:.4f} µm")
    
    print(f"\n🎯 Précision dans tolérance:")
    print(f"   Gap (±{gap_tolerance} µm): {gap_within_tolerance}/{n_samples} ({gap_accuracy:.1%})")
    print(f"   L_ecran (±{L_ecran_tolerance} µm): {L_ecran_within_tolerance}/{n_samples} ({L_ecran_accuracy:.1%})")
    
    # Retourner les résultats
    results = {
        'n_samples': n_samples,
        'metrics': {
            'gap_r2': gap_r2,
            'L_ecran_r2': L_ecran_r2,
            'combined_r2': combined_r2,
            'gap_mae': gap_mae,
            'L_ecran_mae': L_ecran_mae,
            'gap_rmse': gap_rmse,
            'L_ecran_rmse': L_ecran_rmse
        },
        'tolerance_analysis': {
            'gap_tolerance': gap_tolerance,
            'L_ecran_tolerance': L_ecran_tolerance,
            'gap_accuracy': gap_accuracy,
            'L_ecran_accuracy': L_ecran_accuracy,
            'gap_within_tolerance': int(gap_within_tolerance),
            'L_ecran_within_tolerance': int(L_ecran_within_tolerance)
        },
        'predictions': {
            'gap_true': true_gap.tolist(),
            'gap_pred': predictions_gap.tolist(),
            'L_ecran_true': true_L_ecran.tolist(),
            'L_ecran_pred': predictions_L_ecran.tolist()
        }
    }
    
    return results

def create_visualizations(results, save_dir="plots"):
    """
    Crée des visualisations des résultats.
    
    Args:
        results: Résultats de l'évaluation
        save_dir: Répertoire de sauvegarde
    """
    print(f"\n📈 Création des visualisations...")
    
    Path(save_dir).mkdir(exist_ok=True)
    
    # Données
    gap_true = np.array(results['predictions']['gap_true'])
    gap_pred = np.array(results['predictions']['gap_pred'])
    L_ecran_true = np.array(results['predictions']['L_ecran_true'])
    L_ecran_pred = np.array(results['predictions']['L_ecran_pred'])
    
    # Figure avec 2 sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gap scatter plot
    ax1.scatter(gap_true, gap_pred, alpha=0.6, s=20)
    ax1.plot([gap_true.min(), gap_true.max()], [gap_true.min(), gap_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Gap Vrai (µm)')
    ax1.set_ylabel('Gap Prédit (µm)')
    ax1.set_title(f'Prédiction Gap (R² = {results["metrics"]["gap_r2"]:.4f})')
    ax1.grid(True, alpha=0.3)
    
    # L_ecran scatter plot
    ax2.scatter(L_ecran_true, L_ecran_pred, alpha=0.6, s=20)
    ax2.plot([L_ecran_true.min(), L_ecran_true.max()], [L_ecran_true.min(), L_ecran_true.max()], 'r--', lw=2)
    ax2.set_xlabel('L_ecran Vrai (µm)')
    ax2.set_ylabel('L_ecran Prédit (µm)')
    ax2.set_title(f'Prédiction L_ecran (R² = {results["metrics"]["L_ecran_r2"]:.4f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"../{save_dir}/test_predictions_improved.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualisations sauvegardées dans {save_dir}/")

def main():
    """
    Fonction principale de test.
    """
    print("🎯 TEST DE PRÉDICTION AMÉLIORÉ - MODÈLE DUAL GAP + L_ECRAN")
    print("="*60)
    print(f"Auteur: Oussama GUELFAA")
    print(f"Date: 19-06-2025")
    print("="*60)
    
    try:
        # 1. Charger le modèle et les données
        model, X_test, y_test, data_loader = load_model_and_data()
        
        # 2. Évaluation complète
        results = evaluate_model_comprehensive(model, X_test, y_test, data_loader)
        
        # 3. Créer les visualisations
        create_visualizations(results)
        
        # 4. Sauvegarder les résultats
        results_path = "../results/test_predictions_improved.json"
        Path("../results").mkdir(exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Test terminé avec succès !")
        print(f"📊 Résultats sauvegardés: {results_path}")
        print(f"📈 Visualisations: plots/test_predictions_improved.png")
        
    except Exception as e:
        print(f"❌ Erreur dans le test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
