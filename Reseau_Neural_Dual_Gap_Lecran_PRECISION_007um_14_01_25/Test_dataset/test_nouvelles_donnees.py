#!/usr/bin/env python3
"""
Test sur Nouvelles Données - Modèle Dual Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 18 - 06 - 2025

Script exemple pour tester le modèle sur de nouvelles données.
Montre le workflow complet étape par étape.
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
    Prédit les paramètres pour un échantillon.
    
    Args:
        model: Modèle entraîné
        profile (np.array): Profil d'intensité [600]
        data_loader: DataLoader pour normalisation
    
    Returns:
        tuple: (gap_pred, L_ecran_pred)
    """
    print(f"🔧 Traitement d'un échantillon:")
    print(f"   Input shape: {profile.shape}")
    print(f"   Input range: [{profile.min():.3f}, {profile.max():.3f}]")
    
    # ÉTAPE 1: Normaliser l'entrée
    profile_scaled = data_loader.input_scaler.transform(profile.reshape(1, -1))
    print(f"   Après normalisation: [{profile_scaled.min():.3f}, {profile_scaled.max():.3f}]")
    
    # ÉTAPE 2: Prédiction
    with torch.no_grad():
        input_tensor = torch.FloatTensor(profile_scaled)
        prediction_scaled = model(input_tensor).numpy()
    print(f"   Prédiction normalisée: {prediction_scaled[0]}")
    
    # ÉTAPE 3: Dénormaliser la sortie
    prediction_original = data_loader.inverse_transform_predictions(prediction_scaled)
    print(f"   Prédiction finale: {prediction_original[0]}")
    
    gap_pred = prediction_original[0, 0]
    L_ecran_pred = prediction_original[0, 1]
    
    return gap_pred, L_ecran_pred

def load_model_for_new_data():
    """
    Charge le modèle et le data_loader pour nouvelles données.
    
    Returns:
        tuple: (model, data_loader)
    """
    print("🔄 Chargement du modèle pour nouvelles données...")
    
    # Charger le modèle
    model_path = "../models/dual_parameter_model.pth"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = DualParameterPredictor(input_size=600)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Modèle chargé: {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # Charger le data_loader avec les scalers d'entraînement
    data_loader = DualDataLoader()
    
    # Configuration minimale pour charger les scalers
    config = {
        'data_processing': {
            'augmentation': {'enable': False},
            'data_splits': {'train': 0.8, 'validation': 0.1, 'test': 0.1},
            'normalization': {'target_scaling': {'separate_scaling': True}}
        },
        'training': {'batch_size': 32}
    }
    
    # Charger le pipeline pour initialiser les scalers
    _ = data_loader.get_complete_pipeline(config)
    
    print(f"✅ DataLoader configuré avec scalers d'entraînement")
    
    return model, data_loader

def create_sample_data():
    """
    Crée des données d'exemple pour démonstration.
    
    Returns:
        tuple: (X_sample, y_sample)
    """
    print("🎲 Création de données d'exemple...")
    
    # Utiliser quelques échantillons du dataset existant comme exemple
    data_loader = DualDataLoader()
    config = {
        'data_processing': {
            'augmentation': {'enable': False},
            'data_splits': {'train': 0.8, 'validation': 0.1, 'test': 0.1},
            'normalization': {'target_scaling': {'separate_scaling': True}}
        },
        'training': {'batch_size': 32}
    }
    
    pipeline_result = data_loader.get_complete_pipeline(config)
    X_test = pipeline_result['raw_data'][2]
    y_test = pipeline_result['raw_data'][5]
    
    # Prendre 5 échantillons aléatoirement
    indices = np.random.choice(len(X_test), 5, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]
    
    print(f"✅ Données d'exemple créées: {X_sample.shape}")
    print(f"   Échantillons sélectionnés: {indices}")
    
    return X_sample, y_sample

def test_nouvelles_donnees(X_new, y_new=None, model=None, data_loader=None):
    """
    Teste le modèle sur de nouvelles données.
    
    Args:
        X_new: Nouvelles données [n_samples, 600]
        y_new: Labels optionnels [n_samples, 2]
        model: Modèle entraîné (optionnel)
        data_loader: DataLoader configuré (optionnel)
    
    Returns:
        dict: Résultats des prédictions
    """
    print(f"\n🎯 TEST SUR NOUVELLES DONNÉES")
    print("="*40)
    print(f"📊 Données d'entrée:")
    print(f"   Shape: {X_new.shape}")
    print(f"   Type: {X_new.dtype}")
    print(f"   Range: [{X_new.min():.3f}, {X_new.max():.3f}]")
    
    # Charger le modèle si non fourni
    if model is None or data_loader is None:
        model, data_loader = load_model_for_new_data()
    
    # Prédictions
    predictions_gap = []
    predictions_L_ecran = []
    
    print(f"\n🔄 Calcul des prédictions...")
    for i in range(len(X_new)):
        print(f"\n--- Échantillon {i+1}/{len(X_new)} ---")
        
        profile = X_new[i]
        gap_pred, L_ecran_pred = predict_sample(model, profile, data_loader)
        
        predictions_gap.append(gap_pred)
        predictions_L_ecran.append(L_ecran_pred)
        
        print(f"   → Gap prédit: {gap_pred:.4f} µm")
        print(f"   → L_ecran prédit: {L_ecran_pred:.2f} µm")
        
        if y_new is not None:
            gap_true = y_new[i, 0]
            L_ecran_true = y_new[i, 1]
            gap_error = abs(gap_pred - gap_true)
            L_ecran_error = abs(L_ecran_pred - L_ecran_true)
            
            print(f"   ✓ Gap vrai: {gap_true:.4f} µm (erreur: {gap_error:.4f} µm)")
            print(f"   ✓ L_ecran vrai: {L_ecran_true:.2f} µm (erreur: {L_ecran_error:.4f} µm)")
    
    # Conversion en arrays
    predictions_gap = np.array(predictions_gap)
    predictions_L_ecran = np.array(predictions_L_ecran)
    
    # Résultats
    results = {
        'n_samples': len(X_new),
        'predictions': {
            'gap': predictions_gap.tolist(),
            'L_ecran': predictions_L_ecran.tolist()
        }
    }
    
    # Évaluation si labels disponibles
    if y_new is not None:
        true_gap = y_new[:, 0]
        true_L_ecran = y_new[:, 1]
        
        # Métriques
        gap_r2 = r2_score(true_gap, predictions_gap)
        L_ecran_r2 = r2_score(true_L_ecran, predictions_L_ecran)
        combined_r2 = (gap_r2 + L_ecran_r2) / 2
        
        gap_mae = mean_absolute_error(true_gap, predictions_gap)
        L_ecran_mae = mean_absolute_error(true_L_ecran, predictions_L_ecran)
        
        # Tolérance
        gap_within_tolerance = np.sum(np.abs(predictions_gap - true_gap) <= 0.01)
        L_ecran_within_tolerance = np.sum(np.abs(predictions_L_ecran - true_L_ecran) <= 0.1)
        
        gap_accuracy = gap_within_tolerance / len(X_new)
        L_ecran_accuracy = L_ecran_within_tolerance / len(X_new)
        
        # Ajouter aux résultats
        results['evaluation'] = {
            'gap_r2': gap_r2,
            'L_ecran_r2': L_ecran_r2,
            'combined_r2': combined_r2,
            'gap_mae': gap_mae,
            'L_ecran_mae': L_ecran_mae,
            'gap_accuracy': gap_accuracy,
            'L_ecran_accuracy': L_ecran_accuracy
        }
        
        print(f"\n📊 ÉVALUATION:")
        print("="*30)
        print(f"🎯 Métriques R²:")
        print(f"   Gap R²: {gap_r2:.4f}")
        print(f"   L_ecran R²: {L_ecran_r2:.4f}")
        print(f"   Combined R²: {combined_r2:.4f}")
        
        print(f"\n📏 Erreurs moyennes:")
        print(f"   Gap MAE: {gap_mae:.4f} µm")
        print(f"   L_ecran MAE: {L_ecran_mae:.4f} µm")
        
        print(f"\n🎯 Précision dans tolérance:")
        print(f"   Gap (±0.01µm): {gap_within_tolerance}/{len(X_new)} ({gap_accuracy:.1%})")
        print(f"   L_ecran (±0.1µm): {L_ecran_within_tolerance}/{len(X_new)} ({L_ecran_accuracy:.1%})")
    
    return results

def save_results(results, filename="nouvelles_predictions.json"):
    """
    Sauvegarde les résultats.
    
    Args:
        results: Résultats des prédictions
        filename: Nom du fichier de sauvegarde
    """
    Path("../results").mkdir(exist_ok=True)
    filepath = f"../results/{filename}"
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Résultats sauvegardés: {filepath}")

def create_visualization(results, save_path="../plots/nouvelles_predictions.png"):
    """
    Crée une visualisation des résultats.
    
    Args:
        results: Résultats des prédictions
        save_path: Chemin de sauvegarde
    """
    if 'evaluation' not in results:
        print("⚠️ Pas d'évaluation disponible, visualisation limitée")
        return
    
    Path("../plots").mkdir(exist_ok=True)
    
    # Données
    gap_pred = np.array(results['predictions']['gap'])
    L_ecran_pred = np.array(results['predictions']['L_ecran'])
    
    # Figure simple
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gap
    ax1.bar(range(len(gap_pred)), gap_pred, alpha=0.7)
    ax1.set_xlabel('Échantillon')
    ax1.set_ylabel('Gap Prédit (µm)')
    ax1.set_title(f'Prédictions Gap - {results["n_samples"]} échantillons')
    ax1.grid(True, alpha=0.3)
    
    # L_ecran
    ax2.bar(range(len(L_ecran_pred)), L_ecran_pred, alpha=0.7, color='orange')
    ax2.set_xlabel('Échantillon')
    ax2.set_ylabel('L_ecran Prédit (µm)')
    ax2.set_title(f'Prédictions L_ecran - {results["n_samples"]} échantillons')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Visualisation sauvegardée: {save_path}")

def main():
    """
    Fonction principale de démonstration.
    """
    print("🎯 TEST SUR NOUVELLES DONNÉES - MODÈLE DUAL GAP + L_ECRAN")
    print("="*60)
    print(f"Auteur: Oussama GUELFAA")
    print(f"Date: 18-06-2025")
    print("="*60)
    
    try:
        # 1. Charger le modèle
        model, data_loader = load_model_for_new_data()
        
        # 2. Créer des données d'exemple (remplacez par vos vraies données)
        print(f"\n📝 ÉTAPE 1: Préparation des données")
        X_sample, y_sample = create_sample_data()
        
        print(f"\n📝 ÉTAPE 2: Test sur les nouvelles données")
        # 3. Tester sur les nouvelles données
        results = test_nouvelles_donnees(X_sample, y_sample, model, data_loader)
        
        print(f"\n📝 ÉTAPE 3: Sauvegarde et visualisation")
        # 4. Sauvegarder les résultats
        save_results(results, "exemple_nouvelles_donnees.json")
        
        # 5. Créer une visualisation
        create_visualization(results, "../plots/exemple_nouvelles_donnees.png")
        
        print(f"\n🎉 Test terminé avec succès !")
        print(f"\n📋 POUR VOS PROPRES DONNÉES:")
        print(f"   1. Remplacez create_sample_data() par le chargement de vos données")
        print(f"   2. Format requis: X_new [n_samples, 600], y_new [n_samples, 2] (optionnel)")
        print(f"   3. Appelez: test_nouvelles_donnees(X_new, y_new)")
        print(f"   4. Les résultats seront automatiquement sauvegardés")
        
    except Exception as e:
        print(f"❌ Erreur dans le test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
