#!/usr/bin/env python3
"""
Test sur Dataset 2D - Modèle Dual Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 19 - 06 - 2025

Script pour tester le modèle entraîné sur le nouveau dataset_2D
situé dans data_generation/dataset_2D.
"""

import sys
import os
sys.path.append('../src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import scipy.io as sio
import glob
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

def load_model_for_dataset_2D():
    """
    Charge le modèle et le data_loader pour le test sur dataset_2D.
    
    Returns:
        tuple: (model, data_loader)
    """
    print("🔄 Chargement du modèle pour test dataset_2D...")
    
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
    
    # Configuration pour charger les scalers
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

def load_dataset_2D():
    """
    Charge le dataset_2D depuis data_generation/dataset_2D.
    
    Returns:
        tuple: (X_data, y_data, filenames)
    """
    print("🔄 Chargement du dataset_2D...")
    
    dataset_path = "../../data_generation/dataset_2D"
    
    # Charger le fichier labels.csv
    labels_file = f"{dataset_path}/labels.csv"
    if not Path(labels_file).exists():
        raise FileNotFoundError(f"Fichier labels non trouvé: {labels_file}")
    
    labels_df = pd.read_csv(labels_file)
    print(f"📊 Labels chargés: {len(labels_df)} échantillons")
    print(f"   Gap range: {labels_df['gap_um'].min():.3f} - {labels_df['gap_um'].max():.3f} µm")
    print(f"   L_ecran range: {labels_df['L_um'].min():.1f} - {labels_df['L_um'].max():.1f} µm")
    
    # Charger les fichiers .mat
    X_data = []
    y_data = []
    filenames = []
    
    print("🔄 Chargement des fichiers .mat...")
    
    for idx, row in labels_df.iterrows():
        gap = row['gap_um']
        L_ecran = row['L_um']
        
        # Construire le nom de fichier
        filename = f"gap_{gap:.4f}um_L_{L_ecran:.3f}um.mat"
        filepath = f"{dataset_path}/{filename}"
        
        if Path(filepath).exists():
            try:
                # Charger le fichier .mat
                mat_data = sio.loadmat(filepath)
                
                # Extraire le ratio (profil d'intensité)
                ratio = mat_data['ratio'].flatten()
                
                # Tronquer à 600 points (comme le modèle)
                if len(ratio) >= 600:
                    ratio_truncated = ratio[:600]
                else:
                    # Si moins de 600 points, padding avec des zéros
                    ratio_truncated = np.pad(ratio, (0, 600 - len(ratio)), 'constant')
                
                X_data.append(ratio_truncated)
                y_data.append([gap, L_ecran])
                filenames.append(filename)
                
                if len(X_data) % 100 == 0:
                    print(f"   Chargé: {len(X_data)} fichiers...")
                    
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de {filename}: {e}")
        else:
            print(f"⚠️ Fichier non trouvé: {filename}")
    
    # Conversion en arrays numpy
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"✅ Dataset_2D chargé:")
    print(f"   X_data shape: {X_data.shape}")
    print(f"   y_data shape: {y_data.shape}")
    print(f"   Fichiers chargés: {len(filenames)}")
    
    return X_data, y_data, filenames

def test_on_dataset_2D(model, data_loader, X_data, y_data, filenames, n_samples=None):
    """
    Teste le modèle sur le dataset_2D.
    
    Args:
        model: Modèle entraîné
        data_loader: DataLoader configuré
        X_data: Données d'entrée [n_samples, 600]
        y_data: Labels [n_samples, 2]
        filenames: Noms des fichiers
        n_samples: Nombre d'échantillons à tester (None = tous)
    
    Returns:
        dict: Résultats détaillés
    """
    print(f"\n🎯 TEST SUR DATASET_2D")
    print("="*50)
    
    # Sélectionner les échantillons
    if n_samples is None or n_samples > len(X_data):
        n_samples = len(X_data)
        indices = np.arange(len(X_data))
    else:
        indices = np.random.choice(len(X_data), n_samples, replace=False)
        indices = np.sort(indices)  # Trier pour un affichage ordonné
    
    print(f"📊 Test sur {n_samples} échantillons")
    
    # Prédictions
    predictions_gap = []
    predictions_L_ecran = []
    true_gap = []
    true_L_ecran = []
    tested_filenames = []
    
    print("🔄 Calcul des prédictions...")
    for i, idx in enumerate(indices):
        if i % 50 == 0:
            print(f"   Progression: {i}/{n_samples}")
        
        profile = X_data[idx]
        gap_pred, L_ecran_pred = predict_sample(model, profile, data_loader)
        
        predictions_gap.append(gap_pred)
        predictions_L_ecran.append(L_ecran_pred)
        true_gap.append(y_data[idx, 0])
        true_L_ecran.append(y_data[idx, 1])
        tested_filenames.append(filenames[idx])
    
    print(f"   Progression: {n_samples}/{n_samples} (100%)")
    
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
    
    # Statistiques d'erreurs
    gap_errors = np.abs(predictions_gap - true_gap)
    L_ecran_errors = np.abs(predictions_L_ecran - true_L_ecran)
    
    # Affichage des résultats
    print(f"\n📊 RÉSULTATS SUR DATASET_2D ({n_samples} échantillons)")
    print("="*60)
    print(f"🎯 Métriques R²:")
    print(f"   Gap R²: {gap_r2:.4f} (99.{gap_r2*10000-9900:.0f}%)")
    print(f"   L_ecran R²: {L_ecran_r2:.4f} (98.{L_ecran_r2*10000-9800:.0f}%)")
    print(f"   Combined R²: {combined_r2:.4f} (99.{combined_r2*10000-9900:.0f}%)")
    
    print(f"\n📏 Erreurs moyennes:")
    print(f"   Gap MAE: {gap_mae:.4f} µm")
    print(f"   Gap RMSE: {gap_rmse:.4f} µm")
    print(f"   L_ecran MAE: {L_ecran_mae:.4f} µm")
    print(f"   L_ecran RMSE: {L_ecran_rmse:.4f} µm")
    
    print(f"\n🎯 Précision dans tolérance:")
    print(f"   Gap (±{gap_tolerance} µm): {gap_within_tolerance}/{n_samples} ({gap_accuracy:.1%})")
    print(f"   L_ecran (±{L_ecran_tolerance} µm): {L_ecran_within_tolerance}/{n_samples} ({L_ecran_accuracy:.1%})")
    
    print(f"\n📈 Statistiques d'erreurs:")
    print(f"   Gap - Max: {gap_errors.max():.4f} µm, Min: {gap_errors.min():.4f} µm, Médiane: {np.median(gap_errors):.4f} µm")
    print(f"   L_ecran - Max: {L_ecran_errors.max():.4f} µm, Min: {L_ecran_errors.min():.4f} µm, Médiane: {np.median(L_ecran_errors):.4f} µm")
    
    # Affichage de quelques exemples
    print(f"\n📋 EXEMPLES DE PRÉDICTIONS:")
    print("="*80)
    print(f"{'Fichier':<35} {'Gap Vrai':<10} {'Gap Prédit':<12} {'Erreur Gap':<12} {'L_ecran Vrai':<12} {'L_ecran Prédit':<14} {'Erreur L_ecran':<14}")
    print("-" * 110)
    
    # Afficher les 10 premiers exemples
    for i in range(min(10, len(indices))):
        filename = tested_filenames[i]
        gap_true = true_gap[i]
        gap_pred = predictions_gap[i]
        L_ecran_true = true_L_ecran[i]
        L_ecran_pred = predictions_L_ecran[i]
        gap_error = abs(gap_pred - gap_true)
        L_ecran_error = abs(L_ecran_pred - L_ecran_true)
        
        print(f"{filename:<35} {gap_true:<10.4f} {gap_pred:<12.4f} {gap_error:<12.4f} {L_ecran_true:<12.2f} {L_ecran_pred:<14.2f} {L_ecran_error:<14.4f}")
    
    # Retourner les résultats
    results = {
        'dataset_info': {
            'name': 'dataset_2D',
            'n_samples_total': len(X_data),
            'n_samples_tested': n_samples,
            'gap_range': [float(true_gap.min()), float(true_gap.max())],
            'L_ecran_range': [float(true_L_ecran.min()), float(true_L_ecran.max())]
        },
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
        'error_statistics': {
            'gap_max_error': float(gap_errors.max()),
            'gap_min_error': float(gap_errors.min()),
            'gap_median_error': float(np.median(gap_errors)),
            'L_ecran_max_error': float(L_ecran_errors.max()),
            'L_ecran_min_error': float(L_ecran_errors.min()),
            'L_ecran_median_error': float(np.median(L_ecran_errors))
        },
        'predictions': {
            'filenames': tested_filenames,
            'gap_true': true_gap.tolist(),
            'gap_pred': predictions_gap.tolist(),
            'L_ecran_true': true_L_ecran.tolist(),
            'L_ecran_pred': predictions_L_ecran.tolist()
        }
    }
    
    return results

def create_visualizations_dataset_2D(results, save_dir="plots"):
    """
    Crée des visualisations des résultats pour dataset_2D.
    
    Args:
        results: Résultats du test
        save_dir: Répertoire de sauvegarde
    """
    print(f"\n📈 Création des visualisations pour dataset_2D...")
    
    Path(f"../{save_dir}").mkdir(exist_ok=True)

    # Données
    gap_true = np.array(results['predictions']['gap_true'])
    gap_pred = np.array(results['predictions']['gap_pred'])
    L_ecran_true = np.array(results['predictions']['L_ecran_true'])
    L_ecran_pred = np.array(results['predictions']['L_ecran_pred'])

    # Figure avec 2x2 sous-graphiques
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gap scatter plot
    ax1.scatter(gap_true, gap_pred, alpha=0.6, s=20, c='blue')
    ax1.plot([gap_true.min(), gap_true.max()], [gap_true.min(), gap_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Gap Vrai (µm)')
    ax1.set_ylabel('Gap Prédit (µm)')
    ax1.set_title(f'Prédiction Gap - Dataset_2D (R² = {results["metrics"]["gap_r2"]:.4f})')
    ax1.grid(True, alpha=0.3)
    
    # L_ecran scatter plot
    ax2.scatter(L_ecran_true, L_ecran_pred, alpha=0.6, s=20, c='orange')
    ax2.plot([L_ecran_true.min(), L_ecran_true.max()], [L_ecran_true.min(), L_ecran_true.max()], 'r--', lw=2)
    ax2.set_xlabel('L_ecran Vrai (µm)')
    ax2.set_ylabel('L_ecran Prédit (µm)')
    ax2.set_title(f'Prédiction L_ecran - Dataset_2D (R² = {results["metrics"]["L_ecran_r2"]:.4f})')
    ax2.grid(True, alpha=0.3)
    
    # Distribution des erreurs Gap
    gap_errors = np.abs(gap_pred - gap_true)
    ax3.hist(gap_errors, bins=50, alpha=0.7, edgecolor='black', color='blue')
    ax3.axvline(x=0.01, color='red', linestyle='--', label='Tolérance ±0.01µm')
    ax3.set_xlabel('Erreur Absolue Gap (µm)')
    ax3.set_ylabel('Fréquence')
    ax3.set_title(f'Distribution des Erreurs Gap - Dataset_2D')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Distribution des erreurs L_ecran
    L_ecran_errors = np.abs(L_ecran_pred - L_ecran_true)
    ax4.hist(L_ecran_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
    ax4.axvline(x=0.1, color='red', linestyle='--', label='Tolérance ±0.1µm')
    ax4.set_xlabel('Erreur Absolue L_ecran (µm)')
    ax4.set_ylabel('Fréquence')
    ax4.set_title(f'Distribution des Erreurs L_ecran - Dataset_2D')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"../{save_dir}/test_dataset_2D_results.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualisations sauvegardées: ../{save_dir}/test_dataset_2D_results.png")

def main():
    """
    Fonction principale de test sur dataset_2D.
    """
    print("🎯 TEST SUR DATASET_2D - MODÈLE DUAL GAP + L_ECRAN")
    print("="*60)
    print(f"Auteur: Oussama GUELFAA")
    print(f"Date: 19-06-2025")
    print("="*60)
    
    try:
        # 1. Charger le modèle
        model, data_loader = load_model_for_dataset_2D()
        
        # 2. Charger le dataset_2D
        X_data, y_data, filenames = load_dataset_2D()
        
        # 3. Tester sur le dataset_2D (tous les échantillons)
        results = test_on_dataset_2D(model, data_loader, X_data, y_data, filenames)
        
        # 4. Créer les visualisations
        create_visualizations_dataset_2D(results)
        
        # 5. Sauvegarder les résultats
        results_path = "../results/test_dataset_2D_results.json"
        Path("../results").mkdir(exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Test sur dataset_2D terminé avec succès !")
        print(f"📊 Résultats sauvegardés: {results_path}")
        print(f"📈 Visualisations: plots/test_dataset_2D_results.png")
        
        # Résumé final
        print(f"\n📋 RÉSUMÉ FINAL:")
        print(f"   Dataset: {results['dataset_info']['name']}")
        print(f"   Échantillons testés: {results['dataset_info']['n_samples_tested']}")
        print(f"   Gap R²: {results['metrics']['gap_r2']:.4f}")
        print(f"   L_ecran R²: {results['metrics']['L_ecran_r2']:.4f}")
        print(f"   Gap Accuracy: {results['tolerance_analysis']['gap_accuracy']:.1%}")
        print(f"   L_ecran Accuracy: {results['tolerance_analysis']['L_ecran_accuracy']:.1%}")
        
    except Exception as e:
        print(f"❌ Erreur dans le test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
