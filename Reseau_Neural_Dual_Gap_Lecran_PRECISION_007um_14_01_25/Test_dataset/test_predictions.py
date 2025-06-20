#!/usr/bin/env python3
"""
Test de Prédiction du Modèle Neural Dual Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 19 - 06 - 2025

Ce script teste le modèle entraîné sur le dataset externe et affiche
les prédictions avec analyse détaillée des performances.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys
import os
import joblib

# Ajouter le chemin vers les modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loader import DualDataLoader

class DualParameterPredictor(nn.Module):
    """
    Modèle de prédiction dual pour Gap et L_ecran - Architecture originale.
    """

    def __init__(self, input_size=600, dropout_rate=0.15):
        super(DualParameterPredictor, self).__init__()

        # Couche d'entrée - 1024 neurones
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Couche cachée 1 - 512 neurones
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Couche cachée 2 - 256 neurones
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Couche cachée 3 - 128 neurones
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(dropout_rate * 0.8)

        # Couche cachée 4 - 64 neurones
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(dropout_rate * 0.5)

        # Couche cachée 5 - 32 neurones
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        self.dropout6 = nn.Dropout(dropout_rate * 0.3)

        # Couche de sortie - 2 paramètres [gap, L_ecran]
        self.fc_out = nn.Linear(32, 2)

    def forward(self, x):
        # Couche 1: 600 → 1024
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        # Couche 2: 1024 → 512
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        # Couche 3: 512 → 256
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        # Couche 4: 256 → 128
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)

        # Couche 5: 128 → 64
        x = torch.relu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)

        # Couche 6: 64 → 32
        x = torch.relu(self.bn6(self.fc6(x)))
        x = self.dropout6(x)

        # Sortie: 32 → 2 (gap, L_ecran)
        x = self.fc_out(x)

        return x

def load_trained_model(model_path="../models/dual_parameter_model.pth"):
    """
    Charge le modèle entraîné (même fonction que demo.py).

    Args:
        model_path (str): Chemin vers le modèle sauvegardé

    Returns:
        tuple: (model, config, training_info, test_metrics)
    """
    print(f"� Chargement du modèle depuis {model_path}...")

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

def make_predictions(model, data_loader, input_scaler_saved, gap_scaler_saved, L_ecran_scaler_saved,
                    dataset_type="test", max_samples=None):
    """
    Fait des prédictions sur le dataset et retourne les résultats.
    
    Args:
        model: Modèle PyTorch
        data_loader: DualDataLoader instance
        input_scaler, gap_scaler, L_ecran_scaler: Scalers
        dataset_type (str): Type de données ("test", "val", "train")
        max_samples (int): Limite le nombre d'échantillons
    
    Returns:
        tuple: (predictions, targets, raw_data)
    """
    print(f"\n🔮 PRÉDICTIONS SUR LE DATASET {dataset_type.upper()}")
    print("="*50)
    
    # Charger et préparer les données
    config = {
        'data_processing': {
            'augmentation': {'gap_density': 1, 'L_ecran_density': 1, 'method': 'linear', 'include_original': True},
            'data_splits': {'train': 0.70, 'validation': 0.15, 'test': 0.15},
            'normalization': {'target_scaling': {'separate_scaling': True}}
        },
        'training': {'batch_size': 32}
    }
    
    # Obtenir les données
    pipeline_result = data_loader.get_complete_pipeline(config)
    
    if dataset_type == "test":
        X_raw, y_raw = pipeline_result['raw_data'][2], pipeline_result['raw_data'][5]
        X_scaled = pipeline_result['scaled_data'][2]
    elif dataset_type == "val":
        X_raw, y_raw = pipeline_result['raw_data'][1], pipeline_result['raw_data'][4]
        X_scaled = pipeline_result['scaled_data'][1]
    else:  # train
        X_raw, y_raw = pipeline_result['raw_data'][0], pipeline_result['raw_data'][3]
        X_scaled = pipeline_result['scaled_data'][0]
    
    # Limiter le nombre d'échantillons si demandé
    if max_samples and len(X_raw) > max_samples:
        indices = np.random.choice(len(X_raw), max_samples, replace=False)
        X_raw = X_raw[indices]
        y_raw = y_raw[indices]
        X_scaled = X_scaled[indices]
    
    print(f"📊 Échantillons à prédire: {len(X_raw)}")
    
    # Faire les prédictions
    device = next(model.parameters()).device
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()
    
    # Dénormaliser les prédictions avec les scalers sauvegardés
    pred_gap = gap_scaler_saved.inverse_transform(y_pred_scaled[:, 0:1])
    pred_L_ecran = L_ecran_scaler_saved.inverse_transform(y_pred_scaled[:, 1:2])
    predictions = np.hstack([pred_gap, pred_L_ecran])
    
    print(f"✅ Prédictions terminées")
    
    return predictions, y_raw, X_raw

def analyze_predictions(predictions, targets, dataset_type="test", tolerance_gap=0.007, tolerance_L=0.1):
    """
    Analyse détaillée des prédictions avec métriques et visualisations.
    
    Args:
        predictions (np.array): Prédictions [n_samples, 2]
        targets (np.array): Valeurs réelles [n_samples, 2]
        dataset_type (str): Type de dataset
        tolerance_gap (float): Tolérance pour gap en µm
        tolerance_L (float): Tolérance pour L_ecran en µm
    
    Returns:
        pd.DataFrame: DataFrame avec résultats détaillés
    """
    print(f"\n📈 ANALYSE DES PRÉDICTIONS - {dataset_type.upper()}")
    print("="*60)
    
    # Créer le DataFrame des résultats
    results_df = pd.DataFrame({
        'GAP_reel': targets[:, 0],
        'LECRAN_reel': targets[:, 1],
        'GAP_pred': predictions[:, 0],
        'LECRAN_pred': predictions[:, 1]
    })
    
    # Calculer les erreurs
    results_df['GAP_erreur'] = np.abs(results_df['GAP_pred'] - results_df['GAP_reel'])
    results_df['LECRAN_erreur'] = np.abs(results_df['LECRAN_pred'] - results_df['LECRAN_reel'])
    
    # Indicateurs de succès
    results_df['GAP_success'] = results_df['GAP_erreur'] <= tolerance_gap
    results_df['LECRAN_success'] = results_df['LECRAN_erreur'] <= tolerance_L
    results_df['BOTH_success'] = results_df['GAP_success'] & results_df['LECRAN_success']
    
    # Métriques de performance
    gap_r2 = r2_score(results_df['GAP_reel'], results_df['GAP_pred'])
    lecran_r2 = r2_score(results_df['LECRAN_reel'], results_df['LECRAN_pred'])
    gap_mae = mean_absolute_error(results_df['GAP_reel'], results_df['GAP_pred'])
    lecran_mae = mean_absolute_error(results_df['LECRAN_reel'], results_df['LECRAN_pred'])
    gap_rmse = np.sqrt(mean_squared_error(results_df['GAP_reel'], results_df['GAP_pred']))
    lecran_rmse = np.sqrt(mean_squared_error(results_df['LECRAN_reel'], results_df['LECRAN_pred']))
    
    # Accuracy avec tolérance
    gap_accuracy = results_df['GAP_success'].mean() * 100
    lecran_accuracy = results_df['LECRAN_success'].mean() * 100
    both_accuracy = results_df['BOTH_success'].mean() * 100
    
    # Afficher les résultats
    print(f"📊 MÉTRIQUES DE PERFORMANCE:")
    print(f"   Échantillons testés: {len(results_df)}")
    print(f"")
    print(f"🎯 GAP (tolérance ±{tolerance_gap}µm):")
    print(f"   R² Score: {gap_r2:.4f}")
    print(f"   MAE: {gap_mae:.4f} µm")
    print(f"   RMSE: {gap_rmse:.4f} µm")
    print(f"   Accuracy: {gap_accuracy:.1f}%")
    print(f"")
    print(f"📏 L_ECRAN (tolérance ±{tolerance_L}µm):")
    print(f"   R² Score: {lecran_r2:.4f}")
    print(f"   MAE: {lecran_mae:.4f} µm")
    print(f"   RMSE: {lecran_rmse:.4f} µm")
    print(f"   Accuracy: {lecran_accuracy:.1f}%")
    print(f"")
    print(f"🎯 PERFORMANCE GLOBALE:")
    print(f"   Both Success: {both_accuracy:.1f}%")
    
    return results_df

def display_sample_predictions(results_df, n_samples=20):
    """
    Affiche un échantillon de prédictions pour inspection visuelle.
    """
    print(f"\n🔍 ÉCHANTILLON DE PRÉDICTIONS ({n_samples} exemples)")
    print("="*80)
    
    # Sélectionner des échantillons aléatoires
    sample_indices = np.random.choice(len(results_df), min(n_samples, len(results_df)), replace=False)
    sample_df = results_df.iloc[sample_indices].copy()
    
    # Afficher le tableau
    print(f"{'#':<3} {'GAP_reel':<10} {'GAP_pred':<10} {'GAP_err':<8} {'✓':<2} {'LECRAN_reel':<12} {'LECRAN_pred':<12} {'LEC_err':<8} {'✓':<2} {'Status':<8}")
    print("-" * 80)
    
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        gap_status = "✅" if row['GAP_success'] else "❌"
        lecran_status = "✅" if row['LECRAN_success'] else "❌"
        overall_status = "✅ OK" if row['BOTH_success'] else "❌ FAIL"
        
        print(f"{i+1:<3} {row['GAP_reel']:<10.4f} {row['GAP_pred']:<10.4f} {row['GAP_erreur']:<8.4f} {gap_status:<2} "
              f"{row['LECRAN_reel']:<12.1f} {row['LECRAN_pred']:<12.1f} {row['LECRAN_erreur']:<8.1f} {lecran_status:<2} {overall_status:<8}")

def main():
    """
    Fonction principale de test des prédictions.
    """
    print("🧪 TEST DE PRÉDICTION DU MODÈLE NEURAL DUAL")
    print("="*60)
    
    # Chemins des fichiers
    model_path = "../models/dual_parameter_model.pth"
    scalers_dir = "../models"
    
    # Vérifier que les fichiers existent
    if not Path(model_path).exists():
        print(f"❌ Erreur: Modèle non trouvé à {model_path}")
        return
    
    # Charger le modèle et les scalers
    try:
        model, input_scaler, gap_scaler, L_ecran_scaler = load_model_and_scalers(model_path, scalers_dir)
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return
    
    # Créer le data loader
    data_loader = DualDataLoader(dataset_path="../../data_generation/dataset_2D")
    
    # Faire les prédictions sur le dataset de test
    try:
        predictions, targets, raw_data = make_predictions(
            model, data_loader, input_scaler, gap_scaler, L_ecran_scaler,
            dataset_type="test", max_samples=200  # Limiter pour le test
        )
        
        # Analyser les résultats
        results_df = analyze_predictions(predictions, targets, "test")
        
        # Afficher des exemples
        display_sample_predictions(results_df, n_samples=15)
        
        # Sauvegarder les résultats
        output_path = "results_predictions_external.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Résultats sauvegardés: {output_path}")
        
        print(f"\n✅ Test terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur lors des prédictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
