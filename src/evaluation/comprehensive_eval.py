#!/usr/bin/env python3
"""
Comprehensive Model Evaluation and Analysis
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Évaluation complète du modèle avec :
- Analyse détaillée des performances
- Visualisations des prédictions vs vraies valeurs
- Diagnostic des problèmes de généralisation
- Rapport complet avec recommandations
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import glob
import seaborn as sns
from train_and_evaluate_complete import OptimizedRegressor

def load_all_data():
    """Charge toutes les données : entraînement et test."""
    
    print("=== CHARGEMENT DE TOUTES LES DONNÉES ===")
    
    # Données d'entraînement (simulées)
    train_data = np.load('processed_data/training_data.npz', allow_pickle=True)
    X_train_full = train_data['X']
    y_train_full = train_data['y']
    
    # Données de test (expérimentales)
    dataset_dir = "../data_generation/dataset"
    labels_df = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))
    
    X_test = []
    y_test = []
    filenames = []
    
    for idx, row in labels_df.iterrows():
        filename = row['filename']
        gap = row['gap_um']
        L_ecran = row['L_um']
        
        mat_filename = filename.replace('.png', '.mat')
        mat_path = os.path.join(dataset_dir, mat_filename)
        
        if os.path.exists(mat_path):
            try:
                data = sio.loadmat(mat_path)
                ratio = data['ratio'].flatten()
                X_test.append(ratio)
                y_test.append([L_ecran, gap])
                filenames.append(filename)
            except Exception as e:
                print(f"Erreur {mat_filename}: {e}")
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Données chargées:")
    print(f"  Entraînement: X{X_train_full.shape}, y{y_train_full.shape}")
    print(f"  Test: X{X_test.shape}, y{y_test.shape}")
    
    return X_train_full, y_train_full, X_test, y_test, filenames

def load_trained_model():
    """Charge le modèle entraîné et les scalers."""
    
    print("\n=== CHARGEMENT DU MODÈLE ENTRAÎNÉ ===")
    
    # Charger le modèle
    model = OptimizedRegressor(input_dim=1000, output_dim=2)
    checkpoint = torch.load('models/final_optimized_regressor.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Charger les scalers
    scalers = np.load('models/final_scalers.npz')
    scaler_X = StandardScaler()
    scaler_X.mean_ = scalers['scaler_X_mean']
    scaler_X.scale_ = scalers['scaler_X_scale']
    
    scaler_y = StandardScaler()
    scaler_y.mean_ = scalers['scaler_y_mean']
    scaler_y.scale_ = scalers['scaler_y_scale']
    
    print(f"Modèle et scalers chargés avec succès")
    print(f"Epoch d'entraînement: {checkpoint.get('epoch', 'N/A')}")
    print(f"R² validation: {checkpoint.get('val_r2', 'N/A'):.6f}")
    
    return model, scaler_X, scaler_y

def evaluate_model_comprehensive(model, X_test, y_test, scaler_X, scaler_y):
    """Évaluation complète du modèle."""
    
    print("\n=== ÉVALUATION COMPLÈTE DU MODÈLE ===")
    
    # Normaliser les données de test
    X_test_scaled = scaler_X.transform(X_test)
    
    # Prédictions
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_pred_scaled = model(X_test_tensor).numpy()
    
    # Dénormaliser
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Métriques globales
    r2_global = r2_score(y_test, y_pred)
    rmse_global = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_global = mean_absolute_error(y_test, y_pred)
    
    # Métriques par paramètre
    r2_L = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_gap = r2_score(y_test[:, 1], y_pred[:, 1])
    
    rmse_L = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    
    mae_L = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_gap = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    
    # Erreurs relatives
    mape_L = np.mean(np.abs((y_test[:, 0] - y_pred[:, 0]) / y_test[:, 0])) * 100
    mape_gap = np.mean(np.abs((y_test[:, 1] - y_pred[:, 1]) / y_test[:, 1])) * 100
    
    metrics = {
        'r2_global': r2_global, 'rmse_global': rmse_global, 'mae_global': mae_global,
        'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'mape_L': mape_L, 'mape_gap': mape_gap
    }
    
    print(f"Métriques de performance:")
    print(f"  R² global: {r2_global:.6f}")
    print(f"  R² L_ecran: {r2_L:.6f}")
    print(f"  R² gap: {r2_gap:.6f}")
    print(f"  RMSE L_ecran: {rmse_L:.6f} µm")
    print(f"  RMSE gap: {rmse_gap:.6f} µm")
    print(f"  MAE L_ecran: {mae_L:.6f} µm")
    print(f"  MAE gap: {mae_gap:.6f} µm")
    print(f"  MAPE L_ecran: {mape_L:.2f}%")
    print(f"  MAPE gap: {mape_gap:.2f}%")
    
    return y_pred, metrics

def create_comprehensive_visualizations(X_train, y_train, X_test, y_test, y_pred, metrics, filenames):
    """Crée des visualisations complètes."""
    
    print("\n=== GÉNÉRATION DES VISUALISATIONS ===")
    
    # Configuration des graphiques
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Prédictions vs Vraies valeurs
    ax1 = plt.subplot(3, 4, 1)
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.7, s=50)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], 
             [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
    plt.xlabel('L_ecran vraie (µm)')
    plt.ylabel('L_ecran prédite (µm)')
    plt.title(f'L_ecran: R² = {metrics["r2_L"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 4, 2)
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.7, s=50)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], 
             [y_test[:, 1].min(), y_test[:, 1].max()], 'r--', lw=2)
    plt.xlabel('gap vraie (µm)')
    plt.ylabel('gap prédite (µm)')
    plt.title(f'gap: R² = {metrics["r2_gap"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    # 2. Erreurs absolues
    ax3 = plt.subplot(3, 4, 3)
    errors_L = np.abs(y_test[:, 0] - y_pred[:, 0])
    plt.hist(errors_L, bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('Erreur absolue L_ecran (µm)')
    plt.ylabel('Fréquence')
    plt.title(f'Distribution erreurs L_ecran\nMAE = {metrics["mae_L"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(3, 4, 4)
    errors_gap = np.abs(y_test[:, 1] - y_pred[:, 1])
    plt.hist(errors_gap, bins=15, alpha=0.7, edgecolor='black')
    plt.xlabel('Erreur absolue gap (µm)')
    plt.ylabel('Fréquence')
    plt.title(f'Distribution erreurs gap\nMAE = {metrics["mae_gap"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    # 3. Comparaison distributions train vs test
    ax5 = plt.subplot(3, 4, 5)
    plt.scatter(y_train[:, 0], y_train[:, 1], alpha=0.3, s=10, label='Entraînement', color='blue')
    plt.scatter(y_test[:, 0], y_test[:, 1], alpha=0.8, s=50, label='Test', color='red', marker='x')
    plt.xlabel('L_ecran (µm)')
    plt.ylabel('gap (µm)')
    plt.title('Distribution des paramètres')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Profils d'intensité moyens
    ax6 = plt.subplot(3, 4, 6)
    train_mean = X_train.mean(axis=0)
    test_mean = X_test.mean(axis=0)
    plt.plot(train_mean, label='Entraînement (simulé)', linewidth=2)
    plt.plot(test_mean, label='Test (expérimental)', linewidth=2)
    plt.xlabel('Position radiale')
    plt.ylabel('Intensité moyenne')
    plt.title('Profils moyens')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Erreurs relatives par valeur de gap
    ax7 = plt.subplot(3, 4, 7)
    gap_unique = np.unique(y_test[:, 1])
    gap_errors = []
    for gap_val in gap_unique:
        mask = y_test[:, 1] == gap_val
        if np.sum(mask) > 0:
            error = np.mean(np.abs(y_test[mask, 1] - y_pred[mask, 1]))
            gap_errors.append(error)
        else:
            gap_errors.append(0)
    
    plt.bar(range(len(gap_unique)), gap_errors, alpha=0.7)
    plt.xlabel('Valeur de gap')
    plt.ylabel('Erreur absolue moyenne')
    plt.title('Erreur par valeur de gap')
    plt.xticks(range(len(gap_unique)), [f'{g:.3f}' for g in gap_unique], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. Erreurs relatives par valeur de L_ecran
    ax8 = plt.subplot(3, 4, 8)
    L_unique = np.unique(y_test[:, 0])
    L_errors = []
    for L_val in L_unique:
        mask = y_test[:, 0] == L_val
        if np.sum(mask) > 0:
            error = np.mean(np.abs(y_test[mask, 0] - y_pred[mask, 0]))
            L_errors.append(error)
        else:
            L_errors.append(0)
    
    plt.bar(range(len(L_unique)), L_errors, alpha=0.7)
    plt.xlabel('Valeur de L_ecran')
    plt.ylabel('Erreur absolue moyenne')
    plt.title('Erreur par valeur de L_ecran')
    plt.xticks(range(len(L_unique)), [f'{l:.1f}' for l in L_unique], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 7. Matrice de corrélation des erreurs
    ax9 = plt.subplot(3, 4, 9)
    error_matrix = np.column_stack([errors_L, errors_gap])
    corr_matrix = np.corrcoef(error_matrix.T)
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks([0, 1], ['L_ecran', 'gap'])
    plt.yticks([0, 1], ['L_ecran', 'gap'])
    plt.title('Corrélation des erreurs')
    
    # 8. Résidus vs prédictions
    ax10 = plt.subplot(3, 4, 10)
    residuals_L = y_test[:, 0] - y_pred[:, 0]
    plt.scatter(y_pred[:, 0], residuals_L, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('L_ecran prédite')
    plt.ylabel('Résidus L_ecran')
    plt.title('Résidus vs Prédictions L_ecran')
    plt.grid(True, alpha=0.3)
    
    ax11 = plt.subplot(3, 4, 11)
    residuals_gap = y_test[:, 1] - y_pred[:, 1]
    plt.scatter(y_pred[:, 1], residuals_gap, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('gap prédite')
    plt.ylabel('Résidus gap')
    plt.title('Résidus vs Prédictions gap')
    plt.grid(True, alpha=0.3)
    
    # 9. Métriques résumées
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    metrics_text = f"""
RÉSUMÉ DES PERFORMANCES

R² Scores:
  Global: {metrics['r2_global']:.4f}
  L_ecran: {metrics['r2_L']:.4f}
  gap: {metrics['r2_gap']:.4f}

RMSE:
  L_ecran: {metrics['rmse_L']:.4f} µm
  gap: {metrics['rmse_gap']:.4f} µm

MAE:
  L_ecran: {metrics['mae_L']:.4f} µm
  gap: {metrics['mae_gap']:.4f} µm

MAPE:
  L_ecran: {metrics['mape_L']:.2f}%
  gap: {metrics['mape_gap']:.2f}%

Objectif R² > 0.8: {'✓ ATTEINT' if metrics['r2_global'] > 0.8 else '✗ NON ATTEINT'}
"""
    ax12.text(0.1, 0.9, metrics_text, transform=ax12.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Sauvegarder
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualisations sauvegardées: plots/comprehensive_evaluation.png")

def main():
    """Fonction principale d'évaluation complète."""
    
    print("="*80)
    print("ÉVALUATION COMPLÈTE ET ANALYSE DU MODÈLE")
    print("="*80)
    
    # Charger toutes les données
    X_train, y_train, X_test, y_test, filenames = load_all_data()
    
    # Charger le modèle entraîné
    model, scaler_X, scaler_y = load_trained_model()
    
    # Évaluation complète
    y_pred, metrics = evaluate_model_comprehensive(model, X_test, y_test, scaler_X, scaler_y)
    
    # Visualisations
    create_comprehensive_visualizations(X_train, y_train, X_test, y_test, y_pred, metrics, filenames)
    
    # Analyse et recommandations
    print(f"\n{'='*80}")
    print(f"ANALYSE ET RECOMMANDATIONS")
    print(f"{'='*80}")
    
    if metrics['r2_global'] > 0.8:
        print("✓ OBJECTIF ATTEINT: R² > 0.8")
        print("Le modèle performe bien sur les données expérimentales.")
    else:
        print("✗ OBJECTIF NON ATTEINT: R² < 0.8")
        print("\nProblèmes identifiés:")
        
        if metrics['r2_L'] > 0.8 and metrics['r2_gap'] < 0.5:
            print("- Le modèle prédit bien L_ecran mais mal gap")
            print("- Problème de généralisation pour le paramètre gap")
            
        print("\nRecommandations:")
        print("1. Augmenter la diversité des données d'entraînement")
        print("2. Utiliser des techniques de domain adaptation")
        print("3. Ajouter de la régularisation spécifique pour gap")
        print("4. Considérer un modèle multi-tâches avec poids adaptatifs")
        print("5. Analyser les différences entre données simulées et expérimentales")
    
    print(f"\nRapport complet généré avec succès!")

if __name__ == "__main__":
    main()
