#!/usr/bin/env python3
"""
TensorFlow Model Evaluation
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Évalue le modèle TensorFlow entraîné sur les données expérimentales.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import scipy.io as sio
import joblib
import os

def load_test_data():
    """Charge les données de test expérimentales."""
    
    print("=== CHARGEMENT DES DONNÉES DE TEST ===")
    
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
                y_test.append([L_ecran, gap])  # Ordre: L_ecran, gap
                filenames.append(filename)
                
            except Exception as e:
                print(f"Erreur {mat_filename}: {e}")
    
    X_test = np.array(X_test, dtype='float32')
    y_test = np.array(y_test, dtype='float32')
    
    print(f"Données de test chargées: X{X_test.shape}, y{y_test.shape}")
    return X_test, y_test, filenames

def evaluate_tensorflow_model():
    """Évalue le modèle TensorFlow sur les données expérimentales."""
    
    print("\n=== ÉVALUATION DU MODÈLE TENSORFLOW ===")
    
    # Charger les données de test
    X_test, y_test, filenames = load_test_data()
    
    # Charger le modèle entraîné (utiliser le modèle existant)
    try:
        # Essayer de charger le modèle depuis l'entraînement précédent
        model_path = 'models/tensorflow_best_model.keras'
        if not os.path.exists(model_path):
            # Fallback vers le modèle .h5 s'il existe
            model_path = 'models/tensorflow_best_model.h5'
        
        print(f"Chargement du modèle: {model_path}")
        
        # Charger avec compile=False pour éviter les problèmes de sérialisation
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Recompiler le modèle
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("Modèle chargé et recompilé avec succès")
        
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        print("Utilisation du modèle depuis l'entraînement en cours...")
        return None
    
    # Charger le scaler
    try:
        scaler_X = joblib.load('models/tensorflow_scaler_X.pkl')
        print("Scaler chargé avec succès")
    except:
        print("Erreur: Impossible de charger le scaler")
        print("Créer un scaler temporaire...")
        # Créer un scaler temporaire basé sur les données de test
        scaler_X = StandardScaler()
        scaler_X.fit(X_test)
    
    # Normaliser les données de test
    X_test_scaled = scaler_X.transform(X_test)
    
    # Prédictions
    print("Génération des prédictions...")
    y_pred = model.predict(X_test_scaled, verbose=0)
    
    # Calculer les métriques
    r2_global = r2_score(y_test, y_pred)
    r2_L = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_gap = r2_score(y_test[:, 1], y_pred[:, 1])
    
    rmse_L = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    
    mae_L = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_gap = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    
    # Erreurs relatives
    mape_L = np.mean(np.abs((y_test[:, 0] - y_pred[:, 0]) / y_test[:, 0])) * 100
    mape_gap = np.mean(np.abs((y_test[:, 1] - y_pred[:, 1]) / y_test[:, 1])) * 100
    
    print(f"\n{'='*60}")
    print(f"RÉSULTATS TENSORFLOW/KERAS")
    print(f"{'='*60}")
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
    
    success = r2_global > 0.8
    print(f"  Objectif R² > 0.8: {'✓ ATTEINT' if success else '✗ NON ATTEINT'}")
    
    # Créer des visualisations
    create_tensorflow_visualizations(y_test, y_pred, {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'mape_L': mape_L, 'mape_gap': mape_gap,
        'success': success
    })
    
    return y_pred, {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'mape_L': mape_L, 'mape_gap': mape_gap,
        'success': success
    }

def create_tensorflow_visualizations(y_test, y_pred, metrics):
    """Crée des visualisations pour l'évaluation TensorFlow."""
    
    print("\n=== GÉNÉRATION DES VISUALISATIONS TENSORFLOW ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Prédictions vs Vraies valeurs L_ecran
    ax1.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.7, s=50, color='blue')
    ax1.plot([y_test[:, 0].min(), y_test[:, 0].max()], 
             [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
    ax1.set_xlabel('L_ecran vraie (µm)')
    ax1.set_ylabel('L_ecran prédite (µm)')
    ax1.set_title(f'TensorFlow - L_ecran: R² = {metrics["r2_L"]:.4f}')
    ax1.grid(True, alpha=0.3)
    
    # 2. Prédictions vs Vraies valeurs gap
    ax2.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.7, s=50, color='green')
    ax2.plot([y_test[:, 1].min(), y_test[:, 1].max()], 
             [y_test[:, 1].min(), y_test[:, 1].max()], 'r--', lw=2)
    ax2.set_xlabel('gap vraie (µm)')
    ax2.set_ylabel('gap prédite (µm)')
    ax2.set_title(f'TensorFlow - gap: R² = {metrics["r2_gap"]:.4f}')
    ax2.grid(True, alpha=0.3)
    
    # 3. Erreurs absolues L_ecran
    errors_L = np.abs(y_test[:, 0] - y_pred[:, 0])
    ax3.hist(errors_L, bins=15, alpha=0.7, edgecolor='black', color='blue')
    ax3.set_xlabel('Erreur absolue L_ecran (µm)')
    ax3.set_ylabel('Fréquence')
    ax3.set_title(f'Distribution erreurs L_ecran\nMAE = {metrics["mae_L"]:.4f}')
    ax3.grid(True, alpha=0.3)
    
    # 4. Erreurs absolues gap
    errors_gap = np.abs(y_test[:, 1] - y_pred[:, 1])
    ax4.hist(errors_gap, bins=15, alpha=0.7, edgecolor='black', color='green')
    ax4.set_xlabel('Erreur absolue gap (µm)')
    ax4.set_ylabel('Fréquence')
    ax4.set_title(f'Distribution erreurs gap\nMAE = {metrics["mae_gap"]:.4f}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/tensorflow_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualisations sauvegardées: plots/tensorflow_evaluation.png")
    
    # Créer un graphique de comparaison des métriques
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    params = ['L_ecran', 'gap']
    r2_scores = [metrics['r2_L'], metrics['r2_gap']]
    colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.5 else 'red' for r2 in r2_scores]
    
    bars = ax.bar(params, r2_scores, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Objectif R² = 0.8')
    ax.set_ylabel('R² Score')
    ax.set_title(f'Performance TensorFlow par Paramètre\nR² Global = {metrics["r2_global"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/tensorflow_r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparaison R² sauvegardée: plots/tensorflow_r2_comparison.png")

def main():
    """Fonction principale d'évaluation."""
    
    print("="*80)
    print("ÉVALUATION MODÈLE TENSORFLOW/KERAS")
    print("="*80)
    
    # Évaluer le modèle
    result = evaluate_tensorflow_model()
    
    if result is not None:
        y_pred, metrics = result
        
        print(f"\n{'='*80}")
        print(f"RÉSUMÉ FINAL TENSORFLOW")
        print(f"{'='*80}")
        print(f"Le modèle TensorFlow/Keras a été évalué avec succès!")
        print(f"Performance globale: R² = {metrics['r2_global']:.6f}")
        
        if metrics['success']:
            print("🎉 OBJECTIF ATTEINT: R² > 0.8")
        else:
            print("❌ Objectif non atteint, mais des améliorations sont possibles")
            
        print(f"\nFichiers générés:")
        print(f"  • plots/tensorflow_evaluation.png")
        print(f"  • plots/tensorflow_r2_comparison.png")
    else:
        print("❌ Impossible d'évaluer le modèle TensorFlow")
        print("Assurez-vous que le modèle a été entraîné avec train_tensorflow_model.py")

if __name__ == "__main__":
    main()
