#!/usr/bin/env python3
"""
Data Analysis Script
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Analyse les données pour comprendre les problèmes de convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

def analyze_training_data():
    """Analyse détaillée des données d'entraînement."""
    
    print("=== ANALYSE DES DONNÉES D'ENTRAÎNEMENT ===")
    
    # Charger les données
    data = np.load('processed_data/training_data.npz', allow_pickle=True)
    X = data['X']
    y = data['y']
    metadata = data['metadata'].item()
    
    print(f"\nInformations générales:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Nombre d'échantillons: {len(X)}")
    
    # Analyse des features (profils d'intensité)
    print(f"\nAnalyse des features (profils d'intensité):")
    print(f"  Min: {X.min():.6f}")
    print(f"  Max: {X.max():.6f}")
    print(f"  Mean: {X.mean():.6f}")
    print(f"  Std: {X.std():.6f}")
    print(f"  Médiane: {np.median(X):.6f}")
    print(f"  Valeurs nulles: {np.sum(X == 0)}")
    print(f"  Valeurs NaN: {np.sum(np.isnan(X))}")
    
    # Analyse des targets
    print(f"\nAnalyse des targets:")
    for i, param in enumerate(['L_ecran', 'gap']):
        vals = y[:, i]
        print(f"  {param}:")
        print(f"    Min: {vals.min():.6f}")
        print(f"    Max: {vals.max():.6f}")
        print(f"    Mean: {vals.mean():.6f}")
        print(f"    Std: {vals.std():.6f}")
        print(f"    Valeurs uniques: {len(np.unique(vals))}")
    
    # Vérifier la distribution des paramètres
    print(f"\nDistribution des paramètres:")
    L_ecran_unique = np.unique(y[:, 0])
    gap_unique = np.unique(y[:, 1])
    print(f"  L_ecran valeurs: {L_ecran_unique}")
    print(f"  gap valeurs: {len(gap_unique)} valeurs de {gap_unique.min():.3f} à {gap_unique.max():.3f}")
    
    # Créer des visualisations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Distribution des paramètres
    axes[0, 0].hist(y[:, 0], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution L_ecran')
    axes[0, 0].set_xlabel('L_ecran')
    axes[0, 0].set_ylabel('Fréquence')
    
    axes[0, 1].hist(y[:, 1], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribution gap')
    axes[0, 1].set_xlabel('gap')
    axes[0, 1].set_ylabel('Fréquence')
    
    # Scatter plot des paramètres
    axes[0, 2].scatter(y[:, 0], y[:, 1], alpha=0.6, s=20)
    axes[0, 2].set_xlabel('L_ecran')
    axes[0, 2].set_ylabel('gap')
    axes[0, 2].set_title('Relation L_ecran vs gap')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Analyse des profils d'intensité
    # Moyennes des profils par paramètre
    sample_indices = np.random.choice(len(X), 100, replace=False)
    for idx in sample_indices[:10]:
        axes[1, 0].plot(X[idx], alpha=0.3, linewidth=0.5)
    axes[1, 0].set_title('Échantillons de profils d\'intensité')
    axes[1, 0].set_xlabel('Position radiale')
    axes[1, 0].set_ylabel('Intensité normalisée')
    
    # Distribution des valeurs d'intensité
    axes[1, 1].hist(X.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution des intensités')
    axes[1, 1].set_xlabel('Intensité')
    axes[1, 1].set_ylabel('Fréquence')
    axes[1, 1].set_yscale('log')
    
    # Corrélation entre profils moyens et paramètres
    profile_means = X.mean(axis=1)
    axes[1, 2].scatter(profile_means, y[:, 0], alpha=0.6, s=20, label='L_ecran')
    axes[1, 2].scatter(profile_means, y[:, 1], alpha=0.6, s=20, label='gap')
    axes[1, 2].set_xlabel('Intensité moyenne du profil')
    axes[1, 2].set_ylabel('Valeur du paramètre')
    axes[1, 2].set_title('Corrélation intensité-paramètres')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyse de corrélation détaillée
    print(f"\nAnalyse de corrélation:")
    profile_stats = {
        'mean': X.mean(axis=1),
        'std': X.std(axis=1),
        'max': X.max(axis=1),
        'min': X.min(axis=1)
    }
    
    for stat_name, stat_values in profile_stats.items():
        corr_L = np.corrcoef(stat_values, y[:, 0])[0, 1]
        corr_gap = np.corrcoef(stat_values, y[:, 1])[0, 1]
        print(f"  Corrélation {stat_name} avec L_ecran: {corr_L:.4f}")
        print(f"  Corrélation {stat_name} avec gap: {corr_gap:.4f}")
    
    # Test de différentes normalisations
    print(f"\nTest de différentes normalisations:")
    
    # StandardScaler
    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(X)
    print(f"  StandardScaler - range: [{X_std.min():.3f}, {X_std.max():.3f}]")
    
    # RobustScaler
    scaler_rob = RobustScaler()
    X_rob = scaler_rob.fit_transform(X)
    print(f"  RobustScaler - range: [{X_rob.min():.3f}, {X_rob.max():.3f}]")
    
    # Min-Max scaling
    X_minmax = (X - X.min()) / (X.max() - X.min())
    print(f"  MinMaxScaler - range: [{X_minmax.min():.3f}, {X_minmax.max():.3f}]")
    
    return X, y, metadata

def analyze_model_complexity():
    """Analyse la complexité nécessaire du modèle."""
    
    print(f"\n=== ANALYSE DE LA COMPLEXITÉ DU MODÈLE ===")
    
    # Charger les données
    data = np.load('processed_data/training_data.npz', allow_pickle=True)
    X = data['X']
    y = data['y']
    
    # Calculer la complexité intrinsèque des données
    from sklearn.decomposition import PCA
    
    # PCA sur les features
    pca_X = PCA()
    X_pca = pca_X.fit_transform(X)
    
    # Variance expliquée
    cumsum_var = np.cumsum(pca_X.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    n_components_99 = np.argmax(cumsum_var >= 0.99) + 1
    
    print(f"Analyse PCA des features:")
    print(f"  Composantes pour 95% variance: {n_components_95}")
    print(f"  Composantes pour 99% variance: {n_components_99}")
    print(f"  Variance des 10 premières composantes: {pca_X.explained_variance_ratio_[:10]}")
    
    # Analyse de la séparabilité des classes
    print(f"\nAnalyse de la séparabilité:")
    
    # Créer des bins pour les paramètres
    L_bins = np.digitize(y[:, 0], bins=np.linspace(y[:, 0].min(), y[:, 0].max(), 10))
    gap_bins = np.digitize(y[:, 1], bins=np.linspace(y[:, 1].min(), y[:, 1].max(), 10))
    
    # Calculer la distance inter-classe vs intra-classe
    from sklearn.metrics import silhouette_score
    
    try:
        sil_L = silhouette_score(X, L_bins)
        sil_gap = silhouette_score(X, gap_bins)
        print(f"  Silhouette score L_ecran: {sil_L:.4f}")
        print(f"  Silhouette score gap: {sil_gap:.4f}")
    except:
        print("  Impossible de calculer le silhouette score")
    
    # Recommandations
    print(f"\nRecommandations:")
    if n_components_95 < 100:
        print(f"  - Les données peuvent être réduites à {n_components_95} dimensions")
        print(f"  - Un modèle plus simple pourrait suffire")
    else:
        print(f"  - Les données sont complexes ({n_components_95} composantes)")
        print(f"  - Un modèle profond est justifié")

if __name__ == "__main__":
    X, y, metadata = analyze_training_data()
    analyze_model_complexity()
    
    print(f"\n=== RECOMMANDATIONS ===")
    print(f"1. Vérifier la qualité des données d'entrée")
    print(f"2. Essayer différentes architectures de modèle")
    print(f"3. Ajuster les hyperparamètres de normalisation")
    print(f"4. Considérer des techniques de régularisation")
    print(f"5. Analyser les profils qui posent problème")
