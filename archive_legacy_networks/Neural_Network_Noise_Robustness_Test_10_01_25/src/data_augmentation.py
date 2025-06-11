#!/usr/bin/env python3
"""
Augmentation de données pour améliorer la robustesse au bruit

Ce script implémente différentes techniques d'augmentation de données
pour améliorer la robustesse du modèle face au bruit.

Auteur: Oussama GUELFAA
Date: 10 - 01 - 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
import pandas as pd

def interpolate_between_profiles(X, y, interpolation_factor=2):
    """
    Augmente les données par interpolation entre profils voisins.
    
    Args:
        X (np.ndarray): Profils d'intensité originaux (n_samples, n_features)
        y (np.ndarray): Valeurs de gap correspondantes (n_samples,)
        interpolation_factor (int): Facteur de multiplication des données
        
    Returns:
        tuple: (X_augmented, y_augmented) données augmentées
    """
    print(f"=== AUGMENTATION PAR INTERPOLATION ===")
    print(f"Données originales: {X.shape}")
    print(f"Facteur d'interpolation: {interpolation_factor}")
    
    # Trier par valeur de gap pour interpolation cohérente
    sort_indices = np.argsort(y)
    X_sorted = X[sort_indices]
    y_sorted = y[sort_indices]
    
    X_augmented = [X_sorted]
    y_augmented = [y_sorted]
    
    # Générer des interpolations entre profils adjacents
    for factor in range(1, interpolation_factor):
        alpha = factor / interpolation_factor
        
        # Interpolation linéaire entre profils adjacents
        X_interp = alpha * X_sorted[1:] + (1 - alpha) * X_sorted[:-1]
        y_interp = alpha * y_sorted[1:] + (1 - alpha) * y_sorted[:-1]
        
        X_augmented.append(X_interp)
        y_augmented.append(y_interp)
    
    # Combiner toutes les données
    X_final = np.vstack(X_augmented)
    y_final = np.hstack(y_augmented)
    
    print(f"Données augmentées: {X_final.shape}")
    print(f"Facteur de multiplication réel: {len(X_final) / len(X):.1f}x")
    
    return X_final, y_final

def add_smooth_variations(X, y, variation_factor=0.02, n_variations=2):
    """
    Ajoute des variations lisses aux profils pour augmentation.
    
    Args:
        X (np.ndarray): Profils originaux
        y (np.ndarray): Valeurs de gap
        variation_factor (float): Amplitude des variations (fraction du signal)
        n_variations (int): Nombre de variations par profil original
        
    Returns:
        tuple: (X_augmented, y_augmented)
    """
    print(f"=== AUGMENTATION PAR VARIATIONS LISSES ===")
    print(f"Amplitude des variations: {variation_factor*100:.1f}%")
    print(f"Variations par profil: {n_variations}")
    
    X_augmented = [X]
    y_augmented = [y]
    
    for var_idx in range(n_variations):
        X_varied = X.copy()
        
        for i in range(len(X)):
            # Générer une variation lisse (sinusoïdale)
            x_coords = np.linspace(0, 2*np.pi, X.shape[1])
            
            # Combinaison de plusieurs fréquences pour variation naturelle
            variation = (
                variation_factor * np.sin(x_coords * (1 + var_idx)) +
                variation_factor * 0.5 * np.sin(x_coords * (3 + var_idx)) +
                variation_factor * 0.25 * np.sin(x_coords * (7 + var_idx))
            )
            
            # Appliquer la variation proportionnellement au signal
            X_varied[i] = X[i] * (1 + variation)
        
        X_augmented.append(X_varied)
        y_augmented.append(y)
    
    X_final = np.vstack(X_augmented)
    y_final = np.hstack(y_augmented)
    
    print(f"Données avec variations: {X_final.shape}")
    
    return X_final, y_final

def elastic_deformation(X, y, sigma=0.1, alpha=1.0, n_deformations=1):
    """
    Applique des déformations élastiques aux profils.
    
    Args:
        X (np.ndarray): Profils originaux
        y (np.ndarray): Valeurs de gap
        sigma (float): Écart-type du lissage gaussien
        alpha (float): Amplitude de la déformation
        n_deformations (int): Nombre de déformations par profil
        
    Returns:
        tuple: (X_augmented, y_augmented)
    """
    print(f"=== AUGMENTATION PAR DÉFORMATIONS ÉLASTIQUES ===")
    print(f"Paramètres: sigma={sigma}, alpha={alpha}")
    
    from scipy.ndimage import gaussian_filter1d
    
    X_augmented = [X]
    y_augmented = [y]
    
    for def_idx in range(n_deformations):
        X_deformed = X.copy()
        
        for i in range(len(X)):
            # Générer un champ de déplacement aléatoire
            displacement = np.random.randn(X.shape[1]) * alpha
            
            # Lisser le champ de déplacement
            displacement = gaussian_filter1d(displacement, sigma * X.shape[1] / 10)
            
            # Appliquer la déformation par interpolation
            original_indices = np.arange(X.shape[1])
            deformed_indices = original_indices + displacement
            
            # S'assurer que les indices restent dans les limites
            deformed_indices = np.clip(deformed_indices, 0, X.shape[1] - 1)
            
            # Interpoler
            f = interpolate.interp1d(original_indices, X[i], 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            X_deformed[i] = f(deformed_indices)
        
        X_augmented.append(X_deformed)
        y_augmented.append(y)
    
    X_final = np.vstack(X_augmented)
    y_final = np.hstack(y_augmented)
    
    print(f"Données avec déformations: {X_final.shape}")
    
    return X_final, y_final

def augment_dataset_comprehensive(X, y, augmentation_config=None):
    """
    Applique une augmentation complète du dataset.
    
    Args:
        X, y: Données originales
        augmentation_config (dict): Configuration de l'augmentation
        
    Returns:
        tuple: (X_augmented, y_augmented)
    """
    if augmentation_config is None:
        augmentation_config = {
            'interpolation_factor': 2,
            'smooth_variations': True,
            'variation_factor': 0.02,
            'n_variations': 1,
            'elastic_deformation': True,
            'elastic_sigma': 0.1,
            'elastic_alpha': 0.5,
            'n_deformations': 1
        }
    
    print(f"=== AUGMENTATION COMPLÈTE DU DATASET ===")
    print(f"Configuration: {augmentation_config}")
    
    X_aug, y_aug = X.copy(), y.copy()
    
    # 1. Interpolation entre profils
    if augmentation_config.get('interpolation_factor', 1) > 1:
        X_aug, y_aug = interpolate_between_profiles(
            X_aug, y_aug, 
            augmentation_config['interpolation_factor']
        )
    
    # 2. Variations lisses
    if augmentation_config.get('smooth_variations', False):
        X_aug, y_aug = add_smooth_variations(
            X_aug, y_aug,
            augmentation_config.get('variation_factor', 0.02),
            augmentation_config.get('n_variations', 1)
        )
    
    # 3. Déformations élastiques
    if augmentation_config.get('elastic_deformation', False):
        X_aug, y_aug = elastic_deformation(
            X_aug, y_aug,
            augmentation_config.get('elastic_sigma', 0.1),
            augmentation_config.get('elastic_alpha', 0.5),
            augmentation_config.get('n_deformations', 1)
        )
    
    print(f"\n=== RÉSUMÉ AUGMENTATION ===")
    print(f"Données originales: {X.shape}")
    print(f"Données augmentées: {X_aug.shape}")
    print(f"Facteur de multiplication: {len(X_aug) / len(X):.1f}x")
    
    return X_aug, y_aug

def visualize_augmentation_examples(X_original, X_augmented, y_original, y_augmented, 
                                  n_examples=6):
    """
    Visualise des exemples d'augmentation de données.
    
    Args:
        X_original, X_augmented: Données avant/après augmentation
        y_original, y_augmented: Labels correspondants
        n_examples: Nombre d'exemples à afficher
    """
    print(f"=== VISUALISATION DES EXEMPLES D'AUGMENTATION ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Sélectionner des exemples représentatifs
    original_indices = np.linspace(0, len(X_original)-1, n_examples, dtype=int)
    
    for i, orig_idx in enumerate(original_indices):
        if i < len(axes):
            ax = axes[i]
            
            # Profil original
            ax.plot(X_original[orig_idx], 'b-', linewidth=2, label='Original', alpha=0.8)
            
            # Trouver des profils augmentés correspondants (même gap approximatif)
            gap_orig = y_original[orig_idx]
            gap_tolerance = 0.01  # Tolérance pour trouver des gaps similaires
            
            # Indices des profils augmentés avec gap similaire
            similar_indices = np.where(np.abs(y_augmented - gap_orig) <= gap_tolerance)[0]
            
            # Exclure l'original s'il est présent
            if len(similar_indices) > 1:
                # Prendre quelques exemples augmentés
                aug_examples = similar_indices[1:min(4, len(similar_indices))]
                
                for j, aug_idx in enumerate(aug_examples):
                    alpha = 0.6 - j * 0.1
                    ax.plot(X_augmented[aug_idx], '--', linewidth=1.5, 
                           alpha=alpha, label=f'Augmenté {j+1}')
            
            ax.set_title(f'Gap = {gap_orig:.3f} µm')
            ax.set_xlabel('Position radiale')
            ax.set_ylabel('Intensité')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../plots/augmentation_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Exemples d'augmentation sauvegardés: ../plots/augmentation_examples.png")

def compare_augmentation_strategies(X, y, strategies_config):
    """
    Compare différentes stratégies d'augmentation.
    
    Args:
        X, y: Données originales
        strategies_config (dict): Configurations à comparer
        
    Returns:
        dict: Résultats de comparaison
    """
    print(f"=== COMPARAISON DES STRATÉGIES D'AUGMENTATION ===")
    
    results = {}
    
    for strategy_name, config in strategies_config.items():
        print(f"\nTestage de la stratégie: {strategy_name}")
        
        X_aug, y_aug = augment_dataset_comprehensive(X, y, config)
        
        results[strategy_name] = {
            'config': config,
            'original_size': len(X),
            'augmented_size': len(X_aug),
            'multiplication_factor': len(X_aug) / len(X),
            'X_augmented': X_aug,
            'y_augmented': y_aug
        }
        
        print(f"  Taille finale: {len(X_aug)} échantillons")
        print(f"  Facteur: {len(X_aug) / len(X):.1f}x")
    
    return results

if __name__ == "__main__":
    print("=== TEST DES TECHNIQUES D'AUGMENTATION ===")
    
    # Exemple avec des données synthétiques
    np.random.seed(42)
    
    # Générer des données d'exemple
    n_samples = 50
    n_features = 1000
    
    X_example = np.random.randn(n_samples, n_features)
    y_example = np.linspace(0.1, 2.0, n_samples)
    
    # Tester l'augmentation
    X_aug, y_aug = augment_dataset_comprehensive(X_example, y_example)
    
    # Visualiser
    visualize_augmentation_examples(X_example, X_aug, y_example, y_aug)
    
    print("Test d'augmentation terminé!")
