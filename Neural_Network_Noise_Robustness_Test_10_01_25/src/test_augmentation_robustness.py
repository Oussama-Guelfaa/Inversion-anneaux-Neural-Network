#!/usr/bin/env python3
"""
Test de l'effet de l'augmentation de données sur la robustesse au bruit

Ce script compare les performances avec et sans augmentation de données
pour évaluer l'amélioration de la robustesse au bruit.

Auteur: Oussama GUELFAA
Date: 10 - 01 - 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split

# Importer les modules locaux
from noise_robustness_test import (
    load_dataset, add_gaussian_noise, train_model_with_noise, 
    evaluate_model, RobustGapPredictor
)
from data_augmentation import augment_dataset_comprehensive

def compare_with_without_augmentation(X, y, noise_levels, augmentation_config=None):
    """
    Compare les performances avec et sans augmentation de données.
    
    Args:
        X, y: Données originales
        noise_levels: Niveaux de bruit à tester
        augmentation_config: Configuration de l'augmentation
        
    Returns:
        dict: Résultats comparatifs
    """
    print("=== COMPARAISON AVEC/SANS AUGMENTATION ===")
    
    # Configuration par défaut pour l'augmentation
    if augmentation_config is None:
        augmentation_config = {
            'interpolation_factor': 2,
            'smooth_variations': True,
            'variation_factor': 0.01,
            'n_variations': 1,
            'elastic_deformation': False  # Désactivé pour ce test initial
        }
    
    # Division des données originales
    X_train_orig, X_temp, y_train_orig, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val_orig, X_test, y_val_orig, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Division originale:")
    print(f"  Train: {len(X_train_orig)} échantillons")
    print(f"  Val: {len(X_val_orig)} échantillons") 
    print(f"  Test: {len(X_test)} échantillons")
    
    # Augmentation des données d'entraînement
    print(f"\nAugmentation des données d'entraînement...")
    X_train_aug, y_train_aug = augment_dataset_comprehensive(
        X_train_orig, y_train_orig, augmentation_config
    )
    
    print(f"Données d'entraînement augmentées:")
    print(f"  Avant: {len(X_train_orig)} échantillons")
    print(f"  Après: {len(X_train_aug)} échantillons")
    print(f"  Facteur: {len(X_train_aug) / len(X_train_orig):.1f}x")
    
    results = {
        'without_augmentation': {},
        'with_augmentation': {},
        'comparison': {}
    }
    
    # Test pour chaque niveau de bruit
    for noise_level in noise_levels:
        print(f"\n{'='*50}")
        print(f"TEST AVEC {noise_level}% DE BRUIT")
        print(f"{'='*50}")
        
        # 1. Sans augmentation
        print(f"\n--- SANS AUGMENTATION ---")
        model_orig, scaler_orig, history_orig, time_orig = train_model_with_noise(
            X_train_orig, y_train_orig, X_val_orig, y_val_orig, 
            noise_level, max_epochs=150, batch_size=16
        )
        metrics_orig = evaluate_model(model_orig, scaler_orig, X_test, y_test)
        
        results['without_augmentation'][noise_level] = {
            'metrics': metrics_orig,
            'history': history_orig,
            'training_time': time_orig,
            'epochs': len(history_orig['train_loss'])
        }
        
        # 2. Avec augmentation
        print(f"\n--- AVEC AUGMENTATION ---")
        model_aug, scaler_aug, history_aug, time_aug = train_model_with_noise(
            X_train_aug, y_train_aug, X_val_orig, y_val_orig,
            noise_level, max_epochs=150, batch_size=16
        )
        metrics_aug = evaluate_model(model_aug, scaler_aug, X_test, y_test)
        
        results['with_augmentation'][noise_level] = {
            'metrics': metrics_aug,
            'history': history_aug,
            'training_time': time_aug,
            'epochs': len(history_aug['train_loss'])
        }
        
        # 3. Comparaison
        r2_improvement = metrics_aug['r2'] - metrics_orig['r2']
        rmse_improvement = metrics_orig['rmse'] - metrics_aug['rmse']  # Positif = amélioration
        
        results['comparison'][noise_level] = {
            'r2_improvement': r2_improvement,
            'rmse_improvement': rmse_improvement,
            'r2_improvement_percent': (r2_improvement / metrics_orig['r2']) * 100,
            'rmse_improvement_percent': (rmse_improvement / metrics_orig['rmse']) * 100
        }
        
        print(f"\nRésultats pour {noise_level}% bruit:")
        print(f"  Sans augmentation: R² = {metrics_orig['r2']:.4f}, RMSE = {metrics_orig['rmse']:.4f}")
        print(f"  Avec augmentation: R² = {metrics_aug['r2']:.4f}, RMSE = {metrics_aug['rmse']:.4f}")
        print(f"  Amélioration R²: {r2_improvement:+.4f} ({r2_improvement/metrics_orig['r2']*100:+.1f}%)")
        print(f"  Amélioration RMSE: {rmse_improvement:+.4f} ({rmse_improvement/metrics_orig['rmse']*100:+.1f}%)")
    
    return results, X_test, y_test

def plot_augmentation_comparison(results, noise_levels):
    """
    Crée des graphiques comparatifs des résultats avec/sans augmentation.
    
    Args:
        results: Résultats de la comparaison
        noise_levels: Niveaux de bruit testés
    """
    print("\n=== GÉNÉRATION DES GRAPHIQUES COMPARATIFS ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extraire les données pour les graphiques
    r2_orig = [results['without_augmentation'][noise]['metrics']['r2'] for noise in noise_levels]
    r2_aug = [results['with_augmentation'][noise]['metrics']['r2'] for noise in noise_levels]
    
    rmse_orig = [results['without_augmentation'][noise]['metrics']['rmse'] for noise in noise_levels]
    rmse_aug = [results['with_augmentation'][noise]['metrics']['rmse'] for noise in noise_levels]
    
    time_orig = [results['without_augmentation'][noise]['training_time'] for noise in noise_levels]
    time_aug = [results['with_augmentation'][noise]['training_time'] for noise in noise_levels]
    
    # 1. Comparaison R²
    axes[0, 0].plot(noise_levels, r2_orig, 'b-o', linewidth=2, label='Sans augmentation')
    axes[0, 0].plot(noise_levels, r2_aug, 'r-o', linewidth=2, label='Avec augmentation')
    axes[0, 0].axhline(y=0.8, color='gray', linestyle='--', alpha=0.7, label='Objectif R² = 0.8')
    axes[0, 0].set_xlabel('Niveau de bruit (%)')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Comparaison R² Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Comparaison RMSE
    axes[0, 1].plot(noise_levels, rmse_orig, 'b-o', linewidth=2, label='Sans augmentation')
    axes[0, 1].plot(noise_levels, rmse_aug, 'r-o', linewidth=2, label='Avec augmentation')
    axes[0, 1].set_xlabel('Niveau de bruit (%)')
    axes[0, 1].set_ylabel('RMSE (µm)')
    axes[0, 1].set_title('Comparaison RMSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Amélioration relative R²
    r2_improvements = [results['comparison'][noise]['r2_improvement_percent'] for noise in noise_levels]
    axes[0, 2].bar(noise_levels, r2_improvements, alpha=0.7, color='green')
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 2].set_xlabel('Niveau de bruit (%)')
    axes[0, 2].set_ylabel('Amélioration R² (%)')
    axes[0, 2].set_title('Amélioration Relative R²')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Amélioration relative RMSE
    rmse_improvements = [results['comparison'][noise]['rmse_improvement_percent'] for noise in noise_levels]
    axes[1, 0].bar(noise_levels, rmse_improvements, alpha=0.7, color='orange')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].set_xlabel('Niveau de bruit (%)')
    axes[1, 0].set_ylabel('Amélioration RMSE (%)')
    axes[1, 0].set_title('Amélioration Relative RMSE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Temps d'entraînement
    axes[1, 1].plot(noise_levels, time_orig, 'b-o', linewidth=2, label='Sans augmentation')
    axes[1, 1].plot(noise_levels, time_aug, 'r-o', linewidth=2, label='Avec augmentation')
    axes[1, 1].set_xlabel('Niveau de bruit (%)')
    axes[1, 1].set_ylabel('Temps d\'entraînement (s)')
    axes[1, 1].set_title('Comparaison Temps d\'Entraînement')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Résumé des améliorations
    axes[1, 2].scatter(r2_improvements, rmse_improvements, s=100, alpha=0.7)
    for i, noise in enumerate(noise_levels):
        axes[1, 2].annotate(f'{noise}%', 
                           (r2_improvements[i], rmse_improvements[i]),
                           xytext=(5, 5), textcoords='offset points')
    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 2].axvline(x=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 2].set_xlabel('Amélioration R² (%)')
    axes[1, 2].set_ylabel('Amélioration RMSE (%)')
    axes[1, 2].set_title('Corrélation des Améliorations')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../plots/augmentation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Graphiques comparatifs sauvegardés: ../plots/augmentation_comparison.png")

def analyze_augmentation_benefits(results, noise_levels):
    """
    Analyse les bénéfices de l'augmentation de données.
    
    Args:
        results: Résultats de la comparaison
        noise_levels: Niveaux de bruit testés
    """
    print("\n" + "="*60)
    print("ANALYSE DES BÉNÉFICES DE L'AUGMENTATION")
    print("="*60)
    
    # Tableau comparatif
    print(f"\n📊 COMPARAISON DÉTAILLÉE:")
    print(f"{'Bruit':<6} {'R² Orig':<8} {'R² Aug':<8} {'Δ R²':<8} {'RMSE Orig':<10} {'RMSE Aug':<10} {'Δ RMSE':<10}")
    print("-" * 70)
    
    total_r2_improvement = 0
    total_rmse_improvement = 0
    significant_improvements = 0
    
    for noise_level in noise_levels:
        r2_orig = results['without_augmentation'][noise_level]['metrics']['r2']
        r2_aug = results['with_augmentation'][noise_level]['metrics']['r2']
        rmse_orig = results['without_augmentation'][noise_level]['metrics']['rmse']
        rmse_aug = results['with_augmentation'][noise_level]['metrics']['rmse']
        
        r2_delta = r2_aug - r2_orig
        rmse_delta = rmse_orig - rmse_aug  # Positif = amélioration
        
        total_r2_improvement += r2_delta
        total_rmse_improvement += rmse_delta
        
        if r2_delta > 0.01:  # Amélioration significative
            significant_improvements += 1
        
        print(f"{noise_level:>4}%  "
              f"{r2_orig:>6.3f}   "
              f"{r2_aug:>6.3f}   "
              f"{r2_delta:>+6.3f}   "
              f"{rmse_orig:>8.4f}   "
              f"{rmse_aug:>8.4f}   "
              f"{rmse_delta:>+8.4f}")
    
    # Analyse globale
    avg_r2_improvement = total_r2_improvement / len(noise_levels)
    avg_rmse_improvement = total_rmse_improvement / len(noise_levels)
    
    print(f"\n🎯 ANALYSE GLOBALE:")
    print(f"  Amélioration moyenne R²: {avg_r2_improvement:+.4f}")
    print(f"  Amélioration moyenne RMSE: {avg_rmse_improvement:+.4f} µm")
    print(f"  Niveaux avec amélioration significative: {significant_improvements}/{len(noise_levels)}")
    
    # Seuil de robustesse
    threshold_orig = None
    threshold_aug = None
    
    for noise_level in noise_levels:
        r2_orig = results['without_augmentation'][noise_level]['metrics']['r2']
        r2_aug = results['with_augmentation'][noise_level]['metrics']['r2']
        
        if r2_orig >= 0.8 and threshold_orig is None:
            threshold_orig = noise_level
        if r2_aug >= 0.8 and threshold_aug is None:
            threshold_aug = noise_level
    
    print(f"\n🎯 SEUILS DE ROBUSTESSE (R² > 0.8):")
    print(f"  Sans augmentation: jusqu'à {threshold_orig}% de bruit")
    print(f"  Avec augmentation: jusqu'à {threshold_aug}% de bruit")
    
    if threshold_aug and threshold_orig:
        improvement = threshold_aug - threshold_orig
        print(f"  Amélioration du seuil: +{improvement}% de bruit toléré")
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    
    if avg_r2_improvement > 0.02:
        print(f"  ✅ Augmentation très bénéfique - Recommandée")
        print(f"  ✅ Amélioration substantielle de la robustesse")
    elif avg_r2_improvement > 0.01:
        print(f"  ✅ Augmentation modérément bénéfique")
        print(f"  ⚠️  Évaluer le coût computationnel vs bénéfice")
    else:
        print(f"  ❌ Augmentation peu bénéfique")
        print(f"  ❌ Chercher d'autres stratégies d'amélioration")
    
    # Coût computationnel
    time_overhead = []
    for noise_level in noise_levels:
        time_orig = results['without_augmentation'][noise_level]['training_time']
        time_aug = results['with_augmentation'][noise_level]['training_time']
        overhead = (time_aug - time_orig) / time_orig * 100
        time_overhead.append(overhead)
    
    avg_overhead = np.mean(time_overhead)
    print(f"\n⏱️  COÛT COMPUTATIONNEL:")
    print(f"  Surcoût moyen d'entraînement: +{avg_overhead:.1f}%")
    
    if avg_overhead < 50:
        print(f"  ✅ Surcoût acceptable")
    elif avg_overhead < 100:
        print(f"  ⚠️  Surcoût modéré")
    else:
        print(f"  ❌ Surcoût élevé - Optimisation nécessaire")

if __name__ == "__main__":
    print("=== TEST DE L'EFFET DE L'AUGMENTATION SUR LA ROBUSTESSE ===")
    
    # Créer les dossiers de sortie
    os.makedirs("../plots", exist_ok=True)
    os.makedirs("../results", exist_ok=True)
    
    # Niveaux de bruit à tester (réduits pour ce test)
    noise_levels = [0, 2, 5, 10]
    
    try:
        # Charger les données
        X, y = load_dataset()
        
        # Configuration d'augmentation
        augmentation_config = {
            'interpolation_factor': 2,
            'smooth_variations': True,
            'variation_factor': 0.015,
            'n_variations': 1,
            'elastic_deformation': False
        }
        
        # Comparaison avec/sans augmentation
        results, X_test, y_test = compare_with_without_augmentation(
            X, y, noise_levels, augmentation_config
        )
        
        # Sauvegarder les résultats
        with open('../results/augmentation_comparison.json', 'w') as f:
            # Convertir les arrays numpy en listes pour JSON
            results_json = {}
            for key, value in results.items():
                if key != 'comparison':
                    results_json[key] = {}
                    for noise, data in value.items():
                        results_json[key][noise] = {
                            'r2': float(data['metrics']['r2']),
                            'rmse': float(data['metrics']['rmse']),
                            'mae': float(data['metrics']['mae']),
                            'training_time': float(data['training_time']),
                            'epochs': int(data['epochs'])
                        }
                else:
                    results_json[key] = {}
                    for noise, data in value.items():
                        results_json[key][noise] = {
                            'r2_improvement': float(data['r2_improvement']),
                            'rmse_improvement': float(data['rmse_improvement']),
                            'r2_improvement_percent': float(data['r2_improvement_percent']),
                            'rmse_improvement_percent': float(data['rmse_improvement_percent'])
                        }
            
            json.dump(results_json, f, indent=2)
        
        # Visualisations
        plot_augmentation_comparison(results, noise_levels)
        
        # Analyse
        analyze_augmentation_benefits(results, noise_levels)
        
        print(f"\n=== TEST D'AUGMENTATION TERMINÉ ===")
        print(f"📊 Résultats sauvegardés: ../results/augmentation_comparison.json")
        print(f"📈 Graphiques générés: ../plots/augmentation_comparison.png")
        
    except Exception as e:
        print(f"Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()
