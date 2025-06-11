#!/usr/bin/env python3
"""
Analyse du décalage entre données d'entraînement et données réelles

Ce script analyse en détail pourquoi le modèle entraîné sur dataset_small_particle
ne fonctionne pas bien sur les données réelles du dataset principal.

Auteur: Oussama GUELFAA
Date: 10 - 01 - 2025
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_training_data_sample():
    """Charge un échantillon des données d'entraînement."""
    
    training_dir = "data_generation/dataset_small_particle"
    
    if not os.path.exists(training_dir):
        print(f"❌ Dossier d'entraînement non trouvé: {training_dir}")
        return None, None
    
    mat_files = [f for f in os.listdir(training_dir) if f.endswith('.mat')]
    
    X_train = []
    y_train = []
    
    print(f"Chargement de {len(mat_files)} échantillons d'entraînement...")
    
    for filename in mat_files:
        try:
            file_path = os.path.join(training_dir, filename)
            data = sio.loadmat(file_path)
            
            ratio = data['ratio'].flatten()
            gap = float(data['gap'][0, 0])
            
            X_train.append(ratio)
            y_train.append(gap)
            
        except Exception as e:
            print(f"Erreur avec {filename}: {e}")
    
    return np.array(X_train), np.array(y_train)

def load_real_data_sample():
    """Charge un échantillon des données réelles."""
    
    real_dir = "data_generation/dataset"
    
    if not os.path.exists(real_dir):
        print(f"❌ Dossier de données réelles non trouvé: {real_dir}")
        return None, None
    
    mat_files = [f for f in os.listdir(real_dir) if f.endswith('.mat')]
    
    X_real = []
    y_real = []
    
    print(f"Chargement de {len(mat_files)} échantillons réels...")
    
    for filename in mat_files:
        try:
            file_path = os.path.join(real_dir, filename)
            data = sio.loadmat(file_path)
            
            ratio = data['ratio'].flatten()
            gap = float(data['gap'][0, 0])
            
            X_real.append(ratio)
            y_real.append(gap)
            
        except Exception as e:
            print(f"Erreur avec {filename}: {e}")
    
    return np.array(X_real), np.array(y_real)

def compare_distributions(X_train, y_train, X_real, y_real):
    """Compare les distributions entre données d'entraînement et réelles."""
    
    print("\n" + "="*60)
    print("COMPARAISON DES DISTRIBUTIONS")
    print("="*60)
    
    # Statistiques des profils d'intensité
    print(f"\n📊 PROFILS D'INTENSITÉ:")
    print(f"{'Statistique':<15} {'Entraînement':<15} {'Réelles':<15} {'Différence':<15}")
    print("-" * 65)
    
    stats = ['Min', 'Max', 'Moyenne', 'Écart-type', 'Médiane']
    train_stats = [X_train.min(), X_train.max(), X_train.mean(), X_train.std(), np.median(X_train)]
    real_stats = [X_real.min(), X_real.max(), X_real.mean(), X_real.std(), np.median(X_real)]
    
    for stat, train_val, real_val in zip(stats, train_stats, real_stats):
        diff = real_val - train_val
        print(f"{stat:<15} {train_val:<15.4f} {real_val:<15.4f} {diff:<15.4f}")
    
    # Statistiques des gaps
    print(f"\n📊 VALEURS DE GAP:")
    print(f"{'Statistique':<15} {'Entraînement':<15} {'Réelles':<15} {'Différence':<15}")
    print("-" * 65)
    
    gap_stats = ['Min', 'Max', 'Moyenne', 'Écart-type', 'Médiane']
    train_gap_stats = [y_train.min(), y_train.max(), y_train.mean(), y_train.std(), np.median(y_train)]
    real_gap_stats = [y_real.min(), y_real.max(), y_real.mean(), y_real.std(), np.median(y_real)]
    
    for stat, train_val, real_val in zip(gap_stats, train_gap_stats, real_gap_stats):
        diff = real_val - train_val
        print(f"{stat:<15} {train_val:<15.4f} {real_val:<15.4f} {diff:<15.4f}")

def visualize_distributions(X_train, y_train, X_real, y_real):
    """Visualise les différences de distribution."""
    
    print(f"\n=== GÉNÉRATION DES VISUALISATIONS ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution des intensités moyennes
    axes[0, 0].hist(X_train.mean(axis=1), bins=30, alpha=0.7, label='Entraînement', color='blue')
    axes[0, 0].hist(X_real.mean(axis=1), bins=30, alpha=0.7, label='Réelles', color='red')
    axes[0, 0].set_xlabel('Intensité moyenne')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].set_title('Distribution des Intensités Moyennes')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution des intensités max
    axes[0, 1].hist(X_train.max(axis=1), bins=30, alpha=0.7, label='Entraînement', color='blue')
    axes[0, 1].hist(X_real.max(axis=1), bins=30, alpha=0.7, label='Réelles', color='red')
    axes[0, 1].set_xlabel('Intensité maximale')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].set_title('Distribution des Intensités Maximales')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution des gaps
    axes[0, 2].hist(y_train, bins=30, alpha=0.7, label='Entraînement', color='blue')
    axes[0, 2].hist(y_real, bins=30, alpha=0.7, label='Réelles', color='red')
    axes[0, 2].set_xlabel('Gap (µm)')
    axes[0, 2].set_ylabel('Fréquence')
    axes[0, 2].set_title('Distribution des Gaps')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Profils moyens
    axes[1, 0].plot(X_train.mean(axis=0), 'b-', linewidth=2, label='Entraînement')
    axes[1, 0].plot(X_real.mean(axis=0), 'r-', linewidth=2, label='Réelles')
    axes[1, 0].set_xlabel('Position radiale')
    axes[1, 0].set_ylabel('Intensité moyenne')
    axes[1, 0].set_title('Profils Moyens')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Écarts-types
    axes[1, 1].plot(X_train.std(axis=0), 'b-', linewidth=2, label='Entraînement')
    axes[1, 1].plot(X_real.std(axis=0), 'r-', linewidth=2, label='Réelles')
    axes[1, 1].set_xlabel('Position radiale')
    axes[1, 1].set_ylabel('Écart-type')
    axes[1, 1].set_title('Variabilité des Profils')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Corrélation gap vs intensité moyenne
    axes[1, 2].scatter(X_train.mean(axis=1), y_train, alpha=0.6, label='Entraînement', color='blue')
    axes[1, 2].scatter(X_real.mean(axis=1), y_real, alpha=0.6, label='Réelles', color='red')
    axes[1, 2].set_xlabel('Intensité moyenne')
    axes[1, 2].set_ylabel('Gap (µm)')
    axes[1, 2].set_title('Corrélation Gap vs Intensité')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Graphique sauvegardé: data_distribution_comparison.png")

def analyze_specific_examples(X_train, y_train, X_real, y_real):
    """Analyse des exemples spécifiques pour comprendre les différences."""
    
    print(f"\n=== ANALYSE D'EXEMPLES SPÉCIFIQUES ===")
    
    # Trouver des gaps similaires
    target_gap = 0.025  # Gap du fichier testé
    
    # Gaps d'entraînement proches
    train_close = np.abs(y_train - target_gap) < 0.01
    real_close = np.abs(y_real - target_gap) < 0.01
    
    print(f"Gap cible: {target_gap} µm")
    print(f"Échantillons d'entraînement proches: {train_close.sum()}")
    print(f"Échantillons réels proches: {real_close.sum()}")
    
    if train_close.sum() > 0 and real_close.sum() > 0:
        # Comparer les profils
        train_example = X_train[train_close][0]
        real_example = X_real[real_close][0]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_example, 'b-', linewidth=2, label=f'Entraînement (gap={y_train[train_close][0]:.4f})')
        plt.plot(real_example, 'r-', linewidth=2, label=f'Réel (gap={y_real[real_close][0]:.4f})')
        plt.xlabel('Position radiale')
        plt.ylabel('Intensité')
        plt.title('Comparaison Profils Similaires')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        diff = real_example - train_example
        plt.plot(diff, 'g-', linewidth=2)
        plt.xlabel('Position radiale')
        plt.ylabel('Différence d\'intensité')
        plt.title('Différence entre Profils')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('profile_comparison_similar_gaps.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparaison sauvegardée: profile_comparison_similar_gaps.png")
        
        # Statistiques de la différence
        print(f"\nStatistiques de la différence:")
        print(f"  Différence moyenne: {diff.mean():.6f}")
        print(f"  Différence RMS: {np.sqrt(np.mean(diff**2)):.6f}")
        print(f"  Différence max: {diff.max():.6f}")
        print(f"  Différence min: {diff.min():.6f}")

def generate_recommendations():
    """Génère des recommandations pour corriger le problème."""
    
    print(f"\n" + "="*60)
    print("RECOMMANDATIONS POUR CORRIGER LE PROBLÈME")
    print("="*60)
    
    print(f"\n🎯 PROBLÈME IDENTIFIÉ:")
    print(f"  • Décalage significatif entre données d'entraînement et réelles")
    print(f"  • Distributions d'intensité différentes")
    print(f"  • Normalisation inadéquate pour les données réelles")
    
    print(f"\n💡 SOLUTIONS RECOMMANDÉES:")
    
    print(f"\n1. 🔄 RÉENTRAÎNEMENT SUR DONNÉES RÉELLES:")
    print(f"   • Utiliser le dataset principal (data_generation/dataset/)")
    print(f"   • Même architecture mais entraînement sur données réelles")
    print(f"   • Validation croisée pour robustesse")
    
    print(f"\n2. 🔧 CALIBRATION DU MODÈLE EXISTANT:")
    print(f"   • Ajouter une couche de calibration post-prédiction")
    print(f"   • Correction linéaire basée sur échantillons réels")
    print(f"   • Fine-tuning sur quelques échantillons réels")
    
    print(f"\n3. 📊 NORMALISATION ADAPTÉE:")
    print(f"   • Recalculer le StandardScaler sur données réelles")
    print(f"   • Normalisation robuste (médiane/MAD)")
    print(f"   • Préprocessing spécifique aux données expérimentales")
    
    print(f"\n4. 🔍 ANALYSE APPROFONDIE:")
    print(f"   • Identifier la source des différences (acquisition, traitement)")
    print(f"   • Harmoniser les protocoles de génération de données")
    print(f"   • Validation sur plus d'échantillons réels")
    
    print(f"\n⚡ ACTION IMMÉDIATE RECOMMANDÉE:")
    print(f"   Réentraîner le modèle sur le dataset principal avec")
    print(f"   la même architecture validée lors des tests de robustesse.")

def main():
    """Fonction principale d'analyse."""
    
    print("="*60)
    print("ANALYSE DU DÉCALAGE DONNÉES ENTRAÎNEMENT vs RÉELLES")
    print("="*60)
    
    try:
        # 1. Charger les données d'entraînement
        print(f"\n1. Chargement des données d'entraînement...")
        X_train, y_train = load_training_data_sample()
        
        if X_train is None:
            print("❌ Impossible de charger les données d'entraînement")
            return
        
        print(f"   ✅ {len(X_train)} échantillons d'entraînement chargés")
        
        # 2. Charger les données réelles
        print(f"\n2. Chargement des données réelles...")
        X_real, y_real = load_real_data_sample()
        
        if X_real is None:
            print("❌ Impossible de charger les données réelles")
            return
        
        print(f"   ✅ {len(X_real)} échantillons réels chargés")
        
        # 3. Comparer les distributions
        compare_distributions(X_train, y_train, X_real, y_real)
        
        # 4. Visualiser les différences
        visualize_distributions(X_train, y_train, X_real, y_real)
        
        # 5. Analyser des exemples spécifiques
        analyze_specific_examples(X_train, y_train, X_real, y_real)
        
        # 6. Générer les recommandations
        generate_recommendations()
        
        print(f"\n{'='*60}")
        print("ANALYSE TERMINÉE")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Erreur durant l'analyse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
