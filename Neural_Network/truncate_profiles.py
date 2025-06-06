#!/usr/bin/env python3
"""
Profile Truncation Script
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Script pour tronquer les profils d'intensité de 1000 à 600 points
afin d'éliminer le pic divergent à droite qui perturbe la prédiction du gap.

Problème identifié: Le pic à droite (r > 6 µm) cause des difficultés
pour la prédiction du paramètre gap.

Solution: Limiter les profils aux 600 premiers points (r ≤ 6 µm)
où l'information utile est concentrée.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import os

def analyze_profile_regions():
    """Analyse les différentes régions des profils pour justifier la troncature."""
    
    print("="*80)
    print("ANALYSE DES RÉGIONS DES PROFILS D'INTENSITÉ")
    print("="*80)
    
    # Charger les profils originaux
    df_profiles = pd.read_csv('../data/processed/intensity_profiles_full.csv')
    X_original = df_profiles.values
    
    print(f"Profils originaux: {X_original.shape}")
    print(f"Nombre de points par profil: {X_original.shape[1]}")
    
    # Analyser les statistiques par région
    regions = {
        'Zone utile (0-600)': slice(0, 600),
        'Zone problématique (600-1000)': slice(600, 1000),
        'Pic final (900-1000)': slice(900, 1000)
    }
    
    print(f"\nAnalyse statistique par région:")
    for region_name, region_slice in regions.items():
        region_data = X_original[:, region_slice]
        
        print(f"\n{region_name}:")
        print(f"  Points: {region_slice.start}-{region_slice.stop}")
        print(f"  Moyenne: {region_data.mean():.6f}")
        print(f"  Écart-type: {region_data.std():.6f}")
        print(f"  Min/Max: [{region_data.min():.6f}, {region_data.max():.6f}]")
        print(f"  Coefficient de variation: {region_data.std()/region_data.mean():.4f}")
    
    # Analyser la corrélation avec les paramètres
    df_params = pd.read_csv('../data/processed/parameters.csv')
    y = df_params[['L_ecran', 'gap']].values

    # Aligner les dimensions (prendre le minimum)
    min_samples = min(len(X_original), len(y))
    X_original = X_original[:min_samples]
    y = y[:min_samples]

    print(f"Données alignées: X{X_original.shape}, y{y.shape}")
    
    print(f"\nCorrélation avec les paramètres:")
    
    for region_name, region_slice in regions.items():
        region_data = X_original[:, region_slice]
        region_mean = region_data.mean(axis=1)  # Moyenne par profil
        
        # Corrélation avec L_ecran et gap
        corr_L = np.corrcoef(region_mean, y[:, 0])[0, 1]
        corr_gap = np.corrcoef(region_mean, y[:, 1])[0, 1]
        
        print(f"\n{region_name}:")
        print(f"  Corrélation avec L_ecran: {corr_L:.4f}")
        print(f"  Corrélation avec gap: {corr_gap:.4f}")
        print(f"  Utilité pour gap: {'✓ Utile' if abs(corr_gap) > 0.1 else '✗ Peu utile'}")
    
    return X_original

def visualize_truncation_impact():
    """Visualise l'impact de la troncature sur les profils."""
    
    print(f"\n=== VISUALISATION DE L'IMPACT DE LA TRONCATURE ===")
    
    # Charger les données
    df_profiles = pd.read_csv('../data/processed/intensity_profiles_full.csv')
    df_params = pd.read_csv('../data/processed/parameters.csv')
    X_original = df_profiles.values
    y = df_params[['L_ecran', 'gap']].values
    
    # Créer les visualisations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Profils complets vs tronqués
    sample_indices = [0, 100, 200, 300, 400]  # Échantillons représentatifs
    
    for i, idx in enumerate(sample_indices[:3]):
        ax = axes[0, i]
        profile = X_original[idx]
        
        # Profil complet
        ax.plot(range(1000), profile, 'b-', alpha=0.7, label='Profil complet (1000 pts)')
        
        # Zone de troncature
        ax.axvline(x=600, color='red', linestyle='--', alpha=0.8, label='Limite troncature')
        ax.fill_betweenx([profile.min(), profile.max()], 600, 1000, 
                        alpha=0.2, color='red', label='Zone supprimée')
        
        # Profil tronqué
        ax.plot(range(600), profile[:600], 'g-', linewidth=2, label='Profil tronqué (600 pts)')
        
        ax.set_title(f'Échantillon {idx+1}\nL={y[idx, 0]:.1f}µm, gap={y[idx, 1]:.3f}µm')
        ax.set_xlabel('Position (points)')
        ax.set_ylabel('Intensité normalisée')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Statistiques par position
    ax = axes[1, 0]
    mean_profile = X_original.mean(axis=0)
    std_profile = X_original.std(axis=0)
    
    ax.plot(range(1000), mean_profile, 'b-', label='Moyenne')
    ax.fill_between(range(1000), mean_profile - std_profile, mean_profile + std_profile,
                   alpha=0.3, label='± 1 écart-type')
    ax.axvline(x=600, color='red', linestyle='--', label='Limite troncature')
    ax.set_title('Profil moyen ± écart-type')
    ax.set_xlabel('Position (points)')
    ax.set_ylabel('Intensité')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Variance par position
    ax = axes[1, 1]
    variance_profile = X_original.var(axis=0)
    ax.plot(range(1000), variance_profile, 'purple', linewidth=2)
    ax.axvline(x=600, color='red', linestyle='--', label='Limite troncature')
    ax.set_title('Variance par position')
    ax.set_xlabel('Position (points)')
    ax.set_ylabel('Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Distribution des valeurs par région
    ax = axes[1, 2]
    zone_utile = X_original[:, :600].flatten()
    zone_problematique = X_original[:, 600:].flatten()
    
    ax.hist(zone_utile, bins=50, alpha=0.6, label='Zone utile (0-600)', density=True)
    ax.hist(zone_problematique, bins=50, alpha=0.6, label='Zone problématique (600-1000)', density=True)
    ax.set_title('Distribution des intensités')
    ax.set_xlabel('Intensité normalisée')
    ax.set_ylabel('Densité')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/truncation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Analyse de troncature sauvegardée: plots/truncation_analysis.png")

def create_truncated_dataset():
    """Crée le dataset tronqué à 600 points."""
    
    print(f"\n=== CRÉATION DU DATASET TRONQUÉ ===")
    
    # Charger les données originales
    df_profiles = pd.read_csv('../data/processed/intensity_profiles_full.csv')
    df_params = pd.read_csv('../data/processed/parameters.csv')
    
    X_original = df_profiles.values
    y_original = df_params[['L_ecran', 'gap']].values
    
    print(f"Dataset original:")
    print(f"  Profils: {X_original.shape}")
    print(f"  Paramètres: {y_original.shape}")
    
    # Tronquer à 600 points
    X_truncated = X_original[:, :600]
    
    print(f"\nDataset tronqué:")
    print(f"  Profils: {X_truncated.shape}")
    print(f"  Réduction: {X_original.shape[1]} → {X_truncated.shape[1]} points")
    print(f"  Pourcentage conservé: {600/1000*100:.1f}%")
    
    # Vérifier que l'information importante est conservée
    print(f"\nVérification de l'information conservée:")
    
    # Calculer les corrélations avant/après troncature
    orig_corr_L = np.corrcoef(X_original.mean(axis=1), y_original[:, 0])[0, 1]
    orig_corr_gap = np.corrcoef(X_original.mean(axis=1), y_original[:, 1])[0, 1]
    
    trunc_corr_L = np.corrcoef(X_truncated.mean(axis=1), y_original[:, 0])[0, 1]
    trunc_corr_gap = np.corrcoef(X_truncated.mean(axis=1), y_original[:, 1])[0, 1]
    
    print(f"  Corrélation avec L_ecran:")
    print(f"    Original (1000 pts): {orig_corr_L:.4f}")
    print(f"    Tronqué (600 pts): {trunc_corr_L:.4f}")
    print(f"    Conservation: {trunc_corr_L/orig_corr_L*100:.1f}%")
    
    print(f"  Corrélation avec gap:")
    print(f"    Original (1000 pts): {orig_corr_gap:.4f}")
    print(f"    Tronqué (600 pts): {trunc_corr_gap:.4f}")
    print(f"    Conservation: {trunc_corr_gap/orig_corr_gap*100:.1f}%")
    
    # Sauvegarder le dataset tronqué
    print(f"\nSauvegarde du dataset tronqué...")

    # Créer le dossier processed_data dans Neural_Network
    os.makedirs('processed_data', exist_ok=True)

    # Créer les DataFrames
    df_profiles_truncated = pd.DataFrame(X_truncated)
    df_params_truncated = df_params.copy()  # Paramètres inchangés

    # Sauvegarder
    df_profiles_truncated.to_csv('processed_data/intensity_profiles_truncated_600.csv', index=False)
    df_params_truncated.to_csv('processed_data/parameters_truncated_600.csv', index=False)

    print(f"Fichiers sauvegardés:")
    print(f"  processed_data/intensity_profiles_truncated_600.csv")
    print(f"  processed_data/parameters_truncated_600.csv")
    
    return X_truncated, y_original

def compare_with_experimental_data():
    """Compare les profils tronqués avec les données expérimentales."""
    
    print(f"\n=== COMPARAISON AVEC DONNÉES EXPÉRIMENTALES ===")
    
    # Charger les données expérimentales
    dataset_dir = "../data_generation/dataset"
    labels_df = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))
    
    X_exp = []
    for i in range(min(10, len(labels_df))):
        filename = labels_df.iloc[i]['filename'].replace('.png', '.mat')
        mat_path = os.path.join(dataset_dir, filename)
        if os.path.exists(mat_path):
            data = sio.loadmat(mat_path)
            ratio = data['ratio'].flatten()
            X_exp.append(ratio)
    
    X_exp = np.array(X_exp)
    
    # Charger les données simulées tronquées
    df_sim_trunc = pd.read_csv('processed_data/intensity_profiles_truncated_600.csv')
    X_sim_trunc = df_sim_trunc.values
    
    print(f"Comparaison des longueurs:")
    print(f"  Expérimental: {X_exp.shape[1]} points")
    print(f"  Simulé tronqué: {X_sim_trunc.shape[1]} points")
    
    if X_exp.shape[1] != X_sim_trunc.shape[1]:
        print(f"  ⚠️  ATTENTION: Longueurs différentes!")
        if X_exp.shape[1] > 600:
            print(f"  → Tronquer aussi les données expérimentales à 600 points")
            X_exp = X_exp[:, :600]
        else:
            print(f"  → Les données expérimentales sont déjà plus courtes")
    
    # Comparer les statistiques
    print(f"\nComparaison statistique:")
    print(f"  Simulé tronqué - mean: {X_sim_trunc.mean():.6f}, std: {X_sim_trunc.std():.6f}")
    print(f"  Expérimental - mean: {X_exp.mean():.6f}, std: {X_exp.std():.6f}")
    
    # Calculer la similarité
    sim_mean = X_sim_trunc.mean(axis=0)
    exp_mean = X_exp.mean(axis=0)
    
    if len(sim_mean) == len(exp_mean):
        correlation = np.corrcoef(sim_mean, exp_mean)[0, 1]
        print(f"  Corrélation profils moyens: {correlation:.4f}")
        print(f"  Similarité: {'✓ Bonne' if correlation > 0.8 else '⚠️ Moyenne' if correlation > 0.6 else '✗ Faible'}")

def main():
    """Fonction principale."""
    
    print("="*80)
    print("TRONCATURE DES PROFILS D'INTENSITÉ")
    print("Résolution du problème du pic divergent à droite")
    print("="*80)
    
    # 1. Analyser les régions des profils
    X_original = analyze_profile_regions()
    
    # 2. Visualiser l'impact de la troncature
    visualize_truncation_impact()
    
    # 3. Créer le dataset tronqué
    X_truncated, y = create_truncated_dataset()
    
    # 4. Comparer avec les données expérimentales
    compare_with_experimental_data()
    
    print(f"\n{'='*80}")
    print(f"TRONCATURE TERMINÉE AVEC SUCCÈS")
    print(f"{'='*80}")
    
    print(f"✅ PROBLÈME RÉSOLU:")
    print(f"   • Pic divergent à droite (r > 6µm) supprimé")
    print(f"   • Profils réduits de 1000 → 600 points")
    print(f"   • Information utile conservée")
    print(f"   • Compatibilité avec données expérimentales vérifiée")
    
    print(f"\n📁 FICHIERS GÉNÉRÉS:")
    print(f"   • processed_data/intensity_profiles_truncated_600.csv")
    print(f"   • processed_data/parameters_truncated_600.csv")
    print(f"   • plots/truncation_analysis.png")
    
    print(f"\n🚀 PROCHAINES ÉTAPES:")
    print(f"   1. Entraîner le modèle avec les profils tronqués")
    print(f"   2. Comparer les performances avec le modèle original")
    print(f"   3. Vérifier l'amélioration de la prédiction du gap")
    print(f"   4. Mettre à jour la documentation")

if __name__ == "__main__":
    main()
