#!/usr/bin/env python3
"""
Comparaison optimale avec translation de -0.4µm (meilleur alignement)
Auteur: Oussama GUELFAA
Date: 04/07/2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d

def load_experimental_data(file_path):
    """Charge les données expérimentales interpolées"""
    print(f"📊 Chargement des données expérimentales: {file_path}")
    df = pd.read_csv(file_path)
    r_exp = df['r_experiment'].values
    I_exp = df['I_experiment'].values
    print(f"   ✅ {len(r_exp)} points chargés")
    return r_exp, I_exp

def load_simulated_data(file_path, max_points=600):
    """Charge les données simulées"""
    print(f"🔬 Chargement des données simulées: {file_path}")
    data = loadmat(file_path)

    # Afficher les clés disponibles pour debug
    print(f"   🔍 Clés disponibles: {list(data.keys())}")

    x_sim = data['x'].flatten()
    ratio_sim = data['ratio'].flatten()

    # Extraire gap et L_ecran du nom du fichier si pas dans le .mat
    import os
    filename = os.path.basename(file_path)
    # Format: gap_0.1499um_L_4.751um.mat
    parts = filename.replace('.mat', '').split('_')
    gap = float(parts[1].replace('um', ''))
    L_ecran = float(parts[3].replace('um', ''))

    # Tronquer à max_points
    if len(x_sim) > max_points:
        x_sim = x_sim[:max_points]
        ratio_sim = ratio_sim[:max_points]
        print(f"   ✅ {len(data['x'].flatten())} points originaux → {max_points} points tronqués")

    print(f"   🎯 Paramètres extraits: gap={gap:.4f}µm, L_écran={L_ecran:.3f}µm")

    return x_sim, ratio_sim, gap, L_ecran

def create_optimal_comparison_plot(r_exp, I_exp, x_sim, ratio_sim, gap, L_ecran, translation_offset=0.4):
    """Crée le graphique de comparaison avec translation optimale"""
    
    print(f"🎨 Création du graphique avec translation optimale: -{translation_offset:.1f}µm")
    
    # Appliquer la translation optimale
    r_exp_translated = r_exp - translation_offset
    valid_indices = r_exp_translated >= 0
    r_exp_translated = r_exp_translated[valid_indices]
    I_exp_translated = I_exp[valid_indices]
    
    print(f"   📊 Points conservés: {len(I_exp_translated)}/{len(I_exp)} ({100*len(I_exp_translated)/len(I_exp):.1f}%)")
    
    # Calculer les métriques d'alignement
    interp_sim = interp1d(x_sim, ratio_sim, kind='linear', 
                         bounds_error=False, fill_value='extrapolate')
    
    common_r = r_exp_translated
    mask = (common_r >= x_sim.min()) & (common_r <= x_sim.max())
    common_r = common_r[mask]
    I_exp_common = I_exp_translated[mask]
    I_sim_common = interp_sim(common_r)
    
    correlation = np.corrcoef(I_exp_common, I_sim_common)[0, 1]
    mse = np.mean((I_exp_common - I_sim_common)**2)
    
    print(f"   🎯 Corrélation: {correlation:.3f}")
    print(f"   📏 MSE: {mse:.4f}")
    
    # Configuration de la figure
    plt.figure(figsize=(16, 10))
    
    # Tracer les courbes
    plt.plot(r_exp_translated, I_exp_translated, 
             color='red', 
             linewidth=2.5, 
             alpha=0.9,
             label=f'Anneau Expérimental (Translaté -{translation_offset:.1f}µm)',
             marker='o',
             markersize=2,
             markevery=15)
    
    plt.plot(x_sim, ratio_sim, 
             color='blue', 
             linewidth=2.5, 
             alpha=0.9,
             label=f'Anneau Simulé (gap={gap:.4f}µm, L_écran={L_ecran:.3f}µm)',
             linestyle='--',
             marker='s',
             markersize=1.5,
             markevery=20)
    
    # Configuration des axes
    plt.xlim(0, 4.2)
    plt.ylim(0, 1.8)
    plt.xlabel('Position Radiale r (µm)', fontsize=14, fontweight='bold')
    plt.ylabel('Intensité Normalisée', fontsize=14, fontweight='bold')
    plt.title('Comparaison Optimale: Anneau Expérimental vs Simulé\n'
              f'Translation Optimale: -{translation_offset:.1f}µm (Corrélation: {correlation:.3f})', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Légende
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    
    # Grille
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Statistiques détaillées
    stats_text = f"""MÉTRIQUES D'ALIGNEMENT:

Translation Appliquée: -{translation_offset:.1f} µm
Points Conservés: {len(I_exp_translated)}/{len(I_exp)} ({100*len(I_exp_translated)/len(I_exp):.1f}%)

DONNÉES EXPÉRIMENTALES (Translatées):
• Range r: {r_exp_translated[0]:.3f} - {r_exp_translated[-1]:.3f} µm
• Range I: {I_exp_translated.min():.3f} - {I_exp_translated.max():.3f}
• Moyenne: {I_exp_translated.mean():.3f}

DONNÉES SIMULÉES:
• Range x: {x_sim[0]:.3f} - {x_sim[-1]:.3f} µm
• Range ratio: {ratio_sim.min():.3f} - {ratio_sim.max():.3f}
• Moyenne: {ratio_sim.mean():.3f}

QUALITÉ D'ALIGNEMENT:
• Corrélation: {correlation:.3f}
• MSE: {mse:.4f}
• Points communs: {len(common_r)}"""
    
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, 
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', 
                      facecolor='lightblue', 
                      alpha=0.8,
                      edgecolor='navy'))
    
    # Sauvegarder
    plt.tight_layout()
    save_path = f"optimal_comparison_translation_{translation_offset:.1f}um.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Graphique sauvegardé: {save_path}")
    plt.close()  # Fermer la figure pour libérer la mémoire

    return correlation, mse

def main():
    print("=" * 80)
    print("COMPARAISON OPTIMALE AVEC TRANSLATION")
    print("=" * 80)
    
    # Charger les données
    exp_file = "data_generation/Experimental_data_analysis/interpolated_profiles_600pts/profile_001_interpolated.csv"
    sim_file = "data_generation/Calcul_Data/dataset/gap_0.1499um_L_4.751um.mat"
    
    r_exp, I_exp = load_experimental_data(exp_file)
    x_sim, ratio_sim, gap, L_ecran = load_simulated_data(sim_file)
    
    # Créer la comparaison optimale
    correlation, mse = create_optimal_comparison_plot(r_exp, I_exp, x_sim, ratio_sim, gap, L_ecran)
    
    print("\n🎉 COMPARAISON OPTIMALE TERMINÉE!")
    print(f"🏆 Corrélation finale: {correlation:.3f}")
    print(f"📏 MSE finale: {mse:.4f}")

if __name__ == "__main__":
    main()
