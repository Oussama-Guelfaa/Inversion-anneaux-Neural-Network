#!/usr/bin/env python3
"""
Script interactif pour tester différentes translations de l'anneau expérimental
Auteur: Oussama GUELFAA
Date: 04/07/2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import argparse

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
    x_sim = data['x'].flatten()
    ratio_sim = data['ratio'].flatten()
    
    # Tronquer à max_points
    if len(x_sim) > max_points:
        x_sim = x_sim[:max_points]
        ratio_sim = ratio_sim[:max_points]
        print(f"   ✅ {len(data['x'].flatten())} points originaux → {max_points} points tronqués")
    
    return x_sim, ratio_sim

def create_interactive_comparison(r_exp, I_exp, x_sim, ratio_sim, translation_offset):
    """Crée une comparaison avec translation donnée"""
    
    # Appliquer la translation
    r_exp_translated = r_exp - translation_offset
    valid_indices = r_exp_translated >= 0
    r_exp_translated = r_exp_translated[valid_indices]
    I_exp_translated = I_exp[valid_indices]
    
    # Calculer les statistiques d'alignement
    # Interpoler les données simulées sur les positions expérimentales translatées
    from scipy.interpolate import interp1d
    
    # Créer l'interpolateur pour les données simulées
    interp_sim = interp1d(x_sim, ratio_sim, kind='linear', 
                         bounds_error=False, fill_value='extrapolate')
    
    # Interpoler aux positions expérimentales translatées
    common_r = r_exp_translated
    mask = (common_r >= x_sim.min()) & (common_r <= x_sim.max())
    common_r = common_r[mask]
    I_exp_common = I_exp_translated[mask]
    I_sim_common = interp_sim(common_r)
    
    # Calculer la corrélation
    correlation = np.corrcoef(I_exp_common, I_sim_common)[0, 1]
    
    # Calculer l'erreur quadratique moyenne
    mse = np.mean((I_exp_common - I_sim_common)**2)
    
    return {
        'r_translated': r_exp_translated,
        'I_translated': I_exp_translated,
        'correlation': correlation,
        'mse': mse,
        'points_kept': len(I_exp_translated),
        'points_original': len(I_exp)
    }

def plot_comparison_grid(r_exp, I_exp, x_sim, ratio_sim, translations):
    """Crée une grille de comparaisons pour différentes translations"""
    
    n_translations = len(translations)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    results = []
    
    for i, offset in enumerate(translations):
        ax = axes[i]
        
        # Calculer la translation
        result = create_interactive_comparison(r_exp, I_exp, x_sim, ratio_sim, offset)
        results.append((offset, result))
        
        # Tracer
        ax.plot(result['r_translated'], result['I_translated'], 
                'r-', linewidth=2, alpha=0.8, 
                label=f'Exp. (Translaté -{offset:.1f}µm)')
        ax.plot(x_sim, ratio_sim, 
                'b-', linewidth=2, alpha=0.8, 
                label='Simulé')
        
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1.8)
        ax.set_xlabel('Position Radiale r (µm)')
        ax.set_ylabel('Intensité Normalisée')
        ax.set_title(f'Translation: -{offset:.1f}µm\n'
                    f'Corrélation: {result["correlation"]:.3f}\n'
                    f'MSE: {result["mse"]:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Supprimer les axes non utilisés
    for i in range(n_translations, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('translation_comparison_grid.png', dpi=300, bbox_inches='tight')
    print("💾 Grille de comparaison sauvegardée: translation_comparison_grid.png")
    
    # Afficher le résumé des résultats
    print("\n📊 RÉSUMÉ DES RÉSULTATS:")
    print("=" * 60)
    print(f"{'Translation (µm)':<15} {'Corrélation':<12} {'MSE':<10} {'Points conservés':<15}")
    print("-" * 60)
    
    best_correlation = -1
    best_offset = 0
    
    for offset, result in results:
        correlation = result['correlation']
        if correlation > best_correlation:
            best_correlation = correlation
            best_offset = offset
            
        print(f"{offset:<15.1f} {correlation:<12.3f} {result['mse']:<10.4f} "
              f"{result['points_kept']}/{result['points_original']} ({100*result['points_kept']/result['points_original']:.1f}%)")
    
    print("-" * 60)
    print(f"🏆 MEILLEURE TRANSLATION: -{best_offset:.1f}µm (Corrélation: {best_correlation:.3f})")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test interactif de translations')
    parser.add_argument('--translations', nargs='+', type=float, 
                       default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                       help='Liste des translations à tester (en µm)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TEST INTERACTIF DE TRANSLATIONS")
    print("=" * 70)
    
    # Charger les données
    exp_file = "data_generation/Experimental_data_analysis/interpolated_profiles_600pts/profile_001_interpolated.csv"
    sim_file = "data_generation/Calcul_Data/dataset/gap_0.1499um_L_4.751um.mat"
    
    r_exp, I_exp = load_experimental_data(exp_file)
    x_sim, ratio_sim = load_simulated_data(sim_file)
    
    # Tester les translations
    results = plot_comparison_grid(r_exp, I_exp, x_sim, ratio_sim, args.translations)
    
    print("\n🎉 ANALYSE TERMINÉE!")
    print("📁 Fichier généré: translation_comparison_grid.png")

if __name__ == "__main__":
    main()
