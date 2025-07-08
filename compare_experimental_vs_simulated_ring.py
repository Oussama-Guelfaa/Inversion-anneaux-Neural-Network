#!/usr/bin/env python3
"""
Script de comparaison entre anneau expérimental et simulé
Auteur: Oussama GUELFAA
Date: 03/07/2025

Ce script trace deux anneaux avec des couleurs différentes :
- Anneau expérimental interpolé (600 points)
- Anneau simulé tronqué (600 premiers points)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path

def load_experimental_data(csv_path):
    """
    Charge les données expérimentales interpolées depuis le CSV
    
    Args:
        csv_path (str): Chemin vers le fichier CSV
        
    Returns:
        tuple: (r_experiment, I_experiment)
    """
    print(f"📊 Chargement des données expérimentales: {csv_path}")
    
    # Charger le CSV
    df = pd.read_csv(csv_path)
    
    # Extraire les colonnes
    r_experiment = df['r_experiment'].values
    I_experiment = df['I_experiment'].values
    
    print(f"   ✅ {len(I_experiment)} points chargés")
    print(f"   📏 Range r: {r_experiment[0]:.3f} - {r_experiment[-1]:.3f} µm")
    print(f"   📈 Range I: {I_experiment.min():.3f} - {I_experiment.max():.3f}")
    
    return r_experiment, I_experiment

def load_simulated_data(mat_path, truncate_to=600):
    """
    Charge les données simulées depuis le fichier .mat et tronque
    
    Args:
        mat_path (str): Chemin vers le fichier .mat
        truncate_to (int): Nombre de points à conserver
        
    Returns:
        tuple: (x_simulated, ratio_simulated, gap, L_ecran)
    """
    print(f"🔬 Chargement des données simulées: {mat_path}")
    
    # Charger le fichier .mat
    data = sio.loadmat(mat_path)
    
    # Extraire les données
    ratio = data['ratio'].flatten()
    x = data['x'].flatten()
    gap = float(data['gap'][0, 0])
    L_ecran = float(data['L_ecran_subs'][0, 0])
    
    # Tronquer aux 600 premiers points
    ratio_truncated = ratio[:truncate_to]
    x_truncated = x[:truncate_to]
    
    print(f"   ✅ {len(ratio)} points originaux → {len(ratio_truncated)} points tronqués")
    print(f"   📏 Range x: {x_truncated[0]:.3f} - {x_truncated[-1]:.3f} µm")
    print(f"   📈 Range ratio: {ratio_truncated.min():.3f} - {ratio_truncated.max():.3f}")
    print(f"   🎯 Paramètres: gap={gap:.4f}µm, L_écran={L_ecran:.3f}µm")
    
    return x_truncated, ratio_truncated, gap, L_ecran

def create_comparison_plot(r_exp, I_exp, x_sim, ratio_sim, gap, L_ecran, save_path=None, translation_offset=0.3):
    """
    Crée le graphique de comparaison entre les deux anneaux

    Args:
        r_exp (array): Positions radiales expérimentales
        I_exp (array): Intensités expérimentales
        x_sim (array): Positions radiales simulées
        ratio_sim (array): Ratios simulés
        gap (float): Valeur du gap
        L_ecran (float): Valeur de L_écran
        save_path (str): Chemin de sauvegarde (optionnel)
        translation_offset (float): Décalage vers la gauche pour les données expérimentales (µm)
    """
    print("🎨 Création du graphique de comparaison...")

    # Appliquer la translation vers la gauche aux données expérimentales
    r_exp_translated = r_exp - translation_offset

    # Filtrer les valeurs négatives après translation
    valid_indices = r_exp_translated >= 0
    r_exp_translated = r_exp_translated[valid_indices]
    I_exp_translated = I_exp[valid_indices]

    print(f"   🔄 Translation appliquée: -{translation_offset:.3f} µm")
    print(f"   📊 Points conservés après translation: {len(I_exp_translated)}/{len(I_exp)} ({100*len(I_exp_translated)/len(I_exp):.1f}%)")

    # Configuration de la figure
    plt.figure(figsize=(14, 8))
    
    # Tracer l'anneau expérimental en rouge (avec translation)
    plt.plot(r_exp_translated, I_exp_translated,
             color='red',
             linewidth=2.0,
             alpha=0.8,
             label=f'Anneau Expérimental (Profile 001 - Translaté -{translation_offset:.1f}µm)',
             marker='o',
             markersize=1.5,
             markevery=20)  # Marqueurs tous les 20 points
    
    # Tracer l'anneau simulé en bleu
    plt.plot(x_sim, ratio_sim, 
             color='blue', 
             linewidth=2.0, 
             alpha=0.8,
             label=f'Anneau Simulé (gap={gap:.4f}µm, L_écran={L_ecran:.3f}µm - 600pts)',
             marker='s',
             markersize=1.5,
             markevery=20)  # Marqueurs tous les 20 points
    
    # Configuration des axes et labels
    plt.xlabel('Position Radiale r (µm)', fontsize=14, fontweight='bold')
    plt.ylabel('Intensité Normalisée', fontsize=14, fontweight='bold')
    plt.title('Comparaison Anneau Expérimental vs Simulé\n' + 
              'Profil Expérimental Interpolé (600pts) vs Simulation Tronquée (600pts)',
              fontsize=16, fontweight='bold', pad=20)
    
    # Grille et légende
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='upper right', framealpha=0.9)
    
    # Améliorer l'apparence
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Ajuster les limites pour une meilleure visualisation
    plt.xlim(0, max(r_exp_translated[-1] if len(r_exp_translated) > 0 else 0, x_sim[-1]))

    # Ajouter des statistiques dans un encadré
    stats_text = f"""Statistiques de Comparaison:

Expérimental (Translaté -{translation_offset:.1f}µm):
• Points: {len(I_exp_translated)} (sur {len(I_exp)} originaux)
• Range r: {r_exp_translated[0]:.3f} - {r_exp_translated[-1]:.3f} µm
• Range I: {I_exp_translated.min():.3f} - {I_exp_translated.max():.3f}
• Moyenne: {I_exp_translated.mean():.3f}

Simulé:
• Points: {len(ratio_sim)}
• Range x: {x_sim[0]:.3f} - {x_sim[-1]:.3f} µm
• Range ratio: {ratio_sim.min():.3f} - {ratio_sim.max():.3f}
• Moyenne: {ratio_sim.mean():.3f}"""
    
    plt.text(0.02, 0.98, stats_text, 
             transform=ax.transAxes, 
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()

    # Sauvegarder si demandé
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Graphique sauvegardé: {save_path}")

    # Fermer la figure pour éviter l'affichage
    plt.close()

    print("✅ Graphique créé avec succès!")

def main():
    """Fonction principale"""
    print("="*70)
    print("COMPARAISON ANNEAU EXPÉRIMENTAL vs SIMULÉ")
    print("="*70)
    
    # Chemins des fichiers
    experimental_csv = "data_generation/Experimental_data_analysis/interpolated_profiles_600pts/profile_001_interpolated.csv"
    simulated_mat = "data_generation/Calcul_Data/dataset/gap_0.2088um_L_3.937um.mat"
    
    # Vérifier l'existence des fichiers
    if not Path(experimental_csv).exists():
        print(f"❌ Fichier expérimental non trouvé: {experimental_csv}")
        return
    
    if not Path(simulated_mat).exists():
        print(f"❌ Fichier simulé non trouvé: {simulated_mat}")
        return
    
    try:
        # Charger les données expérimentales
        r_exp, I_exp = load_experimental_data(experimental_csv)
        
        # Charger les données simulées
        x_sim, ratio_sim, gap, L_ecran = load_simulated_data(simulated_mat, truncate_to=600)
        
        # Créer plusieurs graphiques avec différentes translations
        translations = [0.3, 0.5, 0.7]

        for offset in translations:
            save_path = f"comparison_experimental_vs_simulated_rings_translated_{offset:.1f}um.png"
            print(f"\n🔄 Création avec translation de {offset:.1f} µm...")
            create_comparison_plot(r_exp, I_exp, x_sim, ratio_sim, gap, L_ecran, save_path, offset)
        
        print(f"\n🎉 COMPARAISON TERMINÉE AVEC SUCCÈS!")
        print(f"📁 Fichier généré: {save_path}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
