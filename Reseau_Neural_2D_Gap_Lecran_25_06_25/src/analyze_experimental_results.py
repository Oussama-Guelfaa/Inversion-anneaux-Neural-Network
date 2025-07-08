#!/usr/bin/env python3
"""
Analyse des résultats du test sur profils expérimentaux
Auteur: Oussama GUELFAA
Date: Juillet 2025

Analyse les prédictions du réseau de neurones sur les profils expérimentaux
et compare avec les performances sur données simulées.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_experimental_results():
    """Charge les résultats du test expérimental."""
    results_file = "../results/test_experimental_profiles_20250701_154851.csv"
    
    if not Path(results_file).exists():
        logger.error(f"❌ Fichier non trouvé: {results_file}")
        return None
    
    df = pd.read_csv(results_file)
    
    # Filtrer les lignes de statistiques
    df_data = df[df['profile_number'] != 'STATS'].copy()
    df_data['profile_number'] = df_data['profile_number'].astype(int)
    
    logger.info(f"✅ Résultats expérimentaux chargés: {len(df_data)} profils")
    
    return df_data

def load_simulation_results():
    """Charge les résultats sur données simulées pour comparaison."""
    sim_files = [
        "../results/test_complet_modele_ameliore_20250626_125500.csv",
        "../results/resume_performances_modele_ameliore.csv"
    ]
    
    for sim_file in sim_files:
        if Path(sim_file).exists():
            df_sim = pd.read_csv(sim_file)
            logger.info(f"✅ Résultats simulation chargés: {sim_file}")
            return df_sim
    
    logger.warning("⚠️  Aucun fichier de résultats simulation trouvé")
    return None

def analyze_prediction_ranges(df_exp):
    """Analyse les ranges de prédictions."""
    logger.info("\n📊 ANALYSE DES RANGES DE PRÉDICTIONS")
    logger.info("="*50)
    
    # Statistiques Gap
    gap_stats = {
        'min': df_exp['Gap_predit_um'].min(),
        'max': df_exp['Gap_predit_um'].max(),
        'mean': df_exp['Gap_predit_um'].mean(),
        'std': df_exp['Gap_predit_um'].std(),
        'median': df_exp['Gap_predit_um'].median()
    }
    
    # Statistiques L_écran
    lecran_stats = {
        'min': df_exp['L_ecran_predit_um'].min(),
        'max': df_exp['L_ecran_predit_um'].max(),
        'mean': df_exp['L_ecran_predit_um'].mean(),
        'std': df_exp['L_ecran_predit_um'].std(),
        'median': df_exp['L_ecran_predit_um'].median()
    }
    
    logger.info(f"GAP PRÉDIT:")
    logger.info(f"  Range: {gap_stats['min']:.3f} à {gap_stats['max']:.3f} µm")
    logger.info(f"  Moyenne: {gap_stats['mean']:.3f} ± {gap_stats['std']:.3f} µm")
    logger.info(f"  Médiane: {gap_stats['median']:.3f} µm")
    
    logger.info(f"\nL_ÉCRAN PRÉDIT:")
    logger.info(f"  Range: {lecran_stats['min']:.1f} à {lecran_stats['max']:.1f} µm")
    logger.info(f"  Moyenne: {lecran_stats['mean']:.1f} ± {lecran_stats['std']:.1f} µm")
    logger.info(f"  Médiane: {lecran_stats['median']:.1f} µm")
    
    # Analyse des valeurs négatives
    gap_negative = (df_exp['Gap_predit_um'] < 0).sum()
    lecran_negative = (df_exp['L_ecran_predit_um'] < 0).sum()
    
    logger.info(f"\nVALEURS NÉGATIVES (non physiques):")
    logger.info(f"  Gap < 0: {gap_negative}/{len(df_exp)} profils ({gap_negative/len(df_exp)*100:.1f}%)")
    logger.info(f"  L_écran < 0: {lecran_negative}/{len(df_exp)} profils ({lecran_negative/len(df_exp)*100:.1f}%)")
    
    return gap_stats, lecran_stats

def analyze_intensity_correlation(df_exp):
    """Analyse la corrélation entre intensités et prédictions."""
    logger.info("\n🔍 CORRÉLATION INTENSITÉS-PRÉDICTIONS")
    logger.info("="*50)
    
    # Calculer les corrélations
    corr_gap_mean = np.corrcoef(df_exp['intensite_moyenne'], df_exp['Gap_predit_um'])[0, 1]
    corr_gap_std = np.corrcoef(df_exp['intensite_std'], df_exp['Gap_predit_um'])[0, 1]
    corr_lecran_mean = np.corrcoef(df_exp['intensite_moyenne'], df_exp['L_ecran_predit_um'])[0, 1]
    corr_lecran_std = np.corrcoef(df_exp['intensite_std'], df_exp['L_ecran_predit_um'])[0, 1]
    
    logger.info(f"CORRÉLATIONS:")
    logger.info(f"  Gap vs Intensité moyenne: {corr_gap_mean:.3f}")
    logger.info(f"  Gap vs Intensité std: {corr_gap_std:.3f}")
    logger.info(f"  L_écran vs Intensité moyenne: {corr_lecran_mean:.3f}")
    logger.info(f"  L_écran vs Intensité std: {corr_lecran_std:.3f}")
    
    return {
        'gap_mean': corr_gap_mean,
        'gap_std': corr_gap_std,
        'lecran_mean': corr_lecran_mean,
        'lecran_std': corr_lecran_std
    }

def compare_with_simulation_ranges(df_exp, df_sim=None):
    """Compare les ranges avec les données de simulation."""
    logger.info("\n⚖️  COMPARAISON SIMULATION vs EXPÉRIMENTAL")
    logger.info("="*50)
    
    if df_sim is None:
        logger.warning("⚠️  Pas de données de simulation pour comparaison")
        return
    
    # Ranges typiques des simulations (à ajuster selon vos données)
    sim_gap_range = (0.1, 3.0)  # µm
    sim_lecran_range = (50, 200)  # µm
    
    exp_gap_range = (df_exp['Gap_predit_um'].min(), df_exp['Gap_predit_um'].max())
    exp_lecran_range = (df_exp['L_ecran_predit_um'].min(), df_exp['L_ecran_predit_um'].max())
    
    logger.info(f"RANGES ATTENDUS (simulation):")
    logger.info(f"  Gap: {sim_gap_range[0]} à {sim_gap_range[1]} µm")
    logger.info(f"  L_écran: {sim_lecran_range[0]} à {sim_lecran_range[1]} µm")
    
    logger.info(f"\nRANGES OBTENUS (expérimental):")
    logger.info(f"  Gap: {exp_gap_range[0]:.3f} à {exp_gap_range[1]:.3f} µm")
    logger.info(f"  L_écran: {exp_lecran_range[0]:.1f} à {exp_lecran_range[1]:.1f} µm")
    
    # Analyse des écarts
    gap_in_range = ((df_exp['Gap_predit_um'] >= sim_gap_range[0]) & 
                    (df_exp['Gap_predit_um'] <= sim_gap_range[1])).sum()
    lecran_in_range = ((df_exp['L_ecran_predit_um'] >= sim_lecran_range[0]) & 
                       (df_exp['L_ecran_predit_um'] <= sim_lecran_range[1])).sum()
    
    logger.info(f"\nPRÉDICTIONS DANS RANGE PHYSIQUE:")
    logger.info(f"  Gap: {gap_in_range}/{len(df_exp)} ({gap_in_range/len(df_exp)*100:.1f}%)")
    logger.info(f"  L_écran: {lecran_in_range}/{len(df_exp)} ({lecran_in_range/len(df_exp)*100:.1f}%)")

def create_detailed_analysis_plots(df_exp):
    """Crée des graphiques d'analyse détaillée."""
    logger.info("\n📊 Génération des graphiques d'analyse...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Graphique 1: Distribution des gaps
    axes[0, 0].hist(df_exp['Gap_predit_um'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Gap = 0 (limite physique)')
    axes[0, 0].set_xlabel('Gap prédit (µm)', fontweight='bold')
    axes[0, 0].set_ylabel('Fréquence', fontweight='bold')
    axes[0, 0].set_title('Distribution des Gaps prédits\n(Profils expérimentaux)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Graphique 2: Distribution des L_écrans
    axes[0, 1].hist(df_exp['L_ecran_predit_um'], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='L_écran = 0 (limite physique)')
    axes[0, 1].set_xlabel('L_écran prédit (µm)', fontweight='bold')
    axes[0, 1].set_ylabel('Fréquence', fontweight='bold')
    axes[0, 1].set_title('Distribution des L_écrans prédits\n(Profils expérimentaux)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Graphique 3: Corrélation Gap vs intensité moyenne
    axes[0, 2].scatter(df_exp['intensite_moyenne'], df_exp['Gap_predit_um'], alpha=0.7, s=50)
    axes[0, 2].set_xlabel('Intensité moyenne', fontweight='bold')
    axes[0, 2].set_ylabel('Gap prédit (µm)', fontweight='bold')
    axes[0, 2].set_title('Gap vs Intensité moyenne', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Graphique 4: Corrélation L_écran vs intensité moyenne
    axes[1, 0].scatter(df_exp['intensite_moyenne'], df_exp['L_ecran_predit_um'], alpha=0.7, s=50, color='orange')
    axes[1, 0].set_xlabel('Intensité moyenne', fontweight='bold')
    axes[1, 0].set_ylabel('L_écran prédit (µm)', fontweight='bold')
    axes[1, 0].set_title('L_écran vs Intensité moyenne', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Graphique 5: Évolution par profil
    axes[1, 1].plot(df_exp['profile_number'], df_exp['Gap_predit_um'], 'bo-', label='Gap', alpha=0.7)
    axes[1, 1].set_xlabel('Numéro de profil', fontweight='bold')
    axes[1, 1].set_ylabel('Gap prédit (µm)', fontweight='bold')
    axes[1, 1].set_title('Évolution Gap par profil', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Graphique 6: Box plots
    gap_data = df_exp['Gap_predit_um']
    lecran_data = df_exp['L_ecran_predit_um']
    
    box_data = [gap_data, lecran_data]
    box_labels = ['Gap (µm)', 'L_écran (µm)']
    
    bp = axes[1, 2].boxplot(box_data, labels=box_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    axes[1, 2].set_ylabel('Valeurs prédites', fontweight='bold')
    axes[1, 2].set_title('Distribution des prédictions\n(Box plots)', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    plots_dir = Path("../plots")
    plots_dir.mkdir(exist_ok=True)
    
    analysis_path = plots_dir / "experimental_detailed_analysis.png"
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Analyse détaillée sauvegardée: {analysis_path}")

def generate_analysis_report(df_exp, gap_stats, lecran_stats, correlations):
    """Génère un rapport d'analyse détaillé."""
    logger.info("\n📋 Génération du rapport d'analyse...")
    
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("RAPPORT D'ANALYSE - PRÉDICTIONS SUR PROFILS EXPÉRIMENTAUX")
    report_lines.append("="*70)
    report_lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Auteur: Oussama GUELFAA")
    report_lines.append(f"Profils analysés: {len(df_exp)}")
    report_lines.append("")
    
    report_lines.append("RÉSUMÉ EXÉCUTIF:")
    report_lines.append("Le réseau de neurones produit des prédictions HORS RANGE PHYSIQUE")
    report_lines.append("pour les profils expérimentaux, indiquant un problème de généralisation")
    report_lines.append("simulation → expérience.")
    report_lines.append("")
    
    report_lines.append("STATISTIQUES DES PRÉDICTIONS:")
    report_lines.append(f"Gap prédit:")
    report_lines.append(f"  • Range: {gap_stats['min']:.3f} à {gap_stats['max']:.3f} µm")
    report_lines.append(f"  • Moyenne: {gap_stats['mean']:.3f} ± {gap_stats['std']:.3f} µm")
    report_lines.append(f"  • Valeurs négatives: {(df_exp['Gap_predit_um'] < 0).sum()}/{len(df_exp)} profils")
    report_lines.append("")
    
    report_lines.append(f"L_écran prédit:")
    report_lines.append(f"  • Range: {lecran_stats['min']:.1f} à {lecran_stats['max']:.1f} µm")
    report_lines.append(f"  • Moyenne: {lecran_stats['mean']:.1f} ± {lecran_stats['std']:.1f} µm")
    report_lines.append(f"  • Valeurs négatives: {(df_exp['L_ecran_predit_um'] < 0).sum()}/{len(df_exp)} profils")
    report_lines.append("")
    
    report_lines.append("PROBLÈMES IDENTIFIÉS:")
    report_lines.append("1. VALEURS NÉGATIVES: Gap et L_écran négatifs sont non physiques")
    report_lines.append("2. RANGE ABERRANT: Prédictions très éloignées des valeurs attendues")
    report_lines.append("3. GÉNÉRALISATION: Échec de la généralisation simulation → expérience")
    report_lines.append("")
    
    report_lines.append("HYPOTHÈSES SUR LES CAUSES:")
    report_lines.append("1. DIFFÉRENCES D'INTENSITÉ: Profils expérimentaux vs simulés")
    report_lines.append("2. NORMALISATION: Scalers inadaptés aux données expérimentales")
    report_lines.append("3. DOMAINE: Données expérimentales hors du domaine d'entraînement")
    report_lines.append("4. BRUIT: Caractéristiques expérimentales non modélisées")
    report_lines.append("")
    
    report_lines.append("RECOMMANDATIONS:")
    report_lines.append("1. ANALYSER les différences simulation vs expérience")
    report_lines.append("2. RÉENTRAÎNER avec données expérimentales étiquetées")
    report_lines.append("3. ADAPTER la normalisation aux données expérimentales")
    report_lines.append("4. DÉVELOPPER une stratégie de domain adaptation")
    
    # Sauvegarder le rapport
    report_path = Path("../results/analysis_experimental_predictions.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"✅ Rapport d'analyse sauvegardé: {report_path}")
    
    # Afficher aussi à l'écran
    print("\n" + '\n'.join(report_lines))

def main():
    """Fonction principale."""
    logger.info("🔍 DÉBUT DE L'ANALYSE DES RÉSULTATS EXPÉRIMENTAUX")
    
    # Charger les données
    df_exp = load_experimental_results()
    if df_exp is None:
        return
    
    df_sim = load_simulation_results()
    
    # Analyses
    gap_stats, lecran_stats = analyze_prediction_ranges(df_exp)
    correlations = analyze_intensity_correlation(df_exp)
    compare_with_simulation_ranges(df_exp, df_sim)
    
    # Visualisations
    create_detailed_analysis_plots(df_exp)
    
    # Rapport final
    generate_analysis_report(df_exp, gap_stats, lecran_stats, correlations)
    
    logger.info("\n✅ ANALYSE TERMINÉE")

if __name__ == "__main__":
    main()
