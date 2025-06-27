#!/usr/bin/env python3
"""
Analyse des résultats du test complet

Auteur: Oussama GUELFAA
Date: 25/06/2025

Analyse et visualise les résultats du test complet sauvegardés dans le CSV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_results():
    """Charge les résultats du CSV le plus récent."""
    results_dir = Path("../results")
    csv_files = list(results_dir.glob("test_complet_modele_ameliore_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("Aucun fichier de résultats trouvé")
    
    # Prendre le fichier le plus récent
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"📊 Chargement des résultats: {latest_file.name}")
    
    # Charger le CSV en excluant la ligne de statistiques
    df = pd.read_csv(latest_file)
    df_data = df[df['filename'] != 'STATISTIQUES_GLOBALES'].copy()
    
    logger.info(f"✅ {len(df_data)} échantillons chargés")
    
    return df_data, latest_file

def analyze_results(df):
    """Analyse détaillée des résultats."""
    logger.info("🔍 ANALYSE DÉTAILLÉE DES RÉSULTATS")
    logger.info("="*50)
    
    # Statistiques de base
    gap_mae = df['erreur_Gap'].mean()
    gap_std = df['erreur_Gap'].std()
    gap_median = df['erreur_Gap'].median()
    gap_q25 = df['erreur_Gap'].quantile(0.25)
    gap_q75 = df['erreur_Gap'].quantile(0.75)
    
    L_ecran_mae = df['erreur_Lecran'].mean()
    L_ecran_std = df['erreur_Lecran'].std()
    L_ecran_median = df['erreur_Lecran'].median()
    L_ecran_q25 = df['erreur_Lecran'].quantile(0.25)
    L_ecran_q75 = df['erreur_Lecran'].quantile(0.75)
    
    logger.info(f"GAP - Erreurs (µm):")
    logger.info(f"   Moyenne: {gap_mae:.4f} ± {gap_std:.4f}")
    logger.info(f"   Médiane: {gap_median:.4f}")
    logger.info(f"   Q25-Q75: {gap_q25:.4f} - {gap_q75:.4f}")
    logger.info(f"   Min-Max: {df['erreur_Gap'].min():.4f} - {df['erreur_Gap'].max():.4f}")
    
    logger.info(f"\nL_ÉCRAN - Erreurs (µm):")
    logger.info(f"   Moyenne: {L_ecran_mae:.1f} ± {L_ecran_std:.1f}")
    logger.info(f"   Médiane: {L_ecran_median:.1f}")
    logger.info(f"   Q25-Q75: {L_ecran_q25:.1f} - {L_ecran_q75:.1f}")
    logger.info(f"   Min-Max: {df['erreur_Lecran'].min():.1f} - {df['erreur_Lecran'].max():.1f}")
    
    # Analyse par plages de gap
    logger.info(f"\n📈 ANALYSE PAR PLAGE DE GAP:")
    gap_ranges = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), 
                  (0.2, 0.25), (0.25, 0.3), (0.3, 0.35), (0.35, 0.4)]
    
    for gap_min, gap_max in gap_ranges:
        mask = (df['Gap_reel'] >= gap_min) & (df['Gap_reel'] < gap_max)
        range_data = df[mask]
        
        if len(range_data) > 0:
            range_mae = range_data['erreur_Gap'].mean()
            range_median = range_data['erreur_Gap'].median()
            range_count = len(range_data)
            
            # Pourcentage dans différentes tolérances
            tol_005 = (range_data['erreur_Gap'] <= 0.005).sum() / range_count * 100
            tol_01 = (range_data['erreur_Gap'] <= 0.01).sum() / range_count * 100
            tol_02 = (range_data['erreur_Gap'] <= 0.02).sum() / range_count * 100
            
            logger.info(f"   {gap_min:.2f}-{gap_max:.2f}µm: {range_count:3d} échantillons, "
                       f"MAE={range_mae:.4f}µm, Med={range_median:.4f}µm, "
                       f"±0.005µm:{tol_005:4.1f}%, ±0.01µm:{tol_01:4.1f}%, ±0.02µm:{tol_02:4.1f}%")
    
    # Analyse par plages de L_écran
    logger.info(f"\n📈 ANALYSE PAR PLAGE DE L_ÉCRAN:")
    L_ranges = [(4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 8.0)]
    
    for L_min, L_max in L_ranges:
        mask = (df['Lecran_reel'] >= L_min) & (df['Lecran_reel'] < L_max)
        range_data = df[mask]
        
        if len(range_data) > 0:
            range_mae = range_data['erreur_Lecran'].mean()
            range_median = range_data['erreur_Lecran'].median()
            range_count = len(range_data)
            
            # Pourcentage dans différentes tolérances
            tol_02 = (range_data['erreur_Lecran'] <= 0.2).sum() / range_count * 100
            tol_05 = (range_data['erreur_Lecran'] <= 0.5).sum() / range_count * 100
            
            logger.info(f"   {L_min:.1f}-{L_max:.1f}µm: {range_count:3d} échantillons, "
                       f"MAE={range_mae:.1f}µm, Med={range_median:.1f}µm, "
                       f"±0.2µm:{tol_02:4.1f}%, ±0.5µm:{tol_05:4.1f}%")
    
    # Identifier les meilleurs et pires cas
    logger.info(f"\n🏆 MEILLEURS CAS (Gap):")
    best_gap = df.nsmallest(5, 'erreur_Gap')
    for _, row in best_gap.iterrows():
        logger.info(f"   {row['filename']}: Gap {row['Gap_reel']:.4f}→{row['Gap_predit']:.4f} "
                   f"(erreur: {row['erreur_Gap']:.4f}µm)")
    
    logger.info(f"\n❌ PIRES CAS (Gap):")
    worst_gap = df.nlargest(5, 'erreur_Gap')
    for _, row in worst_gap.iterrows():
        logger.info(f"   {row['filename']}: Gap {row['Gap_reel']:.4f}→{row['Gap_predit']:.4f} "
                   f"(erreur: {row['erreur_Gap']:.4f}µm)")
    
    return df

def create_visualizations(df, output_dir="../plots"):
    """Crée des visualisations des résultats."""
    logger.info(f"\n📊 CRÉATION DES VISUALISATIONS")
    
    # Créer le dossier de sortie
    Path(output_dir).mkdir(exist_ok=True)
    
    # Configuration matplotlib
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: Histogrammes des erreurs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogramme Gap
    ax1.hist(df['erreur_Gap'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(df['erreur_Gap'].mean(), color='red', linestyle='--', 
                label=f'Moyenne: {df["erreur_Gap"].mean():.4f}µm')
    ax1.axvline(df['erreur_Gap'].median(), color='orange', linestyle='--', 
                label=f'Médiane: {df["erreur_Gap"].median():.4f}µm')
    ax1.set_xlabel('Erreur Gap (µm)')
    ax1.set_ylabel('Fréquence')
    ax1.set_title('Distribution des Erreurs - Gap')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogramme L_écran
    ax2.hist(df['erreur_Lecran'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(df['erreur_Lecran'].mean(), color='red', linestyle='--', 
                label=f'Moyenne: {df["erreur_Lecran"].mean():.1f}µm')
    ax2.axvline(df['erreur_Lecran'].median(), color='orange', linestyle='--', 
                label=f'Médiane: {df["erreur_Lecran"].median():.1f}µm')
    ax2.set_xlabel('Erreur L_écran (µm)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution des Erreurs - L_écran')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/histogrammes_erreurs.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Scatter plots prédictions vs réel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter Gap
    ax1.scatter(df['Gap_reel'], df['Gap_predit'], alpha=0.6, s=20)
    min_gap = min(df['Gap_reel'].min(), df['Gap_predit'].min())
    max_gap = max(df['Gap_reel'].max(), df['Gap_predit'].max())
    ax1.plot([min_gap, max_gap], [min_gap, max_gap], 'r--', label='Prédiction parfaite')
    ax1.set_xlabel('Gap Réel (µm)')
    ax1.set_ylabel('Gap Prédit (µm)')
    ax1.set_title('Gap: Prédit vs Réel')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter L_écran
    ax2.scatter(df['Lecran_reel'], df['Lecran_predit'], alpha=0.6, s=20)
    min_L = min(df['Lecran_reel'].min(), df['Lecran_predit'].min())
    max_L = max(df['Lecran_reel'].max(), df['Lecran_predit'].max())
    ax2.plot([min_L, max_L], [min_L, max_L], 'r--', label='Prédiction parfaite')
    ax2.set_xlabel('L_écran Réel (µm)')
    ax2.set_ylabel('L_écran Prédit (µm)')
    ax2.set_title('L_écran: Prédit vs Réel')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Erreurs en fonction des valeurs réelles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Erreur Gap vs Gap réel
    ax1.scatter(df['Gap_reel'], df['erreur_Gap'], alpha=0.6, s=20)
    ax1.axhline(0.01, color='red', linestyle='--', label='Tolérance ±0.01µm')
    ax1.axhline(0.02, color='orange', linestyle='--', label='Tolérance ±0.02µm')
    ax1.set_xlabel('Gap Réel (µm)')
    ax1.set_ylabel('Erreur Gap (µm)')
    ax1.set_title('Erreur Gap en fonction de la valeur réelle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Erreur L_écran vs L_écran réel
    ax2.scatter(df['Lecran_reel'], df['erreur_Lecran'], alpha=0.6, s=20)
    ax2.axhline(0.2, color='red', linestyle='--', label='Tolérance ±0.2µm')
    ax2.axhline(0.5, color='orange', linestyle='--', label='Tolérance ±0.5µm')
    ax2.set_xlabel('L_écran Réel (µm)')
    ax2.set_ylabel('Erreur L_écran (µm)')
    ax2.set_title('Erreur L_écran en fonction de la valeur réelle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/erreurs_vs_valeurs.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Visualisations sauvegardées dans {output_dir}/")
    logger.info(f"   - histogrammes_erreurs.png")
    logger.info(f"   - scatter_predictions.png") 
    logger.info(f"   - erreurs_vs_valeurs.png")

def main():
    """Fonction principale d'analyse."""
    logger.info("🔍 ANALYSE DES RÉSULTATS DU TEST COMPLET")
    logger.info("="*60)
    
    # Charger les résultats
    df, csv_file = load_results()
    
    # Analyser les résultats
    df_analyzed = analyze_results(df)
    
    # Créer les visualisations
    create_visualizations(df_analyzed)
    
    logger.info(f"\n✅ ANALYSE TERMINÉE")
    logger.info(f"   Fichier source: {csv_file.name}")
    logger.info(f"   Échantillons analysés: {len(df_analyzed)}")

if __name__ == "__main__":
    main()
