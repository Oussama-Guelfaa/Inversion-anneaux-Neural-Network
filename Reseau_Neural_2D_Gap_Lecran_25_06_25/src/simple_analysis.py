#!/usr/bin/env python3
"""
Analyse simple des résultats CSV

Auteur: Oussama GUELFAA
Date: 25/06/2025

Analyse simple et robuste des résultats du test complet.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_csv_results():
    """Analyse simple des résultats CSV."""
    logger.info("📊 ANALYSE SIMPLE DES RÉSULTATS")
    logger.info("="*50)
    
    # Charger le CSV
    csv_file = Path("../results/test_complet_modele_ameliore_20250626_125500.csv")
    
    # Lire seulement les lignes de données (exclure la ligne de stats)
    df = pd.read_csv(csv_file)
    
    # Filtrer les données numériques valides
    df_clean = df[df['filename'] != 'STATISTIQUES_GLOBALES'].copy()
    
    # Convertir en numérique et supprimer les NaN
    df_clean['Gap_reel'] = pd.to_numeric(df_clean['Gap_reel'], errors='coerce')
    df_clean['Gap_predit'] = pd.to_numeric(df_clean['Gap_predit'], errors='coerce')
    df_clean['Lecran_reel'] = pd.to_numeric(df_clean['Lecran_reel'], errors='coerce')
    df_clean['Lecran_predit'] = pd.to_numeric(df_clean['Lecran_predit'], errors='coerce')
    df_clean['erreur_Gap'] = pd.to_numeric(df_clean['erreur_Gap'], errors='coerce')
    df_clean['erreur_Lecran'] = pd.to_numeric(df_clean['erreur_Lecran'], errors='coerce')
    
    # Supprimer les lignes avec des NaN
    df_valid = df_clean.dropna()
    
    logger.info(f"✅ Données valides: {len(df_valid)}/{len(df_clean)} échantillons")
    
    if len(df_valid) == 0:
        logger.error("❌ Aucune donnée valide trouvée")
        return
    
    # Statistiques Gap
    gap_mae = df_valid['erreur_Gap'].mean()
    gap_std = df_valid['erreur_Gap'].std()
    gap_median = df_valid['erreur_Gap'].median()
    gap_min = df_valid['erreur_Gap'].min()
    gap_max = df_valid['erreur_Gap'].max()
    
    # Statistiques L_écran
    L_ecran_mae = df_valid['erreur_Lecran'].mean()
    L_ecran_std = df_valid['erreur_Lecran'].std()
    L_ecran_median = df_valid['erreur_Lecran'].median()
    L_ecran_min = df_valid['erreur_Lecran'].min()
    L_ecran_max = df_valid['erreur_Lecran'].max()
    
    # Calculer R²
    from sklearn.metrics import r2_score
    gap_r2 = r2_score(df_valid['Gap_reel'], df_valid['Gap_predit'])
    L_ecran_r2 = r2_score(df_valid['Lecran_reel'], df_valid['Lecran_predit'])
    
    logger.info(f"\n📊 STATISTIQUES GLOBALES ({len(df_valid)} échantillons)")
    logger.info("="*60)
    logger.info(f"GAP:")
    logger.info(f"   MAE: {gap_mae:.4f} ± {gap_std:.4f} µm")
    logger.info(f"   Médiane: {gap_median:.4f} µm")
    logger.info(f"   Min-Max: {gap_min:.4f} - {gap_max:.4f} µm")
    logger.info(f"   R²: {gap_r2:.3f}")
    
    logger.info(f"\nL_ÉCRAN:")
    logger.info(f"   MAE: {L_ecran_mae:.1f} ± {L_ecran_std:.1f} µm")
    logger.info(f"   Médiane: {L_ecran_median:.1f} µm")
    logger.info(f"   Min-Max: {L_ecran_min:.1f} - {L_ecran_max:.1f} µm")
    logger.info(f"   R²: {L_ecran_r2:.3f}")
    
    # Précision par tolérance
    logger.info(f"\n🎯 PRÉCISION PAR TOLÉRANCE:")
    
    # Gap
    tolerances_gap = [0.005, 0.01, 0.02, 0.05]
    logger.info("Gap:")
    for tol in tolerances_gap:
        good = (df_valid['erreur_Gap'] <= tol).sum()
        accuracy = good / len(df_valid) * 100
        logger.info(f"   ±{tol:.3f}µm: {good}/{len(df_valid)} = {accuracy:.1f}%")
    
    # L_écran
    tolerances_L_ecran = [0.2, 0.5, 1.0]
    logger.info("L_écran:")
    for tol in tolerances_L_ecran:
        good = (df_valid['erreur_Lecran'] <= tol).sum()
        accuracy = good / len(df_valid) * 100
        logger.info(f"   ±{tol:.1f}µm: {good}/{len(df_valid)} = {accuracy:.1f}%")
    
    # Meilleurs et pires cas
    logger.info(f"\n🏆 MEILLEURS CAS (Gap - 5 plus petites erreurs):")
    best_gap = df_valid.nsmallest(5, 'erreur_Gap')
    for _, row in best_gap.iterrows():
        logger.info(f"   {row['filename']}: Gap {row['Gap_reel']:.4f}→{row['Gap_predit']:.4f} "
                   f"(erreur: {row['erreur_Gap']:.6f}µm)")
    
    logger.info(f"\n❌ PIRES CAS (Gap - 5 plus grandes erreurs):")
    worst_gap = df_valid.nlargest(5, 'erreur_Gap')
    for _, row in worst_gap.iterrows():
        logger.info(f"   {row['filename']}: Gap {row['Gap_reel']:.4f}→{row['Gap_predit']:.4f} "
                   f"(erreur: {row['erreur_Gap']:.4f}µm)")
    
    # Analyse par plages de gap
    logger.info(f"\n📈 ANALYSE PAR PLAGE DE GAP:")
    gap_ranges = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
    
    for gap_min_range, gap_max_range in gap_ranges:
        mask = (df_valid['Gap_reel'] >= gap_min_range) & (df_valid['Gap_reel'] < gap_max_range)
        range_data = df_valid[mask]
        
        if len(range_data) > 0:
            range_mae = range_data['erreur_Gap'].mean()
            range_median = range_data['erreur_Gap'].median()
            range_count = len(range_data)
            
            # Pourcentage dans tolérance ±0.01µm
            tol_01 = (range_data['erreur_Gap'] <= 0.01).sum() / range_count * 100
            
            logger.info(f"   {gap_min_range:.1f}-{gap_max_range:.1f}µm: {range_count:3d} échantillons, "
                       f"MAE={range_mae:.4f}µm, Med={range_median:.4f}µm, ±0.01µm:{tol_01:4.1f}%")
    
    # Évaluation finale
    logger.info(f"\n🏆 ÉVALUATION FINALE:")
    
    if gap_r2 > 0.8 and L_ecran_r2 > 0.95:
        logger.info(f"   🎉 MODÈLE EXCELLENT !")
    elif gap_r2 > 0.6 and L_ecran_r2 > 0.9:
        logger.info(f"   ✅ MODÈLE DE BONNE QUALITÉ")
    elif gap_r2 > 0.4 and L_ecran_r2 > 0.8:
        logger.info(f"   ⚠️ MODÈLE ACCEPTABLE")
    else:
        logger.info(f"   ❌ MODÈLE À AMÉLIORER")
    
    # Recommandations
    if gap_mae <= 0.02:
        logger.info(f"   ✨ Gap: Précision excellente (MAE ≤ 0.02µm)")
    elif gap_mae <= 0.05:
        logger.info(f"   ✅ Gap: Précision bonne (MAE ≤ 0.05µm)")
    else:
        logger.info(f"   ⚠️ Gap: Précision à améliorer (MAE > 0.05µm)")
    
    if L_ecran_r2 > 0.98:
        logger.info(f"   ✨ L_écran: Prédiction quasi-parfaite (R² > 0.98)")
    elif L_ecran_r2 > 0.95:
        logger.info(f"   ✅ L_écran: Prédiction excellente (R² > 0.95)")
    
    # Créer un résumé CSV propre
    summary_data = {
        'Parametre': ['Gap', 'L_ecran'],
        'MAE': [gap_mae, L_ecran_mae],
        'R2': [gap_r2, L_ecran_r2],
        'Mediane_erreur': [gap_median, L_ecran_median],
        'Min_erreur': [gap_min, L_ecran_min],
        'Max_erreur': [gap_max, L_ecran_max],
        'Precision_tolerance_stricte': [
            (df_valid['erreur_Gap'] <= 0.01).sum() / len(df_valid) * 100,
            (df_valid['erreur_Lecran'] <= 0.5).sum() / len(df_valid) * 100
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = "../results/resume_performances_modele_ameliore.csv"
    summary_df.to_csv(summary_file, index=False, float_format='%.4f')
    
    logger.info(f"\n💾 RÉSUMÉ SAUVEGARDÉ: {summary_file}")
    
    return df_valid

def main():
    """Fonction principale."""
    df_valid = analyze_csv_results()
    logger.info(f"\n✅ ANALYSE TERMINÉE")

if __name__ == "__main__":
    main()
