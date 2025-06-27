#!/usr/bin/env python3
"""
Création d'un CSV propre avec tous les résultats

Auteur: Oussama GUELFAA
Date: 25/06/2025

Crée un fichier CSV propre avec tous les résultats du test complet.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_clean_csv():
    """Crée un CSV propre avec tous les résultats."""
    logger.info("🧹 CRÉATION D'UN CSV PROPRE")
    logger.info("="*40)
    
    # Charger le CSV original
    csv_file = Path("../results/test_complet_modele_ameliore_20250626_125500.csv")
    df = pd.read_csv(csv_file)
    
    # Filtrer les données numériques valides
    df_clean = df[df['filename'] != 'STATISTIQUES_GLOBALES'].copy()
    
    # Convertir en numérique
    df_clean['Gap_reel'] = pd.to_numeric(df_clean['Gap_reel'], errors='coerce')
    df_clean['Gap_predit'] = pd.to_numeric(df_clean['Gap_predit'], errors='coerce')
    df_clean['Lecran_reel'] = pd.to_numeric(df_clean['Lecran_reel'], errors='coerce')
    df_clean['Lecran_predit'] = pd.to_numeric(df_clean['Lecran_predit'], errors='coerce')
    df_clean['erreur_Gap'] = pd.to_numeric(df_clean['erreur_Gap'], errors='coerce')
    df_clean['erreur_Lecran'] = pd.to_numeric(df_clean['erreur_Lecran'], errors='coerce')
    
    # Supprimer les lignes avec des NaN
    df_valid = df_clean.dropna()
    
    logger.info(f"✅ Données valides: {len(df_valid)}/{len(df_clean)} échantillons")
    
    # Arrondir les valeurs pour plus de lisibilité
    df_valid['Gap_reel'] = df_valid['Gap_reel'].round(4)
    df_valid['Gap_predit'] = df_valid['Gap_predit'].round(4)
    df_valid['Lecran_reel'] = df_valid['Lecran_reel'].round(1)
    df_valid['Lecran_predit'] = df_valid['Lecran_predit'].round(1)
    df_valid['erreur_Gap'] = df_valid['erreur_Gap'].round(4)
    df_valid['erreur_Lecran'] = df_valid['erreur_Lecran'].round(1)
    
    # Ajouter des colonnes d'analyse
    df_valid['Gap_precision_001'] = (df_valid['erreur_Gap'] <= 0.01).astype(int)
    df_valid['Gap_precision_002'] = (df_valid['erreur_Gap'] <= 0.02).astype(int)
    df_valid['Lecran_precision_05'] = (df_valid['erreur_Lecran'] <= 0.5).astype(int)
    
    # Réorganiser les colonnes
    columns_order = [
        'filename',
        'Gap_reel', 'Gap_predit', 'erreur_Gap',
        'Lecran_reel', 'Lecran_predit', 'erreur_Lecran',
        'Gap_precision_001', 'Gap_precision_002', 'Lecran_precision_05'
    ]
    
    df_final = df_valid[columns_order].copy()
    
    # Trier par erreur Gap croissante
    df_final = df_final.sort_values('erreur_Gap')
    
    # Sauvegarder le CSV propre
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_csv_file = f"../results/resultats_complets_propres_{timestamp}.csv"
    
    df_final.to_csv(clean_csv_file, index=False)
    
    logger.info(f"💾 CSV propre sauvegardé: {clean_csv_file}")
    logger.info(f"   Colonnes: {list(df_final.columns)}")
    logger.info(f"   Échantillons: {len(df_final)}")
    
    # Afficher un aperçu
    logger.info(f"\n📋 APERÇU (10 meilleures prédictions Gap):")
    for i, (_, row) in enumerate(df_final.head(10).iterrows()):
        logger.info(f"   {i+1:2d}. {row['filename']}: "
                   f"Gap {row['Gap_reel']:.4f}→{row['Gap_predit']:.4f} "
                   f"(±{row['erreur_Gap']:.4f}µm), "
                   f"L_écran {row['Lecran_reel']:.1f}→{row['Lecran_predit']:.1f} "
                   f"(±{row['erreur_Lecran']:.1f}µm)")
    
    # Statistiques finales
    gap_mae = df_final['erreur_Gap'].mean()
    L_ecran_mae = df_final['erreur_Lecran'].mean()
    gap_precision_001 = df_final['Gap_precision_001'].mean() * 100
    gap_precision_002 = df_final['Gap_precision_002'].mean() * 100
    L_ecran_precision_05 = df_final['Lecran_precision_05'].mean() * 100
    
    logger.info(f"\n📊 STATISTIQUES FINALES:")
    logger.info(f"   Gap MAE: {gap_mae:.4f} µm")
    logger.info(f"   L_écran MAE: {L_ecran_mae:.1f} µm")
    logger.info(f"   Gap ±0.01µm: {gap_precision_001:.1f}%")
    logger.info(f"   Gap ±0.02µm: {gap_precision_002:.1f}%")
    logger.info(f"   L_écran ±0.5µm: {L_ecran_precision_05:.1f}%")
    
    return clean_csv_file, df_final

def main():
    """Fonction principale."""
    clean_csv_file, df_final = create_clean_csv()
    
    logger.info(f"\n✅ CSV PROPRE CRÉÉ AVEC SUCCÈS")
    logger.info(f"   Fichier: {Path(clean_csv_file).name}")
    logger.info(f"   Chemin complet: {Path(clean_csv_file).resolve()}")

if __name__ == "__main__":
    main()
