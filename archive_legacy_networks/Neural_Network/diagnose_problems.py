#!/usr/bin/env python3
"""
Comprehensive Problem Diagnosis
Author: Oussama GUELFAA
Date: 06 - 06 - 2025

Script de diagnostic complet pour identifier tous les problèmes potentiels
similaires à la précision excessive des labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

def diagnose_all_problems():
    """Diagnostic complet de tous les problèmes potentiels."""
    
    print("="*80)
    print("DIAGNOSTIC COMPLET DES PROBLÈMES POTENTIELS")
    print("="*80)
    
    # 1. Charger les données
    df_profiles = pd.read_csv('processed_data/intensity_profiles_truncated_600.csv')
    df_params = pd.read_csv('processed_data/parameters_truncated_600.csv')
    
    X = df_profiles.values
    y = df_params[['L_ecran', 'gap']].values
    
    print(f"Données chargées: X{X.shape}, y{y.shape}")
    
    # 2. PROBLÈME 1: Précision excessive
    print(f"\n{'='*60}")
    print(f"1. ANALYSE DE LA PRÉCISION")
    print(f"{'='*60}")
    
    print(f"Précision L_ecran:")
    for i in range(min(3, len(y))):
        val = y[i, 0]
        decimals = len(f"{val:.15f}".split('.')[-1].rstrip('0'))
        print(f"  {val:.15f} → {decimals} décimales")
    
    print(f"\nPrécision gap:")
    for i in range(min(3, len(y))):
        val = y[i, 1]
        decimals = len(f"{val:.15f}".split('.')[-1].rstrip('0'))
        print(f"  {val:.15f} → {decimals} décimales")
    
    # 3. PROBLÈME 2: Échelles déséquilibrées
    print(f"\n{'='*60}")
    print(f"2. ANALYSE DES ÉCHELLES")
    print(f"{'='*60}")
    
    L_range = y[:, 0].max() - y[:, 0].min()
    gap_range = y[:, 1].max() - y[:, 1].min()
    scale_ratio = L_range / gap_range
    
    print(f"Plages des paramètres:")
    print(f"  L_ecran: [{y[:, 0].min():.3f}, {y[:, 0].max():.3f}] → plage = {L_range:.3f}")
    print(f"  gap: [{y[:, 1].min():.6f}, {y[:, 1].max():.6f}] → plage = {gap_range:.6f}")
    print(f"  Ratio d'échelle: {scale_ratio:.1f}x")
    
    if scale_ratio > 10:
        print(f"  ⚠️  PROBLÈME: Échelles très déséquilibrées!")
    
    # 4. PROBLÈME 3: Distribution des profils
    print(f"\n{'='*60}")
    print(f"3. ANALYSE DES PROFILS D'INTENSITÉ")
    print(f"{'='*60}")
    
    print(f"Statistiques des profils:")
    print(f"  Min global: {X.min():.6f}")
    print(f"  Max global: {X.max():.6f}")
    print(f"  Moyenne globale: {X.mean():.6f}")
    print(f"  Écart-type global: {X.std():.6f}")
    
    # Analyser la variabilité par profil
    profile_means = X.mean(axis=1)
    profile_stds = X.std(axis=1)
    
    print(f"\nVariabilité entre profils:")
    print(f"  Moyennes par profil: [{profile_means.min():.6f}, {profile_means.max():.6f}]")
    print(f"  Écart-types par profil: [{profile_stds.min():.6f}, {profile_stds.max():.6f}]")
    
    # 5. PROBLÈME 4: Outliers
    print(f"\n{'='*60}")
    print(f"4. DÉTECTION D'OUTLIERS")
    print(f"{'='*60}")
    
    # Outliers dans les paramètres
    z_scores_y = np.abs(stats.zscore(y, axis=0))
    outliers_L = z_scores_y[:, 0] > 3
    outliers_gap = z_scores_y[:, 1] > 3
    
    print(f"Outliers dans les paramètres (|z| > 3):")
    print(f"  L_ecran: {outliers_L.sum()}/{len(y)} ({outliers_L.mean()*100:.1f}%)")
    print(f"  gap: {outliers_gap.sum()}/{len(y)} ({outliers_gap.mean()*100:.1f}%)")
    
    # Outliers dans les profils
    profile_z_scores = np.abs(stats.zscore(profile_means))
    profile_outliers = profile_z_scores > 3
    print(f"  Profils: {profile_outliers.sum()}/{len(X)} ({profile_outliers.mean()*100:.1f}%)")
    
    # 6. PROBLÈME 5: Distribution non uniforme
    print(f"\n{'='*60}")
    print(f"5. ANALYSE DE LA DISTRIBUTION")
    print(f"{'='*60}")
    
    # Analyser la distribution dans l'espace des paramètres
    L_unique = len(np.unique(y[:, 0]))
    gap_unique = len(np.unique(y[:, 1]))
    
    print(f"Diversité des paramètres:")
    print(f"  L_ecran: {L_unique} valeurs uniques")
    print(f"  gap: {gap_unique} valeurs uniques")
    print(f"  Combinaisons théoriques: {L_unique * gap_unique}")
    print(f"  Combinaisons réelles: {len(y)}")
    
    coverage = len(y) / (L_unique * gap_unique)
    print(f"  Couverture de l'espace: {coverage:.1%}")
    
    # 7. PROBLÈME 6: Corrélations parasites
    print(f"\n{'='*60}")
    print(f"6. ANALYSE DES CORRÉLATIONS")
    print(f"{'='*60}")
    
    # Corrélation entre L_ecran et gap
    corr_params = np.corrcoef(y[:, 0], y[:, 1])[0, 1]
    print(f"Corrélation L_ecran ↔ gap: {corr_params:.4f}")
    
    if abs(corr_params) > 0.3:
        print(f"  ⚠️  PROBLÈME: Corrélation forte entre paramètres!")
    
    # Corrélations avec les profils
    profile_mean = X.mean(axis=1)
    corr_profile_L = np.corrcoef(profile_mean, y[:, 0])[0, 1]
    corr_profile_gap = np.corrcoef(profile_mean, y[:, 1])[0, 1]
    
    print(f"Corrélations profil ↔ paramètres:")
    print(f"  Profil ↔ L_ecran: {corr_profile_L:.4f}")
    print(f"  Profil ↔ gap: {corr_profile_gap:.4f}")
    
    # 8. PROBLÈME 7: Zones problématiques
    print(f"\n{'='*60}")
    print(f"7. IDENTIFICATION DES ZONES PROBLÉMATIQUES")
    print(f"{'='*60}")
    
    # Analyser les profils par zones de paramètres
    gap_low = y[:, 1] < 0.2
    gap_high = y[:, 1] > 1.0
    
    print(f"Analyse par zones de gap:")
    print(f"  gap < 0.2 µm: {gap_low.sum()} échantillons")
    print(f"  gap > 1.0 µm: {gap_high.sum()} échantillons")
    
    if gap_low.sum() > 0 and gap_high.sum() > 0:
        profile_low = X[gap_low].mean(axis=0)
        profile_high = X[gap_high].mean(axis=0)
        profile_diff = np.abs(profile_low - profile_high).mean()
        print(f"  Différence moyenne profils: {profile_diff:.6f}")
        
        if profile_diff < 0.01:
            print(f"  ⚠️  PROBLÈME: Profils très similaires pour gaps différents!")
    
    # 9. RECOMMANDATIONS
    print(f"\n{'='*60}")
    print(f"8. RECOMMANDATIONS PRIORITAIRES")
    print(f"{'='*60}")
    
    recommendations = []
    
    # Précision
    max_decimals_L = max(len(f"{val:.15f}".split('.')[-1].rstrip('0')) for val in y[:3, 0])
    max_decimals_gap = max(len(f"{val:.15f}".split('.')[-1].rstrip('0')) for val in y[:3, 1])
    
    if max_decimals_L > 6 or max_decimals_gap > 6:
        recommendations.append("🎯 Arrondir les labels à 3-6 décimales maximum")
    
    if scale_ratio > 10:
        recommendations.append("⚖️ Normaliser les paramètres séparément")
    
    if abs(corr_params) > 0.3:
        recommendations.append("🔄 Vérifier l'indépendance des paramètres")
    
    if outliers_L.sum() > 0 or outliers_gap.sum() > 0:
        recommendations.append("🚫 Supprimer ou traiter les outliers")
    
    if coverage < 0.8:
        recommendations.append("📊 Améliorer la couverture de l'espace des paramètres")
    
    if abs(corr_profile_gap) < 0.2:
        recommendations.append("🔍 Le signal gap est très faible dans les profils")
    
    print(f"Actions recommandées:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    if not recommendations:
        print(f"  ✅ Aucun problème majeur détecté dans les données")
    
    return {
        'scale_ratio': scale_ratio,
        'outliers_L': outliers_L.sum(),
        'outliers_gap': outliers_gap.sum(),
        'corr_params': corr_params,
        'corr_profile_gap': corr_profile_gap,
        'coverage': coverage,
        'recommendations': recommendations
    }

def main():
    """Fonction principale."""
    
    print("="*80)
    print("DIAGNOSTIC COMPLET DES PROBLÈMES POTENTIELS")
    print("Identification de tous les problèmes similaires à la précision excessive")
    print("="*80)
    
    results = diagnose_all_problems()
    
    print(f"\n{'='*80}")
    print(f"RÉSUMÉ DU DIAGNOSTIC")
    print(f"{'='*80}")
    
    print(f"Problèmes détectés: {len(results['recommendations'])}")
    print(f"Priorité: Traiter les problèmes dans l'ordre listé")
    print(f"\nProchain test recommandé:")
    print(f"  1. Arrondir les labels à 3 décimales")
    print(f"  2. Normaliser L_ecran et gap séparément")
    print(f"  3. Utiliser une loss pondérée pour gap")

if __name__ == "__main__":
    main()
