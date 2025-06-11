#!/usr/bin/env python3
"""
Analyse des erreurs relatives pour le bruit 2% avec seuil 10%

Auteur: Oussama GUELFAA
Date: 11 - 01 - 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def main():
    print("="*80)
    print("ANALYSE ERREURS RELATIVES - BRUIT 2% - SEUIL 10%")
    print("="*80)
    
    # Charger les données 2%
    df = pd.read_csv('../results/predictions_noise_2percent.csv')
    print(f"✅ Données chargées: {len(df)} prédictions")
    
    # Filtrer la plage 0.5-1.0 µm
    mask = (df['gap_true'] >= 0.5) & (df['gap_true'] <= 1.0)
    data = df[mask]
    
    y_true = data['gap_true'].values
    y_pred = data['gap_predicted'].values
    
    print(f"📊 Points dans la plage [0.5-1.0 µm]: {len(data)}")
    
    # Calculer erreurs relatives
    rel_err = np.abs((y_pred - y_true) / y_true) * 100
    
    print(f"\n📈 STATISTIQUES DES ERREURS RELATIVES:")
    print(f"  Erreur minimale:  {rel_err.min():.3f}%")
    print(f"  Erreur maximale:  {rel_err.max():.3f}%")
    print(f"  Erreur moyenne:   {rel_err.mean():.3f}%")
    print(f"  Erreur médiane:   {np.median(rel_err):.3f}%")
    print(f"  Écart-type:       {rel_err.std():.3f}%")
    
    # Évaluation seuil 10%
    under_10 = (rel_err < 10).sum()
    pct_under_10 = 100 * under_10 / len(rel_err)
    
    print(f"\n🎯 ÉVALUATION DU SEUIL 10%:")
    print(f"  Points < 10%: {under_10}/{len(rel_err)} ({pct_under_10:.1f}%)")
    
    if rel_err.max() < 10:
        print(f"  ✅ EXCELLENT: Toutes les erreurs < 10%")
        confirmation = "OUI - Toutes les erreurs restent inférieures à 10%"
    else:
        print(f"  ❌ Quelques erreurs dépassent 10%")
        confirmation = f"PARTIELLEMENT - {pct_under_10:.1f}% des erreurs < 10%"
    
    print(f"\n✅ CONFIRMATION CLAIRE:")
    print(f"   {confirmation}")
    
    # Top 5 erreurs
    print(f"\n📋 TOP 5 ERREURS:")
    sorted_idx = np.argsort(rel_err)[::-1]
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {y_true[idx]:.4f} µm → {y_pred[idx]:.4f} µm (erreur: {rel_err[idx]:.3f}%)")
    
    # Créer les visualisations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.7, s=60, c='blue', edgecolors='black', linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite (y=x)')
    
    r2_local = r2_score(y_true, y_pred)
    axes[0].set_xlabel('Gap réel (µm)', fontweight='bold')
    axes[0].set_ylabel('Gap prédit (µm)', fontweight='bold')
    axes[0].set_title(f'Bruit 2% - Plage [0.5-1.0 µm]\\n{len(y_true)} points, R² = {r2_local:.4f}', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axis('equal')
    
    # 2. Distribution des erreurs
    axes[1].hist(rel_err, bins=12, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(x=10.0, color='red', linestyle='--', linewidth=2, label='Seuil 10%')
    axes[1].axvline(x=rel_err.mean(), color='green', linestyle='-', linewidth=2, 
                   label=f'Moyenne: {rel_err.mean():.2f}%')
    axes[1].axvline(x=np.median(rel_err), color='blue', linestyle='-', linewidth=2, 
                   label=f'Médiane: {np.median(rel_err):.2f}%')
    
    axes[1].set_xlabel('Erreur relative (%)', fontweight='bold')
    axes[1].set_ylabel('Fréquence', fontweight='bold')
    axes[1].set_title('Distribution des Erreurs Relatives\\nBruit 2%', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. Erreur vs Gap réel
    colors = ['green' if err < 10 else 'red' for err in rel_err]
    axes[2].scatter(y_true, rel_err, alpha=0.7, s=60, c=colors, edgecolors='black', linewidth=0.5)
    axes[2].axhline(y=10.0, color='red', linestyle='--', linewidth=2, label='Seuil 10%')
    axes[2].axhline(y=rel_err.mean(), color='green', linestyle='-', linewidth=2, 
                   label=f'Moyenne: {rel_err.mean():.2f}%')
    
    axes[2].set_xlabel('Gap réel (µm)', fontweight='bold')
    axes[2].set_ylabel('Erreur relative (%)', fontweight='bold')
    axes[2].set_title('Erreur Relative vs Gap Réel\\nBruit 2%', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('../plots/local_generalization_2percent_seuil10.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Graphique sauvegardé: ../plots/local_generalization_2percent_seuil10.png")
    
    # Résumé final
    print(f"\n" + "="*80)
    print("RÉSUMÉ FINAL - BRUIT 2% AVEC SEUIL 10%")
    print("="*80)
    print(f"🎯 PLAGE ANALYSÉE: 0.5-1.0 µm")
    print(f"📊 POINTS ANALYSÉS: {len(rel_err)}")
    print(f"📈 ERREUR MAXIMALE: {rel_err.max():.3f}%")
    print(f"📈 ERREUR MOYENNE: {rel_err.mean():.3f}%")
    print(f"📈 ERREUR MÉDIANE: {np.median(rel_err):.3f}%")
    print(f"🎯 SEUIL 10%: {under_10}/{len(rel_err)} points ({pct_under_10:.1f}%)")
    
    if rel_err.max() < 10:
        print(f"✅ CONFIRMATION: Toutes les erreurs < 10%")
        print(f"✅ GÉNÉRALISATION LOCALE EXCELLENTE")
    else:
        print(f"⚠️  ATTENTION: Erreur max = {rel_err.max():.3f}% > 10%")
        print(f"⚠️  GÉNÉRALISATION LOCALE ACCEPTABLE")

if __name__ == "__main__":
    main()
