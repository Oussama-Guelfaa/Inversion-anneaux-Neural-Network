#!/usr/bin/env python3
"""
Analyse des erreurs relatives à partir des résultats existants

Ce script utilise les prédictions déjà sauvegardées du test de robustesse au bruit (5%)
pour calculer les erreurs relatives dans la plage 0.5-1.0 µm.

Auteur: Oussama GUELFAA
Date: 11 - 01 - 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def load_existing_predictions():
    """Charge les prédictions existantes du test de robustesse au bruit 5%."""
    
    print("=== CHARGEMENT DES RÉSULTATS EXISTANTS ===")
    
    # Chemin vers les prédictions du test 2% bruit
    predictions_file = "../results/predictions_noise_2percent.csv"
    
    if not os.path.exists(predictions_file):
        print(f"❌ Fichier non trouvé: {predictions_file}")
        print("Fichiers disponibles dans ../results/:")
        if os.path.exists("../results/"):
            for f in os.listdir("../results/"):
                if f.endswith('.csv'):
                    print(f"  - {f}")
        return None, None
    
    # Charger les données
    df = pd.read_csv(predictions_file)
    
    print(f"✅ Données chargées: {len(df)} prédictions")
    print(f"Colonnes disponibles: {list(df.columns)}")
    
    # Extraire les valeurs
    y_true = df['gap_true'].values
    y_pred = df['gap_predicted'].values
    
    print(f"Gap range: {y_true.min():.4f} - {y_true.max():.4f} µm")
    
    return y_true, y_pred

def analyze_local_generalization(y_true, y_pred, range_min=0.5, range_max=1.0, threshold=10.0):
    """
    Analyse la généralisation locale dans la plage spécifiée.
    
    Args:
        y_true: Valeurs réelles de gap
        y_pred: Valeurs prédites de gap
        range_min: Limite inférieure de la plage (µm)
        range_max: Limite supérieure de la plage (µm)
    """
    
    print(f"\n=== ANALYSE GÉNÉRALISATION LOCALE [{range_min}-{range_max} µm] ===")
    
    # Filtrer les points dans la plage spécifiée
    mask = (y_true >= range_min) & (y_true <= range_max)
    y_true_range = y_true[mask]
    y_pred_range = y_pred[mask]
    
    print(f"Nombre de points dans la plage [{range_min}-{range_max} µm]: {len(y_true_range)}")
    
    if len(y_true_range) == 0:
        print("❌ Aucun point dans la plage spécifiée")
        return None
    
    # Calculer les erreurs relatives selon la formule demandée
    # Erreur relative (%) = |Gap prédit - Gap réel| / Gap réel × 100
    relative_errors = np.abs((y_pred_range - y_true_range) / y_true_range) * 100
    
    # Statistiques des erreurs relatives
    max_error = np.max(relative_errors)
    mean_error = np.mean(relative_errors)
    median_error = np.median(relative_errors)
    std_error = np.std(relative_errors)
    min_error = np.min(relative_errors)
    
    print(f"\n📊 STATISTIQUES DES ERREURS RELATIVES:")
    print(f"  Erreur minimale:  {min_error:.3f}%")
    print(f"  Erreur maximale:  {max_error:.3f}%")
    print(f"  Erreur moyenne:   {mean_error:.3f}%")
    print(f"  Erreur médiane:   {median_error:.3f}%")
    print(f"  Écart-type:       {std_error:.3f}%")
    
    # Vérification du seuil (maintenant 10% pour le bruit 2%)
    points_under_threshold = np.sum(relative_errors < threshold)
    percentage_under_threshold = (points_under_threshold / len(relative_errors)) * 100

    print(f"\n🎯 ÉVALUATION DU SEUIL {threshold}%:")
    print(f"  Points < {threshold}% d'erreur: {points_under_threshold}/{len(relative_errors)} ({percentage_under_threshold:.1f}%)")

    if max_error < threshold:
        print(f"  ✅ EXCELLENT: Toutes les erreurs < {threshold}%")
        print(f"  ✅ Généralisation locale exceptionnelle")
        confirmation = f"OUI - Toutes les erreurs restent inférieures à {threshold}%"
    elif percentage_under_threshold >= 95:
        print(f"  ✅ TRÈS BON: >95% des erreurs < {threshold}%")
        print(f"  ⚠️  Quelques points dépassent {threshold}% (max: {max_error:.3f}%)")
        confirmation = f"QUASI-OUI - {percentage_under_threshold:.1f}% des erreurs < {threshold}%"
    elif percentage_under_threshold >= 90:
        print(f"  ⚠️  ACCEPTABLE: >90% des erreurs < {threshold}%")
        print(f"  ⚠️  Erreur maximale: {max_error:.3f}%")
        confirmation = f"PARTIELLEMENT - {percentage_under_threshold:.1f}% des erreurs < {threshold}%"
    else:
        print(f"  ❌ PROBLÉMATIQUE: <90% des erreurs < {threshold}%")
        print(f"  ❌ Généralisation locale insuffisante")
        confirmation = f"NON - Seulement {percentage_under_threshold:.1f}% des erreurs < {threshold}%"
    
    # Analyse détaillée des points
    print(f"\n📋 ANALYSE DÉTAILLÉE DES POINTS (Top 10 erreurs):")
    print(f"{'Gap Réel':<10} {'Gap Prédit':<12} {'Erreur Abs':<12} {'Erreur Rel':<12}")
    print("-" * 50)
    
    # Trier par erreur relative décroissante
    sorted_indices = np.argsort(relative_errors)[::-1]
    
    # Afficher les 10 premiers (pires erreurs)
    for i in sorted_indices[:min(10, len(sorted_indices))]:
        gap_real = y_true_range[i]
        gap_pred = y_pred_range[i]
        abs_error = abs(gap_pred - gap_real)
        rel_error = relative_errors[i]
        
        print(f"{gap_real:<10.4f} {gap_pred:<12.4f} {abs_error:<12.4f} {rel_error:<12.3f}%")
    
    # Créer un graphique spécifique pour cette plage
    create_local_analysis_plot(y_true_range, y_pred_range, relative_errors, range_min, range_max, threshold)
    
    return {
        'n_points': len(y_true_range),
        'min_error': min_error,
        'max_error': max_error,
        'mean_error': mean_error,
        'median_error': median_error,
        'std_error': std_error,
        'points_under_threshold': points_under_threshold,
        'percentage_under_threshold': percentage_under_threshold,
        'all_under_threshold': max_error < threshold,
        'threshold': threshold,
        'confirmation': confirmation
    }

def create_local_analysis_plot(y_true, y_pred, relative_errors, range_min, range_max, threshold=10.0):
    """Crée un graphique spécifique pour l'analyse locale."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Scatter plot pour la plage spécifique
    axes[0].scatter(y_true, y_pred, alpha=0.7, s=60, c='blue', edgecolors='black', linewidth=0.5)
    
    # Ligne parfaite
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite (y=x)')
    
    # Calculer R² pour cette plage
    from sklearn.metrics import r2_score
    r2_local = r2_score(y_true, y_pred)
    
    axes[0].set_xlabel('Gap réel (µm)', fontweight='bold')
    axes[0].set_ylabel('Gap prédit (µm)', fontweight='bold')
    axes[0].set_title(f'Généralisation Locale [{range_min}-{range_max} µm]\n{len(y_true)} points, R² = {r2_local:.4f}', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axis('equal')
    
    # 2. Distribution des erreurs relatives
    axes[1].hist(relative_errors, bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Seuil {threshold}%')
    axes[1].axvline(x=np.mean(relative_errors), color='green', linestyle='-', linewidth=2, 
                   label=f'Moyenne: {np.mean(relative_errors):.2f}%')
    axes[1].axvline(x=np.median(relative_errors), color='blue', linestyle='-', linewidth=2, 
                   label=f'Médiane: {np.median(relative_errors):.2f}%')
    
    axes[1].set_xlabel('Erreur relative (%)', fontweight='bold')
    axes[1].set_ylabel('Fréquence', fontweight='bold')
    axes[1].set_title('Distribution des Erreurs Relatives', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. Erreur relative vs Gap réel
    colors = ['green' if err < threshold else 'red' for err in relative_errors]
    axes[2].scatter(y_true, relative_errors, alpha=0.7, s=60, c=colors, edgecolors='black', linewidth=0.5)
    axes[2].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Seuil {threshold}%')
    axes[2].axhline(y=np.mean(relative_errors), color='green', linestyle='-', linewidth=2, 
                   label=f'Moyenne: {np.mean(relative_errors):.2f}%')
    
    axes[2].set_xlabel('Gap réel (µm)', fontweight='bold')
    axes[2].set_ylabel('Erreur relative (%)', fontweight='bold')
    axes[2].set_title('Erreur Relative vs Gap Réel', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'../plots/local_generalization_analysis_{range_min}_{range_max}um_2percent.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Graphique d'analyse locale sauvegardé: ../plots/local_generalization_analysis_{range_min}_{range_max}um_2percent.png")

def create_summary_report(results):
    """Crée un rapport de synthèse."""
    
    print(f"\n" + "="*80)
    print("RAPPORT DE SYNTHÈSE - GÉNÉRALISATION LOCALE [0.5-1.0 µm]")
    print("="*80)
    
    print(f"\n🎯 OBJECTIF:")
    print(f"   Évaluer la finesse de généralisation locale du réseau")
    print(f"   dans une plage réaliste d'usage (0.5-1.0 µm)")
    print(f"   Seuil d'acceptabilité: {results['threshold']}%")
    
    print(f"\n📊 RÉSULTATS QUANTITATIFS:")
    print(f"   • Points analysés:     {results['n_points']}")
    print(f"   • Erreur maximale:     {results['max_error']:.3f}%")
    print(f"   • Erreur moyenne:      {results['mean_error']:.3f}%")
    print(f"   • Erreur médiane:      {results['median_error']:.3f}%")
    print(f"   • Erreur minimale:     {results['min_error']:.3f}%")
    
    print(f"\n🎯 ÉVALUATION DU SEUIL {results['threshold']}%:")
    print(f"   • Points < {results['threshold']}%:         {results['points_under_threshold']}/{results['n_points']}")
    print(f"   • Pourcentage < {results['threshold']}%:    {results['percentage_under_threshold']:.1f}%")
    
    print(f"\n✅ CONFIRMATION CLAIRE:")
    print(f"   {results['confirmation']}")
    
    if results['all_under_threshold']:
        print(f"\n🎉 CONCLUSION:")
        print(f"   ✅ GÉNÉRALISATION LOCALE EXCEPTIONNELLE")
        print(f"   ✅ Toutes les erreurs restent inférieures à {results['threshold']}%")
        print(f"   ✅ Le réseau démontre une finesse remarquable")
        print(f"   ✅ Prêt pour utilisation en conditions réelles")
    elif results['percentage_under_threshold'] >= 95:
        print(f"\n👍 CONCLUSION:")
        print(f"   ✅ GÉNÉRALISATION LOCALE TRÈS BONNE")
        print(f"   ✅ >95% des erreurs < {results['threshold']}% (quasi-parfait)")
        print(f"   ⚠️  Quelques points isolés dépassent le seuil")
        print(f"   ✅ Acceptable pour utilisation pratique")
    else:
        print(f"\n⚠️  CONCLUSION:")
        print(f"   ⚠️  GÉNÉRALISATION LOCALE À AMÉLIORER")
        print(f"   ❌ Trop de points dépassent le seuil de {results['threshold']}%")
        print(f"   📈 Recommandation: Optimisation nécessaire")

def main():
    """Fonction principale d'analyse."""
    
    print("="*80)
    print("ANALYSE DES ERREURS RELATIVES - RÉSULTATS EXISTANTS")
    print("="*80)
    print("Utilisation des prédictions du test de robustesse au bruit (2%)")
    print("Calcul des erreurs relatives dans la plage 0.5-1.0 µm")
    print("="*80)
    
    try:
        # Charger les prédictions existantes
        y_true, y_pred = load_existing_predictions()
        
        if y_true is None:
            print("❌ Impossible de charger les données existantes")
            return
        
        # Analyser la généralisation locale dans la plage 0.5-1.0 µm avec seuil 10%
        results = analyze_local_generalization(y_true, y_pred, range_min=0.5, range_max=1.0, threshold=10.0)
        
        if results is None:
            print("❌ Aucun point dans la plage spécifiée")
            return
        
        # Créer le rapport de synthèse
        create_summary_report(results)
        
        # Sauvegarder les résultats
        with open('../results/local_generalization_analysis_2percent.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n📋 Résultats sauvegardés: ../results/local_generalization_analysis_2percent.json")
        print(f"📊 Graphiques générés: ../plots/local_generalization_analysis_0.5_1.0um_2percent.png")
        
    except Exception as e:
        print(f"❌ Erreur durant l'analyse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
