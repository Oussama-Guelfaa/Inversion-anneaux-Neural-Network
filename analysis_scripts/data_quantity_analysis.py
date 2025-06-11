#!/usr/bin/env python3
"""
Analyse de l'impact de la quantité de données sur les performances

Ce script compare les performances avec différentes quantités de données
pour établir la relation entre taille du dataset et capacité de généralisation.

Auteur: Oussama GUELFAA
Date: 11 - 01 - 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def load_experiment_results():
    """Charge les résultats des différentes expériences."""
    
    print("=== CHARGEMENT DES RÉSULTATS D'EXPÉRIENCES ===")
    
    results = {}
    
    # Résultats de l'expérience originale (240 train, 80 test)
    original_path = "../results/noise_robustness_summary.json"
    if os.path.exists(original_path):
        with open(original_path, 'r') as f:
            original_data = json.load(f)
            if '5' in original_data.get('results_by_noise', {}):
                results['original'] = {
                    'n_train': 240,  # 60% de 400
                    'n_test': 80,    # 20% de 400
                    'r2': original_data['results_by_noise']['5']['r2'],
                    'rmse': original_data['results_by_noise']['5']['rmse'],
                    'mae': original_data['results_by_noise']['5']['mae']
                }
                print(f"✅ Expérience originale chargée: R² = {results['original']['r2']:.4f}")
    
    # Résultats de l'expérience avec données réduites (300 train, 100 test)
    reduced_path = "../results/reduced_data_experiment.json"
    if os.path.exists(reduced_path):
        with open(reduced_path, 'r') as f:
            reduced_data = json.load(f)
            results['reduced'] = {
                'n_train': reduced_data['experiment_config']['n_train'],
                'n_test': reduced_data['experiment_config']['n_test'],
                'r2': reduced_data['performance']['r2'],
                'rmse': reduced_data['performance']['rmse'],
                'mae': reduced_data['performance']['mae']
            }
            print(f"✅ Expérience données réduites chargée: R² = {results['reduced']['r2']:.4f}")
    
    return results

def create_comprehensive_analysis(results):
    """Crée une analyse complète de l'impact de la quantité de données."""
    
    print(f"\n=== ANALYSE COMPARATIVE ===")
    
    if len(results) < 2:
        print("❌ Pas assez de résultats pour la comparaison")
        return
    
    # Créer le graphique comparatif
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extraire les données pour les graphiques
    experiments = list(results.keys())
    n_trains = [results[exp]['n_train'] for exp in experiments]
    r2_scores = [results[exp]['r2'] for exp in experiments]
    rmse_values = [results[exp]['rmse'] for exp in experiments]
    mae_values = [results[exp]['mae'] for exp in experiments]
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # 1. R² vs Nombre d'échantillons d'entraînement
    axes[0, 0].scatter(n_trains, r2_scores, s=100, c=colors[:len(experiments)], alpha=0.7)
    for i, exp in enumerate(experiments):
        axes[0, 0].annotate(f'{exp}\n({n_trains[i]} échantillons)', 
                           (n_trains[i], r2_scores[i]), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
    
    axes[0, 0].set_xlabel('Nombre d\'échantillons d\'entraînement')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Performance vs Quantité de Données d\'Entraînement')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Seuil acceptable (0.8)')
    axes[0, 0].legend()
    
    # 2. RMSE vs Nombre d'échantillons
    axes[0, 1].scatter(n_trains, rmse_values, s=100, c=colors[:len(experiments)], alpha=0.7)
    for i, exp in enumerate(experiments):
        axes[0, 1].annotate(f'{exp}\n({rmse_values[i]:.3f} µm)', 
                           (n_trains[i], rmse_values[i]), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
    
    axes[0, 1].set_xlabel('Nombre d\'échantillons d\'entraînement')
    axes[0, 1].set_ylabel('RMSE (µm)')
    axes[0, 1].set_title('Erreur RMSE vs Quantité de Données')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Comparaison des métriques
    metrics = ['R²', 'RMSE (µm)', 'MAE (µm)']
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    if 'original' in results and 'reduced' in results:
        original_values = [results['original']['r2'], results['original']['rmse'], results['original']['mae']]
        reduced_values = [results['reduced']['r2'], results['reduced']['rmse'], results['reduced']['mae']]
        
        axes[1, 0].bar(x_pos - width/2, original_values, width, label=f'Original ({results["original"]["n_train"]} train)', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, reduced_values, width, label=f'Réduit ({results["reduced"]["n_train"]} train)', alpha=0.7)
        
        axes[1, 0].set_xlabel('Métriques')
        axes[1, 0].set_ylabel('Valeur')
        axes[1, 0].set_title('Comparaison des Métriques de Performance')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for i, (orig, red) in enumerate(zip(original_values, reduced_values)):
            axes[1, 0].text(i - width/2, orig + orig*0.01, f'{orig:.3f}', ha='center', va='bottom', fontweight='bold')
            axes[1, 0].text(i + width/2, red + red*0.01, f'{red:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Efficacité des données (Performance par échantillon)
    efficiency = [r2 / n_train for r2, n_train in zip(r2_scores, n_trains)]
    
    axes[1, 1].bar(experiments, efficiency, color=colors[:len(experiments)], alpha=0.7)
    axes[1, 1].set_xlabel('Expérience')
    axes[1, 1].set_ylabel('R² / Nombre d\'échantillons')
    axes[1, 1].set_title('Efficacité des Données\n(Performance par échantillon)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for i, (exp, eff) in enumerate(zip(experiments, efficiency)):
        axes[1, 1].text(i, eff + eff*0.01, f'{eff:.6f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../plots/data_quantity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Analyse comparative sauvegardée: ../plots/data_quantity_analysis.png")

def generate_insights(results):
    """Génère des insights sur l'impact de la quantité de données."""
    
    print(f"\n" + "="*60)
    print("INSIGHTS SUR L'IMPACT DE LA QUANTITÉ DE DONNÉES")
    print("="*60)
    
    if 'original' in results and 'reduced' in results:
        orig = results['original']
        red = results['reduced']
        
        print(f"\n📊 COMPARAISON DÉTAILLÉE:")
        print(f"{'Métrique':<15} {'Original':<15} {'Réduit':<15} {'Différence':<15}")
        print("-" * 65)
        print(f"{'N° train':<15} {orig['n_train']:<15} {red['n_train']:<15} {red['n_train'] - orig['n_train']:+<15}")
        print(f"{'R² Score':<15} {orig['r2']:<15.4f} {red['r2']:<15.4f} {red['r2'] - orig['r2']:+<15.4f}")
        print(f"{'RMSE (µm)':<15} {orig['rmse']:<15.4f} {red['rmse']:<15.4f} {red['rmse'] - orig['rmse']:+<15.4f}")
        print(f"{'MAE (µm)':<15} {orig['mae']:<15.4f} {red['mae']:<15.4f} {red['mae'] - orig['mae']:+<15.4f}")
        
        # Calcul des améliorations relatives
        r2_improvement = ((red['r2'] - orig['r2']) / orig['r2']) * 100
        rmse_improvement = ((orig['rmse'] - red['rmse']) / orig['rmse']) * 100
        data_increase = ((red['n_train'] - orig['n_train']) / orig['n_train']) * 100
        
        print(f"\n📈 AMÉLIORATIONS RELATIVES:")
        print(f"  Augmentation données: +{data_increase:.1f}%")
        print(f"  Amélioration R²: {r2_improvement:+.2f}%")
        print(f"  Amélioration RMSE: {rmse_improvement:+.2f}%")
        
        print(f"\n🎯 CONCLUSIONS:")
        
        if abs(r2_improvement) < 1:
            print(f"  ✅ RÉSULTAT SURPRENANT: Performance quasi-identique")
            print(f"     • +{data_increase:.0f}% de données → {r2_improvement:+.1f}% de performance")
            print(f"     • Le modèle semble déjà bien optimisé avec 240 échantillons")
            print(f"     • Rendements décroissants observés")
        elif r2_improvement > 2:
            print(f"  📈 AMÉLIORATION SIGNIFICATIVE avec plus de données")
            print(f"     • +{data_increase:.0f}% de données → +{r2_improvement:.1f}% de performance")
            print(f"     • Bénéfice clair de l'augmentation des données")
        else:
            print(f"  ⚖️  AMÉLIORATION MODÉRÉE")
            print(f"     • +{data_increase:.0f}% de données → +{r2_improvement:.1f}% de performance")
        
        # Efficacité des données
        orig_efficiency = orig['r2'] / orig['n_train']
        red_efficiency = red['r2'] / red['n_train']
        
        print(f"\n⚡ EFFICACITÉ DES DONNÉES:")
        print(f"  Original: {orig_efficiency:.6f} R²/échantillon")
        print(f"  Réduit: {red_efficiency:.6f} R²/échantillon")
        
        if red_efficiency > orig_efficiency:
            print(f"  ✅ Plus de données = Meilleure efficacité")
        else:
            print(f"  ⚠️  Rendements décroissants observés")
    
    print(f"\n💡 RECOMMANDATIONS GÉNÉRALES:")
    
    # Analyser les performances absolues
    best_r2 = max([results[exp]['r2'] for exp in results])
    
    if best_r2 > 0.99:
        print(f"  🎉 PERFORMANCE EXCEPTIONNELLE (R² > 0.99)")
        print(f"     • Le modèle atteint déjà une précision quasi-parfaite")
        print(f"     • Augmenter les données n'apportera que des gains marginaux")
        print(f"     • Focus recommandé: Validation sur données réelles")
    elif best_r2 > 0.95:
        print(f"  ✅ PERFORMANCE TRÈS BONNE (R² > 0.95)")
        print(f"     • Qualité suffisante pour la plupart des applications")
        print(f"     • Gains potentiels limités avec plus de données")
    elif best_r2 > 0.8:
        print(f"  👍 PERFORMANCE ACCEPTABLE (R² > 0.8)")
        print(f"     • Plus de données pourraient améliorer significativement")
        print(f"     • Recommandation: Tester avec 500+ échantillons")
    else:
        print(f"  ❌ PERFORMANCE INSUFFISANTE")
        print(f"     • Augmentation substantielle des données nécessaire")
        print(f"     • Ou révision de l'architecture du modèle")
    
    print(f"\n🔬 POUR CETTE TÂCHE SPÉCIFIQUE:")
    print(f"  • 240-300 échantillons semblent suffisants")
    print(f"  • Performance plateau atteint autour de R² = 0.99")
    print(f"  • Priorité: Validation sur données expérimentales réelles")

def main():
    """Fonction principale d'analyse."""
    
    print("="*60)
    print("ANALYSE DE L'IMPACT DE LA QUANTITÉ DE DONNÉES")
    print("="*60)
    
    try:
        # Charger les résultats des expériences
        results = load_experiment_results()
        
        if len(results) == 0:
            print("❌ Aucun résultat d'expérience trouvé")
            return
        
        # Créer l'analyse comparative
        create_comprehensive_analysis(results)
        
        # Générer les insights
        generate_insights(results)
        
        print(f"\n{'='*60}")
        print("ANALYSE TERMINÉE")
        print(f"{'='*60}")
        print(f"📊 Graphiques générés: ../plots/data_quantity_analysis.png")
        
    except Exception as e:
        print(f"❌ Erreur durant l'analyse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
