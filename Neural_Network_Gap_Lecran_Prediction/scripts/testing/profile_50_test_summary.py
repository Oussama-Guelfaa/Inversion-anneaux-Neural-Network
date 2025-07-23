#!/usr/bin/env python3
"""
Résumé du test du profil 50 avec le modèle ULTRA_DEEP
Auteur: Oussama GUELFAA
Date: 18/07/2025
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def create_test_summary():
    """Crée un résumé visuel du test."""
    
    print("📊 RÉSUMÉ DU TEST - PROFIL 50 ULTRA_DEEP")
    print("=" * 50)
    
    # Charger les résultats
    with open("../../results/predictions/profile_50_ultra_deep_prediction.json", 'r') as f:
        results = json.load(f)
    
    gap_pred = results['gap_predicted_um']
    L_ecran_pred = results['L_ecran_predicted_um']
    
    print(f"🎯 RÉSULTATS DE PRÉDICTION:")
    print(f"   Gap prédit: {gap_pred:.6f} µm")
    print(f"   L'écran prédit: {L_ecran_pred:.3f} µm")
    print(f"   Modèle: {results['model_used']}")
    
    # Créer une figure de résumé
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('RÉSUMÉ TEST PROFIL 50 - MODÈLE ULTRA_DEEP', fontsize=14, fontweight='bold')
    
    # Subplot 1: Résultats de prédiction
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    summary_text = f"""
PRÉDICTIONS DU MODÈLE:

Gap prédit: {gap_pred:.6f} µm
L'écran prédit: {L_ecran_pred:.3f} µm

MODÈLE UTILISÉ:
{results['model_used']}

PREPROCESSING:
• Points finaux: {results['preprocessing']['final_points']}
• Plage radiale: {results['preprocessing']['r_min']:.3f} - {results['preprocessing']['r_max']:.3f} µm
• Résolution: {results['preprocessing']['delta_r']:.6f} µm/point

ARCHITECTURE:
• Type: MLP Ultra-Profond avec attention
• Couches: 601 → 1024 → ... → 64 → 2
• Blocs résiduels: 8 blocs
• Mécanisme d'attention: MultiheadAttention
• Paramètres: ~2.5M
"""
    
    ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Subplot 2: Comparaison avec plages typiques
    ax2 = axes[0, 1]
    
    # Plages typiques des paramètres
    gap_range = [0.005, 0.700]  # µm
    L_ecran_range = [8.0, 12.0]  # µm
    
    # Graphique gap
    ax2.barh(['Gap prédit', 'Plage simulation'], 
             [abs(gap_pred), gap_range[1] - gap_range[0]], 
             color=['red' if gap_pred < 0 else 'green', 'lightblue'])
    
    ax2.set_xlabel('Gap (µm)')
    ax2.set_title('Comparaison Gap')
    ax2.grid(True, alpha=0.3)
    
    # Ajouter annotation pour gap négatif
    if gap_pred < 0:
        ax2.text(0.1, 0, f'⚠️ Gap négatif!\n{gap_pred:.6f} µm', 
                fontsize=10, color='red', fontweight='bold')
    
    # Subplot 3: L'écran dans la plage
    ax3 = axes[1, 0]
    
    x_pos = [0, 1]
    heights = [L_ecran_pred, (L_ecran_range[1] + L_ecran_range[0]) / 2]
    colors = ['green' if L_ecran_range[0] <= L_ecran_pred <= L_ecran_range[1] else 'orange', 'lightblue']
    
    bars = ax3.bar(x_pos, heights, color=colors, alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['L\'écran prédit', 'L\'écran moyen\n(simulation)'])
    ax3.set_ylabel('L\'écran (µm)')
    ax3.set_title('Comparaison L\'écran')
    ax3.grid(True, alpha=0.3)
    
    # Ajouter les plages de simulation
    ax3.axhline(y=L_ecran_range[0], color='blue', linestyle='--', alpha=0.7, label='Min simulation')
    ax3.axhline(y=L_ecran_range[1], color='blue', linestyle='--', alpha=0.7, label='Max simulation')
    ax3.legend()
    
    # Subplot 4: Évaluation qualitative
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Évaluation des résultats
    gap_status = "❌ PROBLÉMATIQUE" if gap_pred < 0 else "✅ ACCEPTABLE" if 0.005 <= gap_pred <= 0.700 else "⚠️ HORS PLAGE"
    L_ecran_status = "✅ DANS LA PLAGE" if L_ecran_range[0] <= L_ecran_pred <= L_ecran_range[1] else "⚠️ HORS PLAGE"
    
    evaluation_text = f"""
ÉVALUATION DES RÉSULTATS:

Gap: {gap_status}
• Valeur: {gap_pred:.6f} µm
• Plage attendue: 0.005 - 0.700 µm
• Commentaire: {'Gap négatif - physiquement impossible' if gap_pred < 0 else 'Dans la plage normale' if 0.005 <= gap_pred <= 0.700 else 'Hors plage de simulation'}

L'écran: {L_ecran_status}
• Valeur: {L_ecran_pred:.3f} µm
• Plage attendue: 8.0 - 12.0 µm
• Commentaire: {'Dans la plage de simulation' if L_ecran_range[0] <= L_ecran_pred <= L_ecran_range[1] else 'Hors plage de simulation'}

CONCLUSION GÉNÉRALE:
{'⚠️ Résultats mitigés - gap négatif problématique' if gap_pred < 0 else '✅ Résultats cohérents' if L_ecran_range[0] <= L_ecran_pred <= L_ecran_range[1] and 0.005 <= gap_pred <= 0.700 else '⚠️ Vérifier la cohérence des données'}
"""
    
    ax4.text(0.05, 0.95, evaluation_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    
    # Sauvegarder
    output_file = "../../visualizations/plots/profile_50_test_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📈 Résumé sauvegardé: {output_file}")
    
    plt.show()
    
    # Analyse des résultats
    print(f"\n📋 ANALYSE DES RÉSULTATS:")
    print(f"   Gap: {gap_status}")
    print(f"   L'écran: {L_ecran_status}")
    
    if gap_pred < 0:
        print(f"\n⚠️  ATTENTION: Gap négatif détecté!")
        print(f"   • Cela peut indiquer:")
        print(f"     - Problème de normalisation des données")
        print(f"     - Données expérimentales hors distribution d'entraînement")
        print(f"     - Besoin d'adaptation de domaine")
        print(f"     - Contraintes physiques non appliquées au modèle")
    
    print(f"\n💡 RECOMMANDATIONS:")
    if gap_pred < 0:
        print(f"   1. Vérifier la cohérence des données expérimentales")
        print(f"   2. Implémenter des contraintes de non-négativité")
        print(f"   3. Considérer l'adaptation de domaine")
        print(f"   4. Tester sur d'autres profils expérimentaux")
    else:
        print(f"   1. Valider avec d'autres profils expérimentaux")
        print(f"   2. Comparer avec des mesures indépendantes si disponibles")
        print(f"   3. Évaluer la robustesse sur l'ensemble du dataset")

def main():
    """Fonction principale."""
    create_test_summary()

if __name__ == "__main__":
    main()
