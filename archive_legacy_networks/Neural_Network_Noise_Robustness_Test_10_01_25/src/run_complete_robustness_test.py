#!/usr/bin/env python3
"""
Script principal pour le test complet de robustesse au bruit

Ce script orchestre tous les tests de robustesse au bruit:
1. Test de base sans augmentation
2. Test avec augmentation de données
3. Comparaison et analyse complète

Auteur: Oussama GUELFAA
Date: 10 - 01 - 2025
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

def run_basic_robustness_test():
    """Exécute le test de robustesse de base."""
    
    print("="*60)
    print("ÉTAPE 1: TEST DE ROBUSTESSE DE BASE")
    print("="*60)
    
    try:
        from noise_robustness_test import main as run_noise_test
        
        # Simuler l'exécution du test principal
        print("Lancement du test de robustesse au bruit...")
        
        # Importer et exécuter le code principal
        exec(open('noise_robustness_test.py').read())
        
        print("✅ Test de robustesse de base terminé")
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans le test de base: {e}")
        return False

def run_augmentation_test():
    """Exécute le test avec augmentation de données."""
    
    print("\n" + "="*60)
    print("ÉTAPE 2: TEST AVEC AUGMENTATION DE DONNÉES")
    print("="*60)
    
    try:
        from test_augmentation_robustness import main as run_aug_test
        
        print("Lancement du test avec augmentation...")
        
        # Importer et exécuter le test d'augmentation
        exec(open('test_augmentation_robustness.py').read())
        
        print("✅ Test avec augmentation terminé")
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans le test d'augmentation: {e}")
        return False

def generate_comprehensive_report():
    """Génère un rapport complet de tous les tests."""
    
    print("\n" + "="*60)
    print("ÉTAPE 3: GÉNÉRATION DU RAPPORT COMPLET")
    print("="*60)
    
    try:
        # Charger les résultats des différents tests
        results = {}
        
        # Résultats du test de base
        if os.path.exists('../results/noise_robustness_summary.json'):
            with open('../results/noise_robustness_summary.json', 'r') as f:
                results['basic_robustness'] = json.load(f)
        
        # Résultats du test d'augmentation
        if os.path.exists('../results/augmentation_comparison.json'):
            with open('../results/augmentation_comparison.json', 'r') as f:
                results['augmentation_comparison'] = json.load(f)
        
        # Générer le rapport consolidé
        report = generate_consolidated_report(results)
        
        # Sauvegarder le rapport
        with open('../results/comprehensive_robustness_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Générer le rapport markdown
        generate_markdown_report(report)
        
        print("✅ Rapport complet généré")
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans la génération du rapport: {e}")
        return False

def generate_consolidated_report(results):
    """Génère un rapport consolidé de tous les tests."""
    
    report = {
        'test_summary': {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_type': 'comprehensive_noise_robustness',
            'tests_completed': list(results.keys())
        },
        'key_findings': {},
        'recommendations': {},
        'detailed_results': results
    }
    
    # Analyser les résultats du test de base
    if 'basic_robustness' in results:
        basic_results = results['basic_robustness']
        
        # Trouver le seuil de tolérance
        threshold_80 = None
        for noise_level, data in basic_results.get('results_by_noise', {}).items():
            if data['r2'] >= 0.8:
                threshold_80 = int(noise_level)
            else:
                break
        
        report['key_findings']['noise_tolerance_threshold'] = threshold_80
        report['key_findings']['max_tested_noise'] = max([int(k) for k in basic_results.get('results_by_noise', {}).keys()])
        
        # Performance de référence (0% bruit)
        if '0' in basic_results.get('results_by_noise', {}):
            ref_performance = basic_results['results_by_noise']['0']
            report['key_findings']['reference_performance'] = {
                'r2': ref_performance['r2'],
                'rmse': ref_performance['rmse']
            }
    
    # Analyser les bénéfices de l'augmentation
    if 'augmentation_comparison' in results:
        aug_results = results['augmentation_comparison']
        
        if 'comparison' in aug_results:
            improvements = []
            for noise_level, data in aug_results['comparison'].items():
                improvements.append(data['r2_improvement_percent'])
            
            avg_improvement = sum(improvements) / len(improvements) if improvements else 0
            report['key_findings']['augmentation_benefit'] = {
                'average_r2_improvement_percent': avg_improvement,
                'beneficial': avg_improvement > 1.0  # Seuil de 1% d'amélioration
            }
    
    # Générer les recommandations
    report['recommendations'] = generate_recommendations(report['key_findings'])
    
    return report

def generate_recommendations(findings):
    """Génère des recommandations basées sur les résultats."""
    
    recommendations = {
        'acquisition_specifications': {},
        'model_deployment': {},
        'future_improvements': {}
    }
    
    # Recommandations d'acquisition
    threshold = findings.get('noise_tolerance_threshold')
    if threshold:
        if threshold >= 5:
            recommendations['acquisition_specifications'] = {
                'snr_requirement': f"SNR > {100/threshold:.1f}",
                'noise_level_max': f"{threshold}%",
                'acquisition_quality': "Standard - Modèle robuste"
            }
        elif threshold >= 2:
            recommendations['acquisition_specifications'] = {
                'snr_requirement': f"SNR > {100/threshold:.1f}",
                'noise_level_max': f"{threshold}%",
                'acquisition_quality': "Élevée - Attention aux conditions"
            }
        else:
            recommendations['acquisition_specifications'] = {
                'snr_requirement': "SNR > 50",
                'noise_level_max': "< 2%",
                'acquisition_quality': "Très élevée - Conditions strictes"
            }
    
    # Recommandations de déploiement
    ref_perf = findings.get('reference_performance', {})
    if ref_perf.get('r2', 0) > 0.95:
        recommendations['model_deployment'] = {
            'readiness': "Prêt pour déploiement",
            'confidence_level': "Élevée",
            'monitoring': "Standard"
        }
    else:
        recommendations['model_deployment'] = {
            'readiness': "Nécessite optimisation",
            'confidence_level': "Modérée",
            'monitoring': "Renforcé"
        }
    
    # Recommandations d'amélioration
    aug_benefit = findings.get('augmentation_benefit', {})
    if aug_benefit.get('beneficial', False):
        recommendations['future_improvements'] = {
            'data_augmentation': "Recommandée - Bénéfice démontré",
            'priority': "Haute",
            'next_steps': ["Optimiser paramètres d'augmentation", "Tester autres techniques"]
        }
    else:
        recommendations['future_improvements'] = {
            'data_augmentation': "Peu bénéfique",
            'priority': "Basse",
            'next_steps': ["Explorer autres architectures", "Améliorer qualité des données"]
        }
    
    return recommendations

def generate_markdown_report(report):
    """Génère un rapport au format Markdown."""
    
    markdown_content = f"""# Rapport Complet - Test de Robustesse au Bruit

**Date:** {report['test_summary']['test_date']}  
**Auteur:** Oussama GUELFAA  
**Type de test:** {report['test_summary']['test_type']}

## 🎯 Résumé Exécutif

### Tests Réalisés
{', '.join(report['test_summary']['tests_completed'])}

### Résultats Clés

#### Seuil de Tolérance au Bruit
- **Niveau maximum toléré (R² > 0.8):** {report['key_findings'].get('noise_tolerance_threshold', 'Non déterminé')}%
- **Niveau maximum testé:** {report['key_findings'].get('max_tested_noise', 'N/A')}%

#### Performance de Référence (0% bruit)
- **R² Score:** {report['key_findings'].get('reference_performance', {}).get('r2', 'N/A'):.4f}
- **RMSE:** {report['key_findings'].get('reference_performance', {}).get('rmse', 'N/A'):.4f} µm

#### Bénéfice de l'Augmentation de Données
- **Amélioration moyenne R²:** {report['key_findings'].get('augmentation_benefit', {}).get('average_r2_improvement_percent', 'N/A'):.1f}%
- **Recommandée:** {'Oui' if report['key_findings'].get('augmentation_benefit', {}).get('beneficial', False) else 'Non'}

## 📋 Recommandations

### Spécifications d'Acquisition
- **SNR requis:** {report['recommendations']['acquisition_specifications'].get('snr_requirement', 'N/A')}
- **Niveau de bruit max:** {report['recommendations']['acquisition_specifications'].get('noise_level_max', 'N/A')}
- **Qualité d'acquisition:** {report['recommendations']['acquisition_specifications'].get('acquisition_quality', 'N/A')}

### Déploiement du Modèle
- **État de préparation:** {report['recommendations']['model_deployment'].get('readiness', 'N/A')}
- **Niveau de confiance:** {report['recommendations']['model_deployment'].get('confidence_level', 'N/A')}
- **Monitoring:** {report['recommendations']['model_deployment'].get('monitoring', 'N/A')}

### Améliorations Futures
- **Augmentation de données:** {report['recommendations']['future_improvements'].get('data_augmentation', 'N/A')}
- **Priorité:** {report['recommendations']['future_improvements'].get('priority', 'N/A')}

## 📊 Fichiers Générés

### Résultats Numériques
- `noise_robustness_summary.json` - Résultats du test de base
- `augmentation_comparison.json` - Comparaison avec/sans augmentation
- `comprehensive_robustness_report.json` - Rapport consolidé

### Visualisations
- `noise_robustness_analysis.png` - Analyse de robustesse
- `predictions_by_noise.png` - Prédictions par niveau de bruit
- `augmentation_comparison.png` - Comparaison augmentation

### Documentation
- `README.md` - Guide du projet
- `comprehensive_robustness_report.md` - Ce rapport

---

**Note:** Ce rapport synthétise tous les tests de robustesse au bruit effectués sur le modèle de prédiction du gap.
"""
    
    with open('../results/comprehensive_robustness_report.md', 'w') as f:
        f.write(markdown_content)

def main():
    """Fonction principale pour exécuter tous les tests."""
    
    print("🚀 LANCEMENT DU TEST COMPLET DE ROBUSTESSE AU BRUIT")
    print("="*60)
    
    start_time = time.time()
    
    # Créer les dossiers nécessaires
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../plots", exist_ok=True)
    os.makedirs("../results", exist_ok=True)
    
    success_count = 0
    total_tests = 3
    
    # Étape 1: Test de robustesse de base
    if run_basic_robustness_test():
        success_count += 1
    
    # Étape 2: Test avec augmentation
    if run_augmentation_test():
        success_count += 1
    
    # Étape 3: Rapport complet
    if generate_comprehensive_report():
        success_count += 1
    
    # Résumé final
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("🏁 RÉSUMÉ FINAL DU TEST COMPLET")
    print("="*60)
    
    print(f"✅ Tests réussis: {success_count}/{total_tests}")
    print(f"⏱️  Temps total: {total_time/60:.1f} minutes")
    
    if success_count == total_tests:
        print("🎉 TOUS LES TESTS TERMINÉS AVEC SUCCÈS!")
        print("\n📁 Fichiers générés:")
        print("   📊 ../results/ - Résultats numériques")
        print("   📈 ../plots/ - Graphiques et visualisations")
        print("   🤖 ../models/ - Modèles entraînés")
        print("   📋 ../results/comprehensive_robustness_report.md - Rapport final")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les logs ci-dessus.")
    
    return success_count == total_tests

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test complet de robustesse au bruit')
    parser.add_argument('--quick', action='store_true', 
                       help='Mode rapide avec moins de niveaux de bruit')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Ignorer les tests d\'augmentation')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🏃 Mode rapide activé")
    
    if args.no_augmentation:
        print("⏭️  Tests d'augmentation ignorés")
    
    success = main()
    sys.exit(0 if success else 1)
