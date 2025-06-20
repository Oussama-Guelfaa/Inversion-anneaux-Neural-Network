#!/usr/bin/env python3
"""
Script Principal - Prédiction Dual Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Script principal pour entraîner et évaluer le réseau de neurones
de prédiction conjointe gap + L_ecran.
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import yaml
import json
import time
from datetime import datetime

# Ajouter le dossier src au path
sys.path.append('src')

from data_loader import DualDataLoader
from trainer import DualParameterTrainer
from dual_parameter_model import DualParameterMetrics, DualParameterVisualizer

def load_config(config_path="../config/dual_prediction_config.yaml"):
    """Charge la configuration."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_results(results, filename="../results/training_results.json"):
    """Sauvegarde les résultats en JSON."""
    Path(filename).parent.mkdir(exist_ok=True)
    
    # Convertir les arrays numpy en listes pour JSON
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, dict):
            json_results[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    json_results[key][k] = v.tolist()
                else:
                    json_results[key][k] = v
        else:
            json_results[key] = value
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"💾 Résultats sauvegardés: {filename}")

def evaluate_model_on_test(model, test_loader, data_loader, metrics_calculator, device):
    """
    Évalue le modèle sur le set de test.
    
    Args:
        model: Modèle entraîné
        test_loader: DataLoader de test
        data_loader: DualDataLoader pour inverse transform
        metrics_calculator: Calculateur de métriques
        device: Device PyTorch
    
    Returns:
        dict: Résultats d'évaluation
    """
    print(f"\n🧪 ÉVALUATION SUR LE SET DE TEST")
    print("="*40)
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Combiner toutes les prédictions
    predictions_scaled = np.vstack(all_predictions)
    targets_scaled = np.vstack(all_targets)
    
    # Inverse transform pour obtenir les valeurs originales
    predictions_original = data_loader.inverse_transform_predictions(predictions_scaled)
    targets_original = data_loader.inverse_transform_predictions(targets_scaled)
    
    # Calculer les métriques sur les données originales
    test_metrics = metrics_calculator.calculate_metrics(predictions_original, targets_original)
    
    # Afficher les résultats
    metrics_calculator.print_metrics(test_metrics, "RÉSULTATS FINAUX SUR TEST SET")

    # Créer le tableau détaillé des résultats
    detailed_results_df = data_loader.create_detailed_results_dataframe(
        predictions_original, targets_original, "test"
    )

    return {
        'predictions_original': predictions_original,
        'targets_original': targets_original,
        'predictions_scaled': predictions_scaled,
        'targets_scaled': targets_scaled,
        'metrics': test_metrics,
        'detailed_results': detailed_results_df
    }

def create_comprehensive_visualizations(history, test_results, visualizer):
    """
    Crée toutes les visualisations.
    
    Args:
        history: Historique d'entraînement
        test_results: Résultats de test
        visualizer: Visualiseur
    """
    print(f"\n🎨 CRÉATION DES VISUALISATIONS")
    print("="*30)
    
    # Courbes d'entraînement
    visualizer.plot_training_curves(history, "training_curves.png")
    print(f"✅ Courbes d'entraînement sauvegardées")
    
    # Scatter plots des prédictions
    visualizer.plot_predictions_scatter(
        test_results['predictions_original'], 
        test_results['targets_original'],
        "test_predictions_scatter.png"
    )
    print(f"✅ Scatter plots sauvegardés")

def run_complete_training():
    """
    Exécute l'entraînement complet du modèle dual.
    """
    print("🚀 ENTRAÎNEMENT COMPLET - PRÉDICTION DUAL GAP + L_ECRAN")
    print("="*70)
    print(f"Auteur: Oussama GUELFAA")
    print(f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    print("="*70)
    
    start_time = time.time()
    
    # 1. Charger la configuration
    print(f"\n📋 Chargement de la configuration...")
    config = load_config()
    print(f"✅ Configuration chargée: {config['experiment']['name']}")
    
    # 2. Préparer les données
    print(f"\n📊 Préparation des données...")
    data_loader = DualDataLoader()
    pipeline_result = data_loader.get_complete_pipeline(config)
    
    train_loader, val_loader, test_loader = pipeline_result['loaders']
    print(f"✅ Données préparées et chargées")
    
    # 3. Configurer et entraîner le modèle
    print(f"\n🏗️ Configuration du modèle...")
    trainer = DualParameterTrainer()
    trainer.setup_model_and_training()
    
    print(f"\n🚀 Entraînement du modèle...")
    history, final_epoch, training_time = trainer.train_model(train_loader, val_loader)
    
    # 4. Évaluation finale sur le test set
    test_results = evaluate_model_on_test(
        trainer.model, test_loader, data_loader, 
        trainer.metrics_calculator, trainer.device
    )
    
    # 5. Créer les visualisations
    create_comprehensive_visualizations(history, test_results, trainer.visualizer)
    
    # 6. Sauvegarder le modèle
    model_path = "../models/dual_parameter_model.pth"
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'config': config,
        'history': history,
        'test_metrics': test_results['metrics'],
        'training_info': {
            'final_epoch': final_epoch,
            'training_time': training_time,
            'total_parameters': sum(p.numel() for p in trainer.model.parameters())
        }
    }, model_path)
    print(f"💾 Modèle sauvegardé: {model_path}")
    
    # 7. Affichage détaillé des résultats
    print(f"\n📊 AFFICHAGE DÉTAILLÉ DES RÉSULTATS")
    print("="*50)

    # Afficher quelques échantillons de test avec comparaison
    data_loader.display_test_samples_comparison(n_samples=15)

    # 8. Sauvegarder les résultats complets
    complete_results = {
        'experiment_info': {
            'name': config['experiment']['name'],
            'date': datetime.now().isoformat(),
            'training_time': training_time,
            'final_epoch': final_epoch,
            'augmentation_methods': ['spline', 'rbf', 'polynomial', 'adaptive', 'synthetic_noise']
        },
        'test_metrics': test_results['metrics'],
        'detailed_results_summary': {
            'total_samples': len(test_results['detailed_results']),
            'gap_accuracy_007um': test_results['detailed_results']['GAP_success'].mean(),
            'lecran_accuracy_01um': test_results['detailed_results']['LECRAN_success'].mean(),
            'both_success_rate': test_results['detailed_results']['BOTH_success'].mean()
        },
        'history': history,
        'config': config
    }

    save_results(complete_results, "../results/complete_results.json")
    
    # 8. Résumé final
    total_time = time.time() - start_time
    
    print(f"\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
    print("="*50)
    print(f"⏱️  Temps total: {total_time:.1f}s")
    print(f"🔄 Epochs: {final_epoch}")
    print(f"📊 Résultats finaux:")
    
    metrics = test_results['metrics']
    print(f"   🎯 Gap R²: {metrics['gap_r2']:.4f}")
    print(f"   🎯 L_ecran R²: {metrics['L_ecran_r2']:.4f}")
    print(f"   🎯 Combined R²: {metrics['combined_r2']:.4f}")
    print(f"   ✅ Gap Accuracy: {metrics['gap_accuracy']:.1%}")
    print(f"   ✅ L_ecran Accuracy: {metrics['L_ecran_accuracy']:.1%}")
    
    # Vérification des objectifs
    gap_success = metrics['gap_accuracy'] >= 0.90
    L_ecran_success = metrics['L_ecran_accuracy'] >= 0.90
    r2_success = metrics['combined_r2'] >= 0.80
    
    print(f"\n🎯 OBJECTIFS ATTEINTS:")
    print(f"   Gap Accuracy > 90%: {'✅ OUI' if gap_success else '❌ NON'}")
    print(f"   L_ecran Accuracy > 90%: {'✅ OUI' if L_ecran_success else '❌ NON'}")
    print(f"   Combined R² > 80%: {'✅ OUI' if r2_success else '❌ NON'}")
    
    if gap_success and L_ecran_success and r2_success:
        print(f"\n🏆 TOUS LES OBJECTIFS ATTEINTS ! MODÈLE RÉUSSI !")
    else:
        print(f"\n⚠️  Certains objectifs non atteints. Optimisation nécessaire.")
    
    print("="*50)
    
    return complete_results

def quick_test():
    """
    Test rapide pour vérifier que tout fonctionne.
    """
    print("🧪 TEST RAPIDE DU SYSTÈME")
    print("="*30)
    
    try:
        # Test configuration
        config = load_config()
        print(f"✅ Configuration OK")
        
        # Test data loader
        data_loader = DualDataLoader()
        print(f"✅ DataLoader OK")
        
        # Test trainer
        trainer = DualParameterTrainer()
        trainer.setup_model_and_training()
        print(f"✅ Trainer OK")
        
        print(f"\n🎉 Tous les composants fonctionnent ! Prêt pour l'entraînement.")
        
    except Exception as e:
        print(f"❌ Erreur dans le test: {e}")
        return False
    
    return True

def main():
    """
    Fonction principale.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraînement Réseau Dual Gap + L_ecran")
    parser.add_argument("--test", action="store_true", help="Exécuter seulement les tests")
    parser.add_argument("--config", default="../config/dual_prediction_config.yaml", 
                       help="Chemin vers le fichier de configuration")
    
    args = parser.parse_args()
    
    if args.test:
        success = quick_test()
        if success:
            print(f"\n🚀 Pour lancer l'entraînement complet:")
            print(f"   python run.py")
    else:
        # Vérifier d'abord que tout fonctionne
        if quick_test():
            print(f"\n🚀 Lancement de l'entraînement complet...")
            time.sleep(2)  # Pause pour lire les tests
            results = run_complete_training()
        else:
            print(f"❌ Tests échoués. Vérifiez la configuration.")

if __name__ == "__main__":
    main()
