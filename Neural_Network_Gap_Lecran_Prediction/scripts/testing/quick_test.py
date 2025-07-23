#!/usr/bin/env python3
"""
Test rapide de l'architecture complète
Auteur: Oussama GUELFAA
Date: 15/07/2025

Ce script effectue un test rapide de tous les composants développés
pour valider le bon fonctionnement de l'architecture avant l'entraînement complet.
"""

import torch
import numpy as np
import os
import sys

# Ajouter le répertoire courant au path pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loader():
    """Test du chargeur de données"""
    print("🔍 Test du chargeur de données...")
    
    try:
        from data_loader import AdvancedDataLoader
        
        loader = AdvancedDataLoader(
            train_dir="Train",
            preprocessed_data_path="preprocessed_data.npz"
        )
        
        # Test avec un très petit échantillon
        X_data, y_data, filenames = loader.load_all_training_data(sample_ratio=0.001)
        
        print(f"   ✅ Données chargées: {X_data.shape[0]} échantillons")
        print(f"   📊 Forme X: {X_data.shape}")
        print(f"   📊 Forme y: {y_data.shape}")
        
        # Test de division
        datasets = loader.create_train_val_test_split()
        print(f"   ✅ Division réussie: Train {datasets['X_train'].shape[0]}, Val {datasets['X_val'].shape[0]}, Test {datasets['X_test'].shape[0]}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_data_augmentation():
    """Test de l'augmentation de données"""
    print("\n🔍 Test de l'augmentation de données...")
    
    try:
        from data_augmentation import AdvancedDataAugmentation
        
        augmenter = AdvancedDataAugmentation(augmentation_factor=2, noise_level=0.01)
        
        # Données de test
        n_points = 601
        x_radial = np.linspace(1.384585, 5.538338, n_points)
        intensity_test = np.exp(-0.5 * ((x_radial - 3.0) / 0.5)**2)
        gap_test = 0.15
        L_ecran_test = 10.0
        
        # Test d'augmentation
        augmented_profiles, augmented_gaps, augmented_L_ecrans = augmenter.interpolate_profile_2d(
            x_radial, intensity_test, gap_test, L_ecran_test, n_variations=3
        )
        
        print(f"   ✅ Augmentation réussie: {len(augmented_profiles)} profils générés")
        print(f"   📊 Plage Gap: [{min(augmented_gaps):.4f}, {max(augmented_gaps):.4f}] µm")
        print(f"   📊 Plage L_écran: [{min(augmented_L_ecrans):.3f}, {max(augmented_L_ecrans):.3f}] µm")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_neural_network():
    """Test de l'architecture de réseau de neurones"""
    print("\n🔍 Test de l'architecture de réseau de neurones...")
    
    try:
        from advanced_neural_network import AdvancedHybridNetwork, create_model_variants
        
        # Test des variantes
        models = create_model_variants()
        
        for name, model in models.items():
            print(f"   🧠 Test du modèle '{name}':")
            
            # Test forward pass
            batch_size = 4
            input_size = 601
            x_test = torch.randn(batch_size, input_size)
            
            with torch.no_grad():
                output = model(x_test)
                
            print(f"      📊 Entrée: {x_test.shape}")
            print(f"      📊 Sortie: {output.shape}")
            
            # Compter les paramètres
            total_params = sum(p.numel() for p in model.parameters())
            print(f"      🔧 Paramètres: {total_params:,}")
        
        print("   ✅ Toutes les architectures fonctionnent correctement")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_training_components():
    """Test des composants d'entraînement"""
    print("\n🔍 Test des composants d'entraînement...")
    
    try:
        from advanced_training import (WeightedMSELoss, AdvancedMetrics, 
                                     AdvancedOptimizer, TrainingConfig, create_loss_function)
        from advanced_neural_network import AdvancedHybridNetwork
        
        # Test de la configuration
        config = TrainingConfig()
        print(f"   ✅ Configuration créée: {len(config.to_dict())} paramètres")
        
        # Test de la loss function
        loss_fn = create_loss_function('weighted_mse', gap_weight=3.0, L_ecran_weight=1.0)
        
        # Données de test
        batch_size = 4
        predictions = torch.randn(batch_size, 2)
        targets = torch.randn(batch_size, 2)
        
        total_loss, gap_loss, L_ecran_loss = loss_fn(predictions, targets)
        print(f"   ✅ Loss function: Total {total_loss:.4f}, Gap {gap_loss:.4f}, L_écran {L_ecran_loss:.4f}")
        
        # Test des métriques
        pred_np = predictions.detach().numpy()
        target_np = targets.detach().numpy()
        
        metrics = AdvancedMetrics.calculate_metrics(pred_np, target_np)
        print(f"   ✅ Métriques calculées: {len(metrics)} métriques")
        
        # Test de l'optimiseur
        model = AdvancedHybridNetwork(base_channels=32, num_encoder_blocks=2)
        optimizer = AdvancedOptimizer.create_optimizer(model, 'adamw', lr=1e-3)
        scheduler = AdvancedOptimizer.create_scheduler(optimizer, 'cosine', epochs=10)
        
        print(f"   ✅ Optimiseur et scheduler créés")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_visualization():
    """Test du système de visualisation"""
    print("\n🔍 Test du système de visualisation...")
    
    try:
        from visualization_monitoring import AdvancedVisualizer
        
        visualizer = AdvancedVisualizer(save_dir="test_plots", experiment_name="quick_test")
        
        # Simuler quelques métriques
        for epoch in range(5):
            train_metrics = {
                'loss': 1.0 - 0.1 * epoch,
                'gap_r2': 0.5 + 0.1 * epoch,
                'L_ecran_r2': 0.6 + 0.08 * epoch,
                'gap_tolerance_0.007um': 50 + 10 * epoch
            }
            
            val_metrics = train_metrics.copy()
            val_metrics['loss'] += 0.1
            
            visualizer.update_training_metrics(epoch, train_metrics, val_metrics, 1e-3)
        
        print(f"   ✅ Visualiseur initialisé et métriques mises à jour")
        
        # Test de génération de graphiques (sans sauvegarde)
        # visualizer.plot_training_curves(save=False, show=False)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def test_integration():
    """Test d'intégration rapide"""
    print("\n🔍 Test d'intégration rapide...")
    
    try:
        from main_training import UltraSophisticatedTrainer
        from advanced_training import TrainingConfig
        
        # Configuration minimale pour test
        config = TrainingConfig()
        config.epochs = 2
        config.batch_size = 4
        
        trainer = UltraSophisticatedTrainer(config, "integration_test")
        
        print(f"   ✅ Entraîneur ultra-sophistiqué créé")
        
        # Note: On ne lance pas l'entraînement complet ici pour éviter les erreurs
        # si les données ne sont pas disponibles
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧠 Test Rapide de l'Architecture Complète")
    print("=" * 60)
    
    tests = [
        ("Chargeur de données", test_data_loader),
        ("Augmentation de données", test_data_augmentation),
        ("Architecture de réseau", test_neural_network),
        ("Composants d'entraînement", test_training_components),
        ("Système de visualisation", test_visualization),
        ("Intégration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Erreur critique dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    print("\n📊 Résumé des Tests:")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat global: {passed}/{total} tests réussis ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Tous les tests sont passés! L'architecture est prête pour l'entraînement.")
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
