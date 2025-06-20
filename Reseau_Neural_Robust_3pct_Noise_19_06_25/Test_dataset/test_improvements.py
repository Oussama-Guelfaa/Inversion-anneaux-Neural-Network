#!/usr/bin/env python3
"""
Test des Améliorations pour Précision Gap 0.007µm

Auteur: Oussama GUELFAA
Date: 14 - 01 - 2025

Ce script teste les améliorations apportées au réseau de neurones
pour atteindre une précision de 0.007µm sur le paramètre gap.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.dual_parameter_model import DualParameterPredictor, DualParameterLoss, DualParameterMetrics
import time

def test_improved_architecture():
    """
    Test de l'architecture améliorée.
    """
    print("🧪 TEST DE L'ARCHITECTURE AMÉLIORÉE")
    print("="*50)
    
    # Créer le modèle amélioré
    model = DualParameterPredictor(input_size=600, dropout_rate=0.15)
    
    # Test avec données factices
    batch_size = 24  # Nouveau batch size
    input_size = 600
    
    # Données d'entrée factices
    x = torch.randn(batch_size, input_size)
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        predictions = model(x)
    forward_time = time.time() - start_time
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {predictions.shape}")
    print(f"✅ Forward pass time: {forward_time*1000:.2f}ms")
    
    # Vérifier que la sortie a la bonne forme
    assert predictions.shape == (batch_size, 2), f"Erreur: forme attendue ({batch_size}, 2), obtenue {predictions.shape}"
    
    # Afficher le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Paramètres totaux: {total_params:,}")
    print(f"📊 Paramètres entraînables: {trainable_params:,}")
    print(f"📊 Augmentation vs modèle original: {(total_params - 482242) / 482242 * 100:.1f}%")
    
    return model

def test_improved_loss_function():
    """
    Test de la fonction de perte améliorée.
    """
    print(f"\n🧪 TEST DE LA FONCTION DE PERTE AMÉLIORÉE")
    print("="*50)
    
    # Créer la loss function améliorée
    loss_fn = DualParameterLoss(gap_weight=3.0, L_ecran_weight=1.0, precision_mode=True)
    
    # Données factices
    batch_size = 24
    predictions = torch.randn(batch_size, 2)
    targets = torch.randn(batch_size, 2)
    
    # Calculer la loss
    total_loss, gap_loss, L_ecran_loss = loss_fn(predictions, targets)
    
    print(f"✅ Total Loss: {total_loss.item():.6f}")
    print(f"✅ Gap Loss: {gap_loss.item():.6f}")
    print(f"✅ L_ecran Loss: {L_ecran_loss.item():.6f}")
    print(f"✅ Ratio Gap/L_ecran: {gap_loss.item() / L_ecran_loss.item():.2f}")
    
    # Test avec erreurs gap importantes
    print(f"\n🎯 Test avec erreurs gap importantes...")
    predictions_bad = targets.clone()
    predictions_bad[:, 0] += 0.01  # Erreur gap de 0.01µm
    
    total_loss_bad, gap_loss_bad, L_ecran_loss_bad = loss_fn(predictions_bad, targets)
    
    print(f"✅ Total Loss (erreur gap): {total_loss_bad.item():.6f}")
    print(f"✅ Gap Loss (erreur gap): {gap_loss_bad.item():.6f}")
    print(f"✅ Pénalité précision activée: {gap_loss_bad.item() > gap_loss.item()}")
    
    return loss_fn

def test_improved_metrics():
    """
    Test des métriques avec nouvelle tolérance 0.007µm.
    """
    print(f"\n🧪 TEST DES MÉTRIQUES HAUTE PRÉCISION")
    print("="*50)
    
    # Créer les métriques avec nouvelle tolérance
    metrics_calc = DualParameterMetrics(gap_tolerance=0.007, L_ecran_tolerance=0.1)
    
    # Données factices avec différents niveaux de précision
    n_samples = 1000
    
    # Cas 1: Précision excellente
    targets = np.random.rand(n_samples, 2)
    targets[:, 0] *= 0.1  # Gap entre 0 et 0.1µm
    targets[:, 1] *= 100  # L_ecran entre 0 et 100µm
    
    # Prédictions avec erreur contrôlée
    predictions = targets.copy()
    predictions[:, 0] += np.random.normal(0, 0.003, n_samples)  # Erreur gap ±3nm
    predictions[:, 1] += np.random.normal(0, 0.05, n_samples)   # Erreur L_ecran ±50nm
    
    metrics = metrics_calc.calculate_metrics(predictions, targets)
    
    print(f"✅ Test avec erreur gap ±3nm:")
    print(f"   Gap Accuracy (0.007µm): {metrics['gap_accuracy']:.1%}")
    print(f"   Gap MAE: {metrics['gap_mae']:.4f}µm")
    print(f"   Gap R²: {metrics['gap_r2']:.4f}")
    
    # Cas 2: Précision limite
    predictions_limit = targets.copy()
    predictions_limit[:, 0] += np.random.normal(0, 0.006, n_samples)  # Erreur gap ±6nm
    predictions_limit[:, 1] += np.random.normal(0, 0.05, n_samples)
    
    metrics_limit = metrics_calc.calculate_metrics(predictions_limit, targets)
    
    print(f"\n✅ Test avec erreur gap ±6nm:")
    print(f"   Gap Accuracy (0.007µm): {metrics_limit['gap_accuracy']:.1%}")
    print(f"   Gap MAE: {metrics_limit['gap_mae']:.4f}µm")
    print(f"   Gap R²: {metrics_limit['gap_r2']:.4f}")
    
    # Cas 3: Précision insuffisante
    predictions_bad = targets.copy()
    predictions_bad[:, 0] += np.random.normal(0, 0.01, n_samples)  # Erreur gap ±10nm
    predictions_bad[:, 1] += np.random.normal(0, 0.05, n_samples)
    
    metrics_bad = metrics_calc.calculate_metrics(predictions_bad, targets)
    
    print(f"\n✅ Test avec erreur gap ±10nm:")
    print(f"   Gap Accuracy (0.007µm): {metrics_bad['gap_accuracy']:.1%}")
    print(f"   Gap MAE: {metrics_bad['gap_mae']:.4f}µm")
    print(f"   Gap R²: {metrics_bad['gap_r2']:.4f}")
    
    return metrics_calc

def test_data_loading():
    """
    Test du chargement des données existantes.
    """
    print(f"\n🧪 TEST DU CHARGEMENT DES DONNÉES")
    print("="*50)
    
    try:
        # Charger le dataset augmenté existant
        data = np.load('data/augmented_dataset.npz')
        X = data['X']
        y = data['y']
        
        print(f"✅ Dataset chargé avec succès:")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Gap range: {np.min(y[:, 0]):.3f} - {np.max(y[:, 0]):.3f} µm")
        print(f"   L_ecran range: {np.min(y[:, 1]):.1f} - {np.max(y[:, 1]):.1f} µm")
        
        # Analyser la distribution des gaps
        gap_std = np.std(y[:, 0])
        gap_mean = np.mean(y[:, 0])
        
        print(f"   Gap mean: {gap_mean:.3f}µm")
        print(f"   Gap std: {gap_std:.3f}µm")
        
        # Identifier les échantillons dans la zone critique (±0.007µm autour de la moyenne)
        critical_mask = np.abs(y[:, 0] - gap_mean) <= 0.007
        critical_samples = np.sum(critical_mask)
        
        print(f"   Échantillons dans zone critique ±0.007µm: {critical_samples} ({critical_samples/len(y)*100:.1f}%)")
        
        return X, y
        
    except Exception as e:
        print(f"❌ Erreur chargement dataset: {e}")
        return None, None

def main():
    """
    Fonction principale de test.
    """
    print("🚀 TEST COMPLET DES AMÉLIORATIONS - PRÉCISION GAP 0.007µm")
    print("="*70)
    print(f"Auteur: Oussama GUELFAA")
    print(f"Date: 14 - 01 - 2025")
    print("="*70)
    
    # Test 1: Architecture améliorée
    model = test_improved_architecture()
    
    # Test 2: Fonction de perte améliorée
    loss_fn = test_improved_loss_function()
    
    # Test 3: Métriques haute précision
    metrics_calc = test_improved_metrics()
    
    # Test 4: Chargement des données
    X, y = test_data_loading()
    
    # Résumé des améliorations
    print(f"\n🎯 RÉSUMÉ DES AMÉLIORATIONS")
    print("="*40)
    print(f"✅ Architecture: 6 couches (vs 4), 1024→512→256→128→64→32→2")
    print(f"✅ Paramètres: ~1.2M (vs 482K), +150% capacité")
    print(f"✅ Loss function: Pondérée 3:1 + pénalité précision")
    print(f"✅ Tolérance gap: 0.007µm (vs 0.01µm), -30% tolérance")
    print(f"✅ Dropout adaptatif: 0.15→0.05 par couche")
    print(f"✅ Batch size: 24 (vs 32) pour stabilité")
    print(f"✅ Learning rate: 0.0008 (vs 0.001) pour précision")
    
    if X is not None and y is not None:
        print(f"✅ Dataset: {len(X)} échantillons prêts pour entraînement")
        
        # Test rapide avec données réelles
        print(f"\n🧪 Test rapide avec données réelles...")
        
        # Sélectionner un petit échantillon
        indices = np.random.choice(len(X), 100, replace=False)
        X_sample = torch.FloatTensor(X[indices])
        y_sample = y[indices]
        
        # Forward pass
        with torch.no_grad():
            predictions = model(X_sample).numpy()
        
        # Calculer métriques
        metrics = metrics_calc.calculate_metrics(predictions, y_sample)
        
        print(f"   Gap R² (modèle non-entraîné): {metrics['gap_r2']:.4f}")
        print(f"   Gap MAE (modèle non-entraîné): {metrics['gap_mae']:.4f}µm")
        print(f"   Gap Accuracy 0.007µm (modèle non-entraîné): {metrics['gap_accuracy']:.1%}")
    
    print(f"\n🏆 PRÊT POUR ENTRAÎNEMENT HAUTE PRÉCISION !")
    print(f"   Objectif: Gap Accuracy > 85% avec tolérance 0.007µm")
    print(f"   Commande: python run.py")

if __name__ == "__main__":
    main()
