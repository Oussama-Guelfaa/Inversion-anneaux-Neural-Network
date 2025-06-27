#!/usr/bin/env python3
"""
Test du Data Loader pour Réseau Neural 2D Gap + L_écran

Auteur: Oussama GUELFAA
Date: 25/06/2025

Script de test pour valider le fonctionnement du data loader.
"""

import yaml
import sys
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_loader import DataLoader2D
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """
    Charge la configuration depuis le fichier YAML.
    """
    config_path = project_root / "config" / "model_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_data_loader():
    """
    Test du data loader avec le dataset augmenté.
    """
    logger.info("🧪 TEST DU DATA LOADER")
    logger.info("="*40)
    
    # 1. Charger la configuration
    config = load_config()
    logger.info("✅ Configuration chargée")
    
    # 2. Créer le data loader
    data_loader = DataLoader2D(config)
    logger.info("✅ DataLoader2D créé")
    
    # 3. Obtenir les statistiques des données
    logger.info("\n📊 Statistiques des données:")
    stats = data_loader.get_data_statistics()
    for key, value in stats.items():
        logger.info(f"   {key}: {value}")
    
    # 4. Préparer les données
    logger.info("\n🔄 Préparation des données...")
    train_loader, val_loader, test_loader, scalers = data_loader.prepare_data(
        use_augmented=True,
        validation_split=0.2
    )
    
    # 5. Tester les data loaders
    logger.info("\n🧪 Test des DataLoaders:")
    
    # Test train loader
    train_batch = next(iter(train_loader))
    X_train_batch, y_train_batch = train_batch
    logger.info(f"   Train batch: X{X_train_batch.shape}, y{y_train_batch.shape}")
    
    # Test validation loader
    val_batch = next(iter(val_loader))
    X_val_batch, y_val_batch = val_batch
    logger.info(f"   Validation batch: X{X_val_batch.shape}, y{y_val_batch.shape}")
    
    # Test test loader
    test_batch = next(iter(test_loader))
    X_test_batch, y_test_batch = test_batch
    logger.info(f"   Test batch: X{X_test_batch.shape}, y{y_test_batch.shape}")
    
    # 6. Vérifier les plages de données normalisées
    logger.info("\n📈 Plages des données normalisées:")
    logger.info(f"   Train X: [{X_train_batch.min():.3f}, {X_train_batch.max():.3f}]")
    logger.info(f"   Train y: [{y_train_batch.min():.3f}, {y_train_batch.max():.3f}]")
    
    # 7. Tester la dénormalisation
    logger.info("\n🔄 Test de dénormalisation:")
    y_denorm = scalers['output_scaler'].inverse_transform(y_train_batch.numpy())
    logger.info(f"   Gap dénormalisé: [{y_denorm[:, 0].min():.4f}, {y_denorm[:, 0].max():.4f}] µm")
    logger.info(f"   L_ecran dénormalisé: [{y_denorm[:, 1].min():.1f}, {y_denorm[:, 1].max():.1f}] µm")
    
    logger.info("\n✅ Test du DataLoader terminé avec succès !")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scalers': scalers,
        'config': config
    }

if __name__ == "__main__":
    test_results = test_data_loader()
