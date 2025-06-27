#!/usr/bin/env python3
"""
Test rapide du script d'entraînement Train.py

Auteur: Oussama GUELFAA
Date: 25/06/2025

Ce script teste le fonctionnement du script Train.py avec un petit échantillon.
"""

import subprocess
import sys
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_training_script():
    """
    Test du script d'entraînement avec un petit échantillon.
    """
    logger.info("🧪 TEST DU SCRIPT D'ENTRAÎNEMENT")
    logger.info("="*50)
    
    # Vérifier que le dataset existe
    dataset_path = Path("../data_generation/dataset_2D_Train_Augmented")
    if not dataset_path.exists():
        logger.error(f"❌ Dataset non trouvé: {dataset_path}")
        return False
    
    # Commande de test avec paramètres réduits
    cmd = [
        sys.executable, "Train.py",
        "--dataset", str(dataset_path),
        "--epochs", "5",  # Seulement 5 epochs pour le test
        "--batch_size", "16",  # Batch size réduit
        "--learning_rate", "0.001",
        "--model_path", "../models/test_model.pt"
    ]
    
    logger.info(f"🚀 Lancement du test d'entraînement...")
    logger.info(f"   Commande: {' '.join(cmd)}")
    
    try:
        # Exécuter le script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        if result.returncode == 0:
            logger.info("✅ Test d'entraînement réussi !")
            logger.info("📊 Sortie du script:")
            for line in result.stdout.split('\n')[-10:]:  # Dernières 10 lignes
                if line.strip():
                    logger.info(f"   {line}")
            return True
        else:
            logger.error("❌ Test d'entraînement échoué !")
            logger.error("📊 Erreur:")
            for line in result.stderr.split('\n')[-10:]:  # Dernières 10 lignes d'erreur
                if line.strip():
                    logger.error(f"   {line}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Test d'entraînement timeout (> 5 minutes)")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du test: {e}")
        return False

def test_prediction():
    """
    Test de prédiction avec le modèle entraîné.
    """
    logger.info("\n🔮 TEST DE PRÉDICTION")
    logger.info("="*30)
    
    # Vérifier que le modèle existe
    model_path = Path("../models/test_model.pt")
    if not model_path.exists():
        logger.warning("⚠️ Modèle de test non trouvé, skip du test de prédiction")
        return False
    
    # Profil d'intensité de test (600 valeurs aléatoires)
    import numpy as np
    test_ratio = np.random.rand(600) * 0.5 + 0.5  # Valeurs entre 0.5 et 1.0
    ratio_str = ','.join([f"{x:.6f}" for x in test_ratio])
    
    # Commande de prédiction
    cmd = [
        sys.executable, "Train.py",
        "--predict_only",
        "--ratio", ratio_str,
        "--model_path", str(model_path)
    ]
    
    logger.info(f"🚀 Test de prédiction...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 1 minute max
        )
        
        if result.returncode == 0:
            logger.info("✅ Test de prédiction réussi !")
            logger.info("📊 Résultat:")
            for line in result.stdout.split('\n')[-5:]:  # Dernières 5 lignes
                if line.strip():
                    logger.info(f"   {line}")
            return True
        else:
            logger.error("❌ Test de prédiction échoué !")
            logger.error("📊 Erreur:")
            for line in result.stderr.split('\n')[-5:]:
                if line.strip():
                    logger.error(f"   {line}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Test de prédiction timeout")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du test de prédiction: {e}")
        return False

def main():
    """
    Fonction principale de test.
    """
    logger.info("🧪 TESTS DU SCRIPT TRAIN.PY")
    logger.info("="*60)
    
    # Test 1: Entraînement
    training_success = test_training_script()
    
    # Test 2: Prédiction (seulement si l'entraînement a réussi)
    prediction_success = False
    if training_success:
        prediction_success = test_prediction()
    
    # Résumé
    logger.info(f"\n📊 RÉSUMÉ DES TESTS:")
    logger.info(f"   Entraînement: {'✅ RÉUSSI' if training_success else '❌ ÉCHOUÉ'}")
    logger.info(f"   Prédiction: {'✅ RÉUSSI' if prediction_success else '❌ ÉCHOUÉ'}")
    
    if training_success and prediction_success:
        logger.info(f"\n🎉 TOUS LES TESTS SONT RÉUSSIS !")
        logger.info(f"   Le script Train.py est prêt pour l'entraînement complet.")
    elif training_success:
        logger.info(f"\n⚠️ ENTRAÎNEMENT OK, PRÉDICTION À VÉRIFIER")
    else:
        logger.info(f"\n❌ PROBLÈMES DÉTECTÉS - VÉRIFIER LE CODE")
    
    return training_success and prediction_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
