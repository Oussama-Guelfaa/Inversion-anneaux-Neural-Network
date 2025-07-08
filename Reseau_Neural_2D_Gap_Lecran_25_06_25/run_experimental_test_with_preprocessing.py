#!/usr/bin/env python3
"""
Script de lancement pour test expérimental avec prétraitement
Auteur: Oussama GUELFAA
Date: Juillet 2025

Lance le test des profils expérimentaux avec prétraitement sophistiqué
pour harmoniser les données avec les caractéristiques des simulations.
"""

import sys
import logging
from pathlib import Path

# Ajouter le chemin vers src
sys.path.append(str(Path(__file__).parent / "src"))

from test_experimental_interpolated_profiles import main

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experimental_test_with_preprocessing.log')
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚀 LANCEMENT DU TEST EXPÉRIMENTAL AVEC PRÉTRAITEMENT")
    logger.info("="*60)
    logger.info("📋 Fonctionnalités activées:")
    logger.info("   ✅ Prétraitement sophistiqué des données expérimentales")
    logger.info("   ✅ Harmonisation avec caractéristiques des simulations")
    logger.info("   ✅ Validation du prétraitement")
    logger.info("   ✅ Graphiques de comparaison avant/après")
    logger.info("   ✅ Métriques détaillées de performance")
    logger.info("="*60)
    
    try:
        main()
        logger.info("✅ Test terminé avec succès!")
    except Exception as e:
        logger.error(f"❌ Erreur lors du test: {e}")
        raise
