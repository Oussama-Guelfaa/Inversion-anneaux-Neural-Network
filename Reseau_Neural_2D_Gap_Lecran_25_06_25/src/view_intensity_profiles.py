#!/usr/bin/env python3
"""
Visualisation des profils d'intensité générés

Auteur: Oussama GUELFAA
Date: 25/06/2025

Script simple pour ouvrir le graphique des profils d'intensité.
"""

import webbrowser
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def open_intensity_profiles():
    """Ouvre le graphique des profils d'intensité dans le navigateur."""
    
    # Chemin vers le fichier graphique
    plot_file = Path("../plots/intensity_profiles_600pts.png").resolve()
    
    if plot_file.exists():
        logger.info(f"📊 Ouverture du graphique des profils d'intensité")
        logger.info(f"   Fichier: {plot_file}")
        
        # Ouvrir dans le navigateur
        webbrowser.open(f"file://{plot_file}")
        
        logger.info(f"✅ Graphique ouvert dans le navigateur")
        
        # Afficher les informations sur les profils
        logger.info(f"\n📋 PROFILS TRACÉS:")
        logger.info(f"   1. Gap 0.0050µm, L_écran 4.0µm (Petit gap, petit L_écran)")
        logger.info(f"   2. Gap 0.0050µm, L_écran 7.0µm (Petit gap, grand L_écran)")
        logger.info(f"   3. Gap 0.3000µm, L_écran 4.0µm (Grand gap, petit L_écran)")
        logger.info(f"   4. Gap 0.3000µm, L_écran 7.0µm (Grand gap, grand L_écran)")
        
        logger.info(f"\n📊 CARACTÉRISTIQUES:")
        logger.info(f"   - Tous tronqués à 600 points")
        logger.info(f"   - Format utilisé par le réseau de neurones")
        logger.info(f"   - Profils représentatifs des extrêmes du dataset")
        logger.info(f"   - Statistiques affichées sur chaque graphique")
        
    else:
        logger.error(f"❌ Fichier graphique non trouvé: {plot_file}")
        logger.info(f"💡 Exécutez d'abord: python plot_intensity_profiles.py")

def main():
    """Fonction principale."""
    logger.info("🚀 VISUALISATION DES PROFILS D'INTENSITÉ")
    logger.info("="*50)
    
    open_intensity_profiles()

if __name__ == "__main__":
    main()
