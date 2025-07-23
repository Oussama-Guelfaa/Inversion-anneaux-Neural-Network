#!/usr/bin/env python3
"""
CLI Principal - Inversion d'Anneaux Holographiques

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Point d'entrée principal du CLI moderne et interactif pour
l'inversion d'anneaux holographiques.

Usage:
    python hologram_cli.py --help
    python hologram_cli.py --interactive
    python hologram_cli.py train --model precision
    python hologram_cli.py predict --input data.mat
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire racine au path Python
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Point d'entrée principal du CLI."""
    try:
        from cli.main import main_cli
        main_cli()
    except ImportError as e:
        print(f"❌ Erreur d'importation: {e}")
        print("🔧 Vérifiez que toutes les dépendances sont installées:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Au revoir !")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
