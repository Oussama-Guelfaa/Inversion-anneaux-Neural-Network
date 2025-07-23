#!/usr/bin/env python3
"""
Finalisation de la réorganisation
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script termine la réorganisation en déplaçant les fichiers restants.
"""

import os
import shutil
from pathlib import Path
import glob

def move_remaining_files():
    """Déplace les fichiers restants dans les bons dossiers"""
    
    print("🔄 FINALISATION DE LA RÉORGANISATION")
    print("=" * 50)
    
    # Mappings pour les fichiers restants
    remaining_mappings = {
        # Visualisations
        "visualizations/plots": [
            "*.png",
        ],
        
        # Rapports
        "reports/technical": [
            "*.md",
            "*report*.txt",
        ],
        
        # Résultats
        "results/predictions": [
            "*.csv",
            "*.json",
        ],
        
        # Données expérimentales
        "data/experimental": [
            "*ps3um*",
            "*experimental*",
            "quick_data_inspection.py",
        ],
        
        # Scripts d'analyse restants
        "scripts/analysis": [
            "analyze_*.py",
            "compare_*.py",
            "residual_error_analysis.py",
        ],
        
        # Utilitaires
        "utils/helpers": [
            "reorganize_directory.py",
            "finish_reorganization.py",
        ],
    }
    
    moved_count = 0
    
    for target_dir, patterns in remaining_mappings.items():
        target_path = Path(target_dir)
        
        print(f"\n📂 Vers {target_dir}/:")
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for file_path in files:
                if Path(file_path).is_file():
                    # Éviter de déplacer les fichiers déjà dans les bons dossiers
                    if not str(file_path).startswith(target_dir):
                        dest_file = target_path / Path(file_path).name
                        if not dest_file.exists():
                            try:
                                shutil.move(file_path, str(dest_file))
                                print(f"   ✅ {file_path} → {target_dir}/")
                                moved_count += 1
                            except Exception as e:
                                print(f"   ❌ Erreur avec {file_path}: {e}")
                        else:
                            print(f"   ⚠️  {Path(file_path).name} déjà existant dans {target_dir}/")
    
    return moved_count

def move_analysis_train_data():
    """Déplace le dossier Analysis_Train_data"""
    
    print(f"\n📂 Déplacement du dossier Analysis_Train_data:")
    
    source = Path("Analysis_Train_data")
    target = Path("scripts/analysis/Analysis_Train_data")
    
    if source.exists() and source.is_dir():
        if not target.exists():
            try:
                shutil.move(str(source), str(target))
                print(f"   ✅ Analysis_Train_data/ → scripts/analysis/")
                return 1
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
        else:
            print(f"   ⚠️  Dossier déjà existant dans scripts/analysis/")
    
    return 0

def clean_pycache():
    """Supprime les dossiers __pycache__"""
    
    print(f"\n🧹 Nettoyage des fichiers __pycache__:")
    
    pycache_dirs = glob.glob("**/__pycache__", recursive=True)
    removed_count = 0
    
    for pycache_dir in pycache_dirs:
        try:
            shutil.rmtree(pycache_dir)
            print(f"   ✅ Supprimé: {pycache_dir}")
            removed_count += 1
        except Exception as e:
            print(f"   ❌ Erreur avec {pycache_dir}: {e}")
    
    return removed_count

def create_index_files():
    """Crée des fichiers d'index pour chaque dossier"""
    
    print(f"\n📝 Création des fichiers d'index:")
    
    # Index pour scripts/training
    training_index = """# Scripts d'Entraînement

## Scripts Disponibles

### Entraînement Principal
- `main_training.py` - Script d'entraînement principal
- `advanced_training.py` - Entraînement avancé avec optimisations

### Entraînement Spécialisé
- `cpu_training.py` - Entraînement optimisé CPU
- `fast_training.py` - Entraînement rapide
- `ultra_fast_training.py` - Entraînement ultra-rapide
- `simplified_training.py` - Entraînement simplifié

### Entraînement Avancé
- `semi_supervised_fine_tuning.py` - Fine-tuning semi-supervisé

## Utilisation
```bash
python main_training.py  # Entraînement standard
python cpu_training.py   # Pour CPU uniquement
```
"""
    
    # Index pour scripts/testing
    testing_index = """# Scripts de Test

## Scripts Disponibles

- `test_model_on_simulation_data.py` - Test sur données simulées
- `test_ultra_deep_on_experimental.py` - Test sur données expérimentales
- `test_all_experimental_profiles.py` - Test complet expérimental
- `quick_test.py` - Test rapide
- `demo_preprocessed_data.py` - Démonstration des données

## Utilisation
```bash
python test_model_on_simulation_data.py  # Test standard
python quick_test.py                     # Test rapide
```
"""
    
    # Index pour data
    data_index = """# Données

## Structure

### raw/
- `Train/` - Données d'entraînement simulées (22,540 profils)
- `Test/` - Données de test

### processed/
- Échantillons préprocessés
- Profils moyens
- Données normalisées

### experimental/
- Données expérimentales PS 3µm (6,596 profils)
- Analyses et rapports

## Utilisation

1. **Entraînement complet** : Utiliser `raw/Train/`
2. **Tests rapides** : Utiliser `processed/`
3. **Validation expérimentale** : Utiliser `experimental/`
"""
    
    indexes = {
        "scripts/training/INDEX.md": training_index,
        "scripts/testing/INDEX.md": testing_index,
        "data/INDEX.md": data_index,
    }
    
    for file_path, content in indexes.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✅ {file_path}")

def show_final_structure():
    """Affiche la structure finale"""
    
    print(f"\n🏗️  STRUCTURE FINALE:")
    print("=" * 50)
    
    structure = """
Neural_Network_Gap_Lecran_Prediction/
├── 📁 data/                     # Données
│   ├── 📂 raw/                  # Données brutes (Train/, Test/)
│   ├── 📂 processed/            # Données préprocessées
│   └── 📂 experimental/         # Données expérimentales PS 3µm
├── 📁 scripts/                  # Scripts principaux
│   ├── 📂 training/             # Entraînement (8 scripts)
│   ├── 📂 analysis/             # Analyse (5+ scripts)
│   ├── 📂 preprocessing/        # Préprocessing (4 scripts)
│   └── 📂 testing/              # Tests (5 scripts)
├── 📁 models/                   # Modèles
│   ├── 📂 architectures/        # Définitions des réseaux
│   ├── 📂 saved_models/         # Modèles entraînés (.pt, .joblib)
│   └── 📂 checkpoints/          # Points de sauvegarde
├── 📁 results/                  # Résultats
│   ├── 📂 training_results/     # Historiques d'entraînement
│   ├── 📂 predictions/          # Prédictions (.csv, .json)
│   └── 📂 evaluations/          # Métriques d'évaluation
├── 📁 visualizations/           # Visualisations
│   ├── 📂 plots/                # Graphiques (.png)
│   ├── 📂 analysis_charts/      # Graphiques d'analyse
│   └── 📂 comparisons/          # Comparaisons visuelles
├── 📁 utils/                    # Utilitaires
│   ├── 📂 data_loaders/         # Chargeurs de données (3 scripts)
│   ├── 📂 monitoring/           # Monitoring (1 script)
│   └── 📂 helpers/              # Fonctions utilitaires
└── 📁 reports/                  # Documentation
    ├── 📂 technical/            # Rapports techniques (.md)
    ├── 📂 summaries/            # Résumés (.txt)
    └── 📂 logs/                 # Logs d'exécution
"""
    
    print(structure)

def main():
    """Fonction principale"""
    
    try:
        # 1. Déplacer les fichiers restants
        moved_count = move_remaining_files()
        
        # 2. Déplacer Analysis_Train_data
        analysis_moved = move_analysis_train_data()
        
        # 3. Nettoyer __pycache__
        cache_removed = clean_pycache()
        
        # 4. Créer les fichiers d'index
        create_index_files()
        
        # 5. Afficher la structure finale
        show_final_structure()
        
        print(f"\n✅ FINALISATION TERMINÉE!")
        print(f"📊 Statistiques:")
        print(f"   • {moved_count} fichiers déplacés")
        print(f"   • {analysis_moved} dossier d'analyse déplacé")
        print(f"   • {cache_removed} dossiers __pycache__ supprimés")
        print(f"   • 3 fichiers d'index créés")
        
        print(f"\n🎯 PROJET MAINTENANT ORGANISÉ!")
        print(f"📁 Structure claire avec 7 dossiers principaux")
        print(f"📝 Documentation mise à jour")
        print(f"🚀 Prêt pour le développement!")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()
