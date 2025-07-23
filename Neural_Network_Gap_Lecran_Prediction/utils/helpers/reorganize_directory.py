#!/usr/bin/env python3
"""
Script de réorganisation du dossier Neural_Network_Gap_Lecran_Prediction
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script réorganise automatiquement tous les fichiers dans une structure claire.
"""

import os
import shutil
from pathlib import Path
import glob

def create_directory_structure():
    """Crée la structure de dossiers organisée"""
    
    base_dir = Path(".")
    
    # Structure de dossiers proposée
    directories = {
        "data": {
            "description": "Données d'entraînement et de test",
            "subdirs": ["raw", "processed", "experimental"]
        },
        "models": {
            "description": "Modèles entraînés et architectures",
            "subdirs": ["saved_models", "architectures", "checkpoints"]
        },
        "scripts": {
            "description": "Scripts d'entraînement et d'analyse",
            "subdirs": ["training", "analysis", "preprocessing", "testing"]
        },
        "results": {
            "description": "Résultats d'entraînement et prédictions",
            "subdirs": ["training_results", "predictions", "evaluations"]
        },
        "visualizations": {
            "description": "Graphiques et visualisations",
            "subdirs": ["plots", "analysis_charts", "comparisons"]
        },
        "reports": {
            "description": "Rapports et documentation",
            "subdirs": ["technical", "summaries", "logs"]
        },
        "utils": {
            "description": "Utilitaires et fonctions communes",
            "subdirs": ["data_loaders", "monitoring", "helpers"]
        }
    }
    
    print("🏗️  CRÉATION DE LA STRUCTURE DE DOSSIERS")
    print("=" * 50)
    
    for main_dir, info in directories.items():
        main_path = base_dir / main_dir
        main_path.mkdir(exist_ok=True)
        print(f"📁 {main_dir}/ - {info['description']}")
        
        for subdir in info['subdirs']:
            sub_path = main_path / subdir
            sub_path.mkdir(exist_ok=True)
            print(f"   📂 {subdir}/")
    
    return directories

def get_file_mappings():
    """Définit où chaque fichier doit être déplacé"""
    
    mappings = {
        # === DONNÉES ===
        "data/raw": [
            "Train/",  # Dossier entier
            "Test/",   # Dossier entier
        ],
        "data/processed": [
            "*samples*.csv",
            "*samples*.mat",
            "*mean_profile*.csv",
            "*interesting_profiles*.csv",
            "extracted_data_full.npz",
        ],
        "data/experimental": [
            "*ps3um*",
            "*experimental*",
        ],
        
        # === MODÈLES ===
        "models/architectures": [
            "advanced_neural_network.py",
        ],
        "models/saved_models": [
            "*.pt",
            "*.pth",
            "*scalers*.joblib",
        ],
        
        # === SCRIPTS D'ENTRAÎNEMENT ===
        "scripts/training": [
            "*training*.py",
            "main_training.py",
            "cpu_training.py",
            "fast_training.py",
            "ultra_fast_training.py",
            "simplified_training.py",
            "semi_supervised_fine_tuning.py",
        ],
        
        # === SCRIPTS D'ANALYSE ===
        "scripts/analysis": [
            "analyze_*.py",
            "*analysis*.py",
            "residual_error_analysis.py",
            "compare_*.py",
        ],
        
        # === SCRIPTS DE PRÉPROCESSING ===
        "scripts/preprocessing": [
            "preprocess_data.py",
            "data_augmentation.py",
            "truncate_and_save.py",
            "extract_*.py",
        ],
        
        # === SCRIPTS DE TEST ===
        "scripts/testing": [
            "test_*.py",
            "quick_test.py",
            "demo_*.py",
        ],
        
        # === UTILITAIRES ===
        "utils/data_loaders": [
            "*data_loader*.py",
            "optimized_data_loader.py",
            "ultra_fast_data_loader.py",
        ],
        "utils/monitoring": [
            "visualization_monitoring.py",
        ],
        
        # === RÉSULTATS ===
        "results/training_results": [
            "results/",  # Dossier entier existant
        ],
        "results/predictions": [
            "*predictions*.csv",
            "*predictions*.json",
            "*test_results*.csv",
            "*test_results*.json",
            "*metrics*.json",
        ],
        
        # === VISUALISATIONS ===
        "visualizations/plots": [
            "*.png",
        ],
        
        # === RAPPORTS ===
        "reports/technical": [
            "*report*.txt",
            "*report*.md",
            "preprocessing_report.md",
        ],
        "reports/summaries": [
            "*summary*.txt",
            "*summary*.md",
        ],
        "reports/logs": [
            "*.log",
        ],
    }
    
    return mappings

def move_files_safely(mappings):
    """Déplace les fichiers selon les mappings définis"""
    
    print("\n📦 DÉPLACEMENT DES FICHIERS")
    print("=" * 50)
    
    moved_count = 0
    
    for target_dir, patterns in mappings.items():
        target_path = Path(target_dir)
        
        print(f"\n📂 Vers {target_dir}/:")
        
        for pattern in patterns:
            # Gérer les dossiers entiers
            if pattern.endswith("/"):
                source_dir = pattern.rstrip("/")
                if Path(source_dir).exists() and Path(source_dir).is_dir():
                    dest_path = target_path / source_dir
                    if not dest_path.exists():
                        shutil.move(source_dir, str(dest_path))
                        print(f"   ✅ {source_dir}/ → {target_dir}/")
                        moved_count += 1
                    else:
                        print(f"   ⚠️  {source_dir}/ déjà existant dans {target_dir}/")
            else:
                # Gérer les fichiers avec patterns
                files = glob.glob(pattern)
                for file_path in files:
                    if Path(file_path).is_file():
                        dest_file = target_path / Path(file_path).name
                        if not dest_file.exists():
                            shutil.move(file_path, str(dest_file))
                            print(f"   ✅ {file_path} → {target_dir}/")
                            moved_count += 1
                        else:
                            print(f"   ⚠️  {Path(file_path).name} déjà existant dans {target_dir}/")
    
    return moved_count

def create_readme_files():
    """Crée des fichiers README pour chaque dossier"""
    
    print("\n📝 CRÉATION DES FICHIERS README")
    print("=" * 50)
    
    readme_contents = {
        "data/README.md": """# Données

## Structure
- `raw/` - Données brutes d'entraînement et de test
- `processed/` - Données préprocessées et échantillons
- `experimental/` - Données expérimentales PS 3µm

## Utilisation
- Utiliser `raw/` pour l'entraînement complet
- Utiliser `processed/` pour les tests rapides
- Utiliser `experimental/` pour la validation expérimentale
""",
        
        "scripts/README.md": """# Scripts

## Structure
- `training/` - Scripts d'entraînement des réseaux de neurones
- `analysis/` - Scripts d'analyse des données et résultats
- `preprocessing/` - Scripts de préprocessing des données
- `testing/` - Scripts de test et validation

## Utilisation
1. Préprocessing → `preprocessing/`
2. Entraînement → `training/`
3. Test → `testing/`
4. Analyse → `analysis/`
""",
        
        "models/README.md": """# Modèles

## Structure
- `architectures/` - Définitions des architectures de réseaux
- `saved_models/` - Modèles entraînés sauvegardés
- `checkpoints/` - Points de sauvegarde pendant l'entraînement

## Utilisation
- Charger les modèles depuis `saved_models/`
- Utiliser les architectures depuis `architectures/`
""",
        
        "results/README.md": """# Résultats

## Structure
- `training_results/` - Résultats d'entraînement détaillés
- `predictions/` - Prédictions sur données de test
- `evaluations/` - Métriques d'évaluation

## Utilisation
- Consulter `training_results/` pour l'historique d'entraînement
- Analyser `predictions/` pour les performances
""",
        
        "visualizations/README.md": """# Visualisations

## Structure
- `plots/` - Graphiques et courbes
- `analysis_charts/` - Graphiques d'analyse
- `comparisons/` - Comparaisons visuelles

## Utilisation
- Tous les graphiques générés sont sauvegardés ici
- Organisés par type d'analyse
""",
        
        "utils/README.md": """# Utilitaires

## Structure
- `data_loaders/` - Chargeurs de données optimisés
- `monitoring/` - Outils de monitoring
- `helpers/` - Fonctions utilitaires

## Utilisation
- Importer les utilitaires depuis ces modules
- Réutiliser les fonctions communes
"""
    }
    
    for file_path, content in readme_contents.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✅ {file_path}")

def create_main_readme():
    """Crée le README principal mis à jour"""
    
    content = """# Neural Network Gap L'écran Prediction

## 🎯 Objectif
Prédiction des paramètres gap et L'écran à partir de profils d'intensité holographique.

## 📁 Structure du Projet

```
Neural_Network_Gap_Lecran_Prediction/
├── data/                    # Données d'entraînement et de test
│   ├── raw/                 # Données brutes (Train/, Test/)
│   ├── processed/           # Données préprocessées
│   └── experimental/        # Données expérimentales PS 3µm
├── scripts/                 # Scripts principaux
│   ├── training/            # Entraînement des modèles
│   ├── analysis/            # Analyse des données
│   ├── preprocessing/       # Préprocessing
│   └── testing/             # Tests et validation
├── models/                  # Modèles et architectures
│   ├── architectures/       # Définitions des réseaux
│   ├── saved_models/        # Modèles entraînés
│   └── checkpoints/         # Points de sauvegarde
├── results/                 # Résultats d'entraînement
│   ├── training_results/    # Historiques d'entraînement
│   ├── predictions/         # Prédictions
│   └── evaluations/         # Métriques
├── visualizations/          # Graphiques et visualisations
│   ├── plots/               # Graphiques généraux
│   ├── analysis_charts/     # Graphiques d'analyse
│   └── comparisons/         # Comparaisons visuelles
├── utils/                   # Utilitaires
│   ├── data_loaders/        # Chargeurs de données
│   ├── monitoring/          # Monitoring
│   └── helpers/             # Fonctions utilitaires
└── reports/                 # Rapports et documentation
    ├── technical/           # Rapports techniques
    ├── summaries/           # Résumés
    └── logs/                # Logs d'exécution
```

## 🚀 Utilisation Rapide

### 1. Entraînement
```bash
cd scripts/training/
python main_training.py
```

### 2. Test
```bash
cd scripts/testing/
python test_model_on_simulation_data.py
```

### 3. Analyse
```bash
cd scripts/analysis/
python analyze_ps3um_data.py
```

## 📊 Données Disponibles

- **Données simulées** : 22,540 profils dans `data/raw/Train/`
- **Données expérimentales** : 6,596 profils PS 3µm dans `data/experimental/`
- **Données préprocessées** : Échantillons dans `data/processed/`

## 🧠 Modèles

- **Architecture principale** : Réseau dense multi-couches
- **Optimisations** : CPU et GPU compatibles
- **Performances** : R² > 0.95 sur données simulées

## 📈 Résultats

- **Précision gap** : ±0.007 µm
- **Précision L'écran** : ±0.5 µm
- **Stabilité** : Excellente sur données expérimentales

## 👨‍💻 Auteur

Oussama GUELFAA - guelfaao@gmail.com

## 📅 Dernière mise à jour

18/07/2025 - Réorganisation complète du projet
"""
    
    with open("README.md", 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n📄 README principal mis à jour")

def main():
    """Fonction principale de réorganisation"""
    
    print("🔄 RÉORGANISATION DU DOSSIER Neural_Network_Gap_Lecran_Prediction")
    print("=" * 70)
    print("⚠️  Cette opération va déplacer tous les fichiers dans une nouvelle structure.")
    
    # Demander confirmation
    response = input("\n❓ Continuer ? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Opération annulée.")
        return
    
    try:
        # 1. Créer la structure de dossiers
        directories = create_directory_structure()
        
        # 2. Définir les mappings
        mappings = get_file_mappings()
        
        # 3. Déplacer les fichiers
        moved_count = move_files_safely(mappings)
        
        # 4. Créer les README
        create_readme_files()
        create_main_readme()
        
        print(f"\n✅ RÉORGANISATION TERMINÉE!")
        print(f"📊 {moved_count} fichiers/dossiers déplacés")
        print(f"📁 {len(directories)} dossiers principaux créés")
        print(f"📝 Documentation mise à jour")
        
        print(f"\n🎯 STRUCTURE FINALE:")
        for main_dir, info in directories.items():
            print(f"   📁 {main_dir}/ - {info['description']}")
        
        print(f"\n💡 PROCHAINES ÉTAPES:")
        print(f"   1. Vérifier que tous les fichiers sont au bon endroit")
        print(f"   2. Mettre à jour les imports dans les scripts si nécessaire")
        print(f"   3. Tester les scripts principaux")
        
    except Exception as e:
        print(f"❌ Erreur pendant la réorganisation: {e}")
        print("🔄 Vous pouvez restaurer manuellement si nécessaire")

if __name__ == "__main__":
    main()
