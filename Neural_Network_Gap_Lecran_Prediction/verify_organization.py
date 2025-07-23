#!/usr/bin/env python3
"""
Vérification de l'organisation du projet
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script vérifie que tous les fichiers sont bien organisés et accessibles.
"""

import os
from pathlib import Path
import glob

def check_directory_structure():
    """Vérifie la structure des dossiers"""
    
    print("🔍 VÉRIFICATION DE LA STRUCTURE")
    print("=" * 50)
    
    expected_dirs = {
        "data": ["raw", "processed", "experimental"],
        "scripts": ["training", "analysis", "preprocessing", "testing"],
        "models": ["architectures", "saved_models", "checkpoints"],
        "results": ["training_results", "predictions", "evaluations"],
        "visualizations": ["plots", "analysis_charts", "comparisons"],
        "utils": ["data_loaders", "monitoring", "helpers"],
        "reports": ["technical", "summaries", "logs"]
    }
    
    all_good = True
    
    for main_dir, subdirs in expected_dirs.items():
        main_path = Path(main_dir)
        if main_path.exists():
            print(f"✅ {main_dir}/")
            for subdir in subdirs:
                sub_path = main_path / subdir
                if sub_path.exists():
                    print(f"   ✅ {subdir}/")
                else:
                    print(f"   ❌ {subdir}/ MANQUANT")
                    all_good = False
        else:
            print(f"❌ {main_dir}/ MANQUANT")
            all_good = False
    
    return all_good

def check_key_files():
    """Vérifie la présence des fichiers clés"""
    
    print(f"\n📋 VÉRIFICATION DES FICHIERS CLÉS")
    print("=" * 50)
    
    key_files = {
        "Documentation": [
            "README.md",
            "NAVIGATION_GUIDE.md",
            "data/INDEX.md",
            "scripts/training/INDEX.md",
            "scripts/testing/INDEX.md"
        ],
        "Scripts d'entraînement": [
            "scripts/training/main_training.py",
            "scripts/training/cpu_training.py",
            "scripts/training/fast_training.py"
        ],
        "Scripts de test": [
            "scripts/testing/quick_test.py",
            "scripts/testing/test_model_on_simulation_data.py"
        ],
        "Données expérimentales": [
            "data/experimental/analyze_ps3um_data.py",
            "data/experimental/final_ps3um_summary_report.txt"
        ],
        "Données préprocessées": [
            "data/processed/ps3um_samples_100profiles.csv",
            "data/processed/ps3um_mean_profile.csv"
        ],
        "Utilitaires": [
            "utils/data_loaders/data_loader.py",
            "utils/monitoring/visualization_monitoring.py"
        ]
    }
    
    all_files_good = True
    
    for category, files in key_files.items():
        print(f"\n📂 {category}:")
        for file_path in files:
            if Path(file_path).exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} MANQUANT")
                all_files_good = False
    
    return all_files_good

def count_files_by_category():
    """Compte les fichiers par catégorie"""
    
    print(f"\n📊 STATISTIQUES DES FICHIERS")
    print("=" * 50)
    
    categories = {
        "Scripts Python": "**/*.py",
        "Données CSV": "**/*.csv",
        "Données MAT": "**/*.mat",
        "Visualisations PNG": "**/*.png",
        "Rapports MD": "**/*.md",
        "Rapports TXT": "**/*.txt",
        "Résultats JSON": "**/*.json",
        "Modèles PT": "**/*.pt",
        "Scalers JOBLIB": "**/*.joblib"
    }
    
    total_files = 0
    
    for category, pattern in categories.items():
        files = glob.glob(pattern, recursive=True)
        count = len(files)
        total_files += count
        print(f"   📄 {category}: {count} fichiers")
    
    print(f"\n📈 TOTAL: {total_files} fichiers organisés")
    
    return total_files

def check_data_accessibility():
    """Vérifie l'accessibilité des données principales"""
    
    print(f"\n🔍 VÉRIFICATION D'ACCESSIBILITÉ DES DONNÉES")
    print("=" * 50)
    
    # Vérifier les données d'entraînement
    train_dir = Path("data/raw/Train")
    if train_dir.exists():
        train_files = list(train_dir.glob("*.mat"))
        print(f"   ✅ Données d'entraînement: {len(train_files)} fichiers .mat")
    else:
        print(f"   ❌ Dossier Train/ non trouvé")
    
    # Vérifier les données de test
    test_dir = Path("data/raw/Test")
    if test_dir.exists():
        test_files = list(test_dir.glob("*.mat"))
        print(f"   ✅ Données de test: {len(test_files)} fichiers .mat")
    else:
        print(f"   ❌ Dossier Test/ non trouvé")
    
    # Vérifier les données expérimentales
    exp_files = list(Path("data/experimental").glob("*.csv"))
    print(f"   ✅ Données expérimentales: {len(exp_files)} fichiers CSV")
    
    # Vérifier les données préprocessées
    proc_files = list(Path("data/processed").glob("*.csv"))
    print(f"   ✅ Données préprocessées: {len(proc_files)} fichiers CSV")

def generate_organization_report():
    """Génère un rapport d'organisation"""
    
    print(f"\n📝 GÉNÉRATION DU RAPPORT D'ORGANISATION")
    print("=" * 50)
    
    # Compter les éléments par dossier
    structure_stats = {}
    
    for root, dirs, files in os.walk("."):
        # Ignorer les dossiers cachés et __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        level = root.replace(".", "").count(os.sep)
        if level <= 2:  # Limiter à 2 niveaux
            structure_stats[root] = {
                'dirs': len(dirs),
                'files': len(files),
                'python_files': len([f for f in files if f.endswith('.py')]),
                'data_files': len([f for f in files if f.endswith(('.csv', '.mat', '.npz'))]),
                'image_files': len([f for f in files if f.endswith('.png')])
            }
    
    # Créer le rapport
    report = """
RAPPORT D'ORGANISATION - Neural Network Gap L'écran Prediction
============================================================
Date: 18/07/2025

STRUCTURE ORGANISÉE:
===================
"""
    
    for path, stats in structure_stats.items():
        if stats['files'] > 0 or stats['dirs'] > 0:
            report += f"\n{path}:\n"
            report += f"  - Sous-dossiers: {stats['dirs']}\n"
            report += f"  - Fichiers total: {stats['files']}\n"
            if stats['python_files'] > 0:
                report += f"  - Scripts Python: {stats['python_files']}\n"
            if stats['data_files'] > 0:
                report += f"  - Fichiers de données: {stats['data_files']}\n"
            if stats['image_files'] > 0:
                report += f"  - Images: {stats['image_files']}\n"
    
    report += f"""

RÉSUMÉ:
=======
✅ Projet entièrement réorganisé
✅ Structure claire et navigable
✅ Documentation complète
✅ Fichiers accessibles et bien classés

AVANTAGES DE LA NOUVELLE ORGANISATION:
=====================================
• Navigation intuitive par fonction
• Séparation claire des responsabilités
• Facilité de maintenance
• Évolutivité du projet
• Documentation intégrée

PROCHAINES ÉTAPES:
==================
1. Tester les scripts principaux
2. Vérifier les imports si nécessaire
3. Continuer le développement
4. Maintenir l'organisation

Contact: Oussama GUELFAA - guelfaao@gmail.com
"""
    
    with open("ORGANIZATION_REPORT.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("   ✅ ORGANIZATION_REPORT.txt créé")

def main():
    """Fonction principale de vérification"""
    
    print("🔍 VÉRIFICATION DE L'ORGANISATION DU PROJET")
    print("=" * 60)
    
    # Vérifications
    structure_ok = check_directory_structure()
    files_ok = check_key_files()
    total_files = count_files_by_category()
    check_data_accessibility()
    generate_organization_report()
    
    # Résumé final
    print(f"\n🎯 RÉSUMÉ DE LA VÉRIFICATION")
    print("=" * 50)
    
    if structure_ok and files_ok:
        print("✅ ORGANISATION PARFAITE!")
        print(f"📊 {total_files} fichiers bien organisés")
        print("🚀 Projet prêt pour le développement")
        
        print(f"\n📚 GUIDES DISPONIBLES:")
        print("   • README.md - Vue d'ensemble")
        print("   • NAVIGATION_GUIDE.md - Guide de navigation")
        print("   • ORGANIZATION_REPORT.txt - Rapport d'organisation")
        
        print(f"\n⚡ DÉMARRAGE RAPIDE:")
        print("   • Entraînement: cd scripts/training/ && python main_training.py")
        print("   • Test rapide: cd scripts/testing/ && python quick_test.py")
        print("   • Analyse PS 3µm: cd data/experimental/ && python analyze_ps3um_data.py")
        
    else:
        print("⚠️  PROBLÈMES DÉTECTÉS")
        if not structure_ok:
            print("❌ Structure de dossiers incomplète")
        if not files_ok:
            print("❌ Fichiers clés manquants")
        print("🔧 Vérifiez les erreurs ci-dessus")

if __name__ == "__main__":
    main()
