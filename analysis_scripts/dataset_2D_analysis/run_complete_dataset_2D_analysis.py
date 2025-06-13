#!/usr/bin/env python3
"""
Script maître pour l'analyse complète du dataset 2D

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Ce script exécute toute la chaîne d'analyse du dataset 2D en une seule commande.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    """
    Exécute un script et affiche le résultat.
    
    Args:
        script_name (str): Nom du script à exécuter
        description (str): Description de l'étape
    """
    print(f"\n🚀 {description}")
    print("="*60)
    
    script_path = Path("analysis_scripts/dataset_2D_analysis") / script_name
    
    if not script_path.exists():
        print(f"❌ Script non trouvé: {script_path}")
        return False
    
    try:
        start_time = time.time()
        
        # Exécuter le script depuis le répertoire racine
        import os
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd=root_dir)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Succès en {duration:.1f}s")
            # Afficher les dernières lignes de sortie pour le feedback
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 3:
                print("📋 Résumé:")
                for line in output_lines[-3:]:
                    if line.strip():
                        print(f"   {line}")
            return True
        else:
            print(f"❌ Échec après {duration:.1f}s")
            print("Erreur:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        return False

def check_prerequisites():
    """
    Vérifie que tous les prérequis sont présents.
    """
    print("🔍 VÉRIFICATION DES PRÉREQUIS")
    print("="*40)
    
    # Vérifier le dossier dataset
    dataset_path = Path("data_generation/dataset_2D")
    if not dataset_path.exists():
        print(f"❌ Dossier dataset non trouvé: {dataset_path}")
        return False
    
    mat_files = list(dataset_path.glob("*.mat"))
    mat_files = [f for f in mat_files if f.name != "labels.mat"]
    
    print(f"✅ Dataset trouvé: {len(mat_files)} fichiers .mat")
    
    # Vérifier les modules Python
    required_modules = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Modules manquants: {missing_modules}")
        print("   Installez avec: pip install " + " ".join(missing_modules))
        return False
    
    print("✅ Tous les modules Python requis sont présents")
    
    # Vérifier l'espace disque (approximatif)
    try:
        import shutil
        free_space = shutil.disk_usage(".").free / (1024*1024)  # MB
        if free_space < 50:  # 50 MB minimum
            print(f"⚠️  Espace disque faible: {free_space:.1f} MB")
        else:
            print(f"✅ Espace disque suffisant: {free_space:.1f} MB")
    except:
        print("⚠️  Impossible de vérifier l'espace disque")
    
    return True

def main():
    """
    Fonction principale - exécute toute la chaîne d'analyse.
    """
    print("🔬 ANALYSE COMPLÈTE DU DATASET 2D - SCRIPT MAÎTRE")
    print("Auteur: Oussama GUELFAA")
    print("Date: 06-01-2025")
    print("="*70)
    
    start_total = time.time()
    
    # Étape 0: Vérifications
    if not check_prerequisites():
        print("\n❌ Prérequis non satisfaits. Arrêt de l'analyse.")
        return
    
    # Définir la séquence d'analyse
    analysis_steps = [
        ("analyze_dataset_2D.py", "ÉTAPE 1: Analyse principale du dataset"),
        ("organize_analysis_outputs.py", "ÉTAPE 2: Organisation des outputs"),
        ("test_dataset_2D_access.py", "ÉTAPE 3: Tests de validation"),
        ("demo_dataset_2D_results.py", "ÉTAPE 4: Démonstration des résultats")
    ]
    
    # Exécuter chaque étape
    success_count = 0
    for script_name, description in analysis_steps:
        success = run_script(script_name, description)
        if success:
            success_count += 1
        else:
            print(f"\n⚠️  Échec de l'étape: {description}")
            print("   L'analyse continue avec les étapes suivantes...")
    
    # Résumé final
    end_total = time.time()
    total_duration = end_total - start_total
    
    print("\n" + "="*70)
    print("📋 RÉSUMÉ FINAL DE L'ANALYSE")
    print("="*70)
    
    print(f"⏱️  Durée totale: {total_duration:.1f} secondes")
    print(f"✅ Étapes réussies: {success_count}/{len(analysis_steps)}")
    
    if success_count == len(analysis_steps):
        print("🎉 ANALYSE COMPLÈTE RÉUSSIE !")
        print("\n📁 Résultats disponibles dans:")
        print("   📊 analysis_scripts/outputs_analysis_2D/")
        print("   📋 analysis_scripts/ANALYSE_DATASET_2D_COMPLETE.md")
        
        print("\n🎯 PROCHAINES ÉTAPES:")
        print("   1. Consultez INDEX.md pour naviguer dans les résultats")
        print("   2. Examinez les visualisations PNG")
        print("   3. Lisez le rapport complet analysis_report.txt")
        print("   4. Utilisez les recommandations pour l'entraînement")
        
    else:
        print("⚠️  Analyse partiellement réussie")
        print("   Vérifiez les erreurs ci-dessus")
        print("   Certains résultats peuvent être disponibles")
    
    # Afficher les fichiers générés
    output_path = Path("analysis_scripts/outputs_analysis_2D")
    if output_path.exists():
        all_files = list(output_path.rglob("*"))
        file_count = len([f for f in all_files if f.is_file()])
        total_size = sum(f.stat().st_size for f in all_files if f.is_file()) / (1024*1024)
        
        print(f"\n📊 OUTPUTS GÉNÉRÉS:")
        print(f"   📁 {file_count} fichiers créés")
        print(f"   💾 {total_size:.1f} MB au total")
        
        # Lister les principaux fichiers
        key_files = [
            "INDEX.md",
            "visualizations/ring_samples_grid.png",
            "reports/analysis_report.txt",
            "statistics/dataset_statistics.csv"
        ]
        
        print(f"\n📋 Fichiers clés:")
        for key_file in key_files:
            file_path = output_path / key_file
            if file_path.exists():
                size = file_path.stat().st_size / 1024
                print(f"   ✅ {key_file} ({size:.1f} KB)")
    
    print("\n" + "="*70)
    print("🔬 Analyse du dataset 2D terminée !")
    print("="*70)

if __name__ == "__main__":
    main()
