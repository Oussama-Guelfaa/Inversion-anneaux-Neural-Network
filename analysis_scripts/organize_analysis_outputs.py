#!/usr/bin/env python3
"""
Organisation et nettoyage des outputs d'analyse

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Ce script organise les fichiers de sortie de l'analyse et crée un index.
"""

import os
import shutil
from pathlib import Path
import datetime

def organize_outputs():
    """
    Organise les fichiers de sortie de l'analyse du dataset 2D.
    """
    print("🗂️  ORGANISATION DES OUTPUTS D'ANALYSE")
    print("="*50)
    
    output_path = Path("analysis_scripts/outputs_analysis_2D")
    
    if not output_path.exists():
        print("❌ Dossier outputs_analysis_2D non trouvé")
        return
    
    # Créer des sous-dossiers
    subdirs = {
        "visualizations": "Graphiques et visualisations",
        "statistics": "Fichiers de statistiques",
        "reports": "Rapports et documentation"
    }
    
    for subdir, description in subdirs.items():
        subdir_path = output_path / subdir
        subdir_path.mkdir(exist_ok=True)
        print(f"📁 Créé: {subdir}/ - {description}")
    
    # Organiser les fichiers
    file_mapping = {
        "*.png": "visualizations",
        "*.csv": "statistics", 
        "*.txt": "reports"
    }
    
    moved_files = 0
    for pattern, target_dir in file_mapping.items():
        files = list(output_path.glob(pattern))
        for file in files:
            if file.parent.name != target_dir:  # Éviter de déplacer si déjà dans le bon dossier
                target_path = output_path / target_dir / file.name
                shutil.move(str(file), str(target_path))
                print(f"   📄 {file.name} → {target_dir}/")
                moved_files += 1
    
    print(f"✅ {moved_files} fichiers organisés")
    
    # Créer un index
    create_index(output_path)

def create_index(output_path):
    """
    Crée un fichier index des outputs générés.
    """
    print("\n📋 Création de l'index des fichiers...")
    
    index_content = f"""# Index des Outputs - Analyse Dataset 2D

**Généré le:** {datetime.datetime.now().strftime('%d/%m/%Y à %H:%M')}
**Auteur:** Oussama GUELFAA

## 📊 Visualisations (visualizations/)

### Distributions et Densités
- **parameter_distributions.png** - Histogrammes et heatmap des paramètres (gap, L_ecran)
- **parameter_density_2D.png** - Densité hexagonale dans l'espace des paramètres

### Couverture du Dataset
- **coverage_matrix.png** - Matrice de couverture pour identifier les zones manquantes

### Échantillons d'Anneaux
- **ring_samples_grid.png** - Grille de 36 échantillons représentatifs d'anneaux
- **rings_by_L_ecran.png** - Évolution des profils d'anneaux par L_ecran fixe

## 📈 Statistiques (statistics/)

- **dataset_statistics.csv** - Statistiques générales du dataset
- **detailed_statistics.csv** - Statistiques détaillées par variable

## 📄 Rapports (reports/)

- **analysis_report.txt** - Rapport complet de l'analyse avec recommandations

## 🎯 Résultats Clés

- **2440 fichiers** analysés (100% de complétude)
- **40 gaps** de 0.005 à 0.2 µm
- **61 L_ecran** de 10.0 à 11.5 µm
- **30.9 MB** de données au total
- **Qualité excellente** (ratios cohérents)

## 🚀 Utilisation

### Pour l'Entraînement de Réseaux
- Train: 1708 échantillons (70%)
- Validation: 366 échantillons (15%) 
- Test: 366 échantillons (15%)

### Préprocessing Recommandé
1. Tronquer les profils à 600 points
2. Normalisation StandardScaler
3. Validation croisée stratifiée

## 📞 Contact

Pour questions sur cette analyse: Oussama GUELFAA
"""
    
    index_path = output_path / "INDEX.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"✅ Index créé: {index_path}")

def generate_summary():
    """
    Génère un résumé final de l'analyse.
    """
    print("\n📝 RÉSUMÉ FINAL")
    print("-"*30)
    
    output_path = Path("analysis_scripts/outputs_analysis_2D")
    
    # Compter les fichiers par catégorie
    viz_files = len(list((output_path / "visualizations").glob("*.png"))) if (output_path / "visualizations").exists() else 0
    stat_files = len(list((output_path / "statistics").glob("*.csv"))) if (output_path / "statistics").exists() else 0
    report_files = len(list((output_path / "reports").glob("*.txt"))) if (output_path / "reports").exists() else 0
    
    print(f"   📊 Visualisations: {viz_files} fichiers PNG")
    print(f"   📈 Statistiques: {stat_files} fichiers CSV") 
    print(f"   📄 Rapports: {report_files} fichiers TXT")
    print(f"   📋 Index: INDEX.md")
    
    total_files = viz_files + stat_files + report_files + 1
    print(f"\n   🎯 Total: {total_files} fichiers organisés")
    
    # Taille totale
    if output_path.exists():
        total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / (1024*1024)
        print(f"   💾 Taille totale: {total_size:.1f} MB")

def main():
    """
    Fonction principale d'organisation.
    """
    print("🗂️  ORGANISATION DES OUTPUTS D'ANALYSE DATASET 2D")
    print("Auteur: Oussama GUELFAA")
    print("Date: 06-01-2025")
    print("="*60)
    
    # Organiser les fichiers
    organize_outputs()
    
    # Générer le résumé
    generate_summary()
    
    print("\n" + "="*60)
    print("✅ Organisation terminée ! Fichiers prêts pour utilisation.")
    print("📁 Consultez analysis_scripts/outputs_analysis_2D/INDEX.md")
    print("="*60)

if __name__ == "__main__":
    main()
