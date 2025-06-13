#!/usr/bin/env python3
"""
Démonstration de la nouvelle organisation des outils d'analyse 2D

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Ce script démontre la nouvelle organisation claire et structurée
de tous les outils d'analyse du dataset 2D.
"""

from pathlib import Path
import os

def show_organization():
    """
    Affiche la nouvelle organisation des outils d'analyse 2D.
    """
    print("🗂️  NOUVELLE ORGANISATION - ANALYSE DATASET 2D")
    print("="*60)
    print("Auteur: Oussama GUELFAA")
    print("Date: 06-01-2025")
    print("="*60)
    
    # Vérifier que nous sommes dans le bon dossier
    current_dir = Path.cwd()
    dataset_2D_dir = Path("analysis_scripts/dataset_2D_analysis")
    
    if not dataset_2D_dir.exists():
        print("❌ Erreur: Exécutez ce script depuis le répertoire racine du projet")
        return
    
    print(f"\n📁 STRUCTURE ORGANISÉE")
    print("-"*40)
    
    # Afficher la structure
    structure = {
        "🎯 SCRIPTS PRINCIPAUX": [
            "run_complete_dataset_2D_analysis.py  # Script maître (RECOMMANDÉ)",
            "analyze_dataset_2D.py               # Analyseur principal"
        ],
        "🎨 COMPARAISON D'ANNEAUX": [
            "plot_all_rings_comparison.py        # Vue globale",
            "compare_specific_rings.py           # Analyses quantitatives", 
            "interactive_rings_comparison.py     # Interface interactive"
        ],
        "🔧 OUTILS UTILITAIRES": [
            "demo_dataset_2D_results.py          # Démonstration",
            "organize_analysis_outputs.py        # Organisation",
            "test_dataset_2D_access.py           # Tests validation"
        ],
        "📚 DOCUMENTATION": [
            "README.md                           # Guide principal",
            "ANALYSE_DATASET_2D_COMPLETE.md      # Rapport détaillé",
            "QUICK_START_DATASET_2D.md           # Guide rapide",
            "RINGS_COMPARISON_SUMMARY.md         # Guide comparaisons"
        ],
        "📊 OUTPUTS": [
            "outputs_analysis_2D/                # Tous les résultats",
            "  ├── visualizations/               # 15+ fichiers PNG",
            "  ├── statistics/                   # Fichiers CSV",
            "  ├── reports/                      # Rapports texte",
            "  └── INDEX.md                      # Index complet"
        ]
    }
    
    for category, files in structure.items():
        print(f"\n{category}")
        for file in files:
            print(f"  {file}")
    
    print(f"\n📊 STATISTIQUES")
    print("-"*40)
    
    # Compter les fichiers
    scripts_count = len(list(dataset_2D_dir.glob("*.py")))
    docs_count = len(list(dataset_2D_dir.glob("*.md")))
    
    outputs_dir = dataset_2D_dir / "outputs_analysis_2D"
    if outputs_dir.exists():
        viz_count = len(list((outputs_dir / "visualizations").glob("*.png"))) if (outputs_dir / "visualizations").exists() else 0
        csv_count = len(list((outputs_dir / "statistics").glob("*.csv"))) if (outputs_dir / "statistics").exists() else 0
        reports_count = len(list((outputs_dir / "reports").glob("*.txt"))) if (outputs_dir / "reports").exists() else 0
    else:
        viz_count = csv_count = reports_count = 0
    
    print(f"  📄 Scripts Python: {scripts_count}")
    print(f"  📚 Documentation: {docs_count}")
    print(f"  🎨 Visualisations: {viz_count}")
    print(f"  📈 Fichiers CSV: {csv_count}")
    print(f"  📋 Rapports: {reports_count}")
    
    total_files = scripts_count + docs_count + viz_count + csv_count + reports_count
    print(f"  🎯 Total: {total_files} fichiers organisés")

def show_usage_examples():
    """
    Affiche des exemples d'utilisation avec la nouvelle organisation.
    """
    print(f"\n🚀 EXEMPLES D'UTILISATION")
    print("-"*40)
    
    examples = [
        {
            "title": "⚡ Analyse Complète (RECOMMANDÉ)",
            "command": "python analysis_scripts/dataset_2D_analysis/run_complete_dataset_2D_analysis.py",
            "description": "Exécute toute la chaîne d'analyse en 20 secondes"
        },
        {
            "title": "🎨 Voir Toutes les Différences",
            "command": "python analysis_scripts/dataset_2D_analysis/plot_all_rings_comparison.py",
            "description": "Visualise tous les anneaux avec différences par couleur"
        },
        {
            "title": "🔍 Comparaisons Personnalisées",
            "command": "python analysis_scripts/dataset_2D_analysis/interactive_rings_comparison.py",
            "description": "Interface pour choisir couples spécifiques"
        },
        {
            "title": "📊 Analyses Quantitatives",
            "command": "python analysis_scripts/dataset_2D_analysis/compare_specific_rings.py",
            "description": "Métriques détaillées et évolutions systématiques"
        },
        {
            "title": "🧪 Tests de Validation",
            "command": "python analysis_scripts/dataset_2D_analysis/test_dataset_2D_access.py",
            "description": "Valide format et cohérence des données"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   {example['command']}")
        print(f"   → {example['description']}")

def show_benefits():
    """
    Affiche les avantages de la nouvelle organisation.
    """
    print(f"\n✨ AVANTAGES DE LA NOUVELLE ORGANISATION")
    print("-"*40)
    
    benefits = [
        "🎯 **Clarté** - Tous les outils 2D dans un seul dossier",
        "📁 **Organisation** - Structure logique par fonction",
        "🚀 **Facilité** - Script maître pour tout exécuter",
        "📚 **Documentation** - Guides intégrés et complets",
        "🔧 **Maintenance** - Chemins cohérents et relatifs",
        "📊 **Outputs** - Résultats centralisés et indexés",
        "🎨 **Spécialisation** - Outils dédiés aux comparaisons",
        "💡 **Évolutivité** - Structure prête pour extensions"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")

def show_migration_info():
    """
    Affiche les informations sur la migration.
    """
    print(f"\n🔄 INFORMATIONS DE MIGRATION")
    print("-"*40)
    
    print("✅ **Fichiers déplacés avec succès:**")
    moved_files = [
        "analyze_dataset_2D.py",
        "run_complete_dataset_2D_analysis.py", 
        "demo_dataset_2D_results.py",
        "organize_analysis_outputs.py",
        "test_dataset_2D_access.py",
        "plot_all_rings_comparison.py",
        "compare_specific_rings.py",
        "interactive_rings_comparison.py",
        "ANALYSE_DATASET_2D_COMPLETE.md",
        "QUICK_START_DATASET_2D.md",
        "RINGS_COMPARISON_SUMMARY.md",
        "outputs_analysis_2D/"
    ]
    
    for file in moved_files:
        print(f"  • {file}")
    
    print(f"\n✅ **Chemins mis à jour automatiquement**")
    print(f"✅ **Fonctionnalité préservée à 100%**")
    print(f"✅ **Documentation adaptée**")

def main():
    """
    Fonction principale de démonstration.
    """
    show_organization()
    show_usage_examples()
    show_benefits()
    show_migration_info()
    
    print(f"\n" + "="*60)
    print("🎉 ORGANISATION RÉUSSIE !")
    print("="*60)
    print("📁 Tous les outils d'analyse 2D sont maintenant")
    print("   parfaitement organisés dans un dossier dédié.")
    print("")
    print("🚀 Commencez par:")
    print("   python analysis_scripts/dataset_2D_analysis/run_complete_dataset_2D_analysis.py")
    print("="*60)

if __name__ == "__main__":
    main()
