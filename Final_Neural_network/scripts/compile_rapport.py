#!/usr/bin/env python3
"""
Script de Compilation du Rapport LaTeX
Author: Oussama GUELFAA
Date: 01/08/2025

Ce script compile le rapport de recherche LaTeX et génère le PDF final.
"""

import subprocess
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = BASE_DIR / 'plots'
DOCS_DIR = BASE_DIR / 'docs'

def check_latex_installation():
    """Vérifier si LaTeX est installé."""
    try:
        result = subprocess.run(['pdflatex', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ pdflatex trouvé")
            return True
        else:
            print("✗ pdflatex non trouvé")
            return False
    except FileNotFoundError:
        print("✗ pdflatex non installé")
        return False

def compile_latex(tex_file, max_runs=3):
    """
    Compiler le fichier LaTeX en PDF.
    
    Args:
        tex_file: Nom du fichier .tex
        max_runs: Nombre maximum de compilations (pour les références)
    
    Returns:
        bool: True si la compilation réussit
    """
    print(f"Compilation de {tex_file}...")
    
    for run in range(max_runs):
        print(f"  Passe {run + 1}/{max_runs}")
        
        try:
            result = subprocess.run([
                'pdflatex', 
                '-interaction=nonstopmode',
                '-output-directory=.',
                tex_file
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"✗ Erreur lors de la compilation (passe {run + 1}):")
                print(result.stdout[-1000:])  # Dernières 1000 caractères
                return False
            else:
                print(f"✓ Passe {run + 1} réussie")
                
        except Exception as e:
            print(f"✗ Erreur d'exécution: {e}")
            return False
    
    return True

def clean_auxiliary_files(base_name):
    """Nettoyer les fichiers auxiliaires LaTeX."""
    extensions_to_remove = ['.aux', '.log', '.toc', '.out', '.fls', '.fdb_latexmk']
    
    print("Nettoyage des fichiers auxiliaires...")
    for ext in extensions_to_remove:
        file_to_remove = base_name + ext
        if os.path.exists(file_to_remove):
            try:
                os.remove(file_to_remove)
                print(f"  ✓ Supprimé: {file_to_remove}")
            except Exception as e:
                print(f"  ⚠ Impossible de supprimer {file_to_remove}: {e}")

def check_images_exist():
    """Vérifier que toutes les images référencées existent."""
    required_images = [
        str(PLOTS_DIR / 'sim_vs_exp_profiles.png'),
        str(PLOTS_DIR / 'domain_adaptive_results_fixed.png'),
        str(PLOTS_DIR / 'experimental_vs_closest_simulation_profiles.png')
    ]

    print("Vérification des images requises...")
    all_exist = True
    
    for image in required_images:
        if os.path.exists(image):
            print(f"  ✓ {image}")
        else:
            print(f"  ✗ {image} manquant")
            all_exist = False
    
    return all_exist

def generate_summary():
    """Générer un résumé du rapport."""
    tex_file = str(DOCS_DIR / 'rapport_recherche_reseaux_neurones.tex')
    pdf_file = str(DOCS_DIR / 'rapport_recherche_reseaux_neurones.pdf')

    print("\n" + "=" * 60)
    print("RÉSUMÉ DU RAPPORT GÉNÉRÉ")
    print("=" * 60)
    
    if os.path.exists(pdf_file):
        pdf_size = os.path.getsize(pdf_file) / (1024 * 1024)  # MB
        print(f"✓ PDF généré: {pdf_file} ({pdf_size:.2f} MB)")
    else:
        print(f"✗ PDF non généré: {pdf_file}")
    
    if os.path.exists(tex_file):
        # Compter les lignes du fichier LaTeX
        with open(tex_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"✓ Source LaTeX: {tex_file} ({len(lines)} lignes)")
        
        # Analyser le contenu
        sections = [line.strip() for line in lines if line.strip().startswith('\\section{')]
        subsections = [line.strip() for line in lines if line.strip().startswith('\\subsection{')]
        figures = [line.strip() for line in lines if '\\includegraphics' in line]
        tables = [line.strip() for line in lines if line.strip().startswith('\\begin{table}')]
        
        print(f"  Sections: {len(sections)}")
        print(f"  Sous-sections: {len(subsections)}")
        print(f"  Figures: {len(figures)}")
        print(f"  Tableaux: {len(tables)}")
    
    print("\nContenu du rapport:")
    print("  1. Introduction - Objectifs et contexte")
    print("  2. Préparation des données - Troncature, normalisation, alignement")
    print("  3. Modèle de réseau de neurones - Architecture DANN")
    print("  4. Entraînement et stabilité - Problèmes et solutions")
    print("  5. Résultats expérimentaux - Prédictions et analyse")
    print("  6. Conclusion - Apprentissages et perspectives")
    
    print("\nStyle narratif personnel avec formulations comme:")
    print("  • 'J'ai d'abord supposé que...'")
    print("  • 'Après plusieurs essais infructueux...'")
    print("  • 'J'ai donc décidé de modifier...'")
    print("  • 'Les résultats se sont améliorés lorsque...'")
    
    print("=" * 60)

def main():
    """Fonction principale de compilation."""
    print("=" * 60)
    print("COMPILATION DU RAPPORT DE RECHERCHE LATEX")
    print("=" * 60)
    
    tex_file = str(DOCS_DIR / 'rapport_recherche_reseaux_neurones.tex')

    # Vérifications préliminaires
    if not os.path.exists(tex_file):
        print(f"✗ Fichier LaTeX non trouvé: {tex_file}")
        return False
    
    if not check_latex_installation():
        print("\nPour installer LaTeX:")
        print("  macOS: brew install --cask mactex")
        print("  Ubuntu: sudo apt-get install texlive-full")
        print("  Windows: Télécharger MiKTeX ou TeX Live")
        return False
    
    if not check_images_exist():
        print("\n⚠ Certaines images sont manquantes.")
        print("Le rapport sera compilé mais certaines figures n'apparaîtront pas.")
        response = input("Continuer quand même? (o/n): ")
        if response.lower() != 'o':
            return False
    
    # Compilation
    print(f"\nDébut de la compilation...")
    success = compile_latex(tex_file)
    
    if success:
        print("\n✓ Compilation réussie!")
        
        # Nettoyage optionnel
        base_name = tex_file.replace('.tex', '')
        response = input("Nettoyer les fichiers auxiliaires? (o/n): ")
        if response.lower() == 'o':
            clean_auxiliary_files(base_name)
        
        # Résumé
        generate_summary()
        
        print(f"\n🎉 Rapport PDF généré: {base_name}.pdf")
        return True
    else:
        print("\n✗ Échec de la compilation")
        print("Vérifiez les erreurs ci-dessus et corrigez le fichier LaTeX.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
