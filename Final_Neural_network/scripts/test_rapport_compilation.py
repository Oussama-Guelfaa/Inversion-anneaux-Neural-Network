#!/usr/bin/env python3
"""
Test de Compilation du Rapport LaTeX Amélioré
Author: Oussama GUELFAA
Date: 01/08/2025

Ce script teste la compilation du rapport LaTeX avec la nouvelle mise en forme.
"""

import os
import subprocess
import sys

def create_minimal_test_document():
    """Créer un document LaTeX minimal pour tester la compilation."""
    
    minimal_tex = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[french]{babel}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{float}
\usepackage{geometry}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{tcolorbox}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{listings}

% Configuration de base
\geometry{margin=2.5cm}

% Couleurs
\definecolor{primaryblue}{RGB}{25,55,109}
\definecolor{accentgreen}{RGB}{39,174,96}

% Boîtes colorées
\tcbuselibrary{most}
\newtcolorbox{definitionbox}{
    colback=primaryblue!5,
    colframe=primaryblue,
    boxrule=1pt,
    arc=3pt,
    title=Définition
}

\title{Test de Compilation LaTeX}
\author{Oussama GUELFAA}
\date{\today}

\begin{document}

\maketitle

\section{Test des Fonctionnalités}

\subsection{Test des Boîtes Colorées}

\begin{definitionbox}
Ceci est un test de boîte de définition avec une couleur bleue.
\end{definitionbox}

\subsection{Test des Équations}

Voici une équation de test :
\begin{equation}
\mathcal{L}_{total} = \mathcal{L}_{regression} + \lambda \times \mathcal{L}_{domain}
\end{equation}

\subsection{Test du Code}

\begin{lstlisting}[language=Python, caption=Test de code]
def test_function():
    return "Hello World"
\end{lstlisting}

\section{Conclusion}

Si vous voyez ce document, la compilation LaTeX fonctionne correctement !

\end{document}
"""
    
    with open('test_minimal.tex', 'w', encoding='utf-8') as f:
        f.write(minimal_tex)
    
    print("✓ Document LaTeX minimal créé : test_minimal.tex")

def test_latex_packages():
    """Tester la disponibilité des packages LaTeX requis."""
    
    print("Test des packages LaTeX requis...")
    
    # Créer un fichier de test pour chaque package critique
    test_packages = [
        'tcolorbox',
        'titlesec', 
        'xcolor',
        'fancyhdr',
        'listings'
    ]
    
    for package in test_packages:
        test_tex = f"""
\\documentclass{{article}}
\\usepackage{{{package}}}
\\begin{{document}}
Test du package {package}
\\end{{document}}
"""
        
        test_file = f'test_{package}.tex'
        with open(test_file, 'w') as f:
            f.write(test_tex)
        
        # Tester la compilation
        try:
            result = subprocess.run([
                'pdflatex', 
                '-interaction=nonstopmode',
                '-output-directory=.',
                test_file
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"  ✓ {package}")
            else:
                print(f"  ✗ {package} - Erreur de compilation")
                
        except subprocess.TimeoutExpired:
            print(f"  ⚠ {package} - Timeout")
        except FileNotFoundError:
            print(f"  ✗ pdflatex non trouvé")
            return False
        finally:
            # Nettoyer les fichiers de test
            for ext in ['.tex', '.pdf', '.aux', '.log']:
                test_clean = f'test_{package}{ext}'
                if os.path.exists(test_clean):
                    os.remove(test_clean)
    
    return True

def compile_minimal_test():
    """Compiler le document de test minimal."""
    
    print("\nCompilation du document de test minimal...")
    
    try:
        result = subprocess.run([
            'pdflatex', 
            '-interaction=nonstopmode',
            'test_minimal.tex'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Compilation réussie !")
            if os.path.exists('test_minimal.pdf'):
                size_mb = os.path.getsize('test_minimal.pdf') / (1024 * 1024)
                print(f"✓ PDF généré : test_minimal.pdf ({size_mb:.2f} MB)")
                return True
            else:
                print("✗ PDF non généré")
                return False
        else:
            print("✗ Erreur de compilation :")
            print(result.stdout[-500:])  # Dernières 500 caractères
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Timeout lors de la compilation")
        return False
    except FileNotFoundError:
        print("✗ pdflatex non trouvé")
        return False

def check_main_report():
    """Vérifier le rapport principal."""
    
    main_report = 'rapport_recherche_reseaux_neurones.tex'
    
    print(f"\nVérification du rapport principal : {main_report}")
    
    if not os.path.exists(main_report):
        print(f"✗ Fichier non trouvé : {main_report}")
        return False
    
    # Analyser le contenu
    with open(main_report, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Vérifications de base
    checks = {
        'documentclass': '\\documentclass' in content,
        'begin_document': '\\begin{document}' in content,
        'end_document': '\\end{document}' in content,
        'tcolorbox': 'tcolorbox' in content,
        'definitionbox': 'definitionbox' in content,
        'conceptbox': 'conceptbox' in content,
        'sections': '\\section{' in content,
        'french_babel': 'french' in content
    }
    
    print("Vérifications du contenu :")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    # Compter les éléments
    sections = content.count('\\section{')
    subsections = content.count('\\subsection{')
    figures = content.count('\\includegraphics')
    equations = content.count('\\begin{equation}')
    
    print(f"\nStatistiques du document :")
    print(f"  Sections : {sections}")
    print(f"  Sous-sections : {subsections}")
    print(f"  Figures : {figures}")
    print(f"  Équations : {equations}")
    print(f"  Taille : {len(content)} caractères")
    
    return all(checks.values())

def cleanup_test_files():
    """Nettoyer les fichiers de test."""
    
    test_files = [
        'test_minimal.tex',
        'test_minimal.pdf',
        'test_minimal.aux',
        'test_minimal.log'
    ]
    
    print("\nNettoyage des fichiers de test...")
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"  ✓ Supprimé : {file}")

def main():
    """Fonction principale de test."""
    
    print("=" * 60)
    print("TEST DE COMPILATION DU RAPPORT LATEX AMÉLIORÉ")
    print("=" * 60)
    
    success = True
    
    # Test 1 : Vérifier le rapport principal
    if not check_main_report():
        print("\n⚠ Problèmes détectés dans le rapport principal")
        success = False
    
    # Test 2 : Tester les packages LaTeX
    if not test_latex_packages():
        print("\n⚠ Problèmes avec les packages LaTeX")
        success = False
    
    # Test 3 : Créer et compiler un document minimal
    create_minimal_test_document()
    if not compile_minimal_test():
        print("\n⚠ Échec de la compilation de test")
        success = False
    
    # Nettoyage
    cleanup_test_files()
    
    # Résumé
    print("\n" + "=" * 60)
    if success:
        print("✅ TOUS LES TESTS SONT PASSÉS !")
        print("Le rapport LaTeX devrait compiler correctement.")
        print("\nPour compiler le rapport complet :")
        print("  pdflatex rapport_recherche_reseaux_neurones.tex")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifiez l'installation LaTeX et les packages requis.")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
