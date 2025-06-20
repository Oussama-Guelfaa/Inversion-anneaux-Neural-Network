# Rapport de Recherche LaTeX

## Description
Ce rapport présente l'amélioration d'un réseau de neurones dual pour la prédiction haute précision des paramètres Gap et L_ecran dans l'analyse holographique.

## Fichiers
- `rapport_recherche.tex` : Document LaTeX principal
- `rapport_recherche.pdf` : Rapport compilé (généré)
- `compile_rapport.py` : Script de compilation
- `plots/` : Figures utilisées dans le rapport

## Compilation
Pour compiler le rapport :
```bash
python compile_rapport.py
```

Ou manuellement :
```bash
pdflatex rapport_recherche.tex
pdflatex rapport_recherche.tex
```

## Contenu du Rapport
1. Introduction et contexte
2. Description de la version initiale
3. Démarche de modification
4. Architecture améliorée
5. Résultats et analyse
6. Conclusion et perspectives

## Performances Obtenues
- Gap Accuracy: 92.9% (±0.007µm)
- L_ecran Accuracy: 94.6% (±0.1µm)
- Combined R²: 99.2%
- Dataset augmenté: 17,080 échantillons (facteur 7.0x)

## Auteur
Oussama GUELFAA - 16 Juin 2025
