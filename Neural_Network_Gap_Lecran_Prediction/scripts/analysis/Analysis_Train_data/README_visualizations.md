# Visualisation des Vecteurs des Ratios - Dataset 2

## Vue d'ensemble

Ce projet contient plusieurs scripts Python pour visualiser et analyser les vecteurs des ratios contenus dans le dataset de fichiers .mat. Le dataset contient **22,540 fichiers** avec différents paramètres de gap et de longueur.

## Structure des données

Chaque fichier .mat contient :
- `ratio` : Vecteur des ratios (1000 éléments)
- `gap` : Valeur du gap en μm
- `L_ecran_subs` : Longueur en μm  
- `x` : Vecteur de positions

**Gammes des paramètres :**
- Gap : 0.0050 - 0.7000 μm
- Longueur : 8.000 - 12.000 μm

## Scripts disponibles

### 1. `visualize_ratio_vectors.py` 
**Script de base pour visualiser des échantillons aléatoires**

```bash
python visualize_ratio_vectors.py
```

**Fonctionnalités :**
- Sélectionne 15 échantillons aléatoires
- Trace les vecteurs des ratios superposés
- Affiche les paramètres (gap, longueur) dans la légende
- Génère : `ratio_vectors_overlay.png`

### 2. `advanced_ratio_analysis.py`
**Analyse complète avec visualisations multiples**

```bash
python advanced_ratio_analysis.py
```

**Fonctionnalités :**
- 6 sous-graphiques différents :
  1. Vecteurs des ratios superposés
  2. Distribution des valeurs de ratios
  3. Relation Gap vs Statistiques
  4. Heatmap des ratios
  5. Longueur vs Moyenne des ratios
  6. Boxplot par gamme de gap
- Statistiques détaillées
- Génère : `comprehensive_ratio_analysis.png`

### 3. `interactive_ratio_viewer.py`
**Interface interactive pour exploration personnalisée**

```bash
python interactive_ratio_viewer.py
```

**Fonctionnalités :**
- Menu interactif avec 6 options
- Filtrage par gamme de gap et/ou longueur
- Comparaison des extrêmes
- Chargement de tout le dataset (peut prendre du temps)

### 4. `quick_demo.py`
**Démonstration rapide avec exemples sélectionnés**

```bash
python quick_demo.py
```

**Fonctionnalités :**
- Sélection d'exemples représentatifs
- Grille de 12 échantillons aléatoires
- Statistiques détaillées par échantillon
- Génère : `demo_ratio_vectors.png` et `grid_ratio_vectors.png`

## Résultats générés

### Images créées :
1. **`ratio_vectors_overlay.png`** - Superposition de 15 échantillons aléatoires
2. **`comprehensive_ratio_analysis.png`** - Analyse complète en 6 panneaux
3. **`demo_ratio_vectors.png`** - Exemples sélectionnés avec différents gaps
4. **`grid_ratio_vectors.png`** - Grille de 12 échantillons individuels

## Observations principales

### Statistiques globales (basées sur 25 échantillons) :
- **Moyenne des ratios :** 0.9017
- **Écart-type :** 0.3434
- **Min/Max :** 0.0158 / 1.8066
- **Médiane :** ~0.88

### Tendances observées :
1. **Stabilité des moyennes :** Les moyennes des ratios restent relativement stables (~0.89-0.91) indépendamment du gap
2. **Variabilité constante :** L'écart-type reste autour de 0.34-0.35 pour tous les échantillons
3. **Distribution :** Les ratios suivent une distribution centrée autour de 0.9

## Utilisation recommandée

### Pour une visualisation rapide :
```bash
python visualize_ratio_vectors.py
```

### Pour une analyse complète :
```bash
python advanced_ratio_analysis.py
```

### Pour une exploration personnalisée :
```bash
python interactive_ratio_viewer.py
```

### Pour une démonstration :
```bash
python quick_demo.py
```

## Dépendances requises

```bash
pip install numpy matplotlib scipy
```

## Structure des fichiers

```
dataset 2/
├── *.mat                              # 22,540 fichiers de données
├── labels.csv                         # Fichier de labels
├── labels.mat                         # Fichier de labels (format .mat)
├── visualize_ratio_vectors.py         # Script de base
├── advanced_ratio_analysis.py         # Analyse avancée
├── interactive_ratio_viewer.py        # Interface interactive
├── quick_demo.py                      # Démonstration rapide
├── ratio_vectors_overlay.png          # Résultat script de base
├── comprehensive_ratio_analysis.png   # Résultat analyse avancée
├── demo_ratio_vectors.png            # Résultat démonstration
├── grid_ratio_vectors.png            # Grille d'échantillons
└── README_visualizations.md          # Ce fichier
```

## Notes techniques

- Les fichiers .mat sont chargés avec `scipy.io.loadmat()`
- Les vecteurs des ratios sont déjà calculés dans les fichiers
- La validation des données gère les valeurs NaN/Inf
- Les couleurs sont automatiquement assignées pour différencier les échantillons
- Les graphiques sont sauvegardés en haute résolution (300 DPI)

## Exemples de commandes

```bash
# Visualisation de base
python visualize_ratio_vectors.py

# Analyse complète
python advanced_ratio_analysis.py

# Démonstration avec exemples sélectionnés
python quick_demo.py

# Interface interactive (suivre les instructions à l'écran)
python interactive_ratio_viewer.py
```

---

**Auteur :** Script généré pour l'analyse des vecteurs des ratios  
**Date :** Juillet 2025  
**Dataset :** 22,540 fichiers .mat avec paramètres gap/longueur variables
