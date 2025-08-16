# Rapport de Recherche - Réseau de Neurones Adaptatif

**Auteur:** Oussama GUELFAA  
**Date:** Janvier 2025  
**Titre:** Développement d'un Réseau de Neurones Adaptatif pour la Généralisation de Données de Simulation vers des Données Expérimentales en Holographie

## 📄 Description du Rapport

Ce rapport LaTeX retrace de manière personnelle et narrative le développement complet d'un modèle de réseau de neurones capable de prédire les paramètres physiques (gap et L_ecran) à partir de profils d'intensité holographiques, en généralisant des données simulées vers des données expérimentales réelles.

## 🎯 Objectif du Travail

Développer un modèle d'apprentissage automatique qui peut :
- S'entraîner sur 22,540 échantillons simulés étiquetés
- Généraliser vers 50 échantillons expérimentaux non étiquetés
- Prédire avec précision les paramètres gap et L_ecran

## 📚 Structure du Rapport

### 1. Introduction
- Présentation du problème d'adaptation de domaine
- Description des données holographiques
- Objectifs et défis du projet

### 2. Préparation des Données
- Chargement et exploration des données
- Troncature des profils (1000 → 750 points)
- Normalisation et alignement des échelles
- Interpolation simulation-expérience
- Visualisations comparatives

### 3. Modèle de Réseau de Neurones
- Première approche naïve (échec)
- Prise de conscience du domain shift
- Architecture DANN (Domain Adversarial Neural Network)
  - Feature Extractor (Conv1D + Dense)
  - Regression Head (gap + L_ecran)
  - Domain Classifier (avec Gradient Reversal Layer)

### 4. Entraînement et Stabilité
- Problèmes initiaux (loss explosion)
- Solutions implémentées :
  - Gradient clipping
  - Planification de λ
  - Réduction de batch size
  - Weight decay
- Courbes de perte et convergence

### 5. Résultats Expérimentaux
- Performance de validation (Gap R² = 0.897)
- Prédictions sur 50 échantillons expérimentaux
- Analyse de correspondance avec simulations
- Statistiques et tendances observées

### 6. Conclusion
- Succès et apprentissages
- Limites identifiées
- Pistes d'amélioration futures

## 📊 Résultats Clés

### Performance du Modèle
- **Gap Parameter:** R² = 0.897, MAE = 0.053 µm
- **L_ecran Parameter:** R² = -0.004, MAE = 1.011 µm
- **Distance moyenne aux simulations:** 0.0019 (très faible)

### Prédictions Expérimentales
- **Gap:** 0.249 à 0.362 µm (CV = 14.1%)
- **L_ecran:** 9.980 à 10.000 µm (CV = 0.1%)
- **Toutes les prédictions dans les plages physiques réalistes**


## 📁 Fichiers Associés

### Document Principal
- `rapport_recherche_reseaux_neurones.tex` - Source LaTeX du rapport
- `rapport_recherche_reseaux_neurones.pdf` - PDF généré (après compilation)

### Scripts de Compilation
- `compile_rapport.py` - Script de compilation automatique
- `README_rapport.md` - Ce fichier de documentation

### Figures Référencées
- `sim_vs_exp_profiles.png` - Comparaison profils simulés vs expérimentaux
- `domain_adaptive_results_fixed.png` - Courbes d'entraînement et validation
- `experimental_vs_closest_simulation_profiles.png` - Correspondances expérience-simulation

## 🔧 Compilation du Rapport

### Prérequis
- Installation LaTeX (TeX Live, MiKTeX, ou MacTeX)
- Python 3.x pour le script de compilation

### Méthodes de Compilation

#### Option 1: Script Automatique (Recommandé)
```bash
python compile_rapport.py
```

#### Option 2: Compilation Manuelle
```bash
pdflatex rapport_recherche_reseaux_neurones.tex
pdflatex rapport_recherche_reseaux_neurones.tex  # Pour les références
```

#### Option 3: Overleaf
1. Télécharger le fichier `.tex`
2. Uploader sur Overleaf avec les images
3. Compiler en ligne

### Installation LaTeX

#### macOS
```bash
brew install --cask mactex
```

#### Ubuntu/Debian
```bash
sudo apt-get install texlive-full
```

#### Windows
Télécharger et installer MiKTeX ou TeX Live

## 📈 Contenu Technique Détaillé

### Architecture du Modèle
```python
# Feature Extractor
Conv1D(1→32, k=5) → ReLU → BatchNorm → MaxPool(2)
Conv1D(32→64, k=3) → ReLU → BatchNorm → MaxPool(2)
Linear(conv_out→256) → ReLU → Dropout(0.3)
Linear(256→128) → ReLU

# Regression Head
Linear(128→2)  # [gap_um, L_um]

# Domain Classifier
GradientReversalLayer(λ)
Linear(128→64) → ReLU → Dropout(0.2)
Linear(64→32) → ReLU
Linear(32→1) → Sigmoid  # 0=sim, 1=exp
```

### Fonction de Perte
```
L_total = L_regression + λ * L_domain
```

### Planification de λ
```python
lambda_param = min(0.1 * (epoch / 10), 0.5)
```

## 🎓 Apprentissages Clés

1. **Importance de l'adaptation de domaine** en apprentissage automatique
2. **Préparation minutieuse des données** (70% du travail)
3. **Stabilisation de l'entraînement** avec techniques de régularisation
4. **Validation croisée** entre domaines pour vérifier la cohérence
5. **Patience et persévérance** face aux échecs initiaux

## 🔮 Perspectives d'Amélioration

- **Plus de données expérimentales** pour validation robuste
- **Pseudo-labelling** des prédictions confiantes
- **Loss MMD** pour meilleure adaptation de domaine
- **Ensemble methods** pour robustesse accrue
- **Analyse d'incertitude** pour quantifier la confiance

## 📞 Contact

**Oussama GUELFAA**  
Email: guelfaao@gmail.com  
Stage de Recherche - Analyse d'Anneaux Holographiques  
Janvier 2025

---

*Ce rapport documente un parcours complet de recherche en apprentissage automatique, de l'identification du problème à la solution finale, en passant par tous les échecs et apprentissages intermédiaires.*
