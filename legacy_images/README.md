# 🖼️ Legacy Images

**Author:** Oussama GUELFAA  
**Date:** 10 - 01 - 2025

## 📖 Description

Ce dossier contient les **images et graphiques historiques** générés lors des développements précédents. Ces visualisations sont conservées pour référence et comparaison avec les nouveaux résultats.

## 📁 Contenu

### Images de Comparaison de Données

#### `data_distribution_comparison.png`
- **Description**: Comparaison des distributions de données
- **Contenu**: Histogrammes des paramètres gap et L_ecran
- **Usage**: Analyse exploratoire des données

#### `profile_comparison_similar_gaps.png`
- **Description**: Comparaison de profils avec gaps similaires
- **Contenu**: Superposition de profils d'intensité
- **Usage**: Validation de la cohérence des données

### Images de Tests de Modèles

#### `test_prediction_gap_0.0250um_L_10.000um.png`
- **Description**: Test de prédiction pour gap=0.025µm, L=10.000µm
- **Contenu**: Comparaison prédiction vs réalité
- **Usage**: Validation sur cas spécifique

## 🔄 Migration vers Nouveaux Réseaux

Ces images historiques ont été remplacées par les visualisations automatiques des nouveaux réseaux modulaires :

### Nouvelles Visualisations Automatiques

#### Reseau_Advanced_Regressor/plots/
- `training_curves.png` : Courbes d'entraînement
- `predictions_scatter.png` : Prédictions vs réalité
- `tolerance_analysis.png` : Analyse de tolérance

#### Reseau_Gap_Prediction_CNN/plots/
- `training_history.png` : Historique d'entraînement
- `evaluation_results.png` : Résultats d'évaluation

#### Reseau_Noise_Robustness/plots/
- `noise_robustness_analysis.png` : Analyse robustesse
- `performance_degradation.png` : Dégradation performance

## 📊 Comparaison Historique

### Avant (Images Legacy)
- **Génération manuelle** des graphiques
- **Fichiers dispersés** à la racine
- **Pas de standardisation** des formats
- **Difficile à retrouver** et organiser

### Maintenant (Réseaux Modulaires)
- **Génération automatique** dans chaque réseau
- **Organisation claire** dans dossiers `plots/`
- **Formats standardisés** et nommage cohérent
- **Facilement accessible** et reproductible

## 🔍 Utilisation des Images Legacy

### Pour Comparaison Historique
```bash
# Voir les anciennes visualisations
ls legacy_images/

# Comparer avec nouvelles visualisations
ls Reseau_Advanced_Regressor/plots/
```

### Pour Référence
- **Validation**: Comparer anciennes et nouvelles métriques
- **Évolution**: Voir l'amélioration des résultats
- **Documentation**: Référence pour publications

## ⚠️ Statut

- **Archivées**: Ces images ne sont plus générées
- **Référence**: Conservées pour comparaison historique
- **Remplacées**: Par les visualisations automatiques des nouveaux réseaux

## 🚀 Nouvelles Visualisations

Pour générer de nouvelles visualisations :

```bash
# Générer automatiquement toutes les visualisations
cd Reseau_Advanced_Regressor
python run.py  # Génère automatiquement dans plots/

# Ou pour un réseau spécifique
cd Reseau_Gap_Prediction_CNN
python run.py --mode train  # Génère plots d'entraînement
```

## 📈 Avantages des Nouvelles Visualisations

### Automatisation
- **Génération automatique** lors de l'entraînement
- **Pas d'intervention manuelle** requise
- **Cohérence** garantie

### Organisation
- **Dossiers dédiés** `plots/` dans chaque réseau
- **Nommage standardisé** et prévisible
- **Facilement archivable** avec le réseau

### Qualité
- **Résolution élevée** (300 DPI)
- **Formats optimisés** (PNG, PDF)
- **Légendes complètes** et informatives

**Ces images legacy sont conservées uniquement pour référence historique.** 📸
