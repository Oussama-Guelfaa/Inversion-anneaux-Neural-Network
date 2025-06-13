# 📊 Analysis Scripts

**Author:** Oussama GUELFAA  
**Date:** 10 - 01 - 2025

## 📖 Description

Ce dossier contient les **scripts d'analyse** utilisés pour étudier les données, diagnostiquer les problèmes et tester les modèles. Ces scripts sont des outils utilitaires pour l'analyse exploratoire et le debugging.

## 📁 Scripts Disponibles

### 🔍 Analyse des Données

#### `data_quantity_analysis.py`
- **Objectif**: Analyse quantitative des données d'entraînement
- **Fonctions**: Distribution des paramètres, statistiques descriptives
- **Usage**: Comprendre la structure du dataset

#### `analyze_data_mismatch.py`
- **Objectif**: Détection des incohérences dans les données
- **Fonctions**: Comparaison train/test, validation cohérence
- **Usage**: Debugging des problèmes de données

#### `analyze_dataset_2D.py` ⭐ **NOUVEAU**
- **Objectif**: Analyse complète du dataset 2D d'anneaux holographiques
- **Fonctions**: Inventaire complet, statistiques, visualisations, détection de trous
- **Usage**: Comprendre la structure et qualité du dataset 2D
- **Outputs**: 8 fichiers (PNG, CSV, rapport texte) dans `outputs_analysis_2D/`

#### `run_complete_dataset_2D_analysis.py` ⭐ **SCRIPT MAÎTRE**
- **Objectif**: Exécution automatique de toute la chaîne d'analyse 2D
- **Fonctions**: Vérifications, analyse, organisation, tests, démonstration
- **Usage**: Une seule commande pour analyse complète (22 secondes)
- **Outputs**: 11 fichiers organisés + documentation complète

### 📈 Analyse des Résultats

#### `analyze_existing_results.py`
- **Objectif**: Analyse des résultats d'entraînement existants
- **Fonctions**: Métriques de performance, comparaisons
- **Usage**: Évaluation post-entraînement

#### `analyze_2percent_noise.py`
- **Objectif**: Analyse spécifique des résultats avec 2% de bruit
- **Fonctions**: Impact du bruit, dégradation performance
- **Usage**: Validation robustesse

### 🧪 Tests de Modèles

#### `test_model_on_real_data.py`
- **Objectif**: Test des modèles sur données réelles
- **Fonctions**: Validation sur données expérimentales
- **Usage**: Validation finale des modèles

## 🚀 Utilisation

### Analyse des Données
```bash
# Analyse quantitative du dataset
python data_quantity_analysis.py

# Détection d'incohérences
python analyze_data_mismatch.py

# Analyse complète du dataset 2D ⭐ NOUVEAU
python analyze_dataset_2D.py

# OU exécution complète automatique ⭐ RECOMMANDÉ
python run_complete_dataset_2D_analysis.py
```

### Analyse des Résultats
```bash
# Analyse des résultats existants
python analyze_existing_results.py

# Analyse spécifique bruit 2%
python analyze_2percent_noise.py
```

### Tests de Validation
```bash
# Test sur données réelles
python test_model_on_real_data.py
```

## 📊 Outputs Générés

### Analyses de Données
- **Statistiques**: Distributions, moyennes, écarts-types
- **Visualisations**: Histogrammes, scatter plots
- **Rapports**: Fichiers CSV avec métriques

### Analyse Dataset 2D ⭐ NOUVEAU
- **Inventaire complet**: 2440 fichiers .mat analysés
- **Statistiques**: 40 gaps × 61 L_ecran, complétude 100%
- **8 fichiers générés**: PNG haute résolution, CSV, rapport texte
- **Visualisations**: Grille d'anneaux, heatmaps, matrices de couverture
- **Recommandations**: Répartition train/val/test optimale

### Analyses de Résultats
- **Métriques**: R², RMSE, MAE par modèle
- **Comparaisons**: Tableaux de performance
- **Graphiques**: Courbes de convergence, résidus

### Tests de Validation
- **Prédictions**: Résultats sur données test
- **Évaluations**: Précision, erreurs
- **Visualisations**: Prédictions vs réalité

## 🔧 Configuration

### Dépendances
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

### Chemins de Données
Les scripts utilisent les chemins relatifs vers :
- `../data_generation/` : Données MATLAB
- `../Reseau_*/results/` : Résultats des réseaux
- `../Reseau_*/models/` : Modèles entraînés

## 📈 Intégration

### Avec les Réseaux Modulaires
Ces scripts peuvent être utilisés pour analyser les résultats des nouveaux réseaux modulaires :

```bash
# Analyser les résultats du réseau avancé
cd analysis_scripts
python analyze_existing_results.py --network ../Reseau_Advanced_Regressor

# Tester un modèle spécifique
python test_model_on_real_data.py --model ../Reseau_Gap_Prediction_CNN/models/best_model.pth
```

### Workflow d'Analyse
1. **Pré-entraînement**: `data_quantity_analysis.py`
2. **Post-entraînement**: `analyze_existing_results.py`
3. **Validation**: `test_model_on_real_data.py`
4. **Debugging**: `analyze_data_mismatch.py`

## 📝 Notes

- **Scripts utilitaires**: Pas d'entraînement, uniquement analyse
- **Flexibles**: Peuvent être adaptés pour nouveaux besoins
- **Indépendants**: Fonctionnent sans les réseaux principaux
- **Documentés**: Commentaires détaillés dans chaque script

**Outils d'analyse pour comprendre et valider les modèles !** 📊
