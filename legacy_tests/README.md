# 🧪 Legacy Tests

**Author:** Oussama GUELFAA  
**Date:** 10 - 01 - 2025

## 📖 Description

Ce dossier contient les **fichiers de tests historiques** générés lors des développements précédents. Ces résultats de tests sont conservés pour référence et validation des améliorations.

## 📁 Contenu

### Résultats de Tests Spécifiques

#### `test_results_gap_0.0250um_L_10.000um.json`
- **Description**: Résultats de test pour gap=0.025µm, L_ecran=10.000µm
- **Contenu**: Métriques de performance, prédictions, erreurs
- **Format**: JSON avec structure détaillée
- **Usage**: Validation sur cas de test spécifique

### Structure des Fichiers JSON
```json
{
  "test_parameters": {
    "gap_target": 0.025,
    "L_ecran_target": 10.000,
    "test_date": "...",
    "model_version": "..."
  },
  "predictions": {
    "gap_predicted": "...",
    "L_ecran_predicted": "...",
    "confidence": "..."
  },
  "metrics": {
    "gap_error": "...",
    "L_ecran_error": "...",
    "relative_error": "...",
    "r2_score": "..."
  }
}
```

## 🔄 Migration vers Nouveaux Tests

Ces tests historiques ont été remplacés par les systèmes de test automatiques des nouveaux réseaux modulaires :

### Nouveaux Tests Automatiques

#### Reseau_Advanced_Regressor/results/
- `evaluation_metrics.json` : Métriques complètes
- `training_history.csv` : Historique détaillé
- `tolerance_analysis.json` : Analyse de tolérance

#### Reseau_Gap_Prediction_CNN/results/
- `training_metrics.json` : Métriques d'entraînement
- `evaluation_report.json` : Rapport d'évaluation

#### Reseau_Overfitting_Test/results/
- `overfitting_test_results.json` : Résultats de validation
- `overfitting_test_summary.csv` : Résumé performance

## 📊 Comparaison des Approches

### Avant (Tests Legacy)
- **Tests manuels** sur cas spécifiques
- **Fichiers isolés** sans contexte
- **Pas de standardisation** des métriques
- **Difficile à reproduire**

### Maintenant (Tests Automatiques)
- **Tests automatiques** lors de l'entraînement
- **Contexte complet** avec configuration
- **Métriques standardisées** et cohérentes
- **Reproductibilité garantie**

## 🔍 Utilisation des Tests Legacy

### Pour Validation Historique
```bash
# Examiner les anciens résultats
cat legacy_tests/test_results_gap_0.0250um_L_10.000um.json

# Comparer avec nouveaux résultats
cat Reseau_Advanced_Regressor/results/evaluation_metrics.json
```

### Pour Benchmarking
- **Référence**: Comparer performance historique vs actuelle
- **Validation**: Vérifier que les améliorations sont réelles
- **Régression**: Détecter d'éventuelles régressions

## 📈 Avantages des Nouveaux Tests

### Automatisation Complète
- **Exécution automatique** lors de l'entraînement
- **Pas d'intervention manuelle** requise
- **Tests systématiques** sur tous les cas

### Couverture Étendue
- **Tests multiples**: Différents paramètres et conditions
- **Métriques complètes**: R², RMSE, MAE, tolérance
- **Validation croisée**: Train, validation, test

### Traçabilité
- **Configuration sauvée** avec chaque test
- **Historique complet** des expériences
- **Reproductibilité** garantie

## 🚀 Nouveaux Tests

Pour exécuter les nouveaux tests automatiques :

```bash
# Tests complets avec métriques
cd Reseau_Advanced_Regressor
python run.py  # Génère automatiquement tous les tests

# Tests de validation spécifiques
cd Reseau_Overfitting_Test
python run.py  # Tests de surapprentissage

# Tests de robustesse
cd Reseau_Noise_Robustness
python run.py  # Tests avec différents niveaux de bruit
```

## 📋 Métriques Modernes

### Métriques Standardisées
- **R² Score**: Coefficient de détermination
- **RMSE**: Erreur quadratique moyenne
- **MAE**: Erreur absolue moyenne
- **Tolérance**: Précision dans seuils définis

### Analyses Avancées
- **Robustesse au bruit**: Performance sous perturbations
- **Généralisation**: Performance sur données non vues
- **Convergence**: Stabilité d'entraînement
- **Overfitting**: Capacité de mémorisation

## ⚠️ Statut

- **Archivés**: Ces tests ne sont plus exécutés
- **Référence**: Conservés pour comparaison historique
- **Remplacés**: Par les tests automatiques des nouveaux réseaux

## 🎯 Recommandations

### Pour Nouveaux Tests
- **Utiliser les réseaux modulaires** avec tests automatiques
- **Configurer via YAML** pour personnalisation
- **Analyser les résultats** dans dossiers `results/`

### Pour Comparaison
- **Comparer métriques** legacy vs modernes
- **Valider améliorations** de performance
- **Documenter évolution** du projet

**Ces tests legacy sont conservés uniquement pour référence historique.** 🧪
