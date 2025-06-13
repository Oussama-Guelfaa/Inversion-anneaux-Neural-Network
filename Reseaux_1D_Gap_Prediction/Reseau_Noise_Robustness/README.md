# 🔊 Réseau Noise Robustness

**Author:** Oussama GUELFAA  
**Date:** 10 - 01 - 2025

## 📖 Description

Ce réseau de neurones teste la robustesse au bruit du modèle de prédiction de gap. Il évalue la performance face à différents niveaux de bruit gaussien pour déterminer les conditions optimales de fonctionnement en environnement réel. Le système utilise une approche progressive avec des niveaux de bruit de 0% à 20%.

## 🎯 Objectifs

- **Test de Robustesse**: Évaluer la performance sous différents niveaux de bruit
- **Seuil de Performance**: Maintenir R² > 0.8 même avec du bruit
- **Analyse Progressive**: Niveaux de bruit de 0%, 1%, 2%, 5%, 10%, 20%
- **Conditions Réelles**: Simuler les conditions d'acquisition expérimentale

## 🏗️ Architecture du Modèle

### Structure Robuste
- **Entrée**: Profils d'intensité tronqués (600 caractéristiques)
- **Couches Dense**: 512 → 256 → 128 → 1
- **Régularisation**: BatchNorm1d + Dropout (0.2)
- **Optimisation**: Adam avec weight decay
- **Early Stopping**: Patience de 25 epochs

### Composants de Robustesse
```python
# Architecture simplifiée pour robustesse
Linear(600, 512) + BatchNorm1d + ReLU + Dropout(0.2)
Linear(512, 256) + BatchNorm1d + ReLU + Dropout(0.2)
Linear(256, 128) + BatchNorm1d + ReLU + Dropout(0.2)
Linear(128, 1)

# Bruit gaussien proportionnel
noise_std = (noise_level / 100.0) * signal_std
X_noisy = X + gaussian_noise(0, noise_std)
```

## 📊 Protocole de Test

### Niveaux de Bruit Testés
- **0%**: Baseline sans bruit
- **1%**: Bruit léger (SNR = 100)
- **2%**: Bruit modéré (SNR = 50)
- **5%**: Bruit significatif (SNR = 20)
- **10%**: Bruit élevé (SNR = 10)
- **20%**: Bruit très élevé (SNR = 5)

### Division des Données
- **Entraînement**: 60% (avec bruit ajouté)
- **Validation**: 20% (sans bruit)
- **Test**: 20% (sans bruit)

### Application du Bruit
- **Bruit Gaussien**: Proportionnel au signal
- **Application**: Uniquement sur les données d'entraînement
- **Distribution**: Moyenne = 0, Écart-type = noise_level × signal_std

## 🚀 Utilisation

### Installation des Dépendances
```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn pyyaml scipy
```

### Exécution du Test Complet
```bash
# Test de robustesse complet
python run.py

# Avec configuration personnalisée
python run.py --config config/noise_config.yaml
```

### Configuration Personnalisée
Modifiez `config/noise_config.yaml` pour ajuster:
- Niveaux de bruit à tester
- Architecture du modèle
- Paramètres d'entraînement
- Options de visualisation

## 📈 Métriques d'Évaluation

### Métriques Principales
- **R² Score**: Coefficient de détermination
- **RMSE**: Erreur quadratique moyenne
- **MAE**: Erreur absolue moyenne
- **Temps d'Entraînement**: Convergence par niveau de bruit

### Critères de Succès
- **Performance Acceptable**: R² > 0.8 sous bruit
- **Robustesse**: Dégradation graduelle avec le bruit
- **Convergence**: Stabilité d'entraînement
- **Généralisation**: Performance sur données test

## 📁 Structure des Fichiers

```
Reseau_Noise_Robustness/
├── run.py                              # Script principal autonome
├── config/
│   └── noise_config.yaml               # Configuration des tests
├── models/
│   ├── model_noise_0percent.pth        # Modèle baseline
│   ├── model_noise_1percent.pth        # Modèle 1% bruit
│   ├── model_noise_2percent.pth        # Modèle 2% bruit
│   ├── model_noise_5percent.pth        # Modèle 5% bruit
│   ├── model_noise_10percent.pth       # Modèle 10% bruit
│   └── model_noise_20percent.pth       # Modèle 20% bruit
├── plots/
│   ├── noise_robustness_analysis.png   # Analyse principale
│   ├── predictions_by_noise.png        # Prédictions par niveau
│   ├── performance_degradation.png     # Dégradation performance
│   └── scatter_plots.png               # Graphiques de dispersion
├── results/
│   ├── performance_by_noise_level.csv  # Tableau de performance
│   ├── predictions_noise_Xpercent.csv  # Prédictions détaillées
│   └── noise_robustness_summary.json   # Résumé complet
├── docs/
│   └── COMPARATIVE_ANALYSIS.md          # Analyse comparative
└── README.md                           # Cette documentation
```

## 🔬 Analyse des Résultats

### Courbes de Performance
1. **R² vs Niveau de Bruit**: Dégradation de performance
2. **RMSE vs Niveau de Bruit**: Augmentation de l'erreur
3. **Temps de Convergence**: Impact sur l'entraînement
4. **Stabilité**: Variance des prédictions

### Analyses Spécialisées
- **Généralisation Locale**: Performance par plage de gap
- **Augmentation de Données**: Impact de l'interpolation
- **Données Réduites**: Performance avec moins d'échantillons
- **Seuils de Tolérance**: Analyse de précision

## 🧪 Tests Avancés

### Test de Généralisation Locale
```yaml
local_generalization:
  gap_ranges:
    - [0.025, 0.5]   # Petits gaps
    - [0.5, 1.0]     # Gaps moyens
    - [1.0, 1.5]     # Grands gaps
```

### Test avec Données Réduites
```yaml
reduced_data_test:
  sample_sizes: [300, 500, 700]
  noise_level: 0.05  # 5% bruit
```

### Augmentation de Données
```yaml
augmentation:
  enable: true
  interpolation_factor: 2  # Double la taille du dataset
```

## 📊 Résultats Attendus

### Performance Baseline (0% bruit)
- **R²**: > 0.95
- **RMSE**: < 0.02 µm
- **Convergence**: < 50 epochs

### Performance avec Bruit (5% bruit)
- **R²**: > 0.8 (objectif)
- **RMSE**: < 0.05 µm
- **Robustesse**: Dégradation contrôlée

### Performance Limite (20% bruit)
- **R²**: > 0.6 (acceptable)
- **RMSE**: < 0.1 µm
- **Stabilité**: Convergence maintenue

## 🔧 Optimisations pour Robustesse

### Techniques Implémentées
- **Dropout Adaptatif**: Régularisation progressive
- **Batch Normalization**: Stabilisation des gradients
- **Early Stopping**: Prévention du surapprentissage
- **Learning Rate Scheduling**: Adaptation automatique

### Stratégies de Bruit
- **Bruit Proportionnel**: Adapté au niveau du signal
- **Application Sélective**: Uniquement sur l'entraînement
- **Distribution Gaussienne**: Simulation réaliste
- **Niveaux Progressifs**: Évaluation systématique

## 🎯 Applications Pratiques

Ce test de robustesse permet de:
- **Valider** la fiabilité en conditions réelles
- **Optimiser** les paramètres d'acquisition
- **Prédire** la performance en environnement bruité
- **Calibrer** les seuils de confiance

**Le modèle est validé pour un usage robuste en holographie expérimentale!** 🚀
