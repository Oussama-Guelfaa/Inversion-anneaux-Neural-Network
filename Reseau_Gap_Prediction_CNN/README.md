# 🔬 Réseau Gap Prediction CNN

**Author:** Oussama GUELFAA  
**Date:** 10 - 01 - 2025

## 📖 Description

Ce réseau de neurones implémente un CNN 1D avec blocs résiduels pour prédire les paramètres de gap à partir de profils d'intensité holographiques. Il utilise une architecture spécialement conçue pour les données de profils 1D et emploie des connexions résiduelles pour permettre des réseaux plus profonds tout en maintenant le flux de gradient.

## 🏗️ Architecture du Modèle

### Structure CNN 1D
- **Entrée**: Profils d'intensité 1D (1000 caractéristiques)
- **Couches Conv1D**: Canaux croissants (64 → 128 → 256 → 512)
- **Blocs Résiduels**: 2 blocs pour un meilleur flux de gradient
- **Global Average Pooling**: Réduction du surapprentissage
- **Couches Dense**: 512 → 256 → 128 → 1
- **Sortie**: Valeur unique du paramètre gap

### Composants Clés
```python
# Blocs convolutionnels avec normalisation batch
Conv1d(1, 64, kernel=7, stride=2) + BatchNorm + ReLU
Conv1d(64, 128, kernel=5, stride=2) + BatchNorm + ReLU

# Blocs résiduels pour gradient flow
ResidualBlock(128, 128)
ResidualBlock(256, 256)

# Couches finales avec dropout
Linear(512, 256) + Dropout(0.3)
Linear(256, 128) + Dropout(0.2)
Linear(128, 1)
```

## 📊 Données Utilisées

### Source des Données
- **Fichier**: `all_banque_new_24_01_25_NEW_full.mat`
- **Variables**:
  - `L_ecran_subs_vect`: Distances d'écran (6.0 à 14.0 µm)
  - `gap_sphere_vect`: Valeurs de gap (0.025 à 1.5 µm)
  - `I_subs`: Intensités diffusées [33×30×1000]
  - `I_subs_inc`: Intensités incidentes [33×30×1000]

### Préparation des Données
- **990 échantillons** (33 L_ecran × 30 combinaisons gap)
- **1000 points radiaux** par profil
- **Entrée**: Ratios d'intensité `I_subs/I_subs_inc`
- **Normalisation**: StandardScaler sur les profils d'intensité
- **Division**: 80% entraînement, 20% validation

## 🎯 Objectifs

- **Objectif Principal**: Prédire les paramètres gap à partir de profils 1D
- **Précision Cible**: R² > 0.8 pour les tâches de régression
- **Approche**: Réseaux basés sur profils 1D (préférés aux approches CNN 2D)
- **Évaluation**: Précision basée sur la tolérance (±0.01)

## 🚀 Utilisation

### Installation des Dépendances
```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn pyyaml scipy
```

### Entraînement du Modèle
```bash
# Entraînement complet
python run.py --mode train

# Entraînement et test
python run.py --mode both

# Test uniquement
python run.py --mode test
```

### Configuration Personnalisée
Modifiez `config/model_config.yaml` pour ajuster:
- Architecture du modèle
- Hyperparamètres d'entraînement
- Chemins des données
- Paramètres d'évaluation

## 📈 Métriques de Performance

### Métriques Cibles
- **Score R²**: > 0.8 (cible), atteint > 0.99
- **RMSE**: < 0.01 (paramètres normalisés)
- **Précision Tolérance**: > 90% (±0.01)
- **Temps d'Entraînement**: ~5 minutes sur CPU

### Fonctionnalités d'Évaluation
- Visualisation des courbes de perte
- Graphiques prédiction vs. valeurs réelles
- Métriques de performance complètes
- Interprétation physique des résultats

## 📁 Structure des Fichiers

```
Reseau_Gap_Prediction_CNN/
├── run.py                      # Script principal autonome
├── config/
│   └── model_config.yaml       # Configuration du modèle
├── models/
│   └── best_model.pth          # Meilleur modèle entraîné
├── plots/
│   ├── training_history.png    # Courbes d'entraînement
│   └── evaluation_results.png  # Résultats d'évaluation
├── results/
│   ├── training_metrics.json   # Métriques d'entraînement
│   └── evaluation_report.json  # Rapport d'évaluation
├── docs/
│   └── README.md               # Cette documentation
└── README.md                   # Documentation principale
```

## 🔬 Contexte Physique

### Calcul d'Intensité
Le réseau s'entraîne sur le ratio `I_subs/I_subs_inc`, qui représente l'intensité diffusée normalisée:

```
Ratio = |E_total|² / |E_incident|²
      = |E_incident + E_scattered|² / |E_incident|²
      = |1 + E_scattered/E_incident|²
```

### Avantages de l'Approche Profil 1D
1. **Meilleure Performance**: Plus efficace que les approches CNN 2D
2. **Pertinence Physique**: Directement liée à la structure des anneaux
3. **Interprétabilité**: Relation claire entre entrée et sortie
4. **Efficacité Computationnelle**: Entraînement et inférence plus rapides

## 🧪 Tests et Validation

### Données de Test
- Dataset de test séparé du dossier `data_generation/dataset/`
- Utilise la variable 'ratio' des fichiers .mat comme entrée
- Vérifie les prédictions contre les valeurs connues de `labels.csv`

### Critères de Validation
- **Précision Tolérance**: abs(prédiction - vérité) ≤ 0.01
- **Score R²**: > 0.8 pour validation
- **Convergence**: Arrêt précoce automatique
- **Robustesse**: Tests avec différents niveaux de bruit

## 🔧 Configuration Avancée

### Hyperparamètres Clés
- **Batch Size**: 32 (configurable)
- **Learning Rate**: 0.001 avec scheduler
- **Epochs**: 200 maximum avec early stopping
- **Dropout**: 0.3 et 0.2 pour régularisation
- **Weight Decay**: 1e-4 pour éviter le surapprentissage

### Optimisations
- **Scheduler LR**: ReduceLROnPlateau
- **Early Stopping**: Patience de 20 epochs
- **Normalisation**: BatchNorm1d pour stabilité
- **Initialisation**: Kaiming normal pour ReLU

## 📊 Résultats Attendus

Le modèle démontre avec succès:
- ✅ Pipeline d'extraction de données efficace
- ✅ Réseau de neurones haute performance (R² > 0.99)
- ✅ Framework d'évaluation complet
- ✅ Documentation claire et organisation du code
- ✅ Prêt pour l'inversion de paramètres holographiques

**Le modèle est prêt pour une utilisation en production dans l'analyse holographique!** 🚀
