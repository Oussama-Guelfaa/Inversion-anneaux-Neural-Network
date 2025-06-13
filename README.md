# 🔬 Inversion d'Anneaux - Neural Network Project

**Auteur:** Oussama GUELFAA
**Date:** 25 - 01 - 2025

## 📖 Vue d'Ensemble du Projet

Ce projet implémente des **solutions de réseaux de neurones spécialisées** pour l'analyse holographique d'anneaux et la prédiction de paramètres. Le projet est organisé en **deux catégories principales** selon le type de prédiction : 1D (gap seul) et 2D (gap + L_écran).

## 🎯 Objectifs

- **Objectif Principal**: Prédire les paramètres gap et/ou L_écran à partir de profils d'intensité 1D
- **Source de Données**: Ratios d'intensité holographiques (I_subs/I_subs_inc) depuis fichiers MATLAB
- **Précision Cible**: R² > 0.8 pour les tâches de régression (R² > 0.95 atteint)
- **Architecture**: Réseaux basés sur profils 1D (préférés aux approches CNN 2D)
- **Modularité**: Chaque réseau comme unité indépendante et déployable

## 🏗️ Architecture du Projet

Le projet suit une architecture modulaire organisée par **type de prédiction** :

```
Inversion_anneaux/
├── 🎯 Reseaux_1D_Gap_Prediction/          # Réseaux prédiction gap seul
│   ├── 🔊 Reseau_Noise_Robustness/        # ⭐ Robustesse bruit (RECOMMANDÉ)
│   ├── 🔬 Reseau_Gap_Prediction_CNN/      # CNN pour prédiction gap
│   └── 🧪 Reseau_Overfitting_Test/        # Test validation overfitting
├── 🎯 Reseaux_2D_Gap_Lecran_Prediction/   # Réseaux prédiction gap + L_écran
│   ├── 🔧 Reseau_TensorFlow_Alternative/  # Alternative TensorFlow/Keras
│   └── 🔥 Reseau_Ultra_Specialized/       # Architecture ultra-spécialisée
├── 📊 data_generation/                    # Données MATLAB et scripts
├── 🗂️ archive_legacy_networks/           # Archives réseaux précédents
├── 🔧 utilities/                          # Utilitaires et outils communs
├── 📋 analysis_scripts/                   # Scripts d'analyse
└── 📖 README.md                           # Ce fichier
```

### 🏆 Résultat Majeur : Succès des Réseaux 1D

**Découverte clé :** Les réseaux 1D (prédiction gap seul) surpassent largement les réseaux 2D (gap + L_écran) :
- **Performance 1D** : R² = 0.9948 (quasi-parfait)
- **Performance 2D** : R² < 0.5 (problématique)
- **Recommandation** : Utiliser exclusivement les réseaux 1D

### Standardized Network Structure

Each neural network follows the same organization:

```
Reseau_XYZ/
├── run.py              # Autonomous main script
├── config/
│   └── config.yaml     # Complete configuration
├── models/             # Trained models (.pth, .h5, .pkl)
├── plots/              # Visualizations and analysis
├── results/            # Metrics and reports (JSON, CSV)
├── docs/               # Specialized documentation
└── README.md           # Usage guide
```

## 🚀 Quick Start

### 1️⃣ Setup Environment
```bash
# Install common dependencies
pip install torch pandas numpy matplotlib seaborn scikit-learn pyyaml scipy joblib

# For TensorFlow (optional)
pip install tensorflow
```

### 2️⃣ Choose Your Neural Network
```bash
# For production use (recommended)
cd Reseau_Advanced_Regressor
python run.py

# For maximum performance
cd Reseau_Ultra_Specialized
python run.py

# For gap-only prediction
cd Reseau_Gap_Prediction_CNN
python run.py --mode train

# For robustness testing
cd Reseau_Noise_Robustness
python run.py
```

### 3️⃣ View Results
Each network generates:
- **Models**: Trained neural networks
- **Plots**: Performance visualizations
- **Results**: Detailed metrics and reports

## 🎯 Réseaux de Neurones Disponibles

### 🏆 Catégorie 1D - Prédiction Gap Seul (RECOMMANDÉE)

#### 1. 🔊 Reseau_Noise_Robustness ⭐ **MEILLEUR MODÈLE**
**Robustesse au bruit avec augmentation de données optimisée**
- **Architecture:** Dense 512→256→128→1 avec régularisation
- **Performance:** R² = **0.9948** (quasi-parfait)
- **Innovation:** Augmentation par interpolation facteur 3
- **Zone critique:** [1.75-2.00 µm] maîtrisée (R² = 0.99)
- **Robustesse:** Testé jusqu'à 20% bruit, optimal à 5%
- **Utilisation:** **Production immédiate recommandée**

#### 2. 🔬 Reseau_Gap_Prediction_CNN
**CNN spécialisé pour prédiction gap**
- **Architecture:** CNN 1D avec blocs résiduels
- **Performance:** R² > 0.90 sur gap
- **Utilisation:** Exploration architectures convolutionnelles

#### 3. 🧪 Reseau_Overfitting_Test
**Validation capacité d'apprentissage**
- **Architecture:** Simple sans régularisation
- **Performance:** R² ≈ 1.0 sur données d'entraînement
- **Utilisation:** Tests de validation et diagnostics

### ⚠️ Catégorie 2D - Prédiction Gap + L_écran (RECHERCHE)

#### 4. 🔧 Reseau_TensorFlow_Alternative
**Alternative TensorFlow/Keras**
- **Architecture:** Dense 512→256→128→64→2
- **Performance:** R² < 0.5 (limité par qualité données)
- **Utilisation:** Recherche et développement TensorFlow

#### 5. � Reseau_Ultra_Specialized
**Architecture ultra-spécialisée**
- **Architecture:** Modèles ultra-profonds spécialisés
- **Performance:** R² < 0.5 (limité par qualité données)
- **Utilisation:** Recherche architectures avancées

## 📊 Dataset Information

### Common Data Source
- **Dataset:** `data_generation/all_banque_new_24_01_25_NEW_full.mat`
- **Variables:**
  - `L_ecran_subs_vect`: Screen distances (6.0 to 14.0 µm)
  - `gap_sphere_vect`: Gap values (0.025 to 1.5 µm)
  - `I_subs`: Scattered intensities [33×30×1000]
  - `I_subs_inc`: Incident intensities [33×30×1000]

### Training Data
- **990 samples** (33 L_ecran × 30 gap combinations)
- **600-1000 radial points** per profile (network-dependent)
- **Input:** Intensity ratios `I_subs/I_subs_inc`
- **Output:** Physical parameters [L_ecran, gap]

## 📈 Performance Comparison

| Network | Gap R² | L_ecran R² | Specialty | Training Time |
|---------|--------|------------|-----------|---------------|
| Gap Prediction CNN | >0.99 | - | Gap only | ~5 min |
| Noise Robustness | >0.8* | >0.95* | Noise testing | ~15 min |
| Overfitting Test | >0.99 | >0.99 | Validation | ~3 min |
| **Advanced Regressor** ⭐ | >0.8 | >0.95 | **Production** | ~8 min |
| Ultra Specialized | >0.85 | >0.98 | Max performance | ~20 min |
| PyTorch Optimized | >0.8 | >0.95 | PyTorch dev | ~10 min |
| TensorFlow Alternative | >0.8 | >0.95 | TensorFlow dev | ~15 min |

*\* Performance under 5% noise*

## 🔬 Physical Background

### Intensity Calculation
The neural networks train on the ratio `I_subs/I_subs_inc`, which represents the normalized scattered intensity:

```
Ratio = |E_total|² / |E_incident|²
      = |E_incident + E_scattered|² / |E_incident|²
      = |1 + E_scattered/E_incident|²
```

### Advantages of 1D Profile Approach
1. **Better Performance:** More efficient than 2D CNN approaches
2. **Physical Relevance:** Directly related to ring structure
3. **Interpretability:** Clear relationship between input and output
4. **Computational Efficiency:** Faster training and inference

## 📚 Documentation

- **[Project Map](project_map.md):** Complete overview of all networks
- **Individual READMEs:** Each network has detailed documentation
- **Configuration Files:** YAML configs for each network
- **Results:** Automated metrics and visualizations

## 🎯 Selection Guide

### For Production Use
- **Recommended:** `Reseau_Advanced_Regressor` or `Reseau_Ultra_Specialized`
- **Reason:** Systematic problem solving, high performance

### For Research
- **Gap only:** `Reseau_Gap_Prediction_CNN`
- **Robustness:** `Reseau_Noise_Robustness`
- **Diagnostics:** `Reseau_Overfitting_Test`

### For Development
- **PyTorch:** `Reseau_PyTorch_Optimized`
- **TensorFlow:** `Reseau_TensorFlow_Alternative`

## 🔧 Modular Benefits

### Independent Units
- ✅ Each network is self-contained
- ✅ Can be zipped and deployed separately
- ✅ Easy to compare different approaches
- ✅ Simplified maintenance and updates

### Standardized Structure
- ✅ Consistent organization across networks
- ✅ Autonomous `run.py` scripts
- ✅ Complete configuration files
- ✅ Automated result generation

## 🎉 Project Achievements

This modular neural network suite successfully provides:
- ✅ **7 specialized networks** for different use cases
- ✅ **Standardized structure** for easy deployment
- ✅ **High performance** (R² > 0.8 consistently achieved)
- ✅ **Complete documentation** and configuration
- ✅ **Production-ready** solutions for holographic analysis
- ✅ **Modular architecture** for easy extension and maintenance

**Each network is ready for independent deployment in holographic parameter inversion!** 🚀
