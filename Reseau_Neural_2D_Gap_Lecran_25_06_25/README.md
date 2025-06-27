# Réseau de Neurones 2D - Prédiction Gap et L_écran

**Auteur:** Oussama GUELFAA  
**Date:** 25/06/2025

## 📋 Description du projet

Ce projet implémente un réseau de neurones pour la prédiction simultanée des paramètres **gap** et **L_écran** à partir de profils d'intensité holographiques 2D.

## 🎯 Objectifs

- Prédiction précise du gap (µm) et de L_écran (µm)
- Utilisation du nouveau dataset 2D divisé (Train/Test)
- Data augmentation sophistiquée basée sur les méthodes éprouvées
- Architecture optimisée pour la régression dual-output

## 📁 Structure du projet

```
Reseau_Neural_2D_Gap_Lecran_25_06_25/
├── README.md                    # Documentation principale
├── config/                      # Configuration
│   └── model_config.yaml       # Paramètres du modèle
├── src/                         # Code source
│   ├── __init__.py
│   ├── data_augmentation_2D.py  # Augmentation des données
│   ├── data_loader.py           # Chargement des données
│   ├── model.py                 # Architecture du réseau
│   ├── trainer.py               # Entraînement
│   └── run.py                   # Script principal
├── data/                        # Données augmentées
├── models/                      # Modèles sauvegardés
├── plots/                       # Graphiques et visualisations
├── results/                     # Résultats d'évaluation
├── docs/                        # Documentation détaillée
└── logs/                        # Logs d'entraînement
```

## 🚀 Utilisation

### 1. Data Augmentation
```bash
cd src
python data_augmentation_2D.py
```

### 2. Entraînement
```bash
cd src
python run.py
```

## 📊 Dataset

- **Source:** `data_generation/dataset_2D_Train/` (11 040 échantillons)
- **Test:** `data_generation/dataset_2D_Test/` (1 840 échantillons)
- **Paramètres:** Gap (0.005-0.400 µm), L_écran (4.0-8.0 µm)

## 🔧 Fonctionnalités

- [x] Structure du projet
- [ ] Data augmentation 2D
- [ ] Architecture du modèle
- [ ] Pipeline d'entraînement
- [ ] Évaluation et métriques
- [ ] Visualisations

## 📈 Performances attendues

- **Objectif R²:** > 0.9 pour les deux paramètres
- **Tolérance gap:** ± 0.01 µm
- **Tolérance L_écran:** ± 0.5 µm

## 📝 Notes

Ce projet s'inspire de l'architecture éprouvée du projet `Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25` en l'adaptant au nouveau dataset 2D divisé.
