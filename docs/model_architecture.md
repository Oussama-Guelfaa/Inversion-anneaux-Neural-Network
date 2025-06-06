# Neural Network pour l'Inversion de Paramètres Holographiques

**Auteur:** Oussama GUELFAA  
**Date:** 05 - 06 - 2025

Ce projet implémente un réseau de neurones pour prédire les paramètres L_ecran et gap à partir de profils radiaux d'intensité extraits du dataset `all_banque_new_24_01_25_NEW_full.mat`.

## 📁 Structure du Projet

```
Neural_Network/
├── extract_training_data.py      # Extraction et préparation des données
├── train_new_dataset.py          # Entraînement du réseau de neurones
├── evaluate_new_model.py         # Évaluation des performances
├── processed_data/               # Données extraites et organisées
│   ├── training_data.npz         # Dataset complet (990 échantillons)
│   ├── intensity_profiles_full.csv # Tous les profils (990×1000)
│   ├── parameters.csv            # Paramètres [L_ecran, gap]
│   └── data_visualization.png    # Visualisation des données
├── models/                       # Modèles entraînés
│   ├── ring_regressor_new.pth    # Meilleur modèle sauvegardé
│   └── scalers_new.npz          # Normalisateurs
└── plots/                        # Résultats d'évaluation
    ├── evaluation_results.png    # Graphiques de performance
    └── predictions_results.csv   # Prédictions vs vraies valeurs
```

## 🚀 Utilisation

### 1. Extraction des données
```bash
python extract_training_data.py
```

### 2. Entraînement du modèle
```bash
python train_new_dataset.py
```

### 3. Évaluation des performances
```bash
python evaluate_new_model.py
```

## 📊 Dataset

- **Source:** `all_banque_new_24_01_25_NEW_full.mat`
- **Échantillons:** 990 (33 L_ecran × 30 gap)
- **Features:** Profils radiaux 1D (1000 points)
- **Targets:** [L_ecran, gap]
- **Plages:**
  - L_ecran: 6.0 à 14.0 µm
  - gap: 0.025 à 1.5 µm

## 🧠 Architecture du Réseau

- **Entrée:** 1000 points (profil radial)
- **Couches cachées:** [512, 256, 128]
- **Sortie:** 2 paramètres [L_ecran, gap]
- **Techniques:** Blocs résiduels, BatchNorm, Dropout
- **Optimiseur:** Adam avec scheduler adaptatif

## 📈 Performances

Le modèle atteint d'excellentes performances :
- **R² Score:** > 0.99 pour L_ecran et gap
- **RMSE:** < 0.01 pour les paramètres normalisés
- **Temps d'entraînement:** ~5 minutes sur CPU

## 🔧 Troubleshooting

1. **Erreur de mémoire** : Réduire `batch_size`
2. **Pas de GPU** : Le modèle fonctionne aussi sur CPU
3. **Données manquantes** : Vérifier les chemins vers les fichiers `.mat`
4. **Overfitting** : Augmenter le dropout ou réduire la complexité
