# 🎯 Réseau Neural Dual Gap + L_ecran - PRECISION 007µm

**Auteur:** Oussama GUELFAA
**Date:** 19-06-2025
**Version:** Organisée et Optimisée

## 📁 Structure du Projet

```
Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25/
├── 📂 src/                     # Code source principal
│   ├── dual_parameter_model.py # Architecture du modèle
│   ├── data_loader.py          # Chargement et préparation des données
│   ├── trainer.py              # Entraînement du modèle
│   ├── run.py                  # Script principal d'exécution
│   ├── demo.py                 # Démonstration du modèle
│   └── data_augmentation_2D.py # Augmentation de données
│
├── 📂 Test_dataset/            # Scripts de test et validation
│   ├── test_dataset_2D.py      # Test sur dataset_2D
│   ├── test_prediction_2400.py # Test sur 2400 échantillons
│   ├── test_nouvelles_donnees.py # Test sur nouvelles données
│   ├── validation_complete.py  # Validation complète
│   └── analyse_comparaison.py  # Analyse comparative
│
├── 📂 docs/                    # Documentation
│   ├── GUIDE_WORKFLOW_PREDICTION.md # Guide détaillé
│   ├── RAPPORT_ANALYSE_COMPARATIVE.md # Rapport d'analyse
│   ├── workflow_prediction_scientifique.tex # Document LaTeX
│   └── *.md                    # Autres documentations
│
├── 📂 models/                  # Modèles entraînés
│   ├── dual_parameter_model.pth # Modèle principal
│   └── *.pkl                   # Scalers sauvegardés
│
├── 📂 data/                    # Données d'entraînement
│   ├── augmented_dataset.npz
│   └── augmented_dataset_advanced.npz
│
├── 📂 results/                 # Résultats des tests
│   ├── test_dataset_2D_results.json
│   ├── test_predictions_2400_samples.json
│   └── *.json                  # Autres résultats
│
├── 📂 plots/                   # Visualisations
│   ├── test_dataset_2D_results.png
│   ├── test_predictions_2400_samples.png
│   └── *.png                   # Autres graphiques
│
├── 📂 config/                  # Configurations
└── 📂 logs/                    # Logs d'entraînement
```

## 🚀 Utilisation

### 1. Entraînement du Modèle
```bash
cd src/
python run.py
```

### 2. Démonstration
```bash
cd src/
python demo.py
```

### 3. Tests sur Nouvelles Données
```bash
cd Test_dataset/
python test_nouvelles_donnees.py
```

### 4. Test sur Dataset_2D
```bash
cd Test_dataset/
python test_dataset_2D.py
```

### 5. Validation Complète
```bash
cd Test_dataset/
python validation_complete.py
```

## 📊 Performances

- **Gap R²:** 0.9952 (99.52%)
- **L_ecran R²:** 0.9888 (98.88%)
- **Gap Accuracy:** 100.0% (±0.01µm)
- **L_ecran Accuracy:** 94.0% (±0.1µm)
- **Validé sur:** 2440 échantillons (dataset_2D)

## 📖 Documentation

Consultez le dossier `docs/` pour :
- Guide détaillé du workflow
- Rapports d'analyse
- Documentation LaTeX scientifique
- Guides d'utilisation

## ✅ Statut

✅ **Modèle validé et prêt pour production**  
✅ **Performances exceptionnelles confirmées**  
✅ **Documentation complète**  
✅ **Structure organisée**
