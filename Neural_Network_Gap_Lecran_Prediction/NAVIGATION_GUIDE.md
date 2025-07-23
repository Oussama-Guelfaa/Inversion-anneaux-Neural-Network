# 🧭 Guide de Navigation - Neural Network Gap L'écran Prediction

## 🎯 Accès Rapide

### 🚀 **Pour commencer rapidement :**
```bash
cd scripts/training/
python main_training.py          # Entraînement standard
```

### 🧪 **Pour tester :**
```bash
cd scripts/testing/
python quick_test.py             # Test rapide
```

### 📊 **Pour analyser les données PS 3µm :**
```bash
cd data/experimental/
python analyze_ps3um_data.py     # Analyse complète
```

---

## 📁 Structure Détaillée

### 🗂️ **data/** - Toutes les données
```
data/
├── raw/                         # 📦 Données brutes
│   ├── Train/                   # 22,540 profils simulés
│   └── Test/                    # Données de test
├── processed/                   # 🔄 Données préprocessées
│   ├── ps3um_samples_100profiles.csv    # Échantillon rapide
│   ├── ps3um_mean_profile.csv           # Profil moyen
│   └── *.mat, *.npz                     # Formats divers
└── experimental/                # 🔬 Données expérimentales PS 3µm
    ├── analyze_ps3um_data.py            # Script d'analyse principal
    ├── final_ps3um_summary_report.txt   # Rapport final
    └── *.png                            # Visualisations
```

### 🛠️ **scripts/** - Tous les scripts
```
scripts/
├── training/                    # 🏋️ Entraînement
│   ├── main_training.py         # ⭐ Script principal
│   ├── cpu_training.py          # CPU optimisé
│   ├── fast_training.py         # Entraînement rapide
│   └── INDEX.md                 # Guide des scripts
├── testing/                     # 🧪 Tests
│   ├── quick_test.py            # ⭐ Test rapide
│   ├── test_model_on_simulation_data.py
│   └── INDEX.md                 # Guide des tests
├── analysis/                    # 📈 Analyse
│   ├── residual_error_analysis.py
│   └── Analysis_Train_data/     # Analyses avancées
└── preprocessing/               # 🔄 Préprocessing
    ├── preprocess_data.py       # Préprocessing principal
    └── data_augmentation.py     # Augmentation de données
```

### 🤖 **models/** - Modèles et architectures
```
models/
├── architectures/               # 🏗️ Définitions des réseaux
│   └── advanced_neural_network.py
├── saved_models/                # 💾 Modèles entraînés
│   └── *.pt, *.joblib          # Modèles et scalers
└── checkpoints/                 # 📍 Points de sauvegarde
```

### 📊 **results/** - Tous les résultats
```
results/
├── training_results/            # 📈 Historiques d'entraînement
├── predictions/                 # 🎯 Prédictions
│   └── *.csv, *.json          # Résultats de prédiction
└── evaluations/                 # 📏 Métriques d'évaluation
```

### 📈 **visualizations/** - Graphiques
```
visualizations/
├── plots/                       # 📊 Graphiques généraux
├── analysis_charts/             # 📉 Graphiques d'analyse
└── comparisons/                 # ⚖️ Comparaisons visuelles
```

### 🔧 **utils/** - Utilitaires
```
utils/
├── data_loaders/                # 📥 Chargeurs de données
│   ├── data_loader.py          # Chargeur standard
│   └── ultra_fast_data_loader.py # Chargeur optimisé
├── monitoring/                  # 👁️ Monitoring
│   └── visualization_monitoring.py
└── helpers/                     # 🛠️ Fonctions utilitaires
```

### 📝 **reports/** - Documentation
```
reports/
├── technical/                   # 📋 Rapports techniques
├── summaries/                   # 📄 Résumés
└── logs/                        # 📜 Logs d'exécution
```

---

## 🎯 Workflows Recommandés

### 🔬 **Workflow Recherche Complète**
1. **Analyse des données** → `data/experimental/analyze_ps3um_data.py`
2. **Préprocessing** → `scripts/preprocessing/preprocess_data.py`
3. **Entraînement** → `scripts/training/main_training.py`
4. **Test** → `scripts/testing/test_model_on_simulation_data.py`
5. **Analyse des résultats** → `scripts/analysis/`

### ⚡ **Workflow Test Rapide**
1. **Test rapide** → `scripts/testing/quick_test.py`
2. **Données préprocessées** → `data/processed/ps3um_samples_100profiles.csv`
3. **Entraînement rapide** → `scripts/training/fast_training.py`

### 🧪 **Workflow Expérimental**
1. **Données PS 3µm** → `data/experimental/`
2. **Analyse complète** → `data/experimental/analyze_ps3um_data.py`
3. **Test sur expérimental** → `data/experimental/test_ultra_deep_on_experimental.py`

---

## 📚 Fichiers Clés à Connaître

### ⭐ **Essentiels**
- `scripts/training/main_training.py` - Entraînement principal
- `scripts/testing/quick_test.py` - Test rapide
- `data/experimental/analyze_ps3um_data.py` - Analyse PS 3µm
- `data/processed/ps3um_samples_100profiles.csv` - Données test

### 📊 **Données Importantes**
- `data/raw/Train/` - 22,540 profils d'entraînement
- `data/experimental/final_ps3um_summary_report.txt` - Rapport PS 3µm
- `data/processed/ps3um_mean_profile.csv` - Profil moyen expérimental

### 🤖 **Modèles**
- `models/architectures/advanced_neural_network.py` - Architecture principale
- `models/saved_models/` - Modèles entraînés

---

## 🆘 Aide Rapide

### ❓ **Questions Fréquentes**

**Q: Comment démarrer un entraînement ?**
```bash
cd scripts/training/
python main_training.py
```

**Q: Comment tester rapidement ?**
```bash
cd scripts/testing/
python quick_test.py
```

**Q: Où sont les données expérimentales ?**
```bash
cd data/experimental/
ls *.png  # Voir les visualisations
```

**Q: Comment voir les résultats ?**
```bash
cd results/predictions/
ls *.csv  # Voir les prédictions
```

### 🔍 **Recherche de Fichiers**
```bash
# Trouver tous les scripts d'entraînement
find scripts/training/ -name "*.py"

# Trouver toutes les visualisations
find visualizations/ -name "*.png"

# Trouver tous les rapports
find reports/ -name "*.txt" -o -name "*.md"
```

---

## 📞 Contact

**Auteur :** Oussama GUELFAA  
**Email :** guelfaao@gmail.com  
**Date :** 18/07/2025

---

*🎉 Projet maintenant parfaitement organisé et prêt pour le développement !*
