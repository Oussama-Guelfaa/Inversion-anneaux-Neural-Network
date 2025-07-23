# 🧠 Neural Network Gap L'écran Prediction

## 🎯 Objectif
Prédiction des paramètres gap et L'écran à partir de profils d'intensité holographique utilisant des réseaux de neurones profonds.

## ⚡ Démarrage Rapide

### 🚀 **Entraînement**
```bash
cd scripts/training/
python main_training.py
```

### 🧪 **Test Rapide**
```bash
cd scripts/testing/
python quick_test.py
```

### 📊 **Analyse des Données PS 3µm**
```bash
cd data/experimental/
python analyze_ps3um_data.py
```

## 📁 Structure du Projet

```
Neural_Network_Gap_Lecran_Prediction/
├── 📁 data/                    # Données d'entraînement et de test
│   ├── 📂 raw/                 # Données brutes (22,540 profils Train/)
│   ├── 📂 processed/           # Données préprocessées et échantillons
│   └── 📂 experimental/        # Données expérimentales PS 3µm (6,596 profils)
├── 📁 scripts/                 # Scripts principaux organisés par fonction
│   ├── 📂 training/            # 7 scripts d'entraînement (main, cpu, fast...)
│   ├── 📂 analysis/            # Scripts d'analyse des données
│   ├── 📂 preprocessing/       # 4 scripts de préprocessing
│   └── 📂 testing/             # 5 scripts de test et validation
├── 📁 models/                  # Modèles et architectures
│   ├── 📂 architectures/       # Définitions des réseaux de neurones
│   ├── 📂 saved_models/        # Modèles entraînés (.pt, .joblib)
│   └── 📂 checkpoints/         # Points de sauvegarde
├── 📁 results/                 # Résultats d'entraînement et prédictions
│   ├── 📂 training_results/    # Historiques d'entraînement détaillés
│   ├── 📂 predictions/         # Prédictions (.csv, .json)
│   └── 📂 evaluations/         # Métriques d'évaluation
├── 📁 visualizations/          # Graphiques et visualisations
│   ├── 📂 plots/               # Graphiques généraux (.png)
│   ├── 📂 analysis_charts/     # Graphiques d'analyse spécialisés
│   └── 📂 comparisons/         # Comparaisons visuelles
├── 📁 utils/                   # Utilitaires et fonctions communes
│   ├── 📂 data_loaders/        # 3 chargeurs de données optimisés
│   ├── 📂 monitoring/          # Outils de monitoring d'entraînement
│   └── 📂 helpers/             # Fonctions utilitaires diverses
└── 📁 reports/                 # Rapports et documentation
    ├── 📂 technical/           # Rapports techniques (.md)
    ├── 📂 summaries/           # Résumés d'analyse (.txt)
    └── 📂 logs/                # Logs d'exécution
```

## 📊 Données Disponibles

### 🔬 **Données Expérimentales PS 3µm**
- **Localisation :** `data/experimental/`
- **Contenu :** 6,596 profils d'intensité holographique
- **Qualité :** Excellente (corrélation temporelle >99%)
- **Anneaux détectés :** 5 anneaux bien définis
- **Résolution :** 0.058 µm/point sur 7 µm

### 🎯 **Données Simulées**
- **Localisation :** `data/raw/Train/`
- **Contenu :** 22,540 profils simulés
- **Paramètres :** Gap (0.005-0.7 µm), L'écran (8-12 µm)
- **Format :** Fichiers .mat individuels

### ⚡ **Données Préprocessées**
- **Localisation :** `data/processed/`
- **Échantillons :** 100 profils représentatifs
- **Formats :** CSV, MAT, NPZ
- **Usage :** Tests rapides et développement

## 🧠 Modèles et Architectures

### 🏗️ **Architecture Principale**
- **Type :** Réseau dense multi-couches
- **Couches :** 512 → 256 → 128 → 64 → 2
- **Activation :** ReLU + Dropout (0.2)
- **Optimiseur :** Adam
- **Loss :** MSE avec pondération

### 🎯 **Performances**
- **Précision gap :** ±0.007 µm
- **Précision L'écran :** ±0.5 µm
- **R² sur simulation :** >0.95
- **Stabilité expérimentale :** Excellente

## 🚀 Workflows Recommandés

### 🔬 **Recherche Complète**
1. **Analyse** → `data/experimental/analyze_ps3um_data.py`
2. **Préprocessing** → `scripts/preprocessing/preprocess_data.py`
3. **Entraînement** → `scripts/training/main_training.py`
4. **Évaluation** → `scripts/testing/test_model_on_simulation_data.py`

### ⚡ **Développement Rapide**
1. **Test rapide** → `scripts/testing/quick_test.py`
2. **Entraînement rapide** → `scripts/training/fast_training.py`
3. **Données échantillon** → `data/processed/ps3um_samples_100profiles.csv`

### 🧪 **Validation Expérimentale**
1. **Données PS 3µm** → `data/experimental/`
2. **Test expérimental** → `data/experimental/test_ultra_deep_on_experimental.py`
3. **Comparaison** → `data/experimental/compare_predicted_vs_experimental.py`

## 📚 Documentation

### 📖 **Guides Principaux**
- `NAVIGATION_GUIDE.md` - Guide de navigation complet
- `data/INDEX.md` - Guide des données
- `scripts/training/INDEX.md` - Guide d'entraînement
- `scripts/testing/INDEX.md` - Guide de test

### 📊 **Rapports Techniques**
- `data/experimental/final_ps3um_summary_report.txt` - Analyse PS 3µm
- `reports/technical/` - Rapports techniques détaillés
- `reports/summaries/` - Résumés d'analyse

## 🛠️ Installation et Prérequis

### 📦 **Dépendances**
```bash
pip install torch numpy matplotlib pandas scipy scikit-learn
```

### 🔧 **Configuration**
- Python 3.8+
- PyTorch (CPU/GPU compatible)
- 8GB RAM minimum (16GB recommandé)
- Espace disque : 5GB pour toutes les données

## 🎯 Utilisation Avancée

### 🏋️ **Entraînement Personnalisé**
```bash
cd scripts/training/
python main_training.py --epochs 100 --batch_size 64 --lr 0.001
```

### 📊 **Analyse Personnalisée**
```bash
cd scripts/analysis/
python residual_error_analysis.py --model_path ../models/saved_models/
```

### 🔄 **Préprocessing Avancé**
```bash
cd scripts/preprocessing/
python data_augmentation.py --factor 3 --noise 0.02
```

## 📈 Résultats et Performances

### 🎯 **Métriques Clés**
- **R² Gap :** 0.987 ± 0.003
- **R² L'écran :** 0.994 ± 0.002
- **MAE Gap :** 0.0045 µm
- **MAE L'écran :** 0.23 µm
- **Temps d'entraînement :** ~2h (CPU), ~30min (GPU)

### 📊 **Validation Expérimentale**
- **Cohérence avec PS 3µm :** Excellente
- **Stabilité temporelle :** >99% corrélation
- **Robustesse au bruit :** Validée jusqu'à 5%

## 🤝 Contribution

### 📝 **Standards de Code**
- Documentation complète des fonctions
- Tests unitaires pour nouvelles fonctionnalités
- Respect de la structure de dossiers

### 🔄 **Workflow Git**
- Commits fréquents et descriptifs
- Branches par fonctionnalité
- Pull requests avec review

## 👨‍💻 Auteur et Contact

**Oussama GUELFAA**  
📧 Email : guelfaao@gmail.com  
🎓 Stage de recherche - Réseaux de neurones holographiques  
📅 Dernière mise à jour : 18/07/2025

## 📄 Licence

Projet de recherche académique - Usage éducatif et scientifique.

---

## 🎉 Statut du Projet

✅ **Données organisées** - Structure claire et navigable  
✅ **Modèles fonctionnels** - Architectures validées  
✅ **Pipeline complet** - De l'analyse à la prédiction  
✅ **Documentation complète** - Guides et rapports détaillés  
✅ **Validation expérimentale** - Données PS 3µm analysées  

**🚀 Projet prêt pour le développement et la recherche avancée !**
