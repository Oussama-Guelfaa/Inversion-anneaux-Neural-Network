# 🔬 Inversion d'Anneaux - Neural Network Project

**Auteur:** Oussama GUELFAA
**Date:** 08 - 07 - 2025
**Version:** 2.0.0 - Avec CLI Moderne

## 📖 Vue d'Ensemble du Projet

Ce projet implémente des **solutions de réseaux de neurones spécialisées** pour l'analyse holographique d'anneaux et la prédiction de paramètres physiques. Le projet contient plusieurs implémentations optimisées pour différents cas d'usage, de la recherche à la production.

### 🎯 Objectifs Principaux

- **Prédiction de paramètres holographiques** : gap et L_écran à partir de profils d'intensité
- **Architectures spécialisées** : Réseaux optimisés pour chaque type de prédiction
- **Performance élevée** : R² > 0.95 pour tous les modèles de production
- **Robustesse validée** : Tests de bruit, overfitting, généralisation

## 🏗️ Architecture du Projet

```
Inversion_anneaux/
├── 🎯 Réseaux de Neurones Principaux
│   ├── Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25/  # ⭐ ULTRA-PRÉCISION - Dual prediction
│   ├── Reseau_Neural_Dual_Gap_Lecran_FINAL_06_01_25/           # Production - Dual prediction
│   └── Reseaux_1D_Gap_Prediction/                              # Prédiction gap seul
│       ├── Reseau_Noise_Robustness/                   # Tests robustesse bruit
│       ├── Reseau_Gap_Prediction_CNN/                 # CNN spécialisé gap
│       └── Reseau_Overfitting_Test/                   # Validation overfitting
├── 🖥️  CLI Moderne (NOUVEAU)
│   ├── cli/                                           # Modules CLI
│   ├── hologram_cli.py                               # Point d'entrée principal
│   ├── demo_cli.py                                   # Démonstration CLI
│   ├── test_cli_quick.py                            # Tests rapides
│   ├── CLI_README.md                                 # Documentation CLI
│   └── EXTENSION_GUIDE.md                           # Guide d'extension
├── 📊 Données et Scripts
│   ├── data_generation/                               # Données MATLAB originales
│   ├── analysis_scripts/                              # Scripts d'analyse
│   └── utilities/                                     # Utilitaires communs
├── 🧪 Tests
│   └── tests/                                         # Tests unitaires
├── 📦 Archives
│   ├── archive_legacy_networks/                       # Anciens réseaux
│   ├── legacy_images/                                 # Images historiques
│   └── legacy_tests/                                  # Tests historiques
└── 📋 README.md                                       # Cette documentation
```

## 📁 Structure Modulaire Standardisée

Chaque réseau de neurones suit la même organisation :

```
Reseau_XYZ/
├── run.py              # Script autonome principal
├── demo.py             # Démonstration rapide (si disponible)
├── config/
│   └── config.yaml     # Configuration complète
├── models/             # Modèles entraînés (.pth, .h5, .pkl)
├── plots/              # Visualisations automatiques
├── results/            # Métriques et rapports (JSON, CSV)
├── docs/               # Documentation spécialisée
├── data/               # Données prétraitées
├── logs/               # Logs d'entraînement
└── src/                # Code source modulaire
```

## 🚀 Guide d'Exécution Rapide

### 1️⃣ Installation des Dépendances

```bash
# Dépendances communes
pip install torch pandas numpy matplotlib seaborn scikit-learn pyyaml scipy joblib

# Pour TensorFlow (optionnel)
pip install tensorflow
```

### 2️⃣ Choix du Réseau selon l'Usage

#### 🏆 Pour Production (Recommandé)
```bash
# Réseau ultra-précision (NOUVEAU - 19/06/2025)
cd Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25
cd src/
python run.py

# Ou démonstration rapide
python demo.py

# Tests avancés
cd ../Test_dataset/
python test_dataset_2D.py  # Test sur 2440 échantillons
```

#### 🎯 Alternative Production
```bash
# Réseau production stable
cd Reseau_Neural_Dual_Gap_Lecran_FINAL_06_01_25
python run.py
```

#### 🔬 Pour Recherche Gap Seul
```bash
# Robustesse au bruit (le plus robuste)
cd Reseaux_1D_Gap_Prediction/Reseau_Noise_Robustness
python run.py

# CNN spécialisé gap
cd Reseaux_1D_Gap_Prediction/Reseau_Gap_Prediction_CNN
python run.py
```

#### 🧪 Pour Tests et Validation
```bash
# Test d'overfitting
cd Reseaux_1D_Gap_Prediction/Reseau_Overfitting_Test
python run.py
```

### 3️⃣ CLI Moderne (⭐ NOUVEAU - RECOMMANDÉ)

```bash
# Interface interactive complète
python hologram_cli.py --interactive

# Commandes directes
python hologram_cli.py train start --model precision
python hologram_cli.py predict single --input data.mat
python hologram_cli.py test accuracy --model production

# Démonstration des fonctionnalités
python demo_cli.py

# Tests rapides
python test_cli_quick.py
```

### 4️⃣ Analyse des Résultats

```bash
# Scripts d'analyse des données
cd analysis_scripts
python analyze_existing_results.py
python test_model_on_real_data.py

# Analyse complète dataset 2D
cd analysis_scripts/dataset_2D_analysis
python run_complete_dataset_2D_analysis.py
```

## 🖥️ CLI Moderne - Interface Unifiée

### ✨ Fonctionnalités Principales

Le nouveau CLI moderne offre une interface unifiée et interactive pour toutes les fonctionnalités :

- **🎯 Interface Interactive** : Menus riches avec navigation intuitive
- **📊 Formatage Avancé** : Tableaux colorés, panneaux stylés, graphiques ASCII
- **⚙️ Configuration Flexible** : Profils YAML pour différents cas d'usage
- **🔄 Barres de Progression** : Suivi visuel des opérations longues
- **🧪 Tests Intégrés** : Validation et tests de robustesse
- **📈 Visualisations** : Graphiques ASCII directement dans le terminal

### 🚀 Démarrage Rapide CLI

```bash
# Mode interactif (recommandé pour débuter)
python hologram_cli.py --interactive

# Aide générale
python hologram_cli.py --help

# Démonstration complète
python demo_cli.py
```

### 📚 Commandes Principales

| Commande | Description | Exemple |
|----------|-------------|---------|
| `train` | Entraînement de modèles | `python hologram_cli.py train start --model precision` |
| `predict` | Prédictions | `python hologram_cli.py predict single --input data.mat` |
| `test` | Tests et évaluation | `python hologram_cli.py test accuracy --model production` |
| `analyze` | Analyse de données | `python hologram_cli.py analyze dataset --dataset-path data/` |
| `visualize` | Visualisations | `python hologram_cli.py visualize results --results-file test.json` |
| `config` | Configuration | `python hologram_cli.py config show` |

### 📖 Documentation CLI

- **[CLI_README.md](CLI_README.md)** - Documentation complète du CLI
- **[EXTENSION_GUIDE.md](EXTENSION_GUIDE.md)** - Guide pour étendre le CLI
- **[tests/test_cli.py](tests/test_cli.py)** - Tests unitaires

## 🎯 Localisation des Modèles Entraînés

### Modèles de Production

#### Réseau Ultra-Précision (⭐ NOUVEAU - RECOMMANDÉ)
- **Localisation** : `Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25/models/`
- **Modèles** :
  - `dual_parameter_model.pth` - Modèle PyTorch ultra-précis (1,318,882 paramètres)
  - `input_scaler.pkl` - Normalisateur des profils d'intensité
  - `gap_scaler.pkl` - Normalisateur spécialisé gap
  - `L_ecran_scaler.pkl` - Normalisateur spécialisé L_écran
- **Performance** : R² = 0.9948 (gap), R² = 0.9891 (L_écran)
- **Précision** : 99.4% gap (±0.01µm), 94.2% L_écran (±0.1µm)
- **Validé sur** : 2440 échantillons (dataset_2D)
- **Usage** : Production industrielle haute précision

#### Réseau Production Stable
- **Localisation** : `Reseau_Neural_Dual_Gap_Lecran_FINAL_06_01_25/models/`
- **Modèles** :
  - `dual_gap_lecran_model.pth` - Modèle PyTorch principal
  - `scaler_X.pkl` - Normalisateur des données d'entrée
  - `scaler_y.pkl` - Normalisateur des paramètres de sortie
- **Performance** : R² = 0.9948 (gap), R² = 0.9949 (L_écran)
- **Usage** : Prédiction conjointe gap + L_écran

#### Réseaux Gap Seul
- **Noise Robustness** : `Reseaux_1D_Gap_Prediction/Reseau_Noise_Robustness/models/`
  - `noise_robust_model_5pct.pth` - Modèle robuste au bruit
  - Performance : R² = 0.9948, robuste jusqu'à 10% de bruit
- **CNN Gap** : `Reseaux_1D_Gap_Prediction/Reseau_Gap_Prediction_CNN/models/`
  - `gap_prediction_cnn.pth` - CNN spécialisé gap
  - Performance : R² > 0.99

### Modèles de Test et Validation
- **Overfitting Test** : `Reseaux_1D_Gap_Prediction/Reseau_Overfitting_Test/models/`
- **Archives** : `archive_legacy_networks/` (modèles historiques)

## 🧭 Navigation entre Implémentations

### Par Type de Prédiction

#### Prédiction Dual (Gap + L_écran)
```bash
# Réseau ultra-précision (NOUVEAU - RECOMMANDÉ)
cd Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25
cd src/
```
- **Architecture** : 1,318,882 paramètres optimisés
- **Données** : 600 points par profil (tronqué optimisé)
- **Innovation** : Data augmentation avancée + scaling séparé
- **Résultats** : R² = 0.9948 (gap), R² = 0.9891 (L_écran)
- **Précision** : 99.4% gap, 94.2% L_écran dans tolérances industrielles
- **Validé** : 2440 échantillons dataset_2D

```bash
# Réseau production stable (Alternative)
cd Reseau_Neural_Dual_Gap_Lecran_FINAL_06_01_25
```
- **Architecture** : Dense layers 512→256→128→64→2
- **Données** : 600 points par profil (optimisé)
- **Innovation** : Data augmentation 2D par interpolation
- **Résultats** : R² > 0.99 pour les deux paramètres

#### Prédiction Gap Seul
```bash
# Pour robustesse maximale
cd Reseaux_1D_Gap_Prediction/Reseau_Noise_Robustness

# Pour performance pure
cd Reseaux_1D_Gap_Prediction/Reseau_Gap_Prediction_CNN

# Pour validation
cd Reseaux_1D_Gap_Prediction/Reseau_Overfitting_Test
```

### Par Framework

#### PyTorch (Principal)
- Tous les réseaux utilisent PyTorch par défaut
- Architecture modulaire avec blocs résiduels
- Optimisation Adam + ReduceLROnPlateau

#### TensorFlow (Alternatif)
- Implémentations alternatives disponibles dans certains réseaux
- Architecture Dense Sequential
- Compatible avec les modèles PyTorch

## 📊 Données et Formats

### Source des Données
- **Fichier principal** : `data_generation/all_banque_new_24_01_25_NEW_full.mat`
- **Variables** : L_ecran_subs_vect, gap_subs_vect, I_subs, I_subs_inc
- **Échantillons** : 990 profils d'intensité originaux
- **Augmentation** : Jusqu'à 12,200 échantillons (facteur 5x)

### Formats de Sortie
- **Modèles** : `.pth` (PyTorch), `.pkl` (Scalers)
- **Résultats** : `.json` (métriques), `.csv` (historiques)
- **Visualisations** : `.png` (plots haute résolution)

## 🏆 Performances et Résultats

### Réseau Ultra-Précision (NOUVEAU - 19/06/2025)
- **Gap** : R² = 0.9948, MAE = 0.0035 µm, RMSE = 0.0042 µm
- **L_écran** : R² = 0.9891, MAE = 0.0335 µm, RMSE = 0.0460 µm
- **Précision industrielle** : 99.4% gap (±0.01µm), 94.2% L_écran (±0.1µm)
- **Validation** : 2440 échantillons dataset_2D
- **Architecture** : 1,318,882 paramètres optimisés
- **Temps d'entraînement** : 300 epochs

### Réseau Dual (Production Stable)
- **Gap** : R² = 0.9946, RMSE = 0.062 µm
- **L_écran** : R² = 0.9949, RMSE = 0.125 µm
- **Précision** : 97% (gap), 99.9% (L_écran)
- **Temps d'entraînement** : 4.2 minutes

### Réseaux Gap Seul
- **Noise Robustness** : R² = 0.9948, robuste jusqu'à 10% bruit
- **CNN Gap** : R² > 0.99, convergence rapide
- **Overfitting Test** : Validation complète, pas de surapprentissage

## 🔧 Utilitaires et Outils

### Scripts d'Analyse
- **`analysis_scripts/`** : Analyse complète des données et résultats
- **Dataset 2D Analysis** : Suite de 8 scripts spécialisés
- **Test sur données réelles** : Validation sur données expérimentales

### Utilitaires Communs
- **`utilities/`** : Fonctions partagées, configurations, exemples
- **Data augmentation** : Scripts d'augmentation 2D par interpolation
- **Validation** : Outils de test et métriques standardisées

## 🆕 Nouveautés Version 2.0.0

### ✨ CLI Moderne Intégré
- **Interface Interactive** : Navigation intuitive avec menus riches
- **Commandes Hiérarchiques** : Structure organisée (train/predict/test/analyze/visualize/config)
- **Formatage Avancé** : Tableaux colorés, panneaux stylés, graphiques ASCII
- **Configuration Flexible** : Profils YAML pour production/recherche/démo
- **Tests Automatisés** : Suite de tests intégrée avec validation

### 🎯 Améliorations Techniques
- **Validation Robuste** : Vérification des chemins, modèles et configurations
- **Gestion d'Erreurs** : Messages d'erreur clairs et solutions proposées
- **Performance** : Détection automatique du meilleur device (CPU/CUDA/MPS)
- **Extensibilité** : Architecture modulaire pour ajout de fonctionnalités

### 📊 Nouvelles Fonctionnalités
- **Visualisations ASCII** : Graphiques directement dans le terminal
- **Analyse Avancée** : Comparaison de modèles, distribution des paramètres
- **Export/Import** : Sauvegarde et partage de configurations
- **Mode Batch** : Traitement de lots de fichiers
- **Simulation** : Génération et test d'anneaux synthétiques

## 📈 Recommandations d'Usage

### 🚀 Pour Nouveaux Utilisateurs (RECOMMANDÉ)
1. **Démarrer avec le CLI** : `python hologram_cli.py --interactive`
2. **Suivre la démonstration** : `python demo_cli.py`
3. **Tester rapidement** : `python test_cli_quick.py`
4. **Consulter la doc** : [CLI_README.md](CLI_README.md)

### 🔬 Pour Utilisateurs Avancés
1. **Utiliser** : `Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25` (ULTRA-PRÉCISION)
2. **CLI Production** : `python hologram_cli.py config set-profile --profile production`
3. **Tests complets** : `python hologram_cli.py test accuracy --model precision --detailed`

### 🧪 Pour Recherche et Développement
1. **Profil recherche** : `python hologram_cli.py config set-profile --profile recherche`
2. **Analyse dataset** : `python hologram_cli.py analyze dataset --dataset-path data/ --detailed`
3. **Extension CLI** : Consulter [EXTENSION_GUIDE.md](EXTENSION_GUIDE.md)
4. **Résultats** : Consultez `results/` et `plots/`
5. **Alternative** : `Reseau_Neural_Dual_Gap_Lecran_FINAL_06_01_25` (stable)

### Pour Développeurs
1. **Étudier** : Structure modulaire dans `src/`
2. **Configurer** : Modifier `config/config.yaml`
3. **Étendre** : Utiliser `utilities/` comme base

### Pour Chercheurs
1. **Analyser** : Scripts dans `analysis_scripts/`
2. **Comparer** : Différents réseaux dans `Reseaux_1D_Gap_Prediction/`
3. **Valider** : Tests de robustesse et overfitting

## 🆕 NOUVEAU - CLI Moderne + Réseau Ultra-Précision

### 🖥️ CLI Moderne (08/07/2025) - Interface Unifiée

**Localisation** : `cli/`, `hologram_cli.py`, `demo_cli.py`

#### ✨ Fonctionnalités Avancées
- **Interface Interactive** : Menus riches avec questionary
- **Commandes Hiérarchiques** : train/predict/test/analyze/visualize/config
- **Formatage Rich** : Tableaux colorés, panneaux stylés, graphiques ASCII
- **Configuration YAML** : Profils pour production/recherche/démo
- **Tests Intégrés** : Validation automatique et tests de robustesse

#### 🚀 Utilisation Simplifiée
```bash
# Interface interactive complète
python hologram_cli.py --interactive

# Commandes directes
python hologram_cli.py train start --model precision
python hologram_cli.py predict single --input data.mat
python hologram_cli.py test accuracy --model production

# Démonstration
python demo_cli.py
```

### 🎯 Réseau Neural Dual Gap + L_ecran - PRECISION 007µm

**Localisation** : `Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25/`

#### ✨ Innovations Clés
- **Architecture optimisée** : 1,318,882 paramètres finement ajustés
- **Scaling séparé** : Normalisation indépendante gap/L_écran
- **Data augmentation avancée** : Interpolation sophistiquée
- **Structure organisée** : Code source dans `src/`, tests dans `Test_dataset/`, docs dans `docs/`
- **Intégration CLI** : Accessible via `python hologram_cli.py train start --model precision`

#### 🏆 Performances Exceptionnelles
- **Gap** : R² = 0.9948 (99.48%), MAE = 0.0035µm
- **L_écran** : R² = 0.9891 (98.91%), MAE = 0.0335µm
- **Précision industrielle** : 99.4% gap (±0.01µm), 94.2% L_écran (±0.1µm)
- **Validation étendue** : 2440 échantillons dataset_2D

#### 🚀 Utilisation Rapide

**Méthode Moderne (CLI) - RECOMMANDÉE :**
```bash
# Interface interactive
python hologram_cli.py --interactive

# Commandes directes
python hologram_cli.py train start --model precision
python hologram_cli.py predict single --input data.mat
python hologram_cli.py test accuracy --model precision --detailed
```

**Méthode Traditionnelle :**
```bash
cd Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25

# Démonstration
cd src/
python demo.py

# Test complet sur dataset_2D
cd ../Test_dataset/
python test_dataset_2D.py
```

#### 📁 Structure Organisée
```
Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25/
├── src/                    # Code source principal
├── Test_dataset/           # Scripts de test et validation
├── docs/                   # Documentation complète
├── models/                 # Modèles entraînés
├── results/                # Résultats des tests
└── plots/                  # Visualisations
```

---

## 🎉 Conclusion

Ce projet représente une solution complète et moderne pour l'inversion d'anneaux holographiques :

### 🏆 Réalisations Principales
- **Précision Ultra-Haute** : ±0.007µm pour le gap, ±0.5µm pour L_écran
- **Interface Moderne** : CLI interactif avec formatage riche et visualisations
- **Architecture Robuste** : Validation, tests automatisés, gestion d'erreurs
- **Flexibilité** : Profils configurables pour différents cas d'usage
- **Extensibilité** : Architecture modulaire pour ajouts futurs

### 🚀 Prochaines Étapes
- **Intégration Continue** : Tests automatisés sur nouvelles données
- **Extensions CLI** : Nouvelles fonctionnalités selon les besoins
- **Optimisations** : Amélioration continue des performances
- **Documentation** : Mise à jour continue de la documentation

### 📞 Support et Contact

**Auteur :** Oussama GUELFAA
**Email :** guelfaao@gmail.com
**Date :** 08 - 07 - 2025
**Version :** 2.0.0 - CLI Moderne Intégré

### 📚 Documentation Complète
- **[CLI_README.md](CLI_README.md)** - Guide complet du CLI moderne
- **[EXTENSION_GUIDE.md](EXTENSION_GUIDE.md)** - Guide d'extension et personnalisation
- **[tests/test_cli.py](tests/test_cli.py)** - Tests unitaires complets

### 🛠️ Démarrage Rapide
```bash
# Démonstration complète du CLI
python demo_cli.py

# Tests rapides de validation
python test_cli_quick.py

# Interface interactive
python hologram_cli.py --interactive

# Aide complète
python hologram_cli.py --help
```

---

**🎯 Résultat : Maîtrise complète de la prédiction de paramètres holographiques avec interface moderne et précision industrielle ultra-haute !**