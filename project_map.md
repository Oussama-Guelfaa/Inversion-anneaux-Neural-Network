# 🗺️ Project Map - Inversion d'Anneaux Neural Networks

**Author:** Oussama GUELFAA  
**Date:** 10 - 01 - 2025

## 📖 Vue d'Ensemble du Projet

Ce projet contient **7 réseaux de neurones modulaires** pour l'analyse holographique et la prédiction de paramètres physiques. Chaque réseau est organisé comme une unité indépendante avec sa propre architecture, configuration et pipeline d'entraînement.

## 🏗️ Architecture Modulaire

```
Inversion_anneaux/
├── 🔬 Reseau_Gap_Prediction_CNN/          # CNN 1D pour prédiction gap
├── 🔊 Reseau_Noise_Robustness/            # Tests robustesse au bruit
├── 🧪 Reseau_Overfitting_Test/            # Validation surapprentissage
├── 🧠 Reseau_Advanced_Regressor/          # Régresseur avancé avec attention
├── 🔥 Reseau_Ultra_Specialized/           # Architecture ultra-spécialisée
├── ⚡ Reseau_PyTorch_Optimized/           # PyTorch ResNet 1D optimisé
├── 🔧 Reseau_TensorFlow_Alternative/      # Alternative TensorFlow/Keras
├── 📊 data_generation/                    # Données MATLAB originales
├── 📋 project_map.md                      # Cette carte du projet
└── 📖 README.md                           # Documentation principale
```

## 🎯 Réseaux de Neurones Disponibles

### 1. 🔬 Reseau_Gap_Prediction_CNN
**Objectif:** Prédiction spécialisée du paramètre gap  
**Architecture:** CNN 1D avec blocs résiduels  
**Spécialité:** Focus sur gap uniquement, architecture robuste  
**Performance:** R² > 0.99 sur gap  

**Utilisation:**
```bash
cd Reseau_Gap_Prediction_CNN
python run.py --mode train
```

**Caractéristiques:**
- ✅ CNN 1D avec connexions résiduelles
- ✅ Global Average Pooling
- ✅ Dropout adaptatif
- ✅ Early stopping automatique

---

### 2. 🔊 Reseau_Noise_Robustness
**Objectif:** Test de robustesse au bruit gaussien  
**Architecture:** Réseau simplifié pour tests  
**Spécialité:** Évaluation progressive 0% à 20% de bruit  
**Performance:** R² > 0.8 même avec 5% de bruit  

**Utilisation:**
```bash
cd Reseau_Noise_Robustness
python run.py
```

**Caractéristiques:**
- ✅ Tests progressifs de bruit
- ✅ Bruit appliqué uniquement sur train
- ✅ Évaluation tolérance adaptative
- ✅ Analyse de généralisation locale

---

### 3. 🧪 Reseau_Overfitting_Test
**Objectif:** Validation capacité de surapprentissage  
**Architecture:** Simple sans régularisation  
**Spécialité:** Mêmes données train/validation  
**Performance:** R² > 0.99 et Loss < 0.001  

**Utilisation:**
```bash
cd Reseau_Overfitting_Test
python run.py
```

**Caractéristiques:**
- ✅ Test de mémorisation parfaite
- ✅ Architecture simple sans dropout
- ✅ Validation diagnostic
- ✅ Surveillance gradients

---

### 4. 🧠 Reseau_Advanced_Regressor
**Objectif:** Prédiction simultanée gap + L_ecran  
**Architecture:** Multi-têtes avec attention  
**Spécialité:** Résolution des 5 problèmes identifiés  
**Performance:** R² > 0.8 gap, R² > 0.95 L_ecran  

**Utilisation:**
```bash
cd Reseau_Advanced_Regressor
python run.py
```

**Caractéristiques:**
- ✅ Résolution systématique des problèmes
- ✅ Loss pondérée (gap × 30)
- ✅ Mécanisme d'attention pour gap
- ✅ Normalisation séparée par paramètre

---

### 5. 🔥 Reseau_Ultra_Specialized
**Objectif:** Performance maximale avec ensemble  
**Architecture:** Ensemble de 3 modèles ultra-profonds  
**Spécialité:** Double attention et optimisations extrêmes  
**Performance:** R² > 0.85 gap, R² > 0.98 L_ecran  

**Utilisation:**
```bash
cd Reseau_Ultra_Specialized
python run.py
```

**Caractéristiques:**
- ✅ Ensemble training (3 modèles)
- ✅ Double attention multiplicative
- ✅ Loss ultra-pondérée (gap × 50)
- ✅ Tolérance ultra-précise (±0.005 µm)

---

### 6. ⚡ Reseau_PyTorch_Optimized
**Objectif:** Implémentation PyTorch optimisée  
**Architecture:** ResNet 1D avec optimisations avancées  
**Spécialité:** Techniques PyTorch de pointe  
**Performance:** R² > 0.95 global  

**Utilisation:**
```bash
cd Reseau_PyTorch_Optimized
python run.py
```

**Caractéristiques:**
- ✅ ResNet 1D avec blocs résiduels
- ✅ CosineAnnealingWarmRestarts scheduler
- ✅ Optimisations mémoire et parallélisation
- ✅ Gradient clipping avancé

---

### 7. 🔧 Reseau_TensorFlow_Alternative
**Objectif:** Alternative TensorFlow/Keras  
**Architecture:** Dense 512→256→128→64→2  
**Spécialité:** API Keras avec callbacks automatiques  
**Performance:** R² > 0.85 global  

**Utilisation:**
```bash
cd Reseau_TensorFlow_Alternative
python run.py
```

**Caractéristiques:**
- ✅ Architecture Dense spécifiée
- ✅ Callbacks Keras automatiques
- ✅ Early stopping et ReduceLROnPlateau
- ✅ Sauvegarde native .h5

## 📊 Comparaison des Performances

| Réseau | Gap R² | L_ecran R² | Spécialité | Temps |
|--------|--------|------------|------------|-------|
| Gap Prediction CNN | >0.99 | - | Gap uniquement | ~5 min |
| Noise Robustness | >0.8* | >0.95* | Robustesse bruit | ~15 min |
| Overfitting Test | >0.99 | >0.99 | Validation diagnostic | ~3 min |
| Advanced Regressor | >0.8 | >0.95 | Résolution problèmes | ~8 min |
| Ultra Specialized | >0.85 | >0.98 | Performance max | ~20 min |
| PyTorch Optimized | >0.8 | >0.95 | Optimisations PyTorch | ~10 min |
| TensorFlow Alternative | >0.8 | >0.95 | API Keras | ~15 min |

*\* Performance sous 5% de bruit*

## 🎯 Guide de Sélection

### Pour la Production
- **Recommandé:** `Reseau_Advanced_Regressor` ou `Reseau_Ultra_Specialized`
- **Raison:** Résolution systématique des problèmes, performance élevée

### Pour la Recherche
- **Gap uniquement:** `Reseau_Gap_Prediction_CNN`
- **Robustesse:** `Reseau_Noise_Robustness`
- **Diagnostic:** `Reseau_Overfitting_Test`

### Pour le Développement
- **PyTorch:** `Reseau_PyTorch_Optimized`
- **TensorFlow:** `Reseau_TensorFlow_Alternative`

## 🔧 Structure Standardisée

Chaque réseau suit la même organisation :

```
Reseau_XYZ/
├── run.py              # Script autonome principal
├── config/
│   └── config.yaml     # Configuration complète
├── models/             # Modèles entraînés (.pth, .h5, .pkl)
├── plots/              # Visualisations et analyses
├── results/            # Métriques et rapports (JSON, CSV)
├── docs/               # Documentation spécialisée
└── README.md           # Guide d'utilisation
```

## 🚀 Démarrage Rapide

### Installation Globale
```bash
# Dépendances communes
pip install torch pandas numpy matplotlib seaborn scikit-learn pyyaml scipy joblib

# Pour TensorFlow (optionnel)
pip install tensorflow

# Pour visualisations avancées (optionnel)
pip install plotly tensorboard
```

### Test Rapide
```bash
# Tester le réseau le plus robuste
cd Reseau_Advanced_Regressor
python run.py

# Ou tester la performance maximale
cd Reseau_Ultra_Specialized
python run.py
```

## 📈 Données et Formats

### Source Commune
- **Fichier:** `data_generation/all_banque_new_24_01_25_NEW_full.mat`
- **Variables:** L_ecran_subs_vect, gap_sphere_vect, I_subs, I_subs_inc
- **Échantillons:** 990 profils d'intensité
- **Caractéristiques:** 600-1000 points radiaux

### Formats de Sortie
- **Modèles:** `.pth` (PyTorch), `.h5` (TensorFlow), `.pkl` (Scalers)
- **Résultats:** `.json` (métriques), `.csv` (historiques)
- **Visualisations:** `.png` (plots), `.html` (interactifs)

## 🎯 Objectifs et Applications

### Objectifs Scientifiques
- **Prédiction Gap:** 0.025-1.5 µm avec R² > 0.8
- **Prédiction L_ecran:** 6.0-14.0 µm avec R² > 0.95
- **Robustesse:** Performance maintenue sous bruit
- **Généralisation:** Validation sur données séparées

### Applications Pratiques
- **Holographie Expérimentale:** Inversion de paramètres en temps réel
- **Contrôle Qualité:** Validation de mesures holographiques
- **Optimisation:** Calibrage d'instruments optiques
- **Recherche:** Développement de nouvelles techniques

## 🏁 Conclusion

Ce projet offre une **suite complète de réseaux de neurones modulaires** pour l'analyse holographique. Chaque réseau est **autonome, documenté et prêt à l'emploi**. La structure modulaire permet de :

- ✅ **Sélectionner** le réseau optimal selon les besoins
- ✅ **Comparer** différentes approches facilement
- ✅ **Archiver** chaque réseau indépendamment
- ✅ **Déployer** en production rapidement
- ✅ **Maintenir** et **étendre** le projet efficacement

**Chaque réseau peut être zippé et utilisé comme unité indépendante !** 🚀
