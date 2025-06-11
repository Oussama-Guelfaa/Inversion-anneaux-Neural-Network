# Résultats d'Entraînement et d'Évaluation du Réseau de Neurones

**Auteur :** Oussama GUELFAA  
**Date :** 05 - 06 - 2025  
**Objectif :** Prédiction des paramètres L_ecran et gap à partir de profils radiaux d'intensité holographique

## 📊 Résumé Exécutif

### Objectif Principal
Atteindre une précision de **R² > 0.8** pour la prédiction simultanée des paramètres L_ecran et gap à partir de profils d'intensité normalisés.

### Résultats Obtenus
- **R² global sur données expérimentales : -3.05** ❌
- **R² L_ecran : 0.942** ✅ (Excellent)
- **R² gap : -7.04** ❌ (Problème majeur)
- **Objectif R² > 0.8 : NON ATTEINT**

## 🗂️ Structure des Données

### Données d'Entraînement (Simulées)
- **Source :** `all_banque_new_24_01_25_NEW_full.mat`
- **Taille :** 990 échantillons (33 L_ecran × 30 gap)
- **Plages :**
  - L_ecran : [6.0 - 14.0] µm
  - gap : [0.025 - 1.5] µm
- **Format :** Profils radiaux 1D de 1000 points (ratio I_subs/I_subs_inc)

### Données de Test (Expérimentales)
- **Source :** Dossier `../data_generation/dataset/`
- **Taille :** 48 échantillons
- **Plages :**
  - L_ecran : [6.0 - 14.0] µm
  - gap : [0.025 - 0.517] µm (sous-ensemble)
- **Format :** Fichiers .mat avec variables `ratio`, `x`, `L_ecran_subs`, `gap`

## 🧠 Architecture du Modèle

### Modèle Optimisé (OptimizedRegressor)
```
Extracteur de Features:
├── Linear(1000 → 512) + BatchNorm + ReLU + Dropout(0.2)
├── Linear(512 → 256) + BatchNorm + ReLU + Dropout(0.15)
├── Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.1)
└── Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.05)

Têtes Spécialisées:
├── L_ecran_head: Linear(64 → 32) + ReLU + Linear(32 → 1)
└── gap_head: Linear(64 → 32) + ReLU + Linear(32 → 1)
```

### Paramètres du Modèle
- **Nombre total de paramètres :** 691,138
- **Fonction de perte :** MSE Loss
- **Optimiseur :** Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler :** ReduceLROnPlateau
- **Early stopping :** Patience = 25 epochs

## 📈 Résultats d'Entraînement

### Convergence
- **Epochs d'entraînement :** 66 (early stopping)
- **Temps d'entraînement :** 0.1 minutes
- **Meilleur R² validation :** 0.509

### Métriques de Validation
- **Train Loss final :** 0.476
- **Validation Loss final :** 0.571
- **Train R² final :** 0.520
- **Validation R² final :** 0.499

## 🎯 Performances sur Données Expérimentales

### Métriques Globales
| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **R² global** | -3.05 | Très mauvaise performance globale |
| **RMSE global** | N/A | - |
| **MAE global** | N/A | - |

### Métriques par Paramètre

#### L_ecran (Excellent)
| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **R²** | 0.942 | Excellente prédiction |
| **RMSE** | 0.584 µm | Erreur acceptable |
| **MAE** | 0.512 µm | Erreur moyenne faible |
| **MAPE** | 5.08% | Erreur relative très faible |

#### gap (Problématique)
| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **R²** | -7.04 | Performance catastrophique |
| **RMSE** | 0.498 µm | Erreur importante |
| **MAE** | 0.451 µm | Erreur moyenne élevée |
| **MAPE** | 803.28% | Erreur relative énorme |

## 🔍 Analyse des Problèmes

### 1. Problème de Généralisation
- **Cause principale :** Différence entre données simulées (entraînement) et expérimentales (test)
- **Manifestation :** Excellent R² pour L_ecran mais catastrophique pour gap
- **Hypothèse :** Le modèle a appris des artefacts spécifiques aux simulations

### 2. Déséquilibre des Paramètres
- **L_ecran :** Plage large [6-14] µm, variations importantes
- **gap :** Plage restreinte [0.025-0.517] µm dans les données test vs [0.025-1.5] µm en entraînement
- **Impact :** Le modèle n'a pas vu assez de variabilité pour gap dans le domaine expérimental

### 3. Analyse PCA Révélatrice
- **Seulement 7 composantes** expliquent 95% de la variance
- **Corrélations très fortes** entre features et L_ecran (-0.86 à 0.97)
- **Silhouette scores négatifs** indiquent un chevauchement des classes

## 📊 Interprétation Physique

### Pourquoi L_ecran fonctionne bien ?
1. **Signal fort :** L_ecran influence directement la forme globale du profil radial
2. **Plage large :** Variations importantes entre 6-14 µm créent des signatures distinctes
3. **Robustesse :** Moins sensible aux variations expérimentales

### Pourquoi gap échoue ?
1. **Signal faible :** gap influence subtilement les détails fins du profil
2. **Plage restreinte :** Peu de variabilité dans les données test [0.025-0.517] µm
3. **Sensibilité :** Plus affecté par le bruit expérimental et les différences de setup

## 🛠️ Recommandations d'Amélioration

### 1. Amélioration des Données
- **Augmenter la diversité** des données d'entraînement
- **Inclure plus de données expérimentales** dans l'entraînement
- **Équilibrer les plages** de gap entre simulation et expérience
- **Ajouter du bruit réaliste** aux simulations

### 2. Améliorations Architecturales
- **Modèle multi-tâches** avec poids adaptatifs pour chaque paramètre
- **Attention spécialisée** pour les features importantes pour gap
- **Régularisation différentielle** selon le paramètre
- **Ensemble de modèles** spécialisés

### 3. Techniques Avancées
- **Domain Adaptation** pour réduire l'écart simulation/expérience
- **Transfer Learning** avec fine-tuning sur données expérimentales
- **Data Augmentation** sophistiquée
- **Adversarial Training** pour la robustesse

### 4. Approches Alternatives
- **Modèles séparés** pour L_ecran et gap
- **Approche hiérarchique** : prédire L_ecran puis gap
- **Méthodes hybrides** combinant ML et physique
- **Optimisation bayésienne** des hyperparamètres

## 📁 Fichiers Générés

### Modèles et Données
- `models/final_optimized_regressor.pth` - Modèle entraîné
- `models/final_scalers.npz` - Scalers de normalisation
- `processed_data/training_data.npz` - Données d'entraînement préparées

### Visualisations et Analyses
- `plots/comprehensive_evaluation.png` - Évaluation complète
- `plots/data_analysis.png` - Analyse des données
- `plots/train_vs_test_comparison.png` - Comparaison train/test

### Scripts
- `train_and_evaluate_complete.py` - Entraînement complet
- `comprehensive_evaluation.py` - Évaluation détaillée
- `analyze_data.py` - Analyse des données
- `load_test_data.py` - Chargement données test

## 🎯 Conclusion

Le modèle développé montre des **performances excellentes pour L_ecran (R² = 0.942)** mais **échoue complètement pour gap (R² = -7.04)**. Cette dichotomie révèle un **problème fondamental de généralisation** entre les données simulées d'entraînement et les données expérimentales de test.

### Points Positifs
- ✅ Architecture robuste et bien optimisée
- ✅ Excellent apprentissage de L_ecran
- ✅ Pipeline complet d'entraînement et d'évaluation
- ✅ Analyse détaillée des problèmes

### Défis Majeurs
- ❌ Généralisation simulation → expérience
- ❌ Prédiction du paramètre gap
- ❌ Objectif R² > 0.8 non atteint

### Prochaines Étapes
1. **Collecter plus de données expérimentales** pour l'entraînement
2. **Implémenter domain adaptation** techniques
3. **Développer modèles spécialisés** par paramètre
4. **Analyser les différences physiques** simulation vs expérience

---

*Ce rapport constitue une base solide pour les améliorations futures du système de prédiction des paramètres holographiques.*
