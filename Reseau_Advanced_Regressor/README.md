# 🧠 Réseau Advanced Regressor

**Author:** Oussama GUELFAA  
**Date:** 10 - 01 - 2025

## 📖 Description

Ce réseau de neurones implémente un régresseur avancé avec mécanisme d'attention pour prédire simultanément les paramètres gap et L_ecran. Il résout systématiquement 5 problèmes identifiés dans les versions précédentes en utilisant une architecture spécialisée avec têtes dédiées et attention pour le signal gap plus faible.

## 🎯 Objectifs

- **Prédiction Simultanée**: Gap et L_ecran en une seule passe
- **Résolution de Problèmes**: 5 problèmes systématiquement adressés
- **Architecture Spécialisée**: Têtes dédiées avec mécanisme d'attention
- **Performance Cible**: R² > 0.8 pour gap, R² > 0.95 pour L_ecran

## 🏗️ Architecture du Modèle

### Structure Multi-Têtes avec Attention
- **Entrée**: Profils d'intensité tronqués (600 caractéristiques)
- **Feature Extractor**: Commun aux deux paramètres
- **Tête L_ecran**: Simple (signal fort)
- **Tête Gap**: Spécialisée avec attention (signal faible)
- **Sortie**: [L_ecran, gap] simultanément

### Composants Architecturaux
```python
# Feature extractor commun
Linear(600, 512) + BatchNorm + ReLU + Dropout(0.2)
Linear(512, 256) + BatchNorm + ReLU + Dropout(0.15)
Linear(256, 128) + BatchNorm + ReLU + Dropout(0.1)

# Tête L_ecran (signal fort)
Linear(128, 64) + ReLU + Dropout(0.05)
Linear(64, 32) + ReLU
Linear(32, 1)

# Tête gap avec attention (signal faible)
Attention: Linear(128, 64) + Tanh + Linear(64, 128) + Sigmoid
Gap Head: Linear(128, 128) + BatchNorm + ReLU + Dropout(0.01)
          Linear(128, 64) + BatchNorm + ReLU + Dropout(0.01)
          Linear(64, 32) + ReLU + Linear(32, 1)
```

## 🔧 Résolution des 5 Problèmes

### Problème 1: Précision Excessive des Labels
- **Solution**: Arrondissement à 3 décimales
- **Impact**: Réduction du bruit dans les labels
- **Implémentation**: `np.round(labels, 3)`

### Problème 2: Échelles Déséquilibrées
- **Solution**: Normalisation séparée par paramètre
- **Impact**: Équilibrage des gradients
- **Implémentation**: StandardScaler individuel pour L_ecran et gap

### Problème 3: Distribution Déséquilibrée
- **Solution**: Focus sur plage expérimentale [0.025, 0.517] µm
- **Impact**: Concentration sur données pertinentes
- **Implémentation**: Filtrage par masque booléen

### Problème 4: Loss Function Inadaptée
- **Solution**: Loss pondérée (gap × 30, L_ecran × 1)
- **Impact**: Attention accrue sur le paramètre gap
- **Implémentation**: WeightedMSELoss personnalisée

### Problème 5: Signal Gap Faible
- **Solution**: Architecture spécialisée avec attention
- **Impact**: Amplification du signal gap
- **Implémentation**: Mécanisme d'attention + tête dédiée

## 📊 Données et Prétraitement

### Source des Données
- **Fichier**: `all_banque_new_24_01_25_NEW_full.mat`
- **Échantillons**: 990 profils d'intensité
- **Caractéristiques**: 600 points radiaux (tronqués)
- **Paramètres**: L_ecran [6.0-14.0 µm], gap [0.025-1.5 µm]

### Pipeline de Prétraitement
1. **Extraction**: Ratios I_subs/I_subs_inc
2. **Troncature**: 1000 → 600 points
3. **Arrondissement**: Labels à 3 décimales
4. **Filtrage**: Focus plage expérimentale
5. **Normalisation**: StandardScaler séparé

## 🚀 Utilisation

### Installation des Dépendances
```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn pyyaml scipy joblib
```

### Entraînement du Modèle
```bash
# Entraînement complet avec résolution des problèmes
python run.py

# Avec configuration personnalisée
python run.py --config config/advanced_config.yaml
```

### Configuration Personnalisée
Modifiez `config/advanced_config.yaml` pour ajuster:
- Poids de la loss pondérée
- Architecture du modèle
- Paramètres d'entraînement
- Critères d'évaluation

## 📈 Métriques de Performance

### Métriques Principales
- **R² Global**: Performance combinée
- **R² L_ecran**: Performance sur L_ecran (cible > 0.95)
- **R² Gap**: Performance sur gap (cible > 0.8)
- **RMSE**: Erreur quadratique moyenne
- **MAE**: Erreur absolue moyenne
- **Tolérance**: Précision dans seuils définis

### Évaluation avec Tolérance
- **Tolérance L_ecran**: ±0.5 µm
- **Tolérance Gap**: ±0.01 µm (2 décimales)
- **Critère de Succès**: 90% des prédictions dans tolérance

## 📁 Structure des Fichiers

```
Reseau_Advanced_Regressor/
├── run.py                              # Script principal autonome
├── config/
│   └── advanced_config.yaml            # Configuration complète
├── models/
│   ├── advanced_regressor_best.pth     # Meilleur modèle
│   ├── advanced_regressor_scaler_X.pkl # Scaler profils
│   ├── advanced_regressor_scaler_L.pkl # Scaler L_ecran
│   └── advanced_regressor_scaler_gap.pkl # Scaler gap
├── plots/
│   ├── training_curves.png             # Courbes d'entraînement
│   ├── predictions_scatter.png         # Prédictions vs réel
│   ├── tolerance_analysis.png          # Analyse tolérance
│   └── attention_weights.png           # Visualisation attention
├── results/
│   ├── training_history.csv            # Historique entraînement
│   ├── evaluation_metrics.json         # Métriques détaillées
│   └── config_used.yaml                # Configuration utilisée
├── docs/
│   ├── EXECUTIVE_SUMMARY.md             # Résumé exécutif
│   ├── README_neural_network_06_06_25.md # Documentation détaillée
│   └── README_tolerance_evaluation.md   # Évaluation tolérance
└── README.md                           # Cette documentation
```

## 🔬 Fonctionnalités Avancées

### Mécanisme d'Attention
- **Type**: Self-attention pour gap
- **Fonction**: Amplification des features pertinentes
- **Architecture**: Linear + Tanh + Linear + Sigmoid
- **Application**: Multiplication élément par élément

### Loss Pondérée
```python
class WeightedMSELoss(nn.Module):
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        weighted_mse = mse * weights  # [1.0, 30.0]
        return weighted_mse.mean()
```

### Gradient Clipping
- **Activation**: Configurable
- **Norme Max**: 1.0 par défaut
- **Objectif**: Stabilité d'entraînement

### Early Stopping
- **Patience**: 25 epochs
- **Critère**: Validation loss
- **Restauration**: Meilleur modèle automatique

## 🧪 Tests et Validation

### Validation Croisée
- **Division**: 80% train, 20% validation
- **Stratégie**: Aléatoire avec seed fixe
- **Métriques**: R², RMSE, MAE, tolérance

### Tests de Robustesse
- **Initialisation**: Kaiming normal
- **Reproductibilité**: Seed fixe (42)
- **Stabilité**: Batch normalization

## 🎯 Résultats Attendus

### Performance L_ecran
- **R²**: > 0.95 (signal fort)
- **RMSE**: < 0.1 µm
- **Tolérance**: > 95% dans ±0.5 µm

### Performance Gap
- **R²**: > 0.8 (signal faible mais amélioré)
- **RMSE**: < 0.05 µm
- **Tolérance**: > 90% dans ±0.01 µm

### Performance Globale
- **Convergence**: < 100 epochs
- **Stabilité**: Entraînement robuste
- **Généralisation**: Bonne performance test

## 🔧 Optimisations Implémentées

### Architecture
- **Têtes Spécialisées**: Adaptation au signal
- **Attention**: Focus sur features importantes
- **Dropout Adaptatif**: Régularisation progressive
- **Batch Normalization**: Stabilisation

### Entraînement
- **Loss Pondérée**: Équilibrage des objectifs
- **Learning Rate**: Adaptatif avec scheduler
- **Weight Decay**: Régularisation L2
- **Gradient Clipping**: Prévention explosion

### Données
- **Prétraitement Systématique**: Résolution des 5 problèmes
- **Normalisation Séparée**: Équilibrage des échelles
- **Focus Expérimental**: Données pertinentes
- **Troncature Optimale**: 600 points

## 📊 Monitoring et Logging

### Métriques Suivies
- **Loss**: Train et validation
- **R²**: Global, L_ecran, gap
- **Tolérance**: Précision pratique
- **Learning Rate**: Adaptation automatique

### Fréquence de Log
- **Entraînement**: Toutes les 10 epochs
- **Sauvegarde**: Meilleur modèle automatique
- **Visualisation**: Graphiques automatiques

**Ce modèle résout systématiquement les problèmes identifiés pour une performance optimale!** 🚀
