# 🔧 Réseau TensorFlow Alternative

**Author:** Oussama GUELFAA  
**Date:** 10 - 01 - 2025

## 📖 Description

Ce réseau de neurones implémente une alternative TensorFlow/Keras avec architecture Dense optimisée. Il utilise l'API Keras intuitive avec callbacks automatiques pour fournir une solution robuste et facile à utiliser pour la prédiction de paramètres holographiques.

## 🎯 Objectifs

- **TensorFlow/Keras**: API intuitive et callbacks automatiques
- **Architecture Dense**: 512→256→128→64→2 selon spécifications
- **Performance**: R² > 0.85 global avec convergence stable
- **Simplicité**: Interface Keras conviviale

## 🏗️ Architecture

### Dense Sequential Keras
- **Entrée**: Profils d'intensité complets (1000 points)
- **Architecture**: Dense layers selon mémoires du projet
- **Couches**: 512 → 256 → 128 → 64 → 2 neurones
- **Régularisation**: Dropout 0.2 + L2 regularization
- **Batch Normalization**: Stabilisation d'entraînement
- **Activation**: ReLU + Linear output

### Callbacks Keras
```python
# Early Stopping automatique
EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Réduction Learning Rate
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

# Sauvegarde automatique
ModelCheckpoint(filepath='models/tensorflow_best_model.h5', save_best_only=True)
```

## 🚀 Utilisation

### Installation
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn pyyaml scipy joblib
```

### Entraînement
```bash
# Entraînement TensorFlow/Keras
python run.py

# Configuration personnalisée
python run.py --config config/tensorflow_config.yaml
```

## 📊 Fonctionnalités TensorFlow

### API Keras Intuitive
- **Sequential Model**: Construction simple
- **Compile**: Configuration automatique
- **Fit**: Entraînement avec callbacks
- **Evaluate**: Évaluation native

### Callbacks Automatiques
- **Early Stopping**: Arrêt automatique optimal
- **ReduceLROnPlateau**: Adaptation learning rate
- **ModelCheckpoint**: Sauvegarde du meilleur modèle
- **TensorBoard**: Monitoring optionnel

### Sauvegarde Native
- **Format .h5**: Sauvegarde complète Keras
- **Model Summary**: Architecture détaillée
- **Weights**: Poids séparés si nécessaire

## 📁 Structure

```
Reseau_TensorFlow_Alternative/
├── run.py                              # Script TensorFlow/Keras
├── config/
│   └── tensorflow_config.yaml          # Configuration TensorFlow
├── models/
│   ├── tensorflow_best_model.h5        # Modèle Keras complet
│   ├── tensorflow_scaler_X.pkl         # Scaler features
│   └── tensorflow_scaler_y.pkl         # Scaler targets
├── plots/
│   └── tensorflow_analysis.png         # Analyse TensorFlow
├── results/
│   ├── tensorflow_training_history.csv # Historique Keras
│   ├── tensorflow_metrics.json         # Métriques détaillées
│   └── tensorflow_model_summary.txt    # Résumé architecture
└── README.md                           # Cette documentation
```

## 🎯 Résultats Attendus

### Performance TensorFlow
- **R² Global**: > 0.85
- **Convergence**: < 150 epochs
- **Temps d'Entraînement**: < 15 minutes
- **Compatibilité Keras**: Complète

### Avantages TensorFlow
- **API Keras intuitive et conviviale**
- **Callbacks automatiques intégrés**
- **Visualisation TensorBoard optionnelle**
- **Sauvegarde native .h5 complète**
- **Écosystème TensorFlow complet**

## 🔧 Configuration Avancée

### Optimisations TensorFlow
```yaml
tensorflow_specific:
  mixed_precision: false      # AMP pour GPU
  distribute_strategy: false  # Multi-GPU
  jit_compile: false         # XLA compilation
  use_multiprocessing: true  # Parallélisation
  workers: 4                 # Nombre de workers
```

### Monitoring
```yaml
tensorboard:
  enable: false              # TensorBoard logging
  log_dir: "logs/"          # Répertoire logs
```

## 📊 Métriques TensorFlow

### Métriques Natives
- **Loss**: MSE TensorFlow native
- **MAE**: Mean Absolute Error
- **Custom Metrics**: R², RMSE calculés

### Historique Complet
- **Training Loss**: Évolution par epoch
- **Validation Loss**: Monitoring overfitting
- **Learning Rate**: Adaptation automatique
- **Callbacks**: Actions automatiques

**Alternative TensorFlow/Keras complète et conviviale!** 🚀
