# 🚀 Guide d'Utilisation - Réseau Dual Gap + L_ecran

## ⚡ Utilisation Rapide

### 1. Entraînement Complet
```bash
python run.py
```

### 2. Démonstration du Modèle
```bash
python demo.py
```

### 3. Tests Préliminaires
```bash
python run.py --test
```

## 📊 Résultats Obtenus

- **Gap R²**: 0.9946 (99.46%)
- **L_ecran R²**: 0.9949 (99.49%)
- **Gap Accuracy**: 97.0%
- **L_ecran Accuracy**: 99.9%

## 📁 Fichiers Importants

- `models/dual_parameter_model.pth` - Modèle entraîné
- `results/complete_results.json` - Résultats détaillés
- `plots/` - Visualisations générées
- `config/dual_prediction_config.yaml` - Configuration

## 🔧 Utilisation Programmatique

```python
import torch
from src.dual_parameter_model import DualParameterPredictor

# Charger le modèle
model = DualParameterPredictor(input_size=600)
checkpoint = torch.load('models/dual_parameter_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prédire (après normalisation des données)
with torch.no_grad():
    predictions = model(input_tensor)
    gap_pred = predictions[0, 0]
    L_ecran_pred = predictions[0, 1]
```

## 🎯 Performance

Ce modèle a **dépassé tous les objectifs** avec une performance exceptionnelle:
- Tous les objectifs atteints ✅
- Prêt pour déploiement en production
- Robuste et fiable
