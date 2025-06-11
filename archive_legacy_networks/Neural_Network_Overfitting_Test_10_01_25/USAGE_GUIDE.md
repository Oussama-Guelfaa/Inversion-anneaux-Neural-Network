# Guide d'Utilisation - Test d'Overfitting

**Auteur:** Oussama GUELFAA  
**Date:** 10 - 01 - 2025

## 🚀 Démarrage Rapide

### Prérequis
```bash
# Python 3.8+
# PyTorch, NumPy, Matplotlib, Pandas, Scikit-learn
pip install -r requirements.txt
```

### Exécution Complète
```bash
cd Neural_Network_Overfitting_Test_10_01_25/src
python overfitting_test.py
```

## 📁 Structure du Projet

```
Neural_Network_Overfitting_Test_10_01_25/
├── EXECUTIVE_SUMMARY.md          # Résumé exécutif des résultats
├── README.md                      # Documentation principale
├── USAGE_GUIDE.md                # Ce guide d'utilisation
├── requirements.txt               # Dépendances Python
│
├── src/                          # Code source
│   ├── overfitting_test.py       # Script principal du test
│   ├── analyze_predictions.py    # Analyse détaillée des résultats
│   └── inspect_data.py          # Inspection du dataset
│
├── docs/                         # Documentation détaillée
│   ├── RESULTS_ANALYSIS.md      # Analyse approfondie des résultats
│   └── NEXT_STEPS.md            # Plan de développement futur
│
├── models/                       # Modèles entraînés
│   └── overfitting_test_model.pth
│
├── plots/                        # Graphiques et visualisations
│   ├── training_curves.png
│   ├── predictions_analysis.png
│   ├── detailed_analysis.png
│   └── sample_profiles_inspection.png
│
└── results/                      # Résultats numériques
    ├── overfitting_test_summary.json
    ├── detailed_predictions.csv
    └── training_history.csv
```

## 🔧 Scripts Disponibles

### 1. Test Principal (`overfitting_test.py`)
**Objectif:** Exécute le test d'overfitting complet

```bash
python overfitting_test.py
```

**Sorties:**
- Modèle entraîné sauvegardé
- Métriques de performance
- Graphiques d'analyse
- Résultats détaillés

**Durée:** ~5-10 minutes

### 2. Analyse des Prédictions (`analyze_predictions.py`)
**Objectif:** Analyse détaillée des résultats

```bash
python analyze_predictions.py
```

**Sorties:**
- Statistiques avancées
- Identification des cas extrêmes
- Visualisations détaillées
- Rapport de synthèse

**Prérequis:** Avoir exécuté `overfitting_test.py`

### 3. Inspection des Données (`inspect_data.py`)
**Objectif:** Examine la structure du dataset

```bash
python inspect_data.py
```

**Sorties:**
- Structure des fichiers .mat
- Distribution des gaps
- Visualisation des profils
- Validation des données

## 📊 Interprétation des Résultats

### Métriques Clés
- **R² Score > 0.999:** Overfitting parfait atteint ✅
- **RMSE < 0.005 µm:** Erreur très faible ✅
- **MAE < 0.005 µm:** Précision excellente ✅
- **Loss < 1e-4:** Convergence optimale ✅

### Graphiques Générés

#### `training_curves.png`
- **Gauche:** Évolution de la loss (échelle log)
- **Droite:** Zoom sur les 50 dernières époques
- **Interprétation:** Décroissance monotone = bon overfitting

#### `predictions_analysis.png`
- **Gauche:** Scatter plot prédictions vs réelles
- **Centre:** Distribution des erreurs
- **Droite:** Erreurs vs valeurs réelles
- **Interprétation:** Points sur diagonale = prédictions parfaites

#### `detailed_analysis.png`
- 12 sous-graphiques d'analyse avancée
- Corrélations, distributions, convergence
- Q-Q plots, heatmaps, évolution temporelle

### Fichiers de Résultats

#### `overfitting_test_summary.json`
```json
{
  "r2": 0.999942,
  "rmse": 0.004388,
  "mae": 0.003092,
  "mse": 1.93e-05
}
```

#### `detailed_predictions.csv`
```csv
gap_true,gap_predicted,error,absolute_error,relative_error_percent
0.005,0.005001,0.000001,0.000001,0.020
...
```

## 🎯 Critères de Validation

### Succès du Test
- [x] R² > 0.99 (atteint: 0.999942)
- [x] MSE < 1e-4 (atteint: 1.93e-05)
- [x] Convergence stable (confirmée)
- [x] Erreurs négligeables (< 0.005 µm)

### Échec Potentiel
- [ ] R² < 0.95
- [ ] MSE > 1e-3
- [ ] Divergence ou instabilité
- [ ] Erreurs importantes (> 0.01 µm)

## 🔧 Personnalisation

### Modifier l'Architecture
```python
# Dans overfitting_test.py, classe SimpleGapPredictor
def __init__(self, input_size=1000):
    super().__init__()
    self.fc1 = nn.Linear(input_size, 1024)  # Plus de neurones
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 256)
    self.fc4 = nn.Linear(256, 1)
```

### Ajuster les Paramètres
```python
# Paramètres d'entraînement
batch_size = 16         # Batch plus grand
learning_rate = 0.001   # Learning rate plus élevé
num_epochs = 100        # Moins d'époques
```

### Changer le Dataset
```python
# Modifier le chemin dans load_dataset()
dataset_dir = "../../data_generation/autre_dataset"
```

## 🐛 Dépannage

### Erreur: "Dataset non trouvé"
```bash
# Vérifier le chemin
ls ../../data_generation/dataset_small_particle/
```

### Erreur: "CUDA non disponible"
```python
# Le code fonctionne sur CPU, pas de problème
device = torch.device('cpu')  # Forcé automatiquement
```

### Performance insuffisante
1. **Vérifier les données:** Exécuter `inspect_data.py`
2. **Augmenter les époques:** Modifier `num_epochs`
3. **Ajuster l'architecture:** Plus de neurones/couches
4. **Réduire le learning rate:** Plus de stabilité

### Graphiques non affichés
```python
# Ajouter en début de script
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
```

## 📝 Logs et Debugging

### Activer le Mode Verbose
```python
# Ajouter dans overfitting_test.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Sauvegarder les Logs
```bash
python overfitting_test.py > test_log.txt 2>&1
```

### Profiling des Performances
```python
import time
start_time = time.time()
# ... code ...
print(f"Temps d'exécution: {time.time() - start_time:.2f}s")
```

## 🔄 Workflow Recommandé

1. **Première exécution**
   ```bash
   python inspect_data.py      # Vérifier les données
   python overfitting_test.py  # Test principal
   python analyze_predictions.py  # Analyse détaillée
   ```

2. **Vérification des résultats**
   - Consulter `EXECUTIVE_SUMMARY.md`
   - Examiner les graphiques dans `plots/`
   - Vérifier les métriques dans `results/`

3. **Itération si nécessaire**
   - Ajuster les paramètres
   - Relancer le test
   - Comparer les résultats

## 📞 Support

### Documentation
- `README.md` - Vue d'ensemble
- `docs/RESULTS_ANALYSIS.md` - Analyse détaillée
- `docs/NEXT_STEPS.md` - Développement futur

### Code Source
- Scripts bien commentés
- Docstrings détaillées
- Gestion d'erreurs intégrée

---

**Note:** Ce guide couvre l'utilisation du test d'overfitting. Pour le développement du modèle complet, consulter `docs/NEXT_STEPS.md`.
