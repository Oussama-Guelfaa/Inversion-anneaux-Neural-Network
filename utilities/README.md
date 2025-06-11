# 🔧 Utilities

**Author:** Oussama GUELFAA  
**Date:** 10 - 01 - 2025

## 📖 Description

Ce dossier contient les **utilitaires et outils de support** pour le développement et la maintenance du projet. Ces composants fournissent des fonctionnalités communes utilisées par les réseaux de neurones modulaires.

## 📁 Structure

```
utilities/
├── src/                    # Code source utilitaire
├── configs/                # Configurations globales
├── data/                   # Données de développement
├── models/                 # Modèles utilitaires
├── results/                # Résultats de développement
├── docs/                   # Documentation technique
├── examples/               # Exemples d'utilisation
├── tests/                  # Tests unitaires
└── README.md               # Cette documentation
```

## 🛠️ Composants Utilitaires

### src/ - Code Source
- **data/**: Modules d'extraction et traitement de données
- **evaluation/**: Outils d'évaluation et métriques
- **training/**: Scripts d'entraînement génériques
- **utils/**: Fonctions utilitaires communes

### configs/ - Configurations
- **training_config.yaml**: Configuration d'entraînement par défaut
- **Templates**: Modèles de configuration pour nouveaux réseaux

### data/ - Données de Développement
- **processed/**: Données prétraitées pour développement
- **Samples**: Échantillons de test et validation

### models/ - Modèles Utilitaires
- **pytorch/**: Modèles PyTorch de référence
- **tensorflow/**: Modèles TensorFlow de référence

### docs/ - Documentation Technique
- **data_extraction.md**: Guide d'extraction de données
- **model_architecture.md**: Architectures de référence
- **results_analysis.md**: Méthodes d'analyse

### examples/ - Exemples
- **quick_start.py**: Exemple de démarrage rapide
- **sample_holograms/**: Échantillons d'hologrammes

### tests/ - Tests Unitaires
- **test_data_processing.py**: Tests de traitement de données
- **Tests de validation**: Validation des composants

## 🚀 Utilisation

### Pour Développement de Nouveaux Réseaux
```bash
# Utiliser les templates de configuration
cp utilities/configs/training_config.yaml nouveau_reseau/config/

# Utiliser les modules utilitaires
from utilities.src.utils.data_utils import extract_matlab_data
```

### Pour Tests et Validation
```bash
# Exécuter les tests unitaires
cd utilities/tests
python -m pytest

# Tester l'extraction de données
python utilities/examples/quick_start.py
```

## 🔗 Intégration avec Réseaux Modulaires

### Fonctions Communes
Les réseaux modulaires peuvent utiliser les utilitaires :

```python
# Dans un réseau modulaire
import sys
sys.path.append('../utilities/src')

from data.extract_training_data import load_matlab_data
from utils.data_utils import preprocess_profiles
from evaluation.evaluate_models import calculate_metrics
```

### Templates et Configurations
- **Configurations de base**: Templates YAML réutilisables
- **Architectures de référence**: Modèles de base à étendre
- **Scripts d'évaluation**: Métriques standardisées

## 📊 Outils d'Analyse

### Extraction de Données
- **Modules MATLAB**: Interface avec fichiers .mat
- **Prétraitement**: Normalisation, filtrage, augmentation
- **Validation**: Vérification cohérence des données

### Évaluation de Modèles
- **Métriques standardisées**: R², RMSE, MAE, tolérance
- **Visualisations**: Graphiques automatiques
- **Rapports**: Génération de rapports détaillés

### Utilitaires de Développement
- **Logging**: Système de logs unifié
- **Configuration**: Gestion des paramètres
- **Debugging**: Outils de diagnostic

## 🔧 Maintenance

### Mise à Jour des Utilitaires
```bash
# Mettre à jour les dépendances
pip install -r utilities/requirements.txt

# Tester les modifications
cd utilities/tests
python -m pytest
```

### Ajout de Nouveaux Utilitaires
1. **Créer le module** dans `src/`
2. **Ajouter les tests** dans `tests/`
3. **Documenter** dans `docs/`
4. **Mettre à jour** ce README

## 📈 Évolution

### Historique
- **Version 1.0**: Utilitaires de base extraits des anciens réseaux
- **Version 2.0**: Standardisation et modularisation
- **Version 3.0**: Intégration avec réseaux modulaires

### Roadmap
- **Templates avancés**: Configurations pour nouveaux types de réseaux
- **Outils de monitoring**: Surveillance en temps réel
- **API unifiée**: Interface commune pour tous les réseaux

## ⚠️ Notes Importantes

### Compatibilité
- **Python 3.8+**: Version minimale requise
- **Dependencies**: Voir `requirements.txt`
- **Cross-platform**: Compatible Windows, macOS, Linux

### Bonnes Pratiques
- **Utiliser les templates** pour nouveaux réseaux
- **Tester les modifications** avant intégration
- **Documenter** les nouveaux utilitaires
- **Maintenir la compatibilité** avec réseaux existants

## 🎯 Objectifs

### Réutilisabilité
- **Code commun**: Éviter la duplication
- **Templates**: Accélérer le développement
- **Standards**: Cohérence entre réseaux

### Maintenabilité
- **Tests unitaires**: Validation automatique
- **Documentation**: Guide complet
- **Modularité**: Composants indépendants

### Extensibilité
- **Architecture ouverte**: Facile à étendre
- **Interfaces claires**: APIs bien définies
- **Backward compatibility**: Compatibilité ascendante

**Outils de support pour un développement efficace et cohérent !** 🔧
