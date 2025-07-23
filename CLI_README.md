# 🔬 CLI Moderne - Inversion d'Anneaux Holographiques

**Auteur:** Oussama GUELFAA  
**Date:** 08 - 07 - 2025  
**Version:** 1.0.0

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Installation](#installation)
- [Démarrage Rapide](#démarrage-rapide)
- [Commandes Principales](#commandes-principales)
- [Configuration](#configuration)
- [Interface Interactive](#interface-interactive)
- [Exemples d'Utilisation](#exemples-dutilisation)
- [Architecture](#architecture)
- [Développement](#développement)

## 🎯 Vue d'ensemble

Ce CLI moderne et interactif fournit une interface unifiée pour toutes les fonctionnalités de notre système de réseaux neuronaux pour la prédiction des paramètres gap et L_écran dans l'analyse d'anneaux holographiques.

### ✨ Fonctionnalités Principales

- **Interface Interactive** : Menus riches avec navigation intuitive
- **Commandes Hiérarchiques** : Structure organisée (train, predict, test, analyze, visualize, config)
- **Formatage Riche** : Texte coloré, tableaux formatés, panneaux stylés
- **Barres de Progression** : Suivi visuel des opérations longues
- **Graphiques ASCII** : Visualisations directement dans le terminal
- **Configuration Avancée** : Gestion de profils YAML avec validation
- **Modes d'Utilisation** : Production, recherche, démo adaptés aux besoins

## 🚀 Installation

### Prérequis

- Python 3.8+
- PyTorch (pour les modèles de réseaux neuronaux)
- Dépendances scientifiques (numpy, pandas, matplotlib, scipy)

### Installation des Dépendances

```bash
# Installation des dépendances principales
pip install -r requirements.txt

# Pour l'interface interactive complète (optionnel)
pip install questionary

# Pour les graphiques avancés (optionnel)
pip install termgraph
```

### Vérification de l'Installation

```bash
# Test de base
python hologram_cli.py --version

# Démonstration complète
python demo_cli.py
```

## ⚡ Démarrage Rapide

### Mode Interactif (Recommandé)

```bash
python hologram_cli.py --interactive
```

### Commandes Directes

```bash
# Afficher l'aide
python hologram_cli.py --help

# Entraîner un modèle
python hologram_cli.py train start --model precision

# Faire une prédiction
python hologram_cli.py predict single --input data.mat

# Tester un modèle
python hologram_cli.py test accuracy --model production
```

## 📚 Commandes Principales

### 🏋️ Train - Entraînement

Commandes pour l'entraînement des modèles de réseaux neuronaux.

```bash
# Démarrer un entraînement
python hologram_cli.py train start --model precision --epochs 200

# Valider un modèle avant entraînement
python hologram_cli.py train validate --model precision

# Lister les modèles disponibles
python hologram_cli.py train list-models
```

**Options principales :**
- `--model` : Type de modèle (precision, production, research, gap-only)
- `--epochs` : Nombre d'époques d'entraînement
- `--batch-size` : Taille des batches
- `--learning-rate` : Taux d'apprentissage
- `--device` : Device de calcul (auto, cpu, cuda, mps)

### 🔮 Predict - Prédiction

Commandes pour faire des prédictions avec les modèles entraînés.

```bash
# Prédiction sur un fichier
python hologram_cli.py predict single --input data.mat --model precision

# Traitement par lot
python hologram_cli.py predict batch --input-dir data/ --model production

# Simulation d'anneau
python hologram_cli.py predict simulate --gap 0.025 --L-ecran 10.0
```

**Options principales :**
- `--input` : Fichier d'entrée (.mat, .csv, .json)
- `--model` : Modèle à utiliser
- `--output` : Fichier de sortie
- `--format` : Format de sortie (json, csv, table)
- `--confidence` : Seuil de confiance minimum

### 🧪 Test - Tests et Évaluation

Commandes pour tester et évaluer les modèles.

```bash
# Test de précision
python hologram_cli.py test accuracy --model precision --detailed

# Test de robustesse au bruit
python hologram_cli.py test robustness --model production --noise-levels "0,2,5,10"

# Test sur données expérimentales
python hologram_cli.py test experimental --model precision --experimental-data exp.mat

# Comparaison de modèles
python hologram_cli.py test compare --model precision --baseline-model production
```

**Options principales :**
- `--tolerance-gap` : Tolérance pour le gap (µm)
- `--tolerance-L-ecran` : Tolérance pour L_écran (µm)
- `--noise-levels` : Niveaux de bruit à tester
- `--detailed` : Rapport détaillé

### 📊 Analyze - Analyse

Commandes pour l'analyse de datasets et de résultats.

```bash
# Analyser un dataset
python hologram_cli.py analyze dataset --dataset-path data/ --generate-plots

# Analyser des résultats de modèle
python hologram_cli.py analyze results --results-file predictions.json

# Comparer deux modèles
python hologram_cli.py analyze compare --model1-results m1.json --model2-results m2.json

# Analyser la distribution des paramètres
python hologram_cli.py analyze distribution --data-path data/ --parameter gap
```

### 📈 Visualize - Visualisation

Commandes pour créer des visualisations et graphiques.

```bash
# Visualiser des résultats
python hologram_cli.py visualize results --results-file predictions.json

# Visualiser un dataset
python hologram_cli.py visualize dataset --dataset-path data/ --ascii-only

# Visualiser des profils d'anneaux
python hologram_cli.py visualize rings --profile-data profiles.mat --n-profiles 5

# Graphique ASCII personnalisé
python hologram_cli.py visualize ascii-chart --data "1,2,3,4,5" --title "Test"
```

### ⚙️ Config - Configuration

Commandes pour gérer la configuration du CLI.

```bash
# Afficher la configuration
python hologram_cli.py config show

# Changer de profil
python hologram_cli.py config set-profile --profile recherche

# Modifier une valeur
python hologram_cli.py config set --key ui.theme --value green

# Lister les profils
python hologram_cli.py config list-profiles

# Exporter/Importer
python hologram_cli.py config export --output my_config.yaml
python hologram_cli.py config import --config-file backup.yaml
```

## ⚙️ Configuration

### Profils Disponibles

Le CLI utilise des profils de configuration pour s'adapter à différents cas d'usage :

#### 🏭 Production
- **Usage** : Utilisation en production avec précision maximale
- **Modèle** : Ultra-précision (±0.007µm)
- **Device** : Auto-détection (GPU préféré)
- **Caractéristiques** : Seuil de confiance élevé, sauvegarde automatique

#### 🔬 Recherche
- **Usage** : Expérimentation et développement
- **Modèle** : Production avec fonctionnalités expérimentales
- **Device** : Auto-détection
- **Caractéristiques** : Logging verbeux, tolérances ajustables

#### 🎯 Démo
- **Usage** : Démonstrations et tests rapides
- **Modèle** : Standard
- **Device** : CPU (compatibilité maximale)
- **Caractéristiques** : Exécution rapide, pas de sauvegarde

### Fichier de Configuration

Le fichier `cli/config/default.yaml` contient la configuration par défaut :

```yaml
version: "1.0.0"
active_profile: "production"

profiles:
  production:
    model_path: "Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25"
    data_path: "data_generation/dataset_2D"
    output_path: "cli/outputs"
    batch_size: 32
    device: "auto"
    precision: "high"

ui:
  theme: "blue"
  progress_bars: true
  ascii_graphs: true
  rich_tables: true

defaults:
  train:
    epochs: 100
    early_stopping: true
  test:
    tolerance_gap: 0.007
    tolerance_L_ecran: 0.5
```

## 🖱️ Interface Interactive

### Activation

```bash
python hologram_cli.py --interactive
```

### Fonctionnalités

- **Menus de Navigation** : Sélection avec flèches ↑↓
- **Assistants Pas-à-Pas** : Configuration guidée
- **Validation en Temps Réel** : Vérification des entrées
- **Aperçus Visuels** : Tableaux et panneaux formatés

### Assistants Disponibles

1. **Assistant d'Entraînement** : Configuration complète d'entraînement
2. **Assistant de Prédiction** : Paramétrage des prédictions
3. **Assistant d'Analyse** : Configuration des analyses
4. **Gestionnaire de Configuration** : Modification interactive des paramètres

## 💡 Exemples d'Utilisation

### Workflow Complet de Production

```bash
# 1. Configurer le profil production
python hologram_cli.py config set-profile --profile production

# 2. Valider le modèle
python hologram_cli.py train validate --model precision

# 3. Tester la précision
python hologram_cli.py test accuracy --model precision --detailed

# 4. Faire des prédictions
python hologram_cli.py predict batch --input-dir new_data/ --output results.json

# 5. Analyser les résultats
python hologram_cli.py analyze results --results-file results.json

# 6. Créer des visualisations
python hologram_cli.py visualize results --results-file results.json --output-dir plots/
```

### Workflow de Recherche

```bash
# 1. Activer le profil recherche
python hologram_cli.py config set-profile --profile recherche

# 2. Analyser le dataset
python hologram_cli.py analyze dataset --dataset-path data/ --detailed --generate-plots

# 3. Entraîner avec paramètres personnalisés
python hologram_cli.py train start --model research --epochs 300 --learning-rate 0.0005

# 4. Test de robustesse
python hologram_cli.py test robustness --model research --noise-levels "0,1,2,5,10,20"

# 5. Comparaison avec modèle de référence
python hologram_cli.py test compare --model research --baseline-model production
```

### Démonstration Rapide

```bash
# Mode démo pour présentation
python hologram_cli.py config set-profile --profile demo

# Simulation d'anneau avec visualisation
python hologram_cli.py predict simulate --gap 0.05 --L-ecran 15.0 --noise-level 2.0

# Graphiques ASCII
python hologram_cli.py visualize ascii-chart --data "0.1,0.08,0.06,0.04,0.02" --title "Convergence"

# Démonstration complète
python demo_cli.py
```

## 🏗️ Architecture

### Structure des Modules

```
cli/
├── __init__.py              # Module principal
├── main.py                  # Point d'entrée CLI
├── interactive.py           # Interface interactive
├── commands/                # Commandes CLI
│   ├── train.py            # Commandes d'entraînement
│   ├── predict.py          # Commandes de prédiction
│   ├── test.py             # Commandes de test
│   ├── analyze.py          # Commandes d'analyse
│   ├── visualize.py        # Commandes de visualisation
│   └── config.py           # Commandes de configuration
├── config/                  # Gestion de configuration
│   ├── config_manager.py   # Gestionnaire principal
│   ├── validators.py       # Validateurs
│   ├── profiles.py         # Gestionnaire de profils
│   └── default.yaml        # Configuration par défaut
└── utils/                   # Utilitaires
    ├── cli_utils.py        # Utilitaires généraux
    ├── formatters.py       # Formatage riche et ASCII
    └── validators.py       # Validateurs de chemins/modèles
```

### Technologies Utilisées

- **Click** : Framework CLI principal
- **Rich** : Formatage riche et interface utilisateur
- **Questionary** : Menus interactifs (optionnel)
- **PyYAML** : Gestion des configurations
- **NumPy/Pandas** : Traitement des données
- **PyTorch** : Intégration avec les modèles

## 🛠️ Développement

### Ajouter une Nouvelle Commande

1. Créer le module dans `cli/commands/`
2. Définir le groupe de commandes avec `@click.group()`
3. Ajouter les sous-commandes avec `@click.command()`
4. Intégrer dans `cli/main.py`

### Exemple de Nouvelle Commande

```python
# cli/commands/export.py
import click
from rich.console import Console

console = Console()

@click.group(name='export')
def export_group():
    """📤 Commandes d'export."""
    pass

@export_group.command()
@click.option('--format', type=click.Choice(['pdf', 'html']))
def report(format):
    """Exporte un rapport."""
    console.print(f"[green]Export en {format}[/green]")
```

### Tests

```bash
# Test de base
python hologram_cli.py --help

# Test interactif
python hologram_cli.py --interactive

# Démonstration complète
python demo_cli.py

# Test de configuration
python hologram_cli.py config validate
```

### Extension

Le CLI est conçu pour être facilement extensible :

- **Nouveaux profils** : Ajouter dans `cli/config/profiles.py`
- **Nouveaux formatters** : Étendre `cli/utils/formatters.py`
- **Nouvelles validations** : Ajouter dans `cli/utils/validators.py`
- **Nouveaux assistants** : Étendre `cli/interactive.py`

---

## 📞 Support

Pour toute question ou problème :

1. Consulter l'aide intégrée : `python hologram_cli.py --help`
2. Lancer la démonstration : `python demo_cli.py`
3. Vérifier la configuration : `python hologram_cli.py config validate`

**Auteur :** Oussama GUELFAA  
**Email :** guelfaao@gmail.com  
**Date :** 08 - 07 - 2025
