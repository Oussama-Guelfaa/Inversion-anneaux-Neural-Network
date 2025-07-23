# 🛠️ Guide d'Extension du CLI

**Auteur:** Oussama GUELFAA  
**Date:** 08 - 07 - 2025

Ce guide explique comment étendre et personnaliser le CLI Inversion d'Anneaux Holographiques.

## 📋 Table des Matières

- [Architecture du CLI](#architecture-du-cli)
- [Ajouter une Nouvelle Commande](#ajouter-une-nouvelle-commande)
- [Créer un Nouveau Profil](#créer-un-nouveau-profil)
- [Étendre les Formateurs](#étendre-les-formateurs)
- [Ajouter des Validateurs](#ajouter-des-validateurs)
- [Personnaliser l'Interface Interactive](#personnaliser-linterface-interactive)
- [Intégrer de Nouveaux Modèles](#intégrer-de-nouveaux-modèles)
- [Bonnes Pratiques](#bonnes-pratiques)

## 🏗️ Architecture du CLI

### Structure des Modules

```
cli/
├── main.py                  # Point d'entrée principal
├── interactive.py           # Interface interactive
├── commands/                # Commandes CLI
│   ├── __init__.py
│   ├── train.py            # Commandes d'entraînement
│   ├── predict.py          # Commandes de prédiction
│   ├── test.py             # Commandes de test
│   ├── analyze.py          # Commandes d'analyse
│   ├── visualize.py        # Commandes de visualisation
│   └── config.py           # Commandes de configuration
├── config/                  # Gestion de configuration
│   ├── config_manager.py   # Gestionnaire principal
│   ├── validators.py       # Validateurs de config
│   ├── profiles.py         # Gestionnaire de profils
│   └── default.yaml        # Configuration par défaut
└── utils/                   # Utilitaires
    ├── cli_utils.py        # Utilitaires généraux
    ├── formatters.py       # Formatage riche et ASCII
    └── validators.py       # Validateurs de chemins/modèles
```

### Principes de Conception

1. **Modularité** : Chaque fonctionnalité dans son propre module
2. **Extensibilité** : Interfaces claires pour l'extension
3. **Configuration** : Tout paramétrable via YAML
4. **Validation** : Validation stricte des entrées
5. **Interface Riche** : Formatage et interactivité avancés

## ➕ Ajouter une Nouvelle Commande

### 1. Créer le Module de Commande

Créez un nouveau fichier dans `cli/commands/` :

```python
# cli/commands/export.py
#!/usr/bin/env python3
"""
Commandes d'Export CLI

Auteur: Votre Nom
Date: Date

Commandes pour exporter des données et résultats.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

@click.group(name='export')
@click.pass_context
def export_group(ctx):
    """📤 Commandes d'export de données et résultats."""
    pass

@export_group.command()
@click.option('--input-file', '-i', required=True,
              help='Fichier de données à exporter')
@click.option('--output-format', '-f',
              type=click.Choice(['pdf', 'html', 'excel']),
              default='pdf',
              help='Format de sortie')
@click.option('--template', '-t',
              help='Template à utiliser')
@click.pass_context
def report(ctx, input_file, output_format, template):
    """
    Exporte un rapport formaté.
    
    Exemples:
        hologram_cli export report --input-file results.json --output-format pdf
    """
    utils = ctx.obj['utils']
    config = ctx.obj['config']
    
    # Validation des entrées
    if not Path(input_file).exists():
        utils.show_error(f"Fichier non trouvé: {input_file}")
        return
    
    # Afficher les informations d'export
    info_panel = Panel(
        f"[bold blue]Export de rapport[/bold blue]\n\n"
        f"📁 Fichier: {input_file}\n"
        f"📄 Format: {output_format}\n"
        f"🎨 Template: {template or 'Par défaut'}",
        title="[bold]Configuration d'Export[/bold]",
        border_style="green"
    )
    console.print(info_panel)
    
    try:
        # Logique d'export ici
        _perform_export(input_file, output_format, template, utils)
        utils.show_success(f"Rapport exporté en {output_format}")
        
    except Exception as e:
        utils.show_error(f"Erreur lors de l'export: {str(e)}")

def _perform_export(input_file, output_format, template, utils):
    """Effectue l'export réel."""
    # Implémentation de l'export
    pass
```

### 2. Intégrer dans le CLI Principal

Modifiez `cli/main.py` pour ajouter la nouvelle commande :

```python
# Dans cli/main.py
from cli.commands import train, predict, test, analyze, visualize, config, export

# Ajouter la commande au CLI principal
main_cli.add_command(export.export_group)
```

### 3. Mettre à Jour l'Interface Interactive

Modifiez `cli/interactive.py` pour inclure la nouvelle commande :

```python
# Dans InteractiveMenu.show_main_menu()
choices = [
    questionary.Choice("🏋️  Entraîner un modèle", value="train"),
    questionary.Choice("🔮 Faire des prédictions", value="predict"),
    questionary.Choice("🧪 Tester un modèle", value="test"),
    questionary.Choice("📊 Analyser des données", value="analyze"),
    questionary.Choice("📈 Créer des visualisations", value="visualize"),
    questionary.Choice("📤 Exporter des rapports", value="export"),  # Nouveau
    questionary.Choice("⚙️  Gérer la configuration", value="config"),
    # ...
]
```

## 🎯 Créer un Nouveau Profil

### 1. Définir le Profil

Ajoutez votre profil dans `cli/config/profiles.py` :

```python
# Dans ConfigProfiles.get_default_profiles()
'mon_profil_custom': {
    'description': 'Profil personnalisé pour cas d\'usage spécifique',
    'model_path': 'Mon_Modele_Custom',
    'data_path': 'data_custom/',
    'output_path': 'cli/outputs/custom',
    'batch_size': 16,
    'device': 'auto',
    'precision': 'high',
    'confidence_threshold': 0.9,
    'tolerance_gap': 0.005,
    'tolerance_L_ecran': 0.3,
    'custom_parameter': 'valeur_specifique',
    'use_advanced_features': True
}
```

### 2. Ajouter la Validation

Étendez la validation dans `cli/config/validators.py` :

```python
# Dans ConfigValidator._validate_values()
if 'custom_parameter' in profile_config:
    custom_param = profile_config['custom_parameter']
    valid_values = ['valeur1', 'valeur2', 'valeur_specifique']
    if custom_param not in valid_values:
        warnings.append(f"Paramètre custom non standard: {custom_param}")
```

### 3. Optimiser pour le Système

Ajoutez l'optimisation dans `ConfigProfiles.optimize_profile_for_system()` :

```python
# Optimisations spécifiques au profil
if 'custom_parameter' in optimized:
    if optimized['device'] == 'cpu':
        optimized['custom_parameter'] = 'valeur_optimisee_cpu'
```

## 🎨 Étendre les Formateurs

### 1. Ajouter un Nouveau Formateur Rich

Étendez `RichFormatter` dans `cli/utils/formatters.py` :

```python
def create_custom_panel(self, data: Dict[str, Any], 
                       title: str = "Données Personnalisées") -> Panel:
    """
    Crée un panneau personnalisé pour vos données.
    
    Args:
        data (Dict): Données à afficher
        title (str): Titre du panneau
        
    Returns:
        Panel: Panneau formaté
    """
    content_lines = []
    
    for key, value in data.items():
        if isinstance(value, float):
            formatted_value = f"{value:.6f}"
        elif isinstance(value, list):
            formatted_value = f"[{len(value)} éléments]"
        else:
            formatted_value = str(value)
        
        content_lines.append(f"[cyan]{key}:[/cyan] [green]{formatted_value}[/green]")
    
    content = "\n".join(content_lines)
    
    return Panel(
        content,
        title=f"[bold {self.theme}]{title}[/bold {self.theme}]",
        border_style=self.theme
    )
```

### 2. Ajouter un Nouveau Graphique ASCII

Étendez `ASCIIGraphs` :

```python
def custom_visualization(self, data: List[float], 
                        title: str = "Visualisation Personnalisée",
                        width: int = None) -> str:
    """
    Crée une visualisation ASCII personnalisée.
    
    Args:
        data (List[float]): Données à visualiser
        title (str): Titre de la visualisation
        width (int): Largeur de la visualisation
        
    Returns:
        str: Visualisation ASCII
    """
    w = width or self.width
    
    if not data:
        return f"{title}\n(Aucune donnée à afficher)"
    
    # Votre logique de visualisation personnalisée
    lines = []
    lines.append(f"🎨 {title}")
    lines.append("─" * w)
    
    # Implémentation de votre visualisation
    for i, value in enumerate(data):
        bar_length = int((value / max(data)) * (w - 10))
        bar = "█" * bar_length
        lines.append(f"Item {i:2d} │{bar:<{w-10}} {value:.3f}")
    
    lines.append("─" * w)
    
    return "\n".join(lines)
```

## ✅ Ajouter des Validateurs

### 1. Créer un Nouveau Validateur

Ajoutez dans `cli/utils/validators.py` :

```python
class CustomValidator:
    """
    Validateur pour vos données personnalisées.
    """
    
    @staticmethod
    def validate_custom_data(data_path: str, 
                           expected_format: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Valide des données dans un format personnalisé.
        
        Args:
            data_path (str): Chemin vers les données
            expected_format (str): Format attendu
            
        Returns:
            Tuple[bool, str, Dict]: (Valide, Message, Informations)
        """
        path = Path(data_path)
        info = {
            'format_detected': 'unknown',
            'data_quality': 'unknown',
            'compatibility': False
        }
        
        if not path.exists():
            return False, f"Données non trouvées: {data_path}", info
        
        # Votre logique de validation
        try:
            # Analyser le format
            if path.suffix == '.custom':
                info['format_detected'] = 'custom'
                # Validation spécifique
                
            info['compatibility'] = True
            return True, "Données valides", info
            
        except Exception as e:
            return False, f"Erreur de validation: {str(e)}", info
```

### 2. Intégrer dans les Utilitaires

Ajoutez dans `CLIUtils` :

```python
def validate_custom_input(self, input_path: str) -> bool:
    """
    Valide une entrée personnalisée.
    
    Args:
        input_path (str): Chemin de l'entrée
        
    Returns:
        bool: True si valide
    """
    from cli.utils.validators import CustomValidator
    
    valid, message, info = CustomValidator.validate_custom_data(
        input_path, 'custom'
    )
    
    if not valid:
        self.show_error(f"Validation échouée: {message}")
    
    return valid
```

## 🖱️ Personnaliser l'Interface Interactive

### 1. Créer un Nouvel Assistant

Ajoutez dans `cli/interactive.py` :

```python
def configure_custom_operation(self) -> Dict[str, Any]:
    """
    Assistant de configuration pour opération personnalisée.
    
    Returns:
        Dict[str, Any]: Configuration personnalisée
    """
    rprint("\n[bold blue]🎯 Assistant d'Opération Personnalisée[/bold blue]")
    
    config = {}
    
    try:
        import questionary
        
        # Paramètres personnalisés
        config['operation_type'] = questionary.select(
            "Type d'opération:",
            choices=[
                questionary.Choice("Analyse avancée", "advanced_analysis"),
                questionary.Choice("Export personnalisé", "custom_export"),
                questionary.Choice("Traitement spécial", "special_processing")
            ]
        ).ask()
        
        config['custom_parameter'] = questionary.text(
            "Paramètre personnalisé:",
            validate=lambda x: len(x) > 0
        ).ask()
        
        config['enable_advanced'] = questionary.confirm(
            "Activer les fonctionnalités avancées ?",
            default=False
        ).ask()
        
    except ImportError:
        # Fallback sans questionary
        config['operation_type'] = Prompt.ask(
            "Type d'opération",
            choices=["advanced_analysis", "custom_export", "special_processing"],
            default="advanced_analysis"
        )
        config['custom_parameter'] = Prompt.ask("Paramètre personnalisé")
        config['enable_advanced'] = Confirm.ask("Fonctionnalités avancées ?", default=False)
    
    return config
```

### 2. Ajouter un Nouvel Assistant Complet

```python
def run_custom_wizard(self, utils, config) -> bool:
    """
    Assistant complet pour opération personnalisée.
    
    Args:
        utils: Utilitaires CLI
        config: Configuration CLI
        
    Returns:
        bool: True si l'opération a été lancée
    """
    rprint("\n[bold blue]🧙‍♂️ Assistant Personnalisé[/bold blue]")
    
    # Configuration
    custom_config = self.configure_custom_operation()
    
    # Validation
    if not self._validate_custom_config(custom_config, utils):
        return False
    
    # Résumé et confirmation
    self._show_custom_summary(custom_config)
    
    if Confirm.ask("\nLancer l'opération personnalisée ?"):
        rprint("[green]🚀 Opération lancée ![/green]")
        return True
    
    return False
```

## 🤖 Intégrer de Nouveaux Modèles

### 1. Étendre la Détection de Modèles

Modifiez `CLIUtils.get_available_models()` :

```python
def _get_model_type(self, model_name: str) -> str:
    """Détermine le type de modèle basé sur son nom."""
    if 'PRECISION' in model_name:
        return 'ultra-precision'
    elif 'FINAL' in model_name:
        return 'production'
    elif 'CUSTOM' in model_name:  # Nouveau type
        return 'custom-model'
    elif '2D' in model_name:
        return 'dual-parameter'
    elif '1D' in model_name:
        return 'gap-only'
    else:
        return 'unknown'
```

### 2. Ajouter la Validation Spécifique

Dans `ModelValidator` :

```python
@staticmethod
def validate_custom_model(model_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Valide un modèle personnalisé.
    
    Args:
        model_path (str): Chemin du modèle
        
    Returns:
        Tuple[bool, str, Dict]: (Valide, Message, Informations)
    """
    info = {
        'model_type': 'custom',
        'custom_features': [],
        'compatibility': 'unknown'
    }
    
    path = Path(model_path)
    
    # Vérifications spécifiques aux modèles personnalisés
    if (path / 'custom_config.yaml').exists():
        info['custom_features'].append('configuration_personnalisee')
    
    if (path / 'preprocessing').exists():
        info['custom_features'].append('preprocessing_integre')
    
    # Validation de compatibilité
    info['compatibility'] = 'compatible'
    
    return True, "Modèle personnalisé valide", info
```

## 📝 Bonnes Pratiques

### 1. Structure du Code

- **Modules séparés** : Une fonctionnalité = un module
- **Imports clairs** : Imports explicites et organisés
- **Documentation** : Docstrings pour toutes les fonctions publiques
- **Type hints** : Annotations de type pour la clarté

### 2. Gestion des Erreurs

```python
try:
    # Opération risquée
    result = risky_operation()
    utils.show_success("Opération réussie")
except SpecificException as e:
    utils.show_error(f"Erreur spécifique: {str(e)}")
except Exception as e:
    utils.show_error(f"Erreur inattendue: {str(e)}")
```

### 3. Configuration

- **Validation stricte** : Toujours valider les configurations
- **Valeurs par défaut** : Fournir des valeurs sensées
- **Documentation** : Commenter les paramètres complexes

### 4. Interface Utilisateur

- **Feedback visuel** : Barres de progression, messages colorés
- **Validation en temps réel** : Vérifier les entrées immédiatement
- **Aide contextuelle** : Messages d'aide clairs et utiles

### 5. Tests

```python
def test_nouvelle_fonctionnalite(self):
    """Test la nouvelle fonctionnalité."""
    # Arrange
    input_data = self._create_test_data()
    
    # Act
    result = nouvelle_fonctionnalite(input_data)
    
    # Assert
    self.assertIsNotNone(result)
    self.assertEqual(result.status, 'success')
```

### 6. Documentation

- **README mis à jour** : Documenter les nouvelles fonctionnalités
- **Exemples d'usage** : Fournir des exemples concrets
- **Guide de migration** : Si changements breaking

## 🚀 Déploiement des Extensions

### 1. Tests

```bash
# Tester votre extension
python test_cli_quick.py

# Tests unitaires spécifiques
python -m pytest tests/test_mon_extension.py
```

### 2. Documentation

```bash
# Mettre à jour la documentation
# Ajouter des exemples dans CLI_README.md
# Documenter dans EXTENSION_GUIDE.md
```

### 3. Intégration

```bash
# Vérifier l'intégration complète
python hologram_cli.py --help
python hologram_cli.py ma-nouvelle-commande --help
```

---

## 📞 Support pour le Développement

Pour toute question sur l'extension du CLI :

1. Consulter ce guide d'extension
2. Examiner le code existant pour les patterns
3. Tester avec `python test_cli_quick.py`
4. Utiliser la démonstration : `python demo_cli.py`

**Auteur :** Oussama GUELFAA  
**Email :** guelfaao@gmail.com  
**Date :** 08 - 07 - 2025
