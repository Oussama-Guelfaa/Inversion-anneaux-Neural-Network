#!/usr/bin/env python3
"""
Test Rapide du CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Script de test rapide pour vérifier le bon fonctionnement du CLI.
"""

import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

console = Console()

def test_imports():
    """Test l'importation de tous les modules."""
    rprint("[blue]🔍 Test des importations...[/blue]")
    
    try:
        from cli.main import main_cli
        rprint("[green]✅ cli.main[/green]")
        
        from cli.config.config_manager import CLIConfig
        rprint("[green]✅ cli.config.config_manager[/green]")
        
        from cli.config.profiles import ConfigProfiles
        rprint("[green]✅ cli.config.profiles[/green]")
        
        from cli.utils.cli_utils import CLIUtils
        rprint("[green]✅ cli.utils.cli_utils[/green]")
        
        from cli.utils.formatters import RichFormatter, ASCIIGraphs
        rprint("[green]✅ cli.utils.formatters[/green]")
        
        from cli.utils.validators import PathValidator, ModelValidator
        rprint("[green]✅ cli.utils.validators[/green]")
        
        from cli.interactive import InteractiveMenu
        rprint("[green]✅ cli.interactive[/green]")
        
        return True
        
    except ImportError as e:
        rprint(f"[red]❌ Erreur d'importation: {e}[/red]")
        return False

def test_configuration():
    """Test la configuration."""
    rprint("\n[blue]⚙️  Test de la configuration...[/blue]")
    
    try:
        from cli.config.config_manager import CLIConfig
        from cli.config.profiles import ConfigProfiles
        
        # Test création de configuration
        config = CLIConfig()
        rprint("[green]✅ Création de configuration[/green]")
        
        # Test profils par défaut
        profiles = ConfigProfiles.get_default_profiles()
        rprint(f"[green]✅ {len(profiles)} profils par défaut chargés[/green]")
        
        # Test validation
        from cli.config.validators import ConfigValidator
        is_valid, errors, warnings = ConfigValidator.validate_full_config(config.config)
        
        if is_valid:
            rprint("[green]✅ Configuration valide[/green]")
        else:
            rprint(f"[yellow]⚠️  Configuration avec {len(errors)} erreurs[/yellow]")
        
        return True
        
    except Exception as e:
        rprint(f"[red]❌ Erreur de configuration: {e}[/red]")
        return False

def test_utilities():
    """Test les utilitaires."""
    rprint("\n[blue]🛠️  Test des utilitaires...[/blue]")
    
    try:
        from cli.utils.cli_utils import CLIUtils
        
        utils = CLIUtils()
        
        # Test détection de device
        device = utils.detect_device()
        rprint(f"[green]✅ Device détecté: {device}[/green]")
        
        # Test validation de chemin
        is_valid = utils.validate_data_path('.')
        rprint(f"[green]✅ Validation de chemin: {is_valid}[/green]")
        
        # Test formatage de métriques
        metrics = {'r2': 0.95, 'mae': 0.01}
        formatted = utils.format_metrics(metrics)
        rprint(f"[green]✅ Formatage de métriques: {len(formatted)} caractères[/green]")
        
        return True
        
    except Exception as e:
        rprint(f"[red]❌ Erreur d'utilitaires: {e}[/red]")
        return False

def test_formatters():
    """Test les formateurs."""
    rprint("\n[blue]🎨 Test des formateurs...[/blue]")
    
    try:
        from cli.utils.formatters import RichFormatter, ASCIIGraphs
        
        # Test RichFormatter
        formatter = RichFormatter()
        panel = formatter.create_header_panel("Test", "Sous-titre")
        rprint("[green]✅ RichFormatter - Panneau créé[/green]")
        
        metrics = {'test': 0.95}
        table = formatter.create_metrics_table(metrics)
        rprint("[green]✅ RichFormatter - Tableau créé[/green]")
        
        # Test ASCIIGraphs
        graphs = ASCIIGraphs()
        data = [1, 2, 3, 4, 5]
        chart = graphs.line_chart(data, "Test Chart")
        rprint(f"[green]✅ ASCIIGraphs - Graphique créé ({len(chart)} caractères)[/green]")
        
        return True
        
    except Exception as e:
        rprint(f"[red]❌ Erreur de formatage: {e}[/red]")
        return False

def test_validators():
    """Test les validateurs."""
    rprint("\n[blue]🔍 Test des validateurs...[/blue]")
    
    try:
        from cli.utils.validators import PathValidator, ModelValidator
        
        # Test PathValidator
        valid, message = PathValidator.validate_file_path(__file__)
        rprint(f"[green]✅ PathValidator - Fichier: {valid}[/green]")
        
        valid, message = PathValidator.validate_directory_path('.')
        rprint(f"[green]✅ PathValidator - Dossier: {valid}[/green]")
        
        # Test ModelValidator (sur dossier courant)
        valid, message, info = ModelValidator.validate_model_directory('.')
        rprint(f"[green]✅ ModelValidator - Analyse: {len(info)} informations[/green]")
        
        return True
        
    except Exception as e:
        rprint(f"[red]❌ Erreur de validation: {e}[/red]")
        return False

def test_interactive():
    """Test l'interface interactive."""
    rprint("\n[blue]🖱️  Test de l'interface interactive...[/blue]")
    
    try:
        from cli.interactive import InteractiveMenu, InteractiveWizard
        
        menu = InteractiveMenu("Test Menu")
        rprint("[green]✅ InteractiveMenu créé[/green]")
        
        wizard = InteractiveWizard("Test Wizard")
        rprint("[green]✅ InteractiveWizard créé[/green]")
        
        # Test questionary (optionnel)
        try:
            import questionary
            rprint("[green]✅ Questionary disponible - Interface complète[/green]")
        except ImportError:
            rprint("[yellow]⚠️  Questionary non installé - Interface basique[/yellow]")
        
        return True
        
    except Exception as e:
        rprint(f"[red]❌ Erreur d'interface: {e}[/red]")
        return False

def test_cli_commands():
    """Test les commandes CLI."""
    rprint("\n[blue]💻 Test des commandes CLI...[/blue]")
    
    try:
        # Test importation des commandes
        from cli.commands import train, predict, test, analyze, visualize, config
        rprint("[green]✅ Toutes les commandes importées[/green]")
        
        # Test du point d'entrée principal
        from cli.main import main_cli
        rprint("[green]✅ Point d'entrée principal disponible[/green]")
        
        return True
        
    except Exception as e:
        rprint(f"[red]❌ Erreur de commandes: {e}[/red]")
        return False

def test_dependencies():
    """Test les dépendances."""
    rprint("\n[blue]📦 Test des dépendances...[/blue]")
    
    dependencies = [
        ('click', 'Framework CLI'),
        ('rich', 'Interface riche'),
        ('yaml', 'Configuration YAML'),
        ('numpy', 'Calculs numériques'),
        ('pandas', 'Manipulation de données'),
        ('pathlib', 'Gestion des chemins')
    ]
    
    missing = []
    
    for dep, description in dependencies:
        try:
            __import__(dep)
            rprint(f"[green]✅ {dep} - {description}[/green]")
        except ImportError:
            rprint(f"[red]❌ {dep} - {description}[/red]")
            missing.append(dep)
    
    # Dépendances optionnelles
    optional_deps = [
        ('questionary', 'Menus interactifs'),
        ('termgraph', 'Graphiques ASCII avancés'),
        ('torch', 'Modèles PyTorch')
    ]
    
    for dep, description in optional_deps:
        try:
            __import__(dep)
            rprint(f"[green]✅ {dep} - {description} (optionnel)[/green]")
        except ImportError:
            rprint(f"[yellow]⚠️  {dep} - {description} (optionnel)[/yellow]")
    
    return len(missing) == 0

def show_summary(results):
    """Affiche le résumé des tests."""
    rprint("\n[bold blue]📊 Résumé des Tests[/bold blue]")
    
    summary_table = Table(title="Résultats des Tests", show_header=True, header_style="bold blue")
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Statut", style="green")
    summary_table.add_column("Description", style="dim")
    
    test_descriptions = {
        'imports': 'Importation des modules',
        'configuration': 'Gestion de configuration',
        'utilities': 'Utilitaires CLI',
        'formatters': 'Formateurs Rich/ASCII',
        'validators': 'Validateurs',
        'interactive': 'Interface interactive',
        'commands': 'Commandes CLI',
        'dependencies': 'Dépendances'
    }
    
    for test_name, success in results.items():
        status = "✅ Réussi" if success else "❌ Échoué"
        description = test_descriptions.get(test_name, "Test")
        summary_table.add_row(test_name.title(), status, description)
    
    console.print(summary_table)
    
    # Statistiques
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    if passed_tests == total_tests:
        status_panel = Panel(
            f"[bold green]🎉 Tous les tests réussis ![/bold green]\n\n"
            f"✅ {passed_tests}/{total_tests} tests passés\n"
            f"🚀 Le CLI est prêt à être utilisé",
            title="[bold green]Succès[/bold green]",
            border_style="green"
        )
    else:
        failed_tests = total_tests - passed_tests
        status_panel = Panel(
            f"[bold yellow]⚠️  Tests partiellement réussis[/bold yellow]\n\n"
            f"✅ {passed_tests}/{total_tests} tests passés\n"
            f"❌ {failed_tests} test(s) échoué(s)\n"
            f"🔧 Vérifiez les dépendances et la configuration",
            title="[bold yellow]Attention[/bold yellow]",
            border_style="yellow"
        )
    
    console.print(status_panel)

def main():
    """Fonction principale de test."""
    rprint("[bold blue]🔬 Test Rapide du CLI Inversion d'Anneaux Holographiques[/bold blue]")
    rprint("[dim]Vérification du bon fonctionnement de tous les composants[/dim]\n")
    
    # Exécuter tous les tests
    tests = {
        'imports': test_imports,
        'configuration': test_configuration,
        'utilities': test_utilities,
        'formatters': test_formatters,
        'validators': test_validators,
        'interactive': test_interactive,
        'commands': test_cli_commands,
        'dependencies': test_dependencies
    }
    
    results = {}
    
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            rprint(f"[red]❌ Erreur dans {test_name}: {e}[/red]")
            results[test_name] = False
    
    # Afficher le résumé
    show_summary(results)
    
    # Code de sortie
    all_passed = all(results.values())
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
