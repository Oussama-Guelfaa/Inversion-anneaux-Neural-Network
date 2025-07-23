#!/usr/bin/env python3
"""
Script de Configuration du CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Script pour configurer et installer le CLI Inversion d'Anneaux Holographiques.
"""

import sys
import subprocess
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich import print as rprint

console = Console()

def check_python_version():
    """Vérifie la version de Python."""
    rprint("[blue]🐍 Vérification de la version Python...[/blue]")
    
    if sys.version_info < (3, 8):
        rprint("[red]❌ Python 3.8+ requis. Version actuelle: {}.{}.{}[/red]".format(
            sys.version_info.major, sys.version_info.minor, sys.version_info.micro
        ))
        return False
    
    rprint(f"[green]✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}[/green]")
    return True

def install_dependencies():
    """Installe les dépendances."""
    rprint("\n[blue]📦 Installation des dépendances...[/blue]")
    
    try:
        # Vérifier si requirements.txt existe
        if not Path("requirements.txt").exists():
            rprint("[red]❌ Fichier requirements.txt non trouvé[/red]")
            return False
        
        # Installer les dépendances
        rprint("[yellow]Installation en cours...[/yellow]")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            rprint("[green]✅ Dépendances installées avec succès[/green]")
            return True
        else:
            rprint(f"[red]❌ Erreur lors de l'installation: {result.stderr}[/red]")
            return False
            
    except Exception as e:
        rprint(f"[red]❌ Erreur: {str(e)}[/red]")
        return False

def create_directories():
    """Crée les dossiers nécessaires."""
    rprint("\n[blue]📁 Création des dossiers...[/blue]")
    
    directories = [
        "cli/outputs",
        "cli/outputs/production",
        "cli/outputs/research", 
        "cli/outputs/demo",
        "cli/exports",
        "cli/logs"
    ]
    
    for directory in track(directories, description="Création des dossiers..."):
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            rprint(f"[green]✅ {directory}[/green]")
        except Exception as e:
            rprint(f"[red]❌ Erreur pour {directory}: {str(e)}[/red]")
            return False
    
    return True

def test_cli_installation():
    """Teste l'installation du CLI."""
    rprint("\n[blue]🧪 Test de l'installation...[/blue]")
    
    try:
        # Test d'importation des modules principaux
        test_imports = [
            ("cli.main", "Module principal"),
            ("cli.config.config_manager", "Gestionnaire de configuration"),
            ("cli.utils.cli_utils", "Utilitaires CLI"),
            ("cli.interactive", "Interface interactive")
        ]
        
        for module, description in test_imports:
            try:
                __import__(module)
                rprint(f"[green]✅ {description}[/green]")
            except ImportError as e:
                rprint(f"[red]❌ {description}: {str(e)}[/red]")
                return False
        
        # Test de la commande CLI
        result = subprocess.run([
            sys.executable, "hologram_cli.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            rprint("[green]✅ CLI fonctionnel[/green]")
            return True
        else:
            rprint(f"[red]❌ Erreur CLI: {result.stderr}[/red]")
            return False
            
    except Exception as e:
        rprint(f"[red]❌ Erreur de test: {str(e)}[/red]")
        return False

def create_config_file():
    """Crée le fichier de configuration par défaut."""
    rprint("\n[blue]⚙️  Création de la configuration...[/blue]")
    
    try:
        from cli.config.config_manager import CLIConfig
        
        # Créer la configuration par défaut
        config = CLIConfig()
        config.save_config()
        
        rprint("[green]✅ Configuration créée[/green]")
        return True
        
    except Exception as e:
        rprint(f"[red]❌ Erreur de configuration: {str(e)}[/red]")
        return False

def show_usage_instructions():
    """Affiche les instructions d'utilisation."""
    instructions_panel = Panel(
        "[bold blue]🚀 CLI Installé avec Succès ![/bold blue]\n\n"
        "[green]Commandes pour commencer:[/green]\n\n"
        "• [cyan]python hologram_cli.py --interactive[/cyan]\n"
        "  Interface interactive complète\n\n"
        "• [cyan]python demo_cli.py[/cyan]\n"
        "  Démonstration des fonctionnalités\n\n"
        "• [cyan]python test_cli_quick.py[/cyan]\n"
        "  Tests rapides de validation\n\n"
        "• [cyan]python hologram_cli.py --help[/cyan]\n"
        "  Aide complète des commandes\n\n"
        "[yellow]Documentation:[/yellow]\n"
        "• CLI_README.md - Guide complet\n"
        "• EXTENSION_GUIDE.md - Guide d'extension\n\n"
        "[dim]Le CLI est maintenant prêt à être utilisé ![/dim]",
        title="[bold green]Installation Terminée[/bold green]",
        border_style="green"
    )
    console.print(instructions_panel)

def main():
    """Fonction principale de setup."""
    # Affichage d'accueil
    welcome_panel = Panel(
        "[bold blue]🔬 Configuration du CLI Inversion d'Anneaux Holographiques[/bold blue]\n\n"
        "Ce script va configurer votre environnement pour utiliser le CLI moderne.\n\n"
        "[yellow]Étapes de configuration:[/yellow]\n"
        "1. Vérification de Python\n"
        "2. Installation des dépendances\n"
        "3. Création des dossiers\n"
        "4. Configuration par défaut\n"
        "5. Tests de validation\n\n"
        "[dim]Auteur: Oussama GUELFAA - 08/07/2025[/dim]",
        title="[bold]Configuration CLI[/bold]",
        border_style="blue"
    )
    console.print(welcome_panel)
    
    # Étapes de configuration
    steps = [
        ("Vérification Python", check_python_version),
        ("Installation dépendances", install_dependencies),
        ("Création dossiers", create_directories),
        ("Configuration", create_config_file),
        ("Tests", test_cli_installation)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        rprint(f"\n[bold blue]📋 Étape: {step_name}[/bold blue]")
        
        if step_func():
            success_count += 1
        else:
            rprint(f"[red]❌ Échec de l'étape: {step_name}[/red]")
            break
    
    # Résumé
    if success_count == len(steps):
        rprint("\n[bold green]🎉 Configuration terminée avec succès ![/bold green]")
        show_usage_instructions()
        return 0
    else:
        rprint(f"\n[bold red]❌ Configuration échouée à l'étape {success_count + 1}/{len(steps)}[/bold red]")
        rprint("[yellow]Vérifiez les erreurs ci-dessus et réessayez.[/yellow]")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        rprint("\n[yellow]Configuration interrompue par l'utilisateur[/yellow]")
        sys.exit(1)
    except Exception as e:
        rprint(f"\n[red]Erreur inattendue: {str(e)}[/red]")
        sys.exit(1)
