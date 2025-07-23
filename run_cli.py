#!/usr/bin/env python3
"""
Lanceur Rapide du CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Script de lancement rapide avec menu de sélection.
"""

import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint

console = Console()

def show_welcome():
    """Affiche l'écran d'accueil."""
    welcome_panel = Panel(
        "[bold blue]🔬 Lanceur Rapide - CLI Inversion d'Anneaux Holographiques[/bold blue]\n\n"
        "[green]Sélectionnez une option pour démarrer rapidement:[/green]\n\n"
        "Ce lanceur vous permet d'accéder facilement aux fonctionnalités\n"
        "principales du CLI sans mémoriser les commandes.\n\n"
        "[dim]Auteur: Oussama GUELFAA - 08/07/2025[/dim]",
        title="[bold]Bienvenue[/bold]",
        border_style="blue"
    )
    console.print(welcome_panel)

def show_menu():
    """Affiche le menu principal."""
    menu_table = Table(title="Options Disponibles", show_header=True, header_style="bold blue")
    menu_table.add_column("Option", style="cyan", width=8)
    menu_table.add_column("Description", style="green")
    menu_table.add_column("Commande", style="dim")
    
    options = [
        ("1", "Interface Interactive Complète", "python hologram_cli.py --interactive"),
        ("2", "Démonstration du CLI", "python demo_cli.py"),
        ("3", "Tests Rapides", "python test_cli_quick.py"),
        ("4", "Configuration du CLI", "python setup_cli.py"),
        ("5", "Aide Générale", "python hologram_cli.py --help"),
        ("6", "Entraîner un Modèle", "python hologram_cli.py train start --model precision"),
        ("7", "Faire une Prédiction", "python hologram_cli.py predict single --input data.mat"),
        ("8", "Tester un Modèle", "python hologram_cli.py test accuracy --model production"),
        ("9", "Analyser des Données", "python hologram_cli.py analyze dataset --dataset-path data/"),
        ("10", "Visualiser des Résultats", "python hologram_cli.py visualize results --results-file test.json"),
        ("11", "Gérer la Configuration", "python hologram_cli.py config show"),
        ("0", "Quitter", "")
    ]
    
    for option, description, command in options:
        menu_table.add_row(option, description, command)
    
    console.print(menu_table)

def run_command(command):
    """Exécute une commande."""
    if not command:
        return
    
    rprint(f"\n[blue]🚀 Exécution: {command}[/blue]")
    
    try:
        # Séparer la commande en arguments
        args = command.split()
        
        # Exécuter la commande
        result = subprocess.run(args, cwd=Path.cwd())
        
        if result.returncode == 0:
            rprint("[green]✅ Commande exécutée avec succès[/green]")
        else:
            rprint(f"[yellow]⚠️  Commande terminée avec code: {result.returncode}[/yellow]")
            
    except KeyboardInterrupt:
        rprint("\n[yellow]Commande interrompue par l'utilisateur[/yellow]")
    except Exception as e:
        rprint(f"[red]❌ Erreur lors de l'exécution: {str(e)}[/red]")

def check_prerequisites():
    """Vérifie les prérequis."""
    rprint("[blue]🔍 Vérification des prérequis...[/blue]")
    
    issues = []
    
    # Vérifier Python
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ requis (actuel: {sys.version_info.major}.{sys.version_info.minor})")
    
    # Vérifier les fichiers principaux
    required_files = [
        "hologram_cli.py",
        "demo_cli.py", 
        "test_cli_quick.py",
        "cli/main.py"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            issues.append(f"Fichier manquant: {file}")
    
    # Vérifier les dépendances critiques
    critical_deps = ["click", "rich", "yaml"]
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            issues.append(f"Dépendance manquante: {dep}")
    
    if issues:
        rprint("[red]❌ Problèmes détectés:[/red]")
        for issue in issues:
            rprint(f"  • {issue}")
        
        if Confirm.ask("\nLancer la configuration automatique ?"):
            run_command("python setup_cli.py")
        
        return False
    else:
        rprint("[green]✅ Tous les prérequis sont satisfaits[/green]")
        return True

def show_quick_help():
    """Affiche l'aide rapide."""
    help_panel = Panel(
        "[bold blue]📚 Aide Rapide[/bold blue]\n\n"
        "[green]Commandes les plus utilisées:[/green]\n\n"
        "• [cyan]Option 1[/cyan] - Interface interactive complète\n"
        "  Recommandée pour débuter, navigation avec menus\n\n"
        "• [cyan]Option 2[/cyan] - Démonstration\n"
        "  Présentation de toutes les fonctionnalités\n\n"
        "• [cyan]Option 3[/cyan] - Tests rapides\n"
        "  Validation que tout fonctionne correctement\n\n"
        "• [cyan]Option 6-11[/cyan] - Commandes directes\n"
        "  Accès direct aux fonctionnalités spécifiques\n\n"
        "[yellow]Documentation complète:[/yellow]\n"
        "• CLI_README.md - Guide complet du CLI\n"
        "• EXTENSION_GUIDE.md - Guide d'extension\n"
        "• README.md - Vue d'ensemble du projet",
        title="[bold]Aide[/bold]",
        border_style="cyan"
    )
    console.print(help_panel)

def main():
    """Fonction principale."""
    show_welcome()
    
    # Vérifier les prérequis
    if not check_prerequisites():
        rprint("\n[yellow]Résolvez les problèmes ci-dessus avant de continuer.[/yellow]")
        return 1
    
    # Menu principal
    while True:
        rprint("\n" + "─" * 60)
        show_menu()
        
        # Options spéciales
        rprint("\n[dim]Options spéciales: 'help' pour l'aide, 'quit' pour quitter[/dim]")
        
        choice = Prompt.ask("\n[bold]Votre choix", default="1")
        
        # Traiter les options spéciales
        if choice.lower() in ['quit', 'q', '0']:
            rprint("[yellow]Au revoir ! 👋[/yellow]")
            break
        elif choice.lower() in ['help', 'h', '?']:
            show_quick_help()
            continue
        
        # Mapper les choix aux commandes
        command_map = {
            "1": "python hologram_cli.py --interactive",
            "2": "python demo_cli.py",
            "3": "python test_cli_quick.py",
            "4": "python setup_cli.py",
            "5": "python hologram_cli.py --help",
            "6": "python hologram_cli.py train start --model precision",
            "7": "python hologram_cli.py predict single --input data.mat",
            "8": "python hologram_cli.py test accuracy --model production",
            "9": "python hologram_cli.py analyze dataset --dataset-path data/",
            "10": "python hologram_cli.py visualize results --results-file test.json",
            "11": "python hologram_cli.py config show"
        }
        
        if choice in command_map:
            command = command_map[choice]
            
            # Demander confirmation pour certaines commandes
            if choice in ["6", "7", "8", "9", "10"]:
                if not Confirm.ask(f"\nExécuter: {command} ?"):
                    continue
            
            run_command(command)
            
            # Pause après exécution
            if choice not in ["1", "2"]:  # Pas de pause pour les modes interactifs
                input("\nAppuyez sur Entrée pour continuer...")
        else:
            rprint(f"[red]Option invalide: {choice}[/red]")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        rprint("\n[yellow]Lanceur interrompu par l'utilisateur[/yellow]")
        sys.exit(1)
    except Exception as e:
        rprint(f"\n[red]Erreur inattendue: {str(e)}[/red]")
        sys.exit(1)
