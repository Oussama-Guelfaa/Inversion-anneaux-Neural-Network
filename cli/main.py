#!/usr/bin/env python3
"""
CLI Principal - Inversion d'Anneaux Holographiques

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Point d'entrée principal du CLI moderne et interactif.
"""

import sys
import os
from pathlib import Path
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).parent.parent))

from cli.commands import train, predict, test, analyze, visualize, config
from cli.utils import CLIUtils
from cli.config import CLIConfig

console = Console()

@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Afficher la version')
@click.option('--interactive', '-i', is_flag=True, help='Mode interactif')
@click.option('--config-file', '-c', default='cli/config/default.yaml', 
              help='Fichier de configuration')
@click.pass_context
def main_cli(ctx, version, interactive, config_file):
    """
    🔬 CLI Moderne pour l'Inversion d'Anneaux Holographiques
    
    Un outil complet pour l'entraînement, la prédiction et l'analyse
    de réseaux neuronaux pour la prédiction des paramètres gap et L_écran.
    """
    if version:
        rprint(f"[bold blue]Inversion Anneaux CLI v1.0.0[/bold blue]")
        rprint(f"[dim]Auteur: Oussama GUELFAA[/dim]")
        return
    
    # Initialiser la configuration
    ctx.ensure_object(dict)
    ctx.obj['config'] = CLIConfig(config_file)
    ctx.obj['utils'] = CLIUtils()
    
    if interactive or ctx.invoked_subcommand is None:
        show_welcome()
        if interactive:
            interactive_mode(ctx)

def show_welcome():
    """Affiche l'écran d'accueil du CLI."""
    welcome_text = Text()
    welcome_text.append("🔬 CLI Inversion d'Anneaux Holographiques\n", style="bold blue")
    welcome_text.append("Prédiction des paramètres gap et L_écran par réseaux neuronaux\n\n", style="dim")
    welcome_text.append("Commandes disponibles:\n", style="bold")
    welcome_text.append("  train     - Entraîner un modèle\n", style="green")
    welcome_text.append("  predict   - Faire des prédictions\n", style="cyan")
    welcome_text.append("  test      - Tester un modèle\n", style="yellow")
    welcome_text.append("  analyze   - Analyser des données\n", style="magenta")
    welcome_text.append("  visualize - Créer des visualisations\n", style="red")
    welcome_text.append("  config    - Gérer la configuration\n", style="blue")
    welcome_text.append("\nUtilisez --help avec chaque commande pour plus d'informations.", style="dim")
    
    panel = Panel(welcome_text, title="[bold]Bienvenue[/bold]", border_style="blue")
    console.print(panel)

def interactive_mode(ctx):
    """Mode interactif pour sélectionner les commandes."""
    from cli.interactive import InteractiveMenu, InteractiveWizard

    menu = InteractiveMenu("CLI Inversion d'Anneaux Holographiques")
    wizard = InteractiveWizard("Assistant CLI")

    while True:
        choice = menu.show_main_menu()

        if choice == "quit":
            rprint("[yellow]Au revoir ! 👋[/yellow]")
            break
        elif choice == "help":
            menu.show_help()
        elif choice == "train":
            # Lancer l'assistant d'entraînement
            wizard.run_training_wizard(ctx.obj['utils'], ctx.obj['config'])
        elif choice == "predict":
            # Lancer l'assistant de prédiction
            wizard.run_prediction_wizard(ctx.obj['utils'], ctx.obj['config'])
        elif choice == "test":
            rprint("[blue]🧪 Lancement du module de test...[/blue]")
            # Ici on pourrait ajouter un assistant de test
        elif choice == "analyze":
            rprint("[blue]📊 Lancement du module d'analyse...[/blue]")
            # Ici on pourrait ajouter un assistant d'analyse
        elif choice == "visualize":
            rprint("[blue]📈 Lancement du module de visualisation...[/blue]")
            # Ici on pourrait ajouter un assistant de visualisation
        elif choice == "config":
            rprint("[blue]⚙️  Lancement du gestionnaire de configuration...[/blue]")
            # Ici on pourrait ajouter un assistant de configuration
        else:
            rprint(f"[red]Commande non reconnue: {choice}[/red]")

# Ajouter les groupes de commandes
main_cli.add_command(train.train_group)
main_cli.add_command(predict.predict_group)
main_cli.add_command(test.test_group)
main_cli.add_command(analyze.analyze_group)
main_cli.add_command(visualize.visualize_group)
main_cli.add_command(config.config_group)

if __name__ == '__main__':
    main_cli()
