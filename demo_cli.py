#!/usr/bin/env python3
"""
Démonstration du CLI Inversion d'Anneaux Holographiques

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Script de démonstration des fonctionnalités du CLI.
"""

import sys
import os
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich import print as rprint

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

console = Console()

def show_welcome():
    """Affiche l'écran d'accueil de la démonstration."""
    welcome_panel = Panel(
        "[bold blue]🔬 Démonstration du CLI Inversion d'Anneaux Holographiques[/bold blue]\n\n"
        "[green]Ce script démontre les fonctionnalités principales du CLI moderne[/green]\n"
        "développé pour l'analyse et la prédiction des paramètres d'anneaux holographiques.\n\n"
        "[yellow]Fonctionnalités démontrées:[/yellow]\n"
        "• Interface interactive avec menus riches\n"
        "• Commandes d'entraînement, prédiction et test\n"
        "• Visualisations ASCII et graphiques\n"
        "• Gestion de configuration avancée\n"
        "• Barres de progression et formatage riche\n\n"
        "[dim]Auteur: Oussama GUELFAA - 08/07/2025[/dim]",
        title="[bold]Bienvenue[/bold]",
        border_style="blue"
    )
    console.print(welcome_panel)

def demo_rich_formatting():
    """Démontre les capacités de formatage riche."""
    rprint("\n[bold blue]📊 Démonstration du Formatage Riche[/bold blue]")
    
    # Tableau de métriques
    metrics_table = Table(title="Métriques de Performance", show_header=True, header_style="bold blue")
    metrics_table.add_column("Métrique", style="cyan")
    metrics_table.add_column("Gap", style="green")
    metrics_table.add_column("L_écran", style="yellow")
    metrics_table.add_column("Unité", style="dim")
    
    metrics_table.add_row("R²", "0.9946", "0.9949", "")
    metrics_table.add_row("MAE", "0.003421", "0.234", "µm")
    metrics_table.add_row("RMSE", "0.005123", "0.456", "µm")
    metrics_table.add_row("Précision", "97.0%", "99.9%", "%")
    
    console.print(metrics_table)
    
    # Panneau de statut
    status_panel = Panel(
        "[green]✅ Modèle entraîné avec succès[/green]\n\n"
        "📊 Époques: 150/200 (arrêt précoce)\n"
        "⏱️  Temps d'entraînement: 2h 34min\n"
        "💾 Modèle sauvegardé: models/precision_model.pth\n"
        "📈 Meilleure validation: Époque 147",
        title="[bold green]Statut d'Entraînement[/bold green]",
        border_style="green"
    )
    console.print(status_panel)

def demo_ascii_graphs():
    """Démontre les graphiques ASCII."""
    rprint("\n[bold blue]📈 Démonstration des Graphiques ASCII[/bold blue]")
    
    from cli.utils.formatters import ASCIIGraphs
    
    ascii_graphs = ASCIIGraphs(width=50, height=15)
    
    # Données simulées
    import numpy as np
    
    # Graphique en ligne - évolution de la loss
    epochs = list(range(1, 21))
    loss_values = [0.1 * np.exp(-0.1 * x) + 0.01 * np.random.random() for x in epochs]
    
    line_chart = ascii_graphs.line_chart(
        loss_values, 
        title="Évolution de la Loss d'Entraînement",
        width=60
    )
    rprint(line_chart)
    
    # Histogramme - distribution des erreurs
    errors = np.random.normal(0, 0.005, 100)
    histogram = ascii_graphs.histogram(
        errors.tolist(),
        bins=10,
        title="Distribution des Erreurs de Prédiction",
        width=50
    )
    rprint(histogram)

def demo_progress_bars():
    """Démontre les barres de progression."""
    rprint("\n[bold blue]⏳ Démonstration des Barres de Progression[/bold blue]")
    
    # Simulation d'entraînement
    rprint("[yellow]Simulation d'entraînement de modèle...[/yellow]")
    
    for epoch in track(range(1, 21), description="Entraînement..."):
        time.sleep(0.1)  # Simuler le travail
    
    rprint("[green]✅ Entraînement terminé ![/green]")
    
    # Simulation de prédiction batch
    rprint("\n[yellow]Simulation de prédiction batch...[/yellow]")
    
    files = [f"profile_{i:03d}.mat" for i in range(1, 51)]
    
    for file in track(files, description="Traitement des fichiers..."):
        time.sleep(0.05)  # Simuler le traitement
    
    rprint("[green]✅ Prédictions terminées ![/green]")

def demo_configuration():
    """Démontre la gestion de configuration."""
    rprint("\n[bold blue]⚙️  Démonstration de la Configuration[/bold blue]")
    
    from cli.config.config_manager import CLIConfig
    from cli.config.profiles import ConfigProfiles
    
    # Créer une configuration de démonstration
    config = CLIConfig()
    
    # Afficher les profils disponibles
    profiles = ConfigProfiles.get_default_profiles()
    
    profiles_table = Table(title="Profils Disponibles", show_header=True, header_style="bold blue")
    profiles_table.add_column("Profil", style="cyan")
    profiles_table.add_column("Description", style="green")
    profiles_table.add_column("Précision", style="yellow")
    profiles_table.add_column("Device", style="magenta")
    
    for profile_name, profile_config in profiles.items():
        profiles_table.add_row(
            profile_name,
            profile_config.get('description', 'N/A')[:50] + "...",
            profile_config.get('precision', 'N/A'),
            profile_config.get('device', 'N/A')
        )
    
    console.print(profiles_table)
    
    # Démonstration de validation
    rprint("\n[yellow]Validation du profil 'production'...[/yellow]")
    
    validation = ConfigProfiles.validate_profile_compatibility(profiles['production'])
    
    if validation['compatible']:
        rprint("[green]✅ Profil compatible[/green]")
    else:
        rprint("[red]❌ Profil incompatible[/red]")
    
    if validation['warnings']:
        rprint(f"[yellow]⚠️  {len(validation['warnings'])} avertissement(s)[/yellow]")
    
    if validation['recommendations']:
        rprint(f"[blue]💡 {len(validation['recommendations'])} recommandation(s)[/blue]")

def demo_interactive_features():
    """Démontre les fonctionnalités interactives."""
    rprint("\n[bold blue]🖱️  Démonstration des Fonctionnalités Interactives[/bold blue]")
    
    try:
        from cli.interactive import InteractiveMenu
        
        rprint("[green]✅ Module interactif disponible[/green]")
        rprint("Le CLI supporte les menus interactifs avec questionary")
        
        # Simuler un menu simple
        menu_demo = Panel(
            "[bold blue]Menu Principal[/bold blue]\n\n"
            "🏋️  1. Entraîner un modèle\n"
            "🔮 2. Faire des prédictions\n"
            "🧪 3. Tester un modèle\n"
            "📊 4. Analyser des données\n"
            "📈 5. Créer des visualisations\n"
            "⚙️  6. Gérer la configuration\n\n"
            "[dim]Navigation avec flèches ↑↓, sélection avec Entrée[/dim]",
            title="[bold]Interface Interactive[/bold]",
            border_style="cyan"
        )
        console.print(menu_demo)
        
    except ImportError:
        rprint("[yellow]⚠️  Module questionary non installé[/yellow]")
        rprint("Pour activer l'interface interactive complète:")
        rprint("[cyan]pip install questionary[/cyan]")

def demo_model_integration():
    """Démontre l'intégration avec les modèles existants."""
    rprint("\n[bold blue]🤖 Démonstration de l'Intégration des Modèles[/bold blue]")
    
    from cli.utils.cli_utils import CLIUtils
    
    utils = CLIUtils()
    
    # Simuler la détection de modèles
    rprint("[yellow]Détection des modèles disponibles...[/yellow]")
    
    models = utils.get_available_models()
    
    if models:
        models_table = Table(title="Modèles Détectés", show_header=True, header_style="bold blue")
        models_table.add_column("Nom", style="cyan")
        models_table.add_column("Type", style="green")
        models_table.add_column("Statut", style="yellow")
        models_table.add_column("Chemin", style="dim")
        
        for model in models[:5]:  # Limiter à 5 pour la démo
            models_table.add_row(
                model['name'][:30] + "...",
                model['type'],
                model['status'],
                model['path'][:40] + "..."
            )
        
        console.print(models_table)
        rprint(f"[green]✅ {len(models)} modèle(s) détecté(s)[/green]")
    else:
        rprint("[yellow]⚠️  Aucun modèle détecté (normal en mode démo)[/yellow]")
    
    # Démonstration de validation de device
    device = utils.detect_device()
    device_panel = Panel(
        f"[bold green]Device détecté: {device}[/bold green]\n\n"
        "Le CLI détecte automatiquement le meilleur device disponible:\n"
        "• CUDA (GPU NVIDIA) - Performance maximale\n"
        "• MPS (GPU Apple Silicon) - Optimisé pour Mac\n"
        "• CPU - Compatible partout",
        title="[bold]Détection de Device[/bold]",
        border_style="green"
    )
    console.print(device_panel)

def demo_command_examples():
    """Montre des exemples de commandes CLI."""
    rprint("\n[bold blue]💻 Exemples de Commandes CLI[/bold blue]")
    
    examples_table = Table(title="Exemples d'Utilisation", show_header=True, header_style="bold blue")
    examples_table.add_column("Commande", style="cyan")
    examples_table.add_column("Description", style="green")
    
    examples = [
        ("python hologram_cli.py --interactive", "Mode interactif complet"),
        ("python hologram_cli.py train start --model precision", "Entraîner modèle ultra-précision"),
        ("python hologram_cli.py predict single --input data.mat", "Prédiction sur un fichier"),
        ("python hologram_cli.py test accuracy --model production", "Test de précision"),
        ("python hologram_cli.py analyze dataset --dataset-path data/", "Analyser un dataset"),
        ("python hologram_cli.py visualize results --results-file test.json", "Visualiser des résultats"),
        ("python hologram_cli.py config show", "Afficher la configuration"),
        ("python hologram_cli.py config set-profile --profile recherche", "Changer de profil")
    ]
    
    for command, description in examples:
        examples_table.add_row(command, description)
    
    console.print(examples_table)

def show_summary():
    """Affiche le résumé de la démonstration."""
    summary_panel = Panel(
        "[bold blue]🎉 Démonstration Terminée[/bold blue]\n\n"
        "[green]Fonctionnalités démontrées:[/green]\n"
        "✅ Formatage riche avec Rich\n"
        "✅ Graphiques ASCII intégrés\n"
        "✅ Barres de progression animées\n"
        "✅ Gestion de configuration avancée\n"
        "✅ Interface interactive (avec questionary)\n"
        "✅ Intégration avec les modèles existants\n"
        "✅ Exemples de commandes complètes\n\n"
        "[yellow]Pour utiliser le CLI:[/yellow]\n"
        "• Mode interactif: [cyan]python hologram_cli.py --interactive[/cyan]\n"
        "• Aide générale: [cyan]python hologram_cli.py --help[/cyan]\n"
        "• Aide spécifique: [cyan]python hologram_cli.py COMMANDE --help[/cyan]\n\n"
        "[dim]Le CLI est prêt pour l'utilisation en production ![/dim]",
        title="[bold green]Résumé[/bold green]",
        border_style="green"
    )
    console.print(summary_panel)

def main():
    """Fonction principale de démonstration."""
    try:
        show_welcome()
        
        # Pause pour lire l'introduction
        input("\nAppuyez sur Entrée pour commencer la démonstration...")
        
        # Démonstrations séquentielles
        demo_rich_formatting()
        input("\nAppuyez sur Entrée pour continuer...")
        
        demo_ascii_graphs()
        input("\nAppuyez sur Entrée pour continuer...")
        
        demo_progress_bars()
        input("\nAppuyez sur Entrée pour continuer...")
        
        demo_configuration()
        input("\nAppuyez sur Entrée pour continuer...")
        
        demo_interactive_features()
        input("\nAppuyez sur Entrée pour continuer...")
        
        demo_model_integration()
        input("\nAppuyez sur Entrée pour continuer...")
        
        demo_command_examples()
        input("\nAppuyez sur Entrée pour voir le résumé...")
        
        show_summary()
        
        rprint("\n[bold green]🎉 Démonstration terminée avec succès ![/bold green]")
        
    except KeyboardInterrupt:
        rprint("\n[yellow]Démonstration interrompue par l'utilisateur[/yellow]")
    except Exception as e:
        rprint(f"\n[red]Erreur durant la démonstration: {e}[/red]")

if __name__ == "__main__":
    main()
