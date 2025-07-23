#!/usr/bin/env python3
"""
Interface Interactive CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Module pour l'interface interactive avec menus et sélections.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint

console = Console()

class InteractiveMenu:
    """
    Classe pour créer des menus interactifs riches.
    
    Fournit des méthodes pour créer des menus de sélection,
    des formulaires interactifs et des assistants pas-à-pas.
    """
    
    def __init__(self, title: str = "Menu Interactif"):
        """
        Initialise le menu interactif.
        
        Args:
            title (str): Titre du menu
        """
        self.title = title
        self.console = console
    
    def show_main_menu(self) -> str:
        """
        Affiche le menu principal et retourne le choix de l'utilisateur.
        
        Returns:
            str: Commande sélectionnée
        """
        try:
            import questionary
            
            # Style personnalisé pour le menu
            custom_style = questionary.Style([
                ('question', 'bold fg:#ff9d00'),
                ('answer', 'fg:#ff9d00 bold'),
                ('pointer', 'fg:#ff9d00 bold'),
                ('highlighted', 'fg:#ff9d00 bold'),
                ('selected', 'fg:#cc5454'),
                ('separator', 'fg:#cc5454'),
                ('instruction', ''),
                ('text', ''),
                ('disabled', 'fg:#858585 italic')
            ])
            
            choices = [
                questionary.Choice("🏋️  Entraîner un modèle", value="train"),
                questionary.Choice("🔮 Faire des prédictions", value="predict"),
                questionary.Choice("🧪 Tester un modèle", value="test"),
                questionary.Choice("📊 Analyser des données", value="analyze"),
                questionary.Choice("📈 Créer des visualisations", value="visualize"),
                questionary.Choice("⚙️  Gérer la configuration", value="config"),
                questionary.Separator(),
                questionary.Choice("❓ Aide", value="help"),
                questionary.Choice("❌ Quitter", value="quit")
            ]
            
            choice = questionary.select(
                "Que souhaitez-vous faire ?",
                choices=choices,
                style=custom_style,
                use_shortcuts=True
            ).ask()
            
            return choice or "quit"
            
        except ImportError:
            # Fallback sans questionary
            return self._show_simple_menu()
    
    def _show_simple_menu(self) -> str:
        """Menu simple sans questionary (fallback)."""
        rprint("\n[bold blue]🔬 Menu Principal - Inversion d'Anneaux Holographiques[/bold blue]")
        rprint("─" * 60)
        
        options = {
            "1": ("🏋️  Entraîner un modèle", "train"),
            "2": ("🔮 Faire des prédictions", "predict"),
            "3": ("🧪 Tester un modèle", "test"),
            "4": ("📊 Analyser des données", "analyze"),
            "5": ("📈 Créer des visualisations", "visualize"),
            "6": ("⚙️  Gérer la configuration", "config"),
            "7": ("❓ Aide", "help"),
            "0": ("❌ Quitter", "quit")
        }
        
        for key, (description, _) in options.items():
            rprint(f"  {key}. {description}")
        
        rprint("─" * 60)
        
        while True:
            choice = Prompt.ask("Votre choix", choices=list(options.keys()))
            if choice in options:
                return options[choice][1]
    
    def select_model(self, available_models: List[Dict[str, str]]) -> Optional[str]:
        """
        Menu de sélection de modèle.
        
        Args:
            available_models (List[Dict]): Liste des modèles disponibles
            
        Returns:
            Optional[str]: Chemin du modèle sélectionné
        """
        if not available_models:
            rprint("[red]Aucun modèle disponible[/red]")
            return None
        
        try:
            import questionary
            
            choices = []
            for model in available_models:
                name = model['name']
                model_type = model.get('type', 'unknown')
                status = model.get('status', 'unknown')
                
                # Créer une description riche
                description = f"{name} ({model_type}, {status})"
                choices.append(questionary.Choice(description, value=model['path']))
            
            choices.append(questionary.Choice("❌ Annuler", value=None))
            
            return questionary.select(
                "Sélectionnez un modèle:",
                choices=choices
            ).ask()
            
        except ImportError:
            # Fallback simple
            rprint("\n[bold blue]Modèles disponibles:[/bold blue]")
            for i, model in enumerate(available_models):
                rprint(f"  {i+1}. {model['name']} ({model.get('type', 'unknown')})")
            rprint(f"  0. Annuler")
            
            while True:
                try:
                    choice = int(Prompt.ask("Sélectionnez un modèle", default="0"))
                    if choice == 0:
                        return None
                    elif 1 <= choice <= len(available_models):
                        return available_models[choice-1]['path']
                    else:
                        rprint("[red]Choix invalide[/red]")
                except ValueError:
                    rprint("[red]Veuillez entrer un nombre[/red]")
    
    def configure_training(self) -> Dict[str, Any]:
        """
        Assistant de configuration d'entraînement.
        
        Returns:
            Dict[str, Any]: Configuration d'entraînement
        """
        rprint("\n[bold blue]🏋️  Assistant de Configuration d'Entraînement[/bold blue]")
        
        config = {}
        
        try:
            import questionary
            
            # Sélection du type de modèle
            model_types = [
                ("Ultra-précision (±0.007µm)", "precision"),
                ("Production (standard)", "production"),
                ("Recherche (expérimental)", "research"),
                ("Gap uniquement", "gap-only")
            ]
            
            model_choice = questionary.select(
                "Type de modèle à entraîner:",
                choices=[questionary.Choice(desc, value=val) for desc, val in model_types]
            ).ask()
            
            config['model_type'] = model_choice
            
            # Paramètres d'entraînement
            config['epochs'] = questionary.text(
                "Nombre d'époques:",
                default="100",
                validate=lambda x: x.isdigit() and int(x) > 0
            ).ask()
            
            config['batch_size'] = questionary.text(
                "Taille des batches:",
                default="32",
                validate=lambda x: x.isdigit() and int(x) > 0
            ).ask()
            
            config['learning_rate'] = questionary.text(
                "Taux d'apprentissage:",
                default="0.001",
                validate=lambda x: self._is_float(x) and float(x) > 0
            ).ask()
            
            # Device
            device_choice = questionary.select(
                "Device de calcul:",
                choices=[
                    questionary.Choice("Automatique", "auto"),
                    questionary.Choice("CPU", "cpu"),
                    questionary.Choice("CUDA (GPU NVIDIA)", "cuda"),
                    questionary.Choice("MPS (GPU Apple)", "mps")
                ]
            ).ask()
            
            config['device'] = device_choice
            
            # Options avancées
            config['early_stopping'] = questionary.confirm(
                "Activer l'arrêt précoce ?",
                default=True
            ).ask()
            
            config['save_best'] = questionary.confirm(
                "Sauvegarder le meilleur modèle ?",
                default=True
            ).ask()
            
        except ImportError:
            # Fallback simple
            config['model_type'] = Prompt.ask(
                "Type de modèle",
                choices=["precision", "production", "research", "gap-only"],
                default="precision"
            )
            
            config['epochs'] = int(Prompt.ask("Nombre d'époques", default="100"))
            config['batch_size'] = int(Prompt.ask("Taille des batches", default="32"))
            config['learning_rate'] = float(Prompt.ask("Taux d'apprentissage", default="0.001"))
            config['device'] = Prompt.ask(
                "Device",
                choices=["auto", "cpu", "cuda", "mps"],
                default="auto"
            )
            config['early_stopping'] = Confirm.ask("Arrêt précoce ?", default=True)
            config['save_best'] = Confirm.ask("Sauvegarder le meilleur ?", default=True)
        
        return config
    
    def configure_prediction(self) -> Dict[str, Any]:
        """
        Assistant de configuration de prédiction.
        
        Returns:
            Dict[str, Any]: Configuration de prédiction
        """
        rprint("\n[bold blue]🔮 Assistant de Configuration de Prédiction[/bold blue]")
        
        config = {}
        
        try:
            import questionary
            
            # Type de prédiction
            prediction_types = [
                ("Fichier unique", "single"),
                ("Traitement par lot", "batch"),
                ("Simulation", "simulate")
            ]
            
            pred_type = questionary.select(
                "Type de prédiction:",
                choices=[questionary.Choice(desc, value=val) for desc, val in prediction_types]
            ).ask()
            
            config['type'] = pred_type
            
            if pred_type == "single":
                config['input_file'] = questionary.path(
                    "Fichier d'entrée:",
                    validate=lambda x: Path(x).exists()
                ).ask()
                
            elif pred_type == "batch":
                config['input_dir'] = questionary.path(
                    "Dossier d'entrée:",
                    validate=lambda x: Path(x).exists() and Path(x).is_dir()
                ).ask()
                
                config['pattern'] = questionary.text(
                    "Motif de fichiers:",
                    default="*.mat"
                ).ask()
                
            elif pred_type == "simulate":
                config['gap'] = float(questionary.text(
                    "Valeur du gap (µm):",
                    validate=lambda x: self._is_float(x) and float(x) > 0
                ).ask())
                
                config['L_ecran'] = float(questionary.text(
                    "Distance L_écran (µm):",
                    validate=lambda x: self._is_float(x) and float(x) > 0
                ).ask())
                
                config['noise_level'] = float(questionary.text(
                    "Niveau de bruit (%):",
                    default="0.0",
                    validate=lambda x: self._is_float(x) and float(x) >= 0
                ).ask())
            
            # Format de sortie
            config['output_format'] = questionary.select(
                "Format de sortie:",
                choices=[
                    questionary.Choice("JSON", "json"),
                    questionary.Choice("CSV", "csv"),
                    questionary.Choice("Tableau (terminal)", "table")
                ]
            ).ask()
            
            # Seuil de confiance
            config['confidence'] = float(questionary.text(
                "Seuil de confiance minimum:",
                default="0.8",
                validate=lambda x: self._is_float(x) and 0 <= float(x) <= 1
            ).ask())
            
        except ImportError:
            # Fallback simple
            config['type'] = Prompt.ask(
                "Type de prédiction",
                choices=["single", "batch", "simulate"],
                default="single"
            )
            
            if config['type'] == "single":
                config['input_file'] = Prompt.ask("Fichier d'entrée")
            elif config['type'] == "batch":
                config['input_dir'] = Prompt.ask("Dossier d'entrée")
                config['pattern'] = Prompt.ask("Motif de fichiers", default="*.mat")
            elif config['type'] == "simulate":
                config['gap'] = float(Prompt.ask("Gap (µm)"))
                config['L_ecran'] = float(Prompt.ask("L_écran (µm)"))
                config['noise_level'] = float(Prompt.ask("Bruit (%)", default="0.0"))
            
            config['output_format'] = Prompt.ask(
                "Format de sortie",
                choices=["json", "csv", "table"],
                default="json"
            )
            config['confidence'] = float(Prompt.ask("Seuil de confiance", default="0.8"))
        
        return config
    
    def configure_analysis(self) -> Dict[str, Any]:
        """
        Assistant de configuration d'analyse.
        
        Returns:
            Dict[str, Any]: Configuration d'analyse
        """
        rprint("\n[bold blue]📊 Assistant de Configuration d'Analyse[/bold blue]")
        
        config = {}
        
        try:
            import questionary
            
            # Type d'analyse
            analysis_types = [
                ("Dataset", "dataset"),
                ("Résultats de modèle", "results"),
                ("Comparaison de modèles", "compare"),
                ("Distribution des paramètres", "distribution")
            ]
            
            analysis_type = questionary.select(
                "Type d'analyse:",
                choices=[questionary.Choice(desc, value=val) for desc, val in analysis_types]
            ).ask()
            
            config['type'] = analysis_type
            
            if analysis_type == "dataset":
                config['dataset_path'] = questionary.path(
                    "Chemin du dataset:",
                    validate=lambda x: Path(x).exists()
                ).ask()
                
                config['max_samples'] = int(questionary.text(
                    "Nombre max d'échantillons:",
                    default="1000",
                    validate=lambda x: x.isdigit() and int(x) > 0
                ).ask())
                
            elif analysis_type == "results":
                config['results_file'] = questionary.path(
                    "Fichier de résultats:",
                    validate=lambda x: Path(x).exists() and x.endswith('.json')
                ).ask()
                
            elif analysis_type == "compare":
                config['model1_results'] = questionary.path(
                    "Résultats modèle 1:",
                    validate=lambda x: Path(x).exists()
                ).ask()
                
                config['model2_results'] = questionary.path(
                    "Résultats modèle 2:",
                    validate=lambda x: Path(x).exists()
                ).ask()
            
            # Options communes
            config['generate_plots'] = questionary.confirm(
                "Générer des graphiques ?",
                default=True
            ).ask()
            
            config['detailed'] = questionary.confirm(
                "Analyse détaillée ?",
                default=False
            ).ask()
            
        except ImportError:
            # Fallback simple
            config['type'] = Prompt.ask(
                "Type d'analyse",
                choices=["dataset", "results", "compare", "distribution"],
                default="dataset"
            )
            
            if config['type'] == "dataset":
                config['dataset_path'] = Prompt.ask("Chemin du dataset")
                config['max_samples'] = int(Prompt.ask("Max échantillons", default="1000"))
            elif config['type'] == "results":
                config['results_file'] = Prompt.ask("Fichier de résultats")
            elif config['type'] == "compare":
                config['model1_results'] = Prompt.ask("Résultats modèle 1")
                config['model2_results'] = Prompt.ask("Résultats modèle 2")
            
            config['generate_plots'] = Confirm.ask("Générer graphiques ?", default=True)
            config['detailed'] = Confirm.ask("Analyse détaillée ?", default=False)
        
        return config
    
    def show_help(self) -> None:
        """Affiche l'aide interactive."""
        help_panel = Panel(
            "[bold blue]🔬 CLI Inversion d'Anneaux Holographiques - Aide[/bold blue]\n\n"
            "[green]Commandes principales:[/green]\n"
            "• [cyan]train[/cyan] - Entraîner un modèle de réseau neuronal\n"
            "• [cyan]predict[/cyan] - Faire des prédictions sur de nouvelles données\n"
            "• [cyan]test[/cyan] - Tester et évaluer un modèle\n"
            "• [cyan]analyze[/cyan] - Analyser des datasets ou des résultats\n"
            "• [cyan]visualize[/cyan] - Créer des visualisations et graphiques\n"
            "• [cyan]config[/cyan] - Gérer la configuration du CLI\n\n"
            "[green]Utilisation:[/green]\n"
            "• Mode interactif: [yellow]python hologram_cli.py --interactive[/yellow]\n"
            "• Commande directe: [yellow]python hologram_cli.py train --model precision[/yellow]\n"
            "• Aide détaillée: [yellow]python hologram_cli.py COMMANDE --help[/yellow]\n\n"
            "[green]Exemples:[/green]\n"
            "• [yellow]python hologram_cli.py train start --model precision --epochs 200[/yellow]\n"
            "• [yellow]python hologram_cli.py predict single --input data.mat[/yellow]\n"
            "• [yellow]python hologram_cli.py test accuracy --model production[/yellow]",
            title="[bold]Aide[/bold]",
            border_style="blue"
        )
        console.print(help_panel)
        
        Prompt.ask("\nAppuyez sur Entrée pour continuer", default="")
    
    def _is_float(self, value: str) -> bool:
        """Vérifie si une chaîne peut être convertie en float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

class InteractiveWizard:
    """
    Assistant interactif pour guider l'utilisateur pas-à-pas.
    """
    
    def __init__(self, title: str = "Assistant"):
        self.title = title
        self.console = console
        self.menu = InteractiveMenu(title)
    
    def run_training_wizard(self, utils, config) -> bool:
        """
        Assistant complet pour l'entraînement d'un modèle.
        
        Args:
            utils: Utilitaires CLI
            config: Configuration CLI
            
        Returns:
            bool: True si l'entraînement a été lancé
        """
        rprint("\n[bold blue]🧙‍♂️ Assistant d'Entraînement[/bold blue]")
        rprint("Cet assistant vous guidera pour configurer et lancer l'entraînement d'un modèle.")
        
        # Étape 1: Sélection du modèle
        models = utils.get_available_models()
        if not models:
            utils.show_error("Aucun modèle disponible pour l'entraînement")
            return False
        
        model_path = self.menu.select_model(models)
        if not model_path:
            rprint("[yellow]Assistant annulé[/yellow]")
            return False
        
        # Étape 2: Configuration
        train_config = self.menu.configure_training()
        
        # Étape 3: Validation
        rprint("\n[bold blue]📋 Résumé de la Configuration[/bold blue]")
        
        summary_table = Table(show_header=True, header_style="bold blue")
        summary_table.add_column("Paramètre", style="cyan")
        summary_table.add_column("Valeur", style="green")
        
        summary_table.add_row("Modèle", model_path)
        for key, value in train_config.items():
            summary_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(summary_table)
        
        # Confirmation
        if not Confirm.ask("\nLancer l'entraînement avec cette configuration ?"):
            rprint("[yellow]Entraînement annulé[/yellow]")
            return False
        
        # Ici on lancerait l'entraînement réel
        rprint("[green]🚀 Entraînement lancé ![/green]")
        return True
    
    def run_prediction_wizard(self, utils, config) -> bool:
        """
        Assistant complet pour faire des prédictions.
        
        Args:
            utils: Utilitaires CLI
            config: Configuration CLI
            
        Returns:
            bool: True si la prédiction a été lancée
        """
        rprint("\n[bold blue]🧙‍♂️ Assistant de Prédiction[/bold blue]")
        
        # Configuration de prédiction
        pred_config = self.menu.configure_prediction()
        
        # Sélection du modèle
        models = [m for m in utils.get_available_models() if m.get('status') == 'trained']
        if not models:
            utils.show_error("Aucun modèle entraîné disponible")
            return False
        
        model_path = self.menu.select_model(models)
        if not model_path:
            return False
        
        # Résumé et confirmation
        rprint("\n[bold blue]📋 Résumé de la Prédiction[/bold blue]")
        
        summary_table = Table(show_header=True, header_style="bold blue")
        summary_table.add_column("Paramètre", style="cyan")
        summary_table.add_column("Valeur", style="green")
        
        summary_table.add_row("Modèle", model_path)
        for key, value in pred_config.items():
            summary_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(summary_table)
        
        if Confirm.ask("\nLancer la prédiction ?"):
            rprint("[green]🔮 Prédiction lancée ![/green]")
            return True
        
        return False
