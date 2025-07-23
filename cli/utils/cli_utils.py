#!/usr/bin/env python3
"""
Utilitaires CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Classe utilitaire principale pour le CLI.
"""

import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

class CLIUtils:
    """
    Classe utilitaire principale pour le CLI.
    
    Fournit des méthodes communes pour la validation, le formatage,
    et l'interaction avec les modèles et données.
    """
    
    def __init__(self):
        """Initialise les utilitaires CLI."""
        self.console = console
    
    def detect_device(self) -> str:
        """
        Détecte automatiquement le meilleur device disponible.
        
        Returns:
            str: 'cuda', 'mps', ou 'cpu'
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def validate_model_path(self, model_path: str) -> bool:
        """
        Valide qu'un chemin de modèle existe et contient les fichiers nécessaires.
        
        Args:
            model_path (str): Chemin vers le dossier du modèle
            
        Returns:
            bool: True si valide, False sinon
        """
        path = Path(model_path)
        if not path.exists():
            return False
        
        # Vérifier la présence des fichiers essentiels
        required_files = ['src', 'models', 'config']
        for req_file in required_files:
            if not (path / req_file).exists():
                return False
        
        return True
    
    def validate_data_path(self, data_path: str) -> bool:
        """
        Valide qu'un chemin de données existe et contient des fichiers .mat.
        
        Args:
            data_path (str): Chemin vers les données
            
        Returns:
            bool: True si valide, False sinon
        """
        path = Path(data_path)
        if not path.exists():
            return False
        
        # Vérifier la présence de fichiers .mat
        mat_files = list(path.glob('*.mat'))
        return len(mat_files) > 0
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """
        Retourne la liste des modèles disponibles.
        
        Returns:
            List[Dict]: Liste des modèles avec leurs informations
        """
        models = []
        base_path = Path('.')
        
        # Chercher les dossiers de réseaux neuronaux
        for model_dir in base_path.glob('Reseau_Neural_*'):
            if model_dir.is_dir() and self.validate_model_path(str(model_dir)):
                models.append({
                    'name': model_dir.name,
                    'path': str(model_dir),
                    'type': self._get_model_type(model_dir.name),
                    'status': self._get_model_status(model_dir)
                })
        
        return models
    
    def _get_model_type(self, model_name: str) -> str:
        """Détermine le type de modèle basé sur son nom."""
        if 'PRECISION' in model_name:
            return 'ultra-precision'
        elif 'FINAL' in model_name:
            return 'production'
        elif '2D' in model_name:
            return 'dual-parameter'
        elif '1D' in model_name:
            return 'gap-only'
        else:
            return 'unknown'
    
    def _get_model_status(self, model_path: Path) -> str:
        """Détermine le statut d'un modèle."""
        if (model_path / 'models').exists():
            model_files = list((model_path / 'models').glob('*.pth'))
            if model_files:
                return 'trained'
        return 'untrained'
    
    def create_progress_bar(self, description: str = "Processing...") -> Progress:
        """
        Crée une barre de progression stylée.
        
        Args:
            description (str): Description de la tâche
            
        Returns:
            Progress: Objet de barre de progression
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )
    
    def create_results_table(self, results: List[Dict[str, Any]], 
                           title: str = "Résultats") -> Table:
        """
        Crée un tableau formaté pour afficher les résultats.
        
        Args:
            results (List[Dict]): Liste des résultats
            title (str): Titre du tableau
            
        Returns:
            Table: Tableau formaté
        """
        table = Table(title=title, show_header=True, header_style="bold blue")
        
        if not results:
            return table
        
        # Ajouter les colonnes basées sur les clés du premier résultat
        for key in results[0].keys():
            table.add_column(key.replace('_', ' ').title(), style="cyan")
        
        # Ajouter les lignes
        for result in results:
            row = [str(value) for value in result.values()]
            table.add_row(*row)
        
        return table
    
    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Formate les métriques pour l'affichage.
        
        Args:
            metrics (Dict): Dictionnaire des métriques
            
        Returns:
            str: Métriques formatées
        """
        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'r2' in key.lower() or 'accuracy' in key.lower():
                    formatted.append(f"{key}: {value:.4f} ({value*100:.2f}%)")
                else:
                    formatted.append(f"{key}: {value:.6f}")
            else:
                formatted.append(f"{key}: {value}")
        
        return " | ".join(formatted)
    
    def show_error(self, message: str, details: str = None) -> None:
        """
        Affiche un message d'erreur formaté.
        
        Args:
            message (str): Message d'erreur principal
            details (str): Détails optionnels
        """
        error_panel = Panel(
            f"[red]{message}[/red]" + (f"\n\n[dim]{details}[/dim]" if details else ""),
            title="[red]Erreur[/red]",
            border_style="red"
        )
        self.console.print(error_panel)
    
    def show_success(self, message: str, details: str = None) -> None:
        """
        Affiche un message de succès formaté.
        
        Args:
            message (str): Message de succès principal
            details (str): Détails optionnels
        """
        success_panel = Panel(
            f"[green]{message}[/green]" + (f"\n\n[dim]{details}[/dim]" if details else ""),
            title="[green]Succès[/green]",
            border_style="green"
        )
        self.console.print(success_panel)
    
    def show_warning(self, message: str, details: str = None) -> None:
        """
        Affiche un message d'avertissement formaté.
        
        Args:
            message (str): Message d'avertissement principal
            details (str): Détails optionnels
        """
        warning_panel = Panel(
            f"[yellow]{message}[/yellow]" + (f"\n\n[dim]{details}[/dim]" if details else ""),
            title="[yellow]Avertissement[/yellow]",
            border_style="yellow"
        )
        self.console.print(warning_panel)
    
    def confirm_action(self, message: str) -> bool:
        """
        Demande confirmation à l'utilisateur.
        
        Args:
            message (str): Message de confirmation
            
        Returns:
            bool: True si confirmé, False sinon
        """
        try:
            import questionary
            return questionary.confirm(message).ask()
        except ImportError:
            response = input(f"{message} (y/N): ").lower().strip()
            return response in ['y', 'yes', 'oui']
