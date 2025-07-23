#!/usr/bin/env python3
"""
Gestionnaire de Configuration CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Classe principale pour la gestion des configurations YAML.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console

console = Console()

class CLIConfig:
    """
    Gestionnaire de configuration pour le CLI.
    
    Gère le chargement, la validation et la sauvegarde des configurations
    YAML avec support pour différents profils (production, recherche, démo).
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialise le gestionnaire de configuration.
        
        Args:
            config_file (str): Chemin vers le fichier de configuration
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config = {}
        self.load_config()
    
    def _get_default_config_path(self) -> str:
        """Retourne le chemin par défaut du fichier de configuration."""
        return str(Path(__file__).parent / "default.yaml")
    
    def load_config(self) -> None:
        """Charge la configuration depuis le fichier YAML."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f) or {}
                console.print(f"[green]✓[/green] Configuration chargée: {self.config_file}")
            else:
                console.print(f"[yellow]⚠[/yellow] Fichier de configuration non trouvé: {self.config_file}")
                self.config = self._get_default_config()
                self.save_config()
        except Exception as e:
            console.print(f"[red]✗[/red] Erreur lors du chargement de la configuration: {e}")
            self.config = self._get_default_config()
    
    def save_config(self) -> None:
        """Sauvegarde la configuration dans le fichier YAML."""
        try:
            # Créer le répertoire si nécessaire
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            console.print(f"[green]✓[/green] Configuration sauvegardée: {self.config_file}")
        except Exception as e:
            console.print(f"[red]✗[/red] Erreur lors de la sauvegarde: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration.
        
        Args:
            key (str): Clé de configuration (support notation pointée)
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            Valeur de configuration ou valeur par défaut
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Définit une valeur de configuration.
        
        Args:
            key (str): Clé de configuration (support notation pointée)
            value: Valeur à définir
        """
        keys = key.split('.')
        config = self.config
        
        # Naviguer jusqu'au dernier niveau
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Définir la valeur
        config[keys[-1]] = value
    
    def get_profile(self, profile_name: str) -> Dict[str, Any]:
        """
        Récupère un profil de configuration.
        
        Args:
            profile_name (str): Nom du profil (production, recherche, démo)
            
        Returns:
            Dict: Configuration du profil
        """
        return self.get(f'profiles.{profile_name}', {})
    
    def set_active_profile(self, profile_name: str) -> None:
        """
        Définit le profil actif.
        
        Args:
            profile_name (str): Nom du profil à activer
        """
        if profile_name in self.get('profiles', {}):
            self.set('active_profile', profile_name)
            console.print(f"[green]✓[/green] Profil actif: {profile_name}")
        else:
            console.print(f"[red]✗[/red] Profil inexistant: {profile_name}")
    
    def get_active_profile(self) -> Dict[str, Any]:
        """Retourne la configuration du profil actif."""
        active_profile = self.get('active_profile', 'production')
        return self.get_profile(active_profile)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Retourne la configuration par défaut."""
        return {
            'version': '1.0.0',
            'active_profile': 'production',
            'profiles': {
                'production': {
                    'model_path': 'Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25',
                    'data_path': 'data_generation/dataset_2D',
                    'output_path': 'cli/outputs',
                    'batch_size': 32,
                    'device': 'auto',
                    'precision': 'high'
                },
                'recherche': {
                    'model_path': 'Reseau_Neural_Dual_Gap_Lecran_FINAL_16_06_25',
                    'data_path': 'data_generation/dataset_2D_Train',
                    'output_path': 'cli/outputs/research',
                    'batch_size': 16,
                    'device': 'auto',
                    'precision': 'ultra'
                },
                'demo': {
                    'model_path': 'Reseau_Neural_Dual_Gap_Lecran_FINAL_16_06_25',
                    'data_path': 'data_generation/dataset_2D_Test',
                    'output_path': 'cli/outputs/demo',
                    'batch_size': 8,
                    'device': 'cpu',
                    'precision': 'standard'
                }
            },
            'ui': {
                'theme': 'blue',
                'progress_bars': True,
                'ascii_graphs': True,
                'rich_tables': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'cli/logs/cli.log',
                'console': True
            }
        }
