#!/usr/bin/env python3
"""
Validateurs de Configuration

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Classes pour valider les configurations YAML.
"""

from typing import Dict, List, Tuple, Any
from pathlib import Path

class ConfigValidator:
    """
    Validateur pour les configurations YAML du CLI.
    
    Vérifie la structure, les types et la cohérence
    des configurations.
    """
    
    @staticmethod
    def validate_full_config(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Valide une configuration complète.
        
        Args:
            config (Dict): Configuration à valider
            
        Returns:
            Tuple[bool, List[str], List[str]]: (Valide, Erreurs, Avertissements)
        """
        errors = []
        warnings = []
        
        # Validation de structure
        struct_valid, struct_errors, struct_warnings = ConfigValidator._validate_structure(config)
        errors.extend(struct_errors)
        warnings.extend(struct_warnings)
        
        # Validation des valeurs
        values_valid, values_errors, values_warnings = ConfigValidator._validate_values(config)
        errors.extend(values_errors)
        warnings.extend(values_warnings)
        
        # Validation des chemins
        paths_valid, paths_errors, paths_warnings = ConfigValidator._validate_paths(config)
        errors.extend(paths_errors)
        warnings.extend(paths_warnings)
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    @staticmethod
    def _validate_structure(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Valide la structure de base de la configuration."""
        errors = []
        warnings = []
        
        # Clés requises au niveau racine
        required_keys = ['version', 'active_profile', 'profiles']
        for key in required_keys:
            if key not in config:
                errors.append(f"Clé requise manquante: {key}")
        
        # Validation des profils
        if 'profiles' in config:
            if not isinstance(config['profiles'], dict):
                errors.append("'profiles' doit être un dictionnaire")
            else:
                for profile_name, profile_config in config['profiles'].items():
                    if not isinstance(profile_config, dict):
                        errors.append(f"Le profil '{profile_name}' doit être un dictionnaire")
                        continue
                    
                    # Clés recommandées pour un profil
                    recommended_profile_keys = [
                        'model_path', 'data_path', 'output_path', 
                        'batch_size', 'device', 'precision'
                    ]
                    
                    for key in recommended_profile_keys:
                        if key not in profile_config:
                            warnings.append(f"Clé recommandée manquante dans '{profile_name}': {key}")
        
        # Validation du profil actif
        if 'active_profile' in config and 'profiles' in config:
            active_profile = config['active_profile']
            if isinstance(config['profiles'], dict) and active_profile not in config['profiles']:
                errors.append(f"Le profil actif '{active_profile}' n'existe pas")
        
        # Validation des sections optionnelles
        optional_sections = ['ui', 'logging', 'defaults']
        for section in optional_sections:
            if section in config and not isinstance(config[section], dict):
                warnings.append(f"La section '{section}' devrait être un dictionnaire")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    @staticmethod
    def _validate_values(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Valide les valeurs de la configuration."""
        errors = []
        warnings = []
        
        # Validation de la version
        if 'version' in config:
            version = config['version']
            if not isinstance(version, str):
                errors.append("La version doit être une chaîne de caractères")
        
        # Validation des profils
        if 'profiles' in config and isinstance(config['profiles'], dict):
            for profile_name, profile_config in config['profiles'].items():
                if isinstance(profile_config, dict):
                    # Validation batch_size
                    if 'batch_size' in profile_config:
                        batch_size = profile_config['batch_size']
                        if not isinstance(batch_size, int) or batch_size <= 0:
                            errors.append(f"batch_size invalide dans '{profile_name}': {batch_size}")
                    
                    # Validation device
                    if 'device' in profile_config:
                        device = profile_config['device']
                        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
                        if device not in valid_devices:
                            warnings.append(f"Device non standard dans '{profile_name}': {device}")
                    
                    # Validation precision
                    if 'precision' in profile_config:
                        precision = profile_config['precision']
                        valid_precisions = ['standard', 'high', 'ultra', 'ultra-high']
                        if precision not in valid_precisions:
                            warnings.append(f"Niveau de précision non standard dans '{profile_name}': {precision}")
        
        # Validation de la configuration UI
        if 'ui' in config and isinstance(config['ui'], dict):
            ui_config = config['ui']
            
            # Validation du thème
            if 'theme' in ui_config:
                theme = ui_config['theme']
                valid_themes = ['blue', 'green', 'red', 'yellow', 'magenta', 'cyan']
                if theme not in valid_themes:
                    warnings.append(f"Thème non standard: {theme}")
            
            # Validation des booléens
            boolean_keys = ['progress_bars', 'ascii_graphs', 'rich_tables', 'interactive_menus']
            for key in boolean_keys:
                if key in ui_config and not isinstance(ui_config[key], bool):
                    errors.append(f"'{key}' doit être un booléen")
        
        # Validation de la configuration de logging
        if 'logging' in config and isinstance(config['logging'], dict):
            logging_config = config['logging']
            
            # Validation du niveau de log
            if 'level' in logging_config:
                level = logging_config['level']
                valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                if level not in valid_levels:
                    errors.append(f"Niveau de log invalide: {level}")
            
            # Validation console
            if 'console' in logging_config and not isinstance(logging_config['console'], bool):
                errors.append("'console' dans logging doit être un booléen")
        
        # Validation des paramètres par défaut
        if 'defaults' in config and isinstance(config['defaults'], dict):
            defaults = config['defaults']
            
            # Validation des paramètres d'entraînement
            if 'train' in defaults and isinstance(defaults['train'], dict):
                train_defaults = defaults['train']
                
                if 'epochs' in train_defaults:
                    epochs = train_defaults['epochs']
                    if not isinstance(epochs, int) or epochs <= 0:
                        errors.append(f"Nombre d'époques invalide: {epochs}")
                
                if 'early_stopping' in train_defaults:
                    if not isinstance(train_defaults['early_stopping'], bool):
                        errors.append("'early_stopping' doit être un booléen")
            
            # Validation des paramètres de test
            if 'test' in defaults and isinstance(defaults['test'], dict):
                test_defaults = defaults['test']
                
                tolerance_keys = ['tolerance_gap', 'tolerance_L_ecran']
                for key in tolerance_keys:
                    if key in test_defaults:
                        tolerance = test_defaults[key]
                        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
                            errors.append(f"Tolérance invalide '{key}': {tolerance}")
                
                if 'noise_levels' in test_defaults:
                    noise_levels = test_defaults['noise_levels']
                    if not isinstance(noise_levels, list):
                        errors.append("'noise_levels' doit être une liste")
                    else:
                        for level in noise_levels:
                            if not isinstance(level, (int, float)) or level < 0:
                                errors.append(f"Niveau de bruit invalide: {level}")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    @staticmethod
    def _validate_paths(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Valide les chemins dans la configuration."""
        errors = []
        warnings = []
        
        # Validation des chemins dans les profils
        if 'profiles' in config and isinstance(config['profiles'], dict):
            for profile_name, profile_config in config['profiles'].items():
                if isinstance(profile_config, dict):
                    # Chemins à vérifier
                    path_keys = ['model_path', 'data_path', 'output_path']
                    
                    for path_key in path_keys:
                        if path_key in profile_config:
                            path_value = profile_config[path_key]
                            if not isinstance(path_value, str):
                                errors.append(f"'{path_key}' dans '{profile_name}' doit être une chaîne")
                                continue
                            
                            path_obj = Path(path_value)
                            
                            # Pour model_path et data_path, vérifier l'existence
                            if path_key in ['model_path', 'data_path']:
                                if not path_obj.exists():
                                    warnings.append(f"Chemin inexistant '{path_key}' dans '{profile_name}': {path_value}")
                            
                            # Pour output_path, vérifier que le parent existe ou peut être créé
                            elif path_key == 'output_path':
                                try:
                                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                                except Exception:
                                    warnings.append(f"Impossible de créer le dossier parent pour '{path_key}' dans '{profile_name}': {path_value}")
        
        # Validation du fichier de log
        if 'logging' in config and isinstance(config['logging'], dict):
            logging_config = config['logging']
            if 'file' in logging_config:
                log_file = logging_config['file']
                if isinstance(log_file, str):
                    log_path = Path(log_file)
                    try:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        warnings.append(f"Impossible de créer le dossier pour le fichier de log: {log_file}")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    @staticmethod
    def validate_profile(profile_config: Dict[str, Any], profile_name: str) -> Tuple[bool, List[str], List[str]]:
        """
        Valide un profil spécifique.
        
        Args:
            profile_config (Dict): Configuration du profil
            profile_name (str): Nom du profil
            
        Returns:
            Tuple[bool, List[str], List[str]]: (Valide, Erreurs, Avertissements)
        """
        errors = []
        warnings = []
        
        if not isinstance(profile_config, dict):
            errors.append(f"Le profil '{profile_name}' doit être un dictionnaire")
            return False, errors, warnings
        
        # Clés requises pour un profil
        required_keys = ['model_path', 'data_path']
        for key in required_keys:
            if key not in profile_config:
                errors.append(f"Clé requise manquante dans '{profile_name}': {key}")
        
        # Validation des chemins
        for path_key in ['model_path', 'data_path']:
            if path_key in profile_config:
                path_value = profile_config[path_key]
                if not isinstance(path_value, str):
                    errors.append(f"'{path_key}' dans '{profile_name}' doit être une chaîne")
                elif not Path(path_value).exists():
                    warnings.append(f"Chemin inexistant '{path_key}' dans '{profile_name}': {path_value}")
        
        # Validation des paramètres numériques
        if 'batch_size' in profile_config:
            batch_size = profile_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append(f"batch_size invalide dans '{profile_name}': {batch_size}")
        
        # Validation du device
        if 'device' in profile_config:
            device = profile_config['device']
            valid_devices = ['auto', 'cpu', 'cuda', 'mps']
            if device not in valid_devices:
                warnings.append(f"Device non standard dans '{profile_name}': {device}")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
