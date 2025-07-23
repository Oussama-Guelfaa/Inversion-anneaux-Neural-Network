#!/usr/bin/env python3
"""
Validateurs CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Classes de validation pour les chemins, modèles et configurations.
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from rich.console import Console

console = Console()

class PathValidator:
    """
    Validateur pour les chemins de fichiers et dossiers.
    
    Vérifie l'existence, les permissions et la structure
    des chemins utilisés par le CLI.
    """
    
    @staticmethod
    def validate_file_path(file_path: str, extensions: List[str] = None) -> Tuple[bool, str]:
        """
        Valide un chemin de fichier.
        
        Args:
            file_path (str): Chemin du fichier à valider
            extensions (List[str]): Extensions autorisées (optionnel)
            
        Returns:
            Tuple[bool, str]: (Valide, Message d'erreur si invalide)
        """
        path = Path(file_path)
        
        # Vérifier l'existence
        if not path.exists():
            return False, f"Fichier non trouvé: {file_path}"
        
        # Vérifier que c'est un fichier
        if not path.is_file():
            return False, f"Le chemin n'est pas un fichier: {file_path}"
        
        # Vérifier l'extension si spécifiée
        if extensions:
            if path.suffix.lower() not in [ext.lower() for ext in extensions]:
                return False, f"Extension non autorisée. Attendu: {extensions}, trouvé: {path.suffix}"
        
        # Vérifier les permissions de lecture
        if not os.access(path, os.R_OK):
            return False, f"Permissions de lecture insuffisantes: {file_path}"
        
        return True, ""
    
    @staticmethod
    def validate_directory_path(dir_path: str, must_exist: bool = True,
                              create_if_missing: bool = False) -> Tuple[bool, str]:
        """
        Valide un chemin de dossier.
        
        Args:
            dir_path (str): Chemin du dossier à valider
            must_exist (bool): Le dossier doit-il exister
            create_if_missing (bool): Créer le dossier s'il n'existe pas
            
        Returns:
            Tuple[bool, str]: (Valide, Message d'erreur si invalide)
        """
        path = Path(dir_path)
        
        if not path.exists():
            if must_exist and not create_if_missing:
                return False, f"Dossier non trouvé: {dir_path}"
            elif create_if_missing:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    return True, f"Dossier créé: {dir_path}"
                except Exception as e:
                    return False, f"Impossible de créer le dossier {dir_path}: {str(e)}"
        
        # Vérifier que c'est un dossier
        if path.exists() and not path.is_dir():
            return False, f"Le chemin n'est pas un dossier: {dir_path}"
        
        # Vérifier les permissions
        if path.exists():
            if not os.access(path, os.R_OK):
                return False, f"Permissions de lecture insuffisantes: {dir_path}"
            if not os.access(path, os.W_OK):
                return False, f"Permissions d'écriture insuffisantes: {dir_path}"
        
        return True, ""
    
    @staticmethod
    def validate_data_directory(data_path: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Valide un dossier de données et analyse son contenu.
        
        Args:
            data_path (str): Chemin du dossier de données
            
        Returns:
            Tuple[bool, str, Dict]: (Valide, Message, Informations sur les données)
        """
        # Valider le dossier de base
        valid, message = PathValidator.validate_directory_path(data_path, must_exist=True)
        if not valid:
            return False, message, {}
        
        path = Path(data_path)
        info = {
            'total_files': 0,
            'data_files': 0,
            'file_types': {},
            'estimated_size': 0,
            'has_mat_files': False,
            'has_csv_files': False,
            'has_json_files': False
        }
        
        # Analyser le contenu
        try:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    info['total_files'] += 1
                    
                    # Analyser l'extension
                    ext = file_path.suffix.lower()
                    info['file_types'][ext] = info['file_types'].get(ext, 0) + 1
                    
                    # Vérifier les types de données
                    if ext == '.mat':
                        info['has_mat_files'] = True
                        info['data_files'] += 1
                    elif ext == '.csv':
                        info['has_csv_files'] = True
                        info['data_files'] += 1
                    elif ext == '.json':
                        info['has_json_files'] = True
                        info['data_files'] += 1
                    
                    # Estimer la taille
                    try:
                        info['estimated_size'] += file_path.stat().st_size
                    except:
                        pass
        
        except Exception as e:
            return False, f"Erreur lors de l'analyse du dossier: {str(e)}", info
        
        # Vérifier qu'il y a des fichiers de données
        if info['data_files'] == 0:
            return False, "Aucun fichier de données trouvé (.mat, .csv, .json)", info
        
        return True, f"Dossier valide avec {info['data_files']} fichiers de données", info

class ModelValidator:
    """
    Validateur pour les modèles de réseaux neuronaux.
    
    Vérifie la structure, les fichiers requis et la compatibilité
    des modèles PyTorch.
    """
    
    @staticmethod
    def validate_model_directory(model_path: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Valide un dossier de modèle complet.
        
        Args:
            model_path (str): Chemin du dossier du modèle
            
        Returns:
            Tuple[bool, str, Dict]: (Valide, Message, Informations sur le modèle)
        """
        # Valider le dossier de base
        valid, message = PathValidator.validate_directory_path(model_path, must_exist=True)
        if not valid:
            return False, message, {}
        
        path = Path(model_path)
        info = {
            'model_type': 'unknown',
            'has_source': False,
            'has_trained_model': False,
            'has_config': False,
            'python_files': [],
            'model_files': [],
            'config_files': [],
            'estimated_precision': 'unknown'
        }
        
        # Vérifier la structure requise
        required_dirs = ['src']
        optional_dirs = ['models', 'config', 'data', 'results', 'plots']
        
        missing_required = []
        for req_dir in required_dirs:
            if not (path / req_dir).exists():
                missing_required.append(req_dir)
            else:
                info['has_source'] = True
        
        if missing_required:
            return False, f"Dossiers requis manquants: {missing_required}", info
        
        # Analyser le contenu
        try:
            # Fichiers Python dans src/
            src_path = path / 'src'
            if src_path.exists():
                python_files = list(src_path.glob('*.py'))
                info['python_files'] = [f.name for f in python_files]
            
            # Modèles entraînés
            models_path = path / 'models'
            if models_path.exists():
                model_files = list(models_path.glob('*.pth')) + list(models_path.glob('*.pt'))
                info['model_files'] = [f.name for f in model_files]
                info['has_trained_model'] = len(model_files) > 0
            
            # Fichiers de configuration
            config_path = path / 'config'
            if config_path.exists():
                config_files = list(config_path.glob('*.yaml')) + list(config_path.glob('*.yml'))
                info['config_files'] = [f.name for f in config_files]
                info['has_config'] = len(config_files) > 0
            
            # Déterminer le type de modèle basé sur le nom
            model_name = path.name.lower()
            if 'precision' in model_name:
                info['model_type'] = 'ultra-precision'
                info['estimated_precision'] = 'ultra-high'
            elif 'final' in model_name:
                info['model_type'] = 'production'
                info['estimated_precision'] = 'high'
            elif '2d' in model_name:
                info['model_type'] = 'dual-parameter'
                info['estimated_precision'] = 'standard'
            elif '1d' in model_name:
                info['model_type'] = 'gap-only'
                info['estimated_precision'] = 'standard'
        
        except Exception as e:
            return False, f"Erreur lors de l'analyse du modèle: {str(e)}", info
        
        return True, "Modèle valide", info
    
    @staticmethod
    def validate_pytorch_model(model_file: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Valide un fichier de modèle PyTorch.
        
        Args:
            model_file (str): Chemin du fichier .pth/.pt
            
        Returns:
            Tuple[bool, str, Dict]: (Valide, Message, Informations sur le modèle)
        """
        # Valider le fichier
        valid, message = PathValidator.validate_file_path(model_file, ['.pth', '.pt'])
        if not valid:
            return False, message, {}
        
        info = {
            'file_size': 0,
            'pytorch_version': 'unknown',
            'model_structure': {},
            'device_compatible': False,
            'loadable': False
        }
        
        try:
            # Informations sur le fichier
            file_path = Path(model_file)
            info['file_size'] = file_path.stat().st_size
            
            # Tenter de charger le modèle
            try:
                # Charger sur CPU pour éviter les problèmes de device
                checkpoint = torch.load(model_file, map_location='cpu')
                info['loadable'] = True
                
                # Analyser la structure
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Compter les paramètres
                    if isinstance(state_dict, dict):
                        total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
                        info['model_structure']['total_parameters'] = total_params
                        info['model_structure']['layers'] = len(state_dict)
                
                # Vérifier la compatibilité avec les devices disponibles
                info['device_compatible'] = True
                
            except Exception as load_error:
                return False, f"Impossible de charger le modèle: {str(load_error)}", info
        
        except Exception as e:
            return False, f"Erreur lors de la validation du modèle: {str(e)}", info
        
        return True, "Modèle PyTorch valide", info
    
    @staticmethod
    def check_model_compatibility(model_path: str, data_path: str) -> Tuple[bool, str]:
        """
        Vérifie la compatibilité entre un modèle et des données.
        
        Args:
            model_path (str): Chemin du modèle
            data_path (str): Chemin des données
            
        Returns:
            Tuple[bool, str]: (Compatible, Message)
        """
        # Valider le modèle
        model_valid, model_msg, model_info = ModelValidator.validate_model_directory(model_path)
        if not model_valid:
            return False, f"Modèle invalide: {model_msg}"
        
        # Valider les données
        data_valid, data_msg, data_info = PathValidator.validate_data_directory(data_path)
        if not data_valid:
            return False, f"Données invalides: {data_msg}"
        
        # Vérifications de compatibilité
        compatibility_issues = []
        
        # Vérifier le type de données vs type de modèle
        model_type = model_info.get('model_type', 'unknown')
        
        if model_type == 'gap-only':
            # Les modèles gap-only nécessitent des profils 1D
            if not data_info.get('has_mat_files') and not data_info.get('has_csv_files'):
                compatibility_issues.append("Modèle gap-only nécessite des fichiers .mat ou .csv")
        
        elif model_type in ['dual-parameter', 'ultra-precision', 'production']:
            # Les modèles dual nécessitent des données 2D
            if not data_info.get('has_mat_files'):
                compatibility_issues.append("Modèle dual-parameter nécessite des fichiers .mat")
        
        # Vérifier la taille des données
        if data_info.get('data_files', 0) < 10:
            compatibility_issues.append("Nombre insuffisant de fichiers de données (< 10)")
        
        if compatibility_issues:
            return False, "Incompatibilités détectées: " + "; ".join(compatibility_issues)
        
        return True, "Modèle et données compatibles"

class ConfigValidator:
    """
    Validateur pour les configurations YAML.
    
    Vérifie la structure, les valeurs et la cohérence
    des fichiers de configuration.
    """
    
    @staticmethod
    def validate_config_structure(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Valide la structure d'une configuration.
        
        Args:
            config (Dict): Configuration à valider
            
        Returns:
            Tuple[bool, List[str], List[str]]: (Valide, Erreurs, Avertissements)
        """
        errors = []
        warnings = []
        
        # Vérifier les clés requises
        required_keys = ['version', 'active_profile', 'profiles']
        for key in required_keys:
            if key not in config:
                errors.append(f"Clé requise manquante: {key}")
        
        # Vérifier les profils
        if 'profiles' in config:
            profiles = config['profiles']
            if not isinstance(profiles, dict):
                errors.append("'profiles' doit être un dictionnaire")
            else:
                # Vérifier chaque profil
                for profile_name, profile_config in profiles.items():
                    if not isinstance(profile_config, dict):
                        errors.append(f"Profil '{profile_name}' doit être un dictionnaire")
                        continue
                    
                    # Clés recommandées pour un profil
                    recommended_keys = ['model_path', 'data_path', 'output_path', 'batch_size']
                    for key in recommended_keys:
                        if key not in profile_config:
                            warnings.append(f"Clé recommandée manquante dans '{profile_name}': {key}")
        
        # Vérifier le profil actif
        if 'active_profile' in config and 'profiles' in config:
            active_profile = config['active_profile']
            if active_profile not in config['profiles']:
                errors.append(f"Profil actif '{active_profile}' n'existe pas dans les profils")
        
        # Vérifier la configuration UI
        if 'ui' in config:
            ui_config = config['ui']
            if not isinstance(ui_config, dict):
                warnings.append("'ui' devrait être un dictionnaire")
        
        # Vérifier la configuration de logging
        if 'logging' in config:
            logging_config = config['logging']
            if not isinstance(logging_config, dict):
                warnings.append("'logging' devrait être un dictionnaire")
            else:
                if 'level' in logging_config:
                    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                    if logging_config['level'] not in valid_levels:
                        warnings.append(f"Niveau de log invalide: {logging_config['level']}")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    @staticmethod
    def validate_config_values(config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Valide les valeurs d'une configuration.
        
        Args:
            config (Dict): Configuration à valider
            
        Returns:
            Tuple[bool, List[str], List[str]]: (Valide, Erreurs, Avertissements)
        """
        errors = []
        warnings = []
        
        # Valider les chemins dans les profils
        if 'profiles' in config:
            for profile_name, profile_config in config['profiles'].items():
                if isinstance(profile_config, dict):
                    # Vérifier les chemins
                    for path_key in ['model_path', 'data_path']:
                        if path_key in profile_config:
                            path_value = profile_config[path_key]
                            if not Path(path_value).exists():
                                warnings.append(f"Chemin inexistant dans {profile_name}.{path_key}: {path_value}")
                    
                    # Vérifier les valeurs numériques
                    if 'batch_size' in profile_config:
                        batch_size = profile_config['batch_size']
                        if not isinstance(batch_size, int) or batch_size <= 0:
                            errors.append(f"batch_size invalide dans {profile_name}: {batch_size}")
                    
                    # Vérifier le device
                    if 'device' in profile_config:
                        device = profile_config['device']
                        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
                        if device not in valid_devices:
                            warnings.append(f"Device non reconnu dans {profile_name}: {device}")
        
        # Valider les paramètres par défaut
        if 'defaults' in config:
            defaults = config['defaults']
            if isinstance(defaults, dict):
                # Vérifier les paramètres d'entraînement
                if 'train' in defaults and isinstance(defaults['train'], dict):
                    train_defaults = defaults['train']
                    if 'epochs' in train_defaults:
                        epochs = train_defaults['epochs']
                        if not isinstance(epochs, int) or epochs <= 0:
                            errors.append(f"Nombre d'époques invalide: {epochs}")
                
                # Vérifier les paramètres de test
                if 'test' in defaults and isinstance(defaults['test'], dict):
                    test_defaults = defaults['test']
                    for tolerance_key in ['tolerance_gap', 'tolerance_L_ecran']:
                        if tolerance_key in test_defaults:
                            tolerance = test_defaults[tolerance_key]
                            if not isinstance(tolerance, (int, float)) or tolerance <= 0:
                                errors.append(f"Tolérance invalide {tolerance_key}: {tolerance}")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
