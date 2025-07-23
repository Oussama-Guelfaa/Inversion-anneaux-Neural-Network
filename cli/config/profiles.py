#!/usr/bin/env python3
"""
Gestionnaire de Profils de Configuration

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Classes pour gérer les profils de configuration.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
from rich.console import Console

console = Console()

class ConfigProfiles:
    """
    Gestionnaire des profils de configuration.
    
    Gère la création, modification et validation des profils
    de configuration pour différents cas d'usage.
    """
    
    def __init__(self):
        """Initialise le gestionnaire de profils."""
        self.console = console
    
    @staticmethod
    def get_default_profiles() -> Dict[str, Dict[str, Any]]:
        """
        Retourne les profils par défaut.
        
        Returns:
            Dict: Profils de configuration par défaut
        """
        return {
            'production': {
                'description': 'Profil pour utilisation en production avec modèle ultra-précision',
                'model_path': 'Reseau_Neural_Dual_Gap_Lecran_PRECISION_007um_14_01_25',
                'data_path': 'data_generation/dataset_2D',
                'output_path': 'cli/outputs/production',
                'batch_size': 32,
                'device': 'auto',
                'precision': 'ultra-high',
                'confidence_threshold': 0.95,
                'tolerance_gap': 0.007,
                'tolerance_L_ecran': 0.5,
                'use_gpu': True,
                'save_predictions': True,
                'generate_reports': True
            },
            'recherche': {
                'description': 'Profil pour recherche et développement',
                'model_path': 'Reseau_Neural_Dual_Gap_Lecran_FINAL_16_06_25',
                'data_path': 'data_generation/dataset_2D_Train',
                'output_path': 'cli/outputs/research',
                'batch_size': 16,
                'device': 'auto',
                'precision': 'high',
                'confidence_threshold': 0.8,
                'tolerance_gap': 0.01,
                'tolerance_L_ecran': 1.0,
                'use_gpu': True,
                'save_predictions': True,
                'generate_reports': True,
                'experimental_features': True,
                'verbose_logging': True
            },
            'demo': {
                'description': 'Profil pour démonstrations et tests rapides',
                'model_path': 'Reseau_Neural_Dual_Gap_Lecran_FINAL_16_06_25',
                'data_path': 'data_generation/dataset_2D_Test',
                'output_path': 'cli/outputs/demo',
                'batch_size': 8,
                'device': 'cpu',
                'precision': 'standard',
                'confidence_threshold': 0.7,
                'tolerance_gap': 0.02,
                'tolerance_L_ecran': 2.0,
                'use_gpu': False,
                'save_predictions': False,
                'generate_reports': False,
                'quick_mode': True
            },
            'gap_only': {
                'description': 'Profil pour prédiction du gap uniquement',
                'model_path': 'Reseaux_1D_Gap_Prediction/Reseau_Noise_Robustness',
                'data_path': 'data_generation/dataset_1D',
                'output_path': 'cli/outputs/gap_only',
                'batch_size': 64,
                'device': 'auto',
                'precision': 'standard',
                'confidence_threshold': 0.85,
                'tolerance_gap': 0.015,
                'use_gpu': True,
                'save_predictions': True,
                'generate_reports': True,
                'single_parameter': True
            }
        }
    
    @staticmethod
    def create_custom_profile(name: str, base_profile: str = 'production',
                            overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Crée un profil personnalisé basé sur un profil existant.
        
        Args:
            name (str): Nom du nouveau profil
            base_profile (str): Profil de base à utiliser
            overrides (Dict): Paramètres à modifier
            
        Returns:
            Dict: Nouveau profil de configuration
        """
        default_profiles = ConfigProfiles.get_default_profiles()
        
        if base_profile not in default_profiles:
            raise ValueError(f"Profil de base '{base_profile}' non trouvé")
        
        # Copier le profil de base
        new_profile = default_profiles[base_profile].copy()
        new_profile['description'] = f'Profil personnalisé basé sur {base_profile}'
        
        # Appliquer les modifications
        if overrides:
            new_profile.update(overrides)
        
        return new_profile
    
    @staticmethod
    def validate_profile_compatibility(profile_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide la compatibilité d'un profil avec le système.
        
        Args:
            profile_config (Dict): Configuration du profil
            
        Returns:
            Dict: Résultats de validation avec recommandations
        """
        validation_results = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'performance_estimate': 'unknown'
        }
        
        # Vérifier les chemins
        model_path = profile_config.get('model_path')
        data_path = profile_config.get('data_path')
        
        if model_path and not Path(model_path).exists():
            validation_results['errors'].append(f"Modèle non trouvé: {model_path}")
            validation_results['compatible'] = False
        
        if data_path and not Path(data_path).exists():
            validation_results['warnings'].append(f"Données non trouvées: {data_path}")
        
        # Vérifier la compatibilité du device
        device = profile_config.get('device', 'cpu')
        if device == 'cuda':
            try:
                import torch
                if not torch.cuda.is_available():
                    validation_results['warnings'].append("CUDA demandé mais non disponible")
                    validation_results['recommendations'].append("Utiliser 'auto' ou 'cpu' pour le device")
            except ImportError:
                validation_results['warnings'].append("PyTorch non disponible pour vérifier CUDA")
        
        elif device == 'mps':
            try:
                import torch
                if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    validation_results['warnings'].append("MPS demandé mais non disponible")
                    validation_results['recommendations'].append("Utiliser 'auto' ou 'cpu' pour le device")
            except ImportError:
                validation_results['warnings'].append("PyTorch non disponible pour vérifier MPS")
        
        # Estimer les performances
        batch_size = profile_config.get('batch_size', 32)
        precision = profile_config.get('precision', 'standard')
        
        if device == 'cpu' and batch_size > 16:
            validation_results['recommendations'].append("Réduire batch_size pour CPU (recommandé: ≤16)")
        
        if precision == 'ultra-high' and device == 'cpu':
            validation_results['warnings'].append("Précision ultra-haute sur CPU sera très lente")
            validation_results['recommendations'].append("Utiliser GPU pour précision ultra-haute")
        
        # Estimer les performances
        if device == 'cuda' and batch_size >= 32:
            validation_results['performance_estimate'] = 'excellent'
        elif device in ['cuda', 'mps'] and batch_size >= 16:
            validation_results['performance_estimate'] = 'good'
        elif device == 'cpu' and batch_size <= 16:
            validation_results['performance_estimate'] = 'acceptable'
        else:
            validation_results['performance_estimate'] = 'slow'
        
        return validation_results
    
    @staticmethod
    def optimize_profile_for_system(profile_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimise un profil pour le système actuel.
        
        Args:
            profile_config (Dict): Configuration du profil
            
        Returns:
            Dict: Profil optimisé
        """
        optimized = profile_config.copy()
        
        try:
            import torch
            
            # Optimiser le device
            if optimized.get('device') == 'auto':
                if torch.cuda.is_available():
                    optimized['device'] = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    optimized['device'] = 'mps'
                else:
                    optimized['device'] = 'cpu'
            
            # Optimiser batch_size selon le device
            device = optimized.get('device', 'cpu')
            current_batch_size = optimized.get('batch_size', 32)
            
            if device == 'cpu':
                # Réduire batch_size pour CPU
                optimized['batch_size'] = min(current_batch_size, 16)
            elif device in ['cuda', 'mps']:
                # Augmenter batch_size pour GPU si trop petit
                optimized['batch_size'] = max(current_batch_size, 32)
            
            # Ajuster la précision selon les capacités
            precision = optimized.get('precision', 'standard')
            if precision == 'ultra-high' and device == 'cpu':
                optimized['precision'] = 'high'
                optimized['_optimization_note'] = 'Précision réduite pour compatibilité CPU'
        
        except ImportError:
            # PyTorch non disponible, utiliser CPU par défaut
            optimized['device'] = 'cpu'
            optimized['batch_size'] = min(optimized.get('batch_size', 32), 8)
        
        return optimized
    
    @staticmethod
    def get_profile_recommendations(use_case: str) -> Dict[str, Any]:
        """
        Retourne des recommandations de profil selon le cas d'usage.
        
        Args:
            use_case (str): Cas d'usage ('production', 'research', 'demo', 'testing')
            
        Returns:
            Dict: Recommandations de configuration
        """
        recommendations = {
            'production': {
                'profile': 'production',
                'description': 'Optimisé pour précision maximale et fiabilité',
                'key_features': [
                    'Modèle ultra-précision (±0.007µm)',
                    'Seuil de confiance élevé (95%)',
                    'Sauvegarde automatique des résultats',
                    'Génération de rapports détaillés'
                ],
                'recommended_settings': {
                    'batch_size': 32,
                    'device': 'auto',
                    'precision': 'ultra-high',
                    'confidence_threshold': 0.95
                }
            },
            'research': {
                'profile': 'recherche',
                'description': 'Optimisé pour expérimentation et développement',
                'key_features': [
                    'Logging verbeux pour debugging',
                    'Fonctionnalités expérimentales activées',
                    'Tolérances ajustables',
                    'Accès aux données d\'entraînement'
                ],
                'recommended_settings': {
                    'batch_size': 16,
                    'device': 'auto',
                    'precision': 'high',
                    'confidence_threshold': 0.8,
                    'experimental_features': True
                }
            },
            'demo': {
                'profile': 'demo',
                'description': 'Optimisé pour démonstrations rapides',
                'key_features': [
                    'Exécution rapide sur CPU',
                    'Pas de sauvegarde automatique',
                    'Tolérances relaxées',
                    'Mode simplifié'
                ],
                'recommended_settings': {
                    'batch_size': 8,
                    'device': 'cpu',
                    'precision': 'standard',
                    'confidence_threshold': 0.7,
                    'quick_mode': True
                }
            },
            'testing': {
                'profile': 'demo',
                'description': 'Optimisé pour tests et validation',
                'key_features': [
                    'Paramètres conservateurs',
                    'Validation stricte',
                    'Logging détaillé',
                    'Sauvegarde des résultats de test'
                ],
                'recommended_settings': {
                    'batch_size': 16,
                    'device': 'auto',
                    'precision': 'high',
                    'confidence_threshold': 0.9,
                    'save_predictions': True,
                    'generate_reports': True
                }
            }
        }
        
        return recommendations.get(use_case, recommendations['production'])
    
    @staticmethod
    def merge_profiles(base_profile: Dict[str, Any], 
                      override_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fusionne deux profils de configuration.
        
        Args:
            base_profile (Dict): Profil de base
            override_profile (Dict): Profil avec les modifications
            
        Returns:
            Dict: Profil fusionné
        """
        merged = base_profile.copy()
        
        for key, value in override_profile.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                # Fusion récursive pour les dictionnaires
                merged[key] = ConfigProfiles.merge_profiles(merged[key], value)
            else:
                # Remplacement direct pour les autres types
                merged[key] = value
        
        return merged
    
    @staticmethod
    def export_profile(profile_config: Dict[str, Any], 
                      profile_name: str, output_path: str) -> bool:
        """
        Exporte un profil vers un fichier YAML.
        
        Args:
            profile_config (Dict): Configuration du profil
            profile_name (str): Nom du profil
            output_path (str): Chemin de sortie
            
        Returns:
            bool: True si l'export a réussi
        """
        try:
            import yaml
            
            export_data = {
                'profile_name': profile_name,
                'exported_at': str(Path().cwd()),
                'configuration': profile_config
            }
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
            
            return True
            
        except Exception as e:
            console.print(f"[red]Erreur lors de l'export: {e}[/red]")
            return False
    
    @staticmethod
    def import_profile(import_path: str) -> Optional[Dict[str, Any]]:
        """
        Importe un profil depuis un fichier YAML.
        
        Args:
            import_path (str): Chemin du fichier à importer
            
        Returns:
            Optional[Dict]: Profil importé ou None si erreur
        """
        try:
            import yaml
            
            import_file = Path(import_path)
            if not import_file.exists():
                console.print(f"[red]Fichier non trouvé: {import_path}[/red]")
                return None
            
            with open(import_file, 'r', encoding='utf-8') as f:
                import_data = yaml.safe_load(f)
            
            if 'configuration' in import_data:
                return import_data['configuration']
            else:
                # Fichier de profil direct
                return import_data
                
        except Exception as e:
            console.print(f"[red]Erreur lors de l'import: {e}[/red]")
            return None
