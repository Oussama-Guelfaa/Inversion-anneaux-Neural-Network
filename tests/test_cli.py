#!/usr/bin/env python3
"""
Tests Unitaires pour le CLI

Auteur: Oussama GUELFAA
Date: 08 - 07 - 2025

Suite de tests pour valider les fonctionnalités du CLI.
"""

import unittest
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.config.config_manager import CLIConfig
from cli.config.profiles import ConfigProfiles
from cli.config.validators import ConfigValidator
from cli.utils.cli_utils import CLIUtils
from cli.utils.formatters import RichFormatter, ASCIIGraphs
from cli.utils.validators import PathValidator, ModelValidator

class TestCLIConfig(unittest.TestCase):
    """Tests pour la gestion de configuration."""
    
    def setUp(self):
        """Prépare les tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"
    
    def tearDown(self):
        """Nettoie après les tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_creation(self):
        """Test la création d'une configuration."""
        config = CLIConfig(str(self.config_file))
        self.assertIsInstance(config.config, dict)
        self.assertIn('version', config.config)
        self.assertIn('profiles', config.config)
    
    def test_config_get_set(self):
        """Test les opérations get/set."""
        config = CLIConfig(str(self.config_file))
        
        # Test set/get simple
        config.set('test_key', 'test_value')
        self.assertEqual(config.get('test_key'), 'test_value')
        
        # Test notation pointée
        config.set('nested.key', 'nested_value')
        self.assertEqual(config.get('nested.key'), 'nested_value')
        
        # Test valeur par défaut
        self.assertEqual(config.get('nonexistent', 'default'), 'default')
    
    def test_profile_management(self):
        """Test la gestion des profils."""
        config = CLIConfig(str(self.config_file))
        
        # Test profil actif
        config.set_active_profile('production')
        self.assertEqual(config.get('active_profile'), 'production')
        
        # Test récupération de profil
        profile = config.get_active_profile()
        self.assertIsInstance(profile, dict)
    
    def test_config_save_load(self):
        """Test la sauvegarde et le chargement."""
        config = CLIConfig(str(self.config_file))
        config.set('test_save', 'saved_value')
        config.save_config()
        
        # Charger une nouvelle instance
        config2 = CLIConfig(str(self.config_file))
        self.assertEqual(config2.get('test_save'), 'saved_value')

class TestConfigProfiles(unittest.TestCase):
    """Tests pour les profils de configuration."""
    
    def test_default_profiles(self):
        """Test les profils par défaut."""
        profiles = ConfigProfiles.get_default_profiles()
        
        self.assertIn('production', profiles)
        self.assertIn('recherche', profiles)
        self.assertIn('demo', profiles)
        
        # Vérifier la structure d'un profil
        prod_profile = profiles['production']
        self.assertIn('model_path', prod_profile)
        self.assertIn('data_path', prod_profile)
        self.assertIn('batch_size', prod_profile)
    
    def test_custom_profile_creation(self):
        """Test la création de profils personnalisés."""
        custom = ConfigProfiles.create_custom_profile(
            'custom_test',
            'production',
            {'batch_size': 64, 'custom_param': 'test'}
        )
        
        self.assertEqual(custom['batch_size'], 64)
        self.assertEqual(custom['custom_param'], 'test')
        self.assertIn('model_path', custom)  # Hérité du profil de base
    
    def test_profile_validation(self):
        """Test la validation de profils."""
        profiles = ConfigProfiles.get_default_profiles()
        prod_profile = profiles['production']
        
        validation = ConfigProfiles.validate_profile_compatibility(prod_profile)
        
        self.assertIn('compatible', validation)
        self.assertIn('warnings', validation)
        self.assertIn('errors', validation)
        self.assertIn('performance_estimate', validation)
    
    def test_profile_optimization(self):
        """Test l'optimisation de profils."""
        test_profile = {
            'device': 'auto',
            'batch_size': 32,
            'precision': 'ultra-high'
        }
        
        optimized = ConfigProfiles.optimize_profile_for_system(test_profile)
        
        self.assertIn('device', optimized)
        self.assertIn('batch_size', optimized)
        # Le device devrait être résolu (pas 'auto')
        self.assertNotEqual(optimized['device'], 'auto')

class TestConfigValidator(unittest.TestCase):
    """Tests pour la validation de configuration."""
    
    def test_structure_validation(self):
        """Test la validation de structure."""
        # Configuration valide
        valid_config = {
            'version': '1.0.0',
            'active_profile': 'production',
            'profiles': {
                'production': {
                    'model_path': 'test/path',
                    'data_path': 'test/data'
                }
            }
        }
        
        is_valid, errors, warnings = ConfigValidator.validate_full_config(valid_config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Configuration invalide
        invalid_config = {
            'version': '1.0.0'
            # Manque active_profile et profiles
        }
        
        is_valid, errors, warnings = ConfigValidator.validate_full_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_values_validation(self):
        """Test la validation des valeurs."""
        config_with_invalid_values = {
            'version': '1.0.0',
            'active_profile': 'test',
            'profiles': {
                'test': {
                    'batch_size': -1,  # Invalide
                    'device': 'invalid_device'  # Avertissement
                }
            }
        }
        
        is_valid, errors, warnings = ConfigValidator.validate_full_config(config_with_invalid_values)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

class TestCLIUtils(unittest.TestCase):
    """Tests pour les utilitaires CLI."""
    
    def setUp(self):
        """Prépare les tests."""
        self.utils = CLIUtils()
    
    def test_device_detection(self):
        """Test la détection de device."""
        device = self.utils.detect_device()
        self.assertIn(device, ['cpu', 'cuda', 'mps'])
    
    def test_model_path_validation(self):
        """Test la validation de chemins de modèles."""
        # Chemin inexistant
        self.assertFalse(self.utils.validate_model_path('nonexistent/path'))
        
        # Chemin existant (répertoire courant)
        self.assertTrue(self.utils.validate_model_path('.'))
    
    def test_data_path_validation(self):
        """Test la validation de chemins de données."""
        # Chemin inexistant
        self.assertFalse(self.utils.validate_data_path('nonexistent/path'))
        
        # Répertoire courant (devrait exister)
        self.assertTrue(self.utils.validate_data_path('.'))
    
    def test_metrics_formatting(self):
        """Test le formatage des métriques."""
        metrics = {
            'r2_gap': 0.9946,
            'mae_gap': 0.003421,
            'accuracy': 0.97
        }
        
        formatted = self.utils.format_metrics(metrics)
        self.assertIsInstance(formatted, str)
        self.assertIn('r2_gap', formatted)
        self.assertIn('%', formatted)  # Pour accuracy

class TestRichFormatter(unittest.TestCase):
    """Tests pour le formateur Rich."""
    
    def setUp(self):
        """Prépare les tests."""
        self.formatter = RichFormatter()
    
    def test_header_panel_creation(self):
        """Test la création de panneaux d'en-tête."""
        panel = self.formatter.create_header_panel("Test Title", "Test Subtitle")
        self.assertIsNotNone(panel)
    
    def test_metrics_table_creation(self):
        """Test la création de tableaux de métriques."""
        metrics = {
            'r2': 0.9946,
            'mae': 0.003421,
            'rmse': 0.005123
        }
        
        table = self.formatter.create_metrics_table(metrics)
        self.assertIsNotNone(table)
    
    def test_comparison_table_creation(self):
        """Test la création de tableaux de comparaison."""
        data1 = {'metric1': 0.95, 'metric2': 0.03}
        data2 = {'metric1': 0.93, 'metric2': 0.04}
        
        table = self.formatter.create_comparison_table(data1, data2, "Model 1", "Model 2")
        self.assertIsNotNone(table)
    
    def test_status_panel_creation(self):
        """Test la création de panneaux de statut."""
        panel = self.formatter.create_status_panel("success", "Test message")
        self.assertIsNotNone(panel)
        
        panel = self.formatter.create_status_panel("error", "Error message")
        self.assertIsNotNone(panel)

class TestASCIIGraphs(unittest.TestCase):
    """Tests pour les graphiques ASCII."""
    
    def setUp(self):
        """Prépare les tests."""
        self.graphs = ASCIIGraphs(width=40, height=10)
    
    def test_line_chart_creation(self):
        """Test la création de graphiques en ligne."""
        data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        chart = self.graphs.line_chart(data, "Test Line Chart")
        
        self.assertIsInstance(chart, str)
        self.assertIn("Test Line Chart", chart)
        self.assertIn("●", chart)  # Points du graphique
    
    def test_bar_chart_creation(self):
        """Test la création de graphiques en barres."""
        data = [10, 20, 15, 25, 30]
        labels = ["A", "B", "C", "D", "E"]
        
        chart = self.graphs.bar_chart(data, labels, "Test Bar Chart")
        
        self.assertIsInstance(chart, str)
        self.assertIn("Test Bar Chart", chart)
        self.assertIn("█", chart)  # Barres du graphique
    
    def test_histogram_creation(self):
        """Test la création d'histogrammes."""
        import numpy as np
        data = np.random.normal(0, 1, 100).tolist()
        
        histogram = self.graphs.histogram(data, bins=5, title="Test Histogram")
        
        self.assertIsInstance(histogram, str)
        self.assertIn("Test Histogram", histogram)
    
    def test_scatter_plot_creation(self):
        """Test la création de nuages de points."""
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 1, 5, 3]
        
        scatter = self.graphs.scatter_plot(x_data, y_data, "Test Scatter")
        
        self.assertIsInstance(scatter, str)
        self.assertIn("Test Scatter", scatter)
    
    def test_empty_data_handling(self):
        """Test la gestion de données vides."""
        chart = self.graphs.line_chart([], "Empty Chart")
        self.assertIn("Aucune donnée", chart)
        
        bar_chart = self.graphs.bar_chart([], title="Empty Bar Chart")
        self.assertIn("Aucune donnée", bar_chart)

class TestPathValidator(unittest.TestCase):
    """Tests pour la validation de chemins."""
    
    def test_file_path_validation(self):
        """Test la validation de fichiers."""
        # Fichier existant (ce script de test)
        valid, message = PathValidator.validate_file_path(__file__)
        self.assertTrue(valid)
        
        # Fichier inexistant
        valid, message = PathValidator.validate_file_path('nonexistent_file.txt')
        self.assertFalse(valid)
        self.assertIn("non trouvé", message)
    
    def test_directory_path_validation(self):
        """Test la validation de dossiers."""
        # Dossier existant
        valid, message = PathValidator.validate_directory_path('.')
        self.assertTrue(valid)
        
        # Dossier inexistant sans création
        valid, message = PathValidator.validate_directory_path('nonexistent_dir', must_exist=True)
        self.assertFalse(valid)
    
    def test_data_directory_validation(self):
        """Test la validation de dossiers de données."""
        # Dossier courant (devrait contenir des fichiers)
        valid, message, info = PathValidator.validate_data_directory('.')
        
        self.assertIsInstance(info, dict)
        self.assertIn('total_files', info)
        self.assertIn('file_types', info)

class TestModelValidator(unittest.TestCase):
    """Tests pour la validation de modèles."""
    
    def test_model_directory_validation(self):
        """Test la validation de dossiers de modèles."""
        # Dossier inexistant
        valid, message, info = ModelValidator.validate_model_directory('nonexistent_model')
        self.assertFalse(valid)
        
        # Dossier existant mais sans structure de modèle
        valid, message, info = ModelValidator.validate_model_directory('.')
        self.assertIsInstance(info, dict)

class TestIntegration(unittest.TestCase):
    """Tests d'intégration pour le CLI."""
    
    def test_config_profile_integration(self):
        """Test l'intégration configuration-profils."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "integration_test.yaml"
            
            # Créer une configuration avec profils
            config = CLIConfig(str(config_file))
            profiles = ConfigProfiles.get_default_profiles()
            
            config.config['profiles'] = profiles
            config.save_config()
            
            # Recharger et vérifier
            config2 = CLIConfig(str(config_file))
            self.assertEqual(len(config2.get('profiles', {})), len(profiles))
    
    def test_formatter_utils_integration(self):
        """Test l'intégration formateur-utilitaires."""
        utils = CLIUtils()
        formatter = RichFormatter()
        
        # Créer des métriques fictives
        metrics = {
            'r2_gap': 0.9946,
            'mae_gap': 0.003421,
            'accuracy': 0.97
        }
        
        # Formater avec les utilitaires
        formatted_str = utils.format_metrics(metrics)
        self.assertIsInstance(formatted_str, str)
        
        # Créer un tableau avec le formateur
        table = formatter.create_metrics_table(metrics)
        self.assertIsNotNone(table)

def run_tests():
    """Lance tous les tests."""
    # Créer une suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajouter les classes de tests
    test_classes = [
        TestCLIConfig,
        TestConfigProfiles,
        TestConfigValidator,
        TestCLIUtils,
        TestRichFormatter,
        TestASCIIGraphs,
        TestPathValidator,
        TestModelValidator,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Lancer les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
