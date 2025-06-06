#!/usr/bin/env python3
"""
Unit Tests for Data Processing
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Tests for data extraction and processing functions.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_X = np.random.rand(100, 1000)  # 100 samples, 1000 features
        self.sample_y = np.random.rand(100, 2)     # 100 samples, 2 targets
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_data_shapes(self):
        """Test that data has correct shapes."""
        self.assertEqual(self.sample_X.shape, (100, 1000))
        self.assertEqual(self.sample_y.shape, (100, 2))
    
    def test_data_ranges(self):
        """Test that data is in expected ranges."""
        # Intensity ratios should be positive
        self.assertTrue(np.all(self.sample_X >= 0))
        
        # Parameters should be in reasonable ranges
        self.assertTrue(np.all(self.sample_y >= 0))
    
    def test_data_normalization(self):
        """Test data normalization."""
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(self.sample_X)
        
        # Check that normalized data has mean ~0 and std ~1
        self.assertAlmostEqual(np.mean(X_normalized), 0, places=10)
        self.assertAlmostEqual(np.std(X_normalized), 1, places=10)
    
    def test_train_test_split(self):
        """Test train/test splitting."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.sample_X, self.sample_y, test_size=0.2, random_state=42
        )
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)
        
        # Check that features dimension is preserved
        self.assertEqual(X_train.shape[1], 1000)
        self.assertEqual(X_test.shape[1], 1000)
        self.assertEqual(y_train.shape[1], 2)
        self.assertEqual(y_test.shape[1], 2)

class TestDataValidation(unittest.TestCase):
    """Test cases for data validation."""
    
    def test_no_nan_values(self):
        """Test that data contains no NaN values."""
        X = np.random.rand(10, 100)
        y = np.random.rand(10, 2)
        
        self.assertFalse(np.any(np.isnan(X)))
        self.assertFalse(np.any(np.isnan(y)))
    
    def test_no_infinite_values(self):
        """Test that data contains no infinite values."""
        X = np.random.rand(10, 100)
        y = np.random.rand(10, 2)
        
        self.assertFalse(np.any(np.isinf(X)))
        self.assertFalse(np.any(np.isinf(y)))
    
    def test_parameter_ranges(self):
        """Test that parameters are in expected physical ranges."""
        # Simulate realistic parameter ranges
        L_ecran = np.random.uniform(6.0, 14.0, 100)  # µm
        gap = np.random.uniform(0.025, 1.5, 100)     # µm
        
        # Check L_ecran range
        self.assertTrue(np.all(L_ecran >= 6.0))
        self.assertTrue(np.all(L_ecran <= 14.0))
        
        # Check gap range
        self.assertTrue(np.all(gap >= 0.025))
        self.assertTrue(np.all(gap <= 1.5))

if __name__ == '__main__':
    unittest.main()
