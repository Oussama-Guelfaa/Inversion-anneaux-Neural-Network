#!/usr/bin/env python3
"""
Ultra Fast Data Loader utilisant extracted_data_full.npz
Auteur: Oussama GUELFAA
Date: 15/07/2025

Data loader ultra-rapide qui charge directement le fichier .npz
au lieu de 22,541 fichiers .mat individuels.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import time
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class UltraFastDataLoader:
    """
    Data Loader ultra-rapide utilisant le fichier extracted_data_full.npz.
    """
    
    def __init__(self, extracted_file="extracted_data_full.npz"):
        self.extracted_file = extracted_file
        self.input_scaler = None  # PAS DE NORMALISATION !
        self.output_scaler = None  # PAS DE NORMALISATION !

        print(f"⚡ UltraFastDataLoader initialisé (SANS NORMALISATION)")
        print(f"   📄 Fichier source: {extracted_file}")
        print(f"   🎯 Mode: DONNÉES BRUTES (préservation des anneaux)")
    
    def load_data(self, sample_ratio=1.0):
        """
        Charge les données depuis le fichier .npz.
        
        Args:
            sample_ratio (float): Ratio d'échantillonnage (1.0 = tous)
        
        Returns:
            tuple: (X_data, y_data)
        """
        print(f"⚡ Chargement ultra-rapide des données...")
        
        start_time = time.time()
        
        # Charger le fichier .npz
        data = np.load(self.extracted_file)
        
        X_data = data['X_data']  # (22540, 601)
        y_data = data['y_data']  # (22540, 2)
        
        load_time = time.time() - start_time
        
        print(f"✅ Données chargées en {load_time:.3f} secondes !")
        print(f"   📊 Forme X: {X_data.shape}")
        print(f"   📊 Forme y: {y_data.shape}")
        
        # Échantillonnage si demandé
        if sample_ratio < 1.0:
            n_samples = int(len(X_data) * sample_ratio)
            indices = np.random.choice(len(X_data), n_samples, replace=False)
            X_data = X_data[indices]
            y_data = y_data[indices]
            
            print(f"   📊 Échantillonnage: {n_samples} profils ({sample_ratio*100:.1f}%)")
        
        # Statistiques
        print(f"   📈 Gap range: [{y_data[:, 0].min():.6f}, {y_data[:, 0].max():.6f}] µm")
        print(f"   📈 L_écran range: [{y_data[:, 1].min():.1f}, {y_data[:, 1].max():.1f}] µm")
        
        return X_data, y_data
    
    def create_train_val_test_split(self, X, y, test_size=0.2, val_size=0.15, random_state=42):
        """
        Divise les données en train/validation/test.
        
        Args:
            X, y: Données d'entrée et labels
            test_size: Proportion du test set
            val_size: Proportion du validation set
            random_state: Graine aléatoire
        
        Returns:
            dict: Dictionnaire avec les splits
        """
        print("🔄 Division train/validation/test...")
        
        # Première division: train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Deuxième division: train / val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        print(f"   📊 Train: {X_train.shape[0]} échantillons ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"   📊 Validation: {X_val.shape[0]} échantillons ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"   📊 Test: {X_test.shape[0]} échantillons ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return splits
    
    def normalize_data(self, splits):
        """
        AUCUNE NORMALISATION - Utilise les données brutes pour préserver les anneaux.

        Args:
            splits: Données divisées

        Returns:
            dict: Données "normalisées" (en fait, données brutes)
        """
        print("🎯 AUCUNE NORMALISATION - Conservation des données brutes...")
        print("   📊 Préservation de la forme des anneaux holographiques")

        # Pas de normalisation - copie directe des données
        normalized_splits = {
            'X_train_norm': splits['X_train'].copy(),
            'X_val_norm': splits['X_val'].copy(),
            'X_test_norm': splits['X_test'].copy(),
            'y_train_norm': splits['y_train'].copy(),
            'y_val_norm': splits['y_val'].copy(),
            'y_test_norm': splits['y_test'].copy()
        }

        print("✅ Données brutes conservées (pas de normalisation)")

        return normalized_splits
    
    def create_data_loaders(self, normalized_splits, batch_size=32):
        """
        Crée les DataLoaders PyTorch.
        
        Args:
            normalized_splits: Données normalisées
            batch_size: Taille des batches
        
        Returns:
            dict: DataLoaders PyTorch
        """
        print(f"🔄 Création des DataLoaders (batch_size={batch_size})...")
        
        # Créer les datasets PyTorch
        train_dataset = TensorDataset(
            torch.FloatTensor(normalized_splits['X_train_norm']),
            torch.FloatTensor(normalized_splits['y_train_norm'])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(normalized_splits['X_val_norm']),
            torch.FloatTensor(normalized_splits['y_val_norm'])
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(normalized_splits['X_test_norm']),
            torch.FloatTensor(normalized_splits['y_test_norm'])
        )
        
        # Créer les data loaders
        data_loaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }
        
        print("✅ DataLoaders créés")
        
        return data_loaders
    
    def save_scalers(self, filepath="ultra_fast_scalers.joblib"):
        """Sauvegarde les scalers (aucun dans ce cas)."""
        scalers = {
            'input_scaler': None,  # Pas de scaler
            'output_scaler': None,  # Pas de scaler
            'normalization': 'none',  # Indicateur
            'note': 'Données brutes sans normalisation pour préserver les anneaux'
        }
        joblib.dump(scalers, filepath)
        print(f"💾 Configuration 'sans scalers' sauvegardée: {filepath}")

    def load_scalers(self, filepath="ultra_fast_scalers.joblib"):
        """Charge les scalers (aucun dans ce cas)."""
        scalers = joblib.load(filepath)
        self.input_scaler = None
        self.output_scaler = None
        print(f"📂 Configuration 'sans scalers' chargée: {filepath}")
        if 'note' in scalers:
            print(f"   📝 {scalers['note']}")
    
    def get_full_pipeline(self, sample_ratio=1.0, batch_size=32, test_size=0.2, val_size=0.15):
        """
        Pipeline complet de chargement des données.
        
        Returns:
            dict: DataLoaders prêts pour l'entraînement
        """
        print("⚡ Pipeline Ultra-Rapide de Chargement des Données")
        print("=" * 55)
        
        start_time = time.time()
        
        # 1. Charger les données
        X, y = self.load_data(sample_ratio=sample_ratio)
        
        # 2. Diviser
        splits = self.create_train_val_test_split(X, y, test_size, val_size)
        
        # 3. Normaliser
        normalized_splits = self.normalize_data(splits)
        
        # 4. Créer les DataLoaders
        data_loaders = self.create_data_loaders(normalized_splits, batch_size)
        
        # 5. Sauvegarder les scalers
        self.save_scalers()
        
        total_time = time.time() - start_time
        
        print("=" * 55)
        print(f"⚡ Pipeline terminé en {total_time:.3f} secondes !")
        print(f"🚀 Données prêtes pour l'entraînement ultra-rapide !")
        
        return data_loaders, normalized_splits

def main():
    """Test du data loader ultra-rapide."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("⚡ Test du UltraFastDataLoader")
    print("=" * 40)
    
    # Créer le data loader
    loader = UltraFastDataLoader()
    
    # Pipeline complet avec 10% des données pour test
    data_loaders, splits = loader.get_full_pipeline(
        sample_ratio=0.1,  # 10% pour test rapide
        batch_size=32
    )
    
    # Test d'un batch
    print("\n🧪 Test d'un batch d'entraînement:")
    train_loader = data_loaders['train']
    for batch_X, batch_y in train_loader:
        print(f"   📊 Batch X shape: {batch_X.shape}")
        print(f"   📊 Batch y shape: {batch_y.shape}")
        print(f"   📈 X range: [{batch_X.min():.3f}, {batch_X.max():.3f}]")
        print(f"   📈 y range: [{batch_y.min():.3f}, {batch_y.max():.3f}]")
        break
    
    print("\n✅ Test terminé avec succès !")
    print("⚡ Le data loader ultra-rapide est prêt !")

if __name__ == "__main__":
    main()
