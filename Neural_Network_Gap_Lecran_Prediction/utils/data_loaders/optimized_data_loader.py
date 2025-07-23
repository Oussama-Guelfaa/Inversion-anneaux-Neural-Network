#!/usr/bin/env python3
"""
Data Loader Optimisé pour Neural_Network_Gap_Lecran_Prediction

Auteur: Oussama GUELFAA
Date: 15/07/2025

Basé sur Reseau_Neural_2D_Gap_Lecran_25_06_25/src/data_loader.py
Modifié pour utiliser la troncature par indices (200-800) au lieu de truncate_to
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
import glob
import os
from typing import Tuple, Optional, Dict, Any
import time

logger = logging.getLogger(__name__)

class HologramDataset(Dataset):
    """
    Dataset PyTorch pour les profils holographiques avec troncature optimisée.
    """
    
    def __init__(self, X, y):
        """
        Initialise le dataset.
        
        Args:
            X (array): Profils d'intensité tronqués (601 points)
            y (array): Paramètres [gap, L_ecran]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class OptimizedDataLoader:
    """
    Data Loader optimisé avec troncature par indices (200-800).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise le data loader optimisé.

        Args:
            config (dict): Configuration (optionnelle)
        """
        # Configuration par défaut
        self.config = config or {
            'data': {
                'train_dir': 'Train',
                'test_dir': 'Test'
            },
            'preprocessing': {
                'truncate_start': 200,  # Index de début
                'truncate_end': 800,    # Index de fin
                'expected_points': 601  # Points après troncature
            },
            'training': {
                'test_size': 0.2,
                'val_size': 0.15,
                'random_state': 42
            }
        }
        
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
        # Paramètres de troncature
        self.truncate_start = self.config['preprocessing']['truncate_start']
        self.truncate_end = self.config['preprocessing']['truncate_end']
        self.expected_points = self.config['preprocessing']['expected_points']
        
        logger.info(f"🔧 OptimizedDataLoader initialisé")
        logger.info(f"✂️ Troncature: indices {self.truncate_start}-{self.truncate_end} ({self.expected_points} points)")
    
    def load_dataset_from_directory(self, dataset_dir: str, max_files: Optional[int] = None, 
                                   sample_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge un dataset depuis un dossier contenant des fichiers .mat.
        
        Args:
            dataset_dir (str): Chemin vers le dossier des données
            max_files (int): Limite le nombre de fichiers (None = tous)
            sample_ratio (float): Ratio d'échantillonnage (1.0 = 100%)
        
        Returns:
            tuple: (X, y) données chargées et tronquées
        """
        logger.info(f"📊 Chargement dataset: {dataset_dir}")
        
        # Obtenir la liste des fichiers .mat
        mat_files = glob.glob(os.path.join(dataset_dir, "*.mat"))
        mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
        
        logger.info(f"   📁 Fichiers trouvés: {len(mat_files)}")
        
        # Échantillonnage si demandé
        if sample_ratio < 1.0:
            n_sample = int(len(mat_files) * sample_ratio)
            mat_files = np.random.choice(mat_files, n_sample, replace=False).tolist()
            logger.info(f"   📊 Échantillonnage: {n_sample} fichiers ({sample_ratio*100:.1f}%)")
        
        # Limitation si demandée
        if max_files is not None:
            mat_files = mat_files[:max_files]
            logger.info(f"   📊 Limitation: {len(mat_files)} fichiers maximum")
        
        # Pré-allouer les arrays pour optimiser les performances
        X_data = np.zeros((len(mat_files), self.expected_points), dtype=np.float32)
        y_data = np.zeros((len(mat_files), 2), dtype=np.float32)
        valid_count = 0
        
        start_time = time.time()
        
        for i, mat_file in enumerate(mat_files):
            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(mat_files) - i) / rate
                logger.info(f"   📈 Progression: {i}/{len(mat_files)} - {rate:.1f} fichiers/s - Reste: {remaining/60:.1f}min")
            
            try:
                # Charger le fichier .mat
                data = loadmat(mat_file)
                
                # Extraire SEULEMENT le profil d'intensité (ratio)
                if 'ratio' not in data:
                    logger.warning(f"   ⚠️ Pas de 'ratio' dans {os.path.basename(mat_file)}")
                    continue
                
                ratio = data['ratio'].flatten()
                
                # Vérifier la taille avant troncature
                if len(ratio) <= self.truncate_end:
                    logger.warning(f"   ⚠️ Profil trop court {os.path.basename(mat_file)}: {len(ratio)} points")
                    continue
                
                # Appliquer la troncature par indices (200-800)
                ratio_truncated = ratio[self.truncate_start:self.truncate_end+1]
                
                # Vérifier la taille après troncature
                if len(ratio_truncated) != self.expected_points:
                    logger.warning(f"   ⚠️ Taille incorrecte après troncature {os.path.basename(mat_file)}: {len(ratio_truncated)}")
                    continue
                
                # Extraire les labels
                gap = None
                L_ecran = None
                
                if 'gap' in data:
                    gap = float(data['gap'][0, 0])
                elif 'gap_um' in data:
                    gap = float(data['gap_um'][0, 0])
                
                if 'L_ecran_subs' in data:
                    L_ecran = float(data['L_ecran_subs'][0, 0])
                elif 'L_um' in data:
                    L_ecran = float(data['L_um'][0, 0])
                
                if gap is None or L_ecran is None:
                    logger.warning(f"   ⚠️ Labels manquants dans {os.path.basename(mat_file)}")
                    continue
                
                # Stocker dans les arrays pré-alloués
                X_data[valid_count] = ratio_truncated
                y_data[valid_count] = [gap, L_ecran]
                valid_count += 1
                
            except Exception as e:
                logger.error(f"   ❌ Erreur avec {os.path.basename(mat_file)}: {e}")
                continue
        
        # Tronquer aux données valides
        X_data = X_data[:valid_count]
        y_data = y_data[:valid_count]
        
        total_time = time.time() - start_time
        logger.info(f"✅ Chargement terminé:")
        logger.info(f"   📊 Données valides: {valid_count}/{len(mat_files)} profils")
        logger.info(f"   📏 Forme X: {X_data.shape}")
        logger.info(f"   📏 Forme y: {y_data.shape}")
        logger.info(f"   ⏱️ Temps: {total_time/60:.1f} minutes")
        logger.info(f"   🚀 Vitesse: {len(mat_files)/total_time:.1f} fichiers/seconde")
        
        return X_data, y_data
    
    def create_train_val_test_split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Divise les données en train/validation/test.
        
        Args:
            X (array): Données d'entrée
            y (array): Labels
        
        Returns:
            dict: Dictionnaire avec les splits
        """
        logger.info("🔄 Division train/validation/test...")
        
        test_size = self.config['training']['test_size']
        val_size = self.config['training']['val_size']
        random_state = self.config['training']['random_state']
        
        # Première division: train+val / test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Deuxième division: train / val
        val_size_adjusted = val_size / (1 - test_size)  # Ajuster la taille de validation
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
        
        logger.info(f"   📊 Train: {X_train.shape[0]} échantillons ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"   📊 Validation: {X_val.shape[0]} échantillons ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"   📊 Test: {X_test.shape[0]} échantillons ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return splits
    
    def normalize_data(self, splits: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalise les données avec StandardScaler.
        
        Args:
            splits (dict): Données divisées
        
        Returns:
            dict: Données normalisées
        """
        logger.info("🔄 Normalisation des données...")
        
        # Normaliser les entrées (X)
        X_train_norm = self.input_scaler.fit_transform(splits['X_train'])
        X_val_norm = self.input_scaler.transform(splits['X_val'])
        X_test_norm = self.input_scaler.transform(splits['X_test'])
        
        # Normaliser les sorties (y)
        y_train_norm = self.output_scaler.fit_transform(splits['y_train'])
        y_val_norm = self.output_scaler.transform(splits['y_val'])
        y_test_norm = self.output_scaler.transform(splits['y_test'])
        
        normalized_splits = {
            'X_train_norm': X_train_norm,
            'X_val_norm': X_val_norm,
            'X_test_norm': X_test_norm,
            'y_train_norm': y_train_norm,
            'y_val_norm': y_val_norm,
            'y_test_norm': y_test_norm
        }
        
        logger.info("✅ Normalisation terminée")
        
        return normalized_splits
    
    def create_data_loaders(self, normalized_splits: Dict[str, np.ndarray], 
                           batch_size: int = 32) -> Dict[str, DataLoader]:
        """
        Crée les DataLoaders PyTorch.
        
        Args:
            normalized_splits (dict): Données normalisées
            batch_size (int): Taille des batches
        
        Returns:
            dict: DataLoaders PyTorch
        """
        logger.info(f"🔄 Création des DataLoaders (batch_size={batch_size})...")
        
        # Créer les datasets
        train_dataset = HologramDataset(
            normalized_splits['X_train_norm'], 
            normalized_splits['y_train_norm']
        )
        val_dataset = HologramDataset(
            normalized_splits['X_val_norm'], 
            normalized_splits['y_val_norm']
        )
        test_dataset = HologramDataset(
            normalized_splits['X_test_norm'], 
            normalized_splits['y_test_norm']
        )
        
        # Créer les data loaders
        data_loaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }
        
        logger.info("✅ DataLoaders créés")
        
        return data_loaders
    
    def save_scalers(self, filepath: str = "scalers.joblib"):
        """Sauvegarde les scalers pour réutilisation."""
        scalers = {
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler
        }
        joblib.dump(scalers, filepath)
        logger.info(f"💾 Scalers sauvegardés: {filepath}")
    
    def load_scalers(self, filepath: str = "scalers.joblib"):
        """Charge les scalers sauvegardés."""
        scalers = joblib.load(filepath)
        self.input_scaler = scalers['input_scaler']
        self.output_scaler = scalers['output_scaler']
        logger.info(f"📂 Scalers chargés: {filepath}")

def main():
    """Fonction de test du data loader optimisé."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("🧠 Test du OptimizedDataLoader")
    print("=" * 50)
    
    # Configuration
    config = {
        'data': {
            'train_dir': 'Train',
            'test_dir': 'Test'
        },
        'preprocessing': {
            'truncate_start': 200,
            'truncate_end': 800,
            'expected_points': 601
        },
        'training': {
            'test_size': 0.2,
            'val_size': 0.15,
            'random_state': 42
        }
    }
    
    # Créer le data loader
    data_loader = OptimizedDataLoader(config)
    
    # Charger un échantillon des données (10% pour test rapide)
    X, y = data_loader.load_dataset_from_directory('Train', sample_ratio=0.1)
    
    # Créer les splits
    splits = data_loader.create_train_val_test_split(X, y)
    
    # Normaliser
    normalized_splits = data_loader.normalize_data(splits)
    
    # Créer les DataLoaders
    data_loaders = data_loader.create_data_loaders(normalized_splits, batch_size=16)
    
    # Sauvegarder les scalers
    data_loader.save_scalers("optimized_scalers.joblib")
    
    print("✅ Test terminé avec succès !")

if __name__ == "__main__":
    main()
