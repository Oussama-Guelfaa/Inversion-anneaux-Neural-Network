#!/usr/bin/env python3
"""
Data Loader pour Réseau Neural 2D Gap + L_écran

Auteur: Oussama GUELFAA
Date: 25/06/2025

Ce module gère le chargement et la préparation des données pour l'entraînement
du réseau de neurones, incluant la normalisation et la division train/validation.
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

logger = logging.getLogger(__name__)

class HologramDataset(Dataset):
    """
    Dataset PyTorch pour les profils holographiques 2D.
    """
    
    def __init__(self, X, y):
        """
        Initialise le dataset.
        
        Args:
            X (array): Profils d'intensité
            y (array): Paramètres [gap, L_ecran]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataLoader2D:
    """
    Classe pour charger et préparer les données 2D.
    """
    
    def __init__(self, config):
        """
        Initialise le data loader.

        Args:
            config (dict): Configuration du modèle
        """
        self.config = config
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        # Déterminer le répertoire racine du projet
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # Remonter de src/ vers la racine

        # Chemins des données (absolus)
        self.train_path = project_root / config['data']['train_dataset']
        self.test_path = project_root / config['data']['test_dataset']
        self.augmented_path = project_root / config['data']['augmented_dataset']
        
        logger.info(f"🔧 Initialisation DataLoader2D")
        logger.info(f"📁 Train: {self.train_path}")
        logger.info(f"📁 Test: {self.test_path}")
        logger.info(f"📁 Augmented: {self.augmented_path}")
    
    def load_dataset_from_path(self, dataset_path, max_files=None):
        """
        Charge un dataset depuis un dossier contenant .mat et labels.csv.
        
        Args:
            dataset_path (Path): Chemin vers le dataset
            max_files (int): Limite le nombre de fichiers
        
        Returns:
            tuple: (X, y) données chargées
        """
        logger.info(f"📊 Chargement dataset: {dataset_path}")
        
        # Charger le fichier labels.csv
        labels_path = dataset_path / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Fichier labels.csv non trouvé: {labels_path}")
        
        labels_df = pd.read_csv(labels_path)
        logger.info(f"   Labels trouvés: {len(labels_df)} entrées")
        
        if max_files:
            labels_df = labels_df.head(max_files)
            logger.info(f"   Limité à {max_files} fichiers")
        
        X_data = []
        y_data = []
        truncate_to = self.config['augmentation']['truncate_to']
        
        for i, (_, row) in enumerate(labels_df.iterrows()):
            if i % 1000 == 0:
                logger.info(f"   Progression: {i}/{len(labels_df)} fichiers...")
            
            filename = row['filename']
            gap = row['gap_um']
            L_ecran = row['L_um']
            
            mat_file_path = dataset_path / filename
            
            try:
                # Charger le fichier .mat
                data = loadmat(str(mat_file_path))
                
                # Extraire le profil d'intensité
                if 'ratio' in data:
                    ratio = data['ratio'].flatten()
                elif 'I_ratio' in data:
                    ratio = data['I_ratio'].flatten()
                else:
                    # Chercher d'autres clés possibles
                    possible_keys = [k for k in data.keys() if not k.startswith('__')]
                    if possible_keys:
                        ratio = data[possible_keys[0]].flatten()
                    else:
                        logger.warning(f"   ⚠️  Aucune donnée trouvée dans {filename}")
                        continue
                
                # Tronquer si nécessaire
                if len(ratio) > truncate_to:
                    ratio = ratio[:truncate_to]
                elif len(ratio) < truncate_to:
                    # Padding avec la dernière valeur si trop court
                    ratio = np.pad(ratio, (0, truncate_to - len(ratio)), 'edge')
                
                # Stocker
                X_data.append(ratio)
                y_data.append([gap, L_ecran])
                
            except Exception as e:
                logger.warning(f"   ⚠️  Erreur avec {filename}: {e}")
                continue
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        logger.info(f"✅ Dataset chargé: X{X.shape}, y{y.shape}")
        logger.info(f"   Gap range: {np.min(y[:, 0]):.4f} - {np.max(y[:, 0]):.4f} µm")
        logger.info(f"   L_ecran range: {np.min(y[:, 1]):.1f} - {np.max(y[:, 1]):.1f} µm")
        
        return X, y
    
    def load_training_data(self, use_augmented=True):
        """
        Charge les données d'entraînement (originales ou augmentées).
        
        Args:
            use_augmented (bool): Utiliser le dataset augmenté
        
        Returns:
            tuple: (X_train, y_train) données d'entraînement
        """
        if use_augmented and self.augmented_path.exists():
            logger.info("🚀 Utilisation du dataset augmenté")
            return self.load_dataset_from_path(self.augmented_path)
        else:
            logger.info("📊 Utilisation du dataset original")
            return self.load_dataset_from_path(self.train_path)
    
    def load_test_data(self):
        """
        Charge les données de test.
        
        Returns:
            tuple: (X_test, y_test) données de test
        """
        logger.info("🧪 Chargement des données de test")
        return self.load_dataset_from_path(self.test_path)
    
    def prepare_data(self, use_augmented=True, validation_split=0.2):
        """
        Prépare les données pour l'entraînement (normalisation + division).
        
        Args:
            use_augmented (bool): Utiliser le dataset augmenté
            validation_split (float): Proportion pour la validation
        
        Returns:
            tuple: (train_loader, val_loader, test_loader, scalers)
        """
        logger.info(f"\n🔄 Préparation des données")
        logger.info("="*30)
        
        # 1. Charger les données
        X_train_full, y_train_full = self.load_training_data(use_augmented)
        X_test, y_test = self.load_test_data()
        
        # 2. Division train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=validation_split,
            random_state=42,
            stratify=None  # Pas de stratification pour la régression
        )
        
        logger.info(f"📊 Division des données:")
        logger.info(f"   Train: {X_train.shape[0]} échantillons")
        logger.info(f"   Validation: {X_val.shape[0]} échantillons")
        logger.info(f"   Test: {X_test.shape[0]} échantillons")
        
        # 3. Normalisation des entrées
        logger.info(f"🔧 Normalisation des données...")
        X_train_scaled = self.input_scaler.fit_transform(X_train)
        X_val_scaled = self.input_scaler.transform(X_val)
        X_test_scaled = self.input_scaler.transform(X_test)
        
        # 4. Normalisation des sorties
        y_train_scaled = self.output_scaler.fit_transform(y_train)
        y_val_scaled = self.output_scaler.transform(y_val)
        y_test_scaled = self.output_scaler.transform(y_test)
        
        # 5. Créer les datasets PyTorch
        train_dataset = HologramDataset(X_train_scaled, y_train_scaled)
        val_dataset = HologramDataset(X_val_scaled, y_val_scaled)
        test_dataset = HologramDataset(X_test_scaled, y_test_scaled)
        
        # 6. Créer les data loaders
        batch_size = self.config['training']['batch_size']
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # 7. Sauvegarder les scalers
        self.save_scalers()
        
        logger.info(f"✅ Données préparées avec succès")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Scalers sauvegardés")
        
        scalers = {
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler
        }
        
        return train_loader, val_loader, test_loader, scalers
    
    def save_scalers(self):
        """
        Sauvegarde les scalers pour utilisation ultérieure.
        """
        # Utiliser le chemin relatif au projet
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        models_dir = project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        input_scaler_path = models_dir / "input_scaler.pkl"
        output_scaler_path = models_dir / "output_scaler.pkl"
        
        joblib.dump(self.input_scaler, input_scaler_path)
        joblib.dump(self.output_scaler, output_scaler_path)
        
        logger.info(f"💾 Scalers sauvegardés:")
        logger.info(f"   Input: {input_scaler_path}")
        logger.info(f"   Output: {output_scaler_path}")
    
    def load_scalers(self):
        """
        Charge les scalers sauvegardés.
        """
        # Utiliser le chemin relatif au projet
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        models_dir = project_root / "models"
        
        input_scaler_path = models_dir / "input_scaler.pkl"
        output_scaler_path = models_dir / "output_scaler.pkl"
        
        if input_scaler_path.exists() and output_scaler_path.exists():
            self.input_scaler = joblib.load(input_scaler_path)
            self.output_scaler = joblib.load(output_scaler_path)
            logger.info(f"✅ Scalers chargés depuis {models_dir}")
        else:
            logger.warning(f"⚠️  Scalers non trouvés dans {models_dir}")
    
    def get_data_statistics(self):
        """
        Retourne les statistiques des données chargées.
        
        Returns:
            dict: Statistiques des données
        """
        # Charger les données pour les statistiques
        X_train, y_train = self.load_training_data(use_augmented=False)
        X_test, y_test = self.load_test_data()
        
        stats = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'input_features': X_train.shape[1],
            'output_features': y_train.shape[1],
            'gap_range_train': (np.min(y_train[:, 0]), np.max(y_train[:, 0])),
            'gap_range_test': (np.min(y_test[:, 0]), np.max(y_test[:, 0])),
            'L_ecran_range_train': (np.min(y_train[:, 1]), np.max(y_train[:, 1])),
            'L_ecran_range_test': (np.min(y_test[:, 1]), np.max(y_test[:, 1]))
        }
        
        return stats
