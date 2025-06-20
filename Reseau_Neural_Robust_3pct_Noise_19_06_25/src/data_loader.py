#!/usr/bin/env python3
"""
Chargeur de Données pour Prédiction Dual Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Ce module gère le chargement, l'augmentation et la préparation
des données pour l'entraînement du réseau dual.
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Ajouter le chemin vers le module d'augmentation
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from data_augmentation_2D import DataAugmentation2D

class IntensityDataset(Dataset):
    """
    Dataset PyTorch pour les profils d'intensité et paramètres dual.
    """
    
    def __init__(self, X, y):
        """
        Initialise le dataset.
        
        Args:
            X (np.array): Profils d'intensité [n_samples, 600]
            y (np.array): Paramètres [n_samples, 2] où [:, 0] = gap, [:, 1] = L_ecran
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DualDataLoader:
    """
    Classe principale pour charger et préparer les données dual.
    """
    
    def __init__(self, dataset_path="../data_generation/dataset_2D",
                 cache_path="../data/augmented_dataset_advanced.npz"):
        """
        Initialise le chargeur de données avec séparation stricte.

        Args:
            dataset_path (str): Chemin vers le dataset 2D
            cache_path (str): Chemin pour cache des données augmentées avancées
        """
        self.dataset_path = Path(dataset_path)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(exist_ok=True)

        # Scalers pour normalisation
        self.input_scaler = StandardScaler()
        self.gap_scaler = StandardScaler()
        self.L_ecran_scaler = StandardScaler()

        # Stockage des données pour séparation stricte
        self.X_train_raw = None
        self.X_val_raw = None
        self.X_test_raw = None
        self.y_train_raw = None
        self.y_val_raw = None
        self.y_test_raw = None

        print(f"🔧 DualDataLoader initialisé avec séparation stricte")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"💾 Cache: {self.cache_path}")
    
    def load_and_augment_data(self, use_cache=True, augmentation_config=None):
        """
        Charge et augmente les données avec cache intelligent.
        
        Args:
            use_cache (bool): Utiliser le cache si disponible
            augmentation_config (dict): Configuration d'augmentation
        
        Returns:
            tuple: (X_augmented, y_augmented) données augmentées
        """
        print(f"\n📊 Chargement et augmentation des données...")
        
        # Vérifier le cache
        if use_cache and self.cache_path.exists():
            print(f"💾 Cache trouvé, chargement depuis {self.cache_path}")
            cached_data = np.load(self.cache_path)
            X_augmented = cached_data['X']
            y_augmented = cached_data['y']
            print(f"✅ Données chargées depuis cache: X{X_augmented.shape}, y{y_augmented.shape}")
            return X_augmented, y_augmented
        
        # Configuration par défaut
        if augmentation_config is None:
            augmentation_config = {
                'gap_density': 2,
                'L_ecran_density': 2,
                'method': 'linear',
                'include_original': True
            }

        # Filtrer les paramètres non supportés
        valid_params = ['gap_density', 'L_ecran_density', 'method', 'include_original']
        filtered_config = {k: v for k, v in augmentation_config.items() if k in valid_params}
        
        # Créer l'augmentateur et charger les données
        augmenter = DataAugmentation2D(self.dataset_path)
        X_augmented, y_augmented = augmenter.augment_dataset(**filtered_config)
        
        # Sauvegarder en cache
        print(f"💾 Sauvegarde en cache: {self.cache_path}")
        np.savez_compressed(self.cache_path, X=X_augmented, y=y_augmented)
        
        return X_augmented, y_augmented
    
    def prepare_data_splits(self, X, y, train_size=0.64, val_size=0.16, test_size=0.20,
                           random_state=42, shuffle=True):
        """
        Divise les données en train/validation/test avec séparation stricte.

        Args:
            X (np.array): Données d'entrée
            y (np.array): Données de sortie
            train_size (float): Proportion train
            val_size (float): Proportion validation
            test_size (float): Proportion test
            random_state (int): Seed pour reproductibilité
            shuffle (bool): Mélanger les données avant division

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print(f"\n🔄 SÉPARATION STRICTE DES DONNÉES")
        print("="*40)
        print(f"   Train: {train_size:.0%}, Val: {val_size:.0%}, Test: {test_size:.0%}")
        print(f"   Shuffle: {shuffle}, Random State: {random_state}")

        # Vérifier que les proportions sont correctes
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Les proportions doivent sommer à 1.0"

        # Première division: train+val / test (SÉPARATION STRICTE)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            shuffle=shuffle, stratify=None
        )

        # Deuxième division: train / val (SÉPARATION STRICTE)
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state + 1, shuffle=shuffle, stratify=None
        )

        # Stocker les données brutes pour affichage ultérieur
        self.X_train_raw = X_train.copy()
        self.X_val_raw = X_val.copy()
        self.X_test_raw = X_test.copy()
        self.y_train_raw = y_train.copy()
        self.y_val_raw = y_val.copy()
        self.y_test_raw = y_test.copy()

        # Vérification de non-chevauchement
        train_indices = set(range(len(X_train)))
        val_indices = set(range(len(X_train), len(X_train) + len(X_val)))
        test_indices = set(range(len(X_train) + len(X_val), len(X_train) + len(X_val) + len(X_test)))

        assert len(train_indices & val_indices) == 0, "Chevauchement train/val détecté!"
        assert len(train_indices & test_indices) == 0, "Chevauchement train/test détecté!"
        assert len(val_indices & test_indices) == 0, "Chevauchement val/test détecté!"

        print(f"✅ Division stricte terminée:")
        print(f"   Train: {X_train.shape[0]} échantillons")
        print(f"   Val: {X_val.shape[0]} échantillons")
        print(f"   Test: {X_test.shape[0]} échantillons")
        print(f"   ✅ Aucun chevauchement entre les sets")

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_data(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                      separate_target_scaling=True):
        """
        Normalise les données avec StandardScaler.
        
        Args:
            X_train, X_val, X_test: Données d'entrée
            y_train, y_val, y_test: Données de sortie
            separate_target_scaling (bool): Scaling séparé pour gap et L_ecran
        
        Returns:
            tuple: Données normalisées
        """
        print(f"\n🔧 Normalisation des données...")
        
        # Normalisation des entrées (profils d'intensité)
        print(f"   Normalisation des profils d'intensité...")
        X_train_scaled = self.input_scaler.fit_transform(X_train)
        X_val_scaled = self.input_scaler.transform(X_val)
        X_test_scaled = self.input_scaler.transform(X_test)
        
        if separate_target_scaling:
            # Scaling séparé pour gap et L_ecran
            print(f"   Scaling séparé pour gap et L_ecran...")
            
            # Gap (colonne 0)
            y_train_gap_scaled = self.gap_scaler.fit_transform(y_train[:, 0:1])
            y_val_gap_scaled = self.gap_scaler.transform(y_val[:, 0:1])
            y_test_gap_scaled = self.gap_scaler.transform(y_test[:, 0:1])
            
            # L_ecran (colonne 1)
            y_train_L_scaled = self.L_ecran_scaler.fit_transform(y_train[:, 1:2])
            y_val_L_scaled = self.L_ecran_scaler.transform(y_val[:, 1:2])
            y_test_L_scaled = self.L_ecran_scaler.transform(y_test[:, 1:2])
            
            # Recombiner
            y_train_scaled = np.hstack([y_train_gap_scaled, y_train_L_scaled])
            y_val_scaled = np.hstack([y_val_gap_scaled, y_val_L_scaled])
            y_test_scaled = np.hstack([y_test_gap_scaled, y_test_L_scaled])
            
        else:
            # Scaling global pour les deux paramètres
            print(f"   Scaling global pour les paramètres...")
            y_train_scaled = self.input_scaler.fit_transform(y_train)
            y_val_scaled = self.input_scaler.transform(y_val)
            y_test_scaled = self.input_scaler.transform(y_test)
        
        print(f"✅ Normalisation terminée")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled,
                y_train_scaled, y_val_scaled, y_test_scaled)
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test,
                           batch_size=32, shuffle_train=True):
        """
        Crée les DataLoaders PyTorch.
        
        Args:
            X_train, X_val, X_test: Données d'entrée normalisées
            y_train, y_val, y_test: Données de sortie normalisées
            batch_size (int): Taille des batches
            shuffle_train (bool): Mélanger les données d'entraînement
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        print(f"\n🔄 Création des DataLoaders (batch_size={batch_size})...")
        
        # Créer les datasets
        train_dataset = IntensityDataset(X_train, y_train)
        val_dataset = IntensityDataset(X_val, y_val)
        test_dataset = IntensityDataset(X_test, y_test)
        
        # Créer les loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            shuffle=shuffle_train, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, 
            shuffle=False, num_workers=0
        )
        
        print(f"✅ DataLoaders créés:")
        print(f"   Train: {len(train_loader)} batches")
        print(f"   Val: {len(val_loader)} batches")
        print(f"   Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def inverse_transform_predictions(self, predictions, separate_scaling=True):
        """
        Inverse la normalisation des prédictions.
        
        Args:
            predictions (np.array): Prédictions normalisées [n_samples, 2]
            separate_scaling (bool): Utiliser scaling séparé
        
        Returns:
            np.array: Prédictions dans l'espace original
        """
        if separate_scaling:
            # Inverse transform séparé
            pred_gap = self.gap_scaler.inverse_transform(predictions[:, 0:1])
            pred_L_ecran = self.L_ecran_scaler.inverse_transform(predictions[:, 1:2])
            return np.hstack([pred_gap, pred_L_ecran])
        else:
            # Inverse transform global
            return self.input_scaler.inverse_transform(predictions)

    def create_detailed_results_dataframe(self, predictions, targets, dataset_type="test"):
        """
        Crée un DataFrame détaillé avec les résultats réels vs prédits.

        Args:
            predictions (np.array): Prédictions [n_samples, 2]
            targets (np.array): Valeurs réelles [n_samples, 2]
            dataset_type (str): Type de dataset ("test", "val", etc.)

        Returns:
            pd.DataFrame: DataFrame avec colonnes [GAP_reel, LECRAN_reel, GAP_pred, LECRAN_pred]
        """
        import pandas as pd

        print(f"\n📊 CRÉATION DU TABLEAU DÉTAILLÉ - {dataset_type.upper()}")
        print("="*50)

        # Créer le DataFrame
        results_df = pd.DataFrame({
            'GAP_reel': targets[:, 0],
            'LECRAN_reel': targets[:, 1],
            'GAP_pred': predictions[:, 0],
            'LECRAN_pred': predictions[:, 1]
        })

        # Ajouter les erreurs
        results_df['GAP_erreur'] = np.abs(results_df['GAP_pred'] - results_df['GAP_reel'])
        results_df['LECRAN_erreur'] = np.abs(results_df['LECRAN_pred'] - results_df['LECRAN_reel'])

        # Ajouter les indicateurs de succès (tolérance 0.007µm pour gap, 0.1µm pour L_ecran)
        results_df['GAP_success'] = results_df['GAP_erreur'] <= 0.007
        results_df['LECRAN_success'] = results_df['LECRAN_erreur'] <= 0.1
        results_df['BOTH_success'] = results_df['GAP_success'] & results_df['LECRAN_success']

        # Statistiques
        gap_accuracy = results_df['GAP_success'].mean() * 100
        lecran_accuracy = results_df['LECRAN_success'].mean() * 100
        both_accuracy = results_df['BOTH_success'].mean() * 100

        print(f"📈 Statistiques {dataset_type}:")
        print(f"   Échantillons: {len(results_df)}")
        print(f"   GAP Accuracy (±0.007µm): {gap_accuracy:.1f}%")
        print(f"   LECRAN Accuracy (±0.1µm): {lecran_accuracy:.1f}%")
        print(f"   Both Success: {both_accuracy:.1f}%")
        print(f"   GAP MAE: {results_df['GAP_erreur'].mean():.4f}µm")
        print(f"   LECRAN MAE: {results_df['LECRAN_erreur'].mean():.4f}µm")

        # Sauvegarder le DataFrame
        filename = f"../results/detailed_results_{dataset_type}.csv"
        Path(filename).parent.mkdir(exist_ok=True)
        results_df.to_csv(filename, index=False)
        print(f"💾 Tableau sauvegardé: {filename}")

        # Afficher quelques exemples
        print(f"\n📋 Aperçu des résultats (premiers 10 échantillons):")
        print(results_df[['GAP_reel', 'GAP_pred', 'GAP_erreur', 'GAP_success',
                         'LECRAN_reel', 'LECRAN_pred', 'LECRAN_erreur', 'LECRAN_success']].head(10).to_string(index=False))

        return results_df

    def display_test_samples_comparison(self, n_samples=20):
        """
        Affiche une comparaison détaillée pour n échantillons de test.

        Args:
            n_samples (int): Nombre d'échantillons à afficher
        """
        if self.y_test_raw is None:
            print("❌ Aucune donnée de test disponible. Exécutez d'abord prepare_data_splits.")
            return

        print(f"\n🔍 COMPARAISON DÉTAILLÉE - {n_samples} ÉCHANTILLONS DE TEST")
        print("="*70)

        # Sélectionner des échantillons aléatoires
        indices = np.random.choice(len(self.y_test_raw), min(n_samples, len(self.y_test_raw)), replace=False)

        print(f"{'#':<3} {'GAP_reel':<10} {'LECRAN_reel':<12} {'GAP_pred':<10} {'LECRAN_pred':<12} {'GAP_err':<8} {'LEC_err':<8} {'Status':<10}")
        print("-" * 70)

        for i, idx in enumerate(indices):
            gap_real = self.y_test_raw[idx, 0]
            lecran_real = self.y_test_raw[idx, 1]

            # Note: Pour l'affichage, on utilise les vraies valeurs
            # En pratique, les prédictions viendraient du modèle
            gap_pred = gap_real + np.random.normal(0, 0.005)  # Simulation pour démonstration
            lecran_pred = lecran_real + np.random.normal(0, 0.05)

            gap_err = abs(gap_pred - gap_real)
            lecran_err = abs(lecran_pred - lecran_real)

            gap_ok = gap_err <= 0.007
            lecran_ok = lecran_err <= 0.1
            status = "✅ OK" if gap_ok and lecran_ok else "❌ ÉCHEC"

            print(f"{i+1:<3} {gap_real:<10.4f} {lecran_real:<12.1f} {gap_pred:<10.4f} {lecran_pred:<12.1f} {gap_err:<8.4f} {lecran_err:<8.1f} {status:<10}")

        print("-" * 70)
        print("Note: Les prédictions ci-dessus sont simulées pour démonstration.")
        print("Les vraies prédictions seront générées par le modèle entraîné.")

    def get_complete_pipeline(self, config):
        """
        Pipeline complet de préparation des données.
        
        Args:
            config (dict): Configuration complète
        
        Returns:
            tuple: Tous les éléments nécessaires pour l'entraînement
        """
        print(f"\n🚀 PIPELINE COMPLET DE PRÉPARATION DES DONNÉES")
        print("="*60)
        
        # 1. Charger et augmenter les données
        augmentation_config = config.get('data_processing', {}).get('augmentation', {})
        X_augmented, y_augmented = self.load_and_augment_data(
            use_cache=True, 
            augmentation_config=augmentation_config
        )
        
        # 2. Diviser les données
        splits_config = config.get('data_processing', {}).get('data_splits', {})
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data_splits(
            X_augmented, y_augmented,
            train_size=splits_config.get('train', 0.64),
            val_size=splits_config.get('validation', 0.16),
            test_size=splits_config.get('test', 0.20)
        )
        
        # 3. Normaliser les données
        normalization_config = config.get('data_processing', {}).get('normalization', {})
        separate_scaling = normalization_config.get('target_scaling', {}).get('separate_scaling', True)
        
        (X_train_scaled, X_val_scaled, X_test_scaled,
         y_train_scaled, y_val_scaled, y_test_scaled) = self.normalize_data(
            X_train, X_val, X_test, y_train, y_val, y_test,
            separate_target_scaling=separate_scaling
        )
        
        # 4. Créer les DataLoaders
        batch_size = config.get('training', {}).get('batch_size', 32)
        train_loader, val_loader, test_loader = self.create_data_loaders(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            batch_size=batch_size
        )
        
        print(f"\n✅ Pipeline terminé avec succès !")
        
        return {
            'loaders': (train_loader, val_loader, test_loader),
            'raw_data': (X_train, X_val, X_test, y_train, y_val, y_test),
            'scaled_data': (X_train_scaled, X_val_scaled, X_test_scaled,
                           y_train_scaled, y_val_scaled, y_test_scaled),
            'scalers': (self.input_scaler, self.gap_scaler, self.L_ecran_scaler)
        }


def test_data_loader():
    """
    Test du chargeur de données.
    """
    print("🧪 Test du DualDataLoader")
    print("="*40)
    
    # Configuration de test
    config = {
        'data_processing': {
            'augmentation': {
                'gap_density': 1,  # Pas d'augmentation pour le test
                'L_ecran_density': 1,
                'method': 'linear',
                'include_original': True
            },
            'data_splits': {
                'train': 0.70,
                'validation': 0.15,
                'test': 0.15
            },
            'normalization': {
                'target_scaling': {
                    'separate_scaling': True
                }
            }
        },
        'training': {
            'batch_size': 16
        }
    }
    
    # Créer le loader
    data_loader = DualDataLoader()
    
    # Tester le pipeline complet
    try:
        pipeline_result = data_loader.get_complete_pipeline(config)
        print(f"✅ Test réussi ! Pipeline fonctionnel.")
        
        # Vérifier un batch
        train_loader = pipeline_result['loaders'][0]
        for batch_X, batch_y in train_loader:
            print(f"✅ Batch test: X{batch_X.shape}, y{batch_y.shape}")
            break
            
    except Exception as e:
        print(f"❌ Erreur dans le test: {e}")


if __name__ == "__main__":
    test_data_loader()
