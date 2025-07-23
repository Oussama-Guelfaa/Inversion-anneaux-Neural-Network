#!/usr/bin/env python3
"""
Chargeur de données avancé pour Neural_Network_Gap_Lecran_Prediction
Auteur: Oussama GUELFAA
Date: 15/07/2025

Ce module implémente le chargement et la division des données d'entraînement
avec support pour l'augmentation de données et la constitution des jeux
train/validation/test selon les spécifications du projet.
"""

import os
import numpy as np
import scipy.io
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import random
from typing import Tuple, Dict, List, Optional
import joblib

class AdvancedDataLoader:
    """
    Chargeur de données avancé avec support pour:
    - Division train/validation/test (70/15/15%)
    - Data augmentation avec interp2D
    - Normalisation et prétraitement
    - Sauvegarde/chargement des datasets
    """
    
    def __init__(self, train_dir: str, preprocessed_data_path: str = None):
        """
        Initialise le chargeur de données
        
        Args:
            train_dir: Chemin vers le dossier Train/
            preprocessed_data_path: Chemin vers les données prétraitées (.npz)
        """
        self.train_dir = train_dir
        self.preprocessed_data_path = preprocessed_data_path
        
        # Paramètres de prétraitement (seront chargés depuis preprocessed_data.npz)
        self.r_min = None
        self.r_max = None
        self.delta_r = None
        self.n_points = None
        
        # Données chargées
        self.X_data = None  # Profils d'intensité
        self.y_data = None  # Labels [gap, L_ecran]
        self.filenames = None
        
        # Datasets divisés
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Scalers
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        print("🔧 AdvancedDataLoader initialisé")
    
    def load_preprocessing_params(self):
        """Charge les paramètres de prétraitement depuis preprocessed_data.npz"""
        if self.preprocessed_data_path and os.path.exists(self.preprocessed_data_path):
            data = np.load(self.preprocessed_data_path)
            self.r_min = float(data['r_min'])
            self.r_max = float(data['r_max'])
            self.delta_r = float(data['delta_r'])
            self.n_points = int(data['points_per_profile'])
            
            print(f"📊 Paramètres de prétraitement chargés:")
            print(f"   - Plage radiale: [{self.r_min:.6f}, {self.r_max:.6f}] µm")
            print(f"   - Espacement: {self.delta_r:.6f} µm")
            print(f"   - Points par profil: {self.n_points}")
        else:
            print("⚠️ Fichier de prétraitement non trouvé, utilisation des paramètres par défaut")
            self.r_min = 1.384585
            self.r_max = 5.538338
            self.delta_r = 0.006923
            self.n_points = 601
    
    def load_all_training_data(self, max_files: Optional[int] = None, 
                              sample_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Charge toutes les données d'entraînement depuis le dossier Train/
        
        Args:
            max_files: Nombre maximum de fichiers à charger (None = tous)
            sample_ratio: Ratio d'échantillonnage (0.1 = 10% des fichiers)
            
        Returns:
            Tuple (X_data, y_data, filenames)
        """
        print("📂 Chargement des données d'entraînement...")
        
        # Obtenir la liste des fichiers .mat
        mat_files = glob.glob(os.path.join(self.train_dir, "*.mat"))
        mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
        
        # Échantillonnage si demandé
        if sample_ratio < 1.0:
            n_sample = int(len(mat_files) * sample_ratio)
            mat_files = random.sample(mat_files, n_sample)
            print(f"   📊 Échantillonnage: {n_sample}/{len(glob.glob(os.path.join(self.train_dir, '*.mat')))} fichiers")
        
        # Limitation si demandée
        if max_files is not None:
            mat_files = mat_files[:max_files]
            print(f"   📊 Limitation: {len(mat_files)} fichiers maximum")
        
        print(f"   📁 Chargement de {len(mat_files)} fichiers...")
        
        # Charger les paramètres de prétraitement
        self.load_preprocessing_params()
        
        # Pré-allouer les arrays pour éviter les append (plus rapide)
        X_data = np.zeros((len(mat_files), self.n_points), dtype=np.float32)
        y_data = np.zeros((len(mat_files), 2), dtype=np.float32)
        filenames = []
        valid_count = 0
        
        # Charger chaque fichier
        for i, mat_file in enumerate(mat_files):

            print(f"   📈 Progression: {i}/{len(mat_files)} fichiers chargés")
            
            try:
                # Charger le fichier .mat avec timeout et gestion d'erreur robuste
                data = scipy.io.loadmat(mat_file)
                
                # Extraire SEULEMENT les données d'intensité (ratio)
                ratio = data['ratio'].flatten()

                # Appliquer la troncature (indices 200 à 800) - SEULEMENT sur ratio
                ratio_truncated = ratio[200:801]
                
                # Vérifier la cohérence des données
                if len(ratio_truncated) != self.n_points:
                    print(f"⚠️ Longueur incorrecte pour {os.path.basename(mat_file)}: {len(ratio_truncated)}")
                    continue
                
                # Extraire les labels
                gap = float(data['gap'][0, 0]) if 'gap' in data else None
                L_ecran = float(data['L_ecran_subs'][0, 0]) if 'L_ecran_subs' in data else None
                
                if gap is None or L_ecran is None:
                    print(f"⚠️ Labels manquants pour {os.path.basename(mat_file)}")
                    continue
                
                # Ajouter aux arrays pré-alloués (plus rapide)
                X_data[valid_count] = ratio_truncated
                y_data[valid_count] = [gap, L_ecran]
                filenames.append(os.path.basename(mat_file))
                valid_count += 1
                
            except Exception as e:
                print(f"❌ ERREUR fichier {i}: {os.path.basename(mat_file)}: {e}")
                print(f"   🔄 Passage au fichier suivant...")
                continue
        
        # Tronquer aux données valides seulement
        X_data = X_data[:valid_count]
        y_data = y_data[:valid_count]
        
        print(f"✅ Chargement terminé:")
        print(f"   📊 Données chargées: {X_data.shape[0]} profils")
        print(f"   📏 Forme X: {X_data.shape}")
        print(f"   📏 Forme y: {y_data.shape}")
        print(f"   📈 Plage Gap: [{y_data[:, 0].min():.6f}, {y_data[:, 0].max():.6f}] µm")
        print(f"   📈 Plage L_écran: [{y_data[:, 1].min():.3f}, {y_data[:, 1].max():.3f}] µm")
        
        # Stocker les données
        self.X_data = X_data
        self.y_data = y_data
        self.filenames = filenames
        
        return X_data, y_data, filenames
    
    def create_train_val_test_split(self, test_size: float = 0.15, val_size: float = 0.15, 
                                   random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Divise les données en train/validation/test (70/15/15% par défaut)
        
        Args:
            test_size: Proportion du jeu de test
            val_size: Proportion du jeu de validation
            random_state: Graine aléatoire pour reproductibilité
            
        Returns:
            Dictionnaire contenant les datasets divisés
        """
        if self.X_data is None or self.y_data is None:
            raise ValueError("Données non chargées. Appelez load_all_training_data() d'abord.")
        
        print(f"🔀 Division des données: Train {1-test_size-val_size:.0%} / Val {val_size:.0%} / Test {test_size:.0%}")
        
        # Première division: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X_data, self.y_data, 
            test_size=test_size, 
            random_state=random_state,
            stratify=None  # Pas de stratification pour la régression
        )
        
        # Deuxième division: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Ajuster la taille de validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state
        )
        
        # Stocker les datasets
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"✅ Division terminée:")
        print(f"   📊 Train: {X_train.shape[0]} échantillons ({X_train.shape[0]/self.X_data.shape[0]:.1%})")
        print(f"   📊 Validation: {X_val.shape[0]} échantillons ({X_val.shape[0]/self.X_data.shape[0]:.1%})")
        print(f"   📊 Test: {X_test.shape[0]} échantillons ({X_test.shape[0]/self.X_data.shape[0]:.1%})")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def normalize_data(self, fit_on_train: bool = True) -> Dict[str, np.ndarray]:
        """
        Normalise les données avec StandardScaler
        
        Args:
            fit_on_train: Si True, fit les scalers sur les données d'entraînement uniquement
            
        Returns:
            Dictionnaire avec les données normalisées
        """
        if self.X_train is None:
            raise ValueError("Datasets non créés. Appelez create_train_val_test_split() d'abord.")
        
        print("🔧 Normalisation des données...")
        
        if fit_on_train:
            # Fit sur train uniquement
            self.scaler_X.fit(self.X_train)
            self.scaler_y.fit(self.y_train)
        else:
            # Fit sur toutes les données
            self.scaler_X.fit(self.X_data)
            self.scaler_y.fit(self.y_data)
        
        # Transform tous les datasets
        X_train_norm = self.scaler_X.transform(self.X_train)
        X_val_norm = self.scaler_X.transform(self.X_val)
        X_test_norm = self.scaler_X.transform(self.X_test)
        
        y_train_norm = self.scaler_y.transform(self.y_train)
        y_val_norm = self.scaler_y.transform(self.y_val)
        y_test_norm = self.scaler_y.transform(self.y_test)
        
        print("✅ Normalisation terminée")
        print(f"   📊 Moyenne X (train): {X_train_norm.mean():.6f}")
        print(f"   📊 Std X (train): {X_train_norm.std():.6f}")
        print(f"   📊 Moyenne y (train): {y_train_norm.mean(axis=0)}")
        print(f"   📊 Std y (train): {y_train_norm.std(axis=0)}")
        
        return {
            'X_train_norm': X_train_norm, 'y_train_norm': y_train_norm,
            'X_val_norm': X_val_norm, 'y_val_norm': y_val_norm,
            'X_test_norm': X_test_norm, 'y_test_norm': y_test_norm
        }
    
    def save_datasets(self, output_path: str):
        """Sauvegarde tous les datasets et scalers"""
        print(f"💾 Sauvegarde des datasets: {output_path}")
        
        # Sauvegarder les données
        np.savez_compressed(output_path,
            X_train=self.X_train, y_train=self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            X_test=self.X_test, y_test=self.y_test,
            X_data=self.X_data, y_data=self.y_data,
            filenames=self.filenames,
            r_min=self.r_min, r_max=self.r_max, delta_r=self.delta_r, n_points=self.n_points
        )
        
        # Sauvegarder les scalers
        scaler_path = output_path.replace('.npz', '_scalers.joblib')
        joblib.dump({
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }, scaler_path)
        
        print(f"✅ Sauvegarde terminée: {output_path} et {scaler_path}")
    
    def load_datasets(self, input_path: str):
        """Charge les datasets et scalers sauvegardés"""
        print(f"📂 Chargement des datasets: {input_path}")
        
        # Charger les données
        data = np.load(input_path)
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.X_data = data['X_data']
        self.y_data = data['y_data']
        self.filenames = data['filenames'].tolist()
        
        self.r_min = float(data['r_min'])
        self.r_max = float(data['r_max'])
        self.delta_r = float(data['delta_r'])
        self.n_points = int(data['n_points'])
        
        # Charger les scalers
        scaler_path = input_path.replace('.npz', '_scalers.joblib')
        if os.path.exists(scaler_path):
            scalers = joblib.load(scaler_path)
            self.scaler_X = scalers['scaler_X']
            self.scaler_y = scalers['scaler_y']
        
        print(f"✅ Chargement terminé")
        print(f"   📊 Train: {self.X_train.shape[0]} échantillons")
        print(f"   📊 Validation: {self.X_val.shape[0]} échantillons")
        print(f"   📊 Test: {self.X_test.shape[0]} échantillons")

def main():
    """Fonction principale de démonstration"""
    print("🧠 AdvancedDataLoader - Constitution des jeux de données")
    print("=" * 60)
    
    # Initialiser le chargeur
    loader = AdvancedDataLoader(
        train_dir="Train",
        preprocessed_data_path="preprocessed_data.npz"
    )
    
    # Charger un échantillon des données pour test (10% des fichiers)
    X_data, y_data, filenames = loader.load_all_training_data(sample_ratio=0.1)
    
    # Créer la division train/val/test
    datasets = loader.create_train_val_test_split()
    
    # Normaliser les données
    normalized_datasets = loader.normalize_data()
    
    # Sauvegarder les datasets
    loader.save_datasets("datasets_sample.npz")
    
    print("\n✅ Constitution des jeux de données terminée!")
    print("📁 Fichiers générés :")
    print("   - datasets_sample.npz")
    print("   - datasets_sample_scalers.joblib")

if __name__ == "__main__":
    main()
