#!/usr/bin/env python3
"""
Data Augmentation 2D par Interpolation pour Prédiction Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Ce module implémente une stratégie sophistiquée de data augmentation
par interpolation 2D dans l'espace des paramètres (gap, L_ecran).
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp2d, griddata
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataAugmentation2D:
    """
    Classe pour l'augmentation de données par interpolation 2D.
    
    Cette classe charge le dataset 2D complet et génère des échantillons
    interpolés physiquement cohérents dans l'espace (gap, L_ecran).
    """
    
    def __init__(self, dataset_path="data_generation/dataset_2D"):
        """
        Initialise l'augmentateur de données.
        
        Args:
            dataset_path (str): Chemin vers le dataset 2D
        """
        self.dataset_path = Path(dataset_path)
        self.original_data = []
        self.gaps = []
        self.L_ecrans = []
        self.profiles = []
        
        print(f"🔧 Initialisation DataAugmentation2D")
        print(f"📁 Dataset: {self.dataset_path}")
        
    def load_complete_dataset(self, max_files=None, truncate_to=600):
        """
        Charge le dataset complet depuis les fichiers .mat.
        
        Args:
            max_files (int): Limite le nombre de fichiers (None = tous)
            truncate_to (int): Tronque les profils à N points (600 recommandé)
        
        Returns:
            tuple: (X, y) où X sont les profils et y les paramètres [gap, L_ecran]
        """
        print(f"\n📊 Chargement du dataset complet...")
        
        # Trouver tous les fichiers .mat
        mat_files = list(self.dataset_path.glob("*.mat"))
        mat_files = [f for f in mat_files if f.name != "labels.mat"]
        
        if max_files:
            mat_files = mat_files[:max_files]
        
        print(f"   Fichiers trouvés: {len(mat_files)}")
        
        X_data = []
        y_data = []
        
        for i, mat_file in enumerate(mat_files):
            if i % 500 == 0:
                print(f"   Progression: {i}/{len(mat_files)} fichiers...")
            
            try:
                # Charger le fichier
                data = loadmat(str(mat_file))
                
                # Extraire les données
                ratio = data['ratio'].flatten()
                gap = float(data['gap'][0, 0])
                L_ecran = float(data['L_ecran_subs'][0, 0])
                
                # Tronquer si nécessaire
                if len(ratio) > truncate_to:
                    ratio = ratio[:truncate_to]
                
                # Stocker
                X_data.append(ratio)
                y_data.append([gap, L_ecran])
                
                # Stocker pour interpolation
                self.gaps.append(gap)
                self.L_ecrans.append(L_ecran)
                self.profiles.append(ratio)
                
            except Exception as e:
                print(f"   ⚠️  Erreur avec {mat_file.name}: {e}")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        print(f"✅ Dataset chargé: X{X.shape}, y{y.shape}")
        print(f"   Gap range: {np.min(y[:, 0]):.3f} - {np.max(y[:, 0]):.3f} µm")
        print(f"   L_ecran range: {np.min(y[:, 1]):.1f} - {np.max(y[:, 1]):.1f} µm")
        
        return X, y
    
    def create_parameter_grid(self, gap_density=2, L_ecran_density=2):
        """
        Crée une grille dense de paramètres pour l'interpolation.
        
        Args:
            gap_density (int): Facteur de densification pour gap
            L_ecran_density (int): Facteur de densification pour L_ecran
        
        Returns:
            tuple: (gap_grid, L_ecran_grid) grilles de paramètres
        """
        print(f"\n🎯 Création de la grille de paramètres...")
        
        # Plages originales
        gap_min, gap_max = min(self.gaps), max(self.gaps)
        L_min, L_max = min(self.L_ecrans), max(self.L_ecrans)
        
        # Nombre de points originaux
        n_gaps_orig = len(set(self.gaps))
        n_L_orig = len(set(self.L_ecrans))
        
        # Nouvelles grilles denses
        n_gaps_new = n_gaps_orig * gap_density
        n_L_new = n_L_orig * L_ecran_density
        
        gap_grid = np.linspace(gap_min, gap_max, n_gaps_new)
        L_ecran_grid = np.linspace(L_min, L_max, n_L_new)
        
        print(f"   Gap: {n_gaps_orig} → {n_gaps_new} points")
        print(f"   L_ecran: {n_L_orig} → {n_L_new} points")
        print(f"   Total: {len(self.gaps)} → {n_gaps_new * n_L_new} combinaisons")
        
        return gap_grid, L_ecran_grid
    
    def interpolate_profiles_2D(self, gap_grid, L_ecran_grid, method='linear'):
        """
        Interpole les profils d'intensité dans l'espace 2D (gap, L_ecran).
        
        Args:
            gap_grid (array): Grille de gaps
            L_ecran_grid (array): Grille de L_ecran
            method (str): Méthode d'interpolation ('linear', 'cubic')
        
        Returns:
            tuple: (X_interpolated, y_interpolated) données interpolées
        """
        print(f"\n🔄 Interpolation 2D des profils (méthode: {method})...")
        
        # Convertir en arrays numpy
        gaps_array = np.array(self.gaps)
        L_ecrans_array = np.array(self.L_ecrans)
        profiles_array = np.array(self.profiles)
        
        # Points d'interpolation
        points = np.column_stack((gaps_array, L_ecrans_array))
        
        # Créer la grille de sortie
        gap_mesh, L_mesh = np.meshgrid(gap_grid, L_ecran_grid)
        xi = np.column_stack((gap_mesh.ravel(), L_mesh.ravel()))
        
        X_interpolated = []
        y_interpolated = []
        
        print(f"   Interpolation de {profiles_array.shape[1]} points par profil...")
        
        # Interpoler chaque point du profil séparément
        for point_idx in range(profiles_array.shape[1]):
            if point_idx % 100 == 0:
                print(f"   Point {point_idx}/{profiles_array.shape[1]}...")
            
            # Valeurs à interpoler pour ce point
            values = profiles_array[:, point_idx]
            
            # Interpolation 2D
            try:
                interpolated_values = griddata(
                    points, values, xi, 
                    method=method, fill_value=np.nan
                )
                
                if point_idx == 0:
                    # Initialiser les arrays de sortie
                    n_interpolated = len(interpolated_values)
                    X_interpolated = np.zeros((n_interpolated, profiles_array.shape[1]))
                    y_interpolated = xi.copy()
                
                X_interpolated[:, point_idx] = interpolated_values
                
            except Exception as e:
                print(f"   ⚠️  Erreur interpolation point {point_idx}: {e}")
                continue
        
        # Supprimer les échantillons avec NaN
        valid_mask = ~np.isnan(X_interpolated).any(axis=1)
        X_interpolated = X_interpolated[valid_mask]
        y_interpolated = y_interpolated[valid_mask]
        
        print(f"✅ Interpolation terminée: {X_interpolated.shape[0]} nouveaux échantillons")
        
        return X_interpolated, y_interpolated
    
    def augment_dataset(self, gap_density=2, L_ecran_density=2, 
                       method='linear', include_original=True):
        """
        Augmente le dataset complet par interpolation 2D.
        
        Args:
            gap_density (int): Facteur de densification gap
            L_ecran_density (int): Facteur de densification L_ecran
            method (str): Méthode d'interpolation
            include_original (bool): Inclure les données originales
        
        Returns:
            tuple: (X_augmented, y_augmented) dataset augmenté
        """
        print(f"\n🚀 AUGMENTATION COMPLÈTE DU DATASET")
        print("="*50)
        
        # 1. Charger le dataset original
        X_original, y_original = self.load_complete_dataset()
        
        # 2. Créer la grille dense
        gap_grid, L_ecran_grid = self.create_parameter_grid(
            gap_density, L_ecran_density
        )
        
        # 3. Interpoler
        X_interpolated, y_interpolated = self.interpolate_profiles_2D(
            gap_grid, L_ecran_grid, method
        )
        
        # 4. Combiner les données
        if include_original:
            X_augmented = np.vstack([X_original, X_interpolated])
            y_augmented = np.vstack([y_original, y_interpolated])
            print(f"✅ Dataset final: {X_original.shape[0]} originaux + {X_interpolated.shape[0]} interpolés")
        else:
            X_augmented = X_interpolated
            y_augmented = y_interpolated
            print(f"✅ Dataset final: {X_interpolated.shape[0]} interpolés seulement")
        
        print(f"   Total: X{X_augmented.shape}, y{y_augmented.shape}")
        print(f"   Facteur d'augmentation: {X_augmented.shape[0] / X_original.shape[0]:.1f}x")
        
        return X_augmented, y_augmented
    
    def validate_interpolation(self, X_augmented, y_augmented, n_samples=5):
        """
        Valide la qualité de l'interpolation avec des visualisations.
        
        Args:
            X_augmented (array): Données augmentées
            y_augmented (array): Paramètres augmentés
            n_samples (int): Nombre d'échantillons à visualiser
        """
        print(f"\n🔍 Validation de l'interpolation...")
        
        # Sélectionner des échantillons aléatoires
        indices = np.random.choice(len(X_augmented), n_samples, replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Validation de l\'Interpolation 2D', fontsize=16, fontweight='bold')
        
        for i, idx in enumerate(indices):
            if i >= 6:  # Limite à 6 échantillons
                break
            
            row = i // 3
            col = i % 3
            
            ax = axes[row, col]
            
            # Tracer le profil interpolé
            x_coords = np.linspace(0, 6.916, len(X_augmented[idx]))
            ax.plot(x_coords, X_augmented[idx], 'b-', linewidth=2, alpha=0.8)
            
            gap = y_augmented[idx, 0]
            L_ecran = y_augmented[idx, 1]
            
            ax.set_title(f'Gap={gap:.3f}µm, L_ecran={L_ecran:.1f}µm')
            ax.set_xlabel('Position (µm)')
            ax.set_ylabel('Ratio I/I₀')
            ax.grid(True, alpha=0.3)
        
        # Supprimer les axes vides
        for i in range(n_samples, 6):
            row = i // 3
            col = i % 3
            axes[row, col].remove()
        
        plt.tight_layout()
        plt.savefig('data_augmentation_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Validation sauvegardée: data_augmentation_validation.png")


def main():
    """
    Fonction principale de démonstration.
    """
    print("🔧 DATA AUGMENTATION 2D - DÉMONSTRATION")
    print("="*60)
    
    # Créer l'augmentateur
    augmenter = DataAugmentation2D()
    
    # Augmenter le dataset (test avec facteur modéré)
    X_augmented, y_augmented = augmenter.augment_dataset(
        gap_density=2,      # Doubler la densité gap
        L_ecran_density=2,  # Doubler la densité L_ecran
        method='linear',
        include_original=True
    )
    
    # Valider l'interpolation
    augmenter.validate_interpolation(X_augmented, y_augmented)
    
    print("\n🎯 RÉSUMÉ")
    print("-"*30)
    print(f"Dataset augmenté: {X_augmented.shape[0]} échantillons")
    print(f"Facteur d'augmentation: {X_augmented.shape[0] / 2440:.1f}x")
    print(f"Prêt pour l'entraînement du réseau de neurones !")


if __name__ == "__main__":
    main()
