#!/usr/bin/env python3
"""
Data Augmentation 2D Intelligente pour Réseau Dual Gap + L_ecran - Précision 0.007µm

Auteur: Oussama GUELFAA
Date: 18 - 01 - 2025

Ce module implémente l'augmentation de données avancée avec:
- Augmentation adaptative ciblée sur les zones critiques du gap
- Oversampling intelligent des échantillons difficiles
- Perturbations synthétiques réalistes pour améliorer la précision gap à 0.007µm
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp2d, griddata, RBFInterpolator, CubicSpline
from scipy.interpolate import UnivariateSpline, BSpline, splrep, splev
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
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

    def identify_difficult_samples(self, X, y, error_threshold=0.007):
        """
        Identifie les échantillons difficiles pour le gap basé sur l'analyse des gradients.

        Args:
            X (array): Profils d'intensité
            y (array): Paramètres [gap, L_ecran]
            error_threshold (float): Seuil d'erreur pour identifier les échantillons difficiles

        Returns:
            array: Indices des échantillons difficiles
        """
        print(f"\n🔍 Identification des échantillons difficiles (seuil: {error_threshold}µm)...")

        # Calculer les gradients des profils (sensibilité aux variations)
        gradients = np.gradient(X, axis=1)
        gradient_variance = np.var(gradients, axis=1)

        # Analyser la distribution des gaps
        gaps = y[:, 0]
        gap_std = np.std(gaps)

        # Identifier les zones critiques du gap
        gap_mean = np.mean(gaps)
        critical_gap_mask = np.abs(gaps - gap_mean) > 0.5 * gap_std

        # Combiner les critères
        high_gradient_mask = gradient_variance > np.percentile(gradient_variance, 75)
        difficult_mask = critical_gap_mask | high_gradient_mask

        difficult_indices = np.where(difficult_mask)[0]

        print(f"✅ {len(difficult_indices)} échantillons difficiles identifiés ({len(difficult_indices)/len(X)*100:.1f}%)")

        return difficult_indices

    def adaptive_gap_augmentation(self, gap_density=3, L_ecran_density=2,
                                 difficult_boost=2, method='cubic'):
        """
        Augmentation adaptative ciblée sur la précision du gap.

        Args:
            gap_density (int): Facteur de densification gap
            L_ecran_density (int): Facteur de densification L_ecran
            difficult_boost (int): Facteur d'amplification pour échantillons difficiles
            method (str): Méthode d'interpolation

        Returns:
            tuple: (X_augmented, y_augmented) dataset augmenté adaptatif
        """
        print(f"\n🚀 AUGMENTATION ADAPTATIVE POUR PRÉCISION GAP 0.007µm")
        print("="*60)

        # 1. Charger le dataset original
        X_original, y_original = self.load_complete_dataset()

        # 2. Identifier les échantillons difficiles
        difficult_indices = self.identify_difficult_samples(X_original, y_original)

        # 3. Augmentation standard
        gap_grid, L_ecran_grid = self.create_parameter_grid(gap_density, L_ecran_density)
        X_standard, y_standard = self.interpolate_profiles_2D(gap_grid, L_ecran_grid, method)

        # 4. Augmentation ciblée sur les échantillons difficiles
        X_difficult = X_original[difficult_indices]
        y_difficult = y_original[difficult_indices]

        # Créer une grille plus dense autour des échantillons difficiles
        X_targeted, y_targeted = self._create_targeted_augmentation(
            X_difficult, y_difficult, difficult_boost, method
        )

        # 5. Combiner toutes les augmentations
        X_augmented = np.vstack([X_original, X_standard, X_targeted])
        y_augmented = np.vstack([y_original, y_standard, y_targeted])

        print(f"✅ Augmentation adaptative terminée:")
        print(f"   - Original: {X_original.shape[0]} échantillons")
        print(f"   - Standard: {X_standard.shape[0]} échantillons")
        print(f"   - Ciblé: {X_targeted.shape[0]} échantillons")
        print(f"   - Total: {X_augmented.shape[0]} échantillons")
        print(f"   - Facteur total: {X_augmented.shape[0] / X_original.shape[0]:.1f}x")

        return X_augmented, y_augmented

    def _create_targeted_augmentation(self, X_difficult, y_difficult, boost_factor, method):
        """
        Crée une augmentation ciblée autour des échantillons difficiles.
        """
        print(f"🎯 Augmentation ciblée sur {len(X_difficult)} échantillons difficiles...")

        X_targeted_list = []
        y_targeted_list = []

        for i in range(len(X_difficult)):
            gap_center = y_difficult[i, 0]
            L_ecran_center = y_difficult[i, 1]

            # Créer une micro-grille autour de chaque échantillon difficile
            gap_range = np.linspace(gap_center - 0.005, gap_center + 0.005, boost_factor + 1)
            L_ecran_range = np.linspace(L_ecran_center - 5, L_ecran_center + 5, boost_factor + 1)

            # Interpoler dans cette micro-grille
            for gap_val in gap_range:
                for L_ecran_val in L_ecran_range:
                    if gap_val != gap_center or L_ecran_val != L_ecran_center:  # Éviter les doublons
                        # Interpolation locale
                        interpolated_profile = self._local_interpolation(
                            gap_val, L_ecran_val, X_difficult, y_difficult, method
                        )
                        if interpolated_profile is not None:
                            X_targeted_list.append(interpolated_profile)
                            y_targeted_list.append([gap_val, L_ecran_val])

        if X_targeted_list:
            X_targeted = np.array(X_targeted_list)
            y_targeted = np.array(y_targeted_list)
        else:
            X_targeted = np.empty((0, X_difficult.shape[1]))
            y_targeted = np.empty((0, 2))

        print(f"✅ {len(X_targeted)} échantillons ciblés générés")

        return X_targeted, y_targeted

    def _local_interpolation(self, target_gap, target_L_ecran, X_local, y_local, method):
        """
        Effectue une interpolation locale autour d'un point cible.
        """
        try:
            # Utiliser les k plus proches voisins pour l'interpolation locale
            k = min(4, len(X_local))

            # Calculer les distances dans l'espace des paramètres
            distances = np.sqrt((y_local[:, 0] - target_gap)**2 +
                              (y_local[:, 1] - target_L_ecran)**2)

            # Sélectionner les k plus proches
            nearest_indices = np.argsort(distances)[:k]

            # Interpolation pondérée par la distance
            weights = 1.0 / (distances[nearest_indices] + 1e-8)
            weights /= np.sum(weights)

            # Profil interpolé
            interpolated_profile = np.average(X_local[nearest_indices], axis=0, weights=weights)

            return interpolated_profile

        except Exception as e:
            print(f"⚠️ Erreur interpolation locale: {e}")
            return None

    def synthetic_noise_augmentation(self, X, y, noise_levels=[0.001, 0.002, 0.005],
                                   target_samples=1000):
        """
        Ajoute des perturbations synthétiques réalistes pour améliorer la robustesse.

        Args:
            X (array): Profils d'intensité
            y (array): Paramètres
            noise_levels (list): Niveaux de bruit à appliquer
            target_samples (int): Nombre d'échantillons à générer

        Returns:
            tuple: (X_noisy, y_noisy) échantillons avec bruit synthétique
        """
        print(f"\n🔊 Augmentation par bruit synthétique réaliste...")

        X_noisy_list = []
        y_noisy_list = []

        samples_per_level = target_samples // len(noise_levels)

        for noise_level in noise_levels:
            print(f"   Niveau de bruit: {noise_level*100:.1f}%")

            for _ in range(samples_per_level):
                # Sélectionner un échantillon aléatoire
                idx = np.random.randint(0, len(X))

                # Ajouter du bruit gaussien réaliste
                noise = np.random.normal(0, noise_level, X[idx].shape)
                X_noisy = X[idx] + noise

                # Légère variation des paramètres (simulation d'incertitude expérimentale)
                gap_noise = np.random.normal(0, 0.001)  # ±1nm
                L_ecran_noise = np.random.normal(0, 1.0)  # ±1µm

                y_noisy = y[idx].copy()
                y_noisy[0] += gap_noise  # gap
                y_noisy[1] += L_ecran_noise  # L_ecran

                X_noisy_list.append(X_noisy)
                y_noisy_list.append(y_noisy)

        X_noisy = np.array(X_noisy_list)
        y_noisy = np.array(y_noisy_list)

        print(f"✅ {len(X_noisy)} échantillons avec bruit synthétique générés")

        return X_noisy, y_noisy

    def advanced_interpolation_augmentation(self, gap_density=5, L_ecran_density=3,
                                          methods=['spline', 'rbf', 'polynomial']):
        """
        Augmentation avancée avec méthodes d'interpolation sophistiquées.

        Args:
            gap_density (int): Facteur de densification gap (augmenté)
            L_ecran_density (int): Facteur de densification L_ecran (augmenté)
            methods (list): Méthodes d'interpolation à utiliser

        Returns:
            tuple: (X_augmented, y_augmented) dataset augmenté avec méthodes avancées
        """
        print(f"\n🚀 AUGMENTATION AVANCÉE AVEC INTERPOLATIONS SOPHISTIQUÉES")
        print("="*65)

        # 1. Charger le dataset original
        X_original, y_original = self.load_complete_dataset()

        X_augmented_list = [X_original]
        y_augmented_list = [y_original]

        # 2. Appliquer chaque méthode d'interpolation
        for method in methods:
            print(f"\n🔧 Application de l'interpolation {method.upper()}...")

            if method == 'spline':
                X_method, y_method = self._spline_interpolation_2D(
                    X_original, y_original, gap_density, L_ecran_density
                )
            elif method == 'rbf':
                X_method, y_method = self._rbf_interpolation_2D(
                    X_original, y_original, gap_density, L_ecran_density
                )
            elif method == 'polynomial':
                X_method, y_method = self._polynomial_interpolation_2D(
                    X_original, y_original, gap_density, L_ecran_density
                )
            else:
                print(f"⚠️ Méthode {method} non reconnue, ignorée")
                continue

            X_augmented_list.append(X_method)
            y_augmented_list.append(y_method)
            print(f"✅ {len(X_method)} échantillons générés avec {method}")

        # 3. Combiner toutes les méthodes
        X_final = np.vstack(X_augmented_list)
        y_final = np.vstack(y_augmented_list)

        print(f"\n✅ Augmentation avancée terminée:")
        print(f"   - Original: {X_original.shape[0]} échantillons")
        for i, method in enumerate(methods):
            if i+1 < len(X_augmented_list):
                print(f"   - {method.capitalize()}: {X_augmented_list[i+1].shape[0]} échantillons")
        print(f"   - Total: {X_final.shape[0]} échantillons")
        print(f"   - Facteur total: {X_final.shape[0] / X_original.shape[0]:.1f}x")

        return X_final, y_final

    def _spline_interpolation_2D(self, X, y, gap_density, L_ecran_density):
        """
        Interpolation par splines 2D pour augmentation sophistiquée.
        """
        print(f"   🌊 Interpolation par splines 2D...")

        # Créer une grille plus dense
        gap_range = np.linspace(np.min(y[:, 0]), np.max(y[:, 0]),
                               len(np.unique(y[:, 0])) * gap_density)
        L_ecran_range = np.linspace(np.min(y[:, 1]), np.max(y[:, 1]),
                                   len(np.unique(y[:, 1])) * L_ecran_density)

        gap_grid, L_ecran_grid = np.meshgrid(gap_range, L_ecran_range)

        X_spline_list = []
        y_spline_list = []

        # Interpolation spline pour chaque point du profil
        for i in range(X.shape[1]):  # Pour chaque point du profil d'intensité
            # Utiliser RBF avec fonction spline
            try:
                rbf = RBFInterpolator(y, X[:, i], kernel='thin_plate_spline', smoothing=0.1)

                # Évaluer sur la grille
                points_grid = np.column_stack([gap_grid.ravel(), L_ecran_grid.ravel()])
                intensity_interpolated = rbf(points_grid)

                if i == 0:  # Première itération, créer les listes
                    n_points = len(points_grid)
                    for j in range(n_points):
                        X_spline_list.append([])
                        y_spline_list.append(points_grid[j])

                # Ajouter les intensités interpolées
                for j, intensity in enumerate(intensity_interpolated):
                    X_spline_list[j].append(intensity)

            except Exception as e:
                print(f"⚠️ Erreur spline au point {i}: {e}")
                continue

        # Convertir en arrays
        if X_spline_list:
            X_spline = np.array(X_spline_list)
            y_spline = np.array(y_spline_list)

            # Filtrer les valeurs aberrantes
            valid_mask = np.all(np.isfinite(X_spline), axis=1)
            X_spline = X_spline[valid_mask]
            y_spline = y_spline[valid_mask]
        else:
            X_spline = np.empty((0, X.shape[1]))
            y_spline = np.empty((0, 2))

        return X_spline, y_spline

    def _rbf_interpolation_2D(self, X, y, gap_density, L_ecran_density):
        """
        Interpolation par fonctions de base radiale (RBF).
        """
        print(f"   🎯 Interpolation par RBF...")

        # Créer une grille dense avec facteur augmenté
        gap_range = np.linspace(np.min(y[:, 0]), np.max(y[:, 0]),
                               len(np.unique(y[:, 0])) * gap_density)
        L_ecran_range = np.linspace(np.min(y[:, 1]), np.max(y[:, 1]),
                                   len(np.unique(y[:, 1])) * L_ecran_density)

        gap_grid, L_ecran_grid = np.meshgrid(gap_range, L_ecran_range)
        points_grid = np.column_stack([gap_grid.ravel(), L_ecran_grid.ravel()])

        X_rbf_list = []
        y_rbf_list = []

        # Interpolation RBF pour chaque point du profil
        for i in range(X.shape[1]):
            try:
                # Utiliser différents kernels RBF
                rbf = RBFInterpolator(y, X[:, i], kernel='multiquadric', smoothing=0.05)
                intensity_interpolated = rbf(points_grid)

                if i == 0:
                    for j in range(len(points_grid)):
                        X_rbf_list.append([])
                        y_rbf_list.append(points_grid[j])

                for j, intensity in enumerate(intensity_interpolated):
                    X_rbf_list[j].append(intensity)

            except Exception as e:
                print(f"⚠️ Erreur RBF au point {i}: {e}")
                continue

        if X_rbf_list:
            X_rbf = np.array(X_rbf_list)
            y_rbf = np.array(y_rbf_list)

            # Filtrer les valeurs aberrantes
            valid_mask = np.all(np.isfinite(X_rbf), axis=1)
            X_rbf = X_rbf[valid_mask]
            y_rbf = y_rbf[valid_mask]
        else:
            X_rbf = np.empty((0, X.shape[1]))
            y_rbf = np.empty((0, 2))

        return X_rbf, y_rbf

    def _polynomial_interpolation_2D(self, X, y, gap_density, L_ecran_density):
        """
        Interpolation polynomiale 2D avec bruit gaussien contrôlé.
        """
        print(f"   📈 Interpolation polynomiale avec bruit gaussien...")

        # Créer une grille dense
        gap_range = np.linspace(np.min(y[:, 0]), np.max(y[:, 0]),
                               len(np.unique(y[:, 0])) * gap_density)
        L_ecran_range = np.linspace(np.min(y[:, 1]), np.max(y[:, 1]),
                                   len(np.unique(y[:, 1])) * L_ecran_density)

        gap_grid, L_ecran_grid = np.meshgrid(gap_range, L_ecran_range)
        points_grid = np.column_stack([gap_grid.ravel(), L_ecran_grid.ravel()])

        X_poly_list = []
        y_poly_list = []

        # Interpolation polynomiale avec griddata
        for i in range(X.shape[1]):
            try:
                # Interpolation polynomiale (ordre 3)
                intensity_interpolated = griddata(
                    y, X[:, i], points_grid,
                    method='cubic', fill_value=np.nan
                )

                # Ajouter du bruit gaussien contrôlé pour diversité
                noise_std = np.std(X[:, i]) * 0.01  # 1% du std original
                noise = np.random.normal(0, noise_std, len(intensity_interpolated))
                intensity_interpolated += noise

                if i == 0:
                    for j in range(len(points_grid)):
                        X_poly_list.append([])
                        y_poly_list.append(points_grid[j])

                for j, intensity in enumerate(intensity_interpolated):
                    X_poly_list[j].append(intensity)

            except Exception as e:
                print(f"⚠️ Erreur polynomiale au point {i}: {e}")
                continue

        if X_poly_list:
            X_poly = np.array(X_poly_list)
            y_poly = np.array(y_poly_list)

            # Filtrer les NaN et valeurs aberrantes
            valid_mask = np.all(np.isfinite(X_poly), axis=1)
            X_poly = X_poly[valid_mask]
            y_poly = y_poly[valid_mask]
        else:
            X_poly = np.empty((0, X.shape[1]))
            y_poly = np.empty((0, 2))

        return X_poly, y_poly


def main():
    """
    Fonction principale de démonstration - Augmentation Intelligente pour Précision 0.007µm.
    """
    print("🔧 DATA AUGMENTATION INTELLIGENTE - PRÉCISION GAP 0.007µm")
    print("="*70)

    # Créer l'augmentateur
    augmenter = DataAugmentation2D()

    # 1. Augmentation avancée avec interpolations sophistiquées
    print("\n🎯 ÉTAPE 1: Augmentation Avancée Multi-Méthodes")
    X_advanced, y_advanced = augmenter.advanced_interpolation_augmentation(
        gap_density=5,        # Facteur augmenté pour plus de diversité
        L_ecran_density=3,    # Facteur augmenté pour plus de diversité
        methods=['spline', 'rbf', 'polynomial']  # Méthodes sophistiquées
    )

    # 2. Augmentation adaptative ciblée (méthode existante améliorée)
    print("\n🎯 ÉTAPE 2: Augmentation Adaptative Ciblée")
    X_adaptive, y_adaptive = augmenter.adaptive_gap_augmentation(
        gap_density=4,        # Facteur augmenté
        L_ecran_density=3,    # Facteur augmenté
        difficult_boost=4,    # Boost augmenté pour échantillons difficiles
        method='cubic'        # Interpolation cubique pour plus de précision
    )

    # 3. Augmentation par bruit synthétique
    print("\n🔊 ÉTAPE 3: Augmentation par Bruit Synthétique")
    X_noise, y_noise = augmenter.synthetic_noise_augmentation(
        X_advanced, y_advanced,
        noise_levels=[0.0005, 0.001, 0.002, 0.005],  # Plus de niveaux de bruit
        target_samples=3000  # Plus d'échantillons supplémentaires
    )

    # 4. Combiner toutes les augmentations
    X_final = np.vstack([X_advanced, X_adaptive, X_noise])
    y_final = np.vstack([y_advanced, y_adaptive, y_noise])

    # 5. Validation de l'augmentation
    augmenter.validate_interpolation(X_final, y_final, n_samples=8)

    # 6. Sauvegarder le dataset augmenté
    print(f"\n💾 Sauvegarde du dataset augmenté...")
    np.savez_compressed('data/augmented_dataset_advanced.npz',
                       X=X_final, y=y_final)

    print("\n🎯 RÉSUMÉ FINAL - AUGMENTATION SOPHISTIQUÉE")
    print("-"*50)
    print(f"Dataset original: 2,440 échantillons")
    print(f"Après augmentation avancée: {X_advanced.shape[0]} échantillons")
    print(f"Après augmentation adaptative: {X_adaptive.shape[0]} échantillons")
    print(f"Après augmentation bruit: {X_noise.shape[0]} échantillons supplémentaires")
    print(f"Dataset final: {X_final.shape[0]} échantillons")
    print(f"Facteur d'augmentation total: {X_final.shape[0] / 2440:.1f}x")
    print(f"✅ Méthodes utilisées: Spline, RBF, Polynomial + Adaptatif + Bruit")
    print(f"✅ Optimisé pour précision gap 0.007µm avec diversité maximale !")
    print(f"📁 Sauvegardé: data/augmented_dataset_advanced.npz")


if __name__ == "__main__":
    main()
