#!/usr/bin/env python3
"""
Data Augmentation 2D pour Réseau Neural Gap + L_écran - Dataset 2D Train

Auteur: Oussama GUELFAA
Date: 25/06/2025

Ce module implémente l'augmentation de données 2D en s'inspirant des méthodes
éprouvées du projet de référence, adaptées au nouveau dataset 2D divisé.

Fonctionnalités:
- Chargement du dataset 2D Train depuis les fichiers .mat
- Augmentation par interpolation 2D (spline, RBF, polynomial)
- Génération d'un dataset augmenté avec fichiers .mat et labels.csv
- Sauvegarde organisée dans un nouveau dossier
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.interpolate import griddata, RBFInterpolator
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAugmentation2D:
    """
    Classe pour l'augmentation de données 2D du dataset Train.
    
    Cette classe charge le dataset 2D Train, applique des transformations
    d'augmentation sophistiquées, et génère un nouveau dataset augmenté
    avec la même structure (fichiers .mat + labels.csv).
    """
    
    def __init__(self, train_dataset_path="data_generation/dataset_2D_Train"):
        """
        Initialise l'augmentateur de données.
        
        Args:
            train_dataset_path (str): Chemin vers le dataset 2D Train
        """
        self.train_dataset_path = Path(train_dataset_path)
        self.augmented_dataset_path = Path("data_generation/dataset_2D_Train_Augmented")
        
        # Données chargées
        self.original_data = []
        self.gaps = []
        self.L_ecrans = []
        self.profiles = []
        self.filenames = []
        
        logger.info(f"🔧 Initialisation DataAugmentation2D")
        logger.info(f"📁 Dataset source: {self.train_dataset_path}")
        logger.info(f"📁 Dataset augmenté: {self.augmented_dataset_path}")
        
        # Créer le dossier de destination
        self.augmented_dataset_path.mkdir(exist_ok=True)
        
    def load_train_dataset(self, max_files=None, truncate_to=600):
        """
        Charge le dataset d'entraînement depuis les fichiers .mat et labels.csv.
        
        Args:
            max_files (int): Limite le nombre de fichiers (None = tous)
            truncate_to (int): Tronque les profils à N points (600 recommandé)
        
        Returns:
            tuple: (X, y, filenames) où X sont les profils, y les paramètres, filenames les noms
        """
        logger.info(f"\n📊 Chargement du dataset d'entraînement...")
        
        # Charger le fichier labels.csv
        labels_path = self.train_dataset_path / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Fichier labels.csv non trouvé: {labels_path}")
        
        labels_df = pd.read_csv(labels_path)
        logger.info(f"   Labels chargés: {len(labels_df)} entrées")
        
        if max_files:
            labels_df = labels_df.head(max_files)
            logger.info(f"   Limité à {max_files} fichiers")
        
        X_data = []
        y_data = []
        filenames = []
        
        for i, (_, row) in enumerate(labels_df.iterrows()):
            if i % 500 == 0:
                logger.info(f"   Progression: {i}/{len(labels_df)} fichiers...")
            
            filename = row['filename']
            gap = row['gap_um']
            L_ecran = row['L_um']
            
            mat_file_path = self.train_dataset_path / filename
            
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
                
                # Stocker
                X_data.append(ratio)
                y_data.append([gap, L_ecran])
                filenames.append(filename)
                
                # Stocker pour interpolation
                self.gaps.append(gap)
                self.L_ecrans.append(L_ecran)
                self.profiles.append(ratio)
                self.filenames.append(filename)
                
            except Exception as e:
                logger.warning(f"   ⚠️  Erreur avec {filename}: {e}")
                continue
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        logger.info(f"✅ Dataset chargé: X{X.shape}, y{y.shape}")
        logger.info(f"   Gap range: {np.min(y[:, 0]):.4f} - {np.max(y[:, 0]):.4f} µm")
        logger.info(f"   L_ecran range: {np.min(y[:, 1]):.1f} - {np.max(y[:, 1]):.1f} µm")
        
        return X, y, filenames
    
    def create_parameter_grid(self, gap_density=2, L_ecran_density=2):
        """
        Crée une grille dense de paramètres pour l'interpolation.
        
        Args:
            gap_density (int): Facteur de densification pour gap
            L_ecran_density (int): Facteur de densification pour L_ecran
        
        Returns:
            tuple: (gap_grid, L_ecran_grid) grilles de paramètres
        """
        logger.info(f"\n🎯 Création de la grille de paramètres...")
        
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
        
        logger.info(f"   Gap: {n_gaps_orig} → {n_gaps_new} points")
        logger.info(f"   L_ecran: {n_L_orig} → {n_L_new} points")
        logger.info(f"   Total: {len(self.gaps)} → {n_gaps_new * n_L_new} combinaisons")
        
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
        logger.info(f"\n🔄 Interpolation 2D des profils (méthode: {method})...")
        
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
        
        logger.info(f"   Interpolation de {profiles_array.shape[1]} points par profil...")
        
        # Interpoler chaque point du profil séparément
        for point_idx in range(profiles_array.shape[1]):
            if point_idx % 100 == 0:
                logger.info(f"   Point {point_idx}/{profiles_array.shape[1]}...")
            
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
                logger.warning(f"   ⚠️  Erreur interpolation point {point_idx}: {e}")
                continue
        
        # Supprimer les échantillons avec NaN
        valid_mask = ~np.isnan(X_interpolated).any(axis=1)
        X_interpolated = X_interpolated[valid_mask]
        y_interpolated = y_interpolated[valid_mask]
        
        logger.info(f"✅ Interpolation terminée: {X_interpolated.shape[0]} nouveaux échantillons")
        
        return X_interpolated, y_interpolated
    
    def generate_filename(self, gap, L_ecran, suffix="interp"):
        """
        Génère un nom de fichier pour un échantillon augmenté.
        
        Args:
            gap (float): Valeur du gap
            L_ecran (float): Valeur de L_ecran
            suffix (str): Suffixe pour identifier l'augmentation
        
        Returns:
            str: Nom de fichier généré
        """
        return f"gap_{gap:.4f}um_L_{L_ecran:.3f}um_{suffix}.mat"
    
    def save_augmented_sample(self, profile, gap, L_ecran, filename):
        """
        Sauvegarde un échantillon augmenté en fichier .mat.
        
        Args:
            profile (array): Profil d'intensité
            gap (float): Valeur du gap
            L_ecran (float): Valeur de L_ecran
            filename (str): Nom du fichier
        """
        file_path = self.augmented_dataset_path / filename
        
        # Préparer les données à sauvegarder
        data_to_save = {
            'ratio': profile.reshape(-1, 1),
            'gap': np.array([[gap]]),
            'L_ecran_subs': np.array([[L_ecran]])
        }
        
        try:
            savemat(str(file_path), data_to_save)
        except Exception as e:
            logger.error(f"Erreur sauvegarde {filename}: {e}")
            raise

    def augment_dataset_standard(self, gap_density=2, L_ecran_density=2,
                                method='linear', include_original=True):
        """
        Augmente le dataset par interpolation 2D standard.

        Args:
            gap_density (int): Facteur de densification gap
            L_ecran_density (int): Facteur de densification L_ecran
            method (str): Méthode d'interpolation
            include_original (bool): Inclure les données originales

        Returns:
            tuple: (X_augmented, y_augmented, filenames_augmented)
        """
        logger.info(f"\n🚀 AUGMENTATION STANDARD DU DATASET")
        logger.info("="*50)

        # 1. Charger le dataset original
        X_original, y_original, filenames_original = self.load_train_dataset()

        # 2. Créer la grille dense
        gap_grid, L_ecran_grid = self.create_parameter_grid(
            gap_density, L_ecran_density
        )

        # 3. Interpoler
        X_interpolated, y_interpolated = self.interpolate_profiles_2D(
            gap_grid, L_ecran_grid, method
        )

        # 4. Générer les noms de fichiers pour les données interpolées
        filenames_interpolated = []
        for i, (gap, L_ecran) in enumerate(y_interpolated):
            filename = self.generate_filename(gap, L_ecran, f"interp_{method}_{i:06d}")
            filenames_interpolated.append(filename)

        # 5. Combiner les données
        if include_original:
            X_augmented = np.vstack([X_original, X_interpolated])
            y_augmented = np.vstack([y_original, y_interpolated])
            filenames_augmented = filenames_original + filenames_interpolated
            logger.info(f"✅ Dataset final: {X_original.shape[0]} originaux + {X_interpolated.shape[0]} interpolés")
        else:
            X_augmented = X_interpolated
            y_augmented = y_interpolated
            filenames_augmented = filenames_interpolated
            logger.info(f"✅ Dataset final: {X_interpolated.shape[0]} interpolés seulement")

        logger.info(f"   Total: X{X_augmented.shape}, y{y_augmented.shape}")
        logger.info(f"   Facteur d'augmentation: {X_augmented.shape[0] / X_original.shape[0]:.1f}x")

        return X_augmented, y_augmented, filenames_augmented

    def augment_with_rbf(self, gap_density=3, L_ecran_density=2):
        """
        Augmentation par interpolation RBF (Radial Basis Function).

        Args:
            gap_density (int): Facteur de densification gap
            L_ecran_density (int): Facteur de densification L_ecran

        Returns:
            tuple: (X_rbf, y_rbf, filenames_rbf)
        """
        logger.info(f"\n🎯 AUGMENTATION PAR RBF")
        logger.info("="*30)

        # Créer une grille dense
        gap_grid, L_ecran_grid = self.create_parameter_grid(gap_density, L_ecran_density)

        # Convertir en arrays numpy
        gaps_array = np.array(self.gaps)
        L_ecrans_array = np.array(self.L_ecrans)
        profiles_array = np.array(self.profiles)

        # Points d'interpolation
        points = np.column_stack((gaps_array, L_ecrans_array))

        # Créer la grille de sortie
        gap_mesh, L_mesh = np.meshgrid(gap_grid, L_ecran_grid)
        xi = np.column_stack((gap_mesh.ravel(), L_mesh.ravel()))

        X_rbf_list = []
        y_rbf_list = []

        logger.info(f"   Interpolation RBF de {profiles_array.shape[1]} points par profil...")

        # Interpolation RBF pour chaque point du profil
        for i in range(profiles_array.shape[1]):
            if i % 100 == 0:
                logger.info(f"   Point {i}/{profiles_array.shape[1]}...")

            try:
                # Utiliser RBF avec kernel thin_plate_spline (pas besoin d'epsilon)
                rbf = RBFInterpolator(points, profiles_array[:, i],
                                    kernel='thin_plate_spline', smoothing=0.05)
                intensity_interpolated = rbf(xi)

                if i == 0:
                    for j in range(len(xi)):
                        X_rbf_list.append([])
                        y_rbf_list.append(xi[j])

                for j, intensity in enumerate(intensity_interpolated):
                    X_rbf_list[j].append(intensity)

            except Exception as e:
                logger.warning(f"   ⚠️  Erreur RBF au point {i}: {e}")
                continue

        if X_rbf_list:
            X_rbf = np.array(X_rbf_list)
            y_rbf = np.array(y_rbf_list)

            # Filtrer les valeurs aberrantes
            valid_mask = np.all(np.isfinite(X_rbf), axis=1)
            X_rbf = X_rbf[valid_mask]
            y_rbf = y_rbf[valid_mask]

            # Générer les noms de fichiers
            filenames_rbf = []
            for i, (gap, L_ecran) in enumerate(y_rbf):
                filename = self.generate_filename(gap, L_ecran, f"rbf_{i:06d}")
                filenames_rbf.append(filename)
        else:
            X_rbf = np.empty((0, profiles_array.shape[1]))
            y_rbf = np.empty((0, 2))
            filenames_rbf = []

        logger.info(f"✅ Augmentation RBF terminée: {len(X_rbf)} échantillons")

        return X_rbf, y_rbf, filenames_rbf

    def save_augmented_dataset(self, X_augmented, y_augmented, filenames_augmented):
        """
        Sauvegarde le dataset augmenté complet (fichiers .mat + labels.csv).

        Args:
            X_augmented (array): Profils augmentés
            y_augmented (array): Paramètres augmentés
            filenames_augmented (list): Noms de fichiers
        """
        logger.info(f"\n💾 Sauvegarde du dataset augmenté...")

        # 1. Sauvegarder les fichiers .mat
        logger.info(f"   Sauvegarde de {len(X_augmented)} fichiers .mat...")

        saved_count = 0
        failed_count = 0

        for i, (profile, (gap, L_ecran), filename) in enumerate(zip(X_augmented, y_augmented, filenames_augmented)):
            if i % 1000 == 0:
                logger.info(f"   Progression: {i}/{len(X_augmented)} fichiers...")

            try:
                self.save_augmented_sample(profile, gap, L_ecran, filename)
                saved_count += 1
            except Exception as e:
                logger.warning(f"   ⚠️  Échec sauvegarde {filename}: {e}")
                failed_count += 1

        # 2. Créer le fichier labels.csv
        logger.info(f"   Création du fichier labels.csv...")

        labels_data = {
            'filename': filenames_augmented,
            'gap_um': y_augmented[:, 0],
            'L_um': y_augmented[:, 1]
        }

        labels_df = pd.DataFrame(labels_data)
        labels_csv_path = self.augmented_dataset_path / "labels.csv"
        labels_df.to_csv(labels_csv_path, index=False)

        logger.info(f"✅ Sauvegarde terminée:")
        logger.info(f"   - Fichiers .mat sauvegardés: {saved_count}")
        logger.info(f"   - Échecs: {failed_count}")
        logger.info(f"   - Labels CSV: {labels_csv_path}")
        logger.info(f"   - Dossier: {self.augmented_dataset_path}")

        return saved_count, failed_count

    def validate_augmentation(self, X_augmented, y_augmented, n_samples=6):
        """
        Valide la qualité de l'augmentation avec des visualisations.

        Args:
            X_augmented (array): Données augmentées
            y_augmented (array): Paramètres augmentés
            n_samples (int): Nombre d'échantillons à visualiser
        """
        logger.info(f"\n🔍 Validation de l'augmentation...")

        # Sélectionner des échantillons aléatoires
        indices = np.random.choice(len(X_augmented), n_samples, replace=False)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Validation de l\'Augmentation 2D', fontsize=16, fontweight='bold')

        for i, idx in enumerate(indices):
            if i >= 6:  # Limite à 6 échantillons
                break

            row = i // 3
            col = i % 3

            ax = axes[row, col]

            # Tracer le profil augmenté
            x_coords = np.linspace(0, 6.916, len(X_augmented[idx]))
            ax.plot(x_coords, X_augmented[idx], 'b-', linewidth=2, alpha=0.8)

            gap = y_augmented[idx, 0]
            L_ecran = y_augmented[idx, 1]

            ax.set_title(f'Gap={gap:.4f}µm, L_ecran={L_ecran:.1f}µm')
            ax.set_xlabel('Position (µm)')
            ax.set_ylabel('Ratio I/I₀')
            ax.grid(True, alpha=0.3)

        # Supprimer les axes vides
        for i in range(n_samples, 6):
            row = i // 3
            col = i % 3
            axes[row, col].remove()

        plt.tight_layout()

        # Sauvegarder dans le dossier plots du projet
        plots_dir = Path("Reseau_Neural_2D_Gap_Lecran_25_06_25/plots")
        plots_dir.mkdir(exist_ok=True)
        validation_path = plots_dir / 'data_augmentation_validation.png'

        plt.savefig(validation_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ Validation sauvegardée: {validation_path}")


def main():
    """
    Fonction principale pour l'augmentation du dataset 2D Train.
    """
    logger.info("🔧 DATA AUGMENTATION 2D - DATASET TRAIN")
    logger.info("="*50)

    # Créer l'augmentateur
    augmenter = DataAugmentation2D()

    # 1. Augmentation standard par interpolation linéaire
    logger.info("\n🎯 ÉTAPE 1: Augmentation Standard (Interpolation Linéaire)")
    X_standard, y_standard, filenames_standard = augmenter.augment_dataset_standard(
        gap_density=2,        # Facteur 2 pour gap
        L_ecran_density=2,    # Facteur 2 pour L_ecran
        method='linear',      # Interpolation linéaire
        include_original=True # Inclure les données originales
    )

    # 2. Augmentation par RBF
    logger.info("\n🎯 ÉTAPE 2: Augmentation RBF")
    X_rbf, y_rbf, filenames_rbf = augmenter.augment_with_rbf(
        gap_density=2,        # Facteur 2 pour gap
        L_ecran_density=2     # Facteur 2 pour L_ecran
    )

    # 3. Combiner les augmentations
    logger.info("\n🔄 ÉTAPE 3: Combinaison des augmentations")
    X_final = np.vstack([X_standard, X_rbf])
    y_final = np.vstack([y_standard, y_rbf])
    filenames_final = filenames_standard + filenames_rbf

    logger.info(f"   Dataset final: {X_final.shape[0]} échantillons")
    logger.info(f"   - Standard: {X_standard.shape[0]} échantillons")
    logger.info(f"   - RBF: {X_rbf.shape[0]} échantillons")

    # 4. Validation de l'augmentation
    logger.info("\n🔍 ÉTAPE 4: Validation")
    augmenter.validate_augmentation(X_final, y_final, n_samples=6)

    # 5. Sauvegarder le dataset augmenté
    logger.info("\n💾 ÉTAPE 5: Sauvegarde")
    saved_count, failed_count = augmenter.save_augmented_dataset(
        X_final, y_final, filenames_final
    )

    # 6. Résumé final
    logger.info("\n🎯 RÉSUMÉ FINAL - AUGMENTATION 2D")
    logger.info("-"*40)
    logger.info(f"Dataset original: {len(augmenter.gaps)} échantillons")
    logger.info(f"Dataset augmenté: {X_final.shape[0]} échantillons")
    logger.info(f"Facteur d'augmentation: {X_final.shape[0] / len(augmenter.gaps):.1f}x")
    logger.info(f"Fichiers sauvegardés: {saved_count}")
    logger.info(f"Échecs: {failed_count}")
    logger.info(f"✅ Méthodes utilisées: Interpolation linéaire + RBF")
    logger.info(f"📁 Dossier de sortie: {augmenter.augmented_dataset_path}")


if __name__ == "__main__":
    main()
