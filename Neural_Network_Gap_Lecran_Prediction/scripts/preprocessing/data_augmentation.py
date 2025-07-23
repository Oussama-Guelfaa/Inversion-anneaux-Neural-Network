#!/usr/bin/env python3
"""
Data Augmentation avancée pour Neural_Network_Gap_Lecran_Prediction
Auteur: Oussama GUELFAA
Date: 15/07/2025

Ce module implémente des techniques d'augmentation de données sophistiquées
incluant l'interpolation 2D, l'augmentation des labels, et diverses transformations
pour améliorer la robustesse et la généralisation du modèle.

Inspiré de: Reseau_Neural_2D_Gap_Lecran_25_06_25/src/data_augmentation_2D.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
import random
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataAugmentation:
    """
    Classe d'augmentation de données avancée avec support pour:
    - Interpolation 2D des profils d'intensité
    - Augmentation cohérente des labels (gap, L_ecran)
    - Transformations physiquement réalistes
    - Génération de variations contrôlées
    """
    
    def __init__(self, augmentation_factor: int = 3, noise_level: float = 0.02):
        """
        Initialise le module d'augmentation
        
        Args:
            augmentation_factor: Facteur de multiplication des données (3-5 recommandé)
            noise_level: Niveau de bruit gaussien à ajouter (0.01-0.05)
        """
        self.augmentation_factor = augmentation_factor
        self.noise_level = noise_level
        
        print(f"🔧 AdvancedDataAugmentation initialisé")
        print(f"   📈 Facteur d'augmentation: {augmentation_factor}x")
        print(f"   🔊 Niveau de bruit: {noise_level:.3f}")
    
    def interpolate_profile_2d(self, x_original: np.ndarray, intensity_original: np.ndarray,
                              gap_original: float, L_ecran_original: float,
                              n_variations: int = None) -> Tuple[List[np.ndarray], List[float], List[float]]:
        """
        Génère des variations d'un profil par interpolation 2D
        
        Args:
            x_original: Vecteur des rayons original
            intensity_original: Profil d'intensité original
            gap_original: Valeur gap originale
            L_ecran_original: Valeur L_ecran originale
            n_variations: Nombre de variations à générer
            
        Returns:
            Tuple (profils_augmentés, gaps_augmentés, L_ecrans_augmentés)
        """
        if n_variations is None:
            n_variations = self.augmentation_factor
        
        # Listes pour stocker les résultats
        augmented_profiles = []
        augmented_gaps = []
        augmented_L_ecrans = []
        
        # Paramètres de variation
        gap_variation_range = 0.05 * gap_original  # ±5% de variation
        L_ecran_variation_range = 0.02 * L_ecran_original  # ±2% de variation
        
        for i in range(n_variations):
            # Générer des variations des paramètres
            gap_variation = np.random.uniform(-gap_variation_range, gap_variation_range)
            L_ecran_variation = np.random.uniform(-L_ecran_variation_range, L_ecran_variation_range)
            
            new_gap = gap_original + gap_variation
            new_L_ecran = L_ecran_original + L_ecran_variation
            
            # S'assurer que les valeurs restent physiquement réalistes
            new_gap = max(0.001, new_gap)  # Gap minimum 1 µm
            new_L_ecran = max(5.0, new_L_ecran)  # L_ecran minimum 5 µm
            
            # Générer une variation du profil d'intensité
            # Méthode 1: Interpolation avec légère déformation
            x_deformed = self._apply_radial_deformation(x_original, deformation_strength=0.01)
            
            # Interpoler le profil sur la nouvelle grille
            interpolator = interp1d(x_original, intensity_original, kind='cubic', 
                                  bounds_error=False, fill_value='extrapolate')
            intensity_deformed = interpolator(x_deformed)
            
            # Méthode 2: Ajouter des variations d'intensité cohérentes
            intensity_augmented = self._apply_intensity_variations(intensity_deformed, 
                                                                 gap_ratio=new_gap/gap_original,
                                                                 L_ecran_ratio=new_L_ecran/L_ecran_original)
            
            # Ajouter du bruit gaussien
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level * np.std(intensity_augmented), 
                                       len(intensity_augmented))
                intensity_augmented += noise
            
            # Lisser légèrement pour maintenir la cohérence physique
            intensity_augmented = gaussian_filter1d(intensity_augmented, sigma=0.5)
            
            # Stocker les résultats
            augmented_profiles.append(intensity_augmented)
            augmented_gaps.append(new_gap)
            augmented_L_ecrans.append(new_L_ecran)
        
        return augmented_profiles, augmented_gaps, augmented_L_ecrans
    
    def _apply_radial_deformation(self, x: np.ndarray, deformation_strength: float = 0.01) -> np.ndarray:
        """
        Applique une déformation radiale subtile pour simuler des variations expérimentales
        
        Args:
            x: Vecteur des rayons
            deformation_strength: Force de la déformation (0.005-0.02)
            
        Returns:
            Vecteur des rayons déformé
        """
        # Générer une déformation sinusoïdale subtile
        n_points = len(x)
        deformation = deformation_strength * np.sin(np.linspace(0, 4*np.pi, n_points))
        deformation += deformation_strength * 0.5 * np.sin(np.linspace(0, 8*np.pi, n_points))
        
        # Appliquer la déformation
        x_deformed = x * (1 + deformation)
        
        return x_deformed
    
    def _apply_intensity_variations(self, intensity: np.ndarray, gap_ratio: float, 
                                  L_ecran_ratio: float) -> np.ndarray:
        """
        Applique des variations d'intensité cohérentes avec les changements de paramètres
        
        Args:
            intensity: Profil d'intensité original
            gap_ratio: Ratio du nouveau gap par rapport à l'original
            L_ecran_ratio: Ratio du nouveau L_ecran par rapport à l'original
            
        Returns:
            Profil d'intensité modifié
        """
        # Facteur d'échelle global basé sur les paramètres physiques
        # Gap plus petit -> intensité généralement plus élevée
        gap_factor = 1.0 + 0.1 * (1.0 - gap_ratio)
        
        # L_ecran plus grand -> légère atténuation
        L_ecran_factor = 1.0 - 0.05 * (L_ecran_ratio - 1.0)
        
        # Appliquer les facteurs
        intensity_modified = intensity * gap_factor * L_ecran_factor
        
        # Ajouter des variations locales subtiles
        n_points = len(intensity)
        local_variations = 0.02 * np.sin(np.linspace(0, 6*np.pi, n_points))
        local_variations += 0.01 * np.random.normal(0, 1, n_points)
        
        intensity_modified *= (1 + local_variations)
        
        return intensity_modified
    
    def augment_dataset(self, X_data: np.ndarray, y_data: np.ndarray, 
                       x_radial: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augmente un dataset complet
        
        Args:
            X_data: Profils d'intensité (N, n_points)
            y_data: Labels [gap, L_ecran] (N, 2)
            x_radial: Vecteur des rayons
            
        Returns:
            Tuple (X_augmented, y_augmented) avec données originales + augmentées
        """
        print(f"🔄 Augmentation du dataset...")
        print(f"   📊 Dataset original: {X_data.shape[0]} échantillons")
        
        # Listes pour stocker toutes les données
        all_X = [X_data]  # Commencer avec les données originales
        all_y = [y_data]
        
        # Augmenter chaque échantillon
        for i in range(X_data.shape[0]):
            if i % 1000 == 0:
                print(f"   📈 Progression: {i}/{X_data.shape[0]} échantillons traités")
            
            # Extraire l'échantillon actuel
            intensity_original = X_data[i]
            gap_original = y_data[i, 0]
            L_ecran_original = y_data[i, 1]
            
            # Générer les variations
            try:
                augmented_profiles, augmented_gaps, augmented_L_ecrans = self.interpolate_profile_2d(
                    x_radial, intensity_original, gap_original, L_ecran_original
                )
                
                # Convertir en arrays
                X_aug = np.array(augmented_profiles)
                y_aug = np.column_stack([augmented_gaps, augmented_L_ecrans])
                
                # Ajouter aux listes
                all_X.append(X_aug)
                all_y.append(y_aug)
                
            except Exception as e:
                print(f"⚠️ Erreur lors de l'augmentation de l'échantillon {i}: {e}")
                continue
        
        # Concaténer toutes les données
        X_final = np.vstack(all_X)
        y_final = np.vstack(all_y)
        
        print(f"✅ Augmentation terminée:")
        print(f"   📊 Dataset final: {X_final.shape[0]} échantillons")
        print(f"   📈 Facteur d'augmentation réel: {X_final.shape[0]/X_data.shape[0]:.1f}x")
        print(f"   📏 Forme finale X: {X_final.shape}")
        print(f"   📏 Forme finale y: {y_final.shape}")
        
        return X_final, y_final
    
    def visualize_augmentation_examples(self, x_radial: np.ndarray, 
                                      intensity_original: np.ndarray,
                                      gap_original: float, L_ecran_original: float,
                                      n_examples: int = 5, save_path: str = None):
        """
        Visualise des exemples d'augmentation pour validation
        
        Args:
            x_radial: Vecteur des rayons
            intensity_original: Profil d'intensité original
            gap_original: Gap original
            L_ecran_original: L_ecran original
            n_examples: Nombre d'exemples à générer
            save_path: Chemin de sauvegarde du graphique
        """
        print(f"📊 Génération d'exemples d'augmentation...")
        
        # Générer les variations
        augmented_profiles, augmented_gaps, augmented_L_ecrans = self.interpolate_profile_2d(
            x_radial, intensity_original, gap_original, L_ecran_original, n_examples
        )
        
        # Créer la visualisation
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Profil original
        axes[0, 0].plot(x_radial, intensity_original, 'b-', linewidth=2, label='Original')
        axes[0, 0].set_title(f'Original\nGap: {gap_original:.4f}µm, L_écran: {L_ecran_original:.3f}µm')
        axes[0, 0].set_xlabel('Rayon (µm)')
        axes[0, 0].set_ylabel('Intensité')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Exemples augmentés
        for i in range(min(5, n_examples)):
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            axes[row, col].plot(x_radial, intensity_original, 'b--', alpha=0.5, label='Original')
            axes[row, col].plot(x_radial, augmented_profiles[i], 'r-', linewidth=2, label='Augmenté')
            axes[row, col].set_title(f'Variation {i+1}\nGap: {augmented_gaps[i]:.4f}µm, L_écran: {augmented_L_ecrans[i]:.3f}µm')
            axes[row, col].set_xlabel('Rayon (µm)')
            axes[row, col].set_ylabel('Intensité')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()
        
        plt.suptitle('Exemples d\'Augmentation de Données', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def create_augmented_dataset_with_validation(self, X_train: np.ndarray, y_train: np.ndarray,
                                               x_radial: np.ndarray, 
                                               validation_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crée un dataset augmenté avec validation visuelle
        
        Args:
            X_train: Données d'entraînement
            y_train: Labels d'entraînement
            x_radial: Vecteur des rayons
            validation_samples: Nombre d'échantillons à visualiser
            
        Returns:
            Tuple (X_augmented, y_augmented)
        """
        print("🔍 Création du dataset augmenté avec validation...")
        
        # Visualiser quelques exemples avant augmentation
        for i in range(min(validation_samples, X_train.shape[0])):
            self.visualize_augmentation_examples(
                x_radial, X_train[i], y_train[i, 0], y_train[i, 1],
                n_examples=3, save_path=f"augmentation_example_{i+1}.png"
            )
        
        # Effectuer l'augmentation complète
        X_augmented, y_augmented = self.augment_dataset(X_train, y_train, x_radial)
        
        return X_augmented, y_augmented

def main():
    """Fonction principale de démonstration"""
    print("🧠 AdvancedDataAugmentation - Démonstration")
    print("=" * 50)
    
    # Simuler des données pour test
    n_points = 601
    x_radial = np.linspace(1.384585, 5.538338, n_points)
    
    # Créer un profil d'intensité synthétique
    intensity_test = np.exp(-0.5 * ((x_radial - 3.0) / 0.5)**2) + 0.3 * np.sin(5 * x_radial)
    gap_test = 0.15
    L_ecran_test = 10.0
    
    # Initialiser l'augmenteur
    augmenter = AdvancedDataAugmentation(augmentation_factor=4, noise_level=0.02)
    
    # Tester l'augmentation sur un échantillon
    augmenter.visualize_augmentation_examples(
        x_radial, intensity_test, gap_test, L_ecran_test,
        n_examples=5, save_path="demo_augmentation.png"
    )
    
    print("\n✅ Démonstration d'augmentation terminée!")
    print("📁 Fichier généré: demo_augmentation.png")

if __name__ == "__main__":
    main()
