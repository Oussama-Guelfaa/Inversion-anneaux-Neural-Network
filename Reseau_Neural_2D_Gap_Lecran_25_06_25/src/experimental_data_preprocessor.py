#!/usr/bin/env python3
"""
Préprocesseur pour données expérimentales holographiques
Auteur: Oussama GUELFAA
Date: Juillet 2025

Module de prétraitement pour harmoniser les données expérimentales
avec les caractéristiques des données simulées d'entraînement.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentalDataPreprocessor:
    """
    Classe pour préprocesser les données expérimentales holographiques
    afin de les rendre compatibles avec les modèles entraînés sur simulations.
    """
    
    def __init__(self):
        # Paramètres cibles basés sur l'analyse des données simulées
        self.target_stats = {
            'mean': 0.898,
            'std': 0.360,
            'min': 0.057,
            'max': 1.625,
            'oscillation_variance': 0.000074
        }
        
        # Paramètres de prétraitement
        self.preprocessing_params = {
            'initial_smoothing_sigma': 0.8,
            'peak_broadening_sigma': 1.2,
            'savgol_window': 11,
            'savgol_polyorder': 3,
            'decay_correction_factor': 0.1,
            'noise_reduction_factor': 0.95
        }
        
        logger.info("✅ Préprocesseur initialisé avec paramètres cibles simulés")
    
    def analyze_raw_data(self, intensities, r=None):
        """
        Analyse les caractéristiques des données brutes
        
        Args:
            intensities (np.array): Intensités expérimentales brutes
            r (np.array): Positions radiales (optionnel)
            
        Returns:
            dict: Statistiques des données brutes
        """
        stats = {
            'mean': np.mean(intensities),
            'std': np.std(intensities),
            'min': np.min(intensities),
            'max': np.max(intensities),
            'range': np.max(intensities) - np.min(intensities),
            'oscillation_variance': np.var(np.diff(intensities)) if len(intensities) > 1 else 0
        }
        
        logger.debug(f"Données brutes - Moyenne: {stats['mean']:.3f}, "
                    f"Écart-type: {stats['std']:.3f}, "
                    f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        return stats
    
    def smooth_initial_noise(self, intensities):
        """
        Lissage initial pour réduire le bruit expérimental
        
        Args:
            intensities (np.array): Intensités brutes
            
        Returns:
            np.array: Intensités lissées
        """
        # Lissage gaussien léger
        smoothed = gaussian_filter1d(intensities, 
                                   sigma=self.preprocessing_params['initial_smoothing_sigma'])
        
        logger.debug(f"Lissage initial appliqué (σ={self.preprocessing_params['initial_smoothing_sigma']})")
        return smoothed
    
    def broaden_initial_peak(self, intensities, r):
        """
        Élargit le pic initial pour correspondre aux simulations
        
        Args:
            intensities (np.array): Intensités
            r (np.array): Positions radiales
            
        Returns:
            np.array: Intensités avec pic élargi
        """
        # Identifier la région du pic initial (r < 0.5 µm)
        peak_region_mask = r < 0.5
        
        if np.sum(peak_region_mask) > 0:
            # Appliquer un lissage plus fort dans la région du pic
            peak_intensities = intensities[peak_region_mask]
            peak_smoothed = gaussian_filter1d(peak_intensities, 
                                            sigma=self.preprocessing_params['peak_broadening_sigma'])
            
            # Remplacer dans le tableau original
            intensities_broadened = intensities.copy()
            intensities_broadened[peak_region_mask] = peak_smoothed
            
            logger.debug(f"Pic initial élargi (σ={self.preprocessing_params['peak_broadening_sigma']})")
            return intensities_broadened
        
        return intensities
    
    def reduce_oscillation_chaos(self, intensities):
        """
        Réduit les oscillations chaotiques pour correspondre aux simulations
        
        Args:
            intensities (np.array): Intensités
            
        Returns:
            np.array: Intensités avec oscillations réduites
        """
        # Filtre Savitzky-Golay pour préserver les oscillations principales
        # tout en réduisant le chaos
        filtered = savgol_filter(intensities, 
                               window_length=self.preprocessing_params['savgol_window'],
                               polyorder=self.preprocessing_params['savgol_polyorder'])
        
        # Mélange avec les données originales pour préserver certaines oscillations
        alpha = self.preprocessing_params['noise_reduction_factor']
        result = alpha * filtered + (1 - alpha) * intensities
        
        logger.debug(f"Oscillations réduites (fenêtre={self.preprocessing_params['savgol_window']}, "
                    f"α={alpha})")
        return result
    
    def correct_decay_profile(self, intensities, r):
        """
        Corrige le profil de décroissance pour correspondre aux simulations
        
        Args:
            intensities (np.array): Intensités
            r (np.array): Positions radiales
            
        Returns:
            np.array: Intensités avec décroissance corrigée
        """
        # Identifier la région de décroissance (r > 1.5 µm)
        decay_region_mask = r > 1.5
        
        if np.sum(decay_region_mask) > 0:
            # Appliquer une correction exponentielle douce
            decay_factor = np.exp(-self.preprocessing_params['decay_correction_factor'] * 
                                (r - 1.5))
            decay_factor[~decay_region_mask] = 1.0  # Pas de correction avant 1.5 µm
            
            corrected = intensities * decay_factor
            logger.debug(f"Décroissance corrigée (facteur={self.preprocessing_params['decay_correction_factor']})")
            return corrected
        
        return intensities
    
    def normalize_to_simulation_range(self, intensities):
        """
        Normalise vers le range des données simulées
        
        Args:
            intensities (np.array): Intensités
            
        Returns:
            np.array: Intensités normalisées
        """
        # Normalisation min-max vers le range cible
        current_min = np.min(intensities)
        current_max = np.max(intensities)
        current_range = current_max - current_min
        
        if current_range > 0:
            # Normalisation vers [0, 1]
            normalized = (intensities - current_min) / current_range
            
            # Mise à l'échelle vers le range cible
            target_range = self.target_stats['max'] - self.target_stats['min']
            scaled = normalized * target_range + self.target_stats['min']
            
            logger.debug(f"Normalisation: [{current_min:.3f}, {current_max:.3f}] → "
                        f"[{self.target_stats['min']:.3f}, {self.target_stats['max']:.3f}]")
            return scaled
        
        return intensities
    
    def adjust_mean_and_std(self, intensities):
        """
        Ajuste la moyenne et l'écart-type pour correspondre aux simulations
        
        Args:
            intensities (np.array): Intensités
            
        Returns:
            np.array: Intensités ajustées
        """
        current_mean = np.mean(intensities)
        current_std = np.std(intensities)
        
        # Centrage
        centered = intensities - current_mean
        
        # Mise à l'échelle de l'écart-type
        if current_std > 0:
            scaled = centered * (self.target_stats['std'] / current_std)
        else:
            scaled = centered
        
        # Recentrage sur la moyenne cible
        adjusted = scaled + self.target_stats['mean']
        
        logger.debug(f"Ajustement statistique: μ={current_mean:.3f}→{self.target_stats['mean']:.3f}, "
                    f"σ={current_std:.3f}→{self.target_stats['std']:.3f}")
        
        return adjusted
    
    def clip_to_physical_range(self, intensities):
        """
        Limite les valeurs au range physiquement acceptable
        
        Args:
            intensities (np.array): Intensités
            
        Returns:
            np.array: Intensités limitées
        """
        clipped = np.clip(intensities, self.target_stats['min'], self.target_stats['max'])
        
        n_clipped = np.sum((intensities < self.target_stats['min']) | 
                          (intensities > self.target_stats['max']))
        
        if n_clipped > 0:
            logger.debug(f"Limitation appliquée: {n_clipped} valeurs limitées")
        
        return clipped
    
    def preprocess_profile(self, intensities, r, profile_name=""):
        """
        Pipeline complet de prétraitement d'un profil expérimental
        
        Args:
            intensities (np.array): Intensités expérimentales brutes
            r (np.array): Positions radiales
            profile_name (str): Nom du profil pour le logging
            
        Returns:
            np.array: Intensités prétraitées
        """
        logger.info(f"🔧 Prétraitement du profil {profile_name}")
        
        # Analyse initiale
        raw_stats = self.analyze_raw_data(intensities, r)
        
        # Pipeline de prétraitement
        processed = intensities.copy()
        
        # 1. Lissage initial du bruit
        processed = self.smooth_initial_noise(processed)
        
        # 2. Élargissement du pic initial
        processed = self.broaden_initial_peak(processed, r)
        
        # 3. Réduction des oscillations chaotiques
        processed = self.reduce_oscillation_chaos(processed)
        
        # 4. Correction du profil de décroissance
        processed = self.correct_decay_profile(processed, r)
        
        # 5. Normalisation vers le range simulé
        processed = self.normalize_to_simulation_range(processed)
        
        # 6. Ajustement de la moyenne et écart-type
        processed = self.adjust_mean_and_std(processed)
        
        # 7. Limitation finale
        processed = self.clip_to_physical_range(processed)
        
        # Analyse finale
        final_stats = self.analyze_raw_data(processed, r)
        
        logger.info(f"✅ Prétraitement terminé pour {profile_name}")
        logger.info(f"   Transformation: μ={raw_stats['mean']:.3f}→{final_stats['mean']:.3f}, "
                   f"σ={raw_stats['std']:.3f}→{final_stats['std']:.3f}")
        
        return processed
    
    def create_comparison_plot(self, original, processed, r, profile_name, save_path=None):
        """
        Crée un graphique de comparaison avant/après prétraitement
        
        Args:
            original (np.array): Intensités originales
            processed (np.array): Intensités prétraitées
            r (np.array): Positions radiales
            profile_name (str): Nom du profil
            save_path (str): Chemin de sauvegarde (optionnel)
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Graphique 1: Comparaison directe
        ax1.plot(r, original, 'r-', linewidth=2, alpha=0.7, label='Original')
        ax1.plot(r, processed, 'b-', linewidth=2, alpha=0.7, label='Prétraité')
        ax1.set_xlabel('Position radiale r (µm)', fontweight='bold')
        ax1.set_ylabel('Intensité', fontweight='bold')
        ax1.set_title(f'Comparaison Avant/Après Prétraitement - {profile_name}', 
                     fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Statistiques
        orig_stats = self.analyze_raw_data(original)
        proc_stats = self.analyze_raw_data(processed)
        
        stats_text = f"ORIGINAL: μ={orig_stats['mean']:.3f}, σ={orig_stats['std']:.3f}\n"
        stats_text += f"PRÉTRAITÉ: μ={proc_stats['mean']:.3f}, σ={proc_stats['std']:.3f}\n"
        stats_text += f"CIBLE: μ={self.target_stats['mean']:.3f}, σ={self.target_stats['std']:.3f}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        # Graphique 2: Différence
        ax2.plot(r, processed - original, 'g-', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Position radiale r (µm)', fontweight='bold')
        ax2.set_ylabel('Différence (Prétraité - Original)', fontweight='bold')
        ax2.set_title('Correction Appliquée', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Graphique 3: Histogrammes
        ax3.hist(original, bins=30, alpha=0.5, color='red', label='Original', density=True)
        ax3.hist(processed, bins=30, alpha=0.5, color='blue', label='Prétraité', density=True)
        ax3.axvline(self.target_stats['mean'], color='green', linestyle='--', 
                   linewidth=2, label='Moyenne cible')
        ax3.set_xlabel('Intensité', fontweight='bold')
        ax3.set_ylabel('Densité', fontweight='bold')
        ax3.set_title('Distribution des Intensités', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Graphique de comparaison sauvegardé: {save_path}")
        
        plt.close()
    
    def validate_preprocessing(self, processed_intensities):
        """
        Valide que le prétraitement a bien fonctionné
        
        Args:
            processed_intensities (np.array): Intensités prétraitées
            
        Returns:
            dict: Résultats de validation
        """
        stats = self.analyze_raw_data(processed_intensities)
        
        validation = {
            'mean_ok': abs(stats['mean'] - self.target_stats['mean']) < 0.05,
            'std_ok': abs(stats['std'] - self.target_stats['std']) < 0.05,
            'range_ok': (stats['min'] >= self.target_stats['min'] - 0.01 and 
                        stats['max'] <= self.target_stats['max'] + 0.01),
            'stats': stats,
            'target_stats': self.target_stats
        }
        
        validation['overall_ok'] = all([validation['mean_ok'], 
                                      validation['std_ok'], 
                                      validation['range_ok']])
        
        return validation

def test_preprocessor():
    """Fonction de test du préprocesseur"""
    logger.info("🧪 Test du préprocesseur")
    
    # Créer des données de test
    r = np.linspace(0, 4.193, 600)
    # Simuler des données expérimentales "problématiques"
    intensities = 1.5 * np.exp(-r/2) * (1 + 0.3*np.sin(10*r) + 0.1*np.random.randn(600))
    intensities[0:50] = 2.0  # Pic initial trop intense
    
    # Initialiser le préprocesseur
    preprocessor = ExperimentalDataPreprocessor()
    
    # Prétraiter
    processed = preprocessor.preprocess_profile(intensities, r, "test_profile")
    
    # Valider
    validation = preprocessor.validate_preprocessing(processed)
    
    logger.info(f"Validation: {validation['overall_ok']}")
    logger.info(f"Statistiques finales: {validation['stats']}")
    
    return preprocessor, processed, validation

if __name__ == "__main__":
    test_preprocessor()
