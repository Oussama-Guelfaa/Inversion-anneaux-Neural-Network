#!/usr/bin/env python3
"""
Analyse des différences entre profils d'intensité

Auteur: Oussama GUELFAA
Date: 25/06/2025

Analyse quantitative des différences entre les 4 profils d'intensité tracés.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_intensity_profile(dataset_path, filename, truncate_to=600):
    """Charge un profil d'intensité depuis un fichier .mat."""
    mat_file_path = Path(dataset_path) / filename
    
    try:
        data = loadmat(str(mat_file_path))
        
        if 'ratio' in data:
            ratio = data['ratio'].flatten()
        elif 'I_ratio' in data:
            ratio = data['I_ratio'].flatten()
        else:
            possible_keys = [k for k in data.keys() if not k.startswith('__')]
            if possible_keys:
                ratio = data[possible_keys[0]].flatten()
            else:
                raise ValueError(f"Aucune donnée trouvée dans {filename}")
        
        if len(ratio) > truncate_to:
            ratio = ratio[:truncate_to]
        elif len(ratio) < truncate_to:
            ratio = np.pad(ratio, (0, truncate_to - len(ratio)), 'edge')
        
        return ratio
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {filename}: {e}")
        return None

def analyze_profile_differences():
    """Analyse les différences entre les 4 profils d'intensité."""
    logger.info("🔍 ANALYSE DES DIFFÉRENCES ENTRE PROFILS")
    logger.info("="*50)
    
    dataset_path = "../../data_generation/dataset_2D_Train_Augmented"
    
    # Définir les 4 profils à analyser
    profiles = [
        {'filename': 'gap_0.0050um_L_4.000um.mat', 'gap': 0.0050, 'L_ecran': 4.0, 'name': 'Petit gap, petit L_écran'},
        {'filename': 'gap_0.0050um_L_7.000um.mat', 'gap': 0.0050, 'L_ecran': 7.0, 'name': 'Petit gap, grand L_écran'},
        {'filename': 'gap_0.3000um_L_4.000um.mat', 'gap': 0.3000, 'L_ecran': 4.0, 'name': 'Grand gap, petit L_écran'},
        {'filename': 'gap_0.3000um_L_7.000um.mat', 'gap': 0.3000, 'L_ecran': 7.0, 'name': 'Grand gap, grand L_écran'}
    ]
    
    # Charger tous les profils
    intensity_profiles = []
    for profile in profiles:
        intensity = load_intensity_profile(dataset_path, profile['filename'])
        if intensity is not None:
            intensity_profiles.append(intensity)
            profile['intensity'] = intensity
        else:
            logger.error(f"❌ Impossible de charger {profile['filename']}")
            return
    
    logger.info(f"✅ {len(intensity_profiles)} profils chargés")
    
    # Calculer les statistiques pour chaque profil
    logger.info(f"\n📊 STATISTIQUES PAR PROFIL:")
    for i, profile in enumerate(profiles):
        intensity = profile['intensity']
        mean_val = np.mean(intensity)
        std_val = np.std(intensity)
        min_val = np.min(intensity)
        max_val = np.max(intensity)
        median_val = np.median(intensity)
        
        logger.info(f"\n{i+1}. {profile['name']}")
        logger.info(f"   Gap: {profile['gap']:.4f}µm, L_écran: {profile['L_ecran']:.1f}µm")
        logger.info(f"   Moyenne: {mean_val:.3f}")
        logger.info(f"   Écart-type: {std_val:.3f}")
        logger.info(f"   Médiane: {median_val:.3f}")
        logger.info(f"   Min-Max: {min_val:.3f} - {max_val:.3f}")
        logger.info(f"   Amplitude: {max_val - min_val:.3f}")
    
    # Analyser les différences entre profils
    logger.info(f"\n🔍 ANALYSE DES DIFFÉRENCES:")
    
    # Comparaisons par paires
    comparisons = [
        (0, 1, "Effet L_écran (gap constant 0.005µm)"),
        (2, 3, "Effet L_écran (gap constant 0.300µm)"),
        (0, 2, "Effet gap (L_écran constant 4.0µm)"),
        (1, 3, "Effet gap (L_écran constant 7.0µm)")
    ]
    
    for idx1, idx2, description in comparisons:
        profile1 = profiles[idx1]
        profile2 = profiles[idx2]
        intensity1 = profile1['intensity']
        intensity2 = profile2['intensity']
        
        # Calculer les différences
        diff = intensity2 - intensity1
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        max_abs_diff = np.max(np.abs(diff))
        
        # Corrélation
        correlation = np.corrcoef(intensity1, intensity2)[0, 1]
        
        # Distance euclidienne normalisée
        euclidean_dist = np.sqrt(np.sum(diff**2)) / len(diff)
        
        logger.info(f"\n📈 {description}:")
        logger.info(f"   Profil 1: {profile1['name']}")
        logger.info(f"   Profil 2: {profile2['name']}")
        logger.info(f"   Différence moyenne: {mean_diff:+.3f}")
        logger.info(f"   Écart-type différence: {std_diff:.3f}")
        logger.info(f"   Différence max absolue: {max_abs_diff:.3f}")
        logger.info(f"   Corrélation: {correlation:.3f}")
        logger.info(f"   Distance euclidienne: {euclidean_dist:.3f}")
    
    # Créer une matrice de corrélation
    logger.info(f"\n📊 MATRICE DE CORRÉLATION:")
    correlation_matrix = np.zeros((4, 4))
    
    for i in range(4):
        for j in range(4):
            if i == j:
                correlation_matrix[i, j] = 1.0
            else:
                corr = np.corrcoef(profiles[i]['intensity'], profiles[j]['intensity'])[0, 1]
                correlation_matrix[i, j] = corr
    
    # Afficher la matrice
    logger.info(f"   {'':25} {'P1':>8} {'P2':>8} {'P3':>8} {'P4':>8}")
    for i, profile in enumerate(profiles):
        name_short = f"P{i+1} ({profile['gap']:.3f}µm, {profile['L_ecran']:.1f}µm)"
        logger.info(f"   {name_short:25} {correlation_matrix[i, 0]:8.3f} {correlation_matrix[i, 1]:8.3f} {correlation_matrix[i, 2]:8.3f} {correlation_matrix[i, 3]:8.3f}")
    
    # Analyser les tendances
    logger.info(f"\n🎯 OBSERVATIONS CLÉS:")
    
    # Effet du gap
    gap_effect_L4 = np.mean(profiles[2]['intensity']) - np.mean(profiles[0]['intensity'])
    gap_effect_L7 = np.mean(profiles[3]['intensity']) - np.mean(profiles[1]['intensity'])
    
    logger.info(f"   Effet du gap (0.005→0.300µm):")
    logger.info(f"     À L_écran=4.0µm: {gap_effect_L4:+.3f} (moyenne)")
    logger.info(f"     À L_écran=7.0µm: {gap_effect_L7:+.3f} (moyenne)")
    
    # Effet de L_écran
    L_effect_gap005 = np.mean(profiles[1]['intensity']) - np.mean(profiles[0]['intensity'])
    L_effect_gap300 = np.mean(profiles[3]['intensity']) - np.mean(profiles[2]['intensity'])
    
    logger.info(f"   Effet de L_écran (4.0→7.0µm):")
    logger.info(f"     À gap=0.005µm: {L_effect_gap005:+.3f} (moyenne)")
    logger.info(f"     À gap=0.300µm: {L_effect_gap300:+.3f} (moyenne)")
    
    # Variabilité
    std_values = [np.std(profile['intensity']) for profile in profiles]
    logger.info(f"   Variabilité (écart-type):")
    logger.info(f"     Min: {min(std_values):.3f}, Max: {max(std_values):.3f}")
    logger.info(f"     Différence: {max(std_values) - min(std_values):.3f}")
    
    # Corrélations moyennes
    off_diag_corr = []
    for i in range(4):
        for j in range(i+1, 4):
            off_diag_corr.append(correlation_matrix[i, j])
    
    mean_correlation = np.mean(off_diag_corr)
    logger.info(f"   Corrélation moyenne entre profils: {mean_correlation:.3f}")
    
    if mean_correlation > 0.9:
        logger.info(f"     → Profils très similaires (défi pour le réseau)")
    elif mean_correlation > 0.7:
        logger.info(f"     → Profils modérément similaires")
    else:
        logger.info(f"     → Profils bien distincts")
    
    return profiles, correlation_matrix

def main():
    """Fonction principale."""
    logger.info("🚀 ANALYSE DES DIFFÉRENCES ENTRE PROFILS D'INTENSITÉ")
    logger.info("="*60)
    
    profiles, correlation_matrix = analyze_profile_differences()
    
    logger.info(f"\n✅ ANALYSE TERMINÉE")
    logger.info(f"   4 profils analysés")
    logger.info(f"   Différences quantifiées")
    logger.info(f"   Matrice de corrélation calculée")

if __name__ == "__main__":
    main()
