#!/usr/bin/env python3
"""
Tracé de profils d'intensité du dataset_2D_Train

Auteur: Oussama GUELFAA
Date: 25/06/2025

Ce script trace 4 profils d'intensité représentatifs du dataset d'entraînement,
tronqués à 600 points comme utilisé dans le réseau de neurones.
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
    """
    Charge un profil d'intensité depuis un fichier .mat.
    
    Args:
        dataset_path: Chemin vers le dataset
        filename: Nom du fichier .mat
        truncate_to: Nombre de points pour la troncature
    
    Returns:
        array: Profil d'intensité tronqué
    """
    mat_file_path = Path(dataset_path) / filename
    
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
                raise ValueError(f"Aucune donnée trouvée dans {filename}")
        
        # Tronquer à la taille désirée
        if len(ratio) > truncate_to:
            ratio = ratio[:truncate_to]
        elif len(ratio) < truncate_to:
            ratio = np.pad(ratio, (0, truncate_to - len(ratio)), 'edge')
        
        return ratio
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {filename}: {e}")
        return None

def select_representative_profiles(dataset_path):
    """
    Sélectionne 4 profils représentatifs avec différents paramètres.
    
    Args:
        dataset_path: Chemin vers le dataset
    
    Returns:
        list: Liste de dictionnaires avec les informations des profils
    """
    # Charger le fichier labels.csv
    labels_path = Path(dataset_path) / "labels.csv"
    labels_df = pd.read_csv(labels_path)
    
    logger.info(f"📊 Dataset chargé: {len(labels_df)} échantillons")
    logger.info(f"   Gap range: {labels_df['gap_um'].min():.4f} - {labels_df['gap_um'].max():.4f} µm")
    logger.info(f"   L_ecran range: {labels_df['L_um'].min():.1f} - {labels_df['L_um'].max():.1f} µm")
    
    # Sélectionner 4 profils représentatifs avec des paramètres variés
    selected_profiles = []
    
    # 1. Petit gap, petit L_écran
    mask1 = (labels_df['gap_um'] <= 0.1) & (labels_df['L_um'] <= 5.0)
    if len(labels_df[mask1]) > 0:
        sample1 = labels_df[mask1].iloc[0]
        selected_profiles.append({
            'filename': sample1['filename'],
            'gap': sample1['gap_um'],
            'L_ecran': sample1['L_um'],
            'description': 'Petit gap, petit L_écran'
        })
    
    # 2. Petit gap, grand L_écran
    mask2 = (labels_df['gap_um'] <= 0.1) & (labels_df['L_um'] >= 7.0)
    if len(labels_df[mask2]) > 0:
        sample2 = labels_df[mask2].iloc[0]
        selected_profiles.append({
            'filename': sample2['filename'],
            'gap': sample2['gap_um'],
            'L_ecran': sample2['L_um'],
            'description': 'Petit gap, grand L_écran'
        })
    
    # 3. Grand gap, petit L_écran
    mask3 = (labels_df['gap_um'] >= 0.3) & (labels_df['L_um'] <= 5.0)
    if len(labels_df[mask3]) > 0:
        sample3 = labels_df[mask3].iloc[0]
        selected_profiles.append({
            'filename': sample3['filename'],
            'gap': sample3['gap_um'],
            'L_ecran': sample3['L_um'],
            'description': 'Grand gap, petit L_écran'
        })
    
    # 4. Grand gap, grand L_écran
    mask4 = (labels_df['gap_um'] >= 0.3) & (labels_df['L_um'] >= 7.0)
    if len(labels_df[mask4]) > 0:
        sample4 = labels_df[mask4].iloc[0]
        selected_profiles.append({
            'filename': sample4['filename'],
            'gap': sample4['gap_um'],
            'L_ecran': sample4['L_um'],
            'description': 'Grand gap, grand L_écran'
        })
    
    # Si on n'a pas 4 profils, compléter avec des échantillons aléatoires
    while len(selected_profiles) < 4:
        random_sample = labels_df.sample(1).iloc[0]
        selected_profiles.append({
            'filename': random_sample['filename'],
            'gap': random_sample['gap_um'],
            'L_ecran': random_sample['L_um'],
            'description': 'Échantillon aléatoire'
        })
    
    return selected_profiles[:4]  # S'assurer qu'on a exactement 4 profils

def plot_intensity_profiles(dataset_path, output_path="../plots/intensity_profiles_600pts.png"):
    """
    Trace 4 profils d'intensité représentatifs.
    
    Args:
        dataset_path: Chemin vers le dataset
        output_path: Chemin de sauvegarde du graphique
    """
    logger.info("📊 TRACÉ DES PROFILS D'INTENSITÉ")
    logger.info("="*50)
    
    # Sélectionner les profils représentatifs
    selected_profiles = select_representative_profiles(dataset_path)
    
    # Créer la figure avec 4 sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Couleurs pour chaque profil
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, profile_info in enumerate(selected_profiles):
        logger.info(f"\n📋 Profil {i+1}: {profile_info['filename']}")
        logger.info(f"   Gap: {profile_info['gap']:.4f} µm")
        logger.info(f"   L_écran: {profile_info['L_ecran']:.1f} µm")
        logger.info(f"   Description: {profile_info['description']}")
        
        # Charger le profil d'intensité
        intensity_profile = load_intensity_profile(dataset_path, profile_info['filename'])
        
        if intensity_profile is not None:
            # Créer l'axe des x (indices des points)
            x_axis = np.arange(len(intensity_profile))
            
            # Tracer le profil
            axes[i].plot(x_axis, intensity_profile, color=colors[i], linewidth=2, alpha=0.8)
            
            # Configuration du graphique
            axes[i].set_title(f"Profil {i+1}: {profile_info['description']}\n"
                             f"Gap = {profile_info['gap']:.4f} µm, L_écran = {profile_info['L_ecran']:.1f} µm",
                             fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Index du point', fontsize=10)
            axes[i].set_ylabel('Intensité (ratio)', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            
            # Ajouter des statistiques sur le graphique
            mean_intensity = np.mean(intensity_profile)
            std_intensity = np.std(intensity_profile)
            min_intensity = np.min(intensity_profile)
            max_intensity = np.max(intensity_profile)
            
            # Texte avec les statistiques
            stats_text = f"Moyenne: {mean_intensity:.3f}\n"
            stats_text += f"Écart-type: {std_intensity:.3f}\n"
            stats_text += f"Min-Max: {min_intensity:.3f} - {max_intensity:.3f}"
            
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Marquer les points caractéristiques
            axes[i].axhline(y=mean_intensity, color='red', linestyle='--', alpha=0.7, label='Moyenne')
            axes[i].legend(fontsize=8)
            
            logger.info(f"   ✅ Profil tracé: {len(intensity_profile)} points")
            logger.info(f"   Statistiques: Moyenne={mean_intensity:.3f}, Écart-type={std_intensity:.3f}")
        else:
            # En cas d'erreur, afficher un message
            axes[i].text(0.5, 0.5, f"Erreur de chargement\n{profile_info['filename']}", 
                        transform=axes[i].transAxes, ha='center', va='center',
                        fontsize=12, color='red')
            axes[i].set_title(f"Profil {i+1}: Erreur", fontsize=12)
            logger.error(f"   ❌ Impossible de charger le profil")
    
    # Configuration générale de la figure
    plt.suptitle('Profils d\'Intensité Représentatifs du Dataset (600 points)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Ajuster l'espacement
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Créer le dossier de sortie si nécessaire
    Path(output_path).parent.mkdir(exist_ok=True)
    
    # Sauvegarder le graphique
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"\n💾 Graphique sauvegardé: {output_path}")
    
    return selected_profiles

def main():
    """Fonction principale."""
    logger.info("🚀 TRACÉ DES PROFILS D'INTENSITÉ DU DATASET")
    logger.info("="*60)
    
    # Chemin vers le dataset d'entraînement
    dataset_path = "../../data_generation/dataset_2D_Train_Augmented"
    
    # Vérifier que le dataset existe
    if not Path(dataset_path).exists():
        logger.error(f"❌ Dataset non trouvé: {dataset_path}")
        return
    
    # Tracer les profils
    selected_profiles = plot_intensity_profiles(dataset_path)
    
    logger.info(f"\n✅ TRACÉ TERMINÉ")
    logger.info(f"   4 profils représentatifs tracés")
    logger.info(f"   Tronqués à 600 points (format réseau de neurones)")
    logger.info(f"   Graphique sauvegardé: ../plots/intensity_profiles_600pts.png")
    
    # Résumé des profils sélectionnés
    logger.info(f"\n📋 PROFILS SÉLECTIONNÉS:")
    for i, profile in enumerate(selected_profiles):
        logger.info(f"   {i+1}. {profile['filename']}")
        logger.info(f"      Gap: {profile['gap']:.4f} µm, L_écran: {profile['L_ecran']:.1f} µm")
        logger.info(f"      Type: {profile['description']}")

if __name__ == "__main__":
    main()
