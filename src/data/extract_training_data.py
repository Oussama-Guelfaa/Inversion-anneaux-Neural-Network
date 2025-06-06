#!/usr/bin/env python3
"""
Training Data Extractor
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Ce script extrait et organise les données du fichier all_banque_new_24_01_25_NEW_full.mat
pour l'entraînement du réseau de neurones. Il calcule le ratio I_subs/I_subs_inc
et l'associe aux paramètres correspondants (L_ecran et gap).

Variables d'intérêt dans le fichier .mat:
- L_ecran_subs_vect: Liste des valeurs L_ecran
- gap_subs_vect: Liste des valeurs gap  
- I_subs: Liste des intensités
- I_subs_inc: Liste des intensités incidentes

Sortie: Dataset organisé pour l'entraînement avec:
- X (features): Profils radiaux normalisés I_subs/I_subs_inc
- y (targets): Paramètres [L_ecran, gap]
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def inspect_mat_structure(mat_file):
    """
    Examine la structure du fichier .mat pour comprendre l'organisation des données.
    
    Args:
        mat_file (str): Chemin vers le fichier .mat
    
    Returns:
        dict: Données chargées du fichier
    """
    print(f"=== Inspection de {mat_file} ===")
    
    try:
        data = sio.loadmat(mat_file)
        
        print("\nVariables disponibles:")
        for key in data.keys():
            if not key.startswith('__'):
                value = data[key]
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                
                # Afficher quelques statistiques pour les arrays numériques
                if hasattr(value, 'flatten') and value.size > 0:
                    flat = value.flatten()
                    if np.issubdtype(flat.dtype, np.number):
                        print(f"    min/max: {flat.min():.6f} / {flat.max():.6f}")
                        print(f"    mean±std: {flat.mean():.6f} ± {flat.std():.6f}")
                        if value.size <= 10:
                            print(f"    valeurs: {flat}")
                        else:
                            print(f"    premiers/derniers: [{flat[0]:.6f}, {flat[1]:.6f}, ..., {flat[-2]:.6f}, {flat[-1]:.6f}]")
        
        return data
        
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        return None

def extract_and_organize_data(mat_file, output_dir="Neural_Network/processed_data"):
    """
    Extrait et organise les données pour l'entraînement du réseau de neurones.
    
    Args:
        mat_file (str): Chemin vers le fichier .mat
        output_dir (str): Dossier de sortie pour les données organisées
    
    Returns:
        tuple: (X, y, metadata) - Features, targets et métadonnées
    """
    print(f"\n=== Extraction des données de {mat_file} ===")
    
    # Charger les données
    data = sio.loadmat(mat_file)
    
    # Extraire les variables d'intérêt
    try:
        L_ecran_vect = data['L_ecran_subs_vect'].flatten()
        gap_vect = data['gap_sphere_vect'].flatten()  # Correction du nom de variable
        I_subs = data['I_subs']
        I_subs_inc = data['I_subs_inc']
        
        print(f"Variables extraites avec succès:")
        print(f"  L_ecran_subs_vect: {L_ecran_vect.shape} - {L_ecran_vect.dtype}")
        print(f"  gap_subs_vect: {gap_vect.shape} - {gap_vect.dtype}")
        print(f"  I_subs: {I_subs.shape} - {I_subs.dtype}")
        print(f"  I_subs_inc: {I_subs_inc.shape} - {I_subs_inc.dtype}")
        
    except KeyError as e:
        print(f"Variable manquante: {e}")
        print("Variables disponibles:", [k for k in data.keys() if not k.startswith('__')])
        return None, None, None
    
    # Analyser la structure des données
    n_L_ecran = len(L_ecran_vect)  # 33
    n_gap = len(gap_vect)          # 30
    print(f"\nDimensions détectées:")
    print(f"  L_ecran: {n_L_ecran} valeurs")
    print(f"  gap: {n_gap} valeurs")
    print(f"  Total combinaisons: {n_L_ecran * n_gap}")

    # Analyser la structure des intensités
    print(f"\nStructure des intensités:")
    print(f"  I_subs shape: {I_subs.shape}")
    print(f"  I_subs_inc shape: {I_subs_inc.shape}")

    # Calculer le ratio I_subs/I_subs_inc
    print(f"\nCalcul du ratio I_subs/I_subs_inc...")

    # Éviter la division par zéro
    I_subs_inc_safe = np.where(I_subs_inc == 0, 1e-10, I_subs_inc)
    intensity_ratio = I_subs / I_subs_inc_safe

    print(f"Ratio calculé: shape={intensity_ratio.shape}")
    print(f"  min/max: {intensity_ratio.min():.6f} / {intensity_ratio.max():.6f}")
    print(f"  mean±std: {intensity_ratio.mean():.6f} ± {intensity_ratio.std():.6f}")

    # Organiser les données 3D en format 2D pour l'entraînement
    # I_subs.shape = (33, 30, 1000) -> (33*30, 1000)
    # Chaque ligne sera un échantillon avec son profil radial de 1000 points

    if len(intensity_ratio.shape) == 3:
        n_L, n_g, n_radial = intensity_ratio.shape

        # Vérifier la cohérence
        if n_L != n_L_ecran or n_g != n_gap:
            print(f"ATTENTION: Incohérence dans les dimensions!")
            print(f"  L_ecran attendu: {n_L_ecran}, trouvé: {n_L}")
            print(f"  gap attendu: {n_gap}, trouvé: {n_g}")

        # Reshape en 2D: (n_samples, n_features)
        X = intensity_ratio.reshape(n_L * n_g, n_radial)

        # Créer les grilles de paramètres correspondantes
        L_grid, gap_grid = np.meshgrid(L_ecran_vect, gap_vect, indexing='ij')
        L_ecran_final = L_grid.flatten()
        gap_final = gap_grid.flatten()

        n_samples_final = len(L_ecran_final)

    else:
        print(f"ERREUR: Structure d'intensité non supportée: {intensity_ratio.shape}")
        print(f"Attendu: 3D (n_L_ecran, n_gap, n_radial)")
        return None, None, None
    
    # Créer les targets (paramètres à prédire)
    y = np.column_stack([L_ecran_final, gap_final])
    
    print(f"\nDonnées finales organisées:")
    print(f"  X (features): {X.shape} - Profils d'intensité normalisés")
    print(f"  y (targets): {y.shape} - [L_ecran, gap]")
    print(f"  Nombre d'échantillons final: {n_samples_final}")
    
    # Statistiques des paramètres
    print(f"\nStatistiques des paramètres:")
    print(f"  L_ecran: {L_ecran_final.min():.6f} à {L_ecran_final.max():.6f} (mean: {L_ecran_final.mean():.6f})")
    print(f"  gap: {gap_final.min():.6f} à {gap_final.max():.6f} (mean: {gap_final.mean():.6f})")
    
    # Métadonnées
    metadata = {
        'n_samples': n_samples_final,
        'n_features': X.shape[1],
        'L_ecran_range': [L_ecran_final.min(), L_ecran_final.max()],
        'gap_range': [gap_final.min(), gap_final.max()],
        'intensity_range': [X.min(), X.max()],
        'source_file': mat_file,
        'processing_date': '05-06-2025'
    }
    
    # Sauvegarder les données organisées
    os.makedirs(output_dir, exist_ok=True)
    
    # Format NumPy
    np.savez(os.path.join(output_dir, 'training_data.npz'),
             X=X, y=y, metadata=metadata)
    
    # Format CSV pour inspection
    df = pd.DataFrame(y, columns=['L_ecran', 'gap'])
    df.to_csv(os.path.join(output_dir, 'parameters.csv'), index=False)
    
    # Sauvegarder les profils d'intensité (échantillon des premiers)
    np.savetxt(os.path.join(output_dir, 'intensity_profiles_sample.csv'),
               X[:min(100, len(X))], delimiter=',', fmt='%.6f')

    # Sauvegarder TOUS les profils d'intensité
    np.savetxt(os.path.join(output_dir, 'intensity_profiles_full.csv'),
               X, delimiter=',', fmt='%.6f')
    
    print(f"\nDonnées sauvegardées dans {output_dir}:")
    print(f"  - training_data.npz: Données complètes pour l'entraînement")
    print(f"  - parameters.csv: Paramètres au format CSV")
    print(f"  - intensity_profiles_sample.csv: Échantillon des 100 premiers profils")
    print(f"  - intensity_profiles_full.csv: TOUS les {len(X)} profils d'intensité")
    
    return X, y, metadata

def visualize_data(X, y, metadata, output_dir="Neural_Network/processed_data"):
    """
    Visualise les données extraites pour validation.
    
    Args:
        X (np.ndarray): Features (profils d'intensité)
        y (np.ndarray): Targets (paramètres)
        metadata (dict): Métadonnées
        output_dir (str): Dossier de sortie
    """
    print(f"\n=== Visualisation des données ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution des paramètres L_ecran
    axes[0, 0].hist(y[:, 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('L_ecran')
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].set_title('Distribution de L_ecran')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribution des paramètres gap
    axes[0, 1].hist(y[:, 1], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_xlabel('gap')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].set_title('Distribution de gap')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Corrélation L_ecran vs gap
    axes[0, 2].scatter(y[:, 0], y[:, 1], alpha=0.6, s=1)
    axes[0, 2].set_xlabel('L_ecran')
    axes[0, 2].set_ylabel('gap')
    axes[0, 2].set_title('Corrélation L_ecran vs gap')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Échantillon de profils d'intensité
    n_profiles_to_show = min(20, X.shape[0])
    for i in range(n_profiles_to_show):
        axes[1, 0].plot(X[i], alpha=0.5, linewidth=0.8)
    axes[1, 0].set_xlabel('Point radial')
    axes[1, 0].set_ylabel('Intensité normalisée')
    axes[1, 0].set_title(f'Échantillon de {n_profiles_to_show} profils')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Statistiques des intensités
    intensity_means = X.mean(axis=1)
    intensity_stds = X.std(axis=1)
    axes[1, 1].scatter(intensity_means, intensity_stds, alpha=0.6, s=1)
    axes[1, 1].set_xlabel('Intensité moyenne')
    axes[1, 1].set_ylabel('Écart-type')
    axes[1, 1].set_title('Variabilité des profils')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Profil moyen et enveloppe
    mean_profile = X.mean(axis=0)
    std_profile = X.std(axis=0)
    x_axis = np.arange(len(mean_profile))
    
    axes[1, 2].plot(x_axis, mean_profile, 'b-', linewidth=2, label='Profil moyen')
    axes[1, 2].fill_between(x_axis, 
                           mean_profile - std_profile,
                           mean_profile + std_profile,
                           alpha=0.3, color='blue', label='±1σ')
    axes[1, 2].set_xlabel('Point radial')
    axes[1, 2].set_ylabel('Intensité normalisée')
    axes[1, 2].set_title('Profil moyen ± écart-type')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_visualization.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Afficher les statistiques finales
    print(f"\nStatistiques finales:")
    print(f"  Nombre total d'échantillons: {metadata['n_samples']}")
    print(f"  Dimension des features: {metadata['n_features']}")
    print(f"  Plage L_ecran: {metadata['L_ecran_range'][0]:.6f} à {metadata['L_ecran_range'][1]:.6f}")
    print(f"  Plage gap: {metadata['gap_range'][0]:.6f} à {metadata['gap_range'][1]:.6f}")
    print(f"  Plage intensité: {metadata['intensity_range'][0]:.6f} à {metadata['intensity_range'][1]:.6f}")

def main():
    """Fonction principale pour extraire et organiser les données."""
    
    # Chemin vers le fichier de données
    mat_file = "data_generation/all_banque_new_24_01_25_NEW_full.mat"
    
    if not os.path.exists(mat_file):
        print(f"ERREUR: Fichier non trouvé - {mat_file}")
        return
    
    # 1. Inspecter la structure
    data = inspect_mat_structure(mat_file)
    if data is None:
        return
    
    # 2. Extraire et organiser
    X, y, metadata = extract_and_organize_data(mat_file)
    if X is None:
        return
    
    # 3. Visualiser
    visualize_data(X, y, metadata)
    
    print(f"\n=== Extraction terminée avec succès ===")
    print(f"Les données sont prêtes pour l'entraînement du réseau de neurones!")

if __name__ == "__main__":
    main()
