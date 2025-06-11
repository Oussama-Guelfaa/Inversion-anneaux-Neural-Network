#!/usr/bin/env python3
"""
Inspection des données du dataset_small_particle pour le test d'overfitting

Auteur: Oussama GUELFAA
Date: 10 - 01 - 2025
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path

def inspect_mat_file(mat_path):
    """
    Inspecte un fichier .mat et affiche sa structure.
    
    Args:
        mat_path (str): Chemin vers le fichier .mat
    """
    print(f"\n=== Inspection de {os.path.basename(mat_path)} ===")
    
    try:
        data = sio.loadmat(mat_path)
        
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
        print(f"Erreur lors de la lecture: {e}")
        return None

def extract_gap_from_filename(filename):
    """
    Extrait la valeur du gap depuis le nom de fichier.
    
    Args:
        filename (str): Nom du fichier (ex: "gap_0.0050um_L_10.000um.mat")
        
    Returns:
        float: Valeur du gap en microns
    """
    try:
        # Format: gap_X.XXXXum_L_Y.YYYum.mat
        parts = filename.split('_')
        gap_part = parts[1]  # "0.0050um"
        gap_value = float(gap_part.replace('um', ''))
        return gap_value
    except Exception as e:
        print(f"Erreur extraction gap de {filename}: {e}")
        return None

def scan_dataset():
    """
    Scanne tout le dataset et affiche un résumé.
    """
    dataset_dir = "../../data_generation/dataset_small_particle"
    
    if not os.path.exists(dataset_dir):
        print(f"Erreur: Le dossier {dataset_dir} n'existe pas")
        return
    
    # Lister tous les fichiers .mat
    mat_files = [f for f in os.listdir(dataset_dir) if f.endswith('.mat') and f.startswith('gap_')]
    mat_files.sort()
    
    print(f"\n=== SCAN DU DATASET ===")
    print(f"Dossier: {dataset_dir}")
    print(f"Nombre de fichiers .mat: {len(mat_files)}")
    
    if len(mat_files) == 0:
        print("Aucun fichier .mat trouvé!")
        return
    
    # Examiner quelques fichiers
    print(f"\nExamen des premiers fichiers:")
    for i, filename in enumerate(mat_files[:3]):
        gap_value = extract_gap_from_filename(filename)
        print(f"  {i+1}. {filename} -> gap = {gap_value} µm")
        
        # Inspecter le premier fichier en détail
        if i == 0:
            mat_path = os.path.join(dataset_dir, filename)
            data = inspect_mat_file(mat_path)
    
    # Analyser la distribution des gaps
    gaps = []
    for filename in mat_files:
        gap = extract_gap_from_filename(filename)
        if gap is not None:
            gaps.append(gap)
    
    gaps = np.array(gaps)
    print(f"\n=== ANALYSE DES GAPS ===")
    print(f"Nombre de valeurs: {len(gaps)}")
    print(f"Plage: {gaps.min():.4f} - {gaps.max():.4f} µm")
    print(f"Pas moyen: {np.mean(np.diff(gaps)):.4f} µm")
    print(f"Premiers gaps: {gaps[:10]}")
    print(f"Derniers gaps: {gaps[-10:]}")
    
    return mat_files, gaps

def visualize_sample_profiles():
    """
    Visualise quelques profils d'intensité pour vérification.
    """
    dataset_dir = "../../data_generation/dataset_small_particle"
    mat_files = [f for f in os.listdir(dataset_dir) if f.endswith('.mat') and f.startswith('gap_')]
    mat_files.sort()
    
    if len(mat_files) == 0:
        print("Aucun fichier trouvé pour la visualisation")
        return
    
    # Sélectionner quelques fichiers représentatifs
    indices = [0, len(mat_files)//4, len(mat_files)//2, 3*len(mat_files)//4, -1]
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(indices):
        filename = mat_files[idx]
        gap_value = extract_gap_from_filename(filename)
        mat_path = os.path.join(dataset_dir, filename)
        
        try:
            data = sio.loadmat(mat_path)
            
            # Chercher la variable contenant le profil d'intensité
            profile = None
            for key in data.keys():
                if not key.startswith('__'):
                    value = data[key]
                    if hasattr(value, 'flatten') and value.size > 100:
                        profile = value.flatten()
                        break
            
            if profile is not None:
                plt.subplot(2, 3, i+1)
                plt.plot(profile)
                plt.title(f'Gap = {gap_value:.4f} µm\n{filename}')
                plt.xlabel('Position radiale')
                plt.ylabel('Intensité')
                plt.grid(True, alpha=0.3)
                
        except Exception as e:
            print(f"Erreur visualisation {filename}: {e}")
    
    plt.tight_layout()
    plt.savefig('../plots/sample_profiles_inspection.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Graphique sauvegardé: ../plots/sample_profiles_inspection.png")

if __name__ == "__main__":
    print("=== INSPECTION DU DATASET SMALL PARTICLE ===")
    
    # Créer le dossier plots s'il n'existe pas
    os.makedirs("../plots", exist_ok=True)
    
    # Scanner le dataset
    mat_files, gaps = scan_dataset()
    
    # Visualiser quelques profils
    if len(mat_files) > 0:
        visualize_sample_profiles()
    
    print("\n=== INSPECTION TERMINÉE ===")
