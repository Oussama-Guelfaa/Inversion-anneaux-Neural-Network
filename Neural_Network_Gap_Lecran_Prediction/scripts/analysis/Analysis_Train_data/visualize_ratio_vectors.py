#!/usr/bin/env python3
"""
Script pour visualiser les vecteurs des ratios superposés à partir d'échantillons aléatoires
du dataset de fichiers .mat
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import random
import glob
from pathlib import Path

def load_mat_file(filepath):
    """Charge un fichier .mat et retourne les données de ratio"""
    try:
        data = scipy.io.loadmat(filepath)
        # Récupérer le vecteur des ratios
        if 'ratio' in data:
            ratio_data = data['ratio']
            # Convertir en vecteur 1D
            if len(ratio_data.shape) > 1:
                ratio_data = ratio_data.flatten()
            return ratio_data
        return None
    except Exception as e:
        print(f"Erreur lors du chargement de {filepath}: {e}")
        return None

def extract_parameters_from_filename(filename):
    """Extrait les paramètres gap et L du nom de fichier"""
    try:
        # Format: gap_X.XXXXum_L_Y.YYYum.mat
        parts = filename.replace('.mat', '').split('_')
        gap_str = parts[1].replace('um', '')
        l_str = parts[3].replace('um', '')
        gap = float(gap_str)
        length = float(l_str)
        return gap, length
    except:
        return None, None

def validate_ratios(ratio_data):
    """Valide et nettoie les données de ratio"""
    if ratio_data is None or ratio_data.size == 0:
        return None

    # Vérifier que les données sont valides (pas de NaN ou Inf)
    if np.any(np.isnan(ratio_data)) or np.any(np.isinf(ratio_data)):
        # Remplacer les valeurs invalides par la médiane
        valid_mask = np.isfinite(ratio_data)
        if np.any(valid_mask):
            median_val = np.median(ratio_data[valid_mask])
            ratio_data = np.where(valid_mask, ratio_data, median_val)
        else:
            return None

    return ratio_data

def plot_overlaid_ratio_vectors(dataset_path, num_samples=10, figsize=(12, 8)):
    """
    Trace les vecteurs des ratios superposés pour des échantillons aléatoires
    
    Args:
        dataset_path: Chemin vers le dossier contenant les fichiers .mat
        num_samples: Nombre d'échantillons aléatoires à tracer
        figsize: Taille de la figure
    """
    
    # Trouver tous les fichiers .mat
    mat_files = glob.glob(os.path.join(dataset_path, "*.mat"))
    mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
    
    if len(mat_files) == 0:
        print("Aucun fichier .mat trouvé dans le dossier spécifié")
        return
    
    print(f"Trouvé {len(mat_files)} fichiers .mat")
    
    # Sélectionner des échantillons aléatoires
    if len(mat_files) < num_samples:
        selected_files = mat_files
        print(f"Utilisation de tous les {len(mat_files)} fichiers disponibles")
    else:
        selected_files = random.sample(mat_files, num_samples)
        print(f"Sélection de {num_samples} échantillons aléatoires")
    
    # Créer la figure
    plt.figure(figsize=figsize)
    
    # Couleurs pour différencier les échantillons
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_files)))
    
    valid_samples = 0
    
    for i, filepath in enumerate(selected_files):
        filename = os.path.basename(filepath)
        gap, length = extract_parameters_from_filename(filename)
        
        # Charger les données de ratio
        ratio_data = load_mat_file(filepath)

        if ratio_data is not None:
            # Valider les ratios
            ratios = validate_ratios(ratio_data)

            if ratios is not None and len(ratios) > 0:
                # Tracer le vecteur des ratios
                x_values = np.arange(len(ratios))
                
                label = f"Gap={gap:.4f}μm, L={length:.3f}μm" if gap is not None else f"Échantillon {i+1}"
                plt.plot(x_values, ratios, 
                        color=colors[i], 
                        alpha=0.7, 
                        linewidth=1.5,
                        label=label)
                
                valid_samples += 1
                print(f"✓ Traçé: {filename}")
            else:
                print(f"✗ Impossible de calculer les ratios pour: {filename}")
        else:
            print(f"✗ Impossible de charger: {filename}")
    
    if valid_samples == 0:
        print("Aucun échantillon valide trouvé pour le tracé")
        return 0
    
    # Configuration du graphique
    plt.title(f'Vecteurs des Ratios Superposés\n({valid_samples} échantillons aléatoires)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Index du Ratio', fontsize=12)
    plt.ylabel('Valeur du Ratio', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Légende (limiter le nombre d'entrées si trop nombreuses)
    if valid_samples <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        plt.legend().set_visible(False)
        plt.text(0.02, 0.98, f'{valid_samples} échantillons tracés', 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    output_path = os.path.join(dataset_path, 'ratio_vectors_overlay.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé: {output_path}")
    
    plt.show()
    
    return valid_samples

def main():
    """Fonction principale"""
    # Chemin vers le dataset
    dataset_path = "."  # Dossier courant (dataset 2)
    
    print("=== Visualisation des Vecteurs des Ratios Superposés ===\n")
    
    # Tracer les vecteurs des ratios pour des échantillons aléatoires
    num_samples = 15  # Nombre d'échantillons à tracer
    
    valid_samples = plot_overlaid_ratio_vectors(
        dataset_path=dataset_path,
        num_samples=num_samples,
        figsize=(14, 8)
    )
    
    if valid_samples > 0:
        print(f"\n✓ Visualisation terminée avec succès!")
        print(f"  - {valid_samples} échantillons tracés")
        print(f"  - Graphique sauvegardé dans le dossier dataset")
    else:
        print("\n✗ Aucune visualisation générée")

if __name__ == "__main__":
    main()
