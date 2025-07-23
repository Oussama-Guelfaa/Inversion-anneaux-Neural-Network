#!/usr/bin/env python3
"""
Analyse avancée des vecteurs des ratios avec visualisations multiples
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import random
import glob
from pathlib import Path
import seaborn as sns

def load_mat_file(filepath):
    """Charge un fichier .mat et retourne les données de ratio et paramètres"""
    try:
        data = scipy.io.loadmat(filepath)
        
        # Récupérer les données
        ratio_data = data['ratio'].flatten() if 'ratio' in data else None
        gap = float(data['gap'][0, 0]) if 'gap' in data else None
        length = float(data['L_ecran_subs'][0, 0]) if 'L_ecran_subs' in data else None
        x_data = data['x'].flatten() if 'x' in data else None
        
        return {
            'ratio': ratio_data,
            'gap': gap,
            'length': length,
            'x': x_data
        }
    except Exception as e:
        print(f"Erreur lors du chargement de {filepath}: {e}")
        return None

def create_comprehensive_visualization(dataset_path, num_samples=20):
    """Crée une visualisation complète avec plusieurs sous-graphiques"""
    
    # Trouver tous les fichiers .mat
    mat_files = glob.glob(os.path.join(dataset_path, "*.mat"))
    mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
    
    if len(mat_files) == 0:
        print("Aucun fichier .mat trouvé")
        return
    
    # Sélectionner des échantillons aléatoires
    selected_files = random.sample(mat_files, min(num_samples, len(mat_files)))
    
    # Charger les données
    samples_data = []
    for filepath in selected_files:
        data = load_mat_file(filepath)
        if data and data['ratio'] is not None:
            samples_data.append(data)
    
    if not samples_data:
        print("Aucune donnée valide trouvée")
        return
    
    # Créer la figure avec plusieurs sous-graphiques
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Vecteurs des ratios superposés
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(samples_data)))
    
    for i, data in enumerate(samples_data):
        x_vals = np.arange(len(data['ratio']))
        plt.plot(x_vals, data['ratio'], 
                color=colors[i], alpha=0.7, linewidth=1,
                label=f"Gap={data['gap']:.3f}μm")
    
    plt.title('Vecteurs des Ratios Superposés')
    plt.xlabel('Index')
    plt.ylabel('Ratio')
    plt.grid(True, alpha=0.3)
    
    # 2. Distribution des valeurs de ratios
    ax2 = plt.subplot(2, 3, 2)
    all_ratios = np.concatenate([data['ratio'] for data in samples_data])
    plt.hist(all_ratios, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution des Valeurs de Ratios')
    plt.xlabel('Valeur du Ratio')
    plt.ylabel('Fréquence')
    plt.grid(True, alpha=0.3)
    
    # 3. Relation Gap vs Statistiques des ratios
    ax3 = plt.subplot(2, 3, 3)
    gaps = [data['gap'] for data in samples_data]
    ratio_means = [np.mean(data['ratio']) for data in samples_data]
    ratio_stds = [np.std(data['ratio']) for data in samples_data]
    
    plt.scatter(gaps, ratio_means, c='red', alpha=0.7, s=50, label='Moyenne')
    plt.scatter(gaps, ratio_stds, c='blue', alpha=0.7, s=50, label='Écart-type')
    plt.title('Gap vs Statistiques des Ratios')
    plt.xlabel('Gap (μm)')
    plt.ylabel('Valeur')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Heatmap des ratios
    ax4 = plt.subplot(2, 3, 4)
    # Créer une matrice avec les ratios (tronquer à la même longueur)
    min_length = min(len(data['ratio']) for data in samples_data)
    ratio_matrix = np.array([data['ratio'][:min_length] for data in samples_data])
    
    im = plt.imshow(ratio_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.title('Heatmap des Ratios')
    plt.xlabel('Index du Ratio')
    plt.ylabel('Échantillon')
    plt.colorbar(im, ax=ax4)
    
    # 5. Relation Longueur vs Statistiques
    ax5 = plt.subplot(2, 3, 5)
    lengths = [data['length'] for data in samples_data]
    plt.scatter(lengths, ratio_means, c=gaps, cmap='plasma', s=60, alpha=0.7)
    plt.title('Longueur vs Moyenne des Ratios')
    plt.xlabel('Longueur (μm)')
    plt.ylabel('Moyenne des Ratios')
    cbar = plt.colorbar()
    cbar.set_label('Gap (μm)')
    plt.grid(True, alpha=0.3)
    
    # 6. Boxplot des ratios par gamme de gap
    ax6 = plt.subplot(2, 3, 6)
    # Grouper par gammes de gap
    gap_ranges = []
    ratio_groups = []
    
    gap_bins = np.linspace(min(gaps), max(gaps), 5)
    for i in range(len(gap_bins)-1):
        range_label = f"{gap_bins[i]:.2f}-{gap_bins[i+1]:.2f}"
        gap_ranges.append(range_label)
        
        # Trouver les échantillons dans cette gamme
        mask = (np.array(gaps) >= gap_bins[i]) & (np.array(gaps) < gap_bins[i+1])
        if i == len(gap_bins)-2:  # Inclure la dernière valeur
            mask = (np.array(gaps) >= gap_bins[i]) & (np.array(gaps) <= gap_bins[i+1])
        
        group_ratios = []
        for j, data in enumerate(samples_data):
            if mask[j]:
                group_ratios.extend(data['ratio'])
        
        if group_ratios:
            ratio_groups.append(group_ratios)
        else:
            ratio_groups.append([0])  # Valeur par défaut si pas de données
    
    plt.boxplot(ratio_groups, labels=gap_ranges)
    plt.title('Distribution des Ratios par Gamme de Gap')
    plt.xlabel('Gamme de Gap (μm)')
    plt.ylabel('Ratio')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = os.path.join(dataset_path, 'comprehensive_ratio_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Analyse complète sauvegardée: {output_path}")
    
    plt.show()
    
    # Afficher quelques statistiques
    print("\n=== Statistiques Globales ===")
    print(f"Nombre d'échantillons analysés: {len(samples_data)}")
    print(f"Gamme des gaps: {min(gaps):.4f} - {max(gaps):.4f} μm")
    print(f"Gamme des longueurs: {min(lengths):.3f} - {max(lengths):.3f} μm")
    print(f"Moyenne globale des ratios: {np.mean(all_ratios):.4f}")
    print(f"Écart-type global des ratios: {np.std(all_ratios):.4f}")
    print(f"Min/Max des ratios: {np.min(all_ratios):.4f} / {np.max(all_ratios):.4f}")

def main():
    """Fonction principale"""
    dataset_path = "."
    
    print("=== Analyse Avancée des Vecteurs des Ratios ===\n")
    
    # Créer l'analyse complète
    create_comprehensive_visualization(dataset_path, num_samples=25)
    
    print("\n✓ Analyse terminée!")

if __name__ == "__main__":
    main()
