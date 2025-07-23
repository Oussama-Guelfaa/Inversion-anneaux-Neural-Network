#!/usr/bin/env python3
"""
Visualiseur interactif des vecteurs des ratios avec sélection par paramètres
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import glob
from pathlib import Path

def load_all_data(dataset_path):
    """Charge toutes les données du dataset"""
    mat_files = glob.glob(os.path.join(dataset_path, "*.mat"))
    mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
    
    all_data = []
    
    print(f"Chargement de {len(mat_files)} fichiers...")
    
    for i, filepath in enumerate(mat_files):
        if i % 1000 == 0:
            print(f"  Progression: {i}/{len(mat_files)}")
        
        try:
            data = scipy.io.loadmat(filepath)
            
            sample_data = {
                'filename': os.path.basename(filepath),
                'ratio': data['ratio'].flatten() if 'ratio' in data else None,
                'gap': float(data['gap'][0, 0]) if 'gap' in data else None,
                'length': float(data['L_ecran_subs'][0, 0]) if 'L_ecran_subs' in data else None,
                'x': data['x'].flatten() if 'x' in data else None
            }
            
            if sample_data['ratio'] is not None:
                all_data.append(sample_data)
                
        except Exception as e:
            continue
    
    print(f"✓ {len(all_data)} fichiers chargés avec succès")
    return all_data

def filter_by_parameters(all_data, gap_range=None, length_range=None):
    """Filtre les données selon les paramètres spécifiés"""
    filtered_data = all_data.copy()
    
    if gap_range:
        filtered_data = [d for d in filtered_data 
                        if gap_range[0] <= d['gap'] <= gap_range[1]]
    
    if length_range:
        filtered_data = [d for d in filtered_data 
                        if length_range[0] <= d['length'] <= length_range[1]]
    
    return filtered_data

def plot_selected_ratios(filtered_data, max_samples=20, title_suffix=""):
    """Trace les vecteurs des ratios pour les données filtrées"""
    
    if not filtered_data:
        print("Aucune donnée correspondant aux critères")
        return
    
    # Limiter le nombre d'échantillons pour la lisibilité
    if len(filtered_data) > max_samples:
        import random
        filtered_data = random.sample(filtered_data, max_samples)
        print(f"Affichage de {max_samples} échantillons aléatoires sur {len(filtered_data)} disponibles")
    
    plt.figure(figsize=(14, 8))
    
    # Couleurs basées sur les valeurs de gap
    gaps = [d['gap'] for d in filtered_data]
    colors = plt.cm.plasma(np.linspace(0, 1, len(filtered_data)))
    
    for i, data in enumerate(filtered_data):
        x_vals = np.arange(len(data['ratio']))
        plt.plot(x_vals, data['ratio'], 
                color=colors[i], alpha=0.7, linewidth=1.5,
                label=f"Gap={data['gap']:.3f}μm, L={data['length']:.1f}μm")
    
    plt.title(f'Vecteurs des Ratios Superposés{title_suffix}')
    plt.xlabel('Index du Ratio')
    plt.ylabel('Valeur du Ratio')
    plt.grid(True, alpha=0.3)
    
    # Légende adaptative
    if len(filtered_data) <= 10:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        # Afficher seulement les statistiques
        plt.text(0.02, 0.98, 
                f'{len(filtered_data)} échantillons\n'
                f'Gap: {min(gaps):.3f}-{max(gaps):.3f}μm\n'
                f'Longueur: {min(d["length"] for d in filtered_data):.1f}-{max(d["length"] for d in filtered_data):.1f}μm',
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques
    all_ratios = np.concatenate([d['ratio'] for d in filtered_data])
    print(f"\n=== Statistiques pour {len(filtered_data)} échantillons ===")
    print(f"Moyenne des ratios: {np.mean(all_ratios):.4f}")
    print(f"Écart-type: {np.std(all_ratios):.4f}")
    print(f"Min/Max: {np.min(all_ratios):.4f} / {np.max(all_ratios):.4f}")

def interactive_exploration(all_data):
    """Interface interactive pour explorer les données"""
    
    # Afficher les gammes disponibles
    gaps = [d['gap'] for d in all_data]
    lengths = [d['length'] for d in all_data]
    
    print("\n=== Gammes de Paramètres Disponibles ===")
    print(f"Gap: {min(gaps):.4f} - {max(gaps):.4f} μm")
    print(f"Longueur: {min(lengths):.3f} - {max(lengths):.3f} μm")
    print(f"Total d'échantillons: {len(all_data)}")
    
    while True:
        print("\n" + "="*50)
        print("OPTIONS DE VISUALISATION:")
        print("1. Échantillons aléatoires (tous paramètres)")
        print("2. Filtrer par gamme de gap")
        print("3. Filtrer par gamme de longueur")
        print("4. Filtrer par gap ET longueur")
        print("5. Comparer gaps extrêmes")
        print("6. Comparer longueurs extrêmes")
        print("0. Quitter")
        
        choice = input("\nVotre choix (0-6): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            import random
            sample_data = random.sample(all_data, min(20, len(all_data)))
            plot_selected_ratios(sample_data, title_suffix=" (Échantillons Aléatoires)")
            
        elif choice == '2':
            try:
                gap_min = float(input(f"Gap minimum (min={min(gaps):.4f}): "))
                gap_max = float(input(f"Gap maximum (max={max(gaps):.4f}): "))
                filtered = filter_by_parameters(all_data, gap_range=(gap_min, gap_max))
                plot_selected_ratios(filtered, title_suffix=f" (Gap: {gap_min:.3f}-{gap_max:.3f}μm)")
            except ValueError:
                print("Valeurs invalides!")
                
        elif choice == '3':
            try:
                len_min = float(input(f"Longueur minimum (min={min(lengths):.1f}): "))
                len_max = float(input(f"Longueur maximum (max={max(lengths):.1f}): "))
                filtered = filter_by_parameters(all_data, length_range=(len_min, len_max))
                plot_selected_ratios(filtered, title_suffix=f" (L: {len_min:.1f}-{len_max:.1f}μm)")
            except ValueError:
                print("Valeurs invalides!")
                
        elif choice == '4':
            try:
                gap_min = float(input(f"Gap minimum (min={min(gaps):.4f}): "))
                gap_max = float(input(f"Gap maximum (max={max(gaps):.4f}): "))
                len_min = float(input(f"Longueur minimum (min={min(lengths):.1f}): "))
                len_max = float(input(f"Longueur maximum (max={max(lengths):.1f}): "))
                filtered = filter_by_parameters(all_data, 
                                              gap_range=(gap_min, gap_max),
                                              length_range=(len_min, len_max))
                plot_selected_ratios(filtered, 
                                   title_suffix=f" (Gap: {gap_min:.3f}-{gap_max:.3f}μm, L: {len_min:.1f}-{len_max:.1f}μm)")
            except ValueError:
                print("Valeurs invalides!")
                
        elif choice == '5':
            # Comparer gaps extrêmes
            gap_sorted = sorted(all_data, key=lambda x: x['gap'])
            low_gap = gap_sorted[:10]  # 10 plus petits gaps
            high_gap = gap_sorted[-10:]  # 10 plus grands gaps
            
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            for i, data in enumerate(low_gap):
                x_vals = np.arange(len(data['ratio']))
                plt.plot(x_vals, data['ratio'], alpha=0.7, linewidth=1.5)
            plt.title(f'Gaps Faibles (≤{low_gap[-1]["gap"]:.3f}μm)')
            plt.xlabel('Index')
            plt.ylabel('Ratio')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            for i, data in enumerate(high_gap):
                x_vals = np.arange(len(data['ratio']))
                plt.plot(x_vals, data['ratio'], alpha=0.7, linewidth=1.5)
            plt.title(f'Gaps Élevés (≥{high_gap[0]["gap"]:.3f}μm)')
            plt.xlabel('Index')
            plt.ylabel('Ratio')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        elif choice == '6':
            # Comparer longueurs extrêmes
            len_sorted = sorted(all_data, key=lambda x: x['length'])
            short_len = len_sorted[:10]  # 10 plus courtes
            long_len = len_sorted[-10:]  # 10 plus longues
            
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            for i, data in enumerate(short_len):
                x_vals = np.arange(len(data['ratio']))
                plt.plot(x_vals, data['ratio'], alpha=0.7, linewidth=1.5)
            plt.title(f'Longueurs Courtes (≤{short_len[-1]["length"]:.1f}μm)')
            plt.xlabel('Index')
            plt.ylabel('Ratio')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            for i, data in enumerate(long_len):
                x_vals = np.arange(len(data['ratio']))
                plt.plot(x_vals, data['ratio'], alpha=0.7, linewidth=1.5)
            plt.title(f'Longueurs Longues (≥{long_len[0]["length"]:.1f}μm)')
            plt.xlabel('Index')
            plt.ylabel('Ratio')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        else:
            print("Choix invalide!")

def main():
    """Fonction principale"""
    dataset_path = "."
    
    print("=== Visualiseur Interactif des Vecteurs des Ratios ===")
    print("Chargement des données...")
    
    # Charger toutes les données (peut prendre du temps)
    all_data = load_all_data(dataset_path)
    
    if not all_data:
        print("Aucune donnée trouvée!")
        return
    
    # Lancer l'exploration interactive
    interactive_exploration(all_data)
    
    print("\nAu revoir!")

if __name__ == "__main__":
    main()
