#!/usr/bin/env python3
"""
Démonstration rapide des vecteurs des ratios avec exemples spécifiques
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import random
import glob

def load_and_plot_specific_examples():
    """Charge et affiche des exemples spécifiques pour démonstration"""
    
    # Trouver tous les fichiers .mat
    mat_files = glob.glob("*.mat")
    mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
    
    # Sélectionner quelques exemples spécifiques
    examples = []
    
    # Chercher des exemples avec différents gaps
    target_gaps = [0.005, 0.1, 0.3, 0.5, 0.7]  # Différentes valeurs de gap
    
    for target_gap in target_gaps:
        # Trouver le fichier le plus proche de ce gap
        best_file = None
        best_diff = float('inf')
        
        for filepath in mat_files:
            try:
                data = scipy.io.loadmat(filepath)
                if 'gap' in data:
                    gap_val = float(data['gap'][0, 0])
                    diff = abs(gap_val - target_gap)
                    if diff < best_diff:
                        best_diff = diff
                        best_file = filepath
            except:
                continue
        
        if best_file:
            examples.append(best_file)
    
    # Ajouter quelques échantillons aléatoires
    random_samples = random.sample(mat_files, min(5, len(mat_files)))
    examples.extend(random_samples)
    
    # Supprimer les doublons
    examples = list(set(examples))
    
    print(f"Affichage de {len(examples)} exemples sélectionnés:")
    
    # Créer la visualisation
    plt.figure(figsize=(16, 10))
    
    # Couleurs distinctes
    colors = plt.cm.Set3(np.linspace(0, 1, len(examples)))
    
    valid_examples = []
    
    for i, filepath in enumerate(examples):
        try:
            data = scipy.io.loadmat(filepath)
            
            if 'ratio' in data and 'gap' in data and 'L_ecran_subs' in data:
                ratio_data = data['ratio'].flatten()
                gap = float(data['gap'][0, 0])
                length = float(data['L_ecran_subs'][0, 0])
                
                # Tracer le vecteur des ratios
                x_vals = np.arange(len(ratio_data))
                plt.plot(x_vals, ratio_data, 
                        color=colors[i], 
                        linewidth=2, 
                        alpha=0.8,
                        label=f"Gap={gap:.3f}μm, L={length:.1f}μm")
                
                valid_examples.append({
                    'filename': filepath,
                    'gap': gap,
                    'length': length,
                    'ratio': ratio_data
                })
                
                print(f"  ✓ {filepath}: Gap={gap:.3f}μm, L={length:.1f}μm")
            
        except Exception as e:
            print(f"  ✗ Erreur avec {filepath}: {e}")
            continue
    
    if not valid_examples:
        print("Aucun exemple valide trouvé!")
        return
    
    # Configuration du graphique
    plt.title('Exemples de Vecteurs des Ratios Superposés\n(Sélection représentative du dataset)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Index du Ratio', fontsize=14)
    plt.ylabel('Valeur du Ratio', fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Légende
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Améliorer l'apparence
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Sauvegarder
    plt.savefig('demo_ratio_vectors.png', dpi=300, bbox_inches='tight')
    print(f"\nGraphique sauvegardé: demo_ratio_vectors.png")
    
    plt.show()
    
    # Afficher des statistiques
    print(f"\n=== Statistiques des {len(valid_examples)} exemples ===")
    
    all_ratios = np.concatenate([ex['ratio'] for ex in valid_examples])
    gaps = [ex['gap'] for ex in valid_examples]
    lengths = [ex['length'] for ex in valid_examples]
    
    print(f"Gamme des gaps: {min(gaps):.4f} - {max(gaps):.4f} μm")
    print(f"Gamme des longueurs: {min(lengths):.1f} - {max(lengths):.1f} μm")
    print(f"Statistiques des ratios:")
    print(f"  - Moyenne: {np.mean(all_ratios):.4f}")
    print(f"  - Écart-type: {np.std(all_ratios):.4f}")
    print(f"  - Min/Max: {np.min(all_ratios):.4f} / {np.max(all_ratios):.4f}")
    print(f"  - Médiane: {np.median(all_ratios):.4f}")
    
    # Analyse par gap
    print(f"\nAnalyse par échantillon:")
    for ex in sorted(valid_examples, key=lambda x: x['gap']):
        ratio_mean = np.mean(ex['ratio'])
        ratio_std = np.std(ex['ratio'])
        print(f"  Gap={ex['gap']:.3f}μm, L={ex['length']:.1f}μm: "
              f"ratio_moy={ratio_mean:.3f}±{ratio_std:.3f}")

def create_summary_plot():
    """Crée un graphique de résumé avec différentes visualisations"""
    
    # Charger quelques échantillons
    mat_files = glob.glob("*.mat")
    mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
    
    # Sélectionner 12 échantillons aléatoires
    selected_files = random.sample(mat_files, min(12, len(mat_files)))
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, filepath in enumerate(selected_files):
        try:
            data = scipy.io.loadmat(filepath)
            
            if 'ratio' in data and 'gap' in data:
                ratio_data = data['ratio'].flatten()
                gap = float(data['gap'][0, 0])
                length = float(data['L_ecran_subs'][0, 0])
                
                # Tracer dans le sous-graphique
                x_vals = np.arange(len(ratio_data))
                axes[i].plot(x_vals, ratio_data, 'b-', linewidth=1.5, alpha=0.8)
                axes[i].set_title(f'Gap={gap:.3f}μm\nL={length:.1f}μm', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlabel('Index', fontsize=8)
                axes[i].set_ylabel('Ratio', fontsize=8)
                
        except Exception as e:
            axes[i].text(0.5, 0.5, 'Erreur\nde chargement', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Échantillon {i+1}', fontsize=10)
    
    plt.suptitle('Grille de Vecteurs des Ratios - Échantillons Aléatoires', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    plt.savefig('grid_ratio_vectors.png', dpi=300, bbox_inches='tight')
    print("Grille sauvegardée: grid_ratio_vectors.png")
    
    plt.show()

def main():
    """Fonction principale de démonstration"""
    print("=== Démonstration Rapide des Vecteurs des Ratios ===\n")
    
    print("1. Exemples sélectionnés avec différents paramètres:")
    load_and_plot_specific_examples()
    
    print("\n" + "="*60 + "\n")
    
    print("2. Grille d'échantillons aléatoires:")
    create_summary_plot()
    
    print("\n✓ Démonstration terminée!")
    print("\nFichiers générés:")
    print("  - demo_ratio_vectors.png")
    print("  - grid_ratio_vectors.png")

if __name__ == "__main__":
    main()
