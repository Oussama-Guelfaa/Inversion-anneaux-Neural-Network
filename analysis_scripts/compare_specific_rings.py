#!/usr/bin/env python3
"""
Comparaison spécifique entre couples d'anneaux holographiques

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Ce script permet de comparer des couples spécifiques d'anneaux
pour visualiser précisément les différences.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from pathlib import Path
import re

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

class SpecificRingsComparator:
    """
    Classe pour comparer des couples spécifiques d'anneaux.
    """
    
    def __init__(self, dataset_path="data_generation/dataset_2D"):
        """
        Initialise le comparateur.
        
        Args:
            dataset_path (str): Chemin vers le dossier dataset_2D
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path("analysis_scripts/outputs_analysis_2D/visualizations")
        
        # Créer le dossier de sortie
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 Comparateur d'anneaux spécifiques initialisé")
        print(f"📁 Dataset: {self.dataset_path}")
    
    def load_ring(self, gap, L_ecran):
        """
        Charge un anneau spécifique.
        
        Args:
            gap (float): Valeur du gap en µm
            L_ecran (float): Valeur de L_ecran en µm
            
        Returns:
            dict: Données de l'anneau ou None si non trouvé
        """
        # Construire le nom de fichier
        filename = f"gap_{gap:.4f}um_L_{L_ecran:.3f}um.mat"
        filepath = self.dataset_path / filename
        
        if not filepath.exists():
            print(f"⚠️  Fichier non trouvé: {filename}")
            return None
        
        try:
            data = loadmat(str(filepath))
            
            if 'ratio' in data and 'x' in data:
                ratio = data['ratio'].flatten()
                x = data['x'].flatten()
                
                # Tronquer à 600 points si nécessaire
                if len(ratio) > 600:
                    ratio = ratio[:600]
                    x = x[:600]
                
                return {
                    'gap': gap,
                    'L_ecran': L_ecran,
                    'ratio': ratio,
                    'x': x,
                    'filename': filename
                }
            else:
                print(f"⚠️  Données manquantes dans: {filename}")
                return None
                
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {filename}: {e}")
            return None
    
    def compare_gap_evolution(self, L_ecran_fixed=10.0, gaps_to_compare=None):
        """
        Compare l'évolution des anneaux pour différents gaps à L_ecran fixe.
        
        Args:
            L_ecran_fixed (float): Valeur de L_ecran fixe
            gaps_to_compare (list): Liste des gaps à comparer
        """
        if gaps_to_compare is None:
            gaps_to_compare = [0.005, 0.02, 0.05, 0.1, 0.15, 0.2]
        
        print(f"\n📊 Comparaison de l'évolution du gap (L_ecran={L_ecran_fixed}µm)")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Évolution des Anneaux avec le Gap (L_ecran={L_ecran_fixed}µm)', 
                    fontsize=16, fontweight='bold')
        
        # Graphique 1: Tous les anneaux superposés
        ax1 = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(gaps_to_compare)))
        
        rings_data = []
        for i, gap in enumerate(gaps_to_compare):
            ring = self.load_ring(gap, L_ecran_fixed)
            if ring:
                rings_data.append(ring)
                ax1.plot(ring['x'], ring['ratio'], color=colors[i], linewidth=2,
                        label=f'Gap={gap:.3f}µm', alpha=0.8)
        
        ax1.set_xlabel('Position (µm)')
        ax1.set_ylabel('Ratio I/I₀')
        ax1.set_title('Superposition des Anneaux')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Différences relatives
        ax2 = axes[0, 1]
        if len(rings_data) > 1:
            reference_ring = rings_data[0]  # Premier gap comme référence
            
            for i, ring in enumerate(rings_data[1:], 1):
                # Interpoler sur la même grille si nécessaire
                if len(ring['ratio']) == len(reference_ring['ratio']):
                    diff = ring['ratio'] - reference_ring['ratio']
                    ax2.plot(ring['x'], diff, color=colors[i], linewidth=2,
                            label=f'Gap={ring["gap"]:.3f}µm - {reference_ring["gap"]:.3f}µm')
            
            ax2.set_xlabel('Position (µm)')
            ax2.set_ylabel('Différence de Ratio')
            ax2.set_title(f'Différences par rapport à Gap={reference_ring["gap"]:.3f}µm')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Intensité maximale vs Gap
        ax3 = axes[1, 0]
        gaps_plot = [ring['gap'] for ring in rings_data]
        max_intensities = [np.max(ring['ratio']) for ring in rings_data]
        
        ax3.plot(gaps_plot, max_intensities, 'o-', linewidth=2, markersize=8, color='red')
        ax3.set_xlabel('Gap (µm)')
        ax3.set_ylabel('Intensité Maximale')
        ax3.set_title('Intensité Maximale vs Gap')
        ax3.grid(True, alpha=0.3)
        
        # Graphique 4: Position du premier minimum vs Gap
        ax4 = axes[1, 1]
        first_min_positions = []
        
        for ring in rings_data:
            # Trouver le premier minimum local
            ratio = ring['ratio']
            x = ring['x']
            
            # Chercher le premier minimum après le maximum central
            max_idx = np.argmax(ratio)
            if max_idx < len(ratio) - 10:
                min_idx = max_idx + np.argmin(ratio[max_idx:max_idx+50])
                first_min_positions.append(x[min_idx])
            else:
                first_min_positions.append(np.nan)
        
        valid_gaps = []
        valid_positions = []
        for gap, pos in zip(gaps_plot, first_min_positions):
            if not np.isnan(pos):
                valid_gaps.append(gap)
                valid_positions.append(pos)
        
        if valid_positions:
            ax4.plot(valid_gaps, valid_positions, 'o-', linewidth=2, markersize=8, color='blue')
            ax4.set_xlabel('Gap (µm)')
            ax4.set_ylabel('Position du 1er Minimum (µm)')
            ax4.set_title('Position du Premier Minimum vs Gap')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / f"gap_evolution_L{L_ecran_fixed:.1f}um.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparaison gap sauvegardée")
    
    def compare_L_ecran_evolution(self, gap_fixed=0.05, L_ecrans_to_compare=None):
        """
        Compare l'évolution des anneaux pour différents L_ecran à gap fixe.
        
        Args:
            gap_fixed (float): Valeur de gap fixe
            L_ecrans_to_compare (list): Liste des L_ecran à comparer
        """
        if L_ecrans_to_compare is None:
            L_ecrans_to_compare = [10.0, 10.3, 10.6, 10.9, 11.2, 11.5]
        
        print(f"\n📊 Comparaison de l'évolution de L_ecran (Gap={gap_fixed}µm)")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Évolution des Anneaux avec L_ecran (Gap={gap_fixed}µm)', 
                    fontsize=16, fontweight='bold')
        
        # Graphique 1: Tous les anneaux superposés
        ax1 = axes[0, 0]
        colors = plt.cm.plasma(np.linspace(0, 1, len(L_ecrans_to_compare)))
        
        rings_data = []
        for i, L_ecran in enumerate(L_ecrans_to_compare):
            ring = self.load_ring(gap_fixed, L_ecran)
            if ring:
                rings_data.append(ring)
                ax1.plot(ring['x'], ring['ratio'], color=colors[i], linewidth=2,
                        label=f'L_ecran={L_ecran:.1f}µm', alpha=0.8)
        
        ax1.set_xlabel('Position (µm)')
        ax1.set_ylabel('Ratio I/I₀')
        ax1.set_title('Superposition des Anneaux')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Zoom sur la région centrale
        ax2 = axes[0, 1]
        for i, ring in enumerate(rings_data):
            # Zoom sur les 200 premiers points
            zoom_points = min(200, len(ring['ratio']))
            ax2.plot(ring['x'][:zoom_points], ring['ratio'][:zoom_points], 
                    color=colors[i], linewidth=2, label=f'L_ecran={ring["L_ecran"]:.1f}µm')
        
        ax2.set_xlabel('Position (µm)')
        ax2.set_ylabel('Ratio I/I₀')
        ax2.set_title('Zoom sur la Région Centrale')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Intensité moyenne vs L_ecran
        ax3 = axes[1, 0]
        L_ecrans_plot = [ring['L_ecran'] for ring in rings_data]
        mean_intensities = [np.mean(ring['ratio']) for ring in rings_data]
        
        ax3.plot(L_ecrans_plot, mean_intensities, 'o-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('L_ecran (µm)')
        ax3.set_ylabel('Intensité Moyenne')
        ax3.set_title('Intensité Moyenne vs L_ecran')
        ax3.grid(True, alpha=0.3)
        
        # Graphique 4: Largeur du pic central vs L_ecran
        ax4 = axes[1, 1]
        peak_widths = []
        
        for ring in rings_data:
            # Calculer la largeur à mi-hauteur du pic central
            ratio = ring['ratio']
            max_val = np.max(ratio)
            half_max = max_val / 2
            
            # Trouver les indices où l'intensité dépasse la mi-hauteur
            above_half = ratio > half_max
            if np.any(above_half):
                indices = np.where(above_half)[0]
                width_points = len(indices)
                # Convertir en distance physique (approximation)
                if len(ring['x']) > 1:
                    dx = ring['x'][1] - ring['x'][0]
                    width_um = width_points * dx
                    peak_widths.append(width_um)
                else:
                    peak_widths.append(np.nan)
            else:
                peak_widths.append(np.nan)
        
        valid_L = []
        valid_widths = []
        for L, width in zip(L_ecrans_plot, peak_widths):
            if not np.isnan(width):
                valid_L.append(L)
                valid_widths.append(width)
        
        if valid_widths:
            ax4.plot(valid_L, valid_widths, 'o-', linewidth=2, markersize=8, color='orange')
            ax4.set_xlabel('L_ecran (µm)')
            ax4.set_ylabel('Largeur du Pic Central (µm)')
            ax4.set_title('Largeur du Pic Central vs L_ecran')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / f"L_ecran_evolution_gap{gap_fixed:.3f}um.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparaison L_ecran sauvegardée")
    
    def compare_specific_couples(self, couples_list):
        """
        Compare des couples spécifiques d'anneaux.
        
        Args:
            couples_list (list): Liste de tuples (gap, L_ecran)
        """
        print(f"\n📊 Comparaison de couples spécifiques")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(couples_list)))
        
        for i, (gap, L_ecran) in enumerate(couples_list):
            ring = self.load_ring(gap, L_ecran)
            if ring:
                ax.plot(ring['x'], ring['ratio'], color=colors[i], linewidth=2,
                       label=f'Gap={gap:.3f}µm, L_ecran={L_ecran:.1f}µm', alpha=0.8)
        
        ax.set_xlabel('Position (µm)', fontsize=14)
        ax.set_ylabel('Ratio I/I₀', fontsize=14)
        ax.set_title('Comparaison de Couples Spécifiques d\'Anneaux', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "specific_couples_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparaison couples spécifiques sauvegardée")
    
    def run_comprehensive_comparison(self):
        """
        Lance une comparaison complète avec plusieurs analyses.
        """
        print("🔍 COMPARAISON COMPLÈTE DES ANNEAUX SPÉCIFIQUES")
        print("="*60)
        
        # 1. Évolution du gap
        self.compare_gap_evolution(L_ecran_fixed=10.0)
        self.compare_gap_evolution(L_ecran_fixed=11.0)
        
        # 2. Évolution de L_ecran
        self.compare_L_ecran_evolution(gap_fixed=0.05)
        self.compare_L_ecran_evolution(gap_fixed=0.1)
        
        # 3. Couples spécifiques intéressants
        interesting_couples = [
            (0.005, 10.0),  # Gap minimal
            (0.05, 10.0),   # Gap moyen
            (0.2, 10.0),    # Gap maximal
            (0.05, 10.0),   # L_ecran minimal
            (0.05, 11.5),   # L_ecran maximal
        ]
        
        self.compare_specific_couples(interesting_couples)
        
        print("\n" + "="*60)
        print("✅ COMPARAISON COMPLÈTE TERMINÉE")
        print(f"📁 Résultats dans: {self.output_path}")
        print("="*60)


def main():
    """
    Fonction principale pour la comparaison spécifique.
    """
    print("🔍 COMPARAISON SPÉCIFIQUE D'ANNEAUX HOLOGRAPHIQUES")
    print("Auteur: Oussama GUELFAA")
    print("Date: 06-01-2025")
    print("-" * 60)
    
    # Vérifier que le dataset existe
    dataset_path = "data_generation/dataset_2D"
    if not Path(dataset_path).exists():
        print(f"❌ Erreur: Le dossier {dataset_path} n'existe pas.")
        return
    
    # Créer le comparateur et lancer l'analyse
    comparator = SpecificRingsComparator(dataset_path)
    comparator.run_comprehensive_comparison()


if __name__ == "__main__":
    main()
