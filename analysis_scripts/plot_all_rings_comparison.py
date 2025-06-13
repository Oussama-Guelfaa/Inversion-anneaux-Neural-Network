#!/usr/bin/env python3
"""
Visualisation comparative de tous les anneaux holographiques

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Ce script trace tous les anneaux dans un seul graphique pour visualiser
les différences entre les couples (gap, L_ecran).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from pathlib import Path
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

class AllRingsVisualizer:
    """
    Classe pour visualiser tous les anneaux holographiques dans un seul graphique.
    """
    
    def __init__(self, dataset_path="data_generation/dataset_2D"):
        """
        Initialise le visualiseur.
        
        Args:
            dataset_path (str): Chemin vers le dossier dataset_2D
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path("analysis_scripts/outputs_analysis_2D/visualizations")
        self.rings_data = []
        
        # Créer le dossier de sortie
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 Initialisation de la visualisation comparative")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"📁 Outputs: {self.output_path}")
    
    def extract_parameters_from_filename(self, filename):
        """
        Extrait les paramètres gap et L_ecran du nom de fichier.
        
        Args:
            filename (str): Nom du fichier
            
        Returns:
            tuple: (gap_value, L_ecran_value) ou (None, None) si échec
        """
        pattern = r'gap_([0-9.]+)um_L_([0-9.]+)um\.mat'
        match = re.match(pattern, filename)
        
        if match:
            gap = float(match.group(1))
            L_ecran = float(match.group(2))
            return gap, L_ecran
        else:
            return None, None
    
    def load_all_rings(self, max_samples=None):
        """
        Charge tous les anneaux du dataset.
        
        Args:
            max_samples (int): Nombre maximum d'échantillons à charger (None = tous)
        """
        print("\n📊 Chargement de tous les anneaux...")
        
        # Trouver tous les fichiers .mat
        mat_files = list(self.dataset_path.glob("*.mat"))
        mat_files = [f for f in mat_files if f.name != "labels.mat"]
        
        if max_samples:
            mat_files = mat_files[:max_samples]
        
        print(f"   Fichiers à traiter: {len(mat_files)}")
        
        for i, mat_file in enumerate(mat_files):
            if i % 200 == 0:
                print(f"   Progression: {i}/{len(mat_files)} fichiers...")
            
            # Extraire les paramètres
            gap, L_ecran = self.extract_parameters_from_filename(mat_file.name)
            
            if gap is None or L_ecran is None:
                continue
            
            try:
                # Charger le fichier
                data = loadmat(str(mat_file))
                
                if 'ratio' in data and 'x' in data:
                    ratio = data['ratio'].flatten()
                    x = data['x'].flatten()
                    
                    # Tronquer à 600 points si nécessaire
                    if len(ratio) > 600:
                        ratio = ratio[:600]
                        x = x[:600]
                    
                    ring_info = {
                        'gap': gap,
                        'L_ecran': L_ecran,
                        'ratio': ratio,
                        'x': x,
                        'filename': mat_file.name
                    }
                    
                    self.rings_data.append(ring_info)
                
            except Exception as e:
                print(f"   ⚠️  Erreur pour {mat_file.name}: {e}")
        
        print(f"✅ {len(self.rings_data)} anneaux chargés avec succès")
    
    def plot_all_rings_by_gap(self):
        """
        Trace tous les anneaux organisés par valeur de gap.
        """
        print("\n🎨 Génération du graphique par gaps...")
        
        # Organiser par gap
        gaps_dict = defaultdict(list)
        for ring in self.rings_data:
            gaps_dict[ring['gap']].append(ring)
        
        # Trier les gaps
        sorted_gaps = sorted(gaps_dict.keys())
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Palette de couleurs
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_gaps)))
        
        for i, gap in enumerate(sorted_gaps):
            rings = gaps_dict[gap]
            color = colors[i]
            
            # Tracer tous les anneaux pour ce gap
            for j, ring in enumerate(rings):
                alpha = 0.3 if len(rings) > 10 else 0.7
                linewidth = 0.5 if len(rings) > 10 else 1.0
                
                label = f"Gap={gap:.3f}µm" if j == 0 else None
                ax.plot(ring['x'], ring['ratio'], color=color, alpha=alpha, 
                       linewidth=linewidth, label=label)
        
        ax.set_xlabel('Position (µm)', fontsize=14)
        ax.set_ylabel('Ratio I/I₀', fontsize=14)
        ax.set_title('Tous les Anneaux Holographiques - Organisés par Gap', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "all_rings_by_gap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Graphique par gaps sauvegardé")
    
    def plot_all_rings_by_L_ecran(self):
        """
        Trace tous les anneaux organisés par valeur de L_ecran.
        """
        print("\n🎨 Génération du graphique par L_ecran...")
        
        # Organiser par L_ecran
        L_dict = defaultdict(list)
        for ring in self.rings_data:
            L_dict[ring['L_ecran']].append(ring)
        
        # Trier les L_ecran
        sorted_L = sorted(L_dict.keys())
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Palette de couleurs
        colors = plt.cm.plasma(np.linspace(0, 1, len(sorted_L)))
        
        for i, L_ecran in enumerate(sorted_L):
            rings = L_dict[L_ecran]
            color = colors[i]
            
            # Tracer tous les anneaux pour ce L_ecran
            for j, ring in enumerate(rings):
                alpha = 0.3 if len(rings) > 10 else 0.7
                linewidth = 0.5 if len(rings) > 10 else 1.0
                
                label = f"L_ecran={L_ecran:.1f}µm" if j == 0 else None
                ax.plot(ring['x'], ring['ratio'], color=color, alpha=alpha, 
                       linewidth=linewidth, label=label)
        
        ax.set_xlabel('Position (µm)', fontsize=14)
        ax.set_ylabel('Ratio I/I₀', fontsize=14)
        ax.set_title('Tous les Anneaux Holographiques - Organisés par L_ecran', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "all_rings_by_L_ecran.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Graphique par L_ecran sauvegardé")
    
    def plot_rings_3D_surface(self):
        """
        Crée une visualisation 3D des anneaux en fonction des paramètres.
        """
        print("\n🎨 Génération de la surface 3D...")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        # Sélectionner un sous-ensemble pour la lisibilité
        step = max(1, len(self.rings_data) // 100)  # Maximum 100 anneaux
        selected_rings = self.rings_data[::step]
        
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        # Palette de couleurs basée sur le gap
        gaps = [ring['gap'] for ring in selected_rings]
        colors = plt.cm.viridis((np.array(gaps) - min(gaps)) / (max(gaps) - min(gaps)))
        
        for i, ring in enumerate(selected_rings):
            # Utiliser L_ecran comme axe Y et position comme axe X
            x = ring['x']
            y = np.full_like(x, ring['L_ecran'])  # L_ecran constant pour chaque courbe
            z = ring['ratio']
            
            ax.plot(x, y, z, color=colors[i], alpha=0.7, linewidth=1.5,
                   label=f"Gap={ring['gap']:.3f}µm" if i < 10 else None)
        
        ax.set_xlabel('Position (µm)', fontsize=12)
        ax.set_ylabel('L_ecran (µm)', fontsize=12)
        ax.set_zlabel('Ratio I/I₀', fontsize=12)
        ax.set_title('Surface 3D des Anneaux Holographiques\n(Gap codé par couleur)', 
                    fontsize=14, fontweight='bold')
        
        # Ajouter une colorbar pour le gap
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(gaps)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Gap (µm)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "rings_3D_surface.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Surface 3D sauvegardée")
    
    def plot_rings_heatmap(self):
        """
        Crée une heatmap des intensités moyennes par couple (gap, L_ecran).
        """
        print("\n🎨 Génération de la heatmap...")
        
        # Calculer l'intensité moyenne pour chaque anneau
        gap_L_intensity = {}
        for ring in self.rings_data:
            key = (ring['gap'], ring['L_ecran'])
            gap_L_intensity[key] = np.mean(ring['ratio'])
        
        # Créer les listes pour la heatmap
        gaps = sorted(list(set([ring['gap'] for ring in self.rings_data])))
        L_ecrans = sorted(list(set([ring['L_ecran'] for ring in self.rings_data])))
        
        # Créer la matrice d'intensités
        intensity_matrix = np.zeros((len(gaps), len(L_ecrans)))
        
        for i, gap in enumerate(gaps):
            for j, L_ecran in enumerate(L_ecrans):
                if (gap, L_ecran) in gap_L_intensity:
                    intensity_matrix[i, j] = gap_L_intensity[(gap, L_ecran)]
                else:
                    intensity_matrix[i, j] = np.nan
        
        # Créer la heatmap
        fig, ax = plt.subplots(figsize=(20, 12))
        
        im = ax.imshow(intensity_matrix, aspect='auto', cmap='viridis', 
                      interpolation='nearest', origin='lower')
        
        # Configurer les axes
        ax.set_xlabel('L_ecran (µm)', fontsize=14)
        ax.set_ylabel('Gap (µm)', fontsize=14)
        ax.set_title('Heatmap des Intensités Moyennes par Couple (Gap, L_ecran)', 
                    fontsize=16, fontweight='bold')
        
        # Ajouter les labels des axes
        step_gap = max(1, len(gaps) // 20)
        step_L = max(1, len(L_ecrans) // 20)
        
        ax.set_yticks(range(0, len(gaps), step_gap))
        ax.set_yticklabels([f"{gap:.3f}" for gap in gaps[::step_gap]])
        ax.set_xticks(range(0, len(L_ecrans), step_L))
        ax.set_xticklabels([f"{L:.1f}" for L in L_ecrans[::step_L]], rotation=45)
        
        # Ajouter la colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensité Moyenne (Ratio I/I₀)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "rings_intensity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Heatmap sauvegardée")
    
    def generate_comparison_report(self):
        """
        Génère un rapport de comparaison des anneaux.
        """
        print("\n📝 Génération du rapport de comparaison...")
        
        # Statistiques globales
        gaps = [ring['gap'] for ring in self.rings_data]
        L_ecrans = [ring['L_ecran'] for ring in self.rings_data]
        intensities = [np.mean(ring['ratio']) for ring in self.rings_data]
        
        report_path = self.output_path.parent / "reports" / "rings_comparison_report.txt"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT DE COMPARAISON DES ANNEAUX HOLOGRAPHIQUES\n")
            f.write("="*80 + "\n")
            f.write(f"Auteur: Oussama GUELFAA\n")
            f.write(f"Date: 06-01-2025\n")
            f.write(f"Anneaux analysés: {len(self.rings_data)}\n")
            f.write("="*80 + "\n\n")
            
            # Statistiques des paramètres
            f.write("1. STATISTIQUES DES PARAMÈTRES\n")
            f.write("-"*40 + "\n")
            f.write(f"Gap minimum: {min(gaps):.4f} µm\n")
            f.write(f"Gap maximum: {max(gaps):.4f} µm\n")
            f.write(f"Nombre de gaps uniques: {len(set(gaps))}\n")
            f.write(f"L_ecran minimum: {min(L_ecrans):.3f} µm\n")
            f.write(f"L_ecran maximum: {max(L_ecrans):.3f} µm\n")
            f.write(f"Nombre de L_ecran uniques: {len(set(L_ecrans))}\n\n")
            
            # Statistiques des intensités
            f.write("2. STATISTIQUES DES INTENSITÉS\n")
            f.write("-"*40 + "\n")
            f.write(f"Intensité moyenne globale: {np.mean(intensities):.4f}\n")
            f.write(f"Écart-type global: {np.std(intensities):.4f}\n")
            f.write(f"Intensité minimale: {min(intensities):.4f}\n")
            f.write(f"Intensité maximale: {max(intensities):.4f}\n\n")
            
            # Analyse par plages de gap
            f.write("3. ANALYSE PAR PLAGES DE GAP\n")
            f.write("-"*40 + "\n")
            
            gap_ranges = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2)]
            for gap_min, gap_max in gap_ranges:
                rings_in_range = [ring for ring in self.rings_data 
                                if gap_min <= ring['gap'] < gap_max]
                if rings_in_range:
                    intensities_range = [np.mean(ring['ratio']) for ring in rings_in_range]
                    f.write(f"Gap [{gap_min:.2f}-{gap_max:.2f}] µm:\n")
                    f.write(f"  Nombre d'anneaux: {len(rings_in_range)}\n")
                    f.write(f"  Intensité moyenne: {np.mean(intensities_range):.4f}\n")
                    f.write(f"  Écart-type: {np.std(intensities_range):.4f}\n\n")
            
            # Fichiers générés
            f.write("4. FICHIERS GÉNÉRÉS\n")
            f.write("-"*40 + "\n")
            f.write("- all_rings_by_gap.png: Tous les anneaux organisés par gap\n")
            f.write("- all_rings_by_L_ecran.png: Tous les anneaux organisés par L_ecran\n")
            f.write("- rings_3D_surface.png: Visualisation 3D des anneaux\n")
            f.write("- rings_intensity_heatmap.png: Heatmap des intensités moyennes\n")
            f.write("- rings_comparison_report.txt: Ce rapport\n\n")
            
            f.write("5. OBSERVATIONS\n")
            f.write("-"*40 + "\n")
            f.write("- Les anneaux montrent une variation claire avec le gap\n")
            f.write("- L'effet de L_ecran est plus subtil mais observable\n")
            f.write("- Les intensités restent dans une plage cohérente\n")
            f.write("- La structure des anneaux est bien préservée\n")
        
        print(f"✅ Rapport sauvegardé: {report_path}")
    
    def run_complete_visualization(self, max_samples=None):
        """
        Lance la visualisation complète de tous les anneaux.
        
        Args:
            max_samples (int): Nombre maximum d'échantillons (None = tous)
        """
        print("🎨 VISUALISATION COMPARATIVE DE TOUS LES ANNEAUX")
        print("="*60)
        
        # 1. Charger tous les anneaux
        self.load_all_rings(max_samples)
        
        if len(self.rings_data) == 0:
            print("❌ Aucun anneau chargé")
            return
        
        # 2. Générer toutes les visualisations
        self.plot_all_rings_by_gap()
        self.plot_all_rings_by_L_ecran()
        self.plot_rings_3D_surface()
        self.plot_rings_heatmap()
        
        # 3. Générer le rapport
        self.generate_comparison_report()
        
        print("\n" + "="*60)
        print("✅ VISUALISATION COMPLÈTE TERMINÉE")
        print(f"📁 Résultats dans: {self.output_path}")
        print("="*60)


def main():
    """
    Fonction principale pour la visualisation comparative.
    """
    print("🎨 VISUALISATION COMPARATIVE DES ANNEAUX HOLOGRAPHIQUES")
    print("Auteur: Oussama GUELFAA")
    print("Date: 06-01-2025")
    print("-" * 60)
    
    # Vérifier que le dataset existe
    dataset_path = "data_generation/dataset_2D"
    if not Path(dataset_path).exists():
        print(f"❌ Erreur: Le dossier {dataset_path} n'existe pas.")
        return
    
    # Créer le visualiseur et lancer l'analyse
    visualizer = AllRingsVisualizer(dataset_path)
    
    # Demander à l'utilisateur s'il veut limiter le nombre d'échantillons
    print("\n🔍 Options de visualisation:")
    print("1. Tous les anneaux (2440 - peut être lent)")
    print("2. Échantillon représentatif (500 anneaux)")
    print("3. Échantillon rapide (100 anneaux)")
    
    choice = input("\nVotre choix (1/2/3) [défaut: 2]: ").strip()
    
    if choice == "1":
        max_samples = None
        print("📊 Visualisation de TOUS les anneaux...")
    elif choice == "3":
        max_samples = 100
        print("📊 Visualisation d'un échantillon rapide (100 anneaux)...")
    else:
        max_samples = 500
        print("📊 Visualisation d'un échantillon représentatif (500 anneaux)...")
    
    visualizer.run_complete_visualization(max_samples)


if __name__ == "__main__":
    main()
