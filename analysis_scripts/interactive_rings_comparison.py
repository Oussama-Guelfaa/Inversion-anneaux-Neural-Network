#!/usr/bin/env python3
"""
Comparaison interactive d'anneaux holographiques

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Ce script permet de comparer interactivement des anneaux spécifiques
choisis par l'utilisateur.
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
plt.rcParams['font.size'] = 12

class InteractiveRingsComparator:
    """
    Classe pour comparer interactivement des anneaux.
    """
    
    def __init__(self, dataset_path="data_generation/dataset_2D"):
        """
        Initialise le comparateur interactif.
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path("analysis_scripts/outputs_analysis_2D/visualizations")
        self.available_gaps = []
        self.available_L_ecrans = []
        
        # Créer le dossier de sortie
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Comparateur interactif d'anneaux initialisé")
        self.scan_available_parameters()
    
    def scan_available_parameters(self):
        """
        Scanne les paramètres disponibles dans le dataset.
        """
        print("🔍 Scan des paramètres disponibles...")
        
        mat_files = list(self.dataset_path.glob("*.mat"))
        mat_files = [f for f in mat_files if f.name != "labels.mat"]
        
        gaps_set = set()
        L_ecrans_set = set()
        
        for mat_file in mat_files:
            pattern = r'gap_([0-9.]+)um_L_([0-9.]+)um\.mat'
            match = re.match(pattern, mat_file.name)
            
            if match:
                gap = float(match.group(1))
                L_ecran = float(match.group(2))
                gaps_set.add(gap)
                L_ecrans_set.add(L_ecran)
        
        self.available_gaps = sorted(list(gaps_set))
        self.available_L_ecrans = sorted(list(L_ecrans_set))
        
        print(f"   📊 {len(self.available_gaps)} gaps disponibles: {self.available_gaps[0]:.3f} - {self.available_gaps[-1]:.3f} µm")
        print(f"   📊 {len(self.available_L_ecrans)} L_ecran disponibles: {self.available_L_ecrans[0]:.1f} - {self.available_L_ecrans[-1]:.1f} µm")
    
    def load_ring(self, gap, L_ecran):
        """
        Charge un anneau spécifique.
        """
        filename = f"gap_{gap:.4f}um_L_{L_ecran:.3f}um.mat"
        filepath = self.dataset_path / filename
        
        if not filepath.exists():
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
                return None
                
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return None
    
    def display_parameter_options(self):
        """
        Affiche les options de paramètres disponibles.
        """
        print("\n📋 PARAMÈTRES DISPONIBLES")
        print("="*50)
        
        print("🔹 Gaps disponibles (µm):")
        for i, gap in enumerate(self.available_gaps):
            if i % 8 == 0:
                print()
            print(f"{gap:.3f}", end="  ")
        print("\n")
        
        print("🔹 L_ecran disponibles (µm):")
        for i, L in enumerate(self.available_L_ecrans):
            if i % 10 == 0:
                print()
            print(f"{L:.1f}", end="  ")
        print("\n")
    
    def get_user_couples(self):
        """
        Demande à l'utilisateur de choisir les couples à comparer.
        """
        print("\n🎯 SÉLECTION DES COUPLES À COMPARER")
        print("="*50)
        
        couples = []
        
        while True:
            print(f"\n📍 Couple #{len(couples)+1}")
            
            # Choisir le gap
            while True:
                try:
                    gap_input = input(f"Gap (µm) [disponibles: {self.available_gaps[0]:.3f}-{self.available_gaps[-1]:.3f}]: ").strip()
                    if gap_input.lower() in ['q', 'quit', 'stop']:
                        return couples
                    
                    gap = float(gap_input)
                    
                    # Trouver le gap le plus proche
                    closest_gap = min(self.available_gaps, key=lambda x: abs(x - gap))
                    if abs(closest_gap - gap) > 0.001:
                        print(f"   ⚠️  Gap {gap:.3f} non trouvé, utilisation du plus proche: {closest_gap:.3f}")
                        gap = closest_gap
                    
                    break
                except ValueError:
                    print("   ❌ Veuillez entrer un nombre valide")
            
            # Choisir L_ecran
            while True:
                try:
                    L_input = input(f"L_ecran (µm) [disponibles: {self.available_L_ecrans[0]:.1f}-{self.available_L_ecrans[-1]:.1f}]: ").strip()
                    if L_input.lower() in ['q', 'quit', 'stop']:
                        return couples
                    
                    L_ecran = float(L_input)
                    
                    # Trouver le L_ecran le plus proche
                    closest_L = min(self.available_L_ecrans, key=lambda x: abs(x - L_ecran))
                    if abs(closest_L - L_ecran) > 0.01:
                        print(f"   ⚠️  L_ecran {L_ecran:.1f} non trouvé, utilisation du plus proche: {closest_L:.1f}")
                        L_ecran = closest_L
                    
                    break
                except ValueError:
                    print("   ❌ Veuillez entrer un nombre valide")
            
            # Vérifier que le couple existe
            ring = self.load_ring(gap, L_ecran)
            if ring:
                couples.append((gap, L_ecran))
                print(f"   ✅ Couple ajouté: Gap={gap:.3f}µm, L_ecran={L_ecran:.1f}µm")
            else:
                print(f"   ❌ Couple non trouvé dans le dataset")
                continue
            
            # Demander s'il faut continuer
            if len(couples) >= 10:
                print("   ⚠️  Maximum 10 couples recommandé pour la lisibilité")
            
            continue_choice = input("\nAjouter un autre couple ? (o/n) [n]: ").strip().lower()
            if continue_choice not in ['o', 'oui', 'y', 'yes']:
                break
        
        return couples
    
    def plot_comparison(self, couples, title_suffix=""):
        """
        Trace la comparaison des couples sélectionnés.
        """
        if not couples:
            print("❌ Aucun couple à comparer")
            return
        
        print(f"\n🎨 Génération du graphique de comparaison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'Comparaison d\'Anneaux Holographiques{title_suffix}', 
                    fontsize=16, fontweight='bold')
        
        # Charger tous les anneaux
        rings_data = []
        for gap, L_ecran in couples:
            ring = self.load_ring(gap, L_ecran)
            if ring:
                rings_data.append(ring)
        
        if not rings_data:
            print("❌ Aucun anneau chargé")
            return
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(rings_data)))
        
        # Graphique 1: Superposition des anneaux
        ax1 = axes[0, 0]
        for i, ring in enumerate(rings_data):
            label = f"Gap={ring['gap']:.3f}µm, L={ring['L_ecran']:.1f}µm"
            ax1.plot(ring['x'], ring['ratio'], color=colors[i], linewidth=2,
                    label=label, alpha=0.8)
        
        ax1.set_xlabel('Position (µm)')
        ax1.set_ylabel('Ratio I/I₀')
        ax1.set_title('Superposition des Anneaux')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Zoom sur la région centrale
        ax2 = axes[0, 1]
        zoom_points = 200
        for i, ring in enumerate(rings_data):
            points_to_plot = min(zoom_points, len(ring['ratio']))
            label = f"Gap={ring['gap']:.3f}µm, L={ring['L_ecran']:.1f}µm"
            ax2.plot(ring['x'][:points_to_plot], ring['ratio'][:points_to_plot], 
                    color=colors[i], linewidth=2, label=label, alpha=0.8)
        
        ax2.set_xlabel('Position (µm)')
        ax2.set_ylabel('Ratio I/I₀')
        ax2.set_title('Zoom sur la Région Centrale')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Statistiques par anneau
        ax3 = axes[1, 0]
        gaps = [ring['gap'] for ring in rings_data]
        L_ecrans = [ring['L_ecran'] for ring in rings_data]
        max_intensities = [np.max(ring['ratio']) for ring in rings_data]
        
        scatter = ax3.scatter(gaps, L_ecrans, c=max_intensities, s=100, 
                             cmap='viridis', alpha=0.8, edgecolors='black')
        ax3.set_xlabel('Gap (µm)')
        ax3.set_ylabel('L_ecran (µm)')
        ax3.set_title('Intensité Maximale par Couple')
        plt.colorbar(scatter, ax=ax3, label='Intensité Max')
        ax3.grid(True, alpha=0.3)
        
        # Graphique 4: Profil des différences
        ax4 = axes[1, 1]
        if len(rings_data) > 1:
            reference_ring = rings_data[0]
            
            for i, ring in enumerate(rings_data[1:], 1):
                if len(ring['ratio']) == len(reference_ring['ratio']):
                    diff = ring['ratio'] - reference_ring['ratio']
                    label = f"Δ({ring['gap']:.3f}, {ring['L_ecran']:.1f}) - ({reference_ring['gap']:.3f}, {reference_ring['L_ecran']:.1f})"
                    ax4.plot(ring['x'], diff, color=colors[i], linewidth=2, label=label)
            
            ax4.set_xlabel('Position (µm)')
            ax4.set_ylabel('Différence de Ratio')
            ax4.set_title(f'Différences par rapport au premier anneau')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Générer un nom de fichier unique
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interactive_comparison_{timestamp}.png"
        
        plt.savefig(self.output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparaison sauvegardée: {filename}")
        
        # Afficher un résumé
        print(f"\n📊 RÉSUMÉ DE LA COMPARAISON")
        print("-"*40)
        for ring in rings_data:
            print(f"Gap={ring['gap']:.3f}µm, L_ecran={ring['L_ecran']:.1f}µm:")
            print(f"  - Intensité max: {np.max(ring['ratio']):.4f}")
            print(f"  - Intensité moyenne: {np.mean(ring['ratio']):.4f}")
            print(f"  - Écart-type: {np.std(ring['ratio']):.4f}")
    
    def run_interactive_session(self):
        """
        Lance une session interactive de comparaison.
        """
        print("🎯 SESSION INTERACTIVE DE COMPARAISON D'ANNEAUX")
        print("="*60)
        
        while True:
            print("\n🔍 OPTIONS DISPONIBLES:")
            print("1. Afficher les paramètres disponibles")
            print("2. Comparer des couples personnalisés")
            print("3. Comparaisons prédéfinies intéressantes")
            print("4. Quitter")
            
            choice = input("\nVotre choix (1-4): ").strip()
            
            if choice == "1":
                self.display_parameter_options()
            
            elif choice == "2":
                couples = self.get_user_couples()
                if couples:
                    self.plot_comparison(couples, " - Sélection Personnalisée")
                else:
                    print("❌ Aucun couple sélectionné")
            
            elif choice == "3":
                print("\n🎯 COMPARAISONS PRÉDÉFINIES:")
                print("a. Évolution du gap (L_ecran=10.0µm)")
                print("b. Évolution de L_ecran (Gap=0.05µm)")
                print("c. Couples extrêmes")
                print("d. Transition critique (gaps 0.04-0.06µm)")
                
                sub_choice = input("Votre choix (a-d): ").strip().lower()
                
                if sub_choice == "a":
                    couples = [(0.005, 10.0), (0.05, 10.0), (0.1, 10.0), (0.15, 10.0), (0.2, 10.0)]
                    self.plot_comparison(couples, " - Évolution Gap")
                
                elif sub_choice == "b":
                    couples = [(0.05, 10.0), (0.05, 10.5), (0.05, 11.0), (0.05, 11.5)]
                    self.plot_comparison(couples, " - Évolution L_ecran")
                
                elif sub_choice == "c":
                    couples = [(0.005, 10.0), (0.2, 11.5), (0.1, 10.75)]
                    self.plot_comparison(couples, " - Couples Extrêmes")
                
                elif sub_choice == "d":
                    couples = [(0.04, 10.0), (0.045, 10.0), (0.05, 10.0), (0.055, 10.0), (0.06, 10.0)]
                    self.plot_comparison(couples, " - Transition Critique")
            
            elif choice == "4":
                print("👋 Au revoir !")
                break
            
            else:
                print("❌ Choix invalide")


def main():
    """
    Fonction principale pour la comparaison interactive.
    """
    print("🎯 COMPARAISON INTERACTIVE D'ANNEAUX HOLOGRAPHIQUES")
    print("Auteur: Oussama GUELFAA")
    print("Date: 06-01-2025")
    print("-" * 60)
    
    # Vérifier que le dataset existe
    dataset_path = "data_generation/dataset_2D"
    if not Path(dataset_path).exists():
        print(f"❌ Erreur: Le dossier {dataset_path} n'existe pas.")
        return
    
    # Importer pandas pour le timestamp
    import pandas as pd
    
    # Créer le comparateur et lancer la session
    comparator = InteractiveRingsComparator(dataset_path)
    comparator.run_interactive_session()


if __name__ == "__main__":
    main()
