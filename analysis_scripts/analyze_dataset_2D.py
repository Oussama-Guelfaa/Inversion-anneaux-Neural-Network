#!/usr/bin/env python3
"""
Analyse complète du dataset 2D d'anneaux holographiques

Auteur: Oussama GUELFAA
Date: 13 - 06 - 2025

Ce script effectue une analyse détaillée du dossier dataset_generation/dataset_2D
pour comprendre la structure, la distribution et la qualité des données 2D.
"""

import os
import glob
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
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class Dataset2DAnalyzer:
    """
    Classe pour analyser le dataset 2D d'anneaux holographiques.
    
    Cette classe charge tous les fichiers .mat du dossier dataset_2D,
    extrait les métadonnées, et génère des analyses statistiques et visuelles.
    """
    
    def __init__(self, dataset_path="data_generation/dataset_2D"):
        """
        Initialise l'analyseur avec le chemin du dataset.
        
        Args:
            dataset_path (str): Chemin vers le dossier dataset_2D
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path("analysis_scripts/outputs_analysis_2D")
        self.data_info = []
        self.loaded_data = {}
        
        # Créer le dossier de sortie
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 Initialisation de l'analyse du dataset: {self.dataset_path}")
        print(f"📁 Dossier de sortie: {self.output_path}")
    
    def extract_parameters_from_filename(self, filename):
        """
        Extrait les paramètres gap et L_ecran du nom de fichier.
        
        Args:
            filename (str): Nom du fichier (ex: gap_0.0050um_L_10.000um.mat)
            
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
    
    def load_dataset_info(self):
        """
        Charge les informations de tous les fichiers .mat du dataset.
        
        Cette méthode parcourt tous les fichiers .mat, extrait les métadonnées
        et stocke les informations dans self.data_info.
        """
        print("\n📊 Chargement des informations du dataset...")
        
        # Trouver tous les fichiers .mat
        mat_files = list(self.dataset_path.glob("*.mat"))
        mat_files = [f for f in mat_files if f.name != "labels.mat"]  # Exclure labels.mat
        
        print(f"   Fichiers .mat trouvés: {len(mat_files)}")
        
        for i, mat_file in enumerate(mat_files):
            if i % 100 == 0:
                print(f"   Progression: {i}/{len(mat_files)} fichiers traités...")
            
            # Extraire les paramètres du nom de fichier
            gap, L_ecran = self.extract_parameters_from_filename(mat_file.name)
            
            if gap is None or L_ecran is None:
                print(f"   ⚠️  Impossible d'extraire les paramètres de: {mat_file.name}")
                continue
            
            try:
                # Charger le fichier .mat
                data = loadmat(str(mat_file))
                
                # Extraire les informations
                info = {
                    'filename': mat_file.name,
                    'filepath': str(mat_file),
                    'gap_um': gap,
                    'L_ecran_um': L_ecran,
                    'ratio_shape': data['ratio'].shape if 'ratio' in data else None,
                    'x_shape': data['x'].shape if 'x' in data else None,
                    'ratio_min': np.min(data['ratio']) if 'ratio' in data else None,
                    'ratio_max': np.max(data['ratio']) if 'ratio' in data else None,
                    'ratio_mean': np.mean(data['ratio']) if 'ratio' in data else None,
                    'ratio_std': np.std(data['ratio']) if 'ratio' in data else None,
                    'file_size_mb': mat_file.stat().st_size / (1024*1024)
                }
                
                self.data_info.append(info)
                
            except Exception as e:
                print(f"   ❌ Erreur lors du chargement de {mat_file.name}: {e}")
        
        print(f"✅ Chargement terminé: {len(self.data_info)} fichiers analysés")
        
        # Convertir en DataFrame pour faciliter l'analyse
        self.df = pd.DataFrame(self.data_info)
        
        if len(self.df) > 0:
            print(f"   Plage des gaps: {self.df['gap_um'].min():.4f} - {self.df['gap_um'].max():.4f} µm")
            print(f"   Plage des L_ecran: {self.df['L_ecran_um'].min():.3f} - {self.df['L_ecran_um'].max():.3f} µm")
    
    def generate_statistical_analysis(self):
        """
        Génère une analyse statistique globale du dataset.
        
        Cette méthode calcule et affiche des statistiques descriptives
        sur la distribution des paramètres et des données.
        """
        print("\n📈 Génération de l'analyse statistique...")
        
        if len(self.df) == 0:
            print("❌ Aucune donnée à analyser")
            return
        
        # Statistiques générales
        stats = {
            'Nombre total de fichiers': len(self.df),
            'Nombre de gaps uniques': self.df['gap_um'].nunique(),
            'Nombre de L_ecran uniques': self.df['L_ecran_um'].nunique(),
            'Taille totale (MB)': self.df['file_size_mb'].sum(),
            'Taille moyenne par fichier (MB)': self.df['file_size_mb'].mean()
        }
        
        print("\n🔢 Statistiques générales:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        # Distribution des gaps
        gap_counts = self.df['gap_um'].value_counts().sort_index()
        print(f"\n📊 Distribution des gaps:")
        print(f"   Gaps disponibles: {sorted(self.df['gap_um'].unique())}")
        print(f"   Nombre de fichiers par gap: {gap_counts.iloc[0]} (uniforme: {gap_counts.nunique() == 1})")
        
        # Distribution des L_ecran
        L_counts = self.df['L_ecran_um'].value_counts().sort_index()
        print(f"\n📊 Distribution des L_ecran:")
        print(f"   L_ecran min: {self.df['L_ecran_um'].min():.3f} µm")
        print(f"   L_ecran max: {self.df['L_ecran_um'].max():.3f} µm")
        print(f"   Pas moyen: {np.diff(sorted(self.df['L_ecran_um'].unique())).mean():.3f} µm")
        
        # Vérification de la complétude
        expected_combinations = len(self.df['gap_um'].unique()) * len(self.df['L_ecran_um'].unique())
        actual_combinations = len(self.df)
        completeness = (actual_combinations / expected_combinations) * 100
        
        print(f"\n✅ Complétude du dataset:")
        print(f"   Combinaisons attendues: {expected_combinations}")
        print(f"   Combinaisons présentes: {actual_combinations}")
        print(f"   Complétude: {completeness:.1f}%")
        
        # Statistiques sur les ratios
        if 'ratio_mean' in self.df.columns and not self.df['ratio_mean'].isna().all():
            print(f"\n📊 Statistiques des ratios d'intensité:")
            print(f"   Ratio moyen global: {self.df['ratio_mean'].mean():.4f}")
            print(f"   Écart-type moyen: {self.df['ratio_std'].mean():.4f}")
            print(f"   Valeur min globale: {self.df['ratio_min'].min():.4f}")
            print(f"   Valeur max globale: {self.df['ratio_max'].max():.4f}")
        
        # Sauvegarder les statistiques
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(self.output_path / "dataset_statistics.csv", index=False)
        
        # Sauvegarder le résumé détaillé
        summary_df = self.df.describe()
        summary_df.to_csv(self.output_path / "detailed_statistics.csv")
        
        print(f"💾 Statistiques sauvegardées dans {self.output_path}")

    def plot_parameter_distributions(self):
        """
        Génère des graphiques de distribution des paramètres.

        Cette méthode crée des histogrammes et des graphiques de densité
        pour visualiser la distribution des gaps et L_ecran.
        """
        print("\n📊 Génération des graphiques de distribution...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Distribution des Paramètres du Dataset 2D', fontsize=16, fontweight='bold')

        # Distribution des gaps
        axes[0, 0].hist(self.df['gap_um'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Gap (µm)')
        axes[0, 0].set_ylabel('Nombre de fichiers')
        axes[0, 0].set_title('Distribution des Gaps')
        axes[0, 0].grid(True, alpha=0.3)

        # Distribution des L_ecran
        axes[0, 1].hist(self.df['L_ecran_um'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('L_ecran (µm)')
        axes[0, 1].set_ylabel('Nombre de fichiers')
        axes[0, 1].set_title('Distribution des L_ecran')
        axes[0, 1].grid(True, alpha=0.3)

        # Heatmap de densité dans l'espace (gap, L_ecran)
        pivot_table = self.df.pivot_table(values='filename', index='gap_um',
                                         columns='L_ecran_um', aggfunc='count', fill_value=0)

        im = axes[1, 0].imshow(pivot_table.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        axes[1, 0].set_xlabel('Index L_ecran')
        axes[1, 0].set_ylabel('Index Gap')
        axes[1, 0].set_title('Heatmap de Densité (Gap vs L_ecran)')

        # Ajouter une colorbar
        cbar = plt.colorbar(im, ax=axes[1, 0])
        cbar.set_label('Nombre de fichiers')

        # Scatter plot gap vs L_ecran
        scatter = axes[1, 1].scatter(self.df['L_ecran_um'], self.df['gap_um'],
                                   alpha=0.6, c=self.df.index, cmap='viridis', s=20)
        axes[1, 1].set_xlabel('L_ecran (µm)')
        axes[1, 1].set_ylabel('Gap (µm)')
        axes[1, 1].set_title('Répartition dans l\'espace des paramètres')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_path / "parameter_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Graphique de densité 2D plus détaillé
        plt.figure(figsize=(12, 8))
        plt.hexbin(self.df['L_ecran_um'], self.df['gap_um'], gridsize=30, cmap='Blues', mincnt=1)
        plt.colorbar(label='Nombre de points')
        plt.xlabel('L_ecran (µm)')
        plt.ylabel('Gap (µm)')
        plt.title('Densité des Points dans l\'Espace des Paramètres (Gap, L_ecran)')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_path / "parameter_density_2D.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Graphiques de distribution sauvegardés")

    def analyze_coverage_gaps(self):
        """
        Analyse les trous ou zones mal couvertes dans l'espace des paramètres.

        Cette méthode identifie les combinaisons manquantes et les zones
        sous-échantillonnées dans l'espace (gap, L_ecran).
        """
        print("\n🔍 Analyse de la couverture de l'espace des paramètres...")

        # Créer une grille complète des paramètres attendus
        unique_gaps = sorted(self.df['gap_um'].unique())
        unique_L_ecran = sorted(self.df['L_ecran_um'].unique())

        print(f"   Gaps uniques: {len(unique_gaps)}")
        print(f"   L_ecran uniques: {len(unique_L_ecran)}")

        # Identifier les combinaisons manquantes
        expected_combinations = set()
        for gap in unique_gaps:
            for L in unique_L_ecran:
                expected_combinations.add((gap, L))

        actual_combinations = set(zip(self.df['gap_um'], self.df['L_ecran_um']))
        missing_combinations = expected_combinations - actual_combinations

        print(f"   Combinaisons attendues: {len(expected_combinations)}")
        print(f"   Combinaisons présentes: {len(actual_combinations)}")
        print(f"   Combinaisons manquantes: {len(missing_combinations)}")

        if missing_combinations:
            print(f"   ⚠️  Premières combinaisons manquantes:")
            for i, (gap, L) in enumerate(sorted(missing_combinations)[:10]):
                print(f"      Gap={gap:.4f}µm, L_ecran={L:.3f}µm")
            if len(missing_combinations) > 10:
                print(f"      ... et {len(missing_combinations)-10} autres")

        # Créer une matrice de couverture
        coverage_matrix = np.zeros((len(unique_gaps), len(unique_L_ecran)))

        for i, gap in enumerate(unique_gaps):
            for j, L in enumerate(unique_L_ecran):
                if (gap, L) in actual_combinations:
                    coverage_matrix[i, j] = 1

        # Visualiser la matrice de couverture
        plt.figure(figsize=(15, 10))
        plt.imshow(coverage_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
        plt.colorbar(label='Présence (1=Oui, 0=Non)')
        plt.xlabel('Index L_ecran')
        plt.ylabel('Index Gap')
        plt.title('Matrice de Couverture du Dataset\n(Vert=Présent, Rouge=Manquant)')

        # Ajouter des labels sur les axes
        step_gap = max(1, len(unique_gaps) // 10)
        step_L = max(1, len(unique_L_ecran) // 10)

        plt.yticks(range(0, len(unique_gaps), step_gap),
                  [f"{gap:.3f}" for gap in unique_gaps[::step_gap]])
        plt.xticks(range(0, len(unique_L_ecran), step_L),
                  [f"{L:.2f}" for L in unique_L_ecran[::step_L]], rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_path / "coverage_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Sauvegarder les combinaisons manquantes
        if missing_combinations:
            missing_df = pd.DataFrame(list(missing_combinations), columns=['gap_um', 'L_ecran_um'])
            missing_df.to_csv(self.output_path / "missing_combinations.csv", index=False)
            print(f"💾 Combinaisons manquantes sauvegardées dans missing_combinations.csv")

        return len(missing_combinations) == 0

    def load_sample_data_for_visualization(self, n_samples=36):
        """
        Charge un échantillon représentatif de données pour la visualisation.

        Args:
            n_samples (int): Nombre d'échantillons à charger (par défaut 36 pour une grille 6x6)

        Returns:
            list: Liste des données chargées avec métadonnées
        """
        print(f"\n📥 Chargement de {n_samples} échantillons pour visualisation...")

        # Sélectionner un échantillon représentatif
        # Stratégie: prendre des échantillons répartis uniformément dans l'espace des paramètres

        # Trier par gap puis par L_ecran
        df_sorted = self.df.sort_values(['gap_um', 'L_ecran_um'])

        # Prendre des échantillons espacés uniformément
        indices = np.linspace(0, len(df_sorted)-1, n_samples, dtype=int)
        sample_df = df_sorted.iloc[indices].copy()

        loaded_samples = []

        for idx, row in sample_df.iterrows():
            try:
                # Charger les données
                data = loadmat(row['filepath'])

                sample_info = {
                    'gap_um': row['gap_um'],
                    'L_ecran_um': row['L_ecran_um'],
                    'filename': row['filename'],
                    'ratio': data['ratio'].flatten(),  # Convertir en 1D
                    'x': data['x'].flatten() if 'x' in data else np.arange(len(data['ratio'].flatten()))
                }

                loaded_samples.append(sample_info)

            except Exception as e:
                print(f"   ⚠️  Erreur lors du chargement de {row['filename']}: {e}")

        print(f"✅ {len(loaded_samples)} échantillons chargés avec succès")
        return loaded_samples

    def plot_ring_samples(self, n_samples=36):
        """
        Trace un échantillon représentatif d'anneaux en grille.

        Args:
            n_samples (int): Nombre d'échantillons à afficher (par défaut 36)
        """
        print(f"\n🎨 Génération de la grille d'anneaux ({n_samples} échantillons)...")

        # Charger les échantillons
        samples = self.load_sample_data_for_visualization(n_samples)

        if len(samples) == 0:
            print("❌ Aucun échantillon chargé pour la visualisation")
            return

        # Calculer la taille de la grille
        grid_size = int(np.ceil(np.sqrt(len(samples))))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        fig.suptitle('Échantillon Représentatif d\'Anneaux Holographiques 2D',
                    fontsize=16, fontweight='bold')

        # Aplatir les axes pour faciliter l'itération
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, sample in enumerate(samples):
            if i >= len(axes):
                break

            ax = axes[i]

            # Tracer le profil d'intensité
            ax.plot(sample['x'], sample['ratio'], 'b-', linewidth=1.5, alpha=0.8)
            ax.set_title(f"Gap={sample['gap_um']:.3f}µm, L={sample['L_ecran_um']:.1f}µm",
                        fontsize=10)
            ax.set_xlabel('Position (µm)', fontsize=8)
            ax.set_ylabel('Ratio I/I₀', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            # Limiter l'affichage pour une meilleure lisibilité
            if len(sample['ratio']) > 600:
                # Tronquer à 600 points comme suggéré dans les mémoires
                ax.plot(sample['x'][:600], sample['ratio'][:600], 'b-', linewidth=1.5, alpha=0.8)
                ax.set_xlim(sample['x'][0], sample['x'][599])

        # Masquer les axes inutilisés
        for i in range(len(samples), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_path / "ring_samples_grid.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Grille d'anneaux sauvegardée")

    def plot_rings_by_parameter(self):
        """
        Trace des anneaux organisés par paramètres (gap croissant, L_ecran fixe).

        Cette méthode crée des visualisations pour montrer l'évolution
        des profils d'anneaux en fonction des paramètres.
        """
        print("\n📊 Génération des graphiques organisés par paramètres...")

        # Sélectionner quelques valeurs de L_ecran représentatives
        L_values = sorted(self.df['L_ecran_um'].unique())
        selected_L = [L_values[0], L_values[len(L_values)//4],
                     L_values[len(L_values)//2], L_values[3*len(L_values)//4], L_values[-1]]

        # Sélectionner quelques valeurs de gap
        gap_values = sorted(self.df['gap_um'].unique())
        selected_gaps = gap_values[::max(1, len(gap_values)//8)]  # Prendre ~8 gaps

        fig, axes = plt.subplots(len(selected_L), 1, figsize=(15, 4*len(selected_L)))
        if len(selected_L) == 1:
            axes = [axes]

        fig.suptitle('Évolution des Profils d\'Anneaux par L_ecran', fontsize=16, fontweight='bold')

        colors = plt.cm.viridis(np.linspace(0, 1, len(selected_gaps)))

        for i, L_ecran in enumerate(selected_L):
            ax = axes[i]

            for j, gap in enumerate(selected_gaps):
                # Trouver le fichier correspondant
                matching_files = self.df[(self.df['gap_um'] == gap) &
                                       (self.df['L_ecran_um'] == L_ecran)]

                if len(matching_files) > 0:
                    try:
                        filepath = matching_files.iloc[0]['filepath']
                        data = loadmat(filepath)

                        ratio = data['ratio'].flatten()
                        x = data['x'].flatten() if 'x' in data else np.arange(len(ratio))

                        # Tronquer à 600 points si nécessaire
                        if len(ratio) > 600:
                            ratio = ratio[:600]
                            x = x[:600]

                        ax.plot(x, ratio, color=colors[j], linewidth=1.5,
                               label=f'Gap={gap:.3f}µm', alpha=0.8)

                    except Exception as e:
                        print(f"   ⚠️  Erreur pour Gap={gap}, L={L_ecran}: {e}")

            ax.set_title(f'L_ecran = {L_ecran:.1f} µm', fontsize=12, fontweight='bold')
            ax.set_xlabel('Position (µm)')
            ax.set_ylabel('Ratio I/I₀')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_path / "rings_by_L_ecran.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Graphiques par paramètres sauvegardés")

    def generate_summary_report(self):
        """
        Génère un rapport de synthèse de l'analyse.

        Cette méthode crée un fichier texte résumant tous les résultats
        de l'analyse du dataset 2D.
        """
        print("\n📝 Génération du rapport de synthèse...")

        report_path = self.output_path / "analysis_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT D'ANALYSE DU DATASET 2D D'ANNEAUX HOLOGRAPHIQUES\n")
            f.write("="*80 + "\n")
            f.write(f"Auteur: Oussama GUELFAA\n")
            f.write(f"Date: 06-01-2025\n")
            f.write(f"Dataset analysé: {self.dataset_path}\n")
            f.write("="*80 + "\n\n")

            # Statistiques générales
            f.write("1. STATISTIQUES GÉNÉRALES\n")
            f.write("-"*40 + "\n")
            f.write(f"Nombre total de fichiers: {len(self.df)}\n")
            f.write(f"Nombre de gaps uniques: {self.df['gap_um'].nunique()}\n")
            f.write(f"Nombre de L_ecran uniques: {self.df['L_ecran_um'].nunique()}\n")
            f.write(f"Taille totale du dataset: {self.df['file_size_mb'].sum():.2f} MB\n")
            f.write(f"Taille moyenne par fichier: {self.df['file_size_mb'].mean():.2f} MB\n\n")

            # Plages de paramètres
            f.write("2. PLAGES DES PARAMÈTRES\n")
            f.write("-"*40 + "\n")
            f.write(f"Gap minimum: {self.df['gap_um'].min():.4f} µm\n")
            f.write(f"Gap maximum: {self.df['gap_um'].max():.4f} µm\n")
            f.write(f"Nombre de gaps: {self.df['gap_um'].nunique()}\n")
            f.write(f"L_ecran minimum: {self.df['L_ecran_um'].min():.3f} µm\n")
            f.write(f"L_ecran maximum: {self.df['L_ecran_um'].max():.3f} µm\n")
            f.write(f"Nombre de L_ecran: {self.df['L_ecran_um'].nunique()}\n\n")

            # Complétude
            expected_combinations = self.df['gap_um'].nunique() * self.df['L_ecran_um'].nunique()
            actual_combinations = len(self.df)
            completeness = (actual_combinations / expected_combinations) * 100

            f.write("3. COMPLÉTUDE DU DATASET\n")
            f.write("-"*40 + "\n")
            f.write(f"Combinaisons attendues: {expected_combinations}\n")
            f.write(f"Combinaisons présentes: {actual_combinations}\n")
            f.write(f"Complétude: {completeness:.1f}%\n")

            if completeness < 100:
                f.write(f"⚠️  Dataset incomplet - voir missing_combinations.csv\n")
            else:
                f.write("✅ Dataset complet\n")
            f.write("\n")

            # Qualité des données
            if 'ratio_mean' in self.df.columns and not self.df['ratio_mean'].isna().all():
                f.write("4. QUALITÉ DES DONNÉES\n")
                f.write("-"*40 + "\n")
                f.write(f"Ratio d'intensité moyen: {self.df['ratio_mean'].mean():.4f}\n")
                f.write(f"Écart-type moyen: {self.df['ratio_std'].mean():.4f}\n")
                f.write(f"Valeur minimale globale: {self.df['ratio_min'].min():.4f}\n")
                f.write(f"Valeur maximale globale: {self.df['ratio_max'].max():.4f}\n\n")

            # Recommandations
            f.write("5. RECOMMANDATIONS\n")
            f.write("-"*40 + "\n")

            if completeness >= 95:
                f.write("✅ Dataset très bien structuré et complet\n")
            elif completeness >= 80:
                f.write("⚠️  Dataset majoritairement complet, quelques données manquantes\n")
            else:
                f.write("❌ Dataset incomplet, vérifier la génération des données\n")

            # Recommandations pour l'entraînement
            f.write("\nPour l'entraînement de réseaux de neurones:\n")
            f.write(f"- Nombre total d'échantillons: {len(self.df)}\n")
            f.write(f"- Recommandation train/val/test: {int(0.7*len(self.df))}/{int(0.15*len(self.df))}/{int(0.15*len(self.df))}\n")
            f.write(f"- Espace des paramètres bien couvert: {'Oui' if completeness > 90 else 'Partiellement'}\n")

            # Fichiers générés
            f.write("\n6. FICHIERS GÉNÉRÉS\n")
            f.write("-"*40 + "\n")
            f.write("- parameter_distributions.png: Distributions des paramètres\n")
            f.write("- parameter_density_2D.png: Densité 2D des paramètres\n")
            f.write("- coverage_matrix.png: Matrice de couverture\n")
            f.write("- ring_samples_grid.png: Grille d'échantillons d'anneaux\n")
            f.write("- rings_by_L_ecran.png: Évolution par L_ecran\n")
            f.write("- dataset_statistics.csv: Statistiques détaillées\n")
            f.write("- detailed_statistics.csv: Statistiques par variable\n")
            if completeness < 100:
                f.write("- missing_combinations.csv: Combinaisons manquantes\n")

        print(f"✅ Rapport de synthèse généré: {report_path}")

    def run_complete_analysis(self):
        """
        Lance l'analyse complète du dataset 2D.

        Cette méthode exécute toutes les étapes d'analyse dans l'ordre approprié.
        """
        print("🚀 DÉBUT DE L'ANALYSE COMPLÈTE DU DATASET 2D")
        print("="*60)

        try:
            # 1. Charger les informations du dataset
            self.load_dataset_info()

            if len(self.df) == 0:
                print("❌ Aucune donnée trouvée. Vérifiez le chemin du dataset.")
                return

            # 2. Analyse statistique
            self.generate_statistical_analysis()

            # 3. Graphiques de distribution
            self.plot_parameter_distributions()

            # 4. Analyse de couverture
            is_complete = self.analyze_coverage_gaps()

            # 5. Visualisation des anneaux
            self.plot_ring_samples(n_samples=36)

            # 6. Graphiques par paramètres
            self.plot_rings_by_parameter()

            # 7. Rapport de synthèse
            self.generate_summary_report()

            print("\n" + "="*60)
            print("✅ ANALYSE COMPLÈTE TERMINÉE AVEC SUCCÈS")
            print(f"📁 Tous les résultats sont disponibles dans: {self.output_path}")
            print("="*60)

            # Résumé final
            print(f"\n📊 RÉSUMÉ:")
            print(f"   • {len(self.df)} fichiers analysés")
            print(f"   • {self.df['gap_um'].nunique()} gaps × {self.df['L_ecran_um'].nunique()} L_ecran")
            print(f"   • Dataset {'complet' if is_complete else 'incomplet'}")
            print(f"   • Taille totale: {self.df['file_size_mb'].sum():.1f} MB")

        except Exception as e:
            print(f"❌ Erreur lors de l'analyse: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Fonction principale pour exécuter l'analyse du dataset 2D.
    """
    print("🔬 ANALYSEUR DE DATASET 2D D'ANNEAUX HOLOGRAPHIQUES")
    print("Auteur: Oussama GUELFAA")
    print("Date: 06-01-2025")
    print("-" * 60)

    # Vérifier que le dossier dataset existe
    dataset_path = "data_generation/dataset_2D"
    if not Path(dataset_path).exists():
        print(f"❌ Erreur: Le dossier {dataset_path} n'existe pas.")
        print("   Vérifiez que vous êtes dans le bon répertoire de travail.")
        return

    # Créer l'analyseur et lancer l'analyse
    analyzer = Dataset2DAnalyzer(dataset_path)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
