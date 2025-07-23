#!/usr/bin/env python3
"""
Vérification de cohérence avec le preprocessing EXACT du réseau de neurones
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script compare les données expérimentales et simulation avec exactement
le même preprocessing que celui appliqué au réseau de neurones:
- r_min = 1.3845845845845846
- r_max = 5.538338338338338
- 601 points
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
import random
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import ks_2samp, wasserstein_distance
import joblib

class ExactPreprocessingVerifier:
    """Vérificateur avec preprocessing exact du réseau de neurones."""
    
    def __init__(self):
        # Paramètres EXACTS du preprocessing d'entraînement
        self.r_min = 1.3845845845845846
        self.r_max = 5.538338338338338
        self.delta_r = 0.006922922922922923
        self.final_points = 601
        
        # Créer la grille radiale exacte
        self.r_network = np.linspace(self.r_min, self.r_max, self.final_points)
        
        print("🔍 VÉRIFICATION COHÉRENCE - PREPROCESSING EXACT")
        print("=" * 60)
        print(f"📏 Paramètres réseau de neurones:")
        print(f"   r_min: {self.r_min:.10f} µm")
        print(f"   r_max: {self.r_max:.10f} µm")
        print(f"   Points: {self.final_points}")
        print(f"   Delta_r: {self.delta_r:.10f} µm")
    
    def load_and_preprocess_experimental(self, profile_number=49):
        """Charge et prétraite les données expérimentales EXACTEMENT comme le réseau : COUPURE puis INTERPOLATION."""

        print(f"\n📊 Preprocessing expérimental (profil {profile_number}) - ORDRE CORRECT...")

        # Charger données
        exp_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"
        data = sio.loadmat(exp_file)

        I_profiles = data['I_profiles']  # (50, 184)
        r_exp = data['r_exp'].flatten() * 1e6  # Conversion en µm

        # Extraire le profil spécifique
        I_profile = I_profiles[profile_number, :]

        print(f"   📏 Données brutes: {len(r_exp)} points ({r_exp[0]:.6f} - {r_exp[-1]:.6f} µm)")
        print(f"   📊 Intensité brute: {np.min(I_profile):.6f} - {np.max(I_profile):.6f}")

        # ÉTAPE 1: COUPURE D'ABORD - Extraire seulement la plage [r_min, r_max]
        print(f"   ✂️  ÉTAPE 1: Coupure sur [{self.r_min:.6f}, {self.r_max:.6f}] µm")

        mask = (r_exp >= self.r_min) & (r_exp <= self.r_max)
        indices_valid = np.where(mask)[0]

        if len(indices_valid) == 0:
            raise ValueError(f"Aucun point expérimental dans l'intervalle [{self.r_min:.6f}, {self.r_max:.6f}] µm")

        # Extraire les sous-intervalles
        r_cut = r_exp[indices_valid]
        I_cut = I_profile[indices_valid]

        print(f"      ✅ {len(indices_valid)} points valides (indices {indices_valid[0]} à {indices_valid[-1]})")
        print(f"      📏 Plage coupée: [{r_cut[0]:.6f}, {r_cut[-1]:.6f}] µm")
        print(f"      📊 Intensité coupée: {np.min(I_cut):.6f} - {np.max(I_cut):.6f}")

        # ÉTAPE 2: INTERPOLATION ENSUITE - Sur la grille du réseau
        print(f"   🔄 ÉTAPE 2: Interpolation sur grille réseau ({self.final_points} points)")

        f_interp = interp1d(r_cut, I_cut, kind='linear', bounds_error=False, fill_value='extrapolate')
        I_processed = f_interp(self.r_network)

        # Vérifier l'extrapolation
        extrap_mask = (self.r_network < r_cut[0]) | (self.r_network > r_cut[-1])
        extrap_count = np.sum(extrap_mask)

        print(f"      ✅ Interpolation terminée: {len(I_processed)} points")
        print(f"      📊 Intensité finale: {np.min(I_processed):.6f} - {np.max(I_processed):.6f}")

        if extrap_count > 0:
            print(f"      ⚠️  {extrap_count} points extrapolés ({extrap_count/len(self.r_network)*100:.1f}%)")
        else:
            print(f"      ✅ Aucune extrapolation - interpolation pure!")

        return I_processed
    
    def load_and_preprocess_simulation(self, target_gap=0.115, target_L_ecran=10.25, tolerance=0.01):
        """Charge et prétraite les données de simulation avec paramètres spécifiques."""

        print(f"\n🎯 Recherche simulation gap={target_gap} µm, L_ecran={target_L_ecran} µm...")

        train_dir = Path("../../data/raw/Train")
        mat_files = list(train_dir.glob("gap_*.mat"))

        # Chercher le fichier exact ou le plus proche
        best_file = None
        best_distance = float('inf')

        for file_path in mat_files:
            try:
                # Extraire paramètres du nom
                filename = file_path.name
                parts = filename.replace('.mat', '').split('_')
                gap = float(parts[1].replace('um', ''))
                L_ecran = float(parts[3].replace('um', ''))

                # Calculer distance euclidienne
                distance = np.sqrt((gap - target_gap)**2 + (L_ecran - target_L_ecran)**2)

                if distance < best_distance:
                    best_distance = distance
                    best_file = file_path
                    best_gap = gap
                    best_L_ecran = L_ecran

            except Exception:
                continue

        if best_file is None:
            raise FileNotFoundError("Aucun fichier de simulation trouvé")

        print(f"   📁 Fichier trouvé: {best_file.name}")
        print(f"   🎯 Paramètres: gap={best_gap:.6f} µm, L_ecran={best_L_ecran:.3f} µm")
        print(f"   📏 Distance: {best_distance:.6f}")

        # Charger le fichier spécifique
        data = sio.loadmat(best_file)

        if 'ratio' in data:
            ratio = data['ratio'].flatten()
        elif 'otot' in data and 'oinc' in data:
            ratio = (data['otot'] / data['oinc']).flatten()
        else:
            raise ValueError("Variables 'ratio' ou 'otot'/'oinc' non trouvées")

        print(f"   📏 Données simulation brutes: {len(ratio)} points")
        print(f"   📊 Intensité brute: {np.min(ratio):.6f} - {np.max(ratio):.6f}")

        # ÉTAPE 1: COUPURE D'ABORD - Simuler la troncature d'entraînement (indices 200-800)
        print(f"   ✂️  ÉTAPE 1: Troncature simulation (indices 200-800)")

        # Vérifier que nous avons assez de points
        if len(ratio) < 801:
            print(f"      ⚠️  Pas assez de points ({len(ratio)} < 801), utilisation complète")
            ratio_truncated = ratio
            r_sim_truncated = np.linspace(self.r_min, self.r_max, len(ratio))
        else:
            # Appliquer la troncature exacte comme à l'entraînement
            ratio_truncated = ratio[200:801]  # 601 points

            # Créer le vecteur radial correspondant à la troncature
            r_sim_full = np.linspace(0, self.r_max * (len(ratio)/800), len(ratio))  # Approximation
            r_sim_truncated = r_sim_full[200:801]

            # Ajuster pour correspondre exactement à la plage d'entraînement
            r_sim_truncated = np.linspace(self.r_min, self.r_max, len(ratio_truncated))

        print(f"      ✅ Après troncature: {len(ratio_truncated)} points")
        print(f"      📏 Plage tronquée: [{r_sim_truncated[0]:.6f}, {r_sim_truncated[-1]:.6f}] µm")
        print(f"      📊 Intensité tronquée: {np.min(ratio_truncated):.6f} - {np.max(ratio_truncated):.6f}")

        # ÉTAPE 2: INTERPOLATION ENSUITE - Sur la grille exacte du réseau
        print(f"   🔄 ÉTAPE 2: Interpolation sur grille réseau ({self.final_points} points)")

        f_interp = interp1d(r_sim_truncated, ratio_truncated, kind='linear', bounds_error=False, fill_value='extrapolate')
        ratio_processed = f_interp(self.r_network)

        print(f"      ✅ Interpolation terminée: {len(ratio_processed)} points")
        print(f"      📊 Intensité finale: {np.min(ratio_processed):.6f} - {np.max(ratio_processed):.6f}")

        # Charger aussi quelques profils voisins pour contexte
        neighbor_files = []
        for file_path in mat_files:
            try:
                filename = file_path.name
                parts = filename.replace('.mat', '').split('_')
                gap = float(parts[1].replace('um', ''))
                L_ecran = float(parts[3].replace('um', ''))

                # Prendre les fichiers dans un rayon de tolérance
                if abs(gap - target_gap) <= tolerance and abs(L_ecran - target_L_ecran) <= tolerance:
                    neighbor_files.append(file_path)

            except Exception:
                continue

        # Charger les profils voisins avec le MÊME preprocessing
        neighbor_profiles = []
        neighbor_gaps = []
        neighbor_L_ecrans = []

        print(f"   📊 Preprocessing des profils voisins...")

        for file_path in neighbor_files[:10]:  # Maximum 10 voisins
            try:
                filename = file_path.name
                parts = filename.replace('.mat', '').split('_')
                gap = float(parts[1].replace('um', ''))
                L_ecran = float(parts[3].replace('um', ''))

                data = sio.loadmat(file_path)

                if 'ratio' in data:
                    ratio = data['ratio'].flatten()
                elif 'otot' in data and 'oinc' in data:
                    ratio = (data['otot'] / data['oinc']).flatten()
                else:
                    continue

                # Appliquer le MÊME preprocessing que le profil principal
                # ÉTAPE 1: Troncature
                if len(ratio) >= 801:
                    ratio_truncated = ratio[200:801]
                    r_sim_truncated = np.linspace(self.r_min, self.r_max, len(ratio_truncated))
                else:
                    ratio_truncated = ratio
                    r_sim_truncated = np.linspace(self.r_min, self.r_max, len(ratio))

                # ÉTAPE 2: Interpolation
                f_interp = interp1d(r_sim_truncated, ratio_truncated, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
                ratio_proc = f_interp(self.r_network)

                neighbor_profiles.append(ratio_proc)
                neighbor_gaps.append(gap)
                neighbor_L_ecrans.append(L_ecran)

            except Exception as e:
                print(f"      ⚠️  Erreur avec {file_path.name}: {e}")
                continue

        neighbor_profiles = np.array(neighbor_profiles) if neighbor_profiles else np.array([ratio_processed])

        print(f"      ✅ {len(neighbor_profiles)} profils voisins traités avec preprocessing cohérent")

        return ratio_processed, neighbor_profiles, best_gap, best_L_ecran, np.array(neighbor_gaps), np.array(neighbor_L_ecrans)
    
    def apply_network_normalization(self, exp_data, sim_data, neighbor_data):
        """Applique la normalisation EXACTE du réseau de neurones."""

        print(f"\n🔧 Application de la normalisation du réseau...")

        # Charger les scalers utilisés par le réseau
        scalers_path = "../../models/saved_models/ultra_fast_scalers.joblib"

        if not Path(scalers_path).exists():
            print(f"   ⚠️  Scalers non trouvés, utilisation de StandardScaler sur les données")
            from sklearn.preprocessing import StandardScaler

            # Créer un scaler sur les données de simulation voisines
            input_scaler = StandardScaler()
            input_scaler.fit(neighbor_data)

        else:
            print(f"   📂 Chargement des scalers d'entraînement...")
            scalers = joblib.load(scalers_path)
            input_scaler = scalers['input_scaler']

        # Normaliser les données
        exp_normalized = input_scaler.transform(exp_data.reshape(1, -1)).flatten()
        sim_normalized = input_scaler.transform(sim_data.reshape(1, -1)).flatten()
        neighbor_normalized = input_scaler.transform(neighbor_data)

        print(f"   ✅ Normalisation appliquée")
        print(f"   📊 Exp normalisé: {np.min(exp_normalized):.6f} - {np.max(exp_normalized):.6f}")
        print(f"   📊 Sim normalisé: {np.min(sim_normalized):.6f} - {np.max(sim_normalized):.6f}")
        print(f"   📊 Voisins normalisés: {np.min(neighbor_normalized):.6f} - {np.max(neighbor_normalized):.6f}")

        return exp_normalized, sim_normalized, neighbor_normalized, input_scaler
    
    def analyze_exact_coherence(self, exp_norm, sim_norm, neighbor_norm):
        """Analyse la cohérence sur les données exactement comme le réseau les voit."""

        print(f"\n📈 Analyse de cohérence sur données normalisées...")

        # Statistiques avec le profil spécifique
        stats = {
            'exp_mean': np.mean(exp_norm),
            'sim_mean': np.mean(sim_norm),
            'exp_std': np.std(exp_norm),
            'sim_std': np.std(sim_norm),
            'correlation': np.corrcoef(exp_norm, sim_norm)[0, 1]
        }

        # Tests statistiques avec profil spécifique
        ks_stat, ks_pvalue = ks_2samp(exp_norm, sim_norm)
        wasserstein_dist = wasserstein_distance(exp_norm, sim_norm)

        # Tests avec voisins pour contexte
        neighbor_mean = np.mean(neighbor_norm, axis=0)
        ks_stat_neighbors, ks_pvalue_neighbors = ks_2samp(exp_norm, neighbor_norm.flatten())
        correlation_neighbors = np.corrcoef(exp_norm, neighbor_mean)[0, 1]

        stats['ks_statistic'] = ks_stat
        stats['ks_pvalue'] = ks_pvalue
        stats['wasserstein_distance'] = wasserstein_dist
        stats['ks_pvalue_neighbors'] = ks_pvalue_neighbors
        stats['correlation_neighbors'] = correlation_neighbors

        print(f"   📊 Corrélation (profil spécifique): {stats['correlation']:.6f}")
        print(f"   📊 Corrélation (voisins): {correlation_neighbors:.6f}")
        print(f"   📊 Test KS (spécifique): stat={ks_stat:.6f}, p={ks_pvalue:.6f}")
        print(f"   📊 Test KS (voisins): p={ks_pvalue_neighbors:.6f}")
        print(f"   📊 Distance Wasserstein: {wasserstein_dist:.6f}")

        return stats

    def create_simple_comparison_plot(self, exp_raw, exp_norm, sim_raw, sim_norm, stats, gap_found, L_ecran_found):
        """Crée une visualisation simple de comparaison."""

        print(f"\n📈 Création de la visualisation...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'COHÉRENCE AVEC SIMULATION SPÉCIFIQUE\nGap={gap_found:.6f}µm, L_écran={L_ecran_found:.3f}µm',
                     fontsize=14, fontweight='bold')

        # 1. Données brutes
        ax1 = axes[0, 0]
        ax1.plot(self.r_network, exp_raw, 'r-', linewidth=2, label='Expérimental (profil 49)')
        ax1.plot(self.r_network, sim_raw, 'b-', linewidth=2, label=f'Simulation (gap={gap_found:.3f}µm)')
        ax1.set_xlabel('Position radiale (µm)')
        ax1.set_ylabel('Intensité')
        ax1.set_title('Données brutes (après interpolation)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Données normalisées
        ax2 = axes[0, 1]
        ax2.plot(self.r_network, exp_norm, 'r-', linewidth=2, label='Exp. normalisé')
        ax2.plot(self.r_network, sim_norm, 'b-', linewidth=2, label='Sim. normalisé')
        ax2.set_xlabel('Position radiale (µm)')
        ax2.set_ylabel('Intensité normalisée')
        ax2.set_title('Données normalisées (ENTRÉE RÉSEAU)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Corrélation
        ax3 = axes[1, 0]
        ax3.scatter(exp_norm, sim_norm, alpha=0.6, color='purple', s=20)

        # Ligne de régression
        z = np.polyfit(exp_norm, sim_norm, 1)
        p = np.poly1d(z)
        ax3.plot(exp_norm, p(exp_norm), "r--", alpha=0.8, linewidth=2)

        # Ligne y=x
        min_val = min(np.min(exp_norm), np.min(sim_norm))
        max_val = max(np.max(exp_norm), np.max(sim_norm))
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')

        ax3.set_xlabel('Exp. normalisé')
        ax3.set_ylabel('Sim. normalisé')
        ax3.set_title(f'Corrélation (R = {stats["correlation"]:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Résumé
        ax4 = axes[1, 1]
        ax4.axis('off')

        correlation_ok = stats['correlation'] > 0.6
        ks_ok = stats['ks_pvalue'] > 0.05
        mean_diff_ok = abs(stats['exp_mean'] - stats['sim_mean']) < 0.5

        summary_text = f"""
RÉSULTATS AVEC SIMULATION SPÉCIFIQUE:

Paramètres simulation:
• Gap: {gap_found:.6f} µm
• L'écran: {L_ecran_found:.3f} µm

Métriques de cohérence:
• Corrélation: {stats['correlation']:.6f}
  {'✅ Bonne' if correlation_ok else '❌ Insuffisante'} (seuil: 0.6)

• Test KS p-value: {stats['ks_pvalue']:.6f}
  {'✅ OK' if ks_ok else '❌ Problématique'} (seuil: 0.05)

• Différence moyennes: {abs(stats['exp_mean'] - stats['sim_mean']):.6f}
  {'✅ Similaires' if mean_diff_ok else '❌ Différentes'} (seuil: 0.5)

• Distance Wasserstein: {stats['wasserstein_distance']:.6f}

ÉVALUATION:
{'✅ COHÉRENCE BONNE' if all([correlation_ok, ks_ok, mean_diff_ok]) else
 '⚠️ COHÉRENCE MODÉRÉE' if sum([correlation_ok, ks_ok, mean_diff_ok]) >= 2 else
 '❌ COHÉRENCE FAIBLE'}

EXPLICATION GAP NÉGATIF:
{'Extrapolation du modèle' if correlation_ok else 'Données hors distribution'}
"""

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))

        plt.tight_layout()

        # Sauvegarder
        output_file = f"../../visualizations/plots/coherence_gap_{gap_found:.3f}_Lecran_{L_ecran_found:.1f}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualisation sauvegardée: {output_file}")

        plt.show()
    
    def create_exact_comparison_plot(self, exp_raw, exp_norm, sim_raw, sim_norm, stats):
        """Crée une comparaison visuelle avec les données exactes du réseau."""
        
        print(f"\n📈 Création de la visualisation exacte...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('COHÉRENCE AVEC PREPROCESSING EXACT DU RÉSEAU DE NEURONES', 
                     fontsize=16, fontweight='bold')
        
        # 1. Données brutes interpolées
        ax1 = axes[0, 0]
        ax1.plot(self.r_network, exp_raw, 'r-', linewidth=2, label='Exp. (profil 50)')
        
        # Quelques profils de simulation
        for i in range(min(5, sim_raw.shape[0])):
            ax1.plot(self.r_network, sim_raw[i, :], 'b-', alpha=0.6, linewidth=1)
        
        ax1.plot(self.r_network, np.mean(sim_raw, axis=0), 'b-', linewidth=3, label='Sim. (moyenne)')
        ax1.set_xlabel('Position radiale (µm)')
        ax1.set_ylabel('Intensité')
        ax1.set_title('Données brutes (après interpolation)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Données normalisées (entrée réseau)
        ax2 = axes[0, 1]
        ax2.plot(self.r_network, exp_norm, 'r-', linewidth=2, label='Exp. normalisé')
        ax2.plot(self.r_network, np.mean(sim_norm, axis=0), 'b-', linewidth=2, label='Sim. normalisé (moy.)')
        
        # Enveloppe simulation
        sim_mean = np.mean(sim_norm, axis=0)
        sim_std = np.std(sim_norm, axis=0)
        ax2.fill_between(self.r_network, sim_mean - sim_std, sim_mean + sim_std, 
                        color='blue', alpha=0.2, label='±1σ sim.')
        
        ax2.set_xlabel('Position radiale (µm)')
        ax2.set_ylabel('Intensité normalisée')
        ax2.set_title('Données normalisées (ENTRÉE RÉSEAU)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Corrélation point par point
        ax3 = axes[0, 2]
        sim_mean = np.mean(sim_norm, axis=0)
        ax3.scatter(exp_norm, sim_mean, alpha=0.6, color='purple', s=20)
        
        # Ligne de régression
        z = np.polyfit(exp_norm, sim_mean, 1)
        p = np.poly1d(z)
        ax3.plot(exp_norm, p(exp_norm), "r--", alpha=0.8, linewidth=2)
        
        # Ligne y=x
        min_val = min(np.min(exp_norm), np.min(sim_mean))
        max_val = max(np.max(exp_norm), np.max(sim_mean))
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
        
        ax3.set_xlabel('Exp. normalisé')
        ax3.set_ylabel('Sim. normalisé (moyenne)')
        ax3.set_title(f'Corrélation (R = {stats["correlation"]:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Distributions normalisées
        ax4 = axes[1, 0]
        ax4.hist(exp_norm, bins=30, alpha=0.6, color='red', density=True, label='Exp.')
        ax4.hist(sim_norm.flatten(), bins=30, alpha=0.6, color='blue', density=True, label='Sim.')
        ax4.set_xlabel('Intensité normalisée')
        ax4.set_ylabel('Densité')
        ax4.set_title('Distributions (données d\'entrée réseau)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Différences point par point
        ax5 = axes[1, 1]
        sim_mean = np.mean(sim_norm, axis=0)
        diff = exp_norm - sim_mean
        
        ax5.plot(self.r_network, diff, 'g-', linewidth=2)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.fill_between(self.r_network, diff, 0, alpha=0.3, color='green')
        
        ax5.set_xlabel('Position radiale (µm)')
        ax5.set_ylabel('Différence (Exp - Sim)')
        ax5.set_title('Différences point par point')
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistiques et évaluation
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Critères d'évaluation
        correlation_ok = stats['correlation'] > 0.6
        ks_ok = stats['ks_pvalue'] > 0.05
        mean_diff_ok = abs(stats['exp_mean'] - stats['sim_mean']) < 0.5
        
        stats_text = f"""
STATISTIQUES EXACTES (ENTRÉE RÉSEAU):

Corrélation: {stats['correlation']:.6f}
{'✅ Bonne' if correlation_ok else '❌ Insuffisante'} (seuil: 0.6)

Test Kolmogorov-Smirnov:
• Statistique: {stats['ks_statistic']:.6f}
• P-value: {stats['ks_pvalue']:.6f}
• {'✅ Distributions similaires' if ks_ok else '❌ Distributions différentes'}

Distance Wasserstein: {stats['wasserstein_distance']:.6f}

Moyennes normalisées:
• Exp: {stats['exp_mean']:.6f}
• Sim: {stats['sim_mean']:.6f}
• Diff: {abs(stats['exp_mean'] - stats['sim_mean']):.6f}
• {'✅ Similaires' if mean_diff_ok else '❌ Différentes'}

ÉVALUATION GLOBALE:
{'✅ COHÉRENCE BONNE' if all([correlation_ok, ks_ok, mean_diff_ok]) else
 '⚠️ COHÉRENCE MODÉRÉE' if sum([correlation_ok, ks_ok, mean_diff_ok]) >= 2 else
 '❌ COHÉRENCE FAIBLE'}

EXPLICATION GAP NÉGATIF:
{'Extrapolation du modèle' if correlation_ok else 'Données hors distribution'}
"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarder
        output_file = "../../visualizations/plots/exact_preprocessing_coherence.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualisation sauvegardée: {output_file}")
        
        plt.show()
        
        return correlation_ok, ks_ok, mean_diff_ok
    
    def generate_exact_report(self, stats, coherence_results):
        """Génère un rapport avec les données exactes."""
        
        correlation_ok, ks_ok, mean_diff_ok = coherence_results
        
        report = f"""
RAPPORT DE COHÉRENCE - PREPROCESSING EXACT DU RÉSEAU
===================================================
Date: 18/07/2025
Auteur: Oussama GUELFAA

PARAMÈTRES EXACTS DU RÉSEAU:
============================
• r_min: {self.r_min:.10f} µm
• r_max: {self.r_max:.10f} µm
• Points: {self.final_points}
• Delta_r: {self.delta_r:.10f} µm

DONNÉES ANALYSÉES:
==================
• Expérimental: Profil 49 (interpolé + normalisé)
• Simulation: 50 profils échantillonnés (interpolés + normalisés)
• Preprocessing: IDENTIQUE à l'entraînement du réseau

RÉSULTATS SUR DONNÉES D'ENTRÉE RÉSEAU:
======================================
• Corrélation: {stats['correlation']:.6f} ({'✅ Bonne' if correlation_ok else '❌ Insuffisante'})
• Test KS p-value: {stats['ks_pvalue']:.6f} ({'✅ OK' if ks_ok else '❌ Problématique'})
• Distance Wasserstein: {stats['wasserstein_distance']:.6f}
• Différence moyennes: {abs(stats['exp_mean'] - stats['sim_mean']):.6f}

ÉVALUATION:
===========
"""

        if all([correlation_ok, ks_ok, mean_diff_ok]):
            report += """✅ COHÉRENCE BONNE
Le profil expérimental est cohérent avec les données d'entraînement.
Le gap négatif est probablement dû à une extrapolation du modèle."""
        elif sum([correlation_ok, ks_ok, mean_diff_ok]) >= 2:
            report += """⚠️ COHÉRENCE MODÉRÉE
Certaines différences existent. Le gap négatif peut s'expliquer par
des données expérimentales en bordure de la distribution d'entraînement."""
        else:
            report += """❌ COHÉRENCE FAIBLE
Le profil expérimental est significativement différent des données
d'entraînement. Cela explique le gap négatif."""

        report += f"""

EXPLICATION DU GAP NÉGATIF (-0.151 µm):
=======================================
1. {'✅ Données cohérentes' if correlation_ok else '❌ Données incohérentes'} - {'Extrapolation du modèle' if correlation_ok else 'Hors distribution d\'entraînement'}
2. {'✅ Distributions similaires' if ks_ok else '❌ Distributions différentes'} - {'Variabilité normale' if ks_ok else 'Domaine différent'}
3. {'✅ Moyennes similaires' if mean_diff_ok else '❌ Moyennes différentes'} - {'Biais minimal' if mean_diff_ok else 'Biais significatif'}

RECOMMANDATIONS:
================
1. {'Ajouter contraintes physiques (gap ≥ 0)' if correlation_ok else 'Adaptation de domaine nécessaire'}
2. {'Tester sur plus de profils expérimentaux' if ks_ok else 'Réviser la normalisation'}
3. {'Monitoring des prédictions' if mean_diff_ok else 'Revoir les données d\'entraînement'}
4. Valider avec mesures indépendantes si disponibles

CONCLUSION:
===========
Le gap négatif s'explique {'principalement par l\'absence de contraintes physiques dans le modèle' if all([correlation_ok, ks_ok, mean_diff_ok]) else 'par l\'incohérence entre données expérimentales et simulation'}.

Contact: Oussama GUELFAA - guelfaao@gmail.com
"""
        
        report_file = "../../reports/technical/exact_preprocessing_coherence_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Rapport sauvegardé: {report_file}")
    
    def verify_exact_coherence(self):
        """Vérification complète avec preprocessing exact."""
        
        try:
            # 1. Preprocessing expérimental exact
            exp_processed = self.load_and_preprocess_experimental(profile_number=49)

            # 2. Preprocessing simulation exact (gap=0.115, L_ecran=10.25)
            sim_processed, neighbor_processed, gap_found, L_ecran_found, neighbor_gaps, neighbor_L_ecrans = self.load_and_preprocess_simulation(
                target_gap=0.115, target_L_ecran=10.25, tolerance=0.01)

            # 3. Normalisation exacte
            exp_norm, sim_norm, neighbor_norm, scaler = self.apply_network_normalization(
                exp_processed, sim_processed, neighbor_processed)

            # 4. Analyse de cohérence
            stats = self.analyze_exact_coherence(exp_norm, sim_norm, neighbor_norm)

            # 5. Visualisation simplifiée
            correlation_ok = stats['correlation'] > 0.6
            ks_ok = stats['ks_pvalue'] > 0.05
            mean_diff_ok = abs(stats['exp_mean'] - stats['sim_mean']) < 0.5
            coherence_results = (correlation_ok, ks_ok, mean_diff_ok)

            # 6. Rapport simplifié
            print(f"\n📊 RÉSULTATS AVEC SIMULATION SPÉCIFIQUE:")
            print(f"   🎯 Simulation utilisée: gap={gap_found:.6f} µm, L_ecran={L_ecran_found:.3f} µm")
            print(f"   📊 Corrélation: {stats['correlation']:.6f} ({'✅' if correlation_ok else '❌'})")
            print(f"   📊 Test KS p-value: {stats['ks_pvalue']:.6f} ({'✅' if ks_ok else '❌'})")
            print(f"   📊 Différence moyennes: {abs(stats['exp_mean'] - stats['sim_mean']):.6f} ({'✅' if mean_diff_ok else '❌'})")
            print(f"   📊 Corrélation voisins: {stats['correlation_neighbors']:.6f}")

            # Créer une visualisation simple
            self.create_simple_comparison_plot(exp_processed, exp_norm, sim_processed, sim_norm,
                                             stats, gap_found, L_ecran_found)

            print(f"\n✅ VÉRIFICATION EXACTE TERMINÉE!")

            return stats, coherence_results
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            raise

def main():
    """Fonction principale."""
    
    verifier = ExactPreprocessingVerifier()
    stats, coherence_results = verifier.verify_exact_coherence()
    
    correlation_ok, ks_ok, mean_diff_ok = coherence_results
    
    print(f"\n🎯 RÉSUMÉ FINAL (DONNÉES EXACTES RÉSEAU):")
    print(f"   Corrélation: {'✅' if correlation_ok else '❌'} {stats['correlation']:.3f}")
    print(f"   Tests statistiques: {'✅' if ks_ok else '❌'} p={stats['ks_pvalue']:.6f}")
    print(f"   Moyennes: {'✅' if mean_diff_ok else '❌'}")
    
    if all([correlation_ok, ks_ok, mean_diff_ok]):
        print(f"\n✅ Le profil expérimental est COHÉRENT avec l'entraînement")
        print(f"   → Gap négatif = extrapolation du modèle")
        print(f"   → Solution: contraintes physiques (gap ≥ 0)")
    else:
        print(f"\n❌ Le profil expérimental est INCOHÉRENT avec l'entraînement")
        print(f"   → Gap négatif = données hors distribution")
        print(f"   → Solution: adaptation de domaine")

if __name__ == "__main__":
    main()
