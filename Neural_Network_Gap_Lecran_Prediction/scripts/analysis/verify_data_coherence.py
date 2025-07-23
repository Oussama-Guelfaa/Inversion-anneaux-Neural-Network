#!/usr/bin/env python3
"""
Vérification de la cohérence entre données expérimentales et simulation
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script compare en détail les caractéristiques des données expérimentales
vs simulation pour identifier les sources d'incohérence.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
import random
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import ks_2samp, wasserstein_distance
import seaborn as sns

class DataCoherenceVerifier:
    """Vérificateur de cohérence des données."""
    
    def __init__(self):
        self.exp_data = None
        self.sim_data = None
        self.r_exp = None
        self.r_sim = None
        
        print("🔍 VÉRIFICATEUR DE COHÉRENCE DES DONNÉES")
        print("=" * 50)
    
    def load_experimental_data(self):
        """Charge les données expérimentales."""
        
        print("📊 Chargement des données expérimentales...")
        
        # Charger le profil 50 spécifiquement testé
        exp_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"
        
        if not Path(exp_file).exists():
            raise FileNotFoundError(f"Fichier expérimental non trouvé: {exp_file}")
        
        data = sio.loadmat(exp_file)
        
        self.exp_data = data['I_profiles']  # (50, 184)
        self.r_exp = data['r_exp'].flatten() * 1e6  # Conversion en µm
        
        print(f"   ✅ {self.exp_data.shape[0]} profils expérimentaux chargés")
        print(f"   📏 Plage radiale exp: {self.r_exp[0]:.6f} - {self.r_exp[-1]:.6f} µm")
        print(f"   📊 Intensité exp: {np.min(self.exp_data):.6f} - {np.max(self.exp_data):.6f}")
        
        return self.exp_data, self.r_exp
    
    def load_simulation_data(self, n_samples=100):
        """Charge un échantillon de données de simulation."""
        
        print(f"🎯 Chargement de {n_samples} échantillons de simulation...")
        
        train_dir = Path("../../data/raw/Train")
        mat_files = list(train_dir.glob("gap_*.mat"))
        
        if len(mat_files) == 0:
            raise FileNotFoundError("Aucun fichier de simulation trouvé")
        
        # Échantillonner aléatoirement
        selected_files = random.sample(mat_files, min(n_samples, len(mat_files)))
        
        sim_profiles = []
        gaps = []
        L_ecrans = []
        
        for file_path in selected_files:
            try:
                # Extraire paramètres du nom
                filename = file_path.name
                parts = filename.replace('.mat', '').split('_')
                gap = float(parts[1].replace('um', ''))
                L_ecran = float(parts[3].replace('um', ''))
                
                # Charger données
                data = sio.loadmat(file_path)
                
                if 'ratio' in data:
                    ratio = data['ratio'].flatten()
                elif 'otot' in data and 'oinc' in data:
                    ratio = (data['otot'] / data['oinc']).flatten()
                else:
                    continue
                
                sim_profiles.append(ratio)
                gaps.append(gap)
                L_ecrans.append(L_ecran)
                
            except Exception as e:
                print(f"   ⚠️  Erreur avec {file_path.name}: {e}")
                continue
        
        self.sim_data = np.array(sim_profiles)
        self.gaps = np.array(gaps)
        self.L_ecrans = np.array(L_ecrans)
        
        # Créer vecteur radial simulation (approximation)
        self.r_sim = np.linspace(0, self.r_exp[-1], self.sim_data.shape[1])
        
        print(f"   ✅ {len(sim_profiles)} profils de simulation chargés")
        print(f"   📏 Plage radiale sim: {self.r_sim[0]:.6f} - {self.r_sim[-1]:.6f} µm")
        print(f"   📊 Intensité sim: {np.min(self.sim_data):.6f} - {np.max(self.sim_data):.6f}")
        print(f"   🎯 Gap range: {np.min(self.gaps):.6f} - {np.max(self.gaps):.6f} µm")
        print(f"   🎯 L'écran range: {np.min(self.L_ecrans):.3f} - {np.max(self.L_ecrans):.3f} µm")
        
        return self.sim_data, self.r_sim, self.gaps, self.L_ecrans
    
    def interpolate_to_common_grid(self):
        """Interpole les données sur une grille commune."""
        
        print("🔄 Interpolation sur grille commune...")
        
        # Utiliser la grille expérimentale comme référence
        r_common = self.r_exp
        
        # Interpoler les données de simulation
        sim_interp = np.zeros((self.sim_data.shape[0], len(r_common)))
        
        for i in range(self.sim_data.shape[0]):
            f = interp1d(self.r_sim, self.sim_data[i, :], kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
            sim_interp[i, :] = f(r_common)
        
        print(f"   ✅ Interpolation terminée: {sim_interp.shape}")
        
        return self.exp_data, sim_interp, r_common
    
    def analyze_statistical_differences(self, exp_data, sim_data):
        """Analyse les différences statistiques."""
        
        print("📈 Analyse des différences statistiques...")
        
        # Statistiques globales
        exp_flat = exp_data.flatten()
        sim_flat = sim_data.flatten()
        
        stats = {
            'exp_mean': np.mean(exp_flat),
            'sim_mean': np.mean(sim_flat),
            'exp_std': np.std(exp_flat),
            'sim_std': np.std(sim_flat),
            'exp_min': np.min(exp_flat),
            'sim_min': np.min(sim_flat),
            'exp_max': np.max(exp_flat),
            'sim_max': np.max(sim_flat),
        }
        
        # Tests statistiques
        ks_stat, ks_pvalue = ks_2samp(exp_flat, sim_flat)
        wasserstein_dist = wasserstein_distance(exp_flat, sim_flat)
        
        stats['ks_statistic'] = ks_stat
        stats['ks_pvalue'] = ks_pvalue
        stats['wasserstein_distance'] = wasserstein_dist
        
        print(f"   📊 Moyennes: Exp={stats['exp_mean']:.6f}, Sim={stats['sim_mean']:.6f}")
        print(f"   📊 Écarts-types: Exp={stats['exp_std']:.6f}, Sim={stats['sim_std']:.6f}")
        print(f"   📊 Test KS: statistic={ks_stat:.6f}, p-value={ks_pvalue:.6f}")
        print(f"   📊 Distance Wasserstein: {wasserstein_dist:.6f}")
        
        return stats
    
    def analyze_spectral_content(self, exp_data, sim_data, r_common):
        """Analyse le contenu spectral."""
        
        print("🌊 Analyse du contenu spectral...")
        
        # Profils moyens
        exp_mean = np.mean(exp_data, axis=0)
        sim_mean = np.mean(sim_data, axis=0)
        
        # FFT
        exp_fft = np.fft.fft(exp_mean)
        sim_fft = np.fft.fft(sim_mean)
        
        freqs = np.fft.fftfreq(len(exp_mean), d=np.diff(r_common)[0])
        
        # Puissance spectrale
        exp_power = np.abs(exp_fft)**2
        sim_power = np.abs(sim_fft)**2
        
        # Fréquences dominantes
        exp_peak_freq = freqs[np.argmax(exp_power[1:len(freqs)//2]) + 1]
        sim_peak_freq = freqs[np.argmax(sim_power[1:len(freqs)//2]) + 1]
        
        print(f"   🌊 Fréquence dominante exp: {exp_peak_freq:.3f} 1/µm")
        print(f"   🌊 Fréquence dominante sim: {sim_peak_freq:.3f} 1/µm")
        
        return freqs, exp_power, sim_power, exp_peak_freq, sim_peak_freq
    
    def analyze_ring_structure(self, exp_data, sim_data, r_common):
        """Analyse la structure des anneaux."""
        
        print("🎯 Analyse de la structure des anneaux...")
        
        # Profils moyens
        exp_mean = np.mean(exp_data, axis=0)
        sim_mean = np.mean(sim_data, axis=0)
        
        # Détection des pics
        exp_peaks, exp_props = find_peaks(exp_mean, height=np.mean(exp_mean)*0.8, distance=5)
        sim_peaks, sim_props = find_peaks(sim_mean, height=np.mean(sim_mean)*0.8, distance=5)
        
        # Positions des anneaux
        exp_ring_positions = r_common[exp_peaks]
        sim_ring_positions = r_common[sim_peaks]
        
        print(f"   🎯 Anneaux exp: {len(exp_peaks)} détectés")
        print(f"   🎯 Anneaux sim: {len(sim_peaks)} détectés")
        print(f"   📏 Positions exp: {exp_ring_positions}")
        print(f"   📏 Positions sim: {sim_ring_positions}")
        
        # Espacement entre anneaux
        if len(exp_ring_positions) > 1:
            exp_spacing = np.diff(exp_ring_positions)
            exp_mean_spacing = np.mean(exp_spacing)
        else:
            exp_spacing = []
            exp_mean_spacing = 0
        
        if len(sim_ring_positions) > 1:
            sim_spacing = np.diff(sim_ring_positions)
            sim_mean_spacing = np.mean(sim_spacing)
        else:
            sim_spacing = []
            sim_mean_spacing = 0
        
        print(f"   📏 Espacement moyen exp: {exp_mean_spacing:.6f} µm")
        print(f"   📏 Espacement moyen sim: {sim_mean_spacing:.6f} µm")
        
        return {
            'exp_peaks': exp_peaks,
            'sim_peaks': sim_peaks,
            'exp_ring_positions': exp_ring_positions,
            'sim_ring_positions': sim_ring_positions,
            'exp_mean_spacing': exp_mean_spacing,
            'sim_mean_spacing': sim_mean_spacing
        }
    
    def create_coherence_visualization(self, exp_data, sim_data, r_common, stats, spectral_data, ring_data):
        """Crée une visualisation complète de la cohérence."""
        
        print("📈 Création de la visualisation de cohérence...")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('VÉRIFICATION COHÉRENCE DONNÉES EXPÉRIMENTALES vs SIMULATION', 
                     fontsize=16, fontweight='bold')
        
        # 1. Profils moyens
        ax1 = axes[0, 0]
        exp_mean = np.mean(exp_data, axis=0)
        sim_mean = np.mean(sim_data, axis=0)
        
        ax1.plot(r_common, exp_mean, 'r-', linewidth=2, label='Expérimental')
        ax1.plot(r_common, sim_mean, 'b-', linewidth=2, label='Simulation')
        ax1.set_xlabel('Position radiale (µm)')
        ax1.set_ylabel('Intensité')
        ax1.set_title('Profils moyens')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distributions d'intensité
        ax2 = axes[0, 1]
        ax2.hist(exp_data.flatten(), bins=50, alpha=0.6, color='red', 
                density=True, label='Expérimental')
        ax2.hist(sim_data.flatten(), bins=50, alpha=0.6, color='blue', 
                density=True, label='Simulation')
        ax2.set_xlabel('Intensité')
        ax2.set_ylabel('Densité')
        ax2.set_title('Distributions d\'intensité')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Analyse spectrale
        ax3 = axes[0, 2]
        freqs, exp_power, sim_power, exp_peak_freq, sim_peak_freq = spectral_data
        
        # Afficher seulement les fréquences positives
        pos_freqs = freqs[:len(freqs)//2]
        ax3.semilogy(pos_freqs, exp_power[:len(freqs)//2], 'r-', label='Expérimental')
        ax3.semilogy(pos_freqs, sim_power[:len(freqs)//2], 'b-', label='Simulation')
        ax3.set_xlabel('Fréquence (1/µm)')
        ax3.set_ylabel('Puissance spectrale')
        ax3.set_title('Contenu spectral')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Structure des anneaux
        ax4 = axes[1, 0]
        ax4.plot(r_common, exp_mean, 'r-', linewidth=2, label='Expérimental')
        ax4.plot(r_common[ring_data['exp_peaks']], exp_mean[ring_data['exp_peaks']], 
                'ro', markersize=8, label=f'{len(ring_data["exp_peaks"])} anneaux exp.')
        
        ax4.plot(r_common, sim_mean, 'b-', linewidth=2, label='Simulation')
        ax4.plot(r_common[ring_data['sim_peaks']], sim_mean[ring_data['sim_peaks']], 
                'bs', markersize=8, label=f'{len(ring_data["sim_peaks"])} anneaux sim.')
        
        ax4.set_xlabel('Position radiale (µm)')
        ax4.set_ylabel('Intensité')
        ax4.set_title('Structure des anneaux')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Corrélation point par point
        ax5 = axes[1, 1]
        ax5.scatter(exp_mean, sim_mean, alpha=0.6, color='purple')
        
        # Ligne de régression
        z = np.polyfit(exp_mean, sim_mean, 1)
        p = np.poly1d(z)
        ax5.plot(exp_mean, p(exp_mean), "r--", alpha=0.8)
        
        correlation = np.corrcoef(exp_mean, sim_mean)[0, 1]
        ax5.set_xlabel('Intensité expérimentale')
        ax5.set_ylabel('Intensité simulation')
        ax5.set_title(f'Corrélation (R = {correlation:.3f})')
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistiques comparatives
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        stats_text = f"""
STATISTIQUES COMPARATIVES:

Moyennes:
• Exp: {stats['exp_mean']:.6f}
• Sim: {stats['sim_mean']:.6f}
• Différence: {abs(stats['exp_mean'] - stats['sim_mean']):.6f}

Écarts-types:
• Exp: {stats['exp_std']:.6f}
• Sim: {stats['sim_std']:.6f}
• Ratio: {stats['exp_std']/stats['sim_std']:.3f}

Tests statistiques:
• KS statistic: {stats['ks_statistic']:.6f}
• KS p-value: {stats['ks_pvalue']:.6f}
• Wasserstein: {stats['wasserstein_distance']:.6f}

Anneaux:
• Exp: {len(ring_data['exp_peaks'])} anneaux
• Sim: {len(ring_data['sim_peaks'])} anneaux
• Espacement exp: {ring_data['exp_mean_spacing']:.6f} µm
• Espacement sim: {ring_data['sim_mean_spacing']:.6f} µm
"""
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 7. Variabilité par position radiale
        ax7 = axes[2, 0]
        exp_std_profile = np.std(exp_data, axis=0)
        sim_std_profile = np.std(sim_data, axis=0)
        
        ax7.plot(r_common, exp_std_profile, 'r-', linewidth=2, label='Exp. std')
        ax7.plot(r_common, sim_std_profile, 'b-', linewidth=2, label='Sim. std')
        ax7.set_xlabel('Position radiale (µm)')
        ax7.set_ylabel('Écart-type')
        ax7.set_title('Variabilité spatiale')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Heatmap des différences
        ax8 = axes[2, 1]
        
        # Calculer la différence moyenne
        diff_profile = exp_mean - sim_mean
        
        # Créer une heatmap 2D (artificielle pour visualisation)
        diff_2d = np.tile(diff_profile, (10, 1))
        
        im = ax8.imshow(diff_2d, aspect='auto', cmap='RdBu_r', 
                       extent=[r_common[0], r_common[-1], 0, 1])
        plt.colorbar(im, ax=ax8, label='Différence (Exp - Sim)')
        ax8.set_xlabel('Position radiale (µm)')
        ax8.set_title('Carte des différences')
        
        # 9. Évaluation globale
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # Critères d'évaluation
        correlation_ok = correlation > 0.6
        ks_ok = stats['ks_pvalue'] > 0.05
        mean_diff_ok = abs(stats['exp_mean'] - stats['sim_mean']) < 0.1
        rings_ok = abs(len(ring_data['exp_peaks']) - len(ring_data['sim_peaks'])) <= 2
        
        evaluation_text = f"""
ÉVALUATION GLOBALE:

Corrélation: {'✅' if correlation_ok else '❌'} {correlation:.3f}
Test KS: {'✅' if ks_ok else '❌'} p={stats['ks_pvalue']:.6f}
Différence moyennes: {'✅' if mean_diff_ok else '❌'} {abs(stats['exp_mean'] - stats['sim_mean']):.6f}
Nombre d'anneaux: {'✅' if rings_ok else '❌'} {len(ring_data['exp_peaks'])} vs {len(ring_data['sim_peaks'])}

COHÉRENCE GLOBALE:
{'✅ BONNE' if all([correlation_ok, ks_ok, mean_diff_ok, rings_ok]) else 
 '⚠️ MODÉRÉE' if sum([correlation_ok, ks_ok, mean_diff_ok, rings_ok]) >= 2 else 
 '❌ FAIBLE'}

RECOMMANDATIONS:
{'• Données cohérentes pour IA' if all([correlation_ok, ks_ok, mean_diff_ok, rings_ok]) else
 '• Adaptation de domaine recommandée' if sum([correlation_ok, ks_ok, mean_diff_ok, rings_ok]) >= 2 else
 '• Révision complète nécessaire'}
"""
        
        ax9.text(0.05, 0.95, evaluation_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarder
        output_file = "../../visualizations/plots/data_coherence_verification.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualisation sauvegardée: {output_file}")
        
        plt.show()
        
        return correlation_ok, ks_ok, mean_diff_ok, rings_ok
    
    def generate_coherence_report(self, stats, ring_data, coherence_results):
        """Génère un rapport de cohérence."""
        
        correlation_ok, ks_ok, mean_diff_ok, rings_ok = coherence_results
        
        report = f"""
RAPPORT DE VÉRIFICATION DE COHÉRENCE
====================================
Date: 18/07/2025
Auteur: Oussama GUELFAA

DONNÉES ANALYSÉES:
==================
• Expérimentales: 50 profils PS 3µm (184 points)
• Simulation: 100 profils échantillonnés (1000 points → interpolés)
• Plage radiale commune: {self.r_exp[0]:.6f} - {self.r_exp[-1]:.6f} µm

STATISTIQUES COMPARATIVES:
==========================
• Moyenne exp: {stats['exp_mean']:.6f}
• Moyenne sim: {stats['sim_mean']:.6f}
• Différence relative: {abs(stats['exp_mean'] - stats['sim_mean'])/stats['exp_mean']*100:.2f}%

• Écart-type exp: {stats['exp_std']:.6f}
• Écart-type sim: {stats['sim_std']:.6f}
• Ratio variabilité: {stats['exp_std']/stats['sim_std']:.3f}

TESTS STATISTIQUES:
===================
• Test Kolmogorov-Smirnov: {stats['ks_statistic']:.6f} (p={stats['ks_pvalue']:.6f})
• Distance Wasserstein: {stats['wasserstein_distance']:.6f}
• Interprétation KS: {'Distributions similaires' if ks_ok else 'Distributions différentes'}

STRUCTURE DES ANNEAUX:
======================
• Anneaux expérimentaux: {len(ring_data['exp_peaks'])}
• Anneaux simulation: {len(ring_data['sim_peaks'])}
• Espacement moyen exp: {ring_data['exp_mean_spacing']:.6f} µm
• Espacement moyen sim: {ring_data['sim_mean_spacing']:.6f} µm
• Cohérence structure: {'✅ Bonne' if rings_ok else '❌ Problématique'}

ÉVALUATION GLOBALE:
===================
• Corrélation profils moyens: {'✅ Acceptable' if correlation_ok else '❌ Insuffisante'}
• Cohérence statistique: {'✅ Bonne' if ks_ok else '❌ Problématique'}
• Similarité moyennes: {'✅ Bonne' if mean_diff_ok else '❌ Problématique'}
• Structure anneaux: {'✅ Cohérente' if rings_ok else '❌ Incohérente'}

CONCLUSION:
===========
"""

        if all([correlation_ok, ks_ok, mean_diff_ok, rings_ok]):
            report += """✅ COHÉRENCE EXCELLENTE
Les données expérimentales et de simulation sont très cohérentes.
Le modèle devrait bien fonctionner sur les données expérimentales."""
        elif sum([correlation_ok, ks_ok, mean_diff_ok, rings_ok]) >= 2:
            report += """⚠️ COHÉRENCE MODÉRÉE
Certaines différences existent entre expérimental et simulation.
Une adaptation de domaine est recommandée."""
        else:
            report += """❌ COHÉRENCE FAIBLE
Différences significatives entre expérimental et simulation.
Révision complète de l'approche nécessaire."""

        report += f"""

RECOMMANDATIONS:
================
1. {'Utiliser directement le modèle' if all([correlation_ok, ks_ok, mean_diff_ok, rings_ok]) else 'Implémenter adaptation de domaine'}
2. {'Valider sur plus de profils expérimentaux' if correlation_ok else 'Revoir la normalisation des données'}
3. {'Contraintes physiques recommandées' if not mean_diff_ok else 'Monitoring des prédictions'}
4. {'Tests de robustesse' if rings_ok else 'Analyse approfondie des différences structurelles'}

EXPLICATION DU GAP NÉGATIF:
===========================
Le gap négatif observé (-0.151 µm) peut s'expliquer par:
• {'Distribution expérimentale hors plage d\'entraînement' if not ks_ok else 'Problème de normalisation'}
• {'Structure d\'anneaux différente' if not rings_ok else 'Extrapolation du modèle'}
• {'Absence de contraintes physiques dans le modèle' if not mean_diff_ok else 'Variabilité expérimentale'}

Contact: Oussama GUELFAA - guelfaao@gmail.com
"""
        
        report_file = "../../reports/technical/data_coherence_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Rapport sauvegardé: {report_file}")
    
    def verify_coherence(self):
        """Vérification complète de la cohérence."""
        
        try:
            # 1. Charger les données
            exp_data, r_exp = self.load_experimental_data()
            sim_data, r_sim, gaps, L_ecrans = self.load_simulation_data(n_samples=100)
            
            # 2. Interpoler sur grille commune
            exp_interp, sim_interp, r_common = self.interpolate_to_common_grid()
            
            # 3. Analyses statistiques
            stats = self.analyze_statistical_differences(exp_interp, sim_interp)
            
            # 4. Analyse spectrale
            spectral_data = self.analyze_spectral_content(exp_interp, sim_interp, r_common)
            
            # 5. Analyse des anneaux
            ring_data = self.analyze_ring_structure(exp_interp, sim_interp, r_common)
            
            # 6. Visualisation
            coherence_results = self.create_coherence_visualization(
                exp_interp, sim_interp, r_common, stats, spectral_data, ring_data)
            
            # 7. Rapport
            self.generate_coherence_report(stats, ring_data, coherence_results)
            
            print(f"\n✅ VÉRIFICATION DE COHÉRENCE TERMINÉE!")
            
            return stats, ring_data, coherence_results
            
        except Exception as e:
            print(f"❌ Erreur pendant la vérification: {e}")
            raise

def main():
    """Fonction principale."""
    
    verifier = DataCoherenceVerifier()
    stats, ring_data, coherence_results = verifier.verify_coherence()
    
    correlation_ok, ks_ok, mean_diff_ok, rings_ok = coherence_results
    
    print(f"\n🎯 RÉSUMÉ DE LA COHÉRENCE:")
    print(f"   Corrélation: {'✅' if correlation_ok else '❌'}")
    print(f"   Tests statistiques: {'✅' if ks_ok else '❌'}")
    print(f"   Moyennes: {'✅' if mean_diff_ok else '❌'}")
    print(f"   Structure anneaux: {'✅' if rings_ok else '❌'}")
    
    if all([correlation_ok, ks_ok, mean_diff_ok, rings_ok]):
        print(f"\n✅ Données cohérentes - Gap négatif probablement dû à l'extrapolation")
    elif sum([correlation_ok, ks_ok, mean_diff_ok, rings_ok]) >= 2:
        print(f"\n⚠️ Cohérence modérée - Adaptation de domaine recommandée")
    else:
        print(f"\n❌ Cohérence faible - Révision complète nécessaire")

if __name__ == "__main__":
    main()
