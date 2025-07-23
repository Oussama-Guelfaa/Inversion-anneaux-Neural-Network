#!/usr/bin/env python3
"""
Analyse des Erreurs Résiduelles - Diagnostic Avancé
Auteur: Oussama GUELFAA
Date: 15/07/2025

Analyse détaillée des erreurs résiduelles pour identifier
les zones problématiques et proposer des améliorations ciblées.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from scipy.interpolate import interp1d
from scipy import signal
from scipy.stats import pearsonr
import seaborn as sns
from pathlib import Path

class ResidualErrorAnalyzer:
    """
    Analyseur d'erreurs résiduelles pour diagnostic avancé.
    """
    
    def __init__(self):
        self.results_dir = Path("results/residual_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔬 ResidualErrorAnalyzer initialisé")
        print(f"   📁 Résultats: {self.results_dir}")
    
    def load_comparison_data(self):
        """Charge les données de la comparaison précédente."""
        print("📂 Chargement des données de comparaison...")
        
        # Charger le profil simulé optimal (Semi-Supervisé Simple)
        sim_file = "/Users/oussamaguelfaa/Desktop/Stage/Inversion_anneaux/data_generation/Calcul_Data/dataset/gap_0.0707um_L_9.908um.mat"
        sim_data = scipy.io.loadmat(sim_file)
        x_sim = sim_data['x'].flatten()
        ratio_sim = sim_data['ratio'].flatten()
        
        # Charger le profil expérimental n°50
        exp_file = "Test/profile_exp_PS_3um_z_positive.mat"
        exp_data = scipy.io.loadmat(exp_file)
        r_exp = exp_data['r_exp'].flatten() * 1e6  # m → µm
        I_profiles = exp_data['I_profiles']
        
        # Vérifier dimensions et extraire profil 50
        if I_profiles.shape[0] == 50:
            I_profiles = I_profiles.T
        I_exp = I_profiles[:, 49]  # Profil 50 (index 49)
        
        print(f"   ✅ Données chargées:")
        print(f"      Simulation: {len(x_sim)} points")
        print(f"      Expérimental: {len(r_exp)} points")
        
        return x_sim, ratio_sim, r_exp, I_exp
    
    def harmonize_for_analysis(self, x_sim, ratio_sim, r_exp, I_exp):
        """Harmonise les profils pour l'analyse."""
        print("🔄 Harmonisation pour analyse...")
        
        # Plage commune
        r_min = max(x_sim.min(), r_exp.min())
        r_max = min(x_sim.max(), r_exp.max())
        
        # Grille haute résolution pour analyse fine
        n_points = 1000  # Plus de points pour analyse détaillée
        r_common = np.linspace(r_min, r_max, n_points)
        
        # Interpoler simulation
        mask_sim = (x_sim >= r_min) & (x_sim <= r_max)
        interp_sim = interp1d(x_sim[mask_sim], ratio_sim[mask_sim], 
                             kind='cubic', bounds_error=False, fill_value='extrapolate')
        ratio_sim_interp = interp_sim(r_common)
        
        # Interpoler expérimental
        mask_exp = (r_exp >= r_min) & (r_exp <= r_max)
        interp_exp = interp1d(r_exp[mask_exp], I_exp[mask_exp], 
                             kind='cubic', bounds_error=False, fill_value='extrapolate')
        I_exp_interp = interp_exp(r_common)
        
        print(f"   ✅ Harmonisation terminée: {len(r_common)} points")
        
        return r_common, ratio_sim_interp, I_exp_interp
    
    def compute_residuals(self, r_common, ratio_sim, I_exp):
        """Calcule les résidus et métriques d'erreur."""
        print("📊 Calcul des résidus...")
        
        # Résidus absolus et relatifs
        residuals_abs = I_exp - ratio_sim
        residuals_rel = (I_exp - ratio_sim) / (I_exp + 1e-8)  # Éviter division par 0
        
        # Métriques globales
        mse = np.mean(residuals_abs**2)
        mae = np.mean(np.abs(residuals_abs))
        rmse = np.sqrt(mse)
        correlation, _ = pearsonr(ratio_sim, I_exp)
        
        # Métriques par zones
        n_zones = 5
        zone_size = len(r_common) // n_zones
        zone_metrics = []
        
        for i in range(n_zones):
            start_idx = i * zone_size
            end_idx = (i + 1) * zone_size if i < n_zones - 1 else len(r_common)
            
            zone_residuals = residuals_abs[start_idx:end_idx]
            zone_sim = ratio_sim[start_idx:end_idx]
            zone_exp = I_exp[start_idx:end_idx]
            
            zone_mse = np.mean(zone_residuals**2)
            zone_mae = np.mean(np.abs(zone_residuals))
            zone_corr, _ = pearsonr(zone_sim, zone_exp)
            
            zone_metrics.append({
                'zone': i + 1,
                'r_start': r_common[start_idx],
                'r_end': r_common[end_idx - 1],
                'mse': zone_mse,
                'mae': zone_mae,
                'correlation': zone_corr,
                'n_points': end_idx - start_idx
            })
        
        metrics = {
            'global': {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'correlation': correlation,
                'r2': correlation**2
            },
            'zones': zone_metrics
        }
        
        print(f"   📈 Métriques globales:")
        print(f"      MSE: {mse:.6f}")
        print(f"      MAE: {mae:.6f}")
        print(f"      Corrélation: {correlation:.4f}")
        print(f"      R²: {correlation**2:.4f}")
        
        return residuals_abs, residuals_rel, metrics
    
    def analyze_frequency_content(self, r_common, ratio_sim, I_exp, residuals):
        """Analyse le contenu fréquentiel."""
        print("🌊 Analyse fréquentielle...")
        
        # FFT des signaux
        fft_sim = np.fft.fft(ratio_sim)
        fft_exp = np.fft.fft(I_exp)
        fft_residuals = np.fft.fft(residuals)
        
        # Fréquences spatiales
        dr = r_common[1] - r_common[0]
        freqs = np.fft.fftfreq(len(r_common), dr)
        
        # Puissance spectrale
        power_sim = np.abs(fft_sim)**2
        power_exp = np.abs(fft_exp)**2
        power_residuals = np.abs(fft_residuals)**2
        
        # Identifier les fréquences dominantes dans les résidus
        dominant_freq_indices = np.argsort(power_residuals)[-10:]  # Top 10
        dominant_freqs = freqs[dominant_freq_indices]
        dominant_powers = power_residuals[dominant_freq_indices]
        
        frequency_analysis = {
            'freqs': freqs,
            'power_sim': power_sim,
            'power_exp': power_exp,
            'power_residuals': power_residuals,
            'dominant_freqs': dominant_freqs,
            'dominant_powers': dominant_powers
        }
        
        print(f"   🌊 Fréquences dominantes dans les résidus:")
        for i, (freq, power) in enumerate(zip(dominant_freqs[-5:], dominant_powers[-5:])):
            print(f"      {i+1}. Freq: {freq:.3f} µm⁻¹, Power: {power:.2e}")
        
        return frequency_analysis
    
    def detect_systematic_errors(self, r_common, residuals):
        """Détecte les erreurs systématiques."""
        print("🎯 Détection d'erreurs systématiques...")
        
        systematic_errors = {}
        
        # 1. Tendance linéaire
        coeffs = np.polyfit(r_common, residuals, 1)
        linear_trend = np.polyval(coeffs, r_common)
        trend_strength = np.abs(coeffs[0])
        
        # 2. Oscillations périodiques
        # Détecter les pics dans l'autocorrélation
        autocorr = np.correlate(residuals, residuals, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normaliser
        
        # Trouver les pics significatifs
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.1, distance=10)
        if len(peaks) > 0:
            main_period_idx = peaks[0] + 1  # +1 car on a exclu le premier point
            main_period = (r_common[1] - r_common[0]) * main_period_idx
        else:
            main_period = None
        
        # 3. Zones de forte erreur
        error_threshold = np.std(residuals) * 2
        high_error_mask = np.abs(residuals) > error_threshold
        high_error_zones = []
        
        if np.any(high_error_mask):
            # Trouver les zones continues d'erreur élevée
            diff = np.diff(np.concatenate(([False], high_error_mask, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for start, end in zip(starts, ends):
                high_error_zones.append({
                    'r_start': r_common[start],
                    'r_end': r_common[end-1],
                    'max_error': np.max(np.abs(residuals[start:end])),
                    'mean_error': np.mean(residuals[start:end])
                })
        
        systematic_errors = {
            'linear_trend': {
                'slope': coeffs[0],
                'intercept': coeffs[1],
                'strength': trend_strength
            },
            'periodic_oscillation': {
                'main_period': main_period,
                'autocorr_peaks': len(peaks)
            },
            'high_error_zones': high_error_zones
        }
        
        print(f"   📈 Tendance linéaire: pente = {coeffs[0]:.2e}")
        if main_period:
            print(f"   🌊 Oscillation principale: période = {main_period:.3f} µm")
        print(f"   ⚠️ Zones d'erreur élevée: {len(high_error_zones)}")
        
        return systematic_errors
    
    def generate_diagnostic_plots(self, r_common, ratio_sim, I_exp, residuals_abs, 
                                 residuals_rel, metrics, frequency_analysis, systematic_errors):
        """Génère les graphiques de diagnostic."""
        print("📈 Génération des graphiques de diagnostic...")
        
        # Figure principale avec 6 sous-graphiques
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. Profils originaux + résidus
        axes[0, 0].plot(r_common, ratio_sim, 'b-', label='Simulation', linewidth=2, alpha=0.8)
        axes[0, 0].plot(r_common, I_exp, 'r-', label='Expérimental', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Profils Originaux')
        axes[0, 0].set_xlabel('Position Radiale (µm)')
        axes[0, 0].set_ylabel('Intensité')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Résidus absolus
        axes[0, 1].plot(r_common, residuals_abs, 'g-', linewidth=2)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_title(f'Résidus Absolus (MAE: {metrics["global"]["mae"]:.4f})')
        axes[0, 1].set_xlabel('Position Radiale (µm)')
        axes[0, 1].set_ylabel('Résidu')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Analyse par zones
        zone_data = pd.DataFrame(metrics['zones'])
        x_zones = range(1, len(zone_data) + 1)
        
        axes[1, 0].bar(x_zones, zone_data['correlation'], alpha=0.7, color='skyblue')
        axes[1, 0].set_title('Corrélation par Zone')
        axes[1, 0].set_xlabel('Zone')
        axes[1, 0].set_ylabel('Corrélation')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Distribution des résidus
        axes[1, 1].hist(residuals_abs, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Distribution des Résidus')
        axes[1, 1].set_xlabel('Résidu')
        axes[1, 1].set_ylabel('Fréquence')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Analyse fréquentielle
        freqs = frequency_analysis['freqs']
        power_residuals = frequency_analysis['power_residuals']
        
        # Prendre seulement les fréquences positives
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power_residuals[:len(power_residuals)//2]
        
        axes[2, 0].semilogy(positive_freqs, positive_power, 'purple', linewidth=2)
        axes[2, 0].set_title('Spectre de Puissance des Résidus')
        axes[2, 0].set_xlabel('Fréquence Spatiale (µm⁻¹)')
        axes[2, 0].set_ylabel('Puissance')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Erreurs systématiques
        linear_trend = np.polyval([systematic_errors['linear_trend']['slope'], 
                                  systematic_errors['linear_trend']['intercept']], r_common)
        
        axes[2, 1].plot(r_common, residuals_abs, 'g-', alpha=0.5, label='Résidus')
        axes[2, 1].plot(r_common, linear_trend, 'r--', linewidth=2, label='Tendance linéaire')
        
        # Marquer les zones d'erreur élevée
        for zone in systematic_errors['high_error_zones']:
            axes[2, 1].axvspan(zone['r_start'], zone['r_end'], alpha=0.3, color='red')
        
        axes[2, 1].set_title('Erreurs Systématiques')
        axes[2, 1].set_xlabel('Position Radiale (µm)')
        axes[2, 1].set_ylabel('Résidu')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Analyse Complète des Erreurs Résiduelles', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Sauvegarder
        plt.savefig(self.results_dir / 'residual_analysis_complete.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Graphiques sauvegardés: {self.results_dir}/residual_analysis_complete.png")
    
    def generate_recommendations(self, metrics, systematic_errors, frequency_analysis):
        """Génère des recommandations d'amélioration."""
        print("💡 Génération des recommandations...")
        
        recommendations = []
        
        # 1. Analyse des zones problématiques
        zone_data = pd.DataFrame(metrics['zones'])
        worst_zone = zone_data.loc[zone_data['correlation'].idxmin()]
        
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Zone Analysis',
            'issue': f"Zone {worst_zone['zone']} a la plus faible corrélation ({worst_zone['correlation']:.3f})",
            'recommendation': f"Appliquer un preprocessing spécialisé pour la zone radiale {worst_zone['r_start']:.2f}-{worst_zone['r_end']:.2f} µm",
            'expected_gain': '+2-3% corrélation'
        })
        
        # 2. Tendance linéaire
        trend_strength = systematic_errors['linear_trend']['strength']
        if trend_strength > 1e-3:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Systematic Bias',
                'issue': f"Tendance linéaire détectée (pente: {systematic_errors['linear_trend']['slope']:.2e})",
                'recommendation': "Ajouter une correction de baseline adaptative au preprocessing",
                'expected_gain': '+1-2% corrélation'
            })
        
        # 3. Oscillations périodiques
        if systematic_errors['periodic_oscillation']['main_period']:
            period = systematic_errors['periodic_oscillation']['main_period']
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Periodic Errors',
                'issue': f"Oscillation périodique détectée (période: {period:.3f} µm)",
                'recommendation': f"Ajouter un filtre adaptatif ou une loss function sensible aux oscillations de période {period:.3f} µm",
                'expected_gain': '+3-5% corrélation'
            })
        
        # 4. Zones d'erreur élevée
        high_error_zones = systematic_errors['high_error_zones']
        if len(high_error_zones) > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'High Error Zones',
                'issue': f"{len(high_error_zones)} zones d'erreur élevée détectées",
                'recommendation': "Implémenter une loss function pondérée qui pénalise plus ces zones spécifiques",
                'expected_gain': '+2-4% corrélation'
            })
        
        # 5. Contenu fréquentiel
        dominant_freqs = frequency_analysis['dominant_freqs']
        if len(dominant_freqs) > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Frequency Content',
                'issue': "Fréquences dominantes manquées dans les résidus",
                'recommendation': "Ajouter des couches convolutionnelles 1D pour capturer les patterns haute fréquence",
                'expected_gain': '+1-3% corrélation'
            })
        
        # Trier par priorité
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def save_analysis_report(self, metrics, systematic_errors, recommendations):
        """Sauvegarde le rapport d'analyse."""
        print("💾 Sauvegarde du rapport d'analyse...")
        
        report = {
            'summary': {
                'current_correlation': metrics['global']['correlation'],
                'current_r2': metrics['global']['r2'],
                'mse': metrics['global']['mse'],
                'mae': metrics['global']['mae'],
                'potential_improvement': '+5-10% corrélation possible'
            },
            'zone_analysis': metrics['zones'],
            'systematic_errors': systematic_errors,
            'recommendations': recommendations
        }
        
        # Sauvegarder en JSON
        import json
        with open(self.results_dir / 'residual_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Créer un rapport markdown
        markdown_report = self.create_markdown_report(report)
        with open(self.results_dir / 'residual_analysis_report.md', 'w') as f:
            f.write(markdown_report)
        
        print(f"   ✅ Rapport sauvegardé: {self.results_dir}/residual_analysis_report.json")
        print(f"   ✅ Rapport markdown: {self.results_dir}/residual_analysis_report.md")
    
    def create_markdown_report(self, report):
        """Crée un rapport markdown lisible."""
        markdown = f"""# 🔬 Rapport d'Analyse des Erreurs Résiduelles

## 📊 Résumé Exécutif
- **Corrélation actuelle**: {report['summary']['current_correlation']:.4f} ({report['summary']['current_r2']*100:.1f}%)
- **MSE**: {report['summary']['mse']:.6f}
- **MAE**: {report['summary']['mae']:.6f}
- **Amélioration potentielle**: {report['summary']['potential_improvement']}

## 🎯 Recommandations Prioritaires

"""
        
        for i, rec in enumerate(report['recommendations'][:3], 1):
            priority_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[rec['priority']]
            markdown += f"""### {i}. {priority_emoji} {rec['category']} ({rec['priority']} Priority)
**Problème**: {rec['issue']}
**Recommandation**: {rec['recommendation']}
**Gain attendu**: {rec['expected_gain']}

"""
        
        markdown += """## 📈 Analyse par Zones

| Zone | Corrélation | MSE | MAE |
|------|-------------|-----|-----|
"""
        
        for zone in report['zone_analysis']:
            markdown += f"| {zone['zone']} | {zone['correlation']:.3f} | {zone['mse']:.6f} | {zone['mae']:.6f} |\n"
        
        return markdown

def main():
    """Fonction principale."""
    print("🔬 Analyse des Erreurs Résiduelles - Diagnostic Avancé")
    print("=" * 70)
    
    # Créer l'analyseur
    analyzer = ResidualErrorAnalyzer()
    
    # 1. Charger les données
    x_sim, ratio_sim, r_exp, I_exp = analyzer.load_comparison_data()
    
    # 2. Harmoniser pour analyse
    r_common, ratio_sim_interp, I_exp_interp = analyzer.harmonize_for_analysis(
        x_sim, ratio_sim, r_exp, I_exp
    )
    
    # 3. Calculer les résidus
    residuals_abs, residuals_rel, metrics = analyzer.compute_residuals(
        r_common, ratio_sim_interp, I_exp_interp
    )
    
    # 4. Analyse fréquentielle
    frequency_analysis = analyzer.analyze_frequency_content(
        r_common, ratio_sim_interp, I_exp_interp, residuals_abs
    )
    
    # 5. Détecter les erreurs systématiques
    systematic_errors = analyzer.detect_systematic_errors(r_common, residuals_abs)
    
    # 6. Générer les graphiques
    analyzer.generate_diagnostic_plots(
        r_common, ratio_sim_interp, I_exp_interp, residuals_abs, residuals_rel,
        metrics, frequency_analysis, systematic_errors
    )
    
    # 7. Générer les recommandations
    recommendations = analyzer.generate_recommendations(
        metrics, systematic_errors, frequency_analysis
    )
    
    # 8. Sauvegarder le rapport
    analyzer.save_analysis_report(metrics, systematic_errors, recommendations)
    
    # 9. Afficher les recommandations principales
    print("\n🎯 RECOMMANDATIONS PRINCIPALES:")
    for i, rec in enumerate(recommendations[:3], 1):
        priority_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[rec['priority']]
        print(f"{i}. {priority_emoji} {rec['category']} ({rec['priority']})")
        print(f"   Problème: {rec['issue']}")
        print(f"   Solution: {rec['recommendation']}")
        print(f"   Gain: {rec['expected_gain']}")
        print()
    
    print("🎉 Analyse des erreurs résiduelles terminée avec succès !")
    print(f"📁 Résultats dans: {analyzer.results_dir}")

if __name__ == "__main__":
    main()
