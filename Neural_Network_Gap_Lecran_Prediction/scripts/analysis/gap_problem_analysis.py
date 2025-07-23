#!/usr/bin/env python3
"""
Analyse approfondie du problème de prédiction du gap
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script analyse pourquoi le gap est si difficile à prédire
et propose des solutions pour atteindre R² = 0.9 et précision ±0.01 µm.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
import seaborn as sns
from sklearn.metrics import r2_score
import pandas as pd

class GapProblemAnalyzer:
    """Analyseur du problème de prédiction du gap."""
    
    def __init__(self):
        print("🔍 ANALYSEUR DU PROBLÈME DU GAP")
        print("=" * 50)
        print("🎯 Objectif: R² = 0.9, précision ±0.01 µm")
    
    def load_training_data(self):
        """Charge les données d'entraînement pour analyse."""
        
        print("📊 Chargement des données d'entraînement...")
        
        data_file = "../../data/processed/extracted_data_full.npz"
        data = np.load(data_file)
        
        X_data = data['X_data']  # (22540, 601) - profils d'intensité
        y_data = data['y_data']  # (22540, 2) - [gap, L_ecran]
        
        gaps = y_data[:, 0]
        L_ecrans = y_data[:, 1]
        
        print(f"   ✅ {len(X_data)} profils chargés")
        print(f"   📊 Gap range: [{gaps.min():.6f}, {gaps.max():.6f}] µm")
        print(f"   📊 L_écran range: [{L_ecrans.min():.3f}, {L_ecrans.max():.3f}] µm")
        
        return X_data, gaps, L_ecrans
    
    def analyze_gap_distribution(self, gaps, L_ecrans):
        """Analyse la distribution des gaps."""
        
        print("\n🔍 ANALYSE DE LA DISTRIBUTION DES GAPS:")
        
        # Statistiques de base
        gap_stats = {
            'min': gaps.min(),
            'max': gaps.max(),
            'mean': gaps.mean(),
            'std': gaps.std(),
            'median': np.median(gaps),
            'unique_values': len(np.unique(gaps))
        }
        
        print(f"   📊 Min: {gap_stats['min']:.6f} µm")
        print(f"   📊 Max: {gap_stats['max']:.6f} µm")
        print(f"   📊 Moyenne: {gap_stats['mean']:.6f} µm")
        print(f"   📊 Écart-type: {gap_stats['std']:.6f} µm")
        print(f"   📊 Médiane: {gap_stats['median']:.6f} µm")
        print(f"   📊 Valeurs uniques: {gap_stats['unique_values']}")
        
        # Distribution par bins
        bins = np.linspace(gaps.min(), gaps.max(), 20)
        hist, _ = np.histogram(gaps, bins=bins)
        
        print(f"\n   📈 Distribution par bins:")
        for i in range(len(bins)-1):
            count = hist[i]
            percentage = count / len(gaps) * 100
            print(f"      [{bins[i]:.3f}-{bins[i+1]:.3f}]: {count:4d} ({percentage:5.1f}%)")
        
        # Problème potentiel: déséquilibre
        target_gap = 0.115  # Gap expérimental typique
        close_gaps = gaps[(gaps >= target_gap - 0.02) & (gaps <= target_gap + 0.02)]
        print(f"\n   🎯 Gaps proches de 0.115 µm (±0.02): {len(close_gaps)} ({len(close_gaps)/len(gaps)*100:.1f}%)")
        
        return gap_stats
    
    def analyze_gap_L_ecran_correlation(self, gaps, L_ecrans):
        """Analyse la corrélation entre gap et L_écran."""
        
        print("\n🔍 ANALYSE CORRÉLATION GAP-L_ÉCRAN:")
        
        correlation = np.corrcoef(gaps, L_ecrans)[0, 1]
        print(f"   📊 Corrélation gap-L_écran: {correlation:.6f}")
        
        if abs(correlation) > 0.7:
            print(f"   ⚠️  FORTE CORRÉLATION détectée!")
            print(f"      Le réseau peut confondre gap et L_écran")
        elif abs(correlation) > 0.3:
            print(f"   ⚠️  Corrélation modérée détectée")
        else:
            print(f"   ✅ Corrélation faible (bon pour l'indépendance)")
        
        return correlation
    
    def analyze_intensity_sensitivity(self, X_data, gaps, L_ecrans):
        """Analyse la sensibilité des profils d'intensité au gap."""
        
        print("\n🔍 ANALYSE SENSIBILITÉ DES PROFILS AU GAP:")
        
        # Sélectionner des profils avec L_écran similaire mais gaps différents
        L_ecran_target = 10.0
        tolerance = 0.5
        
        mask = (L_ecrans >= L_ecran_target - tolerance) & (L_ecrans <= L_ecran_target + tolerance)
        similar_L_ecran_indices = np.where(mask)[0]
        
        if len(similar_L_ecran_indices) < 10:
            print("   ⚠️  Pas assez de profils avec L_écran similaire")
            return None
        
        similar_gaps = gaps[similar_L_ecran_indices]
        similar_profiles = X_data[similar_L_ecran_indices]
        
        # Trier par gap
        sort_indices = np.argsort(similar_gaps)
        sorted_gaps = similar_gaps[sort_indices]
        sorted_profiles = similar_profiles[sort_indices]
        
        print(f"   📊 {len(similar_L_ecran_indices)} profils avec L_écran ≈ {L_ecran_target:.1f} µm")
        print(f"   📊 Gap range: [{sorted_gaps.min():.6f}, {sorted_gaps.max():.6f}] µm")
        
        # Calculer la variance entre profils
        profile_variance = np.var(sorted_profiles, axis=0)
        mean_variance = np.mean(profile_variance)
        
        print(f"   📊 Variance moyenne entre profils: {mean_variance:.6f}")
        
        # Sensibilité: différence entre profils extrêmes
        if len(sorted_profiles) >= 2:
            diff_profile = np.abs(sorted_profiles[-1] - sorted_profiles[0])
            max_diff = np.max(diff_profile)
            mean_diff = np.mean(diff_profile)
            
            print(f"   📊 Différence max entre profils extrêmes: {max_diff:.6f}")
            print(f"   📊 Différence moyenne: {mean_diff:.6f}")
            
            if mean_diff < 0.01:
                print("   ⚠️  FAIBLE SENSIBILITÉ au gap détectée!")
                print("      Les profils changent peu avec le gap")
            else:
                print("   ✅ Sensibilité suffisante au gap")
        
        return {
            'variance': mean_variance,
            'max_diff': max_diff if 'max_diff' in locals() else 0,
            'mean_diff': mean_diff if 'mean_diff' in locals() else 0
        }
    
    def create_diagnostic_plots(self, X_data, gaps, L_ecrans):
        """Crée des graphiques de diagnostic."""
        
        print("\n📈 Création des graphiques de diagnostic...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DIAGNOSTIC DU PROBLÈME DE PRÉDICTION DU GAP', fontsize=16, fontweight='bold')
        
        # 1. Distribution des gaps
        ax1 = axes[0, 0]
        ax1.hist(gaps, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(0.115, color='red', linestyle='--', linewidth=2, label='Gap expérimental (0.115 µm)')
        ax1.set_xlabel('Gap (µm)')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution des Gaps')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution des L_écrans
        ax2 = axes[0, 1]
        ax2.hist(L_ecrans, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(10.3, color='red', linestyle='--', linewidth=2, label='L_écran expérimental (10.3 µm)')
        ax2.set_xlabel('L_écran (µm)')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution des L_écrans')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Corrélation gap-L_écran
        ax3 = axes[0, 2]
        ax3.scatter(gaps, L_ecrans, alpha=0.5, s=1)
        ax3.scatter(0.115, 10.3, color='red', s=100, marker='x', linewidth=3, label='Cible expérimentale')
        correlation = np.corrcoef(gaps, L_ecrans)[0, 1]
        ax3.set_xlabel('Gap (µm)')
        ax3.set_ylabel('L_écran (µm)')
        ax3.set_title(f'Corrélation Gap-L_écran (R={correlation:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Heatmap 2D
        ax4 = axes[1, 0]
        H, xedges, yedges = np.histogram2d(gaps, L_ecrans, bins=30)
        im = ax4.imshow(H.T, origin='lower', aspect='auto', cmap='viridis',
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        ax4.scatter(0.115, 10.3, color='red', s=100, marker='x', linewidth=3)
        ax4.set_xlabel('Gap (µm)')
        ax4.set_ylabel('L_écran (µm)')
        ax4.set_title('Densité des Données (Gap vs L_écran)')
        plt.colorbar(im, ax=ax4)
        
        # 5. Profils d'intensité pour différents gaps
        ax5 = axes[1, 1]
        
        # Sélectionner quelques profils avec gaps différents
        L_ecran_target = 10.0
        tolerance = 0.2
        mask = (L_ecrans >= L_ecran_target - tolerance) & (L_ecrans <= L_ecran_target + tolerance)
        indices = np.where(mask)[0]
        
        if len(indices) >= 5:
            selected_indices = indices[::len(indices)//5][:5]  # 5 profils équidistants
            r_network = np.linspace(1.385, 5.538, 601)
            
            for i, idx in enumerate(selected_indices):
                gap_val = gaps[idx]
                profile = X_data[idx]
                ax5.plot(r_network, profile, label=f'Gap={gap_val:.3f} µm', alpha=0.8)
            
            ax5.set_xlabel('Position radiale (µm)')
            ax5.set_ylabel('Intensité')
            ax5.set_title(f'Profils pour L_écran ≈ {L_ecran_target} µm')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Analyse de la précision requise
        ax6 = axes[1, 2]
        
        # Calculer la distribution des erreurs pour différentes précisions
        precisions = [0.001, 0.005, 0.01, 0.02, 0.05]
        gap_range = gaps.max() - gaps.min()
        
        percentages = []
        for prec in precisions:
            # Pourcentage de gaps dans une fenêtre de ±prec autour de 0.115
            target = 0.115
            in_range = np.sum((gaps >= target - prec) & (gaps <= target + prec))
            percentage = in_range / len(gaps) * 100
            percentages.append(percentage)
        
        ax6.bar(range(len(precisions)), percentages, alpha=0.7, color='orange')
        ax6.set_xticks(range(len(precisions)))
        ax6.set_xticklabels([f'±{p:.3f}' for p in precisions])
        ax6.set_xlabel('Précision (µm)')
        ax6.set_ylabel('% de données dans la fenêtre')
        ax6.set_title('Données disponibles pour gap=0.115 µm')
        ax6.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(percentages):
            ax6.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Sauvegarder
        output_file = "../../visualizations/plots/gap_problem_diagnosis.png"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Diagnostic sauvegardé: {output_file}")
        
        plt.show()
    
    def propose_solutions(self, gap_stats, correlation, sensitivity_analysis):
        """Propose des solutions pour améliorer la prédiction du gap."""
        
        print("\n💡 SOLUTIONS PROPOSÉES POUR ATTEINDRE R² = 0.9:")
        print("=" * 60)
        
        solutions = []
        
        # Solution 1: Architecture spécialisée
        print("🔧 SOLUTION 1: ARCHITECTURE SPÉCIALISÉE POUR LE GAP")
        print("   • Réseau dual avec branches séparées gap/L_écran")
        print("   • Couches dédiées au gap avec plus de neurones")
        print("   • Loss function pondérée (gap x10)")
        print("   • Régularisation spécifique au gap")
        solutions.append("architecture_specialisee")
        
        # Solution 2: Augmentation de données ciblée
        print("\n🔧 SOLUTION 2: AUGMENTATION DE DONNÉES CIBLÉE")
        print("   • Générer plus de données autour de gap=0.115 µm")
        print("   • Interpolation entre profils proches")
        print("   • Bruit gaussien calibré sur le gap")
        print("   • Équilibrage de la distribution")
        solutions.append("augmentation_ciblee")
        
        # Solution 3: Préprocessing avancé
        print("\n🔧 SOLUTION 3: PRÉPROCESSING AVANCÉ")
        print("   • Extraction de features spécifiques au gap")
        print("   • Analyse fréquentielle (FFT)")
        print("   • Dérivées et gradients des profils")
        print("   • Normalisation adaptative par région")
        solutions.append("preprocessing_avance")
        
        # Solution 4: Ensemble de modèles
        print("\n🔧 SOLUTION 4: ENSEMBLE DE MODÈLES")
        print("   • Modèle spécialisé uniquement pour le gap")
        print("   • Combinaison de plusieurs architectures")
        print("   • Voting ou stacking")
        print("   • Cross-validation sophistiquée")
        solutions.append("ensemble_modeles")
        
        # Solution 5: Transfer learning
        print("\n🔧 SOLUTION 5: TRANSFER LEARNING PROGRESSIF")
        print("   • Pré-entraînement sur L_écran (facile)")
        print("   • Fine-tuning progressif vers le gap")
        print("   • Curriculum learning (facile → difficile)")
        print("   • Contraintes physiques adaptatives")
        solutions.append("transfer_learning")
        
        # Recommandation basée sur l'analyse
        print("\n🎯 RECOMMANDATION PRIORITAIRE:")
        
        if sensitivity_analysis and sensitivity_analysis['mean_diff'] < 0.01:
            print("   ⚠️  FAIBLE SENSIBILITÉ détectée")
            print("   🥇 PRIORITÉ 1: Augmentation de données ciblée")
            print("   🥈 PRIORITÉ 2: Préprocessing avancé (features)")
            print("   🥉 PRIORITÉ 3: Architecture spécialisée")
        elif abs(correlation) > 0.5:
            print("   ⚠️  FORTE CORRÉLATION gap-L_écran détectée")
            print("   🥇 PRIORITÉ 1: Architecture spécialisée (branches séparées)")
            print("   🥈 PRIORITÉ 2: Ensemble de modèles")
            print("   🥉 PRIORITÉ 3: Transfer learning")
        else:
            print("   ✅ Données équilibrées")
            print("   🥇 PRIORITÉ 1: Architecture spécialisée")
            print("   🥈 PRIORITÉ 2: Loss function pondérée")
            print("   🥉 PRIORITÉ 3: Augmentation de données")
        
        return solutions
    
    def run_complete_analysis(self):
        """Lance l'analyse complète du problème du gap."""
        
        try:
            # 1. Charger les données
            X_data, gaps, L_ecrans = self.load_training_data()
            
            # 2. Analyser la distribution
            gap_stats = self.analyze_gap_distribution(gaps, L_ecrans)
            
            # 3. Analyser la corrélation
            correlation = self.analyze_gap_L_ecran_correlation(gaps, L_ecrans)
            
            # 4. Analyser la sensibilité
            sensitivity_analysis = self.analyze_intensity_sensitivity(X_data, gaps, L_ecrans)
            
            # 5. Créer les graphiques
            self.create_diagnostic_plots(X_data, gaps, L_ecrans)
            
            # 6. Proposer des solutions
            solutions = self.propose_solutions(gap_stats, correlation, sensitivity_analysis)
            
            print(f"\n✅ ANALYSE TERMINÉE!")
            print(f"   📊 {len(solutions)} solutions identifiées")
            print(f"   📈 Graphiques sauvegardés")
            
            return {
                'gap_stats': gap_stats,
                'correlation': correlation,
                'sensitivity': sensitivity_analysis,
                'solutions': solutions
            }
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            raise

def main():
    """Fonction principale."""
    
    analyzer = GapProblemAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print(f"\n🎯 PROCHAINE ÉTAPE:")
    print(f"   Implémenter la solution prioritaire identifiée")

if __name__ == "__main__":
    main()
