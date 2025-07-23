#!/usr/bin/env python3
"""
Comparaison de différents normaliseurs pour préserver la forme des anneaux
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script teste différentes méthodes de normalisation pour voir
laquelle préserve le mieux la cohérence entre données exp et sim.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import random

class NormalizerComparator:
    """Comparateur de méthodes de normalisation."""
    
    def __init__(self):
        # Paramètres de preprocessing
        self.r_min = 1.3845845845845846
        self.r_max = 5.538338338338338
        self.final_points = 601
        self.r_network = np.linspace(self.r_min, self.r_max, self.final_points)
        
        print("🔍 COMPARATEUR DE NORMALISEURS")
        print("=" * 50)
    
    def load_and_preprocess_experimental(self, profile_number=49):
        """Charge et prétraite les données expérimentales."""
        
        print(f"📊 Chargement données expérimentales (profil {profile_number})...")
        
        exp_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"
        data = sio.loadmat(exp_file)
        
        I_profiles = data['I_profiles']
        r_exp = data['r_exp'].flatten() * 1e6
        I_profile = I_profiles[profile_number, :]
        
        # COUPURE puis INTERPOLATION
        mask = (r_exp >= self.r_min) & (r_exp <= self.r_max)
        indices_valid = np.where(mask)[0]
        r_cut = r_exp[indices_valid]
        I_cut = I_profile[indices_valid]
        
        f_interp = interp1d(r_cut, I_cut, kind='linear', bounds_error=False, fill_value='extrapolate')
        I_processed = f_interp(self.r_network)
        
        print(f"   ✅ Profil exp traité: {len(I_processed)} points")
        
        return I_processed
    
    def load_and_preprocess_simulation(self, target_gap=0.115, target_L_ecran=10.25):
        """Charge et prétraite une simulation spécifique."""
        
        print(f"🎯 Chargement simulation gap={target_gap}, L_écran={target_L_ecran}...")
        
        train_dir = Path("../../data/raw/Train")
        mat_files = list(train_dir.glob("gap_*.mat"))
        
        # Chercher le fichier le plus proche
        best_file = None
        best_distance = float('inf')
        
        for file_path in mat_files:
            try:
                filename = file_path.name
                parts = filename.replace('.mat', '').split('_')
                gap = float(parts[1].replace('um', ''))
                L_ecran = float(parts[3].replace('um', ''))
                
                distance = np.sqrt((gap - target_gap)**2 + (L_ecran - target_L_ecran)**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_file = file_path
                    best_gap = gap
                    best_L_ecran = L_ecran
            except Exception:
                continue
        
        print(f"   📁 Fichier trouvé: gap={best_gap:.6f}, L_écran={best_L_ecran:.3f}")
        
        # Charger et traiter
        data = sio.loadmat(best_file)
        
        if 'ratio' in data:
            ratio = data['ratio'].flatten()
        else:
            raise ValueError("Variable 'ratio' non trouvée")
        
        # COUPURE puis INTERPOLATION (même preprocessing)
        if len(ratio) >= 801:
            ratio_truncated = ratio[200:801]
            r_sim_truncated = np.linspace(self.r_min, self.r_max, len(ratio_truncated))
        else:
            ratio_truncated = ratio
            r_sim_truncated = np.linspace(self.r_min, self.r_max, len(ratio))
        
        f_interp = interp1d(r_sim_truncated, ratio_truncated, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
        ratio_processed = f_interp(self.r_network)
        
        print(f"   ✅ Simulation traitée: {len(ratio_processed)} points")
        
        return ratio_processed
    
    def test_normalizers(self, exp_data, sim_data):
        """Teste différents normaliseurs."""
        
        print(f"\n🔧 Test de différents normaliseurs...")
        
        # Définir les normaliseurs à tester
        normalizers = {
            'Aucune': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer_L2': Normalizer(norm='l2'),
            'PowerTransformer': PowerTransformer(method='yeo-johnson'),
            'QuantileTransformer': QuantileTransformer(output_distribution='normal'),
            'MinMax_0_1': MinMaxScaler(feature_range=(0, 1)),
            'MinMax_-1_1': MinMaxScaler(feature_range=(-1, 1)),
            'StandardScaler_Global': StandardScaler(),  # Entraîné sur les deux datasets
        }
        
        results = {}
        
        for name, normalizer in normalizers.items():
            try:
                print(f"   🔧 Test: {name}")
                
                if name == 'Aucune':
                    # Pas de normalisation
                    exp_norm = exp_data.copy()
                    sim_norm = sim_data.copy()
                    
                elif name == 'StandardScaler_Global':
                    # Entraîner sur les deux datasets combinés
                    combined_data = np.vstack([exp_data.reshape(1, -1), sim_data.reshape(1, -1)])
                    normalizer.fit(combined_data)
                    exp_norm = normalizer.transform(exp_data.reshape(1, -1)).flatten()
                    sim_norm = normalizer.transform(sim_data.reshape(1, -1)).flatten()
                    
                else:
                    # Entraîner sur simulation, appliquer aux deux
                    normalizer.fit(sim_data.reshape(1, -1))
                    exp_norm = normalizer.transform(exp_data.reshape(1, -1)).flatten()
                    sim_norm = normalizer.transform(sim_data.reshape(1, -1)).flatten()
                
                # Calculer la corrélation
                correlation = np.corrcoef(exp_norm, sim_norm)[0, 1]
                
                # Calculer la différence RMS
                rms_diff = np.sqrt(np.mean((exp_norm - sim_norm)**2))
                
                # Calculer la préservation de la forme (variance relative)
                exp_var = np.var(exp_norm)
                sim_var = np.var(sim_norm)
                var_ratio = min(exp_var, sim_var) / max(exp_var, sim_var)
                
                results[name] = {
                    'exp_norm': exp_norm,
                    'sim_norm': sim_norm,
                    'correlation': correlation,
                    'rms_diff': rms_diff,
                    'var_ratio': var_ratio,
                    'score': correlation * var_ratio - rms_diff * 0.1  # Score composite
                }
                
                print(f"      Corrélation: {correlation:.6f}")
                print(f"      RMS diff: {rms_diff:.6f}")
                print(f"      Var ratio: {var_ratio:.6f}")
                
            except Exception as e:
                print(f"      ❌ Erreur: {e}")
                continue
        
        return results
    
    def create_comparison_plot(self, exp_raw, sim_raw, results):
        """Crée une visualisation comparative des normaliseurs."""
        
        print(f"\n📈 Création de la visualisation comparative...")
        
        # Sélectionner les meilleurs normaliseurs
        sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
        top_normalizers = sorted_results[:6]  # Top 6
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('COMPARAISON DES NORMALISEURS - PRÉSERVATION DE LA FORME DES ANNEAUX', 
                     fontsize=16, fontweight='bold')
        
        # 1. Données brutes (référence)
        ax1 = axes[0, 0]
        ax1.plot(self.r_network, exp_raw, 'r-', linewidth=2, label='Expérimental')
        ax1.plot(self.r_network, sim_raw, 'b-', linewidth=2, label='Simulation')
        correlation_raw = np.corrcoef(exp_raw, sim_raw)[0, 1]
        ax1.set_title(f'Données brutes\nR = {correlation_raw:.3f}')
        ax1.set_xlabel('Position radiale (µm)')
        ax1.set_ylabel('Intensité')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2-7. Top 6 normaliseurs
        for i, (name, result) in enumerate(top_normalizers):
            row = (i + 1) // 3
            col = (i + 1) % 3
            ax = axes[row, col]
            
            exp_norm = result['exp_norm']
            sim_norm = result['sim_norm']
            correlation = result['correlation']
            
            ax.plot(self.r_network, exp_norm, 'r-', linewidth=2, label='Exp. normalisé')
            ax.plot(self.r_network, sim_norm, 'b-', linewidth=2, label='Sim. normalisé')
            ax.set_title(f'{name}\nR = {correlation:.3f}')
            ax.set_xlabel('Position radiale (µm)')
            ax.set_ylabel('Intensité normalisée')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 8. Tableau de scores
        ax_table = axes[2, 2]
        ax_table.axis('off')
        
        table_text = "CLASSEMENT DES NORMALISEURS:\n\n"
        for i, (name, result) in enumerate(sorted_results[:8]):
            score = result['score']
            corr = result['correlation']
            table_text += f"{i+1:2d}. {name:<20} Score: {score:6.3f} (R={corr:.3f})\n"
        
        ax_table.text(0.05, 0.95, table_text, transform=ax_table.transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarder
        output_file = "../../visualizations/plots/normalizers_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualisation sauvegardée: {output_file}")
        
        plt.show()
        
        return top_normalizers
    
    def generate_recommendation(self, results):
        """Génère une recommandation de normaliseur."""
        
        print(f"\n💡 RECOMMANDATIONS:")
        
        # Classer par score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
        
        best_name, best_result = sorted_results[0]
        
        print(f"🏆 MEILLEUR NORMALISEUR: {best_name}")
        print(f"   📊 Corrélation: {best_result['correlation']:.6f}")
        print(f"   📊 RMS diff: {best_result['rms_diff']:.6f}")
        print(f"   📊 Var ratio: {best_result['var_ratio']:.6f}")
        print(f"   📊 Score: {best_result['score']:.6f}")
        
        # Recommandations spécifiques
        if best_name == 'Aucune':
            print(f"\n✅ RECOMMANDATION: Utiliser les données BRUTES sans normalisation!")
            print(f"   La normalisation détruit la cohérence des anneaux.")
        elif 'MinMax' in best_name:
            print(f"\n✅ RECOMMANDATION: Utiliser MinMaxScaler")
            print(f"   Préserve mieux les relations relatives entre les valeurs.")
        elif 'Robust' in best_name:
            print(f"\n✅ RECOMMANDATION: Utiliser RobustScaler")
            print(f"   Moins sensible aux outliers que StandardScaler.")
        else:
            print(f"\n✅ RECOMMANDATION: Utiliser {best_name}")
        
        print(f"\n⚠️  À ÉVITER:")
        worst_name, worst_result = sorted_results[-1]
        print(f"   {worst_name} (corrélation: {worst_result['correlation']:.6f})")
        
        return best_name, best_result
    
    def compare_normalizers(self):
        """Comparaison complète des normaliseurs."""
        
        try:
            # 1. Charger les données
            exp_data = self.load_and_preprocess_experimental(profile_number=49)
            sim_data = self.load_and_preprocess_simulation(target_gap=0.115, target_L_ecran=10.25)
            
            # 2. Tester les normaliseurs
            results = self.test_normalizers(exp_data, sim_data)
            
            # 3. Visualiser
            top_normalizers = self.create_comparison_plot(exp_data, sim_data, results)
            
            # 4. Recommandation
            best_normalizer, best_result = self.generate_recommendation(results)
            
            print(f"\n✅ COMPARAISON TERMINÉE!")
            
            return results, best_normalizer
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            raise

def main():
    """Fonction principale."""
    
    comparator = NormalizerComparator()
    results, best_normalizer = comparator.compare_normalizers()
    
    print(f"\n🎯 RÉSULTAT FINAL:")
    print(f"   Le meilleur normaliseur est: {best_normalizer}")

if __name__ == "__main__":
    main()
