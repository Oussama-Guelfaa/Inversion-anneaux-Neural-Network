#!/usr/bin/env python3
"""
Test du modèle ultra-précis sur le profil expérimental 49
Auteur: Oussama GUELFAA
Date: 18/07/2025

Test final du modèle ultra-précis pour vérifier si on atteint
gap=0.115 µm ±0.01 sur les données expérimentales.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from scipy.interpolate import interp1d
import sys
sys.path.append('../training')
from domain_adaptation_training import DomainAdaptiveNetwork

class UltraPreciseTester:
    """Testeur pour le modèle ultra-précis."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paramètres de preprocessing
        self.r_min = 1.3845845845845846
        self.r_max = 5.538338338338338
        self.final_points = 601
        self.r_network = np.linspace(self.r_min, self.r_max, self.final_points)
        
        print("🎯 TEST MODÈLE ULTRA-PRÉCIS")
        print("=" * 50)
        print(f"   🖥️ Device: {self.device}")
        print(f"   🎯 Objectif: Gap = 0.115 µm ±0.01")
        print(f"   🔬 Modèle: Ultra-Précision")
    
    def load_ultra_precise_model(self):
        """Charge le modèle ultra-précis."""
        
        print("📂 Chargement du modèle ultra-précis...")
        
        model_path = "../training/results/ultra_precise_domain_model.pt"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        # Charger le checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Créer et charger le modèle
        self.model = DomainAdaptiveNetwork().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"   ✅ Modèle chargé (époque {checkpoint['epoch']})")
        print(f"   📊 Meilleure erreur Gap: {checkpoint['best_gap_error']:.6f} µm")
        
        return checkpoint
    
    def load_experimental_profile(self, profile_idx=49):
        """Charge et prétraite le profil expérimental."""
        
        print(f"📊 Chargement du profil expérimental {profile_idx}...")
        
        exp_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"
        
        if not Path(exp_file).exists():
            raise FileNotFoundError(f"Fichier expérimental non trouvé: {exp_file}")
        
        data = sio.loadmat(exp_file)
        I_profiles = data['I_profiles']
        r_exp = data['r_exp'].flatten() * 1e6
        
        if profile_idx >= I_profiles.shape[0]:
            raise ValueError(f"Profil {profile_idx} non disponible (max: {I_profiles.shape[0]-1})")
        
        I_profile = I_profiles[profile_idx, :]
        
        print(f"   ✅ Profil {profile_idx} chargé")
        
        return I_profile, r_exp
    
    def preprocess_experimental_profile(self, I_profile, r_exp):
        """Prétraite le profil expérimental."""
        
        print("🔄 Prétraitement du profil expérimental...")
        
        # Coupure
        mask = (r_exp >= self.r_min) & (r_exp <= self.r_max)
        indices_valid = np.where(mask)[0]
        
        if len(indices_valid) == 0:
            raise ValueError(f"Aucun point dans la plage [{self.r_min:.3f}, {self.r_max:.3f}] µm")
        
        r_cut = r_exp[indices_valid]
        I_cut = I_profile[indices_valid]
        
        # Interpolation
        f_interp = interp1d(r_cut, I_cut, kind='linear', bounds_error=False, fill_value='extrapolate')
        I_processed = f_interp(self.r_network)
        
        # Traitement des NaN
        if np.any(np.isnan(I_processed)):
            I_processed = np.nan_to_num(I_processed, nan=np.mean(I_processed[~np.isnan(I_processed)]))
        
        print(f"   ✅ Profil traité: {len(I_processed)} points")
        
        return I_processed
    
    def predict_ultra_precise(self, I_processed):
        """Fait la prédiction ultra-précise."""
        
        print("🔮 Prédiction ULTRA-PRÉCISE...")
        
        # Convertir en tensor
        X_tensor = torch.FloatTensor(I_processed).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            predictions, domain_pred, features = self.model(X_tensor, alpha=0.0)
            pred_np = predictions.cpu().numpy()[0]
            domain_prob = domain_pred.cpu().numpy()[0, 0]
        
        gap_pred = pred_np[0]
        L_ecran_pred = pred_np[1]
        
        print(f"   ✅ Prédiction effectuée")
        print(f"   🎯 Gap prédit: {gap_pred:.6f} µm")
        print(f"   🎯 L'écran prédit: {L_ecran_pred:.3f} µm")
        print(f"   🎭 Probabilité domaine expérimental: {domain_prob:.3f}")
        
        return gap_pred, L_ecran_pred, domain_prob
    
    def evaluate_ultra_precision(self, gap_pred, L_ecran_pred, gap_target=0.115, L_ecran_target=10.30):
        """Évalue la précision ultra-fine."""
        
        print(f"\n📊 ÉVALUATION ULTRA-PRÉCISE:")
        
        gap_error = abs(gap_pred - gap_target)
        L_ecran_error = abs(L_ecran_pred - L_ecran_target)
        
        print(f"   🎯 GAP:")
        print(f"      Prédit: {gap_pred:.6f} µm")
        print(f"      Cible:  {gap_target:.6f} µm")
        print(f"      Erreur: {gap_error:.6f} µm")
        
        print(f"   🎯 L'ÉCRAN:")
        print(f"      Prédit: {L_ecran_pred:.3f} µm")
        print(f"      Cible:  {L_ecran_target:.2f} µm")
        print(f"      Erreur: {L_ecran_error:.3f} µm")
        
        # Évaluation de précision ultra-fine
        gap_precision_001 = gap_error <= 0.001
        gap_precision_005 = gap_error <= 0.005
        gap_precision_01 = gap_error <= 0.01
        gap_precision_02 = gap_error <= 0.02
        
        L_ecran_precision_01 = L_ecran_error <= 0.01
        L_ecran_precision_1 = L_ecran_error <= 0.1
        L_ecran_precision_5 = L_ecran_error <= 0.5
        
        print(f"\n📊 PRÉCISION ULTRA-FINE:")
        print(f"   🎯 Gap:")
        print(f"      ±0.001 µm: {'✅' if gap_precision_001 else '❌'} (Ultra-précision)")
        print(f"      ±0.005 µm: {'✅' if gap_precision_005 else '❌'} (Très haute précision)")
        print(f"      ±0.01 µm:  {'✅' if gap_precision_01 else '❌'} (OBJECTIF)")
        print(f"      ±0.02 µm:  {'✅' if gap_precision_02 else '❌'} (Acceptable)")
        
        print(f"   🎯 L'écran:")
        print(f"      ±0.01 µm: {'✅' if L_ecran_precision_01 else '❌'} (Ultra-précision)")
        print(f"      ±0.1 µm:  {'✅' if L_ecran_precision_1 else '❌'} (Très bonne)")
        print(f"      ±0.5 µm:  {'✅' if L_ecran_precision_5 else '❌'} (Acceptable)")
        
        # Verdict final
        print(f"\n🏆 VERDICT FINAL:")
        if gap_precision_01 and L_ecran_precision_5:
            print(f"   ✅ OBJECTIF ATTEINT ! Gap ±0.01 µm ET L'écran ±0.5 µm")
        elif gap_precision_01:
            print(f"   🎯 Gap OBJECTIF ATTEINT ! (±0.01 µm)")
            print(f"   ⚠️  L'écran à améliorer")
        elif L_ecran_precision_5:
            print(f"   🎯 L'écran OBJECTIF ATTEINT ! (±0.5 µm)")
            print(f"   ⚠️  Gap à améliorer")
        else:
            print(f"   ❌ OBJECTIFS NON ATTEINTS")
        
        return {
            'gap_error': gap_error,
            'L_ecran_error': L_ecran_error,
            'gap_precision_01': gap_precision_01,
            'L_ecran_precision_5': L_ecran_precision_5,
            'objective_reached': gap_precision_01 and L_ecran_precision_5
        }
    
    def compare_all_models(self, I_processed, gap_target=0.115, L_ecran_target=10.30):
        """Compare tous les modèles développés."""
        
        print(f"\n🔍 COMPARAISON DE TOUS LES MODÈLES:")
        
        results = {}
        
        # 1. Modèle ultra-précis (actuel)
        gap_ultra, L_ecran_ultra, domain_prob = self.predict_ultra_precise(I_processed)
        results['Ultra-Précis'] = {
            'gap': gap_ultra,
            'L_ecran': L_ecran_ultra,
            'gap_error': abs(gap_ultra - gap_target),
            'L_ecran_error': abs(L_ecran_ultra - L_ecran_target)
        }
        
        # 2. Modèle domain adaptation original
        try:
            checkpoint_da = torch.load("../training/results/domain_adapted_model.pt", map_location=self.device, weights_only=False)
            model_da = DomainAdaptiveNetwork().to(self.device)
            model_da.load_state_dict(checkpoint_da['model_state_dict'])
            model_da.eval()
            
            X_tensor = torch.FloatTensor(I_processed).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_da, _, _ = model_da(X_tensor, alpha=0.0)
                pred_da_np = pred_da.cpu().numpy()[0]
            
            results['Domain Adaptation'] = {
                'gap': pred_da_np[0],
                'L_ecran': pred_da_np[1],
                'gap_error': abs(pred_da_np[0] - gap_target),
                'L_ecran_error': abs(pred_da_np[1] - L_ecran_target)
            }
        except Exception as e:
            print(f"   ⚠️  Modèle Domain Adaptation non disponible: {e}")
        
        # 3. Modèle spécialisé original
        try:
            from specialized_gap_training import DualSpecializedNetwork
            checkpoint_spec = torch.load("../training/results/specialized_gap_training/best_specialized_model.pt",
                                       map_location=self.device, weights_only=False)
            model_spec = DualSpecializedNetwork().to(self.device)
            model_spec.load_state_dict(checkpoint_spec['model_state_dict'])
            model_spec.eval()
            
            X_tensor = torch.FloatTensor(I_processed).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred_spec = model_spec(X_tensor).cpu().numpy()[0]
            
            results['Spécialisé'] = {
                'gap': pred_spec[0],
                'L_ecran': pred_spec[1],
                'gap_error': abs(pred_spec[0] - gap_target),
                'L_ecran_error': abs(pred_spec[1] - L_ecran_target)
            }
        except Exception as e:
            print(f"   ⚠️  Modèle Spécialisé non disponible: {e}")
        
        # Affichage comparatif
        print(f"\n   📊 COMPARAISON DES PERFORMANCES:")
        print(f"   {'Modèle':<20} {'Gap (µm)':<12} {'L_écran (µm)':<12} {'Err Gap':<10} {'Err L_écran':<12}")
        print(f"   {'-'*70}")
        
        for name, res in results.items():
            print(f"   {name:<20} {res['gap']:<12.6f} {res['L_ecran']:<12.3f} "
                  f"{res['gap_error']:<10.6f} {res['L_ecran_error']:<12.3f}")
        
        print(f"   {'Cible':<20} {gap_target:<12.6f} {L_ecran_target:<12.3f} {'0.000000':<10} {'0.000':<12}")
        
        # Meilleur modèle
        best_model = min(results.items(), key=lambda x: x[1]['gap_error'])
        print(f"\n   🏆 MEILLEUR MODÈLE: {best_model[0]} (Gap error: {best_model[1]['gap_error']:.6f} µm)")
        
        return results
    
    def create_final_visualization(self, I_profile_raw, r_exp, I_processed, results, evaluation):
        """Crée la visualisation finale."""
        
        print("📈 Création de la visualisation finale...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('RÉSULTATS FINAUX - MODÈLE ULTRA-PRÉCIS', fontsize=16, fontweight='bold')
        
        # 1. Profil expérimental
        ax1 = axes[0, 0]
        ax1.plot(r_exp, I_profile_raw, 'b-', linewidth=1, alpha=0.7, label='Profil brut')
        ax1.axvline(self.r_min, color='red', linestyle='--', alpha=0.7, label='Limites réseau')
        ax1.axvline(self.r_max, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Position radiale (µm)')
        ax1.set_ylabel('Intensité')
        ax1.set_title('Profil Expérimental 49')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Profil traité
        ax2 = axes[0, 1]
        ax2.plot(self.r_network, I_processed, 'g-', linewidth=2, label='Profil traité')
        ax2.set_xlabel('Position radiale (µm)')
        ax2.set_ylabel('Intensité')
        ax2.set_title('Profil Traité (Entrée Réseau)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Comparaison des modèles
        ax3 = axes[1, 0]
        
        models = list(results.keys())
        gap_values = [results[m]['gap'] for m in models]
        L_ecran_values = [results[m]['L_ecran'] for m in models]
        
        # Ajouter la cible
        models.append('Cible')
        gap_values.append(0.115)
        L_ecran_values.append(10.30)
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, gap_values, width, label='Gap (µm)', alpha=0.8)
        bars2 = ax3.bar(x + width/2, [l/10 for l in L_ecran_values], width, 
                       label="L'écran (µm/10)", alpha=0.8)
        
        ax3.set_xlabel('Modèles')
        ax3.set_ylabel('Valeur')
        ax3.set_title('Comparaison Finale des Modèles')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Résumé final
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        ultra_result = results['Ultra-Précis']
        
        summary_text = f"""RÉSULTATS FINAUX

🎯 MODÈLE ULTRA-PRÉCIS

📊 PRÉDICTIONS:
   Gap: {ultra_result['gap']:.6f} µm
   L'écran: {ultra_result['L_ecran']:.3f} µm

🎯 CIBLES:
   Gap: 0.115000 µm
   L'écran: 10.30 µm

📊 ERREURS:
   Gap: {ultra_result['gap_error']:.6f} µm
   L'écran: {ultra_result['L_ecran_error']:.3f} µm

📊 PRÉCISION:
   Gap ±0.01 µm: {'✅' if evaluation['gap_precision_01'] else '❌'}
   L'écran ±0.5 µm: {'✅' if evaluation['L_ecran_precision_5'] else '❌'}

🏆 OBJECTIF:
   {'✅ ATTEINT !' if evaluation['objective_reached'] else '❌ Non atteint'}

🧠 MODÈLE: Ultra-Précision
🎭 MÉTHODE: Domain Adaptation
📊 PONDÉRATION: Gap x1000
        """
        
        color = "lightgreen" if evaluation['objective_reached'] else "lightyellow"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarder
        output_file = "../../visualizations/plots/ultra_precise_final_results.png"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualisation finale sauvegardée: {output_file}")
        
        plt.show()
    
    def run_final_test(self, profile_idx=49, gap_target=0.115, L_ecran_target=10.30):
        """Lance le test final complet."""
        
        try:
            # 1. Charger le modèle ultra-précis
            checkpoint = self.load_ultra_precise_model()
            
            # 2. Charger le profil expérimental
            I_profile_raw, r_exp = self.load_experimental_profile(profile_idx)
            
            # 3. Prétraiter le profil
            I_processed = self.preprocess_experimental_profile(I_profile_raw, r_exp)
            
            # 4. Prédiction ultra-précise
            gap_pred, L_ecran_pred, domain_prob = self.predict_ultra_precise(I_processed)
            
            # 5. Évaluation ultra-précise
            evaluation = self.evaluate_ultra_precision(gap_pred, L_ecran_pred, gap_target, L_ecran_target)
            
            # 6. Comparaison de tous les modèles
            results = self.compare_all_models(I_processed, gap_target, L_ecran_target)
            
            # 7. Visualisation finale
            self.create_final_visualization(I_profile_raw, r_exp, I_processed, results, evaluation)
            
            print(f"\n✅ TEST FINAL TERMINÉ!")
            print(f"   🎯 Gap ultra-précis: {gap_pred:.6f} µm")
            print(f"   🎯 L'écran ultra-précis: {L_ecran_pred:.3f} µm")
            print(f"   🏆 Objectif: {'ATTEINT !' if evaluation['objective_reached'] else 'Non atteint'}")
            
            return results, evaluation
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            raise

def main():
    """Fonction principale."""
    
    tester = UltraPreciseTester()
    results, evaluation = tester.run_final_test(profile_idx=49, gap_target=0.115, L_ecran_target=10.30)
    
    print(f"\n🎉 TEST ULTRA-PRÉCIS TERMINÉ!")

if __name__ == "__main__":
    main()
