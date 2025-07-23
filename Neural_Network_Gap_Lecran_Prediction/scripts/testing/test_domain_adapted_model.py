#!/usr/bin/env python3
"""
Test du modèle adapté au domaine sur le profil expérimental 49
Auteur: Oussama GUELFAA
Date: 18/07/2025

Test du modèle avec domain adaptation sur les données expérimentales.
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

class DomainAdaptedTester:
    """Testeur pour le modèle adapté au domaine."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paramètres de preprocessing (identiques à l'entraînement)
        self.r_min = 1.3845845845845846
        self.r_max = 5.538338338338338
        self.final_points = 601
        self.r_network = np.linspace(self.r_min, self.r_max, self.final_points)
        
        print("🧪 TEST MODÈLE ADAPTÉ AU DOMAINE")
        print("=" * 50)
        print(f"   🖥️ Device: {self.device}")
        print(f"   🎯 Profil cible: 49")
        print(f"   🧠 Modèle: Domain Adaptation")
    
    def load_domain_adapted_model(self):
        """Charge le modèle adapté au domaine."""
        
        print("📂 Chargement du modèle adapté au domaine...")
        
        model_path = "../training/results/domain_adapted_model.pt"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        # Charger le checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Créer et charger le modèle
        self.model = DomainAdaptiveNetwork().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"   ✅ Modèle chargé (époque {checkpoint['epoch']})")
        print(f"   📊 Loss principale: {checkpoint['val_main_loss']:.6f}")
        print(f"   📊 Accuracy domaine: {checkpoint['domain_accuracy']:.1f}%")
        
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
        print(f"   📊 Points originaux: {len(I_profile)}")
        print(f"   📊 r_exp range: [{r_exp.min():.3f}, {r_exp.max():.3f}] µm")
        
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
        print(f"   📊 Intensité range: [{I_processed.min():.6f}, {I_processed.max():.6f}]")
        
        return I_processed
    
    def predict_with_domain_adaptation(self, I_processed):
        """Fait la prédiction avec le modèle adapté au domaine."""
        
        print("🔮 Prédiction avec domain adaptation...")
        
        # Convertir en tensor
        X_tensor = torch.FloatTensor(I_processed).unsqueeze(0).to(self.device)
        
        # Prédiction (alpha=0 pour désactiver le gradient reversal en test)
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
    
    def compare_models(self, I_processed, gap_target=0.115, L_ecran_target=10.30):
        """Compare le modèle adapté avec le modèle spécialisé original."""
        
        print("\n🔍 COMPARAISON DES MODÈLES:")
        
        # 1. Modèle adapté au domaine
        gap_adapted, L_ecran_adapted, domain_prob = self.predict_with_domain_adaptation(I_processed)
        
        # 2. Modèle spécialisé original (si disponible)
        try:
            from specialized_gap_training import DualSpecializedNetwork
            
            specialized_model_path = "../training/results/specialized_gap_training/best_specialized_model.pt"
            if Path(specialized_model_path).exists():
                checkpoint_spec = torch.load(specialized_model_path, map_location=self.device, weights_only=False)
                model_spec = DualSpecializedNetwork().to(self.device)
                model_spec.load_state_dict(checkpoint_spec['model_state_dict'])
                model_spec.eval()
                
                X_tensor = torch.FloatTensor(I_processed).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred_spec = model_spec(X_tensor).cpu().numpy()[0]
                
                gap_spec = pred_spec[0]
                L_ecran_spec = pred_spec[1]
                
                print(f"   📊 MODÈLE SPÉCIALISÉ ORIGINAL:")
                print(f"      Gap: {gap_spec:.6f} µm")
                print(f"      L'écran: {L_ecran_spec:.3f} µm")
                print(f"      Erreur Gap: {abs(gap_spec - gap_target):.6f} µm")
                print(f"      Erreur L'écran: {abs(L_ecran_spec - L_ecran_target):.3f} µm")
            else:
                gap_spec, L_ecran_spec = None, None
                print(f"   ⚠️  Modèle spécialisé non trouvé")
        except Exception as e:
            gap_spec, L_ecran_spec = None, None
            print(f"   ⚠️  Erreur chargement modèle spécialisé: {e}")
        
        print(f"\n   📊 MODÈLE ADAPTÉ AU DOMAINE:")
        print(f"      Gap: {gap_adapted:.6f} µm")
        print(f"      L'écran: {L_ecran_adapted:.3f} µm")
        print(f"      Erreur Gap: {abs(gap_adapted - gap_target):.6f} µm")
        print(f"      Erreur L'écran: {abs(L_ecran_adapted - L_ecran_target):.3f} µm")
        print(f"      Probabilité expérimental: {domain_prob:.3f}")
        
        # Évaluation
        gap_error_adapted = abs(gap_adapted - gap_target)
        L_ecran_error_adapted = abs(L_ecran_adapted - L_ecran_target)
        
        print(f"\n   📊 ÉVALUATION DOMAIN ADAPTATION:")
        print(f"      Gap ±0.001 µm: {'✅' if gap_error_adapted <= 0.001 else '❌'}")
        print(f"      Gap ±0.01 µm:  {'✅' if gap_error_adapted <= 0.01 else '❌'}")
        print(f"      Gap ±0.05 µm:  {'✅' if gap_error_adapted <= 0.05 else '❌'}")
        print(f"      L'écran ±0.5 µm: {'✅' if L_ecran_error_adapted <= 0.5 else '❌'}")
        
        # Amélioration par rapport au modèle spécialisé
        if gap_spec is not None:
            gap_error_spec = abs(gap_spec - gap_target)
            L_ecran_error_spec = abs(L_ecran_spec - L_ecran_target)
            
            gap_improvement = gap_error_spec - gap_error_adapted
            L_ecran_improvement = L_ecran_error_spec - L_ecran_error_adapted
            
            print(f"\n   📈 AMÉLIORATION vs MODÈLE SPÉCIALISÉ:")
            print(f"      Gap: {gap_improvement:+.6f} µm ({'✅' if gap_improvement > 0 else '❌'})")
            print(f"      L'écran: {L_ecran_improvement:+.3f} µm ({'✅' if L_ecran_improvement > 0 else '❌'})")
        
        return {
            'gap_adapted': gap_adapted,
            'L_ecran_adapted': L_ecran_adapted,
            'domain_prob': domain_prob,
            'gap_spec': gap_spec,
            'L_ecran_spec': L_ecran_spec,
            'gap_error_adapted': gap_error_adapted,
            'L_ecran_error_adapted': L_ecran_error_adapted
        }
    
    def create_comparison_visualization(self, I_profile_raw, r_exp, I_processed, results, 
                                      gap_target=0.115, L_ecran_target=10.30):
        """Crée une visualisation comparative."""
        
        print("📈 Création de la visualisation comparative...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('COMPARAISON MODÈLES - DOMAIN ADAPTATION vs SPÉCIALISÉ', 
                     fontsize=16, fontweight='bold')
        
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
        
        # 3. Comparaison des prédictions
        ax3 = axes[1, 0]
        
        models = []
        gap_preds = []
        L_ecran_preds = []
        
        # Modèle adapté
        models.append('Domain\nAdapted')
        gap_preds.append(results['gap_adapted'])
        L_ecran_preds.append(results['L_ecran_adapted'])
        
        # Modèle spécialisé (si disponible)
        if results['gap_spec'] is not None:
            models.append('Specialized\nOriginal')
            gap_preds.append(results['gap_spec'])
            L_ecran_preds.append(results['L_ecran_spec'])
        
        # Cibles
        models.append('Target')
        gap_preds.append(gap_target)
        L_ecran_preds.append(L_ecran_target)
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, gap_preds, width, label='Gap (µm)', alpha=0.8)
        bars2 = ax3.bar(x + width/2, [l/10 for l in L_ecran_preds], width, 
                       label="L'écran (µm/10)", alpha=0.8)
        
        ax3.set_xlabel('Modèles')
        ax3.set_ylabel('Valeur')
        ax3.set_title('Comparaison des Prédictions')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, (bar, val) in enumerate(zip(bars1, gap_preds)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Résumé des résultats
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""RÉSULTATS DOMAIN ADAPTATION

🎯 PROFIL EXPÉRIMENTAL 49

📊 MODÈLE ADAPTÉ AU DOMAINE:
   Gap: {results['gap_adapted']:.6f} µm
   L'écran: {results['L_ecran_adapted']:.3f} µm
   Prob. expérimental: {results['domain_prob']:.3f}

🎯 CIBLES:
   Gap: {gap_target:.6f} µm
   L'écran: {L_ecran_target:.2f} µm

📊 ERREURS:
   Gap: {results['gap_error_adapted']:.6f} µm
   L'écran: {results['L_ecran_error_adapted']:.3f} µm

📊 PRÉCISION:
   Gap ±0.01 µm: {'✅' if results['gap_error_adapted'] <= 0.01 else '❌'}
   Gap ±0.05 µm: {'✅' if results['gap_error_adapted'] <= 0.05 else '❌'}
   L'écran ±0.5 µm: {'✅' if results['L_ecran_error_adapted'] <= 0.5 else '❌'}

🧠 MODÈLE: Domain Adaptation
🎭 ADVERSARIAL: Gradient Reversal
📊 DONNÉES: Mix Sim + Exp
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarder
        output_file = "../../visualizations/plots/domain_adaptation_test_results.png"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualisation sauvegardée: {output_file}")
        
        plt.show()
    
    def run_complete_test(self, profile_idx=49, gap_target=0.115, L_ecran_target=10.30):
        """Lance le test complet du modèle adapté au domaine."""
        
        try:
            # 1. Charger le modèle adapté
            checkpoint = self.load_domain_adapted_model()
            
            # 2. Charger le profil expérimental
            I_profile_raw, r_exp = self.load_experimental_profile(profile_idx)
            
            # 3. Prétraiter le profil
            I_processed = self.preprocess_experimental_profile(I_profile_raw, r_exp)
            
            # 4. Comparer les modèles
            results = self.compare_models(I_processed, gap_target, L_ecran_target)
            
            # 5. Créer la visualisation
            self.create_comparison_visualization(I_profile_raw, r_exp, I_processed, 
                                               results, gap_target, L_ecran_target)
            
            print(f"\n✅ TEST DOMAIN ADAPTATION TERMINÉ!")
            print(f"   🎯 Gap adapté: {results['gap_adapted']:.6f} µm")
            print(f"   🎯 L'écran adapté: {results['L_ecran_adapted']:.3f} µm")
            print(f"   🎭 Probabilité expérimental: {results['domain_prob']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            raise

def main():
    """Fonction principale."""
    
    tester = DomainAdaptedTester()
    results = tester.run_complete_test(profile_idx=49, gap_target=0.115, L_ecran_target=10.30)
    
    print(f"\n🎉 TEST DOMAIN ADAPTATION TERMINÉ!")

if __name__ == "__main__":
    main()
