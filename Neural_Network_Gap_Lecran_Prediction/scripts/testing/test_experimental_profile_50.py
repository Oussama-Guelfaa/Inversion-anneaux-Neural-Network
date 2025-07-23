#!/usr/bin/env python3
"""
Test du modèle spécialisé sur le profil expérimental 50
Auteur: Oussama GUELFAA
Date: 18/07/2025

Test du modèle spécialisé pour le gap sur le profil expérimental 50
pour évaluer ses performances sur des données réelles.
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
from specialized_gap_training import DualSpecializedNetwork

class ExperimentalProfileTester:
    """Testeur pour le profil expérimental 50."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paramètres de preprocessing (identiques à l'entraînement)
        self.r_min = 1.3845845845845846
        self.r_max = 5.538338338338338
        self.final_points = 601
        self.r_network = np.linspace(self.r_min, self.r_max, self.final_points)
        
        print("🧪 TEST SUR PROFIL EXPÉRIMENTAL 50")
        print("=" * 50)
        print(f"   🖥️ Device: {self.device}")
        print(f"   🎯 Profil cible: 50")
        print(f"   📏 Grille réseau: {self.final_points} points")
    
    def load_specialized_model(self):
        """Charge le modèle spécialisé (pas le fine-tuné qui a échoué)."""
        
        print("📂 Chargement du modèle spécialisé original...")
        
        model_path = "../training/results/specialized_gap_training/best_specialized_model.pt"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        # Charger le checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Créer et charger le modèle
        self.model = DualSpecializedNetwork().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"   ✅ Modèle chargé (époque {checkpoint['epoch']})")
        print(f"   📊 Gap R² d'entraînement: {checkpoint['gap_r2']:.4f}")
        print(f"   📊 L_écran R² d'entraînement: {checkpoint['L_ecran_r2']:.4f}")
        
        return checkpoint
    
    def load_experimental_profile(self, profile_idx=50):
        """Charge et prétraite le profil expérimental."""
        
        print(f"📊 Chargement du profil expérimental {profile_idx}...")
        
        # Charger le fichier expérimental
        exp_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"
        
        if not Path(exp_file).exists():
            raise FileNotFoundError(f"Fichier expérimental non trouvé: {exp_file}")
        
        data = sio.loadmat(exp_file)
        
        # Extraire les données
        I_profiles = data['I_profiles']
        r_exp = data['r_exp'].flatten() * 1e6  # Conversion en µm
        
        # Vérifier l'index
        if profile_idx >= I_profiles.shape[0]:
            raise ValueError(f"Profil {profile_idx} non disponible (max: {I_profiles.shape[0]-1})")
        
        I_profile = I_profiles[profile_idx, :]
        
        print(f"   ✅ Profil {profile_idx} chargé")
        print(f"   📊 Points originaux: {len(I_profile)}")
        print(f"   📊 r_exp range: [{r_exp.min():.3f}, {r_exp.max():.3f}] µm")
        print(f"   📊 Intensité range: [{I_profile.min():.6f}, {I_profile.max():.6f}]")
        
        return I_profile, r_exp
    
    def preprocess_experimental_profile(self, I_profile, r_exp):
        """Prétraite le profil expérimental (même preprocessing que l'entraînement)."""
        
        print("🔄 Prétraitement du profil expérimental...")
        
        # ÉTAPE 1: COUPURE dans la plage d'entraînement
        mask = (r_exp >= self.r_min) & (r_exp <= self.r_max)
        indices_valid = np.where(mask)[0]
        
        if len(indices_valid) == 0:
            raise ValueError(f"Aucun point dans la plage [{self.r_min:.3f}, {self.r_max:.3f}] µm")
        
        r_cut = r_exp[indices_valid]
        I_cut = I_profile[indices_valid]
        
        print(f"   📊 Après coupure: {len(I_cut)} points")
        print(f"   📊 r_cut range: [{r_cut.min():.3f}, {r_cut.max():.3f}] µm")
        
        # ÉTAPE 2: INTERPOLATION sur la grille réseau
        f_interp = interp1d(r_cut, I_cut, kind='linear', bounds_error=False, fill_value='extrapolate')
        I_processed = f_interp(self.r_network)
        
        print(f"   ✅ Après interpolation: {len(I_processed)} points")
        print(f"   📊 Intensité traitée range: [{I_processed.min():.6f}, {I_processed.max():.6f}]")
        
        # Vérifier les NaN
        if np.any(np.isnan(I_processed)):
            print(f"   ⚠️  {np.sum(np.isnan(I_processed))} valeurs NaN détectées")
            I_processed = np.nan_to_num(I_processed, nan=np.mean(I_processed[~np.isnan(I_processed)]))
            print(f"   ✅ NaN remplacés par la moyenne")
        
        return I_processed
    
    def predict_on_experimental_profile(self, I_processed):
        """Fait la prédiction sur le profil expérimental."""
        
        print("🔮 Prédiction sur le profil expérimental...")
        
        # Convertir en tensor (batch de 1)
        X_tensor = torch.FloatTensor(I_processed).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            prediction = self.model(X_tensor)
            pred_np = prediction.cpu().numpy()[0]  # Extraire le premier (et seul) échantillon
        
        gap_pred = pred_np[0]
        L_ecran_pred = pred_np[1]
        
        print(f"   ✅ Prédiction effectuée")
        print(f"   🎯 Gap prédit: {gap_pred:.6f} µm")
        print(f"   🎯 L'écran prédit: {L_ecran_pred:.3f} µm")
        
        return gap_pred, L_ecran_pred
    
    def compare_with_targets(self, gap_pred, L_ecran_pred, gap_target=0.115, L_ecran_target=10.30):
        """Compare avec les valeurs cibles expérimentales."""
        
        print(f"\n📊 COMPARAISON AVEC LES CIBLES EXPÉRIMENTALES:")
        
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
        
        # Évaluation de la précision
        gap_precision_001 = gap_error <= 0.001
        gap_precision_01 = gap_error <= 0.01
        gap_precision_05 = gap_error <= 0.05
        
        L_ecran_precision_01 = L_ecran_error <= 0.01
        L_ecran_precision_1 = L_ecran_error <= 0.1
        L_ecran_precision_5 = L_ecran_error <= 0.5
        
        print(f"\n📊 ÉVALUATION DE LA PRÉCISION:")
        print(f"   🎯 Gap:")
        print(f"      ±0.001 µm: {'✅' if gap_precision_001 else '❌'}")
        print(f"      ±0.01 µm:  {'✅' if gap_precision_01 else '❌'}")
        print(f"      ±0.05 µm:  {'✅' if gap_precision_05 else '❌'}")
        
        print(f"   🎯 L'écran:")
        print(f"      ±0.01 µm: {'✅' if L_ecran_precision_01 else '❌'}")
        print(f"      ±0.1 µm:  {'✅' if L_ecran_precision_1 else '❌'}")
        print(f"      ±0.5 µm:  {'✅' if L_ecran_precision_5 else '❌'}")
        
        return {
            'gap_error': gap_error,
            'L_ecran_error': L_ecran_error,
            'gap_precision_001': gap_precision_001,
            'gap_precision_01': gap_precision_01,
            'gap_precision_05': gap_precision_05,
            'L_ecran_precision_01': L_ecran_precision_01,
            'L_ecran_precision_1': L_ecran_precision_1,
            'L_ecran_precision_5': L_ecran_precision_5
        }
    
    def create_visualization(self, I_profile_raw, r_exp, I_processed, gap_pred, L_ecran_pred, 
                           gap_target=0.115, L_ecran_target=10.30):
        """Crée une visualisation complète des résultats."""
        
        print("📈 Création de la visualisation...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'TEST MODÈLE SPÉCIALISÉ - PROFIL EXPÉRIMENTAL 50', 
                     fontsize=16, fontweight='bold')
        
        # 1. Profil expérimental brut
        ax1 = axes[0, 0]
        ax1.plot(r_exp, I_profile_raw, 'b-', linewidth=1, alpha=0.7, label='Profil brut')
        ax1.axvline(self.r_min, color='red', linestyle='--', alpha=0.7, label='Limites réseau')
        ax1.axvline(self.r_max, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Position radiale (µm)')
        ax1.set_ylabel('Intensité')
        ax1.set_title('Profil Expérimental Brut')
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
        
        categories = ['Gap', "L'écran"]
        predicted = [gap_pred, L_ecran_pred]
        targets = [gap_target, L_ecran_target]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, predicted, width, label='Prédit', alpha=0.8, color='blue')
        bars2 = ax3.bar(x + width/2, targets, width, label='Cible', alpha=0.8, color='red')
        
        ax3.set_xlabel('Paramètres')
        ax3.set_ylabel('Valeur (µm)')
        ax3.set_title('Prédictions vs Cibles')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars1, predicted):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        for bar, val in zip(bars2, targets):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Résumé des résultats
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        gap_error = abs(gap_pred - gap_target)
        L_ecran_error = abs(L_ecran_pred - L_ecran_target)
        
        summary_text = f"""RÉSULTATS DU TEST
        
🎯 PROFIL EXPÉRIMENTAL 50

📊 PRÉDICTIONS:
   Gap: {gap_pred:.6f} µm
   L'écran: {L_ecran_pred:.3f} µm

🎯 CIBLES:
   Gap: {gap_target:.6f} µm
   L'écran: {L_ecran_target:.2f} µm

📊 ERREURS:
   Gap: {gap_error:.6f} µm
   L'écran: {L_ecran_error:.3f} µm

📊 PRÉCISION:
   Gap ±0.01 µm: {'✅' if gap_error <= 0.01 else '❌'}
   Gap ±0.05 µm: {'✅' if gap_error <= 0.05 else '❌'}
   L'écran ±0.5 µm: {'✅' if L_ecran_error <= 0.5 else '❌'}

🧠 MODÈLE: Architecture Spécialisée
📊 DONNÉES: Sans normalisation
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarder
        output_file = "../../visualizations/plots/experimental_profile_50_test.png"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualisation sauvegardée: {output_file}")
        
        plt.show()
    
    def run_complete_test(self, profile_idx=50, gap_target=0.115, L_ecran_target=10.30):
        """Lance le test complet sur le profil expérimental."""
        
        try:
            # 1. Charger le modèle
            checkpoint = self.load_specialized_model()
            
            # 2. Charger le profil expérimental
            I_profile_raw, r_exp = self.load_experimental_profile(profile_idx)
            
            # 3. Prétraiter le profil
            I_processed = self.preprocess_experimental_profile(I_profile_raw, r_exp)
            
            # 4. Faire la prédiction
            gap_pred, L_ecran_pred = self.predict_on_experimental_profile(I_processed)
            
            # 5. Comparer avec les cibles
            precision_results = self.compare_with_targets(gap_pred, L_ecran_pred, gap_target, L_ecran_target)
            
            # 6. Créer la visualisation
            self.create_visualization(I_profile_raw, r_exp, I_processed, 
                                    gap_pred, L_ecran_pred, gap_target, L_ecran_target)
            
            print(f"\n✅ TEST COMPLET TERMINÉ!")
            print(f"   🎯 Gap: {gap_pred:.6f} µm (erreur: {precision_results['gap_error']:.6f})")
            print(f"   🎯 L'écran: {L_ecran_pred:.3f} µm (erreur: {precision_results['L_ecran_error']:.3f})")
            
            return {
                'gap_pred': gap_pred,
                'L_ecran_pred': L_ecran_pred,
                'gap_target': gap_target,
                'L_ecran_target': L_ecran_target,
                **precision_results
            }
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            raise

def main():
    """Fonction principale."""
    
    tester = ExperimentalProfileTester()
    results = tester.run_complete_test(profile_idx=49, gap_target=0.115, L_ecran_target=10.30)
    
    print(f"\n🎉 TEST SUR PROFIL EXPÉRIMENTAL 50 TERMINÉ!")

if __name__ == "__main__":
    main()
