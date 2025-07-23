#!/usr/bin/env python3
"""
Test du modèle ULTRA_DEEP sur le profil 50 de profile_exp_PS_3um_z_positive.mat
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script teste le modèle entraîné en appliquant exactement le même preprocessing
que celui utilisé pendant l'entraînement.
"""

import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import joblib
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    """Bloc résiduel avec connexions skip."""

    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )

        # Connexion résiduelle
        if in_features != out_features:
            self.skip_connection = nn.Linear(in_features, out_features)
        else:
            self.skip_connection = nn.Identity()

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.skip_connection(x)
        out = self.main_path(x)
        out = out + residual
        out = self.activation(out)
        out = self.dropout(out)
        return out

class UltraDeepNetwork(nn.Module):
    """
    Architecture ULTRA-PROFONDE identique à celle utilisée pour l'entraînement.
    """

    def __init__(self, input_size=601, output_size=2, dropout=0.3):
        super().__init__()

        # Architecture ULTRA-PROFONDE avec connexions résiduelles
        self.input_layer = nn.Linear(input_size, 1024)
        self.input_bn = nn.BatchNorm1d(1024)

        # Blocs résiduels profonds
        self.deep_blocks = nn.ModuleList([
            self._make_residual_block(1024, 1024, dropout),
            self._make_residual_block(1024, 512, dropout),
            self._make_residual_block(512, 512, dropout),
            self._make_residual_block(512, 256, dropout),
            self._make_residual_block(256, 256, dropout),
            self._make_residual_block(256, 128, dropout),
            self._make_residual_block(128, 128, dropout),
            self._make_residual_block(128, 64, dropout),
        ])

        # Couches finales avec attention
        self.attention = nn.MultiheadAttention(64, num_heads=8, dropout=dropout, batch_first=True)
        self.final_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, output_size)
        )

        # Initialisation des poids
        self._initialize_weights()

    def _make_residual_block(self, in_features, out_features, dropout):
        """Crée un bloc résiduel."""
        return ResidualBlock(in_features, out_features, dropout)

    def _initialize_weights(self):
        """Initialise les poids du réseau."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Couche d'entrée
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = torch.relu(x)

        # Blocs résiduels profonds
        for block in self.deep_blocks:
            x = block(x)

        # Attention mechanism
        x_unsqueezed = x.unsqueeze(1)  # (batch, 1, features)
        attn_out, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        x = attn_out.squeeze(1)  # (batch, features)

        # Couches finales
        x = self.final_layers(x)

        return x

class ProfileTester:
    """Testeur pour le profil 50 avec preprocessing exact."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_scaler = None
        self.output_scaler = None
        
        # Paramètres de preprocessing extraits de l'entraînement
        self.r_min = 1.3845845845845846
        self.r_max = 5.538338338338338
        self.delta_r = 0.006922922922922923
        self.final_points = 601
        
        print(f"🔧 Testeur initialisé")
        print(f"   💻 Device: {self.device}")
        print(f"   📏 Preprocessing: r_min={self.r_min:.6f}, r_max={self.r_max:.6f}")
        print(f"   📊 Points finaux: {self.final_points}")
    
    def load_experimental_data(self, profile_number=50):
        """Charge le profil expérimental spécifié."""

        data_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"

        if not Path(data_file).exists():
            raise FileNotFoundError(f"Fichier non trouvé: {data_file}")

        print(f"📊 Chargement du profil {profile_number} depuis {data_file}")

        # Charger les données
        data = sio.loadmat(data_file)

        # Afficher les variables disponibles
        print("   📋 Variables disponibles:")
        for key, value in data.items():
            if not key.startswith('__'):
                if hasattr(value, 'shape'):
                    print(f"      {key}: shape = {value.shape}")

        # Extraire les données selon la structure identifiée
        if 'I_profiles' in data and 'r_exp' in data:
            I_profiles = data['I_profiles']  # (50, 184)
            r_exp = data['r_exp'].flatten() * 1e6  # Conversion en µm (1, 184) -> (184,)

            print(f"   ✅ Données chargées: {I_profiles.shape[0]} profils, {I_profiles.shape[1]} points")
            print(f"   📏 Plage radiale: {r_exp[0]:.6f} - {r_exp[-1]:.6f} µm")

            # Vérifier que le profil demandé existe
            if profile_number >= I_profiles.shape[0]:
                print(f"   ⚠️  Profil {profile_number} non disponible (max: {I_profiles.shape[0]-1})")
                profile_number = I_profiles.shape[0] - 1
                print(f"   🔄 Utilisation du profil {profile_number}")

            I_profile = I_profiles[profile_number, :]

            print(f"   🎯 Profil {profile_number} extrait")
            print(f"   📊 Intensité: min={np.min(I_profile):.6f}, max={np.max(I_profile):.6f}")

            return r_exp, I_profile

        else:
            raise ValueError("Variables 'I_profiles' et 'r_exp' non trouvées dans le fichier")
    
    def preprocess_profile(self, r_exp, I_profile):
        """Applique le preprocessing EXACT utilisé pendant l'entraînement : COUPURE puis INTERPOLATION."""

        print("🔄 Application du preprocessing CORRECT...")

        # ÉTAPE 1: COUPURE D'ABORD - Extraire seulement la plage [r_min, r_max]
        print(f"   ✂️  ÉTAPE 1: Coupure sur la plage [{self.r_min:.6f}, {self.r_max:.6f}] µm")

        # Trouver les indices correspondant à l'intervalle [r_min, r_max]
        mask = (r_exp >= self.r_min) & (r_exp <= self.r_max)
        indices_valid = np.where(mask)[0]

        if len(indices_valid) == 0:
            print(f"   ❌ ERREUR: Aucun point expérimental dans l'intervalle de référence!")
            print(f"      Intervalle de référence : [{self.r_min:.6f}, {self.r_max:.6f}] µm")
            print(f"      Plage expérimentale : [{r_exp[0]:.6f}, {r_exp[-1]:.6f}] µm")
            raise ValueError("Aucun recouvrement entre données exp et plage d'entraînement")

        print(f"      ✅ {len(indices_valid)} points valides trouvés (indices {indices_valid[0]} à {indices_valid[-1]})")

        # Extraire les sous-intervalles
        r_cut = r_exp[indices_valid]
        I_cut = I_profile[indices_valid]

        print(f"      📏 Plage coupée: [{r_cut[0]:.6f}, {r_cut[-1]:.6f}] µm")
        print(f"      📊 Intensité coupée: min={np.min(I_cut):.6f}, max={np.max(I_cut):.6f}")

        # ÉTAPE 2: INTERPOLATION ENSUITE - Sur la grille d'entraînement
        print(f"   🔄 ÉTAPE 2: Interpolation sur la grille d'entraînement")

        # Créer le nouveau vecteur r avec l'espacement de référence
        r_new = np.arange(self.r_min, self.r_max + self.delta_r, self.delta_r)

        # Ajuster pour avoir exactement 601 points
        if len(r_new) != self.final_points:
            r_new = np.linspace(self.r_min, self.r_max, self.final_points)

        print(f"      📏 Grille cible: {len(r_new)} points ({r_new[0]:.6f} - {r_new[-1]:.6f} µm)")

        # Interpolation linéaire (maintenant sans extrapolation !)
        try:
            f_interp = interp1d(r_cut, I_cut, kind='linear', bounds_error=False, fill_value='extrapolate')
            I_interpolated = f_interp(r_new)

            print(f"      ✅ Interpolation réussie: {len(I_interpolated)} points")

            # Vérifier s'il y a encore de l'extrapolation
            extrap_mask = (r_new < r_cut[0]) | (r_new > r_cut[-1])
            extrap_count = np.sum(extrap_mask)

            if extrap_count > 0:
                print(f"      ⚠️  {extrap_count} points encore extrapolés ({extrap_count/len(r_new)*100:.1f}%)")
                print(f"         Plage données: [{r_cut[0]:.6f}, {r_cut[-1]:.6f}] µm")
                print(f"         Plage grille: [{r_new[0]:.6f}, {r_new[-1]:.6f}] µm")
            else:
                print(f"      ✅ Aucune extrapolation - interpolation pure!")

        except Exception as e:
            print(f"      ❌ Erreur d'interpolation: {e}")
            raise

        # ÉTAPE 3: Vérifications finales
        print(f"   🔍 ÉTAPE 3: Vérifications finales")

        if np.any(np.isnan(I_interpolated)):
            nan_count = np.sum(np.isnan(I_interpolated))
            print(f"      ⚠️  {nan_count} valeurs NaN détectées - correction...")
            I_interpolated = np.nan_to_num(I_interpolated)

        print(f"      📊 Résultat final:")
        print(f"         Forme: {I_interpolated.shape}")
        print(f"         Min: {np.min(I_interpolated):.6f}")
        print(f"         Max: {np.max(I_interpolated):.6f}")
        print(f"         Moyenne: {np.mean(I_interpolated):.6f}")
        print(f"         Std: {np.std(I_interpolated):.6f}")

        return r_new, I_interpolated
    
    def load_model_and_scalers(self):
        """Charge le modèle et les scalers."""
        
        print("📂 Chargement du modèle et des scalers...")
        
        # Chemins
        model_path = "../../results/ULTRA_DEEP_NETWORK_ALL_22540_PROFILES/best_model.pt"
        scalers_path = "../../models/saved_models/ultra_fast_scalers.joblib"
        
        # Vérifier l'existence des fichiers
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        if not Path(scalers_path).exists():
            raise FileNotFoundError(f"Scalers non trouvés: {scalers_path}")
        
        # Charger le modèle
        print("   🧠 Chargement du modèle...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = UltraDeepNetwork(
            input_size=601,
            output_size=2,
            dropout=0.3
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"      ✅ Modèle chargé (époque {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f})")
        
        # Charger les scalers
        print("   📊 Chargement des scalers...")
        scalers = joblib.load(scalers_path)
        self.input_scaler = scalers['input_scaler']
        self.output_scaler = scalers['output_scaler']
        
        print("      ✅ Scalers chargés")
    
    def predict(self, I_profile):
        """Fait la prédiction."""
        
        print("🔮 Prédiction...")
        
        # Normaliser l'entrée
        I_normalized = self.input_scaler.transform(I_profile.reshape(1, -1))
        
        # Convertir en tensor
        I_tensor = torch.FloatTensor(I_normalized).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            prediction_normalized = self.model(I_tensor)
        
        # Dénormaliser la sortie
        prediction = self.output_scaler.inverse_transform(
            prediction_normalized.cpu().numpy()
        )
        
        gap_pred = prediction[0, 0]
        L_ecran_pred = prediction[0, 1]
        
        print(f"   🎯 Gap prédit: {gap_pred:.6f} µm")
        print(f"   🎯 L'écran prédit: {L_ecran_pred:.3f} µm")
        
        return gap_pred, L_ecran_pred
    
    def visualize_profile(self, r_new, I_profile, gap_pred, L_ecran_pred):
        """Visualise le profil et les prédictions."""
        
        print("📈 Création de la visualisation...")
        
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Profil original
        plt.subplot(2, 2, 1)
        plt.plot(r_new, I_profile, 'b-', linewidth=2)
        plt.xlabel('Position radiale (µm)')
        plt.ylabel('Intensité')
        plt.title('Profil expérimental (après preprocessing)')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Détection des anneaux
        plt.subplot(2, 2, 2)
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(I_profile, height=np.mean(I_profile), distance=5)
        plt.plot(r_new, I_profile, 'b-', linewidth=2, label='Profil')
        plt.plot(r_new[peaks], I_profile[peaks], 'ro', markersize=8, label=f'{len(peaks)} anneaux')
        
        for i, peak in enumerate(peaks):
            plt.annotate(f'A{i+1}', (r_new[peak], I_profile[peak]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Position radiale (µm)')
        plt.ylabel('Intensité')
        plt.title('Détection des anneaux')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Résultats de prédiction
        plt.subplot(2, 2, 3)
        plt.axis('off')
        
        results_text = f"""
PRÉDICTIONS DU MODÈLE ULTRA-PROFOND

Gap prédit: {gap_pred:.6f} µm
L'écran prédit: {L_ecran_pred:.3f} µm

Modèle: ULTRA_DEEP_NETWORK_ALL_22540_PROFILES
Architecture: 601 → 1024 → ... → 2
Preprocessing: Interpolation {self.final_points} points
Plage radiale: {self.r_min:.3f} - {self.r_max:.3f} µm

Anneaux détectés: {len(peaks)}
Qualité du profil: {'Bonne' if len(peaks) >= 3 else 'Modérée'}
"""
        
        plt.text(0.1, 0.9, results_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Subplot 4: Distribution des intensités
        plt.subplot(2, 2, 4)
        plt.hist(I_profile, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Intensité')
        plt.ylabel('Fréquence')
        plt.title('Distribution des intensités')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        output_file = "../../visualizations/plots/profile_50_prediction_ultra_deep.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualisation sauvegardée: {output_file}")
        
        plt.show()
    
    def test_profile_50(self):
        """Test complet du profil 50."""
        
        print("🧪 TEST DU MODÈLE ULTRA-PROFOND SUR PROFIL 50")
        print("=" * 60)
        
        try:
            # 1. Charger les données expérimentales
            r_exp, I_profile = self.load_experimental_data(profile_number=50)
            
            # 2. Prétraiter
            r_new, I_processed = self.preprocess_profile(r_exp, I_profile)
            
            # 3. Charger le modèle
            self.load_model_and_scalers()
            
            # 4. Prédire
            gap_pred, L_ecran_pred = self.predict(I_processed)
            
            # 5. Visualiser
            self.visualize_profile(r_new, I_processed, gap_pred, L_ecran_pred)
            
            # 6. Sauvegarder les résultats
            results = {
                'profile_number': 50,
                'gap_predicted_um': float(gap_pred),
                'L_ecran_predicted_um': float(L_ecran_pred),
                'model_used': 'ULTRA_DEEP_NETWORK_ALL_22540_PROFILES',
                'preprocessing': {
                    'r_min': self.r_min,
                    'r_max': self.r_max,
                    'delta_r': self.delta_r,
                    'final_points': self.final_points
                }
            }
            
            import json
            results_file = "../../results/predictions/profile_50_ultra_deep_prediction.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ TEST TERMINÉ AVEC SUCCÈS!")
            print(f"📊 Gap prédit: {gap_pred:.6f} µm")
            print(f"📊 L'écran prédit: {L_ecran_pred:.3f} µm")
            print(f"💾 Résultats sauvegardés: {results_file}")
            
            return gap_pred, L_ecran_pred
            
        except Exception as e:
            print(f"❌ Erreur pendant le test: {e}")
            raise

def main():
    """Fonction principale."""
    
    tester = ProfileTester()
    gap_pred, L_ecran_pred = tester.test_profile_50()
    
    print(f"\n🎯 RÉSULTATS FINAUX:")
    print(f"   Gap: {gap_pred:.6f} µm")
    print(f"   L'écran: {L_ecran_pred:.3f} µm")

if __name__ == "__main__":
    main()
