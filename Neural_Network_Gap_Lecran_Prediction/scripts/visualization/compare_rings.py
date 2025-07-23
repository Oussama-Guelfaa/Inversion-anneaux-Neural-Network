#!/usr/bin/env python3
"""
Comparaison visuelle des anneaux : Expérimental vs Prédit
Auteur: Oussama GUELFAA
Date: 18/07/2025

Script simple pour comparer le profil expérimental 49 
avec le profil simulé prédit (gap=0.1014µm, L=9.351µm).
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import interp1d

def load_experimental_profile():
    """Charge le profil expérimental 49."""
    
    print("📊 Chargement du profil expérimental 49...")
    
    exp_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"
    data = sio.loadmat(exp_file)
    
    I_profiles = data['I_profiles']
    r_exp = data['r_exp'].flatten() * 1e6  # Conversion en µm
    
    I_profile_49 = I_profiles[49, :]
    
    print(f"   ✅ Profil 49 chargé: {len(I_profile_49)} points")
    
    return I_profile_49, r_exp

def load_predicted_profile():
    """Charge le profil simulé prédit."""

    print("📊 Chargement du profil simulé prédit...")

    pred_file = "/Users/oussamaguelfaa/Desktop/Stage/Inversion_anneaux/data_generation/Calcul_Data/dataset/gap_0.1014um_L_9.351um.mat"

    try:
        data = sio.loadmat(pred_file)

        # Extraire les données correctement
        I_pred = data['ratio'].flatten()  # Shape (1000,)
        r_pred = data['x'].flatten()      # Shape (1000,) déjà en µm
        gap_val = data['gap'][0, 0]       # Valeur scalaire
        L_ecran_val = data['L_ecran_subs'][0, 0]  # Valeur scalaire

        print(f"   ✅ Profil prédit chargé: {len(I_pred)} points")
        print(f"   📊 Gap: {gap_val:.6f} µm")
        print(f"   📊 L_écran: {L_ecran_val:.3f} µm")
        print(f"   📊 r range: [{r_pred.min():.3f}, {r_pred.max():.3f}] µm")

        return I_pred, r_pred

    except Exception as e:
        print(f"   ❌ Erreur chargement profil prédit: {e}")
        return None, None

def align_profiles(I_exp, r_exp, I_pred, r_pred):
    """Aligne les profils sur la même grille radiale."""
    
    print("🔄 Alignement des profils...")
    
    # Définir une grille commune
    r_min = max(r_exp.min(), r_pred.min())
    r_max = min(r_exp.max(), r_pred.max())
    r_common = np.linspace(r_min, r_max, 500)
    
    # Interpoler le profil expérimental
    f_exp = interp1d(r_exp, I_exp, kind='linear', bounds_error=False, fill_value='extrapolate')
    I_exp_aligned = f_exp(r_common)
    
    # Interpoler le profil prédit
    f_pred = interp1d(r_pred, I_pred, kind='linear', bounds_error=False, fill_value='extrapolate')
    I_pred_aligned = f_pred(r_common)
    
    # Traiter les NaN
    I_exp_aligned = np.nan_to_num(I_exp_aligned, nan=np.mean(I_exp_aligned[~np.isnan(I_exp_aligned)]))
    I_pred_aligned = np.nan_to_num(I_pred_aligned, nan=np.mean(I_pred_aligned[~np.isnan(I_pred_aligned)]))
    
    print(f"   ✅ Profils alignés sur {len(r_common)} points")
    print(f"   📊 Plage radiale: [{r_min:.3f}, {r_max:.3f}] µm")
    
    return I_exp_aligned, I_pred_aligned, r_common

def plot_comparison(I_exp, I_pred, r_common):
    """Trace la comparaison des anneaux."""
    
    print("📈 Création de la comparaison visuelle...")
    
    plt.figure(figsize=(12, 8))
    
    # Tracer les deux profils
    plt.plot(r_common, I_exp, 'r-', linewidth=2, label='Profil Expérimental 49', alpha=0.8)
    plt.plot(r_common, I_pred, 'b--', linewidth=2, label='Profil Simulé Prédit\n(gap=0.1014µm, L=9.351µm)', alpha=0.8)
    
    plt.xlabel('Position radiale (µm)', fontsize=12)
    plt.ylabel('Intensité', fontsize=12)
    plt.title('COMPARAISON ANNEAUX : EXPÉRIMENTAL vs PRÉDIT\nModèle Ultra-Précis', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Ajouter des informations
    plt.text(0.02, 0.98, 'Prédiction du modèle ultra-précis:\nGap = 0.1014 µm\nL\'écran = 9.351 µm', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    
    # Sauvegarder
    output_file = "../../visualizations/plots/comparison_rings_exp_vs_pred.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Comparaison sauvegardée: {output_file}")
    
    plt.show()

def calculate_similarity(I_exp, I_pred):
    """Calcule la similarité entre les profils."""
    
    print("📊 Calcul de la similarité...")
    
    # Normaliser les profils
    I_exp_norm = (I_exp - I_exp.min()) / (I_exp.max() - I_exp.min())
    I_pred_norm = (I_pred - I_pred.min()) / (I_pred.max() - I_pred.min())
    
    # Corrélation
    correlation = np.corrcoef(I_exp_norm, I_pred_norm)[0, 1]
    
    # Erreur quadratique moyenne
    mse = np.mean((I_exp_norm - I_pred_norm)**2)
    rmse = np.sqrt(mse)
    
    # Erreur absolue moyenne
    mae = np.mean(np.abs(I_exp_norm - I_pred_norm))
    
    print(f"   📊 Corrélation: {correlation:.4f}")
    print(f"   📊 RMSE: {rmse:.4f}")
    print(f"   📊 MAE: {mae:.4f}")
    
    if correlation > 0.9:
        print(f"   ✅ Excellente correspondance !")
    elif correlation > 0.8:
        print(f"   ✅ Bonne correspondance")
    elif correlation > 0.7:
        print(f"   ⚠️  Correspondance acceptable")
    else:
        print(f"   ❌ Correspondance faible")
    
    return correlation, rmse, mae

def main():
    """Fonction principale."""
    
    print("🎯 COMPARAISON ANNEAUX EXPÉRIMENTAL vs PRÉDIT")
    print("=" * 60)
    
    # 1. Charger le profil expérimental
    I_exp, r_exp = load_experimental_profile()
    
    # 2. Charger le profil prédit
    I_pred, r_pred = load_predicted_profile()
    
    if I_pred is None:
        print("❌ Impossible de charger le profil prédit")
        return
    
    # 3. Aligner les profils
    I_exp_aligned, I_pred_aligned, r_common = align_profiles(I_exp, r_exp, I_pred, r_pred)
    
    # 4. Calculer la similarité
    correlation, rmse, mae = calculate_similarity(I_exp_aligned, I_pred_aligned)
    
    # 5. Tracer la comparaison
    plot_comparison(I_exp_aligned, I_pred_aligned, r_common)
    
    print(f"\n✅ COMPARAISON TERMINÉE!")
    print(f"   🎯 Corrélation: {correlation:.4f}")
    print(f"   📊 Qualité: {'Excellente' if correlation > 0.9 else 'Bonne' if correlation > 0.8 else 'Acceptable' if correlation > 0.7 else 'Faible'}")

if __name__ == "__main__":
    main()
