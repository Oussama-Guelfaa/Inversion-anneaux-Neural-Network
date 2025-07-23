#!/usr/bin/env python3
"""
Debug du preprocessing étape par étape
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script montre EXACTEMENT ce qui se passe avec les données de test
étape par étape pour identifier le problème.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
from scipy.interpolate import interp1d
import joblib

def debug_experimental_preprocessing():
    """Debug complet du preprocessing expérimental."""
    
    print("🔍 DEBUG PREPROCESSING EXPÉRIMENTAL - ÉTAPE PAR ÉTAPE")
    print("=" * 60)
    
    # ÉTAPE 1: Charger les données brutes
    print("\n📊 ÉTAPE 1: Chargement des données brutes")
    print("-" * 40)
    
    exp_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"
    data = sio.loadmat(exp_file)
    
    print(f"Variables dans le fichier:")
    for key, value in data.items():
        if not key.startswith('__'):
            if hasattr(value, 'shape'):
                print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
    
    # Extraire les données
    I_profiles = data['I_profiles']  # (50, 184)
    r_exp = data['r_exp'].flatten() * 1e6  # Conversion en µm
    
    print(f"\n✅ Données extraites:")
    print(f"  I_profiles: {I_profiles.shape} (50 profils, 184 points chacun)")
    print(f"  r_exp: {r_exp.shape} ({r_exp[0]:.6f} - {r_exp[-1]:.6f} µm)")
    
    # ÉTAPE 2: Sélectionner le profil 49 (dernier)
    print(f"\n📊 ÉTAPE 2: Sélection du profil 49")
    print("-" * 40)
    
    profile_idx = 49
    I_profile = I_profiles[profile_idx, :]
    
    print(f"✅ Profil {profile_idx} sélectionné:")
    print(f"  Forme: {I_profile.shape}")
    print(f"  Min: {np.min(I_profile):.6f}")
    print(f"  Max: {np.max(I_profile):.6f}")
    print(f"  Moyenne: {np.mean(I_profile):.6f}")
    print(f"  Std: {np.std(I_profile):.6f}")
    
    # ÉTAPE 3: Paramètres du réseau de neurones
    print(f"\n🧠 ÉTAPE 3: Paramètres EXACTS du réseau de neurones")
    print("-" * 40)
    
    # Ces paramètres viennent de l'entraînement
    r_min = 1.3845845845845846
    r_max = 5.538338338338338
    delta_r = 0.006922922922922923
    final_points = 601
    
    print(f"  r_min: {r_min:.10f} µm")
    print(f"  r_max: {r_max:.10f} µm")
    print(f"  delta_r: {delta_r:.10f} µm")
    print(f"  final_points: {final_points}")
    
    # Créer la grille du réseau
    r_network = np.linspace(r_min, r_max, final_points)
    
    print(f"✅ Grille réseau créée:")
    print(f"  Forme: {r_network.shape}")
    print(f"  Premier point: {r_network[0]:.10f} µm")
    print(f"  Dernier point: {r_network[-1]:.10f} µm")
    print(f"  Espacement réel: {np.diff(r_network)[0]:.10f} µm")
    
    # ÉTAPE 4: Vérifier la compatibilité des plages
    print(f"\n🔍 ÉTAPE 4: Vérification compatibilité des plages")
    print("-" * 40)
    
    print(f"Plage expérimentale: {r_exp[0]:.6f} - {r_exp[-1]:.6f} µm")
    print(f"Plage réseau:        {r_network[0]:.6f} - {r_network[-1]:.6f} µm")
    
    # Vérifier le recouvrement
    overlap_start = max(r_exp[0], r_network[0])
    overlap_end = min(r_exp[-1], r_network[-1])
    
    print(f"Recouvrement:        {overlap_start:.6f} - {overlap_end:.6f} µm")
    
    if r_exp[0] > r_network[0]:
        print(f"⚠️  PROBLÈME: Données exp commencent APRÈS la grille réseau!")
        print(f"   Manque: {r_network[0]:.6f} - {r_exp[0]:.6f} µm")
    
    if r_exp[-1] < r_network[-1]:
        print(f"⚠️  PROBLÈME: Données exp finissent AVANT la grille réseau!")
        print(f"   Manque: {r_exp[-1]:.6f} - {r_network[-1]:.6f} µm")
    
    # ÉTAPE 5: Interpolation
    print(f"\n🔄 ÉTAPE 5: Interpolation sur la grille réseau")
    print("-" * 40)
    
    print(f"Méthode: interp1d avec extrapolation")
    print(f"De: {len(I_profile)} points → {len(r_network)} points")
    
    # Interpolation
    f_interp = interp1d(r_exp, I_profile, kind='linear', bounds_error=False, fill_value='extrapolate')
    I_interpolated = f_interp(r_network)
    
    print(f"✅ Interpolation terminée:")
    print(f"  Forme: {I_interpolated.shape}")
    print(f"  Min: {np.min(I_interpolated):.6f}")
    print(f"  Max: {np.max(I_interpolated):.6f}")
    print(f"  Moyenne: {np.mean(I_interpolated):.6f}")
    print(f"  Std: {np.std(I_interpolated):.6f}")
    
    # Vérifier les NaN
    nan_count = np.sum(np.isnan(I_interpolated))
    if nan_count > 0:
        print(f"⚠️  {nan_count} valeurs NaN détectées!")
    
    # Vérifier les valeurs extrapolées
    exp_indices = (r_network >= r_exp[0]) & (r_network <= r_exp[-1])
    extrap_indices = ~exp_indices
    extrap_count = np.sum(extrap_indices)
    
    print(f"📊 Points interpolés: {np.sum(exp_indices)}")
    print(f"📊 Points extrapolés: {extrap_count}")
    
    if extrap_count > 0:
        print(f"⚠️  {extrap_count} points sont EXTRAPOLÉS (potentiellement problématiques)!")
        extrap_values = I_interpolated[extrap_indices]
        print(f"   Valeurs extrapolées: min={np.min(extrap_values):.6f}, max={np.max(extrap_values):.6f}")
    
    # ÉTAPE 6: Normalisation
    print(f"\n🔧 ÉTAPE 6: Normalisation avec les scalers du réseau")
    print("-" * 40)
    
    scalers_path = "../../models/saved_models/ultra_fast_scalers.joblib"
    
    if Path(scalers_path).exists():
        print(f"📂 Chargement des scalers d'entraînement...")
        scalers = joblib.load(scalers_path)
        input_scaler = scalers['input_scaler']
        
        print(f"✅ Scalers chargés:")
        print(f"  Type: {type(input_scaler).__name__}")
        print(f"  Moyenne d'entraînement: {input_scaler.mean_[:5]}... (5 premiers)")
        print(f"  Std d'entraînement: {input_scaler.scale_[:5]}... (5 premiers)")
        
    else:
        print(f"❌ Scalers non trouvés: {scalers_path}")
        return None, None, None, None, None
    
    # Normaliser
    I_normalized = input_scaler.transform(I_interpolated.reshape(1, -1)).flatten()
    
    print(f"✅ Normalisation appliquée:")
    print(f"  Forme: {I_normalized.shape}")
    print(f"  Min: {np.min(I_normalized):.6f}")
    print(f"  Max: {np.max(I_normalized):.6f}")
    print(f"  Moyenne: {np.mean(I_normalized):.6f}")
    print(f"  Std: {np.std(I_normalized):.6f}")
    
    # ÉTAPE 7: Visualisation comparative
    print(f"\n📈 ÉTAPE 7: Visualisation comparative")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DEBUG PREPROCESSING EXPÉRIMENTAL - ÉTAPE PAR ÉTAPE', fontsize=16, fontweight='bold')
    
    # 1. Données brutes
    ax1 = axes[0, 0]
    ax1.plot(r_exp, I_profile, 'r-', linewidth=2, label='Données brutes')
    ax1.axvline(r_network[0], color='blue', linestyle='--', alpha=0.7, label='r_min réseau')
    ax1.axvline(r_network[-1], color='blue', linestyle='--', alpha=0.7, label='r_max réseau')
    ax1.set_xlabel('Position radiale (µm)')
    ax1.set_ylabel('Intensité')
    ax1.set_title('Données expérimentales brutes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Comparaison plages
    ax2 = axes[0, 1]
    ax2.plot(r_exp, I_profile, 'r-', linewidth=2, label='Exp. brut')
    ax2.plot(r_network, I_interpolated, 'b-', linewidth=2, label='Interpolé')
    ax2.axvspan(r_exp[0], r_exp[-1], alpha=0.2, color='red', label='Plage exp.')
    ax2.axvspan(r_network[0], r_network[-1], alpha=0.2, color='blue', label='Plage réseau')
    ax2.set_xlabel('Position radiale (µm)')
    ax2.set_ylabel('Intensité')
    ax2.set_title('Comparaison des plages')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Zoom sur l'extrapolation
    ax3 = axes[0, 2]
    if extrap_count > 0:
        ax3.plot(r_network, I_interpolated, 'b-', linewidth=2)
        ax3.scatter(r_network[extrap_indices], I_interpolated[extrap_indices], 
                   color='red', s=20, label=f'{extrap_count} pts extrapolés')
        ax3.scatter(r_network[exp_indices], I_interpolated[exp_indices], 
                   color='blue', s=10, alpha=0.5, label=f'{np.sum(exp_indices)} pts interpolés')
        ax3.set_xlabel('Position radiale (µm)')
        ax3.set_ylabel('Intensité')
        ax3.set_title('Points extrapolés vs interpolés')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Pas d\'extrapolation', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Extrapolation')
    
    # 4. Données normalisées
    ax4 = axes[1, 0]
    ax4.plot(r_network, I_normalized, 'g-', linewidth=2, label='Normalisé')
    ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Position radiale (µm)')
    ax4.set_ylabel('Intensité normalisée')
    ax4.set_title('Données normalisées (ENTRÉE RÉSEAU)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Histogramme normalisé
    ax5 = axes[1, 1]
    ax5.hist(I_normalized, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax5.axvline(np.mean(I_normalized), color='red', linestyle='--', label=f'Moyenne: {np.mean(I_normalized):.3f}')
    ax5.axvline(0, color='black', linestyle='-', alpha=0.5, label='Zéro')
    ax5.set_xlabel('Intensité normalisée')
    ax5.set_ylabel('Fréquence')
    ax5.set_title('Distribution normalisée')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Résumé des problèmes
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    problems = []
    if r_exp[0] > r_network[0]:
        problems.append(f"❌ Début exp > début réseau")
    if r_exp[-1] < r_network[-1]:
        problems.append(f"❌ Fin exp < fin réseau")
    if extrap_count > 0:
        problems.append(f"⚠️ {extrap_count} points extrapolés")
    if nan_count > 0:
        problems.append(f"❌ {nan_count} valeurs NaN")
    
    if not problems:
        problems.append("✅ Aucun problème détecté")
    
    summary_text = f"""
RÉSUMÉ DU PREPROCESSING:

Données d'entrée:
• Profil: {profile_idx}
• Points: {len(I_profile)} → {len(I_normalized)}
• Plage: {r_exp[0]:.3f}-{r_exp[-1]:.3f} µm

Grille réseau:
• Points: {final_points}
• Plage: {r_network[0]:.3f}-{r_network[-1]:.3f} µm

Problèmes détectés:
{chr(10).join(problems)}

Statistiques finales:
• Min norm: {np.min(I_normalized):.3f}
• Max norm: {np.max(I_normalized):.3f}
• Moy norm: {np.mean(I_normalized):.3f}
• Std norm: {np.std(I_normalized):.3f}

PRÊT POUR LE RÉSEAU: {'✅ OUI' if not any('❌' in p for p in problems) else '❌ NON'}
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    
    # Sauvegarder
    output_file = "../../visualizations/plots/debug_preprocessing_step_by_step.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualisation sauvegardée: {output_file}")
    
    plt.show()
    
    return r_exp, I_profile, r_network, I_interpolated, I_normalized

def debug_simulation_preprocessing():
    """Debug du preprocessing d'une simulation pour comparaison."""
    
    print(f"\n🎯 DEBUG PREPROCESSING SIMULATION (pour comparaison)")
    print("=" * 60)
    
    # Chercher un fichier de simulation
    train_dir = Path("../../data/raw/Train")
    mat_files = list(train_dir.glob("gap_*.mat"))
    
    if not mat_files:
        print("❌ Aucun fichier de simulation trouvé")
        return None
    
    # Prendre le premier fichier
    sim_file = mat_files[0]
    print(f"📁 Fichier: {sim_file.name}")
    
    # Extraire paramètres
    filename = sim_file.name
    parts = filename.replace('.mat', '').split('_')
    gap = float(parts[1].replace('um', ''))
    L_ecran = float(parts[3].replace('um', ''))
    
    print(f"🎯 Paramètres: gap={gap:.6f} µm, L_ecran={L_ecran:.3f} µm")
    
    # Charger données
    data = sio.loadmat(sim_file)
    
    print(f"Variables dans le fichier:")
    for key, value in data.items():
        if not key.startswith('__'):
            if hasattr(value, 'shape'):
                print(f"  {key}: shape = {value.shape}")
    
    if 'ratio' in data:
        ratio = data['ratio'].flatten()
        print(f"✅ Variable 'ratio' trouvée: {ratio.shape}")
    elif 'otot' in data and 'oinc' in data:
        ratio = (data['otot'] / data['oinc']).flatten()
        print(f"✅ Ratio calculé otot/oinc: {ratio.shape}")
    else:
        print(f"❌ Variables ratio/otot/oinc non trouvées")
        return None
    
    print(f"📊 Simulation brute:")
    print(f"  Min: {np.min(ratio):.6f}")
    print(f"  Max: {np.max(ratio):.6f}")
    print(f"  Moyenne: {np.mean(ratio):.6f}")
    print(f"  Std: {np.std(ratio):.6f}")
    
    # Créer vecteur radial (approximation)
    r_max = 5.538338338338338
    r_sim = np.linspace(0, r_max, len(ratio))
    
    print(f"📏 Vecteur radial simulation: {len(r_sim)} points (0 - {r_max:.6f} µm)")
    
    return ratio, r_sim, gap, L_ecran

def main():
    """Fonction principale de debug."""
    
    print("🔍 DEBUG COMPLET DU PREPROCESSING")
    print("=" * 60)
    
    # Debug expérimental
    r_exp, I_profile, r_network, I_interpolated, I_normalized = debug_experimental_preprocessing()
    
    if I_normalized is None:
        print("❌ Échec du debug expérimental")
        return
    
    # Debug simulation
    sim_data = debug_simulation_preprocessing()
    
    print(f"\n🎯 CONCLUSIONS:")
    print("=" * 30)
    
    if np.any(np.isnan(I_normalized)):
        print("❌ PROBLÈME: Valeurs NaN dans les données normalisées")
    
    if np.min(I_normalized) < -5 or np.max(I_normalized) > 5:
        print("⚠️ ATTENTION: Valeurs normalisées extrêmes (possibles outliers)")
    
    extrap_count = np.sum((r_network < r_exp[0]) | (r_network > r_exp[-1]))
    if extrap_count > 0:
        print(f"⚠️ ATTENTION: {extrap_count} points extrapolés sur {len(r_network)} ({extrap_count/len(r_network)*100:.1f}%)")
    
    print(f"\n✅ Données finales prêtes pour le réseau:")
    print(f"   Forme: {I_normalized.shape}")
    print(f"   Plage: [{np.min(I_normalized):.6f}, {np.max(I_normalized):.6f}]")
    print(f"   Moyenne: {np.mean(I_normalized):.6f}")
    print(f"   Std: {np.std(I_normalized):.6f}")

if __name__ == "__main__":
    main()
