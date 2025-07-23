#!/usr/bin/env python3
"""
Comparaison rapide des anneaux expérimentaux vs simulation
Auteur: Oussama GUELFAA
Date: 18/07/2025

Script simple pour tracer rapidement les anneaux expérimentaux et simulés.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
import random
from scipy.interpolate import interp1d

def quick_comparison():
    """Comparaison rapide et simple"""
    
    print("🔬 COMPARAISON RAPIDE - ANNEAUX EXPÉRIMENTAUX vs SIMULATION")
    print("=" * 60)
    
    # 1. Charger les données expérimentales
    exp_file = "../../data/raw/Test/Intensity_profiles_exp_PS_3um_20250715_confined5.mat"
    
    if not Path(exp_file).exists():
        print(f"❌ Fichier expérimental non trouvé")
        return
    
    print("📊 Chargement données expérimentales...")
    exp_data = sio.loadmat(exp_file)
    I_exp = exp_data['I_exp']
    r_exp = exp_data['r_exp'].flatten() * 1e6  # Conversion en µm
    
    print(f"   ✅ {I_exp.shape[0]} profils expérimentaux ({I_exp.shape[1]} points)")
    
    # 2. Charger quelques données de simulation
    train_dir = Path("../../data/raw/Train")
    mat_files = list(train_dir.glob("gap_*.mat"))
    
    if len(mat_files) == 0:
        print("❌ Aucun fichier de simulation trouvé")
        return
    
    print(f"📊 Chargement données simulation ({len(mat_files)} fichiers disponibles)...")
    
    # Sélectionner 5 fichiers aléatoires
    selected_files = random.sample(mat_files, min(5, len(mat_files)))
    
    sim_ratios = []
    sim_params = []
    
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
            
            # Créer vecteur radial simulation
            r_sim = np.linspace(0, r_exp[-1], len(ratio))
            
            # Interpoler pour matcher les données expérimentales
            f = interp1d(r_sim, ratio, kind='linear', bounds_error=False, fill_value='extrapolate')
            ratio_interp = f(r_exp)
            
            sim_ratios.append(ratio_interp)
            sim_params.append((gap, L_ecran))
            
        except Exception as e:
            print(f"   ⚠️  Erreur avec {file_path.name}: {e}")
            continue
    
    print(f"   ✅ {len(sim_ratios)} profils de simulation chargés")
    
    # 3. Créer la figure de comparaison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('COMPARAISON ANNEAUX EXPÉRIMENTAUX PS 3µm vs SIMULATION', 
                 fontsize=14, fontweight='bold')
    
    # Couleurs
    exp_color = 'red'
    sim_colors = ['blue', 'green', 'orange', 'purple', 'brown']
    
    # Subplot 1: Profils individuels
    ax1 = axes[0, 0]
    
    # Quelques profils expérimentaux
    exp_indices = [0, len(I_exp)//4, len(I_exp)//2, 3*len(I_exp)//4, len(I_exp)-1]
    for i, idx in enumerate(exp_indices[:3]):
        alpha = 0.8 - i * 0.1
        ax1.plot(r_exp, I_exp[idx, :], color=exp_color, alpha=alpha, 
                linewidth=1.5, label='Expérimental' if i == 0 else "")
    
    # Profils de simulation
    for i, (ratio, (gap, L_ecran)) in enumerate(zip(sim_ratios, sim_params)):
        color = sim_colors[i % len(sim_colors)]
        ax1.plot(r_exp, ratio, color=color, alpha=0.8, linewidth=1.5,
                label=f'Sim. gap={gap:.3f}µm, L={L_ecran:.1f}µm')
    
    ax1.set_xlabel('Position radiale (µm)')
    ax1.set_ylabel('Intensité')
    ax1.set_title('Profils individuels')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Profils moyens
    ax2 = axes[0, 1]
    
    mean_exp = np.mean(I_exp, axis=0)
    std_exp = np.std(I_exp, axis=0)
    
    mean_sim = np.mean(sim_ratios, axis=0)
    std_sim = np.std(sim_ratios, axis=0)
    
    ax2.plot(r_exp, mean_exp, color=exp_color, linewidth=3, label='Expérimental (moyenne)')
    ax2.fill_between(r_exp, mean_exp - std_exp, mean_exp + std_exp, 
                     color=exp_color, alpha=0.2)
    
    ax2.plot(r_exp, mean_sim, color='blue', linewidth=3, label='Simulation (moyenne)')
    ax2.fill_between(r_exp, mean_sim - std_sim, mean_sim + std_sim, 
                     color='blue', alpha=0.2)
    
    ax2.set_xlabel('Position radiale (µm)')
    ax2.set_ylabel('Intensité')
    ax2.set_title('Profils moyens avec variabilité')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Détection des anneaux
    ax3 = axes[1, 0]
    
    from scipy.signal import find_peaks
    
    # Pics expérimentaux
    peaks_exp, _ = find_peaks(mean_exp, height=np.mean(mean_exp)*0.8, distance=5)
    ax3.plot(r_exp, mean_exp, color=exp_color, linewidth=2, label='Expérimental')
    ax3.plot(r_exp[peaks_exp], mean_exp[peaks_exp], 'o', color=exp_color, 
            markersize=8, label=f'{len(peaks_exp)} anneaux exp.')
    
    # Pics simulation
    peaks_sim, _ = find_peaks(mean_sim, height=np.mean(mean_sim)*0.8, distance=5)
    ax3.plot(r_exp, mean_sim, color='blue', linewidth=2, label='Simulation')
    ax3.plot(r_exp[peaks_sim], mean_sim[peaks_sim], 's', color='blue', 
            markersize=8, label=f'{len(peaks_sim)} anneaux sim.')
    
    # Annoter les positions des anneaux
    for i, peak in enumerate(peaks_exp):
        ax3.annotate(f'E{i+1}', (r_exp[peak], mean_exp[peak]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, color=exp_color)
    
    for i, peak in enumerate(peaks_sim):
        ax3.annotate(f'S{i+1}', (r_exp[peak], mean_sim[peak]), 
                    xytext=(5, -15), textcoords='offset points', fontsize=8, color='blue')
    
    ax3.set_xlabel('Position radiale (µm)')
    ax3.set_ylabel('Intensité')
    ax3.set_title('Détection des anneaux')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Corrélation
    ax4 = axes[1, 1]
    
    correlation = np.corrcoef(mean_exp, mean_sim)[0, 1]
    
    ax4.scatter(mean_exp, mean_sim, alpha=0.6, color='purple', s=30)
    
    # Ligne de régression
    z = np.polyfit(mean_exp, mean_sim, 1)
    p = np.poly1d(z)
    ax4.plot(mean_exp, p(mean_exp), "r--", alpha=0.8, linewidth=2)
    
    # Ligne y=x
    min_val = min(np.min(mean_exp), np.min(mean_sim))
    max_val = max(np.max(mean_exp), np.max(mean_sim))
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    ax4.set_xlabel('Intensité expérimentale')
    ax4.set_ylabel('Intensité simulation')
    ax4.set_title(f'Corrélation (R = {correlation:.3f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_file = "../../visualizations/plots/quick_rings_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📈 Figure sauvegardée: {output_file}")
    
    plt.show()
    
    # Résumé
    print(f"\n📊 RÉSUMÉ DE LA COMPARAISON:")
    print(f"   • Corrélation: R = {correlation:.3f}")
    print(f"   • Anneaux expérimentaux: {len(peaks_exp)} détectés")
    print(f"   • Anneaux simulation: {len(peaks_sim)} détectés")
    print(f"   • Intensité exp.: {np.mean(I_exp):.3f} ± {np.std(I_exp):.3f}")
    print(f"   • Intensité sim.: {np.mean(sim_ratios):.3f} ± {np.std(sim_ratios):.3f}")
    
    if correlation > 0.6:
        print("   ✅ Bonne corrélation entre expérimental et simulation")
    elif correlation > 0.4:
        print("   ⚠️  Corrélation modérée - adaptation de domaine recommandée")
    else:
        print("   ❌ Faible corrélation - vérifier la cohérence des données")
    
    print(f"\n🎯 Les anneaux expérimentaux et simulés sont maintenant tracés ensemble !")

if __name__ == "__main__":
    quick_comparison()
