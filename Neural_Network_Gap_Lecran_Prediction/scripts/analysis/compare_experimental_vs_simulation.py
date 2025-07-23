#!/usr/bin/env python3
"""
Comparaison des anneaux expérimentaux PS 3µm avec les anneaux de simulation
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script trace les anneaux expérimentaux et simulés dans la même figure
pour comparaison visuelle directe.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pathlib import Path
import random

def load_experimental_data():
    """Charge les données expérimentales PS 3µm"""
    
    exp_file = "../../data/raw/Test/Intensity_profiles_exp_PS_3um_20250715_confined5.mat"
    
    if not Path(exp_file).exists():
        print(f"❌ Fichier expérimental non trouvé: {exp_file}")
        return None, None
    
    print("📊 Chargement des données expérimentales PS 3µm...")
    data = sio.loadmat(exp_file)
    
    I_exp = data['I_exp']        # (6596, 121)
    r_exp = data['r_exp']        # (1, 121)
    
    r_vector = r_exp.flatten() * 1e6  # Conversion en µm
    
    print(f"   ✅ {I_exp.shape[0]} profils expérimentaux chargés")
    print(f"   📏 Plage radiale: {r_vector[0]:.3f} - {r_vector[-1]:.3f} µm")
    
    return I_exp, r_vector

def load_simulation_data():
    """Charge les données de simulation depuis les fichiers individuels"""

    train_dir = Path("../../data/raw/Train")

    if not train_dir.exists():
        print(f"❌ Dossier de simulation non trouvé: {train_dir}")
        return None, None, None, None

    print("🎯 Chargement des données de simulation...")

    # Chercher les fichiers .mat individuels
    mat_files = list(train_dir.glob("gap_*.mat"))

    if len(mat_files) == 0:
        print("❌ Aucun fichier de simulation trouvé")
        return None, None, None, None

    print(f"   📁 {len(mat_files)} fichiers de simulation trouvés")

    # Charger un échantillon de fichiers (pour éviter de surcharger)
    n_samples = min(50, len(mat_files))  # Limiter à 50 fichiers
    selected_files = random.sample(mat_files, n_samples)

    ratios = []
    gaps = []
    L_ecrans = []

    for i, file_path in enumerate(selected_files):
        try:
            # Extraire gap et L_ecran du nom du fichier
            filename = file_path.name
            # Format: gap_X.XXXXum_L_YY.YYYum.mat
            parts = filename.replace('.mat', '').split('_')
            gap_str = parts[1].replace('um', '')
            L_str = parts[3].replace('um', '')

            gap = float(gap_str)
            L_ecran = float(L_str)

            # Charger le fichier
            data = sio.loadmat(file_path)

            # Extraire le ratio (utiliser les variables disponibles)
            if 'ratio' in data:
                ratio = data['ratio'].flatten()
            elif 'otot' in data and 'oinc' in data:
                ratio = (data['otot'] / data['oinc']).flatten()
            else:
                print(f"   ⚠️  Variables ratio non trouvées dans {filename}")
                continue

            ratios.append(ratio)
            gaps.append(gap)
            L_ecrans.append(L_ecran)

            if (i + 1) % 10 == 0:
                print(f"   📊 {i + 1}/{n_samples} fichiers chargés...")

        except Exception as e:
            print(f"   ❌ Erreur avec {file_path.name}: {e}")
            continue

    if len(ratios) == 0:
        print("❌ Aucune donnée de simulation chargée")
        return None, None, None, None

    # Convertir en arrays numpy
    ratio_sim = np.array(ratios)
    gap_vect = np.array(gaps)
    L_vect = np.array(L_ecrans)

    print(f"   ✅ {len(ratios)} profils simulés chargés")
    print(f"   📏 Gap range: {gap_vect.min():.4f} - {gap_vect.max():.4f} µm")
    print(f"   📏 L'écran range: {L_vect.min():.1f} - {L_vect.max():.1f} µm")
    print(f"   📏 Taille des profils: {ratio_sim.shape[1]} points")

    return ratio_sim, gap_vect, L_vect, None

def create_radial_vector_simulation(n_points, r_max=7.0):
    """Crée un vecteur radial pour les simulations (approximation)"""
    return np.linspace(0, r_max, n_points)

def select_representative_profiles(I_exp, ratio_sim, gap_vect, L_vect, n_profiles=5):
    """Sélectionne des profils représentatifs pour la comparaison"""
    
    print("🎯 Sélection de profils représentatifs...")
    
    # Sélection expérimentale
    exp_indices = []
    n_exp = I_exp.shape[0]
    
    # Profils expérimentaux: début, milieu, fin, min intensité, max intensité
    exp_indices.append(0)  # Premier
    exp_indices.append(n_exp // 2)  # Milieu
    exp_indices.append(n_exp - 1)  # Dernier
    
    # Profil avec intensité moyenne minimale
    mean_intensities = np.mean(I_exp, axis=1)
    exp_indices.append(np.argmin(mean_intensities))
    
    # Profil avec intensité moyenne maximale
    exp_indices.append(np.argmax(mean_intensities))
    
    # Sélection simulation - différents gaps et L'écran
    sim_indices = []
    
    # Gap faible, L'écran moyen
    mask1 = (gap_vect < 0.02) & (L_vect > 9.5) & (L_vect < 10.5)
    if np.any(mask1):
        sim_indices.append(np.where(mask1)[0][0])
    
    # Gap moyen, L'écran faible
    mask2 = (gap_vect > 0.1) & (gap_vect < 0.2) & (L_vect < 9.0)
    if np.any(mask2):
        sim_indices.append(np.where(mask2)[0][0])
    
    # Gap élevé, L'écran élevé
    mask3 = (gap_vect > 0.3) & (L_vect > 11.0)
    if np.any(mask3):
        sim_indices.append(np.where(mask3)[0][0])
    
    # Quelques profils aléatoires
    remaining_indices = list(range(len(gap_vect)))
    for idx in sim_indices:
        if idx in remaining_indices:
            remaining_indices.remove(idx)
    
    additional = random.sample(remaining_indices, min(2, len(remaining_indices)))
    sim_indices.extend(additional)
    
    print(f"   ✅ {len(exp_indices)} profils expérimentaux sélectionnés")
    print(f"   ✅ {len(sim_indices)} profils simulés sélectionnés")
    
    return exp_indices, sim_indices

def create_comparison_plot(I_exp, r_exp, ratio_sim, r_sim, gap_vect, L_vect,
                          exp_indices, sim_indices):
    """Crée la figure de comparaison"""

    print("📈 Création de la figure de comparaison...")

    # Interpoler les données de simulation pour qu'elles aient la même taille que les expérimentales
    from scipy.interpolate import interp1d

    print(f"   🔄 Interpolation: simulation {ratio_sim.shape[1]} → {len(r_exp)} points")

    ratio_sim_interp = np.zeros((ratio_sim.shape[0], len(r_exp)))
    for i in range(ratio_sim.shape[0]):
        f = interp1d(r_sim, ratio_sim[i, :], kind='linear', bounds_error=False, fill_value='extrapolate')
        ratio_sim_interp[i, :] = f(r_exp)

    # Utiliser les données interpolées
    ratio_sim = ratio_sim_interp
    r_sim = r_exp.copy()

    # Configuration de la figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('COMPARAISON ANNEAUX EXPÉRIMENTAUX PS 3µm vs SIMULATION',
                 fontsize=16, fontweight='bold')

    # Couleurs
    exp_color = 'red'
    sim_color = 'blue'
    
    # 1. Profils individuels superposés
    ax1 = axes[0, 0]
    
    # Tracer quelques profils expérimentaux
    for i, idx in enumerate(exp_indices[:3]):
        alpha = 0.7 - i * 0.1
        ax1.plot(r_exp, I_exp[idx, :], color=exp_color, alpha=alpha, 
                linewidth=2, label=f'Exp. {idx}' if i == 0 else "")
    
    # Tracer quelques profils simulés
    for i, idx in enumerate(sim_indices[:3]):
        alpha = 0.7 - i * 0.1
        gap = gap_vect[idx]
        L_ecran = L_vect[idx]
        ax1.plot(r_sim, ratio_sim[idx, :], color=sim_color, alpha=alpha, 
                linewidth=2, label=f'Sim. gap={gap:.3f}µm' if i == 0 else "")
    
    ax1.set_xlabel('Position radiale (µm)')
    ax1.set_ylabel('Intensité')
    ax1.set_title('Profils individuels superposés')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Profils moyens
    ax2 = axes[0, 1]
    
    mean_exp = np.mean(I_exp, axis=0)
    std_exp = np.std(I_exp, axis=0)
    
    mean_sim = np.mean(ratio_sim, axis=0)
    std_sim = np.std(ratio_sim, axis=0)
    
    ax2.plot(r_exp, mean_exp, color=exp_color, linewidth=3, label='Expérimental (moyenne)')
    ax2.fill_between(r_exp, mean_exp - std_exp, mean_exp + std_exp, 
                     color=exp_color, alpha=0.2, label='±1σ exp.')
    
    ax2.plot(r_sim, mean_sim, color=sim_color, linewidth=3, label='Simulation (moyenne)')
    ax2.fill_between(r_sim, mean_sim - std_sim, mean_sim + std_sim, 
                     color=sim_color, alpha=0.2, label='±1σ sim.')
    
    ax2.set_xlabel('Position radiale (µm)')
    ax2.set_ylabel('Intensité')
    ax2.set_title('Profils moyens avec variabilité')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Détection des anneaux
    ax3 = axes[0, 2]
    
    from scipy.signal import find_peaks
    
    # Pics expérimentaux
    peaks_exp, _ = find_peaks(mean_exp, height=np.mean(mean_exp), distance=5)
    ax3.plot(r_exp, mean_exp, color=exp_color, linewidth=2, label='Expérimental')
    ax3.plot(r_exp[peaks_exp], mean_exp[peaks_exp], 'o', color=exp_color, 
            markersize=8, label=f'{len(peaks_exp)} anneaux exp.')
    
    # Pics simulation
    peaks_sim, _ = find_peaks(mean_sim, height=np.mean(mean_sim), distance=5)
    ax3.plot(r_sim, mean_sim, color=sim_color, linewidth=2, label='Simulation')
    ax3.plot(r_sim[peaks_sim], mean_sim[peaks_sim], 's', color=sim_color, 
            markersize=8, label=f'{len(peaks_sim)} anneaux sim.')
    
    ax3.set_xlabel('Position radiale (µm)')
    ax3.set_ylabel('Intensité')
    ax3.set_title('Détection des anneaux')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution des intensités
    ax4 = axes[1, 0]
    
    ax4.hist(I_exp.flatten(), bins=50, alpha=0.6, color=exp_color, 
            density=True, label='Expérimental')
    ax4.hist(ratio_sim.flatten(), bins=50, alpha=0.6, color=sim_color, 
            density=True, label='Simulation')
    
    ax4.set_xlabel('Intensité')
    ax4.set_ylabel('Densité')
    ax4.set_title('Distribution des intensités')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Comparaison par zones radiales
    ax5 = axes[1, 1]
    
    n_zones = 10
    zone_size = len(r_exp) // n_zones
    
    zones_exp = []
    zones_sim = []
    zone_centers = []
    
    for i in range(n_zones):
        start_idx = i * zone_size
        end_idx = min((i+1) * zone_size, len(r_exp))
        
        zone_exp = np.mean(I_exp[:, start_idx:end_idx])
        zone_sim = np.mean(ratio_sim[:, start_idx:end_idx])
        zone_center = np.mean(r_exp[start_idx:end_idx])
        
        zones_exp.append(zone_exp)
        zones_sim.append(zone_sim)
        zone_centers.append(zone_center)
    
    ax5.plot(zone_centers, zones_exp, 'o-', color=exp_color, linewidth=2, 
            markersize=6, label='Expérimental')
    ax5.plot(zone_centers, zones_sim, 's-', color=sim_color, linewidth=2, 
            markersize=6, label='Simulation')
    
    ax5.set_xlabel('Position radiale (µm)')
    ax5.set_ylabel('Intensité moyenne par zone')
    ax5.set_title('Comparaison par zones radiales')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Corrélation croisée
    ax6 = axes[1, 2]
    
    # Calculer la corrélation entre profils moyens
    correlation = np.corrcoef(mean_exp, mean_sim)[0, 1]
    
    ax6.scatter(mean_exp, mean_sim, alpha=0.6, color='purple')
    
    # Ligne de régression
    z = np.polyfit(mean_exp, mean_sim, 1)
    p = np.poly1d(z)
    ax6.plot(mean_exp, p(mean_exp), "r--", alpha=0.8)
    
    ax6.set_xlabel('Intensité expérimentale')
    ax6.set_ylabel('Intensité simulation')
    ax6.set_title(f'Corrélation (R = {correlation:.3f})')
    ax6.grid(True, alpha=0.3)
    
    # Ligne y=x pour référence
    min_val = min(np.min(mean_exp), np.min(mean_sim))
    max_val = max(np.max(mean_exp), np.max(mean_sim))
    ax6.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    ax6.legend()
    
    plt.tight_layout()
    
    # Sauvegarder
    output_file = "../../visualizations/comparisons/experimental_vs_simulation_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Figure sauvegardée: {output_file}")
    
    plt.show()
    
    return correlation, len(peaks_exp), len(peaks_sim)

def generate_comparison_report(correlation, n_peaks_exp, n_peaks_sim, 
                              I_exp, ratio_sim, gap_vect, L_vect):
    """Génère un rapport de comparaison"""
    
    report = f"""
RAPPORT DE COMPARAISON - ANNEAUX EXPÉRIMENTAUX vs SIMULATION
===========================================================
Date: 18/07/2025
Auteur: Oussama GUELFAA

DONNÉES COMPARÉES:
==================
• Expérimental: {I_exp.shape[0]} profils PS 3µm (121 points)
• Simulation: {ratio_sim.shape[0]} profils (121 points)
• Gap simulation: {gap_vect.min():.4f} - {gap_vect.max():.4f} µm
• L'écran simulation: {L_vect.min():.1f} - {L_vect.max():.1f} µm

RÉSULTATS DE LA COMPARAISON:
============================
• Corrélation profils moyens: R = {correlation:.3f}
• Anneaux détectés (expérimental): {n_peaks_exp}
• Anneaux détectés (simulation): {n_peaks_sim}

STATISTIQUES:
=============
• Intensité exp. moyenne: {np.mean(I_exp):.3f} ± {np.std(I_exp):.3f}
• Intensité sim. moyenne: {np.mean(ratio_sim):.3f} ± {np.std(ratio_sim):.3f}
• Plage exp.: [{np.min(I_exp):.3f}, {np.max(I_exp):.3f}]
• Plage sim.: [{np.min(ratio_sim):.3f}, {np.max(ratio_sim):.3f}]

ÉVALUATION:
===========
"""
    
    if correlation > 0.8:
        report += "✅ EXCELLENTE corrélation entre expérimental et simulation\n"
    elif correlation > 0.6:
        report += "✅ BONNE corrélation entre expérimental et simulation\n"
    elif correlation > 0.4:
        report += "⚠️  CORRÉLATION MODÉRÉE entre expérimental et simulation\n"
    else:
        report += "❌ FAIBLE corrélation entre expérimental et simulation\n"
    
    if abs(n_peaks_exp - n_peaks_sim) <= 1:
        report += "✅ Nombre d'anneaux cohérent entre exp. et sim.\n"
    else:
        report += "⚠️  Différence significative dans le nombre d'anneaux\n"
    
    report += f"""
RECOMMANDATIONS:
================
1. {'Données compatibles' if correlation > 0.6 else 'Vérifier la cohérence des données'}
2. {'Structure d\'anneaux similaire' if abs(n_peaks_exp - n_peaks_sim) <= 1 else 'Analyser les différences de structure'}
3. Considérer l'adaptation de domaine si nécessaire
4. Valider avec plus de profils expérimentaux

FICHIERS GÉNÉRÉS:
=================
• experimental_vs_simulation_comparison.png
• experimental_vs_simulation_report.txt

Contact: Oussama GUELFAA - guelfaao@gmail.com
"""
    
    report_file = "../../reports/technical/experimental_vs_simulation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Rapport sauvegardé: {report_file}")

def main():
    """Fonction principale"""
    
    print("🔬 COMPARAISON ANNEAUX EXPÉRIMENTAUX vs SIMULATION")
    print("=" * 60)
    
    # Chargement des données
    I_exp, r_exp = load_experimental_data()
    if I_exp is None:
        return
    
    ratio_sim, gap_vect, L_vect, sim_data = load_simulation_data()
    if ratio_sim is None:
        return
    
    # Créer le vecteur radial pour la simulation
    r_sim = create_radial_vector_simulation(ratio_sim.shape[1], r_exp[-1])
    
    # Sélectionner des profils représentatifs
    exp_indices, sim_indices = select_representative_profiles(
        I_exp, ratio_sim, gap_vect, L_vect)
    
    # Créer la figure de comparaison
    correlation, n_peaks_exp, n_peaks_sim = create_comparison_plot(
        I_exp, r_exp, ratio_sim, r_sim, gap_vect, L_vect, 
        exp_indices, sim_indices)
    
    # Générer le rapport
    generate_comparison_report(correlation, n_peaks_exp, n_peaks_sim,
                              I_exp, ratio_sim, gap_vect, L_vect)
    
    print(f"\n✅ COMPARAISON TERMINÉE!")
    print(f"📊 Corrélation: R = {correlation:.3f}")
    print(f"🎯 Anneaux: {n_peaks_exp} (exp.) vs {n_peaks_sim} (sim.)")
    print(f"📁 Fichiers dans visualizations/comparisons/ et reports/technical/")

if __name__ == "__main__":
    main()
