#!/usr/bin/env python3
"""
Création des datasets d'entraînement (simulé) et de test (expérimental bruité)
pour l'entraînement d'un réseau de neurones
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import random
import glob
import pickle
from experimental_noise_simulator import ExperimentalNoiseSimulator

def create_training_dataset(num_samples=5000):
    """Crée le dataset d'entraînement à partir des données simulées propres"""
    
    # Trouver tous les fichiers .mat
    mat_files = glob.glob("*.mat")
    mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
    
    if len(mat_files) == 0:
        print("Aucun fichier .mat trouvé")
        return None
    
    print(f"Création du dataset d'entraînement avec {num_samples} échantillons...")
    
    # Sélectionner des échantillons aléatoires
    if len(mat_files) < num_samples:
        print(f"Attention: seulement {len(mat_files)} fichiers disponibles")
        selected_files = mat_files
    else:
        selected_files = random.sample(mat_files, num_samples)
    
    training_data = {
        'ratios': [],
        'gaps': [],
        'lengths': [],
        'filenames': []
    }
    
    valid_samples = 0
    
    for i, filepath in enumerate(selected_files):
        try:
            data = scipy.io.loadmat(filepath)
            
            if 'ratio' in data and 'gap' in data and 'L_ecran_subs' in data:
                ratio = data['ratio'].flatten()
                gap = float(data['gap'][0, 0])
                length = float(data['L_ecran_subs'][0, 0])
                
                training_data['ratios'].append(ratio)
                training_data['gaps'].append(gap)
                training_data['lengths'].append(length)
                training_data['filenames'].append(os.path.basename(filepath))
                
                valid_samples += 1
                
                if (i + 1) % 1000 == 0:
                    print(f"  Progression: {i + 1}/{len(selected_files)}")
                    
        except Exception as e:
            continue
    
    # Convertir en arrays numpy
    training_data['ratios'] = np.array(training_data['ratios'])
    training_data['gaps'] = np.array(training_data['gaps'])
    training_data['lengths'] = np.array(training_data['lengths'])
    
    print(f"✓ Dataset d'entraînement créé: {valid_samples} échantillons")
    
    return training_data

def create_test_dataset(num_samples=500, noise_distribution=None):
    """Crée le dataset de test avec bruit expérimental"""
    
    if noise_distribution is None:
        noise_distribution = {
            'light': 0.2,      # 20% bruit léger
            'realistic': 0.5,   # 50% bruit réaliste
            'heavy': 0.25,      # 25% bruit important
            'extreme': 0.05     # 5% bruit extrême
        }
    
    # Trouver tous les fichiers .mat
    mat_files = glob.glob("*.mat")
    mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
    
    if len(mat_files) == 0:
        print("Aucun fichier .mat trouvé")
        return None
    
    print(f"Création du dataset de test avec {num_samples} échantillons bruités...")
    
    # Sélectionner des échantillons aléatoires
    if len(mat_files) < num_samples:
        print(f"Attention: seulement {len(mat_files)} fichiers disponibles")
        selected_files = mat_files
    else:
        selected_files = random.sample(mat_files, num_samples)
    
    simulator = ExperimentalNoiseSimulator()
    
    test_data = {
        'clean_ratios': [],
        'noisy_ratios': [],
        'gaps': [],
        'lengths': [],
        'noise_profiles': [],
        'filenames': []
    }
    
    # Créer la distribution des profils de bruit
    noise_profiles = []
    for profile, proportion in noise_distribution.items():
        count = int(num_samples * proportion)
        noise_profiles.extend([profile] * count)
    
    # Compléter si nécessaire
    while len(noise_profiles) < num_samples:
        noise_profiles.append('realistic')
    
    # Mélanger
    random.shuffle(noise_profiles)
    
    valid_samples = 0
    
    for i, filepath in enumerate(selected_files):
        try:
            data = scipy.io.loadmat(filepath)
            
            if 'ratio' in data and 'gap' in data and 'L_ecran_subs' in data:
                clean_ratio = data['ratio'].flatten()
                gap = float(data['gap'][0, 0])
                length = float(data['L_ecran_subs'][0, 0])
                
                # Appliquer le bruit expérimental
                noise_profile = noise_profiles[valid_samples] if valid_samples < len(noise_profiles) else 'realistic'
                noisy_ratio = simulator.simulate_experimental_data(clean_ratio, noise_profile)
                
                test_data['clean_ratios'].append(clean_ratio)
                test_data['noisy_ratios'].append(noisy_ratio)
                test_data['gaps'].append(gap)
                test_data['lengths'].append(length)
                test_data['noise_profiles'].append(noise_profile)
                test_data['filenames'].append(os.path.basename(filepath))
                
                valid_samples += 1
                
                if (i + 1) % 100 == 0:
                    print(f"  Progression: {i + 1}/{len(selected_files)}")
                    
        except Exception as e:
            continue
    
    # Convertir en arrays numpy
    test_data['clean_ratios'] = np.array(test_data['clean_ratios'])
    test_data['noisy_ratios'] = np.array(test_data['noisy_ratios'])
    test_data['gaps'] = np.array(test_data['gaps'])
    test_data['lengths'] = np.array(test_data['lengths'])
    
    print(f"✓ Dataset de test créé: {valid_samples} échantillons")
    
    return test_data

def save_datasets(training_data, test_data, output_dir="ml_datasets"):
    """Sauvegarde les datasets dans différents formats"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Format pickle (pour Python)
    with open(os.path.join(output_dir, 'training_data.pkl'), 'wb') as f:
        pickle.dump(training_data, f)
    
    with open(os.path.join(output_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    
    # Format .mat (pour MATLAB)
    scipy.io.savemat(os.path.join(output_dir, 'training_data.mat'), training_data)
    scipy.io.savemat(os.path.join(output_dir, 'test_data.mat'), test_data)
    
    # Format numpy (pour facilité d'utilisation)
    np.savez(os.path.join(output_dir, 'training_data.npz'), **training_data)
    np.savez(os.path.join(output_dir, 'test_data.npz'), **test_data)
    
    print(f"✓ Datasets sauvegardés dans le dossier '{output_dir}'")

def visualize_datasets(training_data, test_data, num_examples=8):
    """Visualise des exemples des datasets d'entraînement et de test"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Exemples d'entraînement (données propres)
    for i in range(4):
        idx = random.randint(0, len(training_data['ratios']) - 1)
        
        ax = axes[0, i]
        x_vals = np.arange(len(training_data['ratios'][idx]))
        ax.plot(x_vals, training_data['ratios'][idx], 'b-', linewidth=2)
        ax.set_title(f'Entraînement\nGap={training_data["gaps"][idx]:.3f}μm\n'
                    f'L={training_data["lengths"][idx]:.1f}μm', fontsize=10)
        ax.set_xlabel('Index')
        ax.set_ylabel('Ratio')
        ax.grid(True, alpha=0.3)
    
    # Exemples de test (données bruitées)
    for i in range(4):
        idx = random.randint(0, len(test_data['noisy_ratios']) - 1)
        
        ax = axes[1, i]
        x_vals = np.arange(len(test_data['noisy_ratios'][idx]))
        
        # Tracer propre et bruité
        ax.plot(x_vals, test_data['clean_ratios'][idx], 'b-', linewidth=2, 
                alpha=0.7, label='Propre')
        ax.plot(x_vals, test_data['noisy_ratios'][idx], 'r-', linewidth=1.5, 
                alpha=0.8, label='Bruité')
        
        ax.set_title(f'Test ({test_data["noise_profiles"][idx]})\n'
                    f'Gap={test_data["gaps"][idx]:.3f}μm\n'
                    f'L={test_data["lengths"][idx]:.1f}μm', fontsize=10)
        ax.set_xlabel('Index')
        ax.set_ylabel('Ratio')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Datasets d\'Entraînement (Simulé) vs Test (Expérimental)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    plt.savefig('training_vs_test_datasets.png', dpi=300, bbox_inches='tight')
    print("Visualisation sauvegardée: training_vs_test_datasets.png")
    
    plt.show()

def analyze_datasets(training_data, test_data):
    """Analyse statistique des datasets"""
    
    print("\n=== Analyse des Datasets ===")
    
    # Dataset d'entraînement
    print(f"\nDataset d'Entraînement:")
    print(f"  Nombre d'échantillons: {len(training_data['ratios'])}")
    print(f"  Gamme des gaps: {np.min(training_data['gaps']):.4f} - {np.max(training_data['gaps']):.4f} μm")
    print(f"  Gamme des longueurs: {np.min(training_data['lengths']):.1f} - {np.max(training_data['lengths']):.1f} μm")
    print(f"  Moyenne des ratios: {np.mean(training_data['ratios']):.4f}")
    print(f"  Écart-type des ratios: {np.std(training_data['ratios']):.4f}")
    
    # Dataset de test
    print(f"\nDataset de Test:")
    print(f"  Nombre d'échantillons: {len(test_data['noisy_ratios'])}")
    print(f"  Gamme des gaps: {np.min(test_data['gaps']):.4f} - {np.max(test_data['gaps']):.4f} μm")
    print(f"  Gamme des longueurs: {np.min(test_data['lengths']):.1f} - {np.max(test_data['lengths']):.1f} μm")
    
    # Analyse du bruit
    noise_counts = {}
    for profile in test_data['noise_profiles']:
        noise_counts[profile] = noise_counts.get(profile, 0) + 1
    
    print(f"  Répartition du bruit:")
    for profile, count in noise_counts.items():
        percentage = (count / len(test_data['noise_profiles'])) * 100
        print(f"    {profile}: {count} échantillons ({percentage:.1f}%)")
    
    # SNR moyen par type de bruit
    print(f"  SNR moyen par type de bruit:")
    for profile in ['light', 'realistic', 'heavy', 'extreme']:
        profile_indices = [i for i, p in enumerate(test_data['noise_profiles']) if p == profile]
        if profile_indices:
            snr_values = []
            for idx in profile_indices:
                signal_power = np.var(test_data['clean_ratios'][idx])
                noise_power = np.var(test_data['noisy_ratios'][idx] - test_data['clean_ratios'][idx])
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                snr_values.append(snr)
            
            avg_snr = np.mean(snr_values)
            print(f"    {profile}: {avg_snr:.1f} dB")

def main():
    """Fonction principale"""
    print("=== Création des Datasets ML pour Prédiction Gap/Longueur ===\n")
    
    # Paramètres
    training_size = 3000  # Taille du dataset d'entraînement
    test_size = 500       # Taille du dataset de test
    
    # Distribution du bruit pour le test
    noise_distribution = {
        'light': 0.15,      # 15% conditions idéales
        'realistic': 0.55,  # 55% conditions normales
        'heavy': 0.25,      # 25% conditions difficiles
        'extreme': 0.05     # 5% conditions extrêmes
    }
    
    # Créer les datasets
    print("1. Création du dataset d'entraînement (données simulées propres)...")
    training_data = create_training_dataset(training_size)
    
    if training_data is None:
        print("Erreur lors de la création du dataset d'entraînement")
        return
    
    print("\n2. Création du dataset de test (données expérimentales bruitées)...")
    test_data = create_test_dataset(test_size, noise_distribution)
    
    if test_data is None:
        print("Erreur lors de la création du dataset de test")
        return
    
    # Sauvegarder les datasets
    print("\n3. Sauvegarde des datasets...")
    save_datasets(training_data, test_data)
    
    # Visualiser
    print("\n4. Génération des visualisations...")
    visualize_datasets(training_data, test_data)
    
    # Analyser
    analyze_datasets(training_data, test_data)
    
    print(f"\n✓ Datasets créés avec succès!")
    print(f"  - Entraînement: {len(training_data['ratios'])} échantillons (données propres)")
    print(f"  - Test: {len(test_data['noisy_ratios'])} échantillons (données bruitées)")
    print(f"  - Sauvegardés dans: ml_datasets/")

if __name__ == "__main__":
    main()
