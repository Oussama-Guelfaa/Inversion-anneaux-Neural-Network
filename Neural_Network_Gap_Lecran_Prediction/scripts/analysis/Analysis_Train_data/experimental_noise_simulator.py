#!/usr/bin/env python3
"""
Simulateur de bruit expérimental pour les anneaux d'interférence
Transforme les données simulées parfaites en données expérimentales réalistes
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import random
import glob
from scipy import ndimage
from scipy.signal import savgol_filter

class ExperimentalNoiseSimulator:
    """Classe pour simuler différents types de bruit expérimental"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
    
    def add_gaussian_noise(self, signal, noise_level=0.05):
        """Ajoute du bruit gaussien"""
        noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
        return signal + noise
    
    def add_shot_noise(self, signal, photon_count=1000):
        """Simule le bruit de photons (Poisson)"""
        # Normaliser le signal pour simuler un nombre de photons
        normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        photons = normalized * photon_count
        noisy_photons = np.random.poisson(photons)
        # Reconvertir en signal
        noisy_signal = noisy_photons / photon_count
        # Remettre à l'échelle originale
        return noisy_signal * (np.max(signal) - np.min(signal)) + np.min(signal)
    
    def add_systematic_drift(self, signal, drift_amplitude=0.02):
        """Ajoute une dérive systématique lente"""
        x = np.linspace(0, 1, len(signal))
        drift = drift_amplitude * np.sin(2 * np.pi * x * 0.5) * np.mean(signal)
        return signal + drift
    
    def add_baseline_fluctuation(self, signal, fluctuation_level=0.03):
        """Ajoute des fluctuations de ligne de base"""
        baseline = np.random.normal(0, fluctuation_level * np.mean(signal))
        return signal + baseline
    
    def add_speckle_noise(self, signal, speckle_strength=0.1):
        """Simule le bruit de speckle optique"""
        # Bruit multiplicatif
        speckle = 1 + np.random.normal(0, speckle_strength, len(signal))
        return signal * speckle
    
    def add_vibration_noise(self, signal, freq_range=(10, 100), amplitude=0.02):
        """Simule les vibrations mécaniques"""
        x = np.linspace(0, 1, len(signal))
        vibration = 0
        # Ajouter plusieurs fréquences de vibration
        for _ in range(3):
            freq = np.random.uniform(freq_range[0], freq_range[1])
            phase = np.random.uniform(0, 2*np.pi)
            vibration += amplitude * np.sin(2 * np.pi * freq * x + phase)
        return signal + vibration * np.mean(signal)
    
    def add_detector_nonlinearity(self, signal, nonlin_coeff=0.05):
        """Simule la non-linéarité du détecteur"""
        normalized = signal / np.max(signal)
        nonlinear = normalized + nonlin_coeff * normalized**2
        return nonlinear * np.max(signal)
    
    def add_quantization_noise(self, signal, bits=12):
        """Simule la quantification ADC"""
        max_val = 2**bits - 1
        normalized = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        quantized = np.round(normalized * max_val) / max_val
        return quantized * (np.max(signal) - np.min(signal)) + np.min(signal)
    
    def add_thermal_noise(self, signal, temperature_drift=0.01):
        """Simule les effets thermiques"""
        x = np.linspace(0, 1, len(signal))
        # Dérive thermique lente
        thermal_drift = temperature_drift * x * np.mean(signal)
        # Fluctuations thermiques rapides
        thermal_fluctuations = np.random.normal(0, 0.005 * np.std(signal), len(signal))
        return signal + thermal_drift + thermal_fluctuations
    
    def simulate_experimental_data(self, clean_signal, noise_profile='realistic'):
        """
        Simule des données expérimentales complètes
        
        noise_profile options:
        - 'light': Bruit léger (conditions idéales)
        - 'realistic': Bruit réaliste (conditions normales)
        - 'heavy': Bruit important (conditions difficiles)
        - 'extreme': Bruit extrême (conditions très difficiles)
        """
        
        noisy_signal = clean_signal.copy()
        
        if noise_profile == 'light':
            noisy_signal = self.add_gaussian_noise(noisy_signal, 0.02)
            noisy_signal = self.add_shot_noise(noisy_signal, 5000)
            noisy_signal = self.add_quantization_noise(noisy_signal, 14)
            
        elif noise_profile == 'realistic':
            noisy_signal = self.add_gaussian_noise(noisy_signal, 0.05)
            noisy_signal = self.add_shot_noise(noisy_signal, 2000)
            noisy_signal = self.add_systematic_drift(noisy_signal, 0.03)
            noisy_signal = self.add_speckle_noise(noisy_signal, 0.08)
            noisy_signal = self.add_vibration_noise(noisy_signal, amplitude=0.02)
            noisy_signal = self.add_thermal_noise(noisy_signal, 0.015)
            noisy_signal = self.add_quantization_noise(noisy_signal, 12)
            
        elif noise_profile == 'heavy':
            noisy_signal = self.add_gaussian_noise(noisy_signal, 0.08)
            noisy_signal = self.add_shot_noise(noisy_signal, 1000)
            noisy_signal = self.add_systematic_drift(noisy_signal, 0.05)
            noisy_signal = self.add_baseline_fluctuation(noisy_signal, 0.04)
            noisy_signal = self.add_speckle_noise(noisy_signal, 0.12)
            noisy_signal = self.add_vibration_noise(noisy_signal, amplitude=0.04)
            noisy_signal = self.add_detector_nonlinearity(noisy_signal, 0.08)
            noisy_signal = self.add_thermal_noise(noisy_signal, 0.025)
            noisy_signal = self.add_quantization_noise(noisy_signal, 10)
            
        elif noise_profile == 'extreme':
            noisy_signal = self.add_gaussian_noise(noisy_signal, 0.12)
            noisy_signal = self.add_shot_noise(noisy_signal, 500)
            noisy_signal = self.add_systematic_drift(noisy_signal, 0.08)
            noisy_signal = self.add_baseline_fluctuation(noisy_signal, 0.06)
            noisy_signal = self.add_speckle_noise(noisy_signal, 0.15)
            noisy_signal = self.add_vibration_noise(noisy_signal, amplitude=0.06)
            noisy_signal = self.add_detector_nonlinearity(noisy_signal, 0.12)
            noisy_signal = self.add_thermal_noise(noisy_signal, 0.04)
            noisy_signal = self.add_quantization_noise(noisy_signal, 8)
        
        return noisy_signal

def load_clean_data(filepath):
    """Charge les données simulées propres"""
    try:
        data = scipy.io.loadmat(filepath)
        return {
            'ratio': data['ratio'].flatten() if 'ratio' in data else None,
            'gap': float(data['gap'][0, 0]) if 'gap' in data else None,
            'length': float(data['L_ecran_subs'][0, 0]) if 'L_ecran_subs' in data else None,
            'x': data['x'].flatten() if 'x' in data else None,
            'filename': os.path.basename(filepath)
        }
    except Exception as e:
        print(f"Erreur lors du chargement de {filepath}: {e}")
        return None

def create_experimental_dataset(num_samples=50, noise_profiles=['light', 'realistic', 'heavy']):
    """Crée un dataset expérimental avec différents niveaux de bruit"""
    
    # Trouver tous les fichiers .mat
    mat_files = glob.glob("*.mat")
    mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
    
    if len(mat_files) == 0:
        print("Aucun fichier .mat trouvé")
        return
    
    # Sélectionner des échantillons aléatoires
    selected_files = random.sample(mat_files, min(num_samples, len(mat_files)))
    
    simulator = ExperimentalNoiseSimulator()
    
    experimental_data = []
    
    print(f"Génération de {len(selected_files)} échantillons expérimentaux...")
    
    for i, filepath in enumerate(selected_files):
        clean_data = load_clean_data(filepath)
        
        if clean_data and clean_data['ratio'] is not None:
            # Choisir un profil de bruit aléatoire
            noise_profile = random.choice(noise_profiles)
            
            # Simuler les données expérimentales
            experimental_ratio = simulator.simulate_experimental_data(
                clean_data['ratio'], noise_profile
            )
            
            experimental_sample = {
                'original_filename': clean_data['filename'],
                'gap': clean_data['gap'],
                'length': clean_data['length'],
                'x': clean_data['x'],
                'clean_ratio': clean_data['ratio'],
                'experimental_ratio': experimental_ratio,
                'noise_profile': noise_profile
            }
            
            experimental_data.append(experimental_sample)
            
            if (i + 1) % 10 == 0:
                print(f"  Progression: {i + 1}/{len(selected_files)}")
    
    print(f"✓ {len(experimental_data)} échantillons expérimentaux générés")
    return experimental_data

def visualize_noise_comparison(experimental_data, num_examples=6):
    """Visualise la comparaison entre données propres et bruitées"""
    
    if len(experimental_data) < num_examples:
        num_examples = len(experimental_data)
    
    selected_samples = random.sample(experimental_data, num_examples)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, sample in enumerate(selected_samples):
        if i >= 6:
            break
            
        ax = axes[i]
        
        x_vals = np.arange(len(sample['clean_ratio']))
        
        # Tracer les données propres et bruitées
        ax.plot(x_vals, sample['clean_ratio'], 'b-', linewidth=2, 
                label='Simulé (propre)', alpha=0.8)
        ax.plot(x_vals, sample['experimental_ratio'], 'r-', linewidth=1.5, 
                label=f'Expérimental ({sample["noise_profile"]})', alpha=0.7)
        
        ax.set_title(f'Gap={sample["gap"]:.3f}μm, L={sample["length"]:.1f}μm\n'
                    f'Bruit: {sample["noise_profile"]}', fontsize=10)
        ax.set_xlabel('Index')
        ax.set_ylabel('Ratio')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Comparaison Données Simulées vs Expérimentales', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    plt.savefig('experimental_vs_simulated_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparaison sauvegardée: experimental_vs_simulated_comparison.png")
    
    plt.show()

def main():
    """Fonction principale"""
    print("=== Simulateur de Bruit Expérimental ===\n")
    
    # Générer des données expérimentales
    experimental_data = create_experimental_dataset(
        num_samples=30,
        noise_profiles=['light', 'realistic', 'heavy', 'extreme']
    )
    
    if not experimental_data:
        print("Aucune donnée expérimentale générée")
        return
    
    # Visualiser la comparaison
    print("\nGénération des visualisations...")
    visualize_noise_comparison(experimental_data, num_examples=6)
    
    # Statistiques sur les niveaux de bruit
    noise_counts = {}
    for sample in experimental_data:
        profile = sample['noise_profile']
        noise_counts[profile] = noise_counts.get(profile, 0) + 1
    
    print(f"\n=== Statistiques des Données Expérimentales ===")
    print(f"Total d'échantillons: {len(experimental_data)}")
    print("Répartition par niveau de bruit:")
    for profile, count in noise_counts.items():
        print(f"  - {profile}: {count} échantillons")
    
    # Calculer l'impact du bruit
    print(f"\n=== Impact du Bruit ===")
    for profile in ['light', 'realistic', 'heavy', 'extreme']:
        profile_samples = [s for s in experimental_data if s['noise_profile'] == profile]
        if profile_samples:
            snr_values = []
            for sample in profile_samples:
                signal_power = np.var(sample['clean_ratio'])
                noise_power = np.var(sample['experimental_ratio'] - sample['clean_ratio'])
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                snr_values.append(snr)
            
            avg_snr = np.mean(snr_values)
            print(f"  {profile}: SNR moyen = {avg_snr:.1f} dB")
    
    print(f"\n✓ Simulation terminée!")
    print("Fichier généré: experimental_vs_simulated_comparison.png")

if __name__ == "__main__":
    main()
