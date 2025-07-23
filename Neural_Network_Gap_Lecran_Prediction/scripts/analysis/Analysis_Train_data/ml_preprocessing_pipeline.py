#!/usr/bin/env python3
"""
Pipeline de préprocessing pour l'entraînement ML avec débruitage
Intègre le débruitage dans le workflow d'entraînement du réseau de neurones
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from denoising_methods import InterferenceDenoiser
from experimental_noise_simulator import ExperimentalNoiseSimulator
import random
import glob
import os

class MLPreprocessingPipeline:
    """Pipeline complet de préprocessing pour ML"""
    
    def __init__(self, denoising_method='adaptive'):
        self.denoiser = InterferenceDenoiser()
        self.noise_simulator = ExperimentalNoiseSimulator()
        self.denoising_method = denoising_method
        
    def preprocess_for_training(self, ratio_data):
        """Préprocessing pour données d'entraînement (données propres)"""
        # Normalisation
        normalized = self.normalize_signal(ratio_data)
        
        # Augmentation de données (optionnel)
        augmented = self.data_augmentation(normalized)
        
        return augmented
    
    def preprocess_for_inference(self, ratio_data, apply_denoising=True):
        """Préprocessing pour inférence (données expérimentales potentiellement bruitées)"""
        processed = ratio_data.copy()
        
        # Débruitage si demandé
        if apply_denoising:
            processed = self.apply_denoising(processed)
        
        # Normalisation
        normalized = self.normalize_signal(processed)
        
        return normalized
    
    def apply_denoising(self, noisy_data):
        """Applique la méthode de débruitage sélectionnée"""
        if self.denoising_method == 'savgol':
            return self.denoiser.savitzky_golay_filter(noisy_data)
        elif self.denoising_method == 'fourier':
            return self.denoiser.fourier_denoising(noisy_data)
        elif self.denoising_method == 'ensemble':
            return self.denoiser.ensemble_denoising(noisy_data)
        elif self.denoising_method == 'adaptive':
            return self.denoiser.adaptive_denoising(noisy_data)
        else:
            return noisy_data
    
    def normalize_signal(self, data):
        """Normalise le signal pour l'entraînement ML"""
        # Normalisation min-max
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val - min_val > 1e-10:
            normalized = (data - min_val) / (max_val - min_val)
        else:
            normalized = data
        
        return normalized
    
    def data_augmentation(self, data, augment_factor=1.0):
        """Augmentation de données pour l'entraînement"""
        if augment_factor <= 1.0:
            return data
        
        augmented_samples = [data]
        
        # Ajouter du bruit léger pour la robustesse
        for _ in range(int(augment_factor) - 1):
            noise_level = np.random.uniform(0.01, 0.03)
            noisy = data + np.random.normal(0, noise_level, len(data))
            augmented_samples.append(noisy)
        
        return augmented_samples
    
    def create_ml_ready_dataset(self, num_train=2000, num_test=500, test_noise_levels=['realistic', 'heavy']):
        """Crée un dataset prêt pour l'entraînement ML"""
        
        # Charger les fichiers
        mat_files = glob.glob("*.mat")
        mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
        
        if len(mat_files) == 0:
            print("Aucun fichier .mat trouvé")
            return None, None
        
        print(f"Création du dataset ML...")
        print(f"  - Entraînement: {num_train} échantillons (propres)")
        print(f"  - Test: {num_test} échantillons (bruités + débruités)")
        
        # Dataset d'entraînement (données propres)
        train_files = random.sample(mat_files, min(num_train, len(mat_files)))
        
        train_data = {
            'X': [],  # Ratios normalisés
            'y_gap': [],  # Gaps
            'y_length': []  # Longueurs
        }
        
        for filepath in train_files:
            try:
                data = scipy.io.loadmat(filepath)
                if 'ratio' in data and 'gap' in data and 'L_ecran_subs' in data:
                    ratio = data['ratio'].flatten()
                    gap = float(data['gap'][0, 0])
                    length = float(data['L_ecran_subs'][0, 0])
                    
                    # Préprocessing pour entraînement
                    processed_ratio = self.preprocess_for_training(ratio)
                    
                    train_data['X'].append(processed_ratio)
                    train_data['y_gap'].append(gap)
                    train_data['y_length'].append(length)
                    
            except Exception as e:
                continue
        
        # Dataset de test (données bruitées)
        test_files = random.sample(mat_files, min(num_test, len(mat_files)))
        
        test_data = {
            'X_clean': [],  # Ratios propres normalisés
            'X_noisy': [],  # Ratios bruités normalisés
            'X_denoised': [],  # Ratios débruités normalisés
            'y_gap': [],  # Gaps
            'y_length': [],  # Longueurs
            'noise_levels': []  # Niveaux de bruit
        }
        
        for filepath in test_files:
            try:
                data = scipy.io.loadmat(filepath)
                if 'ratio' in data and 'gap' in data and 'L_ecran_subs' in data:
                    clean_ratio = data['ratio'].flatten()
                    gap = float(data['gap'][0, 0])
                    length = float(data['L_ecran_subs'][0, 0])
                    
                    # Ajouter du bruit expérimental
                    noise_level = random.choice(test_noise_levels)
                    noisy_ratio = self.noise_simulator.simulate_experimental_data(clean_ratio, noise_level)
                    
                    # Préprocessing
                    clean_processed = self.preprocess_for_training(clean_ratio)
                    noisy_processed = self.normalize_signal(noisy_ratio)
                    denoised_processed = self.preprocess_for_inference(noisy_ratio, apply_denoising=True)
                    
                    test_data['X_clean'].append(clean_processed)
                    test_data['X_noisy'].append(noisy_processed)
                    test_data['X_denoised'].append(denoised_processed)
                    test_data['y_gap'].append(gap)
                    test_data['y_length'].append(length)
                    test_data['noise_levels'].append(noise_level)
                    
            except Exception as e:
                continue
        
        # Convertir en arrays numpy
        for key in train_data:
            train_data[key] = np.array(train_data[key])
        
        for key in test_data:
            if key != 'noise_levels':
                test_data[key] = np.array(test_data[key])
        
        print(f"✓ Dataset créé:")
        print(f"  - Entraînement: {len(train_data['X'])} échantillons")
        print(f"  - Test: {len(test_data['X_clean'])} échantillons")
        
        return train_data, test_data

def evaluate_denoising_impact(test_data):
    """Évalue l'impact du débruitage sur la qualité des prédictions"""
    
    print("\n=== Évaluation de l'Impact du Débruitage ===")
    
    # Calculer les métriques de qualité
    mse_noisy = []
    mse_denoised = []
    snr_improvements = []
    
    for i in range(len(test_data['X_clean'])):
        clean = test_data['X_clean'][i]
        noisy = test_data['X_noisy'][i]
        denoised = test_data['X_denoised'][i]
        
        # MSE par rapport au signal propre
        mse_n = np.mean((noisy - clean)**2)
        mse_d = np.mean((denoised - clean)**2)
        
        mse_noisy.append(mse_n)
        mse_denoised.append(mse_d)
        
        # SNR improvement
        noise_original = noisy - clean
        noise_denoised = denoised - clean
        snr_improvement = 10 * np.log10(np.var(noise_original) / (np.var(noise_denoised) + 1e-10))
        snr_improvements.append(snr_improvement)
    
    print(f"MSE moyen (bruité): {np.mean(mse_noisy):.6f}")
    print(f"MSE moyen (débruité): {np.mean(mse_denoised):.6f}")
    print(f"Amélioration MSE: {((np.mean(mse_noisy) - np.mean(mse_denoised)) / np.mean(mse_noisy) * 100):.1f}%")
    print(f"SNR moyen d'amélioration: +{np.mean(snr_improvements):.1f} ± {np.std(snr_improvements):.1f} dB")
    
    # Analyse par niveau de bruit
    noise_levels = set(test_data['noise_levels'])
    for noise_level in noise_levels:
        indices = [i for i, level in enumerate(test_data['noise_levels']) if level == noise_level]
        if indices:
            level_snr = [snr_improvements[i] for i in indices]
            level_mse_improvement = [((mse_noisy[i] - mse_denoised[i]) / mse_noisy[i] * 100) for i in indices]
            print(f"  {noise_level}: SNR +{np.mean(level_snr):.1f}dB, MSE amélioration {np.mean(level_mse_improvement):.1f}%")

def visualize_preprocessing_pipeline(test_data, num_examples=4):
    """Visualise le pipeline de préprocessing"""
    
    fig, axes = plt.subplots(4, num_examples, figsize=(20, 12))
    
    selected_indices = random.sample(range(len(test_data['X_clean'])), min(num_examples, len(test_data['X_clean'])))
    
    for col, idx in enumerate(selected_indices):
        x_vals = np.arange(len(test_data['X_clean'][idx]))
        
        # Signal propre
        axes[0, col].plot(x_vals, test_data['X_clean'][idx], 'g-', linewidth=2)
        axes[0, col].set_title(f'Propre\nGap={test_data["y_gap"][idx]:.3f}μm')
        axes[0, col].grid(True, alpha=0.3)
        
        # Signal bruité
        axes[1, col].plot(x_vals, test_data['X_noisy'][idx], 'r-', linewidth=1.5)
        axes[1, col].set_title(f'Bruité ({test_data["noise_levels"][idx]})')
        axes[1, col].grid(True, alpha=0.3)
        
        # Signal débruité
        axes[2, col].plot(x_vals, test_data['X_denoised'][idx], 'b-', linewidth=2)
        axes[2, col].set_title('Débruité')
        axes[2, col].grid(True, alpha=0.3)
        
        # Comparaison
        axes[3, col].plot(x_vals, test_data['X_clean'][idx], 'g-', linewidth=2, alpha=0.7, label='Propre')
        axes[3, col].plot(x_vals, test_data['X_denoised'][idx], 'b-', linewidth=1.5, alpha=0.8, label='Débruité')
        axes[3, col].set_title('Comparaison')
        axes[3, col].legend(fontsize=8)
        axes[3, col].grid(True, alpha=0.3)
    
    plt.suptitle('Pipeline de Préprocessing ML avec Débruitage', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    plt.savefig('ml_preprocessing_pipeline.png', dpi=300, bbox_inches='tight')
    print("Pipeline sauvegardé: ml_preprocessing_pipeline.png")
    
    plt.show()

def main():
    """Fonction principale"""
    print("=== Pipeline de Préprocessing ML avec Débruitage ===\n")
    
    # Créer le pipeline
    pipeline = MLPreprocessingPipeline(denoising_method='adaptive')
    
    # Créer les datasets
    print("Création des datasets ML...")
    train_data, test_data = pipeline.create_ml_ready_dataset(
        num_train=1000,
        num_test=200,
        test_noise_levels=['realistic', 'heavy']
    )
    
    if train_data is None or test_data is None:
        print("Erreur lors de la création des datasets")
        return
    
    # Évaluer l'impact du débruitage
    evaluate_denoising_impact(test_data)
    
    # Visualiser le pipeline
    print("\nGénération des visualisations...")
    visualize_preprocessing_pipeline(test_data)
    
    # Sauvegarder les datasets
    print("\nSauvegarde des datasets...")
    os.makedirs('ml_datasets_with_denoising', exist_ok=True)
    
    np.savez('ml_datasets_with_denoising/train_data.npz', **train_data)
    np.savez('ml_datasets_with_denoising/test_data.npz', **test_data)
    
    print(f"✓ Pipeline terminé!")
    print(f"Datasets sauvegardés dans: ml_datasets_with_denoising/")
    print(f"\nRésumé:")
    print(f"  - Entraînement: {len(train_data['X'])} échantillons propres normalisés")
    print(f"  - Test: {len(test_data['X_clean'])} échantillons (propres, bruités, débruités)")
    print(f"  - Méthode de débruitage: {pipeline.denoising_method}")

if __name__ == "__main__":
    main()
