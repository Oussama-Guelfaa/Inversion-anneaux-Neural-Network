#!/usr/bin/env python3
"""
Méthodes de débruitage pour les anneaux d'interférence expérimentaux
Différentes approches pour nettoyer les signaux bruités
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal, ndimage
from scipy.signal import savgol_filter, medfilt, wiener
from scipy.fft import fft, ifft, fftfreq
from sklearn.decomposition import PCA
import random
import glob

class InterferenceDenoiser:
    """Classe pour le débruitage des anneaux d'interférence"""
    
    def __init__(self):
        pass
    
    def savitzky_golay_filter(self, data, window_length=51, polyorder=3):
        """
        Filtre Savitzky-Golay - excellent pour préserver les pics
        Idéal pour les anneaux d'interférence car préserve la forme des oscillations
        """
        if window_length >= len(data):
            window_length = len(data) - 1
        if window_length % 2 == 0:
            window_length -= 1
        
        return savgol_filter(data, window_length, polyorder)
    
    def median_filter(self, data, kernel_size=5):
        """
        Filtre médian - excellent pour éliminer les pics de bruit impulsionnel
        """
        return medfilt(data, kernel_size)
    
    def gaussian_filter(self, data, sigma=1.0):
        """
        Filtre gaussien - lissage doux, bon pour le bruit gaussien
        """
        return ndimage.gaussian_filter1d(data, sigma)
    
    def wiener_filter(self, data, noise_variance=None):
        """
        Filtre de Wiener - optimal pour le bruit gaussien additif
        """
        if noise_variance is None:
            # Estimer la variance du bruit à partir des hautes fréquences
            noise_variance = np.var(np.diff(data))
        
        return wiener(data, noise_variance)
    
    def fourier_denoising(self, data, cutoff_freq=0.3):
        """
        Débruitage par filtrage fréquentiel
        Supprime les hautes fréquences (bruit) tout en préservant le signal
        """
        # Transformée de Fourier
        fft_data = fft(data)
        freqs = fftfreq(len(data))
        
        # Créer un filtre passe-bas
        filter_mask = np.abs(freqs) < cutoff_freq
        
        # Appliquer le filtre
        fft_filtered = fft_data * filter_mask
        
        # Transformée inverse
        denoised = np.real(ifft(fft_filtered))
        
        return denoised
    
    def adaptive_fourier_denoising(self, data, noise_threshold=0.1):
        """
        Débruitage adaptatif par seuillage dans le domaine fréquentiel
        """
        # Transformée de Fourier
        fft_data = fft(data)
        
        # Calculer le seuil adaptatif
        magnitude = np.abs(fft_data)
        threshold = noise_threshold * np.max(magnitude)
        
        # Appliquer le seuillage doux
        fft_filtered = fft_data * (magnitude > threshold)
        
        # Transformée inverse
        denoised = np.real(ifft(fft_filtered))
        
        return denoised
    
    def wavelet_denoising(self, data, wavelet='db4', threshold_mode='soft'):
        """
        Débruitage par ondelettes (nécessite PyWavelets)
        Très efficace pour préserver les détails tout en supprimant le bruit
        """
        try:
            import pywt
            
            # Décomposition en ondelettes
            coeffs = pywt.wavedec(data, wavelet, level=6)
            
            # Estimation du seuil de bruit
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(data)))
            
            # Seuillage des coefficients
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [pywt.threshold(detail, threshold, threshold_mode) 
                                for detail in coeffs_thresh[1:]]
            
            # Reconstruction
            denoised = pywt.waverec(coeffs_thresh, wavelet)
            
            return denoised[:len(data)]  # Assurer la même longueur
            
        except ImportError:
            print("PyWavelets non disponible, utilisation du filtre Savitzky-Golay")
            return self.savitzky_golay_filter(data)
    
    def bilateral_filter(self, data, sigma_spatial=2.0, sigma_intensity=0.1):
        """
        Filtre bilatéral adapté 1D - préserve les bords tout en lissant
        """
        denoised = np.zeros_like(data)
        
        for i in range(len(data)):
            # Fenêtre locale
            window_size = int(3 * sigma_spatial)
            start = max(0, i - window_size)
            end = min(len(data), i + window_size + 1)
            
            # Poids spatiaux
            spatial_weights = np.exp(-0.5 * ((np.arange(start, end) - i) / sigma_spatial) ** 2)
            
            # Poids d'intensité
            intensity_weights = np.exp(-0.5 * ((data[start:end] - data[i]) / sigma_intensity) ** 2)
            
            # Poids combinés
            weights = spatial_weights * intensity_weights
            weights /= np.sum(weights)
            
            # Valeur filtrée
            denoised[i] = np.sum(weights * data[start:end])
        
        return denoised
    
    def total_variation_denoising(self, data, lambda_reg=0.1, num_iterations=100):
        """
        Débruitage par variation totale - préserve les discontinuités
        """
        denoised = data.copy()
        
        for _ in range(num_iterations):
            # Gradient
            grad = np.gradient(denoised)
            
            # Divergence du gradient normalisé
            grad_norm = np.sqrt(grad**2 + 1e-8)
            div = np.gradient(grad / grad_norm)
            
            # Mise à jour
            denoised = denoised + lambda_reg * div
        
        return denoised
    
    def ensemble_denoising(self, data, methods=['savgol', 'fourier', 'median'], weights=None):
        """
        Débruitage par ensemble de méthodes
        Combine plusieurs techniques pour un résultat optimal
        """
        if weights is None:
            weights = [1.0] * len(methods)
        
        results = []
        
        for method in methods:
            if method == 'savgol':
                result = self.savitzky_golay_filter(data)
            elif method == 'fourier':
                result = self.fourier_denoising(data)
            elif method == 'median':
                result = self.median_filter(data)
            elif method == 'gaussian':
                result = self.gaussian_filter(data)
            elif method == 'wiener':
                result = self.wiener_filter(data)
            elif method == 'wavelet':
                result = self.wavelet_denoising(data)
            elif method == 'bilateral':
                result = self.bilateral_filter(data)
            else:
                result = data
            
            results.append(result)
        
        # Moyenne pondérée
        weights = np.array(weights)
        weights /= np.sum(weights)
        
        ensemble_result = np.zeros_like(data)
        for i, result in enumerate(results):
            ensemble_result += weights[i] * result
        
        return ensemble_result
    
    def adaptive_denoising(self, data, noise_level='auto'):
        """
        Débruitage adaptatif basé sur l'estimation du niveau de bruit
        """
        # Estimer le niveau de bruit
        if noise_level == 'auto':
            # Utiliser la variance des différences pour estimer le bruit
            noise_var = np.var(np.diff(data))
            
            if noise_var < 0.001:
                noise_level = 'light'
            elif noise_var < 0.01:
                noise_level = 'moderate'
            elif noise_var < 0.1:
                noise_level = 'heavy'
            else:
                noise_level = 'extreme'
        
        # Choisir la méthode selon le niveau de bruit
        if noise_level == 'light':
            return self.savitzky_golay_filter(data, window_length=21, polyorder=3)
        elif noise_level == 'moderate':
            return self.ensemble_denoising(data, ['savgol', 'fourier'], [0.7, 0.3])
        elif noise_level == 'heavy':
            return self.ensemble_denoising(data, ['median', 'savgol', 'fourier'], [0.3, 0.4, 0.3])
        else:  # extreme
            return self.ensemble_denoising(data, ['median', 'bilateral', 'fourier'], [0.4, 0.3, 0.3])

def load_noisy_data():
    """Charge des données bruitées pour tester le débruitage"""
    from experimental_noise_simulator import ExperimentalNoiseSimulator
    
    # Charger quelques échantillons propres
    mat_files = glob.glob("*.mat")
    mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
    
    if not mat_files:
        return None
    
    # Sélectionner quelques échantillons
    selected_files = random.sample(mat_files, min(5, len(mat_files)))
    
    simulator = ExperimentalNoiseSimulator()
    samples = []
    
    for filepath in selected_files:
        try:
            data = scipy.io.loadmat(filepath)
            if 'ratio' in data:
                clean_ratio = data['ratio'].flatten()
                gap = float(data['gap'][0, 0])
                length = float(data['L_ecran_subs'][0, 0])
                
                # Créer différents niveaux de bruit
                for noise_profile in ['realistic', 'heavy']:
                    noisy_ratio = simulator.simulate_experimental_data(clean_ratio, noise_profile)
                    
                    samples.append({
                        'clean': clean_ratio,
                        'noisy': noisy_ratio,
                        'gap': gap,
                        'length': length,
                        'noise_profile': noise_profile
                    })
        except:
            continue
    
    return samples

def compare_denoising_methods(samples, num_examples=4):
    """Compare différentes méthodes de débruitage"""
    
    denoiser = InterferenceDenoiser()
    
    methods = {
        'Savitzky-Golay': lambda x: denoiser.savitzky_golay_filter(x),
        'Fourier': lambda x: denoiser.fourier_denoising(x),
        'Médian': lambda x: denoiser.median_filter(x),
        'Ensemble': lambda x: denoiser.ensemble_denoising(x),
        'Adaptatif': lambda x: denoiser.adaptive_denoising(x)
    }
    
    fig, axes = plt.subplots(len(methods) + 2, num_examples, figsize=(20, 16))
    
    for col, sample in enumerate(samples[:num_examples]):
        x_vals = np.arange(len(sample['clean']))
        
        # Signal propre (référence)
        axes[0, col].plot(x_vals, sample['clean'], 'g-', linewidth=2, label='Propre')
        axes[0, col].set_title(f'Référence\nGap={sample["gap"]:.3f}μm, L={sample["length"]:.1f}μm')
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend()
        
        # Signal bruité
        axes[1, col].plot(x_vals, sample['noisy'], 'r-', linewidth=1.5, label=f'Bruité ({sample["noise_profile"]})')
        axes[1, col].set_title('Signal Bruité')
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend()
        
        # Méthodes de débruitage
        for i, (method_name, method_func) in enumerate(methods.items()):
            try:
                denoised = method_func(sample['noisy'])
                
                axes[i + 2, col].plot(x_vals, sample['clean'], 'g-', linewidth=1, alpha=0.7, label='Référence')
                axes[i + 2, col].plot(x_vals, denoised, 'b-', linewidth=2, label='Débruité')
                axes[i + 2, col].set_title(f'{method_name}')
                axes[i + 2, col].grid(True, alpha=0.3)
                axes[i + 2, col].legend(fontsize=8)
                
                # Calculer le SNR d'amélioration
                noise_original = sample['noisy'] - sample['clean']
                noise_denoised = denoised - sample['clean']
                
                snr_improvement = 10 * np.log10(np.var(noise_original) / (np.var(noise_denoised) + 1e-10))
                axes[i + 2, col].text(0.02, 0.98, f'SNR: +{snr_improvement:.1f}dB', 
                                     transform=axes[i + 2, col].transAxes, 
                                     verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
            except Exception as e:
                axes[i + 2, col].text(0.5, 0.5, f'Erreur:\n{str(e)[:20]}...', 
                                     ha='center', va='center', 
                                     transform=axes[i + 2, col].transAxes)
    
    plt.suptitle('Comparaison des Méthodes de Débruitage', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    plt.savefig('denoising_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparaison sauvegardée: denoising_methods_comparison.png")
    
    plt.show()

def main():
    """Fonction principale de démonstration"""
    print("=== Méthodes de Débruitage pour Anneaux d'Interférence ===\n")
    
    # Charger des données bruitées
    print("Chargement des données bruitées...")
    samples = load_noisy_data()
    
    if not samples:
        print("Aucune donnée trouvée")
        return
    
    print(f"✓ {len(samples)} échantillons bruités chargés")
    
    # Comparer les méthodes
    print("\nComparaison des méthodes de débruitage...")
    compare_denoising_methods(samples, num_examples=4)
    
    # Analyse quantitative
    print("\n=== Analyse Quantitative ===")
    denoiser = InterferenceDenoiser()
    
    methods = {
        'Savitzky-Golay': lambda x: denoiser.savitzky_golay_filter(x),
        'Fourier': lambda x: denoiser.fourier_denoising(x),
        'Ensemble': lambda x: denoiser.ensemble_denoising(x),
        'Adaptatif': lambda x: denoiser.adaptive_denoising(x)
    }
    
    for method_name, method_func in methods.items():
        snr_improvements = []
        mse_values = []
        
        for sample in samples:
            try:
                denoised = method_func(sample['noisy'])
                
                # SNR improvement
                noise_original = sample['noisy'] - sample['clean']
                noise_denoised = denoised - sample['clean']
                snr_improvement = 10 * np.log10(np.var(noise_original) / (np.var(noise_denoised) + 1e-10))
                snr_improvements.append(snr_improvement)
                
                # MSE
                mse = np.mean((denoised - sample['clean'])**2)
                mse_values.append(mse)
                
            except:
                continue
        
        if snr_improvements:
            print(f"{method_name}:")
            print(f"  SNR moyen: +{np.mean(snr_improvements):.1f} ± {np.std(snr_improvements):.1f} dB")
            print(f"  MSE moyen: {np.mean(mse_values):.6f}")
    
    print(f"\n✓ Analyse terminée!")

if __name__ == "__main__":
    main()
