#!/usr/bin/env python3
"""
Prétraitement des données d'entraînement et de test
Auteur: Oussama GUELFAA
Date: 15/07/2025

Ce script implémente le prétraitement des données selon les spécifications :
- Étape 1 : Prétraitement des données d'entraînement (Train/)
- Étape 2 : Prétraitement des données de test (Test/)
- Étape 3 : Visualisation comparative
"""

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import glob

def preprocess_training_data(train_dir):
    """
    Étape 1 : Prétraitement des données d'entraînement
    
    Args:
        train_dir (str): Chemin vers le dossier Train/
        
    Returns:
        tuple: (r_min, r_max, delta_r, processed_files_info)
    """
    print("=== Étape 1 : Prétraitement des données d'entraînement ===")
    
    # Obtenir la liste des fichiers .mat dans Train/
    mat_files = glob.glob(os.path.join(train_dir, "*.mat"))
    mat_files = [f for f in mat_files if not f.endswith('labels.mat')]  # Exclure labels.mat
    
    print(f"Nombre de fichiers d'entraînement trouvés : {len(mat_files)}")
    
    if len(mat_files) == 0:
        raise ValueError("Aucun fichier .mat trouvé dans le dossier Train/")
    
    # Charger le premier fichier pour déterminer les paramètres
    first_file = mat_files[0]
    data = scipy.io.loadmat(first_file)
    
    # Extraire x et ratio
    x = data['x'].flatten()
    ratio = data['ratio'].flatten()
    
    print(f"Longueur originale des vecteurs : {len(x)} points")
    print(f"Plage originale de x : [{x.min():.6f}, {x.max():.6f}]")
    
    # Appliquer la troncature : indices 200 à 800 inclus
    x_truncated = x[200:801]  # 200:801 pour inclure l'indice 800
    ratio_truncated = ratio[200:801]
    
    print(f"Longueur après troncature : {len(x_truncated)} points")
    print(f"Plage après troncature : [{x_truncated.min():.6f}, {x_truncated.max():.6f}]")
    
    # Calculer les paramètres de référence
    r_min = x_truncated[0]
    r_max = x_truncated[-1]
    delta_r = x_truncated[1] - x_truncated[0]
    
    print(f"Paramètres de référence :")
    print(f"  r_min = {r_min:.6f}")
    print(f"  r_max = {r_max:.6f}")
    print(f"  delta_r = {delta_r:.6f}")
    
    # Vérifier la cohérence sur quelques autres fichiers
    processed_files_info = []
    for i, mat_file in enumerate(mat_files[:5]):  # Vérifier les 5 premiers fichiers
        data = scipy.io.loadmat(mat_file)
        x_check = data['x'].flatten()[200:801]
        ratio_check = data['ratio'].flatten()[200:801]
        
        # Vérifier que x est cohérent
        if not np.allclose(x_check, x_truncated, rtol=1e-10):
            print(f"ATTENTION: Le vecteur x du fichier {os.path.basename(mat_file)} diffère!")
        
        processed_files_info.append({
            'filename': os.path.basename(mat_file),
            'x_truncated': x_check,
            'ratio_truncated': ratio_check,
            'gap': data['gap'][0, 0] if 'gap' in data else None,
            'L_ecran': data['L_ecran_subs'][0, 0] if 'L_ecran_subs' in data else None
        })
    
    print(f"Vérification effectuée sur {len(processed_files_info)} fichiers")
    
    return r_min, r_max, delta_r, processed_files_info

def preprocess_test_data(test_file, r_min, r_max, delta_r):
    """
    Étape 2 : Prétraitement des données de test
    
    Args:
        test_file (str): Chemin vers le fichier de test
        r_min (float): Rayon minimum de référence
        r_max (float): Rayon maximum de référence
        delta_r (float): Espacement de référence
        
    Returns:
        tuple: (r_new, I_profiles_interpolated)
    """
    print("\n=== Étape 2 : Prétraitement des données de test ===")
    
    # Charger le fichier de test
    data = scipy.io.loadmat(test_file)
    
    r_exp = data['r_exp'].flatten()
    I_profiles = data['I_profiles']  # Shape: (50, 184)
    
    print(f"Données expérimentales :")
    print(f"  r_exp shape: {r_exp.shape}")
    print(f"  I_profiles shape: {I_profiles.shape}")
    print(f"  Plage de r_exp : [{r_exp.min():.8f}, {r_exp.max():.8f}]")
    
    # Convertir r_exp en micromètres (il semble être en mètres)
    r_exp_um = r_exp * 1e6  # Conversion m -> µm
    print(f"  Plage de r_exp (µm) : [{r_exp_um.min():.6f}, {r_exp_um.max():.6f}]")
    
    # Trouver les indices correspondant à l'intervalle [r_min, r_max]
    mask = (r_exp_um >= r_min) & (r_exp_um <= r_max)
    indices_valid = np.where(mask)[0]
    
    if len(indices_valid) == 0:
        print("ERREUR: Aucun point expérimental dans l'intervalle de référence!")
        print(f"Intervalle de référence : [{r_min:.6f}, {r_max:.6f}] µm")
        print(f"Plage expérimentale : [{r_exp_um.min():.6f}, {r_exp_um.max():.6f}] µm")
        return None, None
    
    print(f"  Nombre de points valides dans l'intervalle : {len(indices_valid)}")
    print(f"  Indices valides : {indices_valid[0]} à {indices_valid[-1]}")
    
    # Extraire les sous-intervalles
    r_cut = r_exp_um[indices_valid]
    I_profiles_cut = I_profiles[:, indices_valid]  # Shape: (50, n_valid)
    
    print(f"  Données découpées :")
    print(f"    r_cut shape: {r_cut.shape}")
    print(f"    I_profiles_cut shape: {I_profiles_cut.shape}")
    
    # Créer le nouveau vecteur r avec l'espacement de référence
    r_new = np.arange(r_min, r_max + delta_r, delta_r)
    print(f"  Nouveau vecteur r : {len(r_new)} points")
    print(f"  Plage : [{r_new.min():.6f}, {r_new.max():.6f}] µm")
    
    # Interpoler chaque profil
    I_profiles_interpolated = np.zeros((I_profiles.shape[0], len(r_new)))
    
    for i in range(I_profiles.shape[0]):
        I_cut = I_profiles_cut[i, :]
        
        # Créer l'interpolateur
        interpolator = interp1d(r_cut, I_cut, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        
        # Interpoler
        I_interp = interpolator(r_new)
        I_profiles_interpolated[i, :] = I_interp
    
    print(f"  Interpolation terminée : {I_profiles_interpolated.shape}")
    
    return r_new, I_profiles_interpolated

def visualize_comparison(r_train, ratio_train, r_test, I_test, profile_idx=0, train_idx=0):
    """
    Étape 3 : Visualisation comparative
    
    Args:
        r_train (array): Vecteur r d'entraînement
        ratio_train (array): Vecteur ratio d'entraînement
        r_test (array): Vecteur r de test
        I_test (array): Profils de test interpolés
        profile_idx (int): Indice du profil de test à afficher
        train_idx (int): Indice du profil d'entraînement à afficher
    """
    print(f"\n=== Étape 3 : Visualisation comparative ===")
    
    plt.figure(figsize=(12, 8))
    
    # Profil d'entraînement
    plt.plot(r_train, ratio_train, 'b-', linewidth=2, 
             label=f'Train - Ratio (profil {train_idx})', alpha=0.8)
    
    # Profil de test
    plt.plot(r_test, I_test[profile_idx, :], 'r-', linewidth=2, 
             label=f'Test - I_profile (profil {profile_idx})', alpha=0.8)
    
    plt.xlabel('Rayon r (µm)', fontsize=12)
    plt.ylabel('Intensité', fontsize=12)
    plt.title('Comparaison Anneau Train vs Test\n(après prétraitement et interpolation)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Ajouter des informations sur le graphique
    plt.text(0.02, 0.98, f'Longueur train: {len(r_train)} points\nLongueur test: {len(r_test)} points', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    output_path = 'comparison_train_test_preprocessed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé : {output_path}")
    
    plt.show()

def main():
    """Fonction principale"""
    print("🧠 Prétraitement des données - Neural_Network_Gap_Lecran_Prediction")
    print("=" * 70)
    
    # Chemins
    train_dir = "Train"
    test_file = "Test/profile_exp_PS_3um_z_positive.mat"
    
    # Vérifier l'existence des fichiers
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Dossier Train non trouvé : {train_dir}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Fichier de test non trouvé : {test_file}")
    
    try:
        # Étape 1 : Prétraitement des données d'entraînement
        r_min, r_max, delta_r, train_info = preprocess_training_data(train_dir)
        
        # Étape 2 : Prétraitement des données de test
        r_new, I_profiles_interpolated = preprocess_test_data(test_file, r_min, r_max, delta_r)
        
        if r_new is None:
            print("Échec du prétraitement des données de test.")
            return
        
        # Étape 3 : Visualisation
        # Utiliser le premier profil d'entraînement pour la comparaison
        r_train = train_info[0]['x_truncated']
        ratio_train = train_info[0]['ratio_truncated']
        
        visualize_comparison(r_train, ratio_train, r_new, I_profiles_interpolated, 
                           profile_idx=0, train_idx=0)
        
        # Sauvegarder les données prétraitées
        print("\n💾 Sauvegarde des données prétraitées...")

        # Créer un dictionnaire avec toutes les données prétraitées
        preprocessed_data = {
            'r_min': r_min,
            'r_max': r_max,
            'delta_r': delta_r,
            'r_train': r_train,
            'r_test': r_new,
            'I_profiles_test_interpolated': I_profiles_interpolated,
            'train_files_count': len(glob.glob(os.path.join(train_dir, "*.mat"))),
            'test_profiles_count': I_profiles_interpolated.shape[0],
            'points_per_profile': len(r_new)
        }

        # Sauvegarder en format .npz (NumPy)
        np.savez('preprocessed_data.npz', **preprocessed_data)
        print(f"   ✅ Données sauvegardées : preprocessed_data.npz")

        # Sauvegarder aussi quelques exemples de données d'entraînement
        train_examples = {
            'examples_count': min(10, len(train_info)),
            'gap_values': [info['gap'] for info in train_info[:10]],
            'L_ecran_values': [info['L_ecran'] for info in train_info[:10]],
            'ratio_examples': np.array([info['ratio_truncated'] for info in train_info[:10]])
        }
        np.savez('train_examples.npz', **train_examples)
        print(f"   ✅ Exemples d'entraînement sauvegardés : train_examples.npz")

        print("\n✅ Prétraitement terminé avec succès!")
        print(f"📊 Paramètres finaux :")
        print(f"   - Plage radiale : [{r_min:.6f}, {r_max:.6f}] µm")
        print(f"   - Espacement : {delta_r:.6f} µm")
        print(f"   - Points par profil : {len(r_new)}")
        print(f"   - Profils de test interpolés : {I_profiles_interpolated.shape[0]}")
        print(f"   - Fichiers d'entraînement : {len(glob.glob(os.path.join(train_dir, '*.mat')))}")

    except Exception as e:
        print(f"❌ Erreur lors du prétraitement : {e}")
        raise

if __name__ == "__main__":
    main()
