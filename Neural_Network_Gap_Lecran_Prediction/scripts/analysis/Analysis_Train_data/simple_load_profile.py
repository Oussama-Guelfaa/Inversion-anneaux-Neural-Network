#!/usr/bin/env python3
"""
Version ULTRA-SIMPLE pour charger un profil expérimental
Équivalent direct du code MATLAB en quelques lignes
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def load_profile_simple():
    """
    Version ultra-simple - équivalent direct du code MATLAB
    """
    
    # Charger le fichier (équivalent à load())
    data = scipy.io.loadmat('profile_exp_PS_3um_z_positive.mat')
    
    # Extraire les variables (équivalent aux variables MATLAB)
    I_profiles = data['I_profiles']
    r_exp = data['r_exp'].flatten()
    
    # Choisir un profil (équivalent à I_zcut2_raw=I_profiles(50,:))
    # Note: Python utilise l'indexation 0-based, donc 50 en MATLAB = 49 en Python
    I_zcut2_raw = I_profiles[49, :]  # Profil 50 en MATLAB
    
    # Afficher (équivalent à figure; plot(r_exp,I_zcut2_raw))
    plt.figure()
    plt.plot(r_exp, I_zcut2_raw)
    plt.xlabel('Position radiale r (μm)')
    plt.ylabel('Intensité')
    plt.title('Profil Expérimental')
    plt.grid(True)
    plt.show()
    
    return r_exp, I_zcut2_raw

# Version encore plus courte - 5 lignes !
def ultra_simple():
    """Version en 5 lignes seulement"""
    data = scipy.io.loadmat('profile_exp_PS_3um_z_positive.mat')
    r_exp = data['r_exp'].flatten()
    I_profile = data['I_profiles'][49, :]  # Profil 50 en MATLAB
    plt.plot(r_exp, I_profile)
    plt.show()
    return r_exp, I_profile 

if __name__ == "__main__":
    print("Chargement du profil expérimental...")
    
    try:
        # Version simple
        r_exp, I_profile = load_profile_simple()
        print("✓ Profil chargé et affiché avec succès!")
        
    except FileNotFoundError:
        print("❌ Fichier 'profile_exp_PS_3um_z_positive.mat' non trouvé")
        print("Placez le fichier dans le même dossier que ce script")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
