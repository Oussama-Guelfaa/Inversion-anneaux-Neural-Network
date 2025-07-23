#!/usr/bin/env python3
"""
Inspection du fichier profile_exp_PS_3um_z_positive.mat
"""

import scipy.io as sio
import numpy as np
from pathlib import Path

def inspect_data():
    data_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"
    
    if not Path(data_file).exists():
        print(f"❌ Fichier non trouvé: {data_file}")
        return
    
    print("🔍 Inspection du fichier profile_exp_PS_3um_z_positive.mat")
    print("=" * 60)
    
    # Charger les données
    data = sio.loadmat(data_file)
    
    print("📋 Variables disponibles:")
    for key, value in data.items():
        if not key.startswith('__'):
            if hasattr(value, 'shape'):
                print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
                if value.size < 20:  # Afficher les petites variables
                    print(f"    valeurs: {value.flatten()}")
                else:
                    print(f"    min: {np.min(value):.6f}, max: {np.max(value):.6f}")
            else:
                print(f"  {key}: type = {type(value)}")

if __name__ == "__main__":
    inspect_data()
