#!/usr/bin/env python3
"""
Vérification de la structure du fichier de simulation
"""

import scipy.io as sio
from pathlib import Path

def check_structure():
    sim_file = "../../data/raw/Train/all_banque_new_04_07_25_NEW_full.mat"
    
    if not Path(sim_file).exists():
        print(f"❌ Fichier non trouvé: {sim_file}")
        return
    
    print("🔍 Chargement et inspection du fichier...")
    data = sio.loadmat(sim_file)
    
    print("\n📋 Variables disponibles:")
    for key, value in data.items():
        if not key.startswith('__'):
            if hasattr(value, 'shape'):
                print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
            else:
                print(f"  {key}: type = {type(value)}")

if __name__ == "__main__":
    check_structure()
