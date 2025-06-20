#!/usr/bin/env python3
"""
Exemple d'utilisation du modèle robuste avec scaler intégré
"""

import numpy as np
import scipy.io as sio
from load_model import predict_gap, load_complete_model

def test_on_our_samples():
    """Test sur nos échantillons 0.2 et 0.21 µm"""
    
    print("🧪 TEST SUR NOS ÉCHANTILLONS")
    print("="*50)
    
    # Chemins des fichiers de test
    files = [
        ('../../data_generation/dataset/gap_0.2000um_L_10.000um.mat', 0.2000),
        ('../../data_generation/dataset/gap_0.2100um_L_10.000um.mat', 0.2100)
    ]
    
    results = []
    
    for filepath, true_gap in files:
        try:
            # Charger les données
            data = sio.loadmat(filepath)
            ratio_data = data['ratio'].flatten()
            
            # Prédiction DIRECTE - pas de normalisation manuelle !
            gap_pred = predict_gap(ratio_data)
            
            error = abs(gap_pred - true_gap)
            
            print(f"Gap {true_gap:.4f} µm:")
            print(f"  Prédit: {gap_pred:.4f} µm")
            print(f"  Erreur: {error:.4f} µm")
            
            results.append((true_gap, gap_pred, error))
            
        except Exception as e:
            print(f"❌ Erreur pour {filepath}: {e}")
    
    # Analyse de discrimination
    if len(results) == 2:
        diff_real = results[1][0] - results[0][0]
        diff_pred = results[1][1] - results[0][1]
        
        print(f"\n🔬 DISCRIMINATION:")
        print(f"  Différence réelle: {diff_real:.4f} µm")
        print(f"  Différence prédite: {diff_pred:.4f} µm")
        print(f"  Précision: {(diff_pred/diff_real)*100:.1f}%")

if __name__ == "__main__":
    test_on_our_samples()
