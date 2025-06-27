#!/usr/bin/env python3
"""
Test des prédictions du modèle entraîné sur le dataset de test

Auteur: Oussama GUELFAA
Date: 25/06/2025

Ce script teste les prédictions du modèle sur des échantillons réels du dataset de test.
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import torch
import joblib
from Train import DualParameterNet, DualParameterTrainer
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trained_model(model_path="../models/dual_parameter_model.pt"):
    """
    Charge le modèle entraîné et les scalers.
    
    Returns:
        tuple: (model, input_scaler, output_scaler)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Charger le modèle
    model = DualParameterNet(input_size=600).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Charger les scalers
    scalers_dir = Path(model_path).parent
    input_scaler = joblib.load(scalers_dir / 'input_scaler.pkl')
    output_scaler = joblib.load(scalers_dir / 'output_scaler.pkl')
    
    logger.info(f"✅ Modèle chargé depuis {model_path}")
    
    return model, input_scaler, output_scaler, device

def load_test_sample(test_dir, filename):
    """
    Charge un échantillon du dataset de test.
    
    Args:
        test_dir: Répertoire du dataset de test
        filename: Nom du fichier .mat
    
    Returns:
        array: Profil d'intensité
    """
    data = loadmat(test_dir / filename)
    
    # Extraire le profil d'intensité
    if 'ratio' in data:
        ratio = data['ratio'].flatten()
    elif 'I_ratio' in data:
        ratio = data['I_ratio'].flatten()
    else:
        possible_keys = [k for k in data.keys() if not k.startswith('__')]
        if possible_keys:
            ratio = data[possible_keys[0]].flatten()
        else:
            raise ValueError(f"Aucune donnée trouvée dans {filename}")
    
    # Tronquer à 600 points
    if len(ratio) > 600:
        ratio = ratio[:600]
    elif len(ratio) < 600:
        ratio = np.pad(ratio, (0, 600 - len(ratio)), 'edge')
    
    return ratio

def predict_sample(model, input_scaler, output_scaler, device, ratio):
    """
    Fait une prédiction sur un profil d'intensité.
    
    Args:
        model: Modèle PyTorch
        input_scaler: Scaler pour les entrées
        output_scaler: Scaler pour les sorties
        device: Device PyTorch
        ratio: Profil d'intensité
    
    Returns:
        tuple: (gap_pred, L_ecran_pred)
    """
    # Normaliser
    ratio_scaled = input_scaler.transform(ratio.reshape(1, -1))
    
    # Prédiction
    with torch.no_grad():
        ratio_tensor = torch.FloatTensor(ratio_scaled).to(device)
        prediction_scaled = model(ratio_tensor)
        prediction = output_scaler.inverse_transform(prediction_scaled.cpu().numpy())
    
    gap_pred = prediction[0, 0]
    L_ecran_pred = prediction[0, 1]
    
    return gap_pred, L_ecran_pred

def test_multiple_samples(n_samples=10):
    """
    Teste le modèle sur plusieurs échantillons du dataset de test.
    
    Args:
        n_samples: Nombre d'échantillons à tester
    """
    logger.info(f"🧪 TEST DU MODÈLE SUR {n_samples} ÉCHANTILLONS DU DATASET DE TEST")
    logger.info("="*70)
    
    # Charger le modèle
    model, input_scaler, output_scaler, device = load_trained_model()
    
    # Charger le dataset de test
    test_dir = Path("../../data_generation/dataset_2D_Test")
    labels_df = pd.read_csv(test_dir / "labels.csv")
    
    logger.info(f"📊 Dataset de test: {len(labels_df)} échantillons disponibles")
    
    # Sélectionner des échantillons aléatoires
    sample_indices = np.random.choice(len(labels_df), min(n_samples, len(labels_df)), replace=False)
    
    results = []
    
    for i, idx in enumerate(sample_indices):
        sample = labels_df.iloc[idx]
        filename = sample['filename']
        true_gap = sample['gap_um']
        true_L_ecran = sample['L_um']
        
        try:
            # Charger le profil
            ratio = load_test_sample(test_dir, filename)
            
            # Faire la prédiction
            gap_pred, L_ecran_pred = predict_sample(
                model, input_scaler, output_scaler, device, ratio
            )
            
            # Calculer les erreurs
            gap_error = abs(gap_pred - true_gap)
            L_ecran_error = abs(L_ecran_pred - true_L_ecran)
            
            # Stocker les résultats
            results.append({
                'filename': filename,
                'true_gap': true_gap,
                'pred_gap': gap_pred,
                'gap_error': gap_error,
                'true_L_ecran': true_L_ecran,
                'pred_L_ecran': L_ecran_pred,
                'L_ecran_error': L_ecran_error
            })
            
            # Afficher le résultat
            logger.info(f"\n📋 Échantillon {i+1}/{len(sample_indices)}: {filename}")
            logger.info(f"   Gap:     Réel={true_gap:.4f}µm, Prédit={gap_pred:.4f}µm, Erreur={gap_error:.4f}µm")
            logger.info(f"   L_écran: Réel={true_L_ecran:.1f}µm, Prédit={L_ecran_pred:.1f}µm, Erreur={L_ecran_error:.1f}µm")
            
        except Exception as e:
            logger.error(f"❌ Erreur avec {filename}: {e}")
            continue
    
    # Calculer les statistiques globales
    if results:
        gap_errors = [r['gap_error'] for r in results]
        L_ecran_errors = [r['L_ecran_error'] for r in results]
        
        gap_mae = np.mean(gap_errors)
        gap_std = np.std(gap_errors)
        L_ecran_mae = np.mean(L_ecran_errors)
        L_ecran_std = np.std(L_ecran_errors)
        
        # Calculer R²
        true_gaps = [r['true_gap'] for r in results]
        pred_gaps = [r['pred_gap'] for r in results]
        true_L_ecrans = [r['true_L_ecran'] for r in results]
        pred_L_ecrans = [r['pred_L_ecran'] for r in results]
        
        from sklearn.metrics import r2_score
        gap_r2 = r2_score(true_gaps, pred_gaps)
        L_ecran_r2 = r2_score(true_L_ecrans, pred_L_ecrans)
        
        logger.info(f"\n📊 STATISTIQUES GLOBALES ({len(results)} échantillons)")
        logger.info("="*50)
        logger.info(f"Gap:")
        logger.info(f"   MAE: {gap_mae:.4f} ± {gap_std:.4f} µm")
        logger.info(f"   R²:  {gap_r2:.3f}")
        logger.info(f"L_écran:")
        logger.info(f"   MAE: {L_ecran_mae:.1f} ± {L_ecran_std:.1f} µm")
        logger.info(f"   R²:  {L_ecran_r2:.3f}")
        
        # Évaluer la qualité
        gap_tolerance = 0.01  # µm
        L_ecran_tolerance = 0.5  # µm
        
        gap_good = sum(1 for e in gap_errors if e <= gap_tolerance)
        L_ecran_good = sum(1 for e in L_ecran_errors if e <= L_ecran_tolerance)
        
        gap_accuracy = gap_good / len(results) * 100
        L_ecran_accuracy = L_ecran_good / len(results) * 100
        
        logger.info(f"\n🎯 PRÉCISION (tolérance: Gap ±{gap_tolerance}µm, L_écran ±{L_ecran_tolerance}µm)")
        logger.info(f"   Gap: {gap_good}/{len(results)} = {gap_accuracy:.1f}%")
        logger.info(f"   L_écran: {L_ecran_good}/{len(results)} = {L_ecran_accuracy:.1f}%")
        
        if gap_r2 > 0.8 and L_ecran_r2 > 0.9:
            logger.info(f"\n🎉 MODÈLE DE QUALITÉ EXCELLENTE !")
        elif gap_r2 > 0.6 and L_ecran_r2 > 0.8:
            logger.info(f"\n✅ MODÈLE DE BONNE QUALITÉ")
        else:
            logger.info(f"\n⚠️ MODÈLE À AMÉLIORER")
    
    return results

def main():
    """
    Fonction principale de test.
    """
    # Tester sur 20 échantillons aléatoires
    results = test_multiple_samples(n_samples=20)
    
    logger.info(f"\n✅ TEST TERMINÉ - {len(results)} échantillons testés")

if __name__ == "__main__":
    main()
