#!/usr/bin/env python3
"""
Test final du modèle amélioré

Auteur: Oussama GUELFAA
Date: 25/06/2025

Test complet du modèle amélioré sur le dataset de test.
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import torch
import joblib
from Train_Improved import ImprovedDualParameterNet
import logging
from sklearn.metrics import r2_score, mean_absolute_error

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_improved_model():
    """Charge le modèle amélioré."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImprovedDualParameterNet(input_size=600).to(device)
    checkpoint = torch.load("../models/dual_parameter_model_improved.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Charger les scalers améliorés
    input_scaler = joblib.load("../models/input_scaler_improved.pkl")
    gap_scaler = joblib.load("../models/gap_scaler_improved.pkl")
    L_ecran_scaler = joblib.load("../models/L_ecran_scaler_improved.pkl")
    
    logger.info(f"✅ Modèle amélioré chargé")
    
    return model, input_scaler, gap_scaler, L_ecran_scaler, device

def load_test_sample(test_dir, filename):
    """Charge un échantillon du dataset de test."""
    data = loadmat(test_dir / filename)
    
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

def predict_improved(model, input_scaler, gap_scaler, L_ecran_scaler, device, ratio):
    """Prédiction avec le modèle amélioré."""
    # Filtrage léger comme dans l'entraînement
    from scipy.ndimage import gaussian_filter1d
    ratio_filtered = gaussian_filter1d(ratio, sigma=0.5)
    
    ratio_scaled = input_scaler.transform(ratio_filtered.reshape(1, -1))
    
    with torch.no_grad():
        ratio_tensor = torch.FloatTensor(ratio_scaled).to(device)
        prediction_scaled = model(ratio_tensor)
        
        # Dénormalisation séparée
        gap_pred = gap_scaler.inverse_transform(
            prediction_scaled[0, 0].cpu().numpy().reshape(-1, 1)
        )[0, 0]
        L_ecran_pred = L_ecran_scaler.inverse_transform(
            prediction_scaled[0, 1].cpu().numpy().reshape(-1, 1)
        )[0, 0]
    
    return gap_pred, L_ecran_pred

def test_improved_model(n_samples=50):
    """Teste le modèle amélioré sur des échantillons de test."""
    logger.info(f"🧪 TEST DU MODÈLE AMÉLIORÉ SUR {n_samples} ÉCHANTILLONS")
    logger.info("="*70)
    
    # Charger le modèle
    model, input_scaler, gap_scaler, L_ecran_scaler, device = load_improved_model()
    
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
            gap_pred, L_ecran_pred = predict_improved(
                model, input_scaler, gap_scaler, L_ecran_scaler, device, ratio
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
            
            # Afficher quelques exemples détaillés
            if i < 10:
                logger.info(f"\n📋 Échantillon {i+1}/{len(sample_indices)}: {filename}")
                logger.info(f"   Gap:     Réel={true_gap:.4f}µm, Prédit={gap_pred:.4f}µm, Erreur={gap_error:.4f}µm")
                logger.info(f"   L_écran: Réel={true_L_ecran:.1f}µm, Prédit={L_ecran_pred:.1f}µm, Erreur={L_ecran_error:.1f}µm")
                
                # Indicateur de qualité
                gap_quality = "🟢" if gap_error <= 0.01 else "🟡" if gap_error <= 0.05 else "🔴"
                L_ecran_quality = "🟢" if L_ecran_error <= 0.5 else "🟡" if L_ecran_error <= 1.0 else "🔴"
                logger.info(f"   Qualité: Gap {gap_quality}, L_écran {L_ecran_quality}")
            
        except Exception as e:
            logger.error(f"❌ Erreur avec {filename}: {e}")
            continue
    
    # Calculer les statistiques globales
    if results:
        gap_errors = [r['gap_error'] for r in results]
        L_ecran_errors = [r['L_ecran_error'] for r in results]
        
        gap_mae = np.mean(gap_errors)
        gap_std = np.std(gap_errors)
        gap_median = np.median(gap_errors)
        L_ecran_mae = np.mean(L_ecran_errors)
        L_ecran_std = np.std(L_ecran_errors)
        L_ecran_median = np.median(L_ecran_errors)
        
        # Calculer R²
        true_gaps = [r['true_gap'] for r in results]
        pred_gaps = [r['pred_gap'] for r in results]
        true_L_ecrans = [r['true_L_ecran'] for r in results]
        pred_L_ecrans = [r['pred_L_ecran'] for r in results]
        
        gap_r2 = r2_score(true_gaps, pred_gaps)
        L_ecran_r2 = r2_score(true_L_ecrans, pred_L_ecrans)
        
        logger.info(f"\n📊 STATISTIQUES GLOBALES ({len(results)} échantillons)")
        logger.info("="*60)
        logger.info(f"GAP:")
        logger.info(f"   MAE: {gap_mae:.4f} ± {gap_std:.4f} µm")
        logger.info(f"   Médiane: {gap_median:.4f} µm")
        logger.info(f"   R²: {gap_r2:.3f}")
        logger.info(f"   Min erreur: {min(gap_errors):.4f} µm")
        logger.info(f"   Max erreur: {max(gap_errors):.4f} µm")
        
        logger.info(f"\nL_ÉCRAN:")
        logger.info(f"   MAE: {L_ecran_mae:.1f} ± {L_ecran_std:.1f} µm")
        logger.info(f"   Médiane: {L_ecran_median:.1f} µm")
        logger.info(f"   R²: {L_ecran_r2:.3f}")
        logger.info(f"   Min erreur: {min(L_ecran_errors):.1f} µm")
        logger.info(f"   Max erreur: {max(L_ecran_errors):.1f} µm")
        
        # Évaluer la précision avec différentes tolérances
        tolerances_gap = [0.005, 0.01, 0.02, 0.05]
        tolerances_L_ecran = [0.2, 0.5, 1.0, 2.0]
        
        logger.info(f"\n🎯 PRÉCISION PAR TOLÉRANCE:")
        logger.info("Gap:")
        for tol in tolerances_gap:
            good = sum(1 for e in gap_errors if e <= tol)
            accuracy = good / len(results) * 100
            logger.info(f"   ±{tol:.3f}µm: {good}/{len(results)} = {accuracy:.1f}%")
        
        logger.info("L_écran:")
        for tol in tolerances_L_ecran:
            good = sum(1 for e in L_ecran_errors if e <= tol)
            accuracy = good / len(results) * 100
            logger.info(f"   ±{tol:.1f}µm: {good}/{len(results)} = {accuracy:.1f}%")
        
        # Analyse par plage de gap
        logger.info(f"\n📈 ANALYSE PAR PLAGE DE GAP:")
        gap_ranges = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
        
        for gap_min, gap_max in gap_ranges:
            range_results = [r for r in results if gap_min <= r['true_gap'] < gap_max]
            if range_results:
                range_gap_errors = [r['gap_error'] for r in range_results]
                range_mae = np.mean(range_gap_errors)
                range_count = len(range_results)
                logger.info(f"   {gap_min:.1f}-{gap_max:.1f}µm: {range_count} échantillons, MAE={range_mae:.4f}µm")
        
        # Évaluation finale
        logger.info(f"\n🏆 ÉVALUATION FINALE:")
        
        if gap_r2 > 0.8 and L_ecran_r2 > 0.95:
            logger.info(f"   🎉 MODÈLE EXCELLENT !")
        elif gap_r2 > 0.6 and L_ecran_r2 > 0.9:
            logger.info(f"   ✅ MODÈLE DE BONNE QUALITÉ")
        elif gap_r2 > 0.4 and L_ecran_r2 > 0.8:
            logger.info(f"   ⚠️ MODÈLE ACCEPTABLE")
        else:
            logger.info(f"   ❌ MODÈLE À AMÉLIORER")
        
        # Recommandations
        if gap_mae > 0.05:
            logger.info(f"   💡 Recommandation: Augmenter encore le poids du gap dans la loss")
        if gap_r2 < 0.7:
            logger.info(f"   💡 Recommandation: Ajouter plus de données d'entraînement")
        if L_ecran_r2 > 0.98:
            logger.info(f"   ✨ L_écran: Prédiction quasi-parfaite !")
    
    return results

def main():
    """Fonction principale de test."""
    results = test_improved_model(n_samples=50)
    logger.info(f"\n✅ TEST FINAL TERMINÉ - {len(results)} échantillons testés")

if __name__ == "__main__":
    main()
