#!/usr/bin/env python3
"""
Test du modèle amélioré vs modèle original

Auteur: Oussama GUELFAA
Date: 25/06/2025

Compare les performances du modèle amélioré avec le modèle original.
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import torch
import joblib
from Train_Improved import ImprovedDualParameterNet
from Train import DualParameterNet
import logging
from sklearn.metrics import r2_score, mean_absolute_error

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_original_model():
    """Charge le modèle original."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DualParameterNet(input_size=600).to(device)
    checkpoint = torch.load("../models/dual_parameter_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Charger les scalers
    input_scaler = joblib.load("../models/input_scaler.pkl")
    output_scaler = joblib.load("../models/output_scaler.pkl")
    
    return model, input_scaler, output_scaler, device

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

def predict_original(model, input_scaler, output_scaler, device, ratio):
    """Prédiction avec le modèle original."""
    ratio_scaled = input_scaler.transform(ratio.reshape(1, -1))
    
    with torch.no_grad():
        ratio_tensor = torch.FloatTensor(ratio_scaled).to(device)
        prediction_scaled = model(ratio_tensor)
        prediction = output_scaler.inverse_transform(prediction_scaled.cpu().numpy())
    
    return prediction[0, 0], prediction[0, 1]

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

def compare_models(n_samples=30):
    """Compare les deux modèles sur des échantillons de test."""
    logger.info(f"🔬 COMPARAISON MODÈLE ORIGINAL vs AMÉLIORÉ")
    logger.info("="*60)
    
    # Charger les modèles
    logger.info("📥 Chargement des modèles...")
    original_model, orig_input_scaler, orig_output_scaler, device = load_original_model()
    improved_model, imp_input_scaler, imp_gap_scaler, imp_L_ecran_scaler, device = load_improved_model()
    
    # Charger le dataset de test
    test_dir = Path("../../data_generation/dataset_2D_Test")
    labels_df = pd.read_csv(test_dir / "labels.csv")
    
    # Sélectionner des échantillons aléatoires
    sample_indices = np.random.choice(len(labels_df), min(n_samples, len(labels_df)), replace=False)
    
    original_results = []
    improved_results = []
    
    logger.info(f"🧪 Test sur {len(sample_indices)} échantillons...")
    
    for i, idx in enumerate(sample_indices):
        sample = labels_df.iloc[idx]
        filename = sample['filename']
        true_gap = sample['gap_um']
        true_L_ecran = sample['L_um']
        
        try:
            # Charger le profil
            ratio = load_test_sample(test_dir, filename)
            
            # Prédictions
            orig_gap, orig_L_ecran = predict_original(
                original_model, orig_input_scaler, orig_output_scaler, device, ratio
            )
            imp_gap, imp_L_ecran = predict_improved(
                improved_model, imp_input_scaler, imp_gap_scaler, imp_L_ecran_scaler, device, ratio
            )
            
            # Erreurs
            orig_gap_error = abs(orig_gap - true_gap)
            orig_L_ecran_error = abs(orig_L_ecran - true_L_ecran)
            imp_gap_error = abs(imp_gap - true_gap)
            imp_L_ecran_error = abs(imp_L_ecran - true_L_ecran)
            
            # Stocker les résultats
            original_results.append({
                'true_gap': true_gap, 'pred_gap': orig_gap, 'gap_error': orig_gap_error,
                'true_L_ecran': true_L_ecran, 'pred_L_ecran': orig_L_ecran, 'L_ecran_error': orig_L_ecran_error
            })
            
            improved_results.append({
                'true_gap': true_gap, 'pred_gap': imp_gap, 'gap_error': imp_gap_error,
                'true_L_ecran': true_L_ecran, 'pred_L_ecran': imp_L_ecran, 'L_ecran_error': imp_L_ecran_error
            })
            
            # Affichage détaillé pour quelques échantillons
            if i < 5:
                logger.info(f"\n📋 Échantillon {i+1}: {filename}")
                logger.info(f"   Gap réel: {true_gap:.4f}µm")
                logger.info(f"   Original:  {orig_gap:.4f}µm (erreur: {orig_gap_error:.4f}µm)")
                logger.info(f"   Amélioré:  {imp_gap:.4f}µm (erreur: {imp_gap_error:.4f}µm)")
                logger.info(f"   L_écran réel: {true_L_ecran:.1f}µm")
                logger.info(f"   Original:  {orig_L_ecran:.1f}µm (erreur: {orig_L_ecran_error:.1f}µm)")
                logger.info(f"   Amélioré:  {imp_L_ecran:.1f}µm (erreur: {imp_L_ecran_error:.1f}µm)")
                
                improvement = "🟢" if imp_gap_error < orig_gap_error else "🔴"
                logger.info(f"   Gap: {improvement} {'Amélioré' if imp_gap_error < orig_gap_error else 'Dégradé'}")
            
        except Exception as e:
            logger.error(f"❌ Erreur avec {filename}: {e}")
            continue
    
    # Calculer les statistiques
    if original_results and improved_results:
        logger.info(f"\n📊 STATISTIQUES COMPARATIVES")
        logger.info("="*50)
        
        # Métriques Gap
        orig_gap_errors = [r['gap_error'] for r in original_results]
        imp_gap_errors = [r['gap_error'] for r in improved_results]
        orig_gap_mae = np.mean(orig_gap_errors)
        imp_gap_mae = np.mean(imp_gap_errors)
        
        orig_true_gaps = [r['true_gap'] for r in original_results]
        orig_pred_gaps = [r['pred_gap'] for r in original_results]
        imp_true_gaps = [r['true_gap'] for r in improved_results]
        imp_pred_gaps = [r['pred_gap'] for r in improved_results]
        
        orig_gap_r2 = r2_score(orig_true_gaps, orig_pred_gaps)
        imp_gap_r2 = r2_score(imp_true_gaps, imp_pred_gaps)
        
        # Métriques L_écran
        orig_L_ecran_errors = [r['L_ecran_error'] for r in original_results]
        imp_L_ecran_errors = [r['L_ecran_error'] for r in improved_results]
        orig_L_ecran_mae = np.mean(orig_L_ecran_errors)
        imp_L_ecran_mae = np.mean(imp_L_ecran_errors)
        
        orig_true_L_ecrans = [r['true_L_ecran'] for r in original_results]
        orig_pred_L_ecrans = [r['pred_L_ecran'] for r in original_results]
        imp_true_L_ecrans = [r['true_L_ecran'] for r in improved_results]
        imp_pred_L_ecrans = [r['pred_L_ecran'] for r in improved_results]
        
        orig_L_ecran_r2 = r2_score(orig_true_L_ecrans, orig_pred_L_ecrans)
        imp_L_ecran_r2 = r2_score(imp_true_L_ecrans, imp_pred_L_ecrans)
        
        # Affichage des résultats
        logger.info(f"GAP:")
        logger.info(f"   Original:  MAE={orig_gap_mae:.4f}µm, R²={orig_gap_r2:.3f}")
        logger.info(f"   Amélioré:  MAE={imp_gap_mae:.4f}µm, R²={imp_gap_r2:.3f}")
        gap_improvement = ((orig_gap_mae - imp_gap_mae) / orig_gap_mae) * 100
        logger.info(f"   Amélioration: {gap_improvement:+.1f}% MAE")
        
        logger.info(f"\nL_ÉCRAN:")
        logger.info(f"   Original:  MAE={orig_L_ecran_mae:.1f}µm, R²={orig_L_ecran_r2:.3f}")
        logger.info(f"   Amélioré:  MAE={imp_L_ecran_mae:.1f}µm, R²={imp_L_ecran_r2:.3f}")
        L_ecran_improvement = ((orig_L_ecran_mae - imp_L_ecran_mae) / orig_L_ecran_mae) * 100
        logger.info(f"   Amélioration: {L_ecran_improvement:+.1f}% MAE")
        
        # Précision avec tolérance
        gap_tolerance = 0.01
        L_ecran_tolerance = 0.5
        
        orig_gap_good = sum(1 for e in orig_gap_errors if e <= gap_tolerance)
        imp_gap_good = sum(1 for e in imp_gap_errors if e <= gap_tolerance)
        orig_L_ecran_good = sum(1 for e in orig_L_ecran_errors if e <= L_ecran_tolerance)
        imp_L_ecran_good = sum(1 for e in imp_L_ecran_errors if e <= L_ecran_tolerance)
        
        orig_gap_acc = orig_gap_good / len(original_results) * 100
        imp_gap_acc = imp_gap_good / len(improved_results) * 100
        orig_L_ecran_acc = orig_L_ecran_good / len(original_results) * 100
        imp_L_ecran_acc = imp_L_ecran_good / len(improved_results) * 100
        
        logger.info(f"\n🎯 PRÉCISION (Gap ±{gap_tolerance}µm, L_écran ±{L_ecran_tolerance}µm):")
        logger.info(f"   Gap Original:  {orig_gap_good}/{len(original_results)} = {orig_gap_acc:.1f}%")
        logger.info(f"   Gap Amélioré:  {imp_gap_good}/{len(improved_results)} = {imp_gap_acc:.1f}%")
        logger.info(f"   L_écran Original:  {orig_L_ecran_good}/{len(original_results)} = {orig_L_ecran_acc:.1f}%")
        logger.info(f"   L_écran Amélioré:  {imp_L_ecran_good}/{len(improved_results)} = {imp_L_ecran_acc:.1f}%")
        
        # Conclusion
        if imp_gap_mae < orig_gap_mae and imp_gap_r2 > orig_gap_r2:
            logger.info(f"\n🎉 MODÈLE AMÉLIORÉ SUPÉRIEUR !")
        elif imp_gap_mae < orig_gap_mae:
            logger.info(f"\n✅ AMÉLIORATION PARTIELLE")
        else:
            logger.info(f"\n⚠️ AMÉLIORATION LIMITÉE")
    
    return original_results, improved_results

def main():
    """Fonction principale de comparaison."""
    original_results, improved_results = compare_models(n_samples=30)
    logger.info(f"\n✅ COMPARAISON TERMINÉE")

if __name__ == "__main__":
    main()
