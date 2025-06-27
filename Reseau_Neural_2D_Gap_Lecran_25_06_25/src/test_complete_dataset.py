#!/usr/bin/env python3
"""
Test complet du modèle amélioré sur tout le dataset_2D_Test

Auteur: Oussama GUELFAA
Date: 25/06/2025

Teste le modèle amélioré sur tous les échantillons du dataset de test
et sauvegarde les résultats dans un fichier CSV détaillé.
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
from datetime import datetime

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

def test_complete_dataset():
    """Teste le modèle sur l'ensemble du dataset de test."""
    logger.info(f"🧪 TEST COMPLET DU MODÈLE AMÉLIORÉ SUR DATASET_2D_TEST")
    logger.info("="*70)
    
    # Charger le modèle
    model, input_scaler, gap_scaler, L_ecran_scaler, device = load_improved_model()
    
    # Charger le dataset de test complet
    test_dir = Path("../../data_generation/dataset_2D_Test")
    labels_df = pd.read_csv(test_dir / "labels.csv")
    
    total_samples = len(labels_df)
    logger.info(f"📊 Dataset de test complet: {total_samples} échantillons")
    
    # Préparer la liste des résultats
    results = []
    
    # Traiter tous les échantillons
    for i, (_, row) in enumerate(labels_df.iterrows()):
        filename = row['filename']
        true_gap = row['gap_um']
        true_L_ecran = row['L_um']
        
        # Affichage de progression
        if (i + 1) % 100 == 0:
            logger.info(f"   Progression: {i+1}/{total_samples} échantillons traités...")
        
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
            
            # Ajouter aux résultats
            results.append({
                'filename': filename,
                'Gap_reel': true_gap,
                'Gap_predit': gap_pred,
                'Lecran_reel': true_L_ecran,
                'Lecran_predit': L_ecran_pred,
                'erreur_Gap': gap_error,
                'erreur_Lecran': L_ecran_error
            })
            
        except Exception as e:
            logger.error(f"❌ Erreur avec {filename}: {e}")
            # Ajouter une ligne avec des NaN pour garder la trace
            results.append({
                'filename': filename,
                'Gap_reel': true_gap,
                'Gap_predit': np.nan,
                'Lecran_reel': true_L_ecran,
                'Lecran_predit': np.nan,
                'erreur_Gap': np.nan,
                'erreur_Lecran': np.nan
            })
            continue
    
    logger.info(f"✅ Traitement terminé: {len(results)} échantillons")
    
    # Créer le DataFrame
    df_results = pd.DataFrame(results)
    
    # Filtrer les résultats valides pour les statistiques
    valid_results = df_results.dropna()
    n_valid = len(valid_results)
    n_errors = len(results) - n_valid
    
    logger.info(f"📊 Résultats valides: {n_valid}/{len(results)} ({n_errors} erreurs)")
    
    if n_valid > 0:
        # Calculer les statistiques globales
        gap_mae = valid_results['erreur_Gap'].mean()
        gap_std = valid_results['erreur_Gap'].std()
        gap_median = valid_results['erreur_Gap'].median()
        gap_min = valid_results['erreur_Gap'].min()
        gap_max = valid_results['erreur_Gap'].max()
        
        L_ecran_mae = valid_results['erreur_Lecran'].mean()
        L_ecran_std = valid_results['erreur_Lecran'].std()
        L_ecran_median = valid_results['erreur_Lecran'].median()
        L_ecran_min = valid_results['erreur_Lecran'].min()
        L_ecran_max = valid_results['erreur_Lecran'].max()
        
        # Calculer R²
        gap_r2 = r2_score(valid_results['Gap_reel'], valid_results['Gap_predit'])
        L_ecran_r2 = r2_score(valid_results['Lecran_reel'], valid_results['Lecran_predit'])
        
        logger.info(f"\n📊 STATISTIQUES GLOBALES ({n_valid} échantillons)")
        logger.info("="*60)
        logger.info(f"GAP:")
        logger.info(f"   MAE: {gap_mae:.4f} ± {gap_std:.4f} µm")
        logger.info(f"   Médiane: {gap_median:.4f} µm")
        logger.info(f"   Min-Max: {gap_min:.4f} - {gap_max:.4f} µm")
        logger.info(f"   R²: {gap_r2:.3f}")
        
        logger.info(f"\nL_ÉCRAN:")
        logger.info(f"   MAE: {L_ecran_mae:.1f} ± {L_ecran_std:.1f} µm")
        logger.info(f"   Médiane: {L_ecran_median:.1f} µm")
        logger.info(f"   Min-Max: {L_ecran_min:.1f} - {L_ecran_max:.1f} µm")
        logger.info(f"   R²: {L_ecran_r2:.3f}")
        
        # Précision par tolérance
        tolerances_gap = [0.005, 0.01, 0.02, 0.05]
        tolerances_L_ecran = [0.2, 0.5, 1.0]
        
        logger.info(f"\n🎯 PRÉCISION PAR TOLÉRANCE:")
        logger.info("Gap:")
        for tol in tolerances_gap:
            good = (valid_results['erreur_Gap'] <= tol).sum()
            accuracy = good / n_valid * 100
            logger.info(f"   ±{tol:.3f}µm: {good}/{n_valid} = {accuracy:.1f}%")
        
        logger.info("L_écran:")
        for tol in tolerances_L_ecran:
            good = (valid_results['erreur_Lecran'] <= tol).sum()
            accuracy = good / n_valid * 100
            logger.info(f"   ±{tol:.1f}µm: {good}/{n_valid} = {accuracy:.1f}%")
        
        # Ajouter les statistiques au DataFrame
        stats_row = {
            'filename': 'STATISTIQUES_GLOBALES',
            'Gap_reel': f'MAE={gap_mae:.4f}',
            'Gap_predit': f'R²={gap_r2:.3f}',
            'Lecran_reel': f'MAE={L_ecran_mae:.1f}',
            'Lecran_predit': f'R²={L_ecran_r2:.3f}',
            'erreur_Gap': f'±{gap_std:.4f}',
            'erreur_Lecran': f'±{L_ecran_std:.1f}'
        }
        
        # Ajouter la ligne de statistiques
        df_results = pd.concat([df_results, pd.DataFrame([stats_row])], ignore_index=True)
    
    # Sauvegarder dans un fichier CSV avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"../results/test_complet_modele_ameliore_{timestamp}.csv"
    
    # Créer le dossier results s'il n'existe pas
    Path("../results").mkdir(exist_ok=True)
    
    # Sauvegarder le CSV
    df_results.to_csv(csv_filename, index=False, float_format='%.6f')
    
    logger.info(f"\n💾 RÉSULTATS SAUVEGARDÉS:")
    logger.info(f"   Fichier: {csv_filename}")
    logger.info(f"   Format: CSV avec {len(df_results)} lignes")
    logger.info(f"   Colonnes: {list(df_results.columns)}")
    
    # Afficher un aperçu des premières lignes
    logger.info(f"\n📋 APERÇU DES RÉSULTATS (5 premières lignes):")
    for i, row in df_results.head(5).iterrows():
        if row['filename'] != 'STATISTIQUES_GLOBALES':
            logger.info(f"   {row['filename']}: Gap {row['Gap_reel']:.4f}→{row['Gap_predit']:.4f} "
                       f"(±{row['erreur_Gap']:.4f}), L_écran {row['Lecran_reel']:.1f}→{row['Lecran_predit']:.1f} "
                       f"(±{row['erreur_Lecran']:.1f})")
    
    return df_results, csv_filename

def main():
    """Fonction principale."""
    logger.info("🚀 DÉBUT DU TEST COMPLET")
    
    start_time = datetime.now()
    
    # Exécuter le test complet
    df_results, csv_filename = test_complete_dataset()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"\n✅ TEST COMPLET TERMINÉ")
    logger.info(f"   Durée: {duration}")
    logger.info(f"   Fichier CSV: {csv_filename}")
    logger.info(f"   Échantillons traités: {len(df_results)-1}")  # -1 pour la ligne de stats
    
    # Afficher le chemin absolu du fichier
    csv_path = Path(csv_filename).resolve()
    logger.info(f"   Chemin complet: {csv_path}")

if __name__ == "__main__":
    main()
