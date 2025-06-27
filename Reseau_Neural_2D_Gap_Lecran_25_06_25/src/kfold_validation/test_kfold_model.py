#!/usr/bin/env python3
"""
Test du modèle K-Fold sur le dataset de test

Auteur: Oussama GUELFAA
Date: 25/06/2025

Teste le modèle entraîné avec K-Fold Cross-Validation sur le dataset de test complet.
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import torch
import joblib
from Train_KFold import ImprovedDualParameterNet
import logging
from sklearn.metrics import r2_score, mean_absolute_error
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_kfold_model():
    """Charge le modèle K-Fold entraîné."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImprovedDualParameterNet(input_size=600).to(device)
    checkpoint = torch.load("../../models/dual_parameter_model_kfold.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Charger les scalers K-Fold
    input_scaler = joblib.load("../../models/input_scaler_kfold.pkl")
    gap_scaler = joblib.load("../../models/gap_scaler_kfold.pkl")
    L_ecran_scaler = joblib.load("../../models/L_ecran_scaler_kfold.pkl")
    
    fold_number = checkpoint.get('fold_number', 'Unknown')
    logger.info(f"✅ Modèle K-Fold chargé (Meilleur fold: {fold_number})")
    
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

def predict_kfold(model, input_scaler, gap_scaler, L_ecran_scaler, device, ratio):
    """Prédiction avec le modèle K-Fold."""
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

def test_kfold_model_complete():
    """Teste le modèle K-Fold sur l'ensemble du dataset de test."""
    logger.info(f"🧪 TEST COMPLET DU MODÈLE K-FOLD SUR DATASET_2D_TEST")
    logger.info("="*70)
    
    # Charger le modèle
    model, input_scaler, gap_scaler, L_ecran_scaler, device = load_kfold_model()
    
    # Charger le dataset de test complet
    test_dir = Path("../../../data_generation/dataset_2D_Test")
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
            gap_pred, L_ecran_pred = predict_kfold(
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
        
        logger.info(f"\n📊 STATISTIQUES GLOBALES MODÈLE K-FOLD ({n_valid} échantillons)")
        logger.info("="*70)
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
        
        # Ajouter des colonnes d'analyse
        valid_results['Gap_precision_001'] = (valid_results['erreur_Gap'] <= 0.01).astype(int)
        valid_results['Gap_precision_002'] = (valid_results['erreur_Gap'] <= 0.02).astype(int)
        valid_results['Lecran_precision_05'] = (valid_results['erreur_Lecran'] <= 0.5).astype(int)
        
        # Arrondir les valeurs
        valid_results['Gap_reel'] = valid_results['Gap_reel'].round(4)
        valid_results['Gap_predit'] = valid_results['Gap_predit'].round(4)
        valid_results['Lecran_reel'] = valid_results['Lecran_reel'].round(1)
        valid_results['Lecran_predit'] = valid_results['Lecran_predit'].round(1)
        valid_results['erreur_Gap'] = valid_results['erreur_Gap'].round(4)
        valid_results['erreur_Lecran'] = valid_results['erreur_Lecran'].round(1)
        
        # Trier par erreur Gap croissante
        valid_results = valid_results.sort_values('erreur_Gap')
        
        # Sauvegarder dans un fichier CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"../../results/test_complet_modele_kfold_{timestamp}.csv"
        
        # Créer le dossier results s'il n'existe pas
        Path("../../results").mkdir(exist_ok=True)
        
        # Réorganiser les colonnes
        columns_order = [
            'filename',
            'Gap_reel', 'Gap_predit', 'erreur_Gap',
            'Lecran_reel', 'Lecran_predit', 'erreur_Lecran',
            'Gap_precision_001', 'Gap_precision_002', 'Lecran_precision_05'
        ]
        
        df_final = valid_results[columns_order].copy()
        
        # Sauvegarder le CSV
        df_final.to_csv(csv_filename, index=False)
        
        logger.info(f"\n💾 RÉSULTATS SAUVEGARDÉS:")
        logger.info(f"   Fichier: {csv_filename}")
        logger.info(f"   Format: CSV avec {len(df_final)} lignes")
        
        # Afficher un aperçu des meilleures prédictions
        logger.info(f"\n📋 APERÇU (10 meilleures prédictions Gap):")
        for i, (_, row) in enumerate(df_final.head(10).iterrows()):
            logger.info(f"   {i+1:2d}. {row['filename']}: "
                       f"Gap {row['Gap_reel']:.4f}→{row['Gap_predit']:.4f} "
                       f"(±{row['erreur_Gap']:.4f}µm), "
                       f"L_écran {row['Lecran_reel']:.1f}→{row['Lecran_predit']:.1f} "
                       f"(±{row['erreur_Lecran']:.1f}µm)")
        
        # Évaluation finale
        logger.info(f"\n🏆 ÉVALUATION FINALE MODÈLE K-FOLD:")
        
        if gap_r2 > 0.7 and L_ecran_r2 > 0.95:
            logger.info(f"   🎉 MODÈLE EXCELLENT ! Généralisation robuste validée.")
        elif gap_r2 > 0.5 and L_ecran_r2 > 0.9:
            logger.info(f"   ✅ MODÈLE DE BONNE QUALITÉ. Généralisation satisfaisante.")
        elif gap_r2 > 0.3 and L_ecran_r2 > 0.8:
            logger.info(f"   ⚠️ MODÈLE ACCEPTABLE. Amélioration possible.")
        else:
            logger.info(f"   ❌ MODÈLE À AMÉLIORER.")
        
        # Comparaison avec les résultats K-Fold
        logger.info(f"\n📊 COMPARAISON AVEC VALIDATION K-FOLD:")
        logger.info(f"   Gap R² K-Fold: 0.556 ± 0.097 → Test: {gap_r2:.3f}")
        logger.info(f"   Gap MAE K-Fold: 0.0441 ± 0.0058µm → Test: {gap_mae:.4f}µm")
        logger.info(f"   L_écran R² K-Fold: 0.989 ± 0.003 → Test: {L_ecran_r2:.3f}")
        
        if abs(gap_r2 - 0.556) < 0.1:
            logger.info(f"   ✅ Cohérence excellente entre K-Fold et test")
        elif abs(gap_r2 - 0.556) < 0.2:
            logger.info(f"   ⚠️ Cohérence acceptable entre K-Fold et test")
        else:
            logger.info(f"   ❌ Écart important entre K-Fold et test - Possible overfitting")
    
    return valid_results, csv_filename

def main():
    """Fonction principale de test."""
    logger.info("🚀 DÉBUT DU TEST COMPLET MODÈLE K-FOLD")
    
    start_time = datetime.now()
    
    # Exécuter le test complet
    df_results, csv_filename = test_kfold_model_complete()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"\n✅ TEST COMPLET TERMINÉ")
    logger.info(f"   Durée: {duration}")
    logger.info(f"   Fichier CSV: {Path(csv_filename).name}")
    logger.info(f"   Échantillons traités: {len(df_results)}")
    
    # Afficher le chemin absolu du fichier
    csv_path = Path(csv_filename).resolve()
    logger.info(f"   Chemin complet: {csv_path}")

if __name__ == "__main__":
    main()
