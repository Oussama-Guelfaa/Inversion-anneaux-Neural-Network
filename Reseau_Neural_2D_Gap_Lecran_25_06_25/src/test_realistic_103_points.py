#!/usr/bin/env python3
"""
Test avec des données réalistes de 103 points

Auteur: Oussama GUELFAA
Date: 25/06/2025

Génère des données réalistes de 103 points basées sur les profils du dataset
d'entraînement et teste les prédictions.
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import logging
import subprocess
import sys

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_real_profile_600(dataset_path, filename):
    """Charge un profil réel de 600 points du dataset."""
    mat_file_path = Path(dataset_path) / filename
    
    try:
        data = loadmat(str(mat_file_path))
        
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
        
        if len(ratio) > 600:
            ratio = ratio[:600]
        elif len(ratio) < 600:
            ratio = np.pad(ratio, (0, 600 - len(ratio)), 'edge')
        
        return ratio
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {filename}: {e}")
        return None

def simulate_103_points_from_600(profile_600, add_noise=True):
    """Simule des données de 103 points à partir d'un profil de 600 points."""
    indices = np.linspace(0, len(profile_600) - 1, 103).astype(int)
    profile_103 = profile_600[indices]
    
    if add_noise:
        noise_level = 0.015  # 1.5% de bruit
        noise = np.random.normal(0, noise_level * np.std(profile_103), 103)
        profile_103 = profile_103 + noise
        profile_103 = np.maximum(profile_103, 0.1)
    
    return profile_103

def create_test_cases():
    """Crée plusieurs cas de test avec des données réalistes."""
    logger.info("🧪 CRÉATION DE CAS DE TEST RÉALISTES")
    logger.info("="*50)
    
    dataset_path = "../../data_generation/dataset_2D_Train_Augmented"
    
    labels_path = Path(dataset_path) / "labels.csv"
    labels_df = pd.read_csv(labels_path)
    
    test_cases = [
        labels_df[(labels_df['gap_um'] <= 0.05) & (labels_df['L_um'] <= 5.0)].iloc[0],
        labels_df[(labels_df['gap_um'] <= 0.05) & (labels_df['L_um'] >= 7.0)].iloc[0],
        labels_df[(labels_df['gap_um'] >= 0.25) & (labels_df['L_um'] <= 5.0)].iloc[0],
        labels_df[(labels_df['gap_um'] >= 0.25) & (labels_df['L_um'] >= 7.0)].iloc[0]
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        logger.info(f"\n📋 Cas de test {i+1}:")
        logger.info(f"   Fichier: {case['filename']}")
        logger.info(f"   Gap réel: {case['gap_um']:.4f} µm")
        logger.info(f"   L_écran réel: {case['L_um']:.1f} µm")
        
        profile_600 = load_real_profile_600(dataset_path, case['filename'])
        
        if profile_600 is not None:
            profile_103 = simulate_103_points_from_600(profile_600, add_noise=True)
            
            test_file = f"test_case_{i+1}_103pts.txt"
            np.savetxt(test_file, profile_103, fmt='%.6f')
            
            logger.info(f"   ✅ Données 103 points générées: {test_file}")
            logger.info(f"   Statistiques: Min={np.min(profile_103):.3f}, "
                       f"Max={np.max(profile_103):.3f}, "
                       f"Moyenne={np.mean(profile_103):.3f}")
            
            results.append({
                'case_number': i + 1,
                'filename': case['filename'],
                'true_gap': case['gap_um'],
                'true_L_ecran': case['L_um'],
                'test_file': test_file,
                'profile_103': profile_103
            })
        else:
            logger.error(f"   ❌ Impossible de charger {case['filename']}")
    
    return results

def test_predictions_on_cases(test_cases):
    """Teste les prédictions sur tous les cas de test."""
    logger.info(f"\n🔮 TEST DES PRÉDICTIONS SUR LES CAS RÉALISTES")
    logger.info("="*60)
    
    methods = ['linear', 'cubic', 'spline', 'padding', 'fourier']
    all_results = []
    
    for case in test_cases:
        logger.info(f"\n📋 Test du cas {case['case_number']}: {case['filename']}")
        logger.info(f"   Valeurs réelles: Gap={case['true_gap']:.4f}µm, L_écran={case['true_L_ecran']:.1f}µm")
        
        case_results = {
            'case': case['case_number'],
            'true_gap': case['true_gap'],
            'true_L_ecran': case['true_L_ecran'],
            'predictions': {}
        }
        
        for method in methods:
            try:
                cmd = [
                    sys.executable, 'predict_103_points.py',
                    '--method', method,
                    '--data', case['test_file']
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    output_lines = result.stdout.split('\n')
                    gap_pred = None
                    L_ecran_pred = None
                    
                    for line in output_lines:
                        if 'Gap prédit:' in line:
                            gap_pred = float(line.split(':')[1].strip().split()[0])
                        elif 'L_écran prédit:' in line:
                            L_ecran_pred = float(line.split(':')[1].strip().split()[0])
                    
                    if gap_pred is not None and L_ecran_pred is not None:
                        gap_error = abs(gap_pred - case['true_gap'])
                        L_ecran_error = abs(L_ecran_pred - case['true_L_ecran'])
                        
                        case_results['predictions'][method] = {
                            'gap_pred': gap_pred,
                            'L_ecran_pred': L_ecran_pred,
                            'gap_error': gap_error,
                            'L_ecran_error': L_ecran_error
                        }
                        
                        logger.info(f"   {method.upper():8}: Gap={gap_pred:7.4f}µm (±{gap_error:.4f}), "
                                   f"L_écran={L_ecran_pred:5.1f}µm (±{L_ecran_error:.1f})")
                    else:
                        logger.error(f"   {method.upper():8}: Impossible de parser les résultats")
                else:
                    logger.error(f"   {method.upper():8}: Erreur d'exécution")
                    
            except Exception as e:
                logger.error(f"   {method.upper():8}: Exception - {e}")
        
        all_results.append(case_results)
    
    return all_results

def analyze_results(all_results):
    """Analyse les résultats de tous les tests."""
    logger.info(f"\n📊 ANALYSE DES RÉSULTATS")
    logger.info("="*50)
    
    methods = ['linear', 'cubic', 'spline', 'padding', 'fourier']
    method_stats = {}
    
    for method in methods:
        gap_errors = []
        L_ecran_errors = []
        
        for case_result in all_results:
            if method in case_result['predictions']:
                pred = case_result['predictions'][method]
                gap_errors.append(pred['gap_error'])
                L_ecran_errors.append(pred['L_ecran_error'])
        
        if gap_errors:
            method_stats[method] = {
                'gap_mae': np.mean(gap_errors),
                'gap_std': np.std(gap_errors),
                'L_ecran_mae': np.mean(L_ecran_errors),
                'L_ecran_std': np.std(L_ecran_errors),
                'n_cases': len(gap_errors)
            }
    
    logger.info(f"\n📈 STATISTIQUES PAR MÉTHODE:")
    logger.info(f"{'Méthode':10} {'Gap MAE':>10} {'Gap Std':>10} {'L_écran MAE':>12} {'L_écran Std':>12} {'N':>3}")
    logger.info("-" * 70)
    
    for method, stats in method_stats.items():
        logger.info(f"{method.upper():10} {stats['gap_mae']:10.4f} {stats['gap_std']:10.4f} "
                   f"{stats['L_ecran_mae']:12.1f} {stats['L_ecran_std']:12.1f} {stats['n_cases']:3d}")
    
    if method_stats:
        best_method_gap = min(method_stats.keys(), key=lambda m: method_stats[m]['gap_mae'])
        best_method_L_ecran = min(method_stats.keys(), key=lambda m: method_stats[m]['L_ecran_mae'])
        
        logger.info(f"\n🏆 MEILLEURES MÉTHODES:")
        logger.info(f"   Gap: {best_method_gap.upper()} (MAE = {method_stats[best_method_gap]['gap_mae']:.4f}µm)")
        logger.info(f"   L_écran: {best_method_L_ecran.upper()} (MAE = {method_stats[best_method_L_ecran]['L_ecran_mae']:.1f}µm)")
        
        if best_method_gap == best_method_L_ecran:
            logger.info(f"\n💡 RECOMMANDATION: Utiliser la méthode {best_method_gap.upper()}")
        else:
            logger.info(f"\n💡 RECOMMANDATION: Tester {best_method_gap.upper()} et {best_method_L_ecran.upper()}")

def main():
    """Fonction principale."""
    logger.info("🚀 TEST COMPLET AVEC DONNÉES RÉALISTES DE 103 POINTS")
    logger.info("="*70)
    
    test_cases = create_test_cases()
    
    if not test_cases:
        logger.error("❌ Aucun cas de test créé")
        return
    
    all_results = test_predictions_on_cases(test_cases)
    analyze_results(all_results)
    
    for case in test_cases:
        try:
            Path(case['test_file']).unlink()
        except:
            pass
    
    logger.info(f"\n✅ TEST COMPLET TERMINÉ")
    logger.info(f"   {len(test_cases)} cas de test traités")
    logger.info(f"   5 méthodes d'adaptation comparées")
    logger.info(f"   Recommandations générées")

if __name__ == "__main__":
    main()
