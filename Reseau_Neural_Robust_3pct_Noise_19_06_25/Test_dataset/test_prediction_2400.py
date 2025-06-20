#!/usr/bin/env python3
"""
Test de Prédiction sur 2400 Échantillons - Modèle Dual Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 19 - 06 - 2025

Script pour tester les prédictions du modèle entraîné
sur 2400 échantillons pour validation étendue.
"""

import sys
import os
sys.path.append('../src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd

from dual_parameter_model import DualParameterPredictor
from data_loader import DualDataLoader

def predict_sample(model, profile, data_loader, add_noise=False, noise_level=0.02):
    """
    Prédit les paramètres pour un échantillon.

    Args:
        model: Modèle entraîné
        profile (np.array): Profil d'intensité [600]
        data_loader: DataLoader pour normalisation
        add_noise (bool): Ajouter du bruit gaussien
        noise_level (float): Niveau de bruit (ex: 0.02 pour 2%)

    Returns:
        tuple: (gap_pred, L_ecran_pred)
    """
    # Ajouter du bruit gaussien si demandé
    if add_noise:
        noise = np.random.normal(0, noise_level * np.std(profile), profile.shape)
        profile_noisy = profile + noise
    else:
        profile_noisy = profile

    # Normaliser l'entrée
    profile_scaled = data_loader.input_scaler.transform(profile_noisy.reshape(1, -1))

    # Prédiction
    with torch.no_grad():
        input_tensor = torch.FloatTensor(profile_scaled)
        prediction_scaled = model(input_tensor).numpy()

    # Dénormaliser la sortie
    prediction_original = data_loader.inverse_transform_predictions(prediction_scaled)

    gap_pred = prediction_original[0, 0]
    L_ecran_pred = prediction_original[0, 1]

    return gap_pred, L_ecran_pred

def load_model_and_data():
    """
    Charge le modèle entraîné et les données de test pour 2400 échantillons.
    
    Returns:
        tuple: (model, X_test, y_test, data_loader)
    """
    print("🔄 Chargement du modèle et des données pour test 2400 échantillons...")
    
    # Charger le modèle
    model_path = "../models/dual_parameter_model.pth"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = DualParameterPredictor(input_size=600)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Modèle chargé: {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # Charger les données avec split modifié pour avoir ~2400 échantillons de test
    data_loader = DualDataLoader()
    
    # Configuration pour avoir ~2400 échantillons de test
    # Avec 17,080 échantillons total: 70/16/14 ≈ 11,956/2,733/2,391 échantillons
    config = {
        'data_processing': {
            'augmentation': {'enable': False},  # Pas d'augmentation pour le test
            'data_splits': {'train': 0.70, 'validation': 0.16, 'test': 0.14},  # Split pour ~2400 test
            'normalization': {'target_scaling': {'separate_scaling': True}}
        },
        'training': {'batch_size': 32}
    }
    
    pipeline_result = data_loader.get_complete_pipeline(config)
    
    # Récupérer les données de test non normalisées
    X_test = pipeline_result['raw_data'][2]  # Données de test non normalisées
    y_test = pipeline_result['raw_data'][5]  # Labels de test
    
    print(f"✅ Données chargées: {len(X_test)} échantillons de test")
    print(f"🎯 Objectif: ~2400 échantillons (obtenu: {len(X_test)})")
    
    return model, X_test, y_test, data_loader

def evaluate_model_2400(model, X_test, y_test, data_loader, test_with_noise=False, noise_level=0.02):
    """
    Évaluation complète du modèle sur 2400 échantillons.

    Args:
        model: Modèle entraîné
        X_test: Données de test
        y_test: Labels de test
        data_loader: DataLoader configuré
        test_with_noise (bool): Tester avec bruit gaussien
        noise_level (float): Niveau de bruit (ex: 0.02 pour 2%)

    Returns:
        dict: Résultats détaillés
    """
    noise_info = f" AVEC BRUIT {noise_level*100:.0f}%" if test_with_noise else ""
    print(f"\n🎯 ÉVALUATION SUR {len(X_test)} ÉCHANTILLONS{noise_info}")
    print("="*50)
    
    # Prédictions
    predictions_gap = []
    predictions_L_ecran = []
    true_gap = []
    true_L_ecran = []
    
    print("🔄 Calcul des prédictions...")
    for i in range(len(X_test)):
        if i % 300 == 0:
            print(f"   Progression: {i}/{len(X_test)} ({i/len(X_test)*100:.1f}%)")
        
        profile = X_test[i]
        gap_pred, L_ecran_pred = predict_sample(model, profile, data_loader,
                                               add_noise=test_with_noise,
                                               noise_level=noise_level)
        
        predictions_gap.append(gap_pred)
        predictions_L_ecran.append(L_ecran_pred)
        true_gap.append(y_test[i, 0])
        true_L_ecran.append(y_test[i, 1])
    
    print(f"   Progression: {len(X_test)}/{len(X_test)} (100.0%)")
    
    # Conversion en arrays
    predictions_gap = np.array(predictions_gap)
    predictions_L_ecran = np.array(predictions_L_ecran)
    true_gap = np.array(true_gap)
    true_L_ecran = np.array(true_L_ecran)
    
    # Calcul des métriques
    gap_r2 = r2_score(true_gap, predictions_gap)
    L_ecran_r2 = r2_score(true_L_ecran, predictions_L_ecran)
    combined_r2 = (gap_r2 + L_ecran_r2) / 2
    
    gap_mae = mean_absolute_error(true_gap, predictions_gap)
    L_ecran_mae = mean_absolute_error(true_L_ecran, predictions_L_ecran)
    
    gap_rmse = np.sqrt(mean_squared_error(true_gap, predictions_gap))
    L_ecran_rmse = np.sqrt(mean_squared_error(true_L_ecran, predictions_L_ecran))
    
    # Analyse de tolérance
    gap_tolerance = 0.01  # µm
    L_ecran_tolerance = 0.1  # µm
    
    gap_within_tolerance = np.sum(np.abs(predictions_gap - true_gap) <= gap_tolerance)
    L_ecran_within_tolerance = np.sum(np.abs(predictions_L_ecran - true_L_ecran) <= L_ecran_tolerance)
    
    gap_accuracy = gap_within_tolerance / len(X_test)
    L_ecran_accuracy = L_ecran_within_tolerance / len(X_test)
    
    # Affichage des résultats
    print(f"\n📊 RÉSULTATS DE L'ÉVALUATION SUR {len(X_test)} ÉCHANTILLONS")
    print("="*60)
    print(f"🎯 Métriques R²:")
    print(f"   Gap R²: {gap_r2:.4f} (99.{gap_r2*10000-9900:.0f}%)")
    print(f"   L_ecran R²: {L_ecran_r2:.4f} (98.{L_ecran_r2*10000-9800:.0f}%)")
    print(f"   Combined R²: {combined_r2:.4f} (99.{combined_r2*10000-9900:.0f}%)")
    
    print(f"\n📏 Erreurs moyennes:")
    print(f"   Gap MAE: {gap_mae:.4f} µm")
    print(f"   Gap RMSE: {gap_rmse:.4f} µm")
    print(f"   L_ecran MAE: {L_ecran_mae:.4f} µm")
    print(f"   L_ecran RMSE: {L_ecran_rmse:.4f} µm")
    
    print(f"\n🎯 Précision dans tolérance:")
    print(f"   Gap (±{gap_tolerance} µm): {gap_within_tolerance}/{len(X_test)} ({gap_accuracy:.1%})")
    print(f"   L_ecran (±{L_ecran_tolerance} µm): {L_ecran_within_tolerance}/{len(X_test)} ({L_ecran_accuracy:.1%})")
    
    # Statistiques supplémentaires
    gap_errors = np.abs(predictions_gap - true_gap)
    L_ecran_errors = np.abs(predictions_L_ecran - true_L_ecran)
    
    print(f"\n📈 Statistiques d'erreurs:")
    print(f"   Gap - Max: {gap_errors.max():.4f} µm, Min: {gap_errors.min():.4f} µm, Médiane: {np.median(gap_errors):.4f} µm")
    print(f"   L_ecran - Max: {L_ecran_errors.max():.4f} µm, Min: {L_ecran_errors.min():.4f} µm, Médiane: {np.median(L_ecran_errors):.4f} µm")
    
    # Retourner les résultats
    results = {
        'n_samples': len(X_test),
        'test_conditions': {
            'with_noise': test_with_noise,
            'noise_level': noise_level if test_with_noise else 0.0,
            'noise_percentage': f"{noise_level*100:.0f}%" if test_with_noise else "0%"
        },
        'metrics': {
            'gap_r2': gap_r2,
            'L_ecran_r2': L_ecran_r2,
            'combined_r2': combined_r2,
            'gap_mae': gap_mae,
            'L_ecran_mae': L_ecran_mae,
            'gap_rmse': gap_rmse,
            'L_ecran_rmse': L_ecran_rmse
        },
        'tolerance_analysis': {
            'gap_tolerance': gap_tolerance,
            'L_ecran_tolerance': L_ecran_tolerance,
            'gap_accuracy': gap_accuracy,
            'L_ecran_accuracy': L_ecran_accuracy,
            'gap_within_tolerance': int(gap_within_tolerance),
            'L_ecran_within_tolerance': int(L_ecran_within_tolerance)
        },
        'error_statistics': {
            'gap_max_error': float(gap_errors.max()),
            'gap_min_error': float(gap_errors.min()),
            'gap_median_error': float(np.median(gap_errors)),
            'L_ecran_max_error': float(L_ecran_errors.max()),
            'L_ecran_min_error': float(L_ecran_errors.min()),
            'L_ecran_median_error': float(np.median(L_ecran_errors))
        },
        'predictions': {
            'gap_true': true_gap.tolist(),
            'gap_pred': predictions_gap.tolist(),
            'L_ecran_true': true_L_ecran.tolist(),
            'L_ecran_pred': predictions_L_ecran.tolist()
        }
    }
    
    return results

def create_visualizations_2400(results, save_dir="plots"):
    """
    Crée des visualisations des résultats pour 2400 échantillons.
    
    Args:
        results: Résultats de l'évaluation
        save_dir: Répertoire de sauvegarde
    """
    print(f"\n📈 Création des visualisations pour {results['n_samples']} échantillons...")
    
    Path(f"../{save_dir}").mkdir(exist_ok=True)
    
    # Données
    gap_true = np.array(results['predictions']['gap_true'])
    gap_pred = np.array(results['predictions']['gap_pred'])
    L_ecran_true = np.array(results['predictions']['L_ecran_true'])
    L_ecran_pred = np.array(results['predictions']['L_ecran_pred'])
    
    # Figure avec 2x2 sous-graphiques
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gap scatter plot
    ax1.scatter(gap_true, gap_pred, alpha=0.5, s=15)
    ax1.plot([gap_true.min(), gap_true.max()], [gap_true.min(), gap_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Gap Vrai (µm)')
    ax1.set_ylabel('Gap Prédit (µm)')
    ax1.set_title(f'Prédiction Gap (R² = {results["metrics"]["gap_r2"]:.4f}) - {results["n_samples"]} échantillons')
    ax1.grid(True, alpha=0.3)
    
    # L_ecran scatter plot
    ax2.scatter(L_ecran_true, L_ecran_pred, alpha=0.5, s=15)
    ax2.plot([L_ecran_true.min(), L_ecran_true.max()], [L_ecran_true.min(), L_ecran_true.max()], 'r--', lw=2)
    ax2.set_xlabel('L_ecran Vrai (µm)')
    ax2.set_ylabel('L_ecran Prédit (µm)')
    ax2.set_title(f'Prédiction L_ecran (R² = {results["metrics"]["L_ecran_r2"]:.4f}) - {results["n_samples"]} échantillons')
    ax2.grid(True, alpha=0.3)
    
    # Distribution des erreurs Gap
    gap_errors = np.abs(gap_pred - gap_true)
    ax3.hist(gap_errors, bins=60, alpha=0.7, edgecolor='black')
    ax3.axvline(x=0.01, color='red', linestyle='--', label='Tolérance ±0.01µm')
    ax3.set_xlabel('Erreur Absolue Gap (µm)')
    ax3.set_ylabel('Fréquence')
    ax3.set_title(f'Distribution des Erreurs Gap - {results["n_samples"]} échantillons')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Distribution des erreurs L_ecran
    L_ecran_errors = np.abs(L_ecran_pred - L_ecran_true)
    ax4.hist(L_ecran_errors, bins=60, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0.1, color='red', linestyle='--', label='Tolérance ±0.1µm')
    ax4.set_xlabel('Erreur Absolue L_ecran (µm)')
    ax4.set_ylabel('Fréquence')
    ax4.set_title(f'Distribution des Erreurs L_ecran - {results["n_samples"]} échantillons')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"../{save_dir}/test_predictions_2400_samples.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualisations sauvegardées: ../{save_dir}/test_predictions_2400_samples.png")

def visualize_noise_effect(X_test, y_test, noise_level=0.01, save_dir="plots"):
    """
    Visualise l'effet du bruit sur un profil d'intensité choisi au hasard.

    Args:
        X_test: Données de test
        y_test: Labels de test
        noise_level: Niveau de bruit à appliquer
        save_dir: Répertoire de sauvegarde
    """
    print(f"\n📊 VISUALISATION DE L'EFFET DU BRUIT {noise_level*100:.0f}%")
    print("="*50)

    # Choisir un échantillon au hasard
    np.random.seed(123)  # Pour la reproductibilité
    random_idx = np.random.randint(0, len(X_test))

    profile_original = X_test[random_idx]
    gap_true = y_test[random_idx, 0]
    L_ecran_true = y_test[random_idx, 1]

    print(f"📋 Échantillon sélectionné: {random_idx}")
    print(f"   Gap vrai: {gap_true:.4f} µm")
    print(f"   L_ecran vrai: {L_ecran_true:.2f} µm")

    # Générer le bruit
    noise = np.random.normal(0, noise_level * np.std(profile_original), profile_original.shape)
    profile_noisy = profile_original + noise

    # Créer la visualisation
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Profil original
    ax1.plot(profile_original, 'b-', linewidth=1.5, label='Profil Original')
    ax1.set_title(f'Profil Original (Gap: {gap_true:.4f}µm, L_ecran: {L_ecran_true:.2f}µm)')
    ax1.set_ylabel('Intensité')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Bruit ajouté
    ax2.plot(noise, 'r-', linewidth=1, alpha=0.7, label=f'Bruit Gaussien {noise_level*100:.0f}%')
    ax2.set_title(f'Bruit Gaussien Ajouté (σ = {noise_level*100:.0f}% × std(signal))')
    ax2.set_ylabel('Amplitude du Bruit')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Profil avec bruit
    ax3.plot(profile_original, 'b-', linewidth=1.5, alpha=0.7, label='Original')
    ax3.plot(profile_noisy, 'g-', linewidth=1.5, label=f'Avec Bruit {noise_level*100:.0f}%')
    ax3.set_title('Comparaison: Original vs Avec Bruit')
    ax3.set_xlabel('Position Radiale (points)')
    ax3.set_ylabel('Intensité')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    # Sauvegarder
    Path(f"../{save_dir}").mkdir(exist_ok=True)
    filename = f"../{save_dir}/noise_effect_visualization_{noise_level*100:.0f}pct.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualisation sauvegardée: {filename}")

    # Statistiques du bruit
    print(f"\n📈 Statistiques du bruit:")
    print(f"   Signal original - Min: {profile_original.min():.4f}, Max: {profile_original.max():.4f}")
    print(f"   Signal original - Std: {np.std(profile_original):.4f}")
    print(f"   Bruit - Min: {noise.min():.4f}, Max: {noise.max():.4f}")
    print(f"   Bruit - Std: {np.std(noise):.4f}")
    print(f"   Signal avec bruit - Min: {profile_noisy.min():.4f}, Max: {profile_noisy.max():.4f}")
    print(f"   Rapport signal/bruit: {np.std(profile_original)/np.std(noise):.2f}")

    return random_idx, profile_original, profile_noisy

def main():
    """
    Fonction principale de test sur 2400 échantillons.
    """
    print("🎯 TEST DE PRÉDICTION SUR ~2400 ÉCHANTILLONS - MODÈLE DUAL GAP + L_ECRAN")
    print("="*70)
    print(f"Auteur: Oussama GUELFAA")
    print(f"Date: 19-06-2025")
    print("="*70)
    
    try:
        # 1. Charger le modèle et les données
        model, X_test, y_test, data_loader = load_model_and_data()

        # 2. Évaluation complète sur ~2400 échantillons (SANS BRUIT)
        print("\n" + "="*70)
        print("🔬 TEST 1: ÉVALUATION SANS BRUIT (CONDITIONS IDÉALES)")
        print("="*70)
        results_clean = evaluate_model_2400(model, X_test, y_test, data_loader,
                                           test_with_noise=False)

        # 3. Visualisation de l'effet du bruit sur un échantillon
        visualize_noise_effect(X_test, y_test, noise_level=0.01)

        # 4. Évaluation avec bruit gaussien 1%
        print("\n" + "="*70)
        print("🔬 TEST 2: ÉVALUATION AVEC BRUIT GAUSSIEN 1%")
        print("="*70)
        # Fixer la seed pour la reproductibilité
        np.random.seed(42)
        results_noise = evaluate_model_2400(model, X_test, y_test, data_loader,
                                           test_with_noise=True, noise_level=0.01)

        # 5. Créer les visualisations pour les deux tests
        create_visualizations_2400(results_clean)

        # 6. Sauvegarder les résultats
        Path("../results").mkdir(exist_ok=True)

        # Résultats sans bruit
        results_clean_path = "../results/test_predictions_2400_samples_clean.json"
        with open(results_clean_path, 'w') as f:
            json.dump(results_clean, f, indent=2)

        # Résultats avec bruit 1%
        results_noise_path = "../results/test_predictions_2400_samples_noise_1pct.json"
        with open(results_noise_path, 'w') as f:
            json.dump(results_noise, f, indent=2)

        # 7. Comparaison des résultats
        print(f"\n📊 COMPARAISON DES RÉSULTATS:")
        print("="*60)
        print(f"{'Métrique':<20} {'Sans Bruit':<15} {'Avec Bruit 1%':<15} {'Dégradation':<12}")
        print("-" * 65)

        gap_r2_clean = results_clean['metrics']['gap_r2']
        gap_r2_noise = results_noise['metrics']['gap_r2']
        gap_r2_degradation = ((gap_r2_clean - gap_r2_noise) / gap_r2_clean) * 100

        L_ecran_r2_clean = results_clean['metrics']['L_ecran_r2']
        L_ecran_r2_noise = results_noise['metrics']['L_ecran_r2']
        L_ecran_r2_degradation = ((L_ecran_r2_clean - L_ecran_r2_noise) / L_ecran_r2_clean) * 100

        gap_acc_clean = results_clean['tolerance_analysis']['gap_accuracy']
        gap_acc_noise = results_noise['tolerance_analysis']['gap_accuracy']
        gap_acc_degradation = ((gap_acc_clean - gap_acc_noise) / gap_acc_clean) * 100

        L_ecran_acc_clean = results_clean['tolerance_analysis']['L_ecran_accuracy']
        L_ecran_acc_noise = results_noise['tolerance_analysis']['L_ecran_accuracy']
        L_ecran_acc_degradation = ((L_ecran_acc_clean - L_ecran_acc_noise) / L_ecran_acc_clean) * 100

        print(f"{'Gap R²':<20} {gap_r2_clean:<15.4f} {gap_r2_noise:<15.4f} {gap_r2_degradation:<12.2f}%")
        print(f"{'L_ecran R²':<20} {L_ecran_r2_clean:<15.4f} {L_ecran_r2_noise:<15.4f} {L_ecran_r2_degradation:<12.2f}%")
        print(f"{'Gap Accuracy':<20} {gap_acc_clean:<15.1%} {gap_acc_noise:<15.1%} {gap_acc_degradation:<12.2f}%")
        print(f"{'L_ecran Accuracy':<20} {L_ecran_acc_clean:<15.1%} {L_ecran_acc_noise:<15.1%} {L_ecran_acc_degradation:<12.2f}%")

        print(f"\n🎉 Tests terminés avec succès !")
        print(f"📊 Résultats sans bruit: {results_clean_path}")
        print(f"📊 Résultats avec bruit 1%: {results_noise_path}")
        print(f"📈 Visualisations: plots/test_predictions_2400_samples.png")

        # Analyse de robustesse
        print(f"\n🔬 ANALYSE DE ROBUSTESSE AU BRUIT:")
        print("="*40)
        if gap_r2_degradation < 5 and L_ecran_r2_degradation < 5:
            print("✅ EXCELLENT: Dégradation < 5% pour les deux paramètres")
        elif gap_r2_degradation < 10 and L_ecran_r2_degradation < 10:
            print("✅ BON: Dégradation < 10% pour les deux paramètres")
        else:
            print("⚠️ ATTENTION: Dégradation significative détectée")

        print(f"   Gap: {gap_r2_degradation:.2f}% de dégradation R²")
        print(f"   L_ecran: {L_ecran_r2_degradation:.2f}% de dégradation R²")
        
    except Exception as e:
        print(f"❌ Erreur dans le test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
