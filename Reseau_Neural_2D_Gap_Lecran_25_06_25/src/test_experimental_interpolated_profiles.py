#!/usr/bin/env python3
"""
Test du modèle sur les 25 profils expérimentaux interpolés avec prétraitement
Auteur: Oussama GUELFAA
Date: Juillet 2025

Teste le modèle amélioré sur les 25 profils expérimentaux interpolés
vers 600 points avec espacement 0.007 µm.
Inclut un prétraitement sophistiqué pour harmoniser les données expérimentales
avec les caractéristiques des données simulées d'entraînement.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import joblib
from Train_Improved import ImprovedDualParameterNet
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from experimental_data_preprocessor import ExperimentalDataPreprocessor

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_improved_model():
    """Charge le modèle amélioré."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Chemins absolus vers les modèles
    base_path = Path(__file__).parent.parent
    model_path = base_path / "models" / "dual_parameter_model_improved.pt"
    input_scaler_path = base_path / "models" / "input_scaler_improved.pkl"
    gap_scaler_path = base_path / "models" / "gap_scaler_improved.pkl"
    L_ecran_scaler_path = base_path / "models" / "L_ecran_scaler_improved.pkl"

    model = ImprovedDualParameterNet(input_size=600).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Charger les scalers améliorés
    input_scaler = joblib.load(input_scaler_path)
    gap_scaler = joblib.load(gap_scaler_path)
    L_ecran_scaler = joblib.load(L_ecran_scaler_path)

    logger.info(f"✅ Modèle amélioré chargé sur {device}")
    logger.info(f"   Modèle: {model_path}")

    return model, input_scaler, gap_scaler, L_ecran_scaler, device

def load_experimental_profile(csv_path, preprocessor=None):
    """
    Charge un profil expérimental interpolé depuis un fichier CSV avec prétraitement

    Args:
        csv_path (Path): Chemin vers le fichier CSV
        preprocessor (ExperimentalDataPreprocessor): Préprocesseur (optionnel)

    Returns:
        tuple: (intensities_raw, intensities_processed, r_values)
    """
    df = pd.read_csv(csv_path)

    # Extraire les intensités et positions radiales
    intensities_raw = df['I_experiment'].values
    r_values = df['r_experiment'].values if 'r_experiment' in df.columns else np.linspace(0, 4.193, len(intensities_raw))

    # Vérifier que nous avons 600 points
    if len(intensities_raw) != 600:
        logger.warning(f"⚠️  Profil {csv_path.name} a {len(intensities_raw)} points au lieu de 600")
        if len(intensities_raw) > 600:
            intensities_raw = intensities_raw[:600]
            r_values = r_values[:600]
        else:
            intensities_raw = np.pad(intensities_raw, (0, 600 - len(intensities_raw)), 'edge')
            r_values = np.pad(r_values, (0, 600 - len(r_values)), 'edge')

    # Appliquer le prétraitement si fourni
    if preprocessor is not None:
        intensities_processed = preprocessor.preprocess_profile(
            intensities_raw, r_values, profile_name=csv_path.stem
        )
        logger.info(f"✅ Prétraitement appliqué à {csv_path.name}")
    else:
        intensities_processed = intensities_raw
        logger.info(f"⚠️  Aucun prétraitement appliqué à {csv_path.name}")

    return intensities_raw, intensities_processed, r_values

def predict_experimental(model, input_scaler, gap_scaler, L_ecran_scaler, device, intensities, use_additional_filtering=True):
    """
    Prédiction sur un profil expérimental (prétraité ou non)

    Args:
        model: Modèle PyTorch
        input_scaler, gap_scaler, L_ecran_scaler: Scalers
        device: Device PyTorch
        intensities (np.array): Intensités expérimentales (déjà prétraitées de préférence)
        use_additional_filtering (bool): Appliquer un filtrage supplémentaire léger

    Returns:
        tuple: (gap_pred, L_ecran_pred)
    """
    # Filtrage léger supplémentaire si demandé (plus conservateur si déjà prétraité)
    if use_additional_filtering:
        intensities_filtered = gaussian_filter1d(intensities, sigma=0.3)  # Réduit de 0.5 à 0.3
    else:
        intensities_filtered = intensities

    # Normalisation
    intensities_scaled = input_scaler.transform(intensities_filtered.reshape(1, -1))

    with torch.no_grad():
        intensities_tensor = torch.FloatTensor(intensities_scaled).to(device)
        prediction_scaled = model(intensities_tensor)

        # Dénormalisation séparée
        gap_pred = gap_scaler.inverse_transform(
            prediction_scaled[0, 0].cpu().numpy().reshape(-1, 1)
        )[0, 0]
        L_ecran_pred = L_ecran_scaler.inverse_transform(
            prediction_scaled[0, 1].cpu().numpy().reshape(-1, 1)
        )[0, 0]

    return gap_pred, L_ecran_pred

def test_experimental_profiles():
    """Teste le modèle sur les 25 profils expérimentaux interpolés avec prétraitement."""
    logger.info("🧪 TEST DU MODÈLE SUR LES PROFILS EXPÉRIMENTAUX INTERPOLÉS AVEC PRÉTRAITEMENT")
    logger.info("="*80)

    # Charger le modèle
    model, input_scaler, gap_scaler, L_ecran_scaler, device = load_improved_model()

    # Initialiser le préprocesseur
    preprocessor = ExperimentalDataPreprocessor()
    logger.info("🔧 Préprocesseur initialisé pour harmonisation avec données simulées")

    # Chemin vers les profils interpolés
    base_path = Path(__file__).parent.parent.parent
    profiles_dir = base_path / "data_generation" / "Experimental_data_analysis" / "interpolated_profiles_600pts"

    if not profiles_dir.exists():
        logger.error(f"❌ Dossier non trouvé: {profiles_dir}")
        return None, None

    # Lister tous les fichiers CSV interpolés
    csv_files = list(profiles_dir.glob("profile_*_interpolated.csv"))
    csv_files.sort()  # Trier par ordre numérique

    logger.info(f"📊 Profils expérimentaux trouvés: {len(csv_files)}")

    if len(csv_files) == 0:
        logger.error("❌ Aucun profil interpolé trouvé!")
        return None, None

    # Créer dossier pour les graphiques de prétraitement
    plots_base_path = Path(__file__).parent.parent
    preprocessing_plots_dir = plots_base_path / "plots" / "preprocessing_comparisons"
    preprocessing_plots_dir.mkdir(parents=True, exist_ok=True)

    # Préparer la liste des résultats
    results = []

    # Traiter chaque profil
    for i, csv_file in enumerate(csv_files):
        profile_name = csv_file.stem  # Nom sans extension
        profile_number = i + 1

        logger.info(f"   Traitement profil {profile_number}/25: {profile_name}")

        try:
            # Charger le profil expérimental avec prétraitement
            intensities_raw, intensities_processed, r_values = load_experimental_profile(
                csv_file, preprocessor=preprocessor
            )

            # Créer graphique de comparaison prétraitement (pour les 5 premiers profils)
            if i < 5:
                comparison_plot_path = preprocessing_plots_dir / f"preprocessing_comparison_{profile_name}.png"
                preprocessor.create_comparison_plot(
                    intensities_raw, intensities_processed, r_values,
                    profile_name, save_path=comparison_plot_path
                )

            # Valider le prétraitement
            validation = preprocessor.validate_preprocessing(intensities_processed)

            # Faire la prédiction avec les données prétraitées
            gap_pred, L_ecran_pred = predict_experimental(
                model, input_scaler, gap_scaler, L_ecran_scaler, device,
                intensities_processed, use_additional_filtering=False  # Pas de filtrage supplémentaire
            )

            # Ajouter aux résultats
            results.append({
                'profile_number': profile_number,
                'filename': csv_file.name,
                'profile_name': profile_name,
                'Gap_predit_um': gap_pred,
                'L_ecran_predit_um': L_ecran_pred,
                'intensite_raw_min': np.min(intensities_raw),
                'intensite_raw_max': np.max(intensities_raw),
                'intensite_raw_moyenne': np.mean(intensities_raw),
                'intensite_raw_std': np.std(intensities_raw),
                'intensite_processed_min': np.min(intensities_processed),
                'intensite_processed_max': np.max(intensities_processed),
                'intensite_processed_moyenne': np.mean(intensities_processed),
                'intensite_processed_std': np.std(intensities_processed),
                'preprocessing_validation_ok': validation['overall_ok'],
                'preprocessing_mean_ok': validation['mean_ok'],
                'preprocessing_std_ok': validation['std_ok'],
                'preprocessing_range_ok': validation['range_ok']
            })

            status_icon = "✅" if validation['overall_ok'] else "⚠️"
            logger.info(f"      {status_icon} Gap: {gap_pred:.4f} µm, L_écran: {L_ecran_pred:.1f} µm")
            logger.info(f"         Prétraitement: {'OK' if validation['overall_ok'] else 'PARTIEL'}")

        except Exception as e:
            logger.error(f"❌ Erreur avec {csv_file.name}: {e}")
            # Ajouter une ligne avec des NaN pour garder la trace
            results.append({
                'profile_number': profile_number,
                'filename': csv_file.name,
                'profile_name': profile_name,
                'Gap_predit_um': np.nan,
                'L_ecran_predit_um': np.nan,
                'intensite_raw_min': np.nan,
                'intensite_raw_max': np.nan,
                'intensite_raw_moyenne': np.nan,
                'intensite_raw_std': np.nan,
                'intensite_processed_min': np.nan,
                'intensite_processed_max': np.nan,
                'intensite_processed_moyenne': np.nan,
                'intensite_processed_std': np.nan,
                'preprocessing_validation_ok': False,
                'preprocessing_mean_ok': False,
                'preprocessing_std_ok': False,
                'preprocessing_range_ok': False
            })
            continue
    
    logger.info(f"✅ Traitement terminé: {len(results)} profils")
    
    # Créer le DataFrame
    df_results = pd.DataFrame(results)
    
    # Filtrer les résultats valides pour les statistiques
    valid_results = df_results.dropna()
    n_valid = len(valid_results)
    n_errors = len(results) - n_valid
    
    logger.info(f"📊 Résultats valides: {n_valid}/{len(results)} ({n_errors} erreurs)")
    
    if n_valid > 0:
        # Calculer les statistiques des prédictions
        gap_mean = valid_results['Gap_predit_um'].mean()
        gap_std = valid_results['Gap_predit_um'].std()
        gap_min = valid_results['Gap_predit_um'].min()
        gap_max = valid_results['Gap_predit_um'].max()
        
        L_ecran_mean = valid_results['L_ecran_predit_um'].mean()
        L_ecran_std = valid_results['L_ecran_predit_um'].std()
        L_ecran_min = valid_results['L_ecran_predit_um'].min()
        L_ecran_max = valid_results['L_ecran_predit_um'].max()
        
        logger.info(f"\n📊 STATISTIQUES DES PRÉDICTIONS ({n_valid} profils)")
        logger.info("="*60)
        logger.info(f"GAP PRÉDIT:")
        logger.info(f"   Moyenne: {gap_mean:.4f} ± {gap_std:.4f} µm")
        logger.info(f"   Min-Max: {gap_min:.4f} - {gap_max:.4f} µm")
        
        logger.info(f"\nL_ÉCRAN PRÉDIT:")
        logger.info(f"   Moyenne: {L_ecran_mean:.1f} ± {L_ecran_std:.1f} µm")
        logger.info(f"   Min-Max: {L_ecran_min:.1f} - {L_ecran_max:.1f} µm")
        
        # Analyser la distribution des intensités (données prétraitées)
        intensite_mean_global = valid_results['intensite_processed_moyenne'].mean()
        intensite_std_global = valid_results['intensite_processed_std'].mean()

        # Analyser aussi les données brutes pour comparaison
        intensite_raw_mean_global = valid_results['intensite_raw_moyenne'].mean()
        intensite_raw_std_global = valid_results['intensite_raw_std'].mean()

        logger.info(f"\nINTENSITÉS EXPÉRIMENTALES (APRÈS PRÉTRAITEMENT):")
        logger.info(f"   Intensité moyenne globale: {intensite_mean_global:.3f}")
        logger.info(f"   Écart-type moyen: {intensite_std_global:.3f}")

        logger.info(f"\nINTENSITÉS EXPÉRIMENTALES (BRUTES):")
        logger.info(f"   Intensité moyenne globale: {intensite_raw_mean_global:.3f}")
        logger.info(f"   Écart-type moyen: {intensite_raw_std_global:.3f}")

        # Analyser l'efficacité du prétraitement
        preprocessing_success_rate = valid_results['preprocessing_validation_ok'].mean() * 100
        logger.info(f"\nEFFICACITÉ DU PRÉTRAITEMENT:")
        logger.info(f"   Taux de validation réussie: {preprocessing_success_rate:.1f}%")
        
        # Ajouter les statistiques au DataFrame
        stats_row = {
            'profile_number': 'STATS',
            'filename': 'STATISTIQUES_GLOBALES',
            'profile_name': 'MOYENNES',
            'Gap_predit_um': gap_mean,
            'L_ecran_predit_um': L_ecran_mean,
            'intensite_min': f'±{gap_std:.4f}',
            'intensite_max': f'±{L_ecran_std:.1f}',
            'intensite_moyenne': intensite_mean_global,
            'intensite_std': intensite_std_global
        }
        
        # Ajouter la ligne de statistiques
        df_results = pd.concat([df_results, pd.DataFrame([stats_row])], ignore_index=True)
    
    return df_results, valid_results

def create_visualizations(df_results, valid_results):
    """Crée des visualisations des résultats."""
    if valid_results is None or len(valid_results) == 0:
        logger.warning("⚠️  Pas de données valides pour les visualisations")
        return
    
    logger.info("📊 Génération des visualisations...")
    
    # Créer le dossier plots s'il n'existe pas
    base_path = Path(__file__).parent.parent
    plots_dir = base_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Graphique 1: Distribution des prédictions Gap
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(valid_results['Gap_predit_um'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Gap prédit (µm)', fontweight='bold')
    plt.ylabel('Fréquence', fontweight='bold')
    plt.title('Distribution des Gaps prédits\n(Profils expérimentaux)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Graphique 2: Distribution des prédictions L_écran
    plt.subplot(1, 3, 2)
    plt.hist(valid_results['L_ecran_predit_um'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('L_écran prédit (µm)', fontweight='bold')
    plt.ylabel('Fréquence', fontweight='bold')
    plt.title('Distribution des L_écrans prédits\n(Profils expérimentaux)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Graphique 3: Relation Gap vs L_écran
    plt.subplot(1, 3, 3)
    plt.scatter(valid_results['Gap_predit_um'], valid_results['L_ecran_predit_um'], 
               alpha=0.7, s=50, color='red', edgecolor='black')
    plt.xlabel('Gap prédit (µm)', fontweight='bold')
    plt.ylabel('L_écran prédit (µm)', fontweight='bold')
    plt.title('Relation Gap vs L_écran\n(Profils expérimentaux)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    plot_path = plots_dir / "experimental_predictions_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Visualisation sauvegardée: {plot_path}")
    
    # Graphique 2: Évolution des prédictions par profil
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(valid_results['profile_number'], valid_results['Gap_predit_um'], 
             'bo-', linewidth=2, markersize=6, alpha=0.7)
    plt.xlabel('Numéro de profil', fontweight='bold')
    plt.ylabel('Gap prédit (µm)', fontweight='bold')
    plt.title('Évolution des prédictions Gap par profil expérimental', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(valid_results['profile_number'], valid_results['L_ecran_predit_um'], 
             'ro-', linewidth=2, markersize=6, alpha=0.7)
    plt.xlabel('Numéro de profil', fontweight='bold')
    plt.ylabel('L_écran prédit (µm)', fontweight='bold')
    plt.title('Évolution des prédictions L_écran par profil expérimental', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    evolution_path = plots_dir / "experimental_predictions_evolution.png"
    plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Évolution sauvegardée: {evolution_path}")

def save_results(df_results):
    """Sauvegarde les résultats dans un fichier CSV."""
    # Chemin absolu vers le dossier results
    base_path = Path(__file__).parent.parent
    results_dir = base_path / "results"
    results_dir.mkdir(exist_ok=True)

    # Sauvegarder dans un fichier CSV avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = results_dir / f"test_experimental_profiles_preprocessed_{timestamp}.csv"

    # Sauvegarder le CSV
    df_results.to_csv(csv_filename, index=False, float_format='%.6f')

    logger.info(f"\n💾 RÉSULTATS SAUVEGARDÉS:")
    logger.info(f"   Fichier: {csv_filename}")
    logger.info(f"   Format: CSV avec {len(df_results)} lignes")
    logger.info(f"   Colonnes: {list(df_results.columns)}")

    return str(csv_filename)

def main():
    """Fonction principale."""
    logger.info("🚀 DÉBUT DU TEST SUR PROFILS EXPÉRIMENTAUX")
    
    start_time = datetime.now()
    
    # Exécuter le test
    df_results, valid_results = test_experimental_profiles()
    
    if df_results is not None:
        # Créer les visualisations
        create_visualizations(df_results, valid_results)
        
        # Sauvegarder les résultats
        csv_filename = save_results(df_results)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n✅ TEST EXPÉRIMENTAL TERMINÉ")
        logger.info(f"   Durée: {duration}")
        logger.info(f"   Fichier CSV: {csv_filename}")
        logger.info(f"   Profils traités: {len(df_results)-1}")  # -1 pour la ligne de stats
        
        # Afficher un aperçu des résultats
        if valid_results is not None and len(valid_results) > 0:
            logger.info(f"\n📋 APERÇU DES PRÉDICTIONS (5 premiers profils):")
            for i, row in valid_results.head(5).iterrows():
                logger.info(f"   {row['profile_name']}: Gap={row['Gap_predit_um']:.4f}µm, "
                           f"L_écran={row['L_ecran_predit_um']:.1f}µm")
    else:
        logger.error("❌ Échec du test expérimental")

if __name__ == "__main__":
    main()
