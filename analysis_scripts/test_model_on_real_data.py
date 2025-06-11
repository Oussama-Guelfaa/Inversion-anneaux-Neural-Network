#!/usr/bin/env python3
"""
Test du modèle model_noise_2percent.pth sur données réelles

Ce script teste le modèle entraîné avec 2% de bruit sur un fichier de données réelles
spécifique pour évaluer ses performances en conditions expérimentales.

Auteur: Oussama GUELFAA
Date: 10 - 01 - 2025
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pickle
import json
from pathlib import Path

class RobustGapPredictor(nn.Module):
    """
    Modèle robuste pour prédiction du gap avec régularisation.
    Architecture identique à celle utilisée pour l'entraînement.
    """
    
    def __init__(self, input_size=1000, dropout_rate=0.2):
        super(RobustGapPredictor, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

def load_model_and_scaler(model_path):
    """
    Charge le modèle entraîné et le scaler associé.
    
    Args:
        model_path (str): Chemin vers le fichier .pth du modèle
        
    Returns:
        tuple: (model, scaler, model_info)
    """
    print(f"=== CHARGEMENT DU MODÈLE ===")
    print(f"Chemin: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_path} n'existe pas")
    
    # Charger le checkpoint (avec weights_only=False pour compatibilité)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    print(f"Contenu du checkpoint: {list(checkpoint.keys())}")
    
    # Extraire les informations
    model_state_dict = checkpoint['model_state_dict']
    scaler = checkpoint['scaler']
    noise_level = checkpoint.get('noise_level', 'unknown')
    metrics = checkpoint.get('metrics', {})
    
    # Créer et charger le modèle
    model = RobustGapPredictor(input_size=1000)
    model.load_state_dict(model_state_dict)
    model.eval()  # Mode évaluation
    
    print(f"Modèle chargé avec succès:")
    print(f"  - Niveau de bruit d'entraînement: {noise_level}%")
    print(f"  - R² sur test: {metrics.get('r2', 'N/A'):.4f}")
    print(f"  - RMSE sur test: {metrics.get('rmse', 'N/A'):.4f} µm")
    
    return model, scaler, {
        'noise_level': noise_level,
        'metrics': metrics
    }

def load_real_data(data_path):
    """
    Charge les données réelles depuis un fichier .mat.
    
    Args:
        data_path (str): Chemin vers le fichier .mat
        
    Returns:
        dict: Données extraites du fichier
    """
    print(f"\n=== CHARGEMENT DES DONNÉES RÉELLES ===")
    print(f"Fichier: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Le fichier {data_path} n'existe pas")
    
    # Charger le fichier .mat
    data = sio.loadmat(data_path)
    
    print(f"Variables disponibles dans le fichier:")
    for key in data.keys():
        if not key.startswith('__'):
            value = data[key]
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Extraire les données principales
    ratio = data['ratio'].flatten() if 'ratio' in data else None
    gap_true = float(data['gap'][0, 0]) if 'gap' in data else None
    L_ecran = float(data['L_ecran_subs'][0, 0]) if 'L_ecran_subs' in data else None
    x_coords = data['x'].flatten() if 'x' in data else None
    
    # Extraire le gap depuis le nom de fichier comme vérification
    filename = os.path.basename(data_path)
    gap_from_filename = extract_gap_from_filename(filename)
    
    print(f"\nDonnées extraites:")
    print(f"  - Profil ratio: {ratio.shape if ratio is not None else 'Non trouvé'}")
    print(f"  - Gap (fichier): {gap_true} µm")
    print(f"  - Gap (nom fichier): {gap_from_filename} µm")
    print(f"  - L_ecran: {L_ecran} µm")
    print(f"  - Coordonnées x: {x_coords.shape if x_coords is not None else 'Non trouvé'}")
    
    if ratio is None:
        raise ValueError("Variable 'ratio' non trouvée dans le fichier")
    
    return {
        'ratio': ratio,
        'gap_true': gap_true,
        'gap_filename': gap_from_filename,
        'L_ecran': L_ecran,
        'x_coords': x_coords,
        'filename': filename
    }

def extract_gap_from_filename(filename):
    """
    Extrait la valeur du gap depuis le nom de fichier.
    
    Args:
        filename (str): Nom du fichier (ex: "gap_0.0250um_L_10.000um.mat")
        
    Returns:
        float: Valeur du gap en microns
    """
    try:
        # Format: gap_X.XXXXum_L_Y.YYYum.mat
        parts = filename.split('_')
        gap_part = parts[1]  # "0.0250um"
        gap_value = float(gap_part.replace('um', ''))
        return gap_value
    except Exception as e:
        print(f"Erreur extraction gap de {filename}: {e}")
        return None

def predict_gap(model, scaler, ratio_profile):
    """
    Prédit le gap à partir du profil d'intensité.
    
    Args:
        model: Modèle PyTorch chargé
        scaler: StandardScaler pour normalisation
        ratio_profile: Profil d'intensité (array 1D)
        
    Returns:
        float: Gap prédit en microns
    """
    print(f"\n=== PRÉDICTION DU GAP ===")
    
    # Vérifier la taille du profil
    print(f"Taille du profil d'entrée: {len(ratio_profile)}")
    
    if len(ratio_profile) != 1000:
        print(f"⚠️  Attention: Le modèle attend 1000 points, reçu {len(ratio_profile)}")
        if len(ratio_profile) > 1000:
            ratio_profile = ratio_profile[:1000]
            print(f"Profil tronqué à 1000 points")
        else:
            # Padding avec des zéros si nécessaire
            ratio_profile = np.pad(ratio_profile, (0, 1000 - len(ratio_profile)), 'constant')
            print(f"Profil complété avec des zéros à 1000 points")
    
    # Normalisation
    ratio_normalized = scaler.transform(ratio_profile.reshape(1, -1))
    
    # Conversion en tensor PyTorch
    ratio_tensor = torch.FloatTensor(ratio_normalized)
    
    # Prédiction
    with torch.no_grad():
        gap_pred = model(ratio_tensor).item()
    
    print(f"Gap prédit: {gap_pred:.4f} µm")
    
    return gap_pred

def analyze_prediction(gap_true, gap_pred, tolerance=0.01):
    """
    Analyse la qualité de la prédiction.
    
    Args:
        gap_true (float): Valeur réelle du gap
        gap_pred (float): Valeur prédite du gap
        tolerance (float): Tolérance acceptable en µm
        
    Returns:
        dict: Résultats de l'analyse
    """
    print(f"\n=== ANALYSE DE LA PRÉDICTION ===")
    
    error = gap_pred - gap_true
    abs_error = abs(error)
    rel_error_percent = (abs_error / gap_true) * 100
    
    within_tolerance = abs_error <= tolerance
    
    print(f"Gap réel:        {gap_true:.4f} µm")
    print(f"Gap prédit:      {gap_pred:.4f} µm")
    print(f"Erreur:          {error:+.4f} µm")
    print(f"Erreur absolue:  {abs_error:.4f} µm")
    print(f"Erreur relative: {rel_error_percent:.2f}%")
    print(f"Tolérance:       ±{tolerance:.3f} µm")
    print(f"Dans tolérance:  {'✅ OUI' if within_tolerance else '❌ NON'}")
    
    # Évaluation qualitative
    if abs_error <= 0.001:
        quality = "EXCELLENT"
        emoji = "🎉"
    elif abs_error <= 0.005:
        quality = "TRÈS BON"
        emoji = "✅"
    elif abs_error <= 0.01:
        quality = "BON"
        emoji = "👍"
    elif abs_error <= 0.05:
        quality = "ACCEPTABLE"
        emoji = "⚠️"
    else:
        quality = "PROBLÉMATIQUE"
        emoji = "❌"
    
    print(f"Qualité:         {emoji} {quality}")
    
    return {
        'gap_true': gap_true,
        'gap_pred': gap_pred,
        'error': error,
        'abs_error': abs_error,
        'rel_error_percent': rel_error_percent,
        'within_tolerance': within_tolerance,
        'quality': quality
    }

def visualize_results(data, gap_pred, analysis):
    """
    Visualise les résultats de la prédiction.
    
    Args:
        data (dict): Données chargées
        gap_pred (float): Gap prédit
        analysis (dict): Résultats de l'analyse
    """
    print(f"\n=== VISUALISATION ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique 1: Profil d'intensité
    ax1 = axes[0]
    if data['x_coords'] is not None:
        ax1.plot(data['x_coords'], data['ratio'], 'b-', linewidth=2, label='Profil expérimental')
        ax1.set_xlabel('Position radiale (µm)')
    else:
        ax1.plot(data['ratio'], 'b-', linewidth=2, label='Profil expérimental')
        ax1.set_xlabel('Index de position')
    
    ax1.set_ylabel('Intensité (ratio)')
    ax1.set_title(f'Profil d\'Intensité\n{data["filename"]}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Graphique 2: Comparaison prédiction vs réalité
    ax2 = axes[1]
    
    gaps = [data['gap_true'], gap_pred]
    labels = ['Réel', 'Prédit']
    colors = ['blue', 'red']
    
    bars = ax2.bar(labels, gaps, color=colors, alpha=0.7)
    ax2.set_ylabel('Gap (µm)')
    ax2.set_title('Comparaison Gap Réel vs Prédit')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{gap:.4f} µm', ha='center', va='bottom', fontweight='bold')
    
    # Ajouter l'erreur
    error_text = f"Erreur: {analysis['error']:+.4f} µm\n"
    error_text += f"Erreur relative: {analysis['rel_error_percent']:.2f}%\n"
    error_text += f"Qualité: {analysis['quality']}"
    
    ax2.text(0.02, 0.98, error_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = f"test_prediction_{data['filename'].replace('.mat', '.png')}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Graphique sauvegardé: {output_path}")

def save_results(data, gap_pred, analysis, model_info):
    """
    Sauvegarde les résultats du test.
    
    Args:
        data (dict): Données d'entrée
        gap_pred (float): Gap prédit
        analysis (dict): Analyse de la prédiction
        model_info (dict): Informations sur le modèle
    """
    results = {
        'test_info': {
            'date': '10-01-2025',
            'model_file': 'model_noise_2percent.pth',
            'data_file': data['filename'],
            'model_noise_level': model_info['noise_level'],
            'model_metrics': model_info['metrics']
        },
        'input_data': {
            'filename': data['filename'],
            'gap_true': float(data['gap_true']),
            'gap_filename': float(data['gap_filename']) if data['gap_filename'] is not None else None,
            'L_ecran': float(data['L_ecran']) if data['L_ecran'] is not None else None,
            'profile_length': len(data['ratio'])
        },
        'prediction': {
            'gap_predicted': float(gap_pred)
        },
        'analysis': analysis
    }
    
    # Sauvegarder en JSON
    output_file = f"test_results_{data['filename'].replace('.mat', '.json')}"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Résultats sauvegardés: {output_file}")

def analyze_data_distribution(data):
    """
    Analyse la distribution des données pour comprendre les différences.

    Args:
        data (dict): Données chargées
    """
    print(f"\n=== ANALYSE DE LA DISTRIBUTION DES DONNÉES ===")

    ratio = data['ratio']

    print(f"Statistiques du profil d'intensité:")
    print(f"  - Min: {ratio.min():.6f}")
    print(f"  - Max: {ratio.max():.6f}")
    print(f"  - Moyenne: {ratio.mean():.6f}")
    print(f"  - Écart-type: {ratio.std():.6f}")
    print(f"  - Médiane: {np.median(ratio):.6f}")

    # Vérifier s'il y a des valeurs anormales
    if ratio.min() < 0:
        print(f"  ⚠️  Valeurs négatives détectées!")

    if ratio.max() > 10:
        print(f"  ⚠️  Valeurs très élevées détectées!")

    # Analyser la forme du profil
    peak_idx = np.argmax(ratio)
    print(f"  - Position du pic: index {peak_idx} ({peak_idx/len(ratio)*100:.1f}%)")
    print(f"  - Valeur du pic: {ratio[peak_idx]:.6f}")

def compare_with_training_data():
    """
    Compare avec les données d'entraînement pour comprendre les différences.
    """
    print(f"\n=== COMPARAISON AVEC DONNÉES D'ENTRAÎNEMENT ===")

    # Charger quelques échantillons des données d'entraînement
    training_dir = "data_generation/dataset_small_particle"

    if os.path.exists(training_dir):
        training_files = [f for f in os.listdir(training_dir) if f.endswith('.mat')][:5]

        print(f"Analyse de {len(training_files)} échantillons d'entraînement:")

        for filename in training_files:
            try:
                file_path = os.path.join(training_dir, filename)
                data = sio.loadmat(file_path)
                ratio = data['ratio'].flatten()
                gap = float(data['gap'][0, 0])

                print(f"  {filename}:")
                print(f"    Gap: {gap:.4f} µm")
                print(f"    Ratio min/max: {ratio.min():.4f} / {ratio.max():.4f}")
                print(f"    Ratio moyenne: {ratio.mean():.4f}")

            except Exception as e:
                print(f"    Erreur: {e}")
    else:
        print(f"Dossier d'entraînement non trouvé: {training_dir}")

def main():
    """Fonction principale pour tester le modèle."""

    print("="*60)
    print("TEST DU MODÈLE SUR DONNÉES RÉELLES")
    print("="*60)

    # Chemins des fichiers
    model_path = "Neural_Network_Noise_Robustness_Test_10_01_25/models/model_noise_2percent.pth"
    data_path = "data_generation/dataset/gap_0.0250um_L_10.000um.mat"

    try:
        # 1. Charger le modèle
        model, scaler, model_info = load_model_and_scaler(model_path)

        # 2. Charger les données réelles
        data = load_real_data(data_path)

        # 3. Analyser la distribution des données
        analyze_data_distribution(data)

        # 4. Comparer avec les données d'entraînement
        compare_with_training_data()

        # 5. Faire la prédiction
        gap_pred = predict_gap(model, scaler, data['ratio'])

        # 6. Analyser les résultats
        analysis = analyze_prediction(data['gap_true'], gap_pred)

        # 7. Visualiser
        visualize_results(data, gap_pred, analysis)

        # 8. Sauvegarder les résultats (ignorer l'erreur JSON pour l'instant)
        try:
            save_results(data, gap_pred, analysis, model_info)
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde JSON (ignorée): {e}")

        print(f"\n{'='*60}")
        print("TEST TERMINÉ AVEC SUCCÈS")
        print(f"{'='*60}")

        # Résumé final
        print(f"\n🎯 RÉSUMÉ:")
        print(f"  Fichier testé: {data['filename']}")
        print(f"  Gap réel: {data['gap_true']:.4f} µm")
        print(f"  Gap prédit: {gap_pred:.4f} µm")
        print(f"  Erreur: {analysis['error']:+.4f} µm ({analysis['rel_error_percent']:.2f}%)")
        print(f"  Qualité: {analysis['quality']}")

        # Diagnostic
        print(f"\n🔍 DIAGNOSTIC:")
        if gap_pred < 0:
            print(f"  ❌ Prédiction négative - Problème potentiel:")
            print(f"     • Différence entre données réelles et d'entraînement")
            print(f"     • Normalisation inadéquate")
            print(f"     • Modèle non adapté à ce type de données")
        elif abs(analysis['error']) > 0.01:
            print(f"  ⚠️  Erreur élevée - Causes possibles:")
            print(f"     • Données hors distribution d'entraînement")
            print(f"     • Bruit ou artefacts dans les données réelles")
            print(f"     • Calibration nécessaire")
        else:
            print(f"  ✅ Prédiction acceptable")

    except Exception as e:
        print(f"❌ Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
