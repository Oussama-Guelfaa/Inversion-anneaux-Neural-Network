#!/usr/bin/env python3
"""
Analyse de la généralisation locale dans la plage 0.5-1.0 µm

Ce script calcule les erreurs relatives entre les prédictions et la droite idéale
pour évaluer la finesse de généralisation du réseau dans une plage réaliste d'usage.

Auteur: Oussama GUELFAA
Date: 11 - 01 - 2025
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F

import json

# Configuration pour reproductibilité
torch.manual_seed(42)
np.random.seed(42)

class RobustGapPredictor(nn.Module):
    """Modèle robuste pour prédiction du gap - Architecture identique."""
    
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

def load_dataset():
    """Charge le dataset complet depuis dataset_small_particle."""
    
    print("=== CHARGEMENT DU DATASET ===")
    
    dataset_dir = "../../data_generation/dataset_small_particle"
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Le dossier {dataset_dir} n'existe pas")
    
    # Lister tous les fichiers .mat
    mat_files = [f for f in os.listdir(dataset_dir) if f.endswith('.mat') and f.startswith('gap_')]
    mat_files.sort()
    
    print(f"Nombre de fichiers trouvés: {len(mat_files)}")
    
    X = []  # Profils d'intensité
    y = []  # Valeurs de gap
    
    for i, filename in enumerate(mat_files):
        mat_path = os.path.join(dataset_dir, filename)
        
        try:
            data = sio.loadmat(mat_path)
            
            # Extraire le profil d'intensité (ratio)
            ratio = data['ratio'].flatten()  # (1000,)
            
            # Extraire la valeur du gap
            gap_value = float(data['gap'][0, 0])
            
            X.append(ratio)
            y.append(gap_value)
            
        except Exception as e:
            print(f"Erreur avec {filename}: {e}")
    
    X = np.array(X)  # (400, 1000)
    y = np.array(y)  # (400,)
    
    print(f"Données chargées: X{X.shape}, y{y.shape}")
    print(f"Gap range: {y.min():.4f} - {y.max():.4f} µm")
    
    return X, y

def add_gaussian_noise(X, noise_level_percent):
    """Ajoute du bruit gaussien proportionnel au signal."""
    
    if noise_level_percent == 0:
        return X.copy()
    
    # Calculer l'écart-type du signal pour chaque échantillon
    signal_std = np.std(X, axis=1, keepdims=True)
    
    # Générer le bruit proportionnel
    noise_std = (noise_level_percent / 100.0) * signal_std
    noise = np.random.normal(0, noise_std, X.shape)
    
    X_noisy = X + noise
    
    return X_noisy

def recreate_experiment_data():
    """Recrée exactement les mêmes données que l'expérience originale."""
    
    print("=== RECRÉATION DES DONNÉES D'EXPÉRIENCE ===")
    
    # Charger le dataset complet
    X, y = load_dataset()
    
    # Reproduire exactement la même division (même seed)
    total_needed = 400  # 300 train + 100 test
    indices = np.random.choice(len(X), size=total_needed, replace=False)
    
    # Diviser en train/test
    train_indices = indices[:300]
    test_indices = indices[300:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"Données recréées:")
    print(f"  Train: X{X_train.shape}, y{y_train.shape}")
    print(f"  Test: X{X_test.shape}, y{y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_and_predict():
    """Entraîne le modèle et génère les prédictions."""
    
    print("=== ENTRAÎNEMENT ET PRÉDICTION ===")
    
    # Recréer les données
    X_train, X_test, y_train, y_test = recreate_experiment_data()
    
    # Ajouter du bruit aux données d'entraînement (5%)
    X_train_noisy = add_gaussian_noise(X_train, 5)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_noisy)
    X_test_scaled = scaler.transform(X_test)  # Test sans bruit
    
    # Modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustGapPredictor(input_size=X_train.shape[1]).to(device)
    
    # Entraînement rapide (version simplifiée pour l'analyse)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Convertir en tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    print("Entraînement en cours...")
    model.train()
    
    # Entraînement simplifié (50 époques pour l'analyse)
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")
    
    # Prédictions sur le test
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze().cpu().numpy()
    
    print(f"Prédictions générées pour {len(y_test)} échantillons de test")
    
    return y_test, y_pred

def analyze_local_generalization(y_true, y_pred, range_min=0.5, range_max=1.0):
    """
    Analyse la généralisation locale dans la plage spécifiée.
    
    Args:
        y_true: Valeurs réelles de gap
        y_pred: Valeurs prédites de gap
        range_min: Limite inférieure de la plage (µm)
        range_max: Limite supérieure de la plage (µm)
    """
    
    print(f"\n=== ANALYSE GÉNÉRALISATION LOCALE [{range_min}-{range_max} µm] ===")
    
    # Filtrer les points dans la plage spécifiée
    mask = (y_true >= range_min) & (y_true <= range_max)
    y_true_range = y_true[mask]
    y_pred_range = y_pred[mask]
    
    print(f"Nombre de points dans la plage [{range_min}-{range_max} µm]: {len(y_true_range)}")
    
    if len(y_true_range) == 0:
        print("❌ Aucun point dans la plage spécifiée")
        return
    
    # Calculer les erreurs relatives
    relative_errors = np.abs((y_pred_range - y_true_range) / y_true_range) * 100
    
    # Statistiques des erreurs relatives
    max_error = np.max(relative_errors)
    mean_error = np.mean(relative_errors)
    median_error = np.median(relative_errors)
    std_error = np.std(relative_errors)
    
    print(f"\n📊 STATISTIQUES DES ERREURS RELATIVES:")
    print(f"  Erreur maximale:  {max_error:.3f}%")
    print(f"  Erreur moyenne:   {mean_error:.3f}%")
    print(f"  Erreur médiane:   {median_error:.3f}%")
    print(f"  Écart-type:       {std_error:.3f}%")
    
    # Vérification du seuil de 5%
    points_under_5_percent = np.sum(relative_errors < 5.0)
    percentage_under_5 = (points_under_5_percent / len(relative_errors)) * 100
    
    print(f"\n🎯 ÉVALUATION DU SEUIL 5%:")
    print(f"  Points < 5% d'erreur: {points_under_5_percent}/{len(relative_errors)} ({percentage_under_5:.1f}%)")
    
    if max_error < 5.0:
        print(f"  ✅ EXCELLENT: Toutes les erreurs < 5%")
        print(f"  ✅ Généralisation locale exceptionnelle")
    elif mean_error < 5.0:
        print(f"  ✅ TRÈS BON: Erreur moyenne < 5%")
        print(f"  ⚠️  Quelques points dépassent 5% (max: {max_error:.3f}%)")
    else:
        print(f"  ❌ PROBLÉMATIQUE: Erreur moyenne > 5%")
        print(f"  ❌ Généralisation locale insuffisante")
    
    # Analyse détaillée des points
    print(f"\n📋 ANALYSE DÉTAILLÉE DES POINTS:")
    print(f"{'Gap Réel':<10} {'Gap Prédit':<12} {'Erreur Abs':<12} {'Erreur Rel':<12}")
    print("-" * 50)
    
    # Trier par erreur relative décroissante
    sorted_indices = np.argsort(relative_errors)[::-1]
    
    # Afficher les 10 premiers (pires erreurs)
    for i in sorted_indices[:min(10, len(sorted_indices))]:
        gap_real = y_true_range[i]
        gap_pred = y_pred_range[i]
        abs_error = abs(gap_pred - gap_real)
        rel_error = relative_errors[i]
        
        print(f"{gap_real:<10.4f} {gap_pred:<12.4f} {abs_error:<12.4f} {rel_error:<12.3f}%")
    
    # Créer un graphique spécifique pour cette plage
    create_local_analysis_plot(y_true_range, y_pred_range, relative_errors, range_min, range_max)
    
    return {
        'n_points': len(y_true_range),
        'max_error': max_error,
        'mean_error': mean_error,
        'median_error': median_error,
        'std_error': std_error,
        'points_under_5_percent': points_under_5_percent,
        'percentage_under_5': percentage_under_5,
        'all_under_5': max_error < 5.0
    }

def create_local_analysis_plot(y_true, y_pred, relative_errors, range_min, range_max):
    """Crée un graphique spécifique pour l'analyse locale."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Scatter plot pour la plage spécifique
    axes[0].scatter(y_true, y_pred, alpha=0.7, s=50, c='blue', edgecolors='black', linewidth=0.5)
    
    # Ligne parfaite
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
    
    axes[0].set_xlabel('Gap réel (µm)')
    axes[0].set_ylabel('Gap prédit (µm)')
    axes[0].set_title(f'Généralisation Locale [{range_min}-{range_max} µm]\n{len(y_true)} points')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axis('equal')
    
    # 2. Distribution des erreurs relatives
    axes[1].hist(relative_errors, bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='Seuil 5%')
    axes[1].axvline(x=np.mean(relative_errors), color='green', linestyle='-', linewidth=2, label=f'Moyenne: {np.mean(relative_errors):.2f}%')
    axes[1].set_xlabel('Erreur relative (%)')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution des Erreurs Relatives')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. Erreur relative vs Gap réel
    axes[2].scatter(y_true, relative_errors, alpha=0.7, s=50, c='red', edgecolors='black', linewidth=0.5)
    axes[2].axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='Seuil 5%')
    axes[2].axhline(y=np.mean(relative_errors), color='green', linestyle='-', linewidth=2, label=f'Moyenne: {np.mean(relative_errors):.2f}%')
    axes[2].set_xlabel('Gap réel (µm)')
    axes[2].set_ylabel('Erreur relative (%)')
    axes[2].set_title('Erreur Relative vs Gap Réel')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'../plots/local_generalization_analysis_{range_min}_{range_max}um.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Graphique d'analyse locale sauvegardé: ../plots/local_generalization_analysis_{range_min}_{range_max}um.png")

def main():
    """Fonction principale d'analyse."""
    
    print("="*60)
    print("ANALYSE DE LA GÉNÉRALISATION LOCALE [0.5-1.0 µm]")
    print("="*60)
    print("Calcul des erreurs relatives par rapport à la droite idéale")
    print("="*60)
    
    try:
        # Entraîner le modèle et obtenir les prédictions
        y_true, y_pred = train_and_predict()
        
        # Analyser la généralisation locale dans la plage 0.5-1.0 µm
        results = analyze_local_generalization(y_true, y_pred, range_min=0.5, range_max=1.0)
        
        # Sauvegarder les résultats
        if results:
            with open('../results/local_generalization_analysis.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n{'='*60}")
            print("RÉSUMÉ FINAL")
            print(f"{'='*60}")
            
            print(f"🎯 PLAGE ANALYSÉE: 0.5-1.0 µm")
            print(f"📊 POINTS ANALYSÉS: {results['n_points']}")
            print(f"📈 ERREUR MAXIMALE: {results['max_error']:.3f}%")
            print(f"📈 ERREUR MOYENNE: {results['mean_error']:.3f}%")
            print(f"📈 ERREUR MÉDIANE: {results['median_error']:.3f}%")
            
            if results['all_under_5']:
                print(f"✅ CONFIRMATION: Toutes les erreurs < 5%")
                print(f"✅ GÉNÉRALISATION LOCALE EXCEPTIONNELLE")
            else:
                print(f"⚠️  ATTENTION: {results['points_under_5_percent']}/{results['n_points']} points < 5%")
                print(f"⚠️  POURCENTAGE SOUS SEUIL: {results['percentage_under_5']:.1f}%")
            
            print(f"\n📋 Résultats sauvegardés: ../results/local_generalization_analysis.json")
        
    except Exception as e:
        print(f"❌ Erreur durant l'analyse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
