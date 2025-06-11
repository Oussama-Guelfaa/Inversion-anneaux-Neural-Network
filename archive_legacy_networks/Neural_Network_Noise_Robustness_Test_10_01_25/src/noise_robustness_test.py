#!/usr/bin/env python3
"""
Test de robustesse au bruit pour le modèle de prédiction du gap

Ce script évalue la performance du modèle face à différents niveaux de bruit gaussien
pour déterminer les conditions optimales de fonctionnement en environnement réel.

Auteur: Oussama GUELFAA
Date: 10 - 01 - 2025
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import json
import time
from collections import defaultdict

# Configuration pour reproductibilité
torch.manual_seed(42)
np.random.seed(42)

class EarlyStopping:
    """Early stopping pour éviter l'overfitting."""
    
    def __init__(self, patience=20, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class IntensityDataset(Dataset):
    """Dataset PyTorch pour les profils d'intensité avec support du bruit."""
    
    def __init__(self, intensity_profiles, gap_values):
        self.intensity_profiles = torch.FloatTensor(intensity_profiles)
        self.gap_values = torch.FloatTensor(gap_values)
    
    def __len__(self):
        return len(self.intensity_profiles)
    
    def __getitem__(self, idx):
        return self.intensity_profiles[idx], self.gap_values[idx]

class RobustGapPredictor(nn.Module):
    """
    Modèle robuste pour prédiction du gap avec régularisation.
    
    Architecture optimisée pour la robustesse au bruit avec dropout et batch normalization.
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

def load_dataset():
    """
    Charge le dataset complet depuis dataset_small_particle.
    
    Returns:
        tuple: (X, y) où X sont les profils d'intensité et y les valeurs de gap
    """
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
    """
    Ajoute du bruit gaussien proportionnel au signal.
    
    Args:
        X (np.ndarray): Données originales
        noise_level_percent (float): Niveau de bruit en pourcentage (0-100)
        
    Returns:
        np.ndarray: Données avec bruit ajouté
    """
    if noise_level_percent == 0:
        return X.copy()
    
    # Calculer l'écart-type du signal pour chaque échantillon
    signal_std = np.std(X, axis=1, keepdims=True)
    
    # Générer le bruit proportionnel
    noise_std = (noise_level_percent / 100.0) * signal_std
    noise = np.random.normal(0, noise_std, X.shape)
    
    X_noisy = X + noise
    
    print(f"Bruit {noise_level_percent}% ajouté - SNR moyen: {1/(noise_level_percent/100):.1f}")
    
    return X_noisy

def prepare_data_splits(X, y, test_size=0.2, val_size=0.2):
    """
    Divise les données en train/validation/test.
    
    Args:
        X, y: Données complètes
        test_size: Proportion pour le test
        val_size: Proportion pour la validation (du reste après test)
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Première division: train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=None
    )
    
    # Deuxième division: train / val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42
    )
    
    print(f"Division des données:")
    print(f"  Train: {X_train.shape[0]} échantillons ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Validation: {X_val.shape[0]} échantillons ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]} échantillons ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model_with_noise(X_train, y_train, X_val, y_val, noise_level, 
                          max_epochs=200, batch_size=16, learning_rate=0.001):
    """
    Entraîne un modèle avec un niveau de bruit spécifique.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        noise_level: Niveau de bruit en pourcentage
        
    Returns:
        tuple: (model, history, training_time)
    """
    print(f"\n=== ENTRAÎNEMENT AVEC {noise_level}% DE BRUIT ===")
    
    start_time = time.time()
    
    # Ajouter du bruit aux données d'entraînement UNIQUEMENT
    X_train_noisy = add_gaussian_noise(X_train, noise_level)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_noisy)
    X_val_scaled = scaler.transform(X_val)  # Validation SANS bruit
    
    # Datasets et DataLoaders
    train_dataset = IntensityDataset(X_train_scaled, y_train)
    val_dataset = IntensityDataset(X_val_scaled, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Modèle et optimisation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustGapPredictor(input_size=X_train.shape[1]).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=20)
    
    # Historique d'entraînement
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': []
    }
    
    print(f"Entraînement sur {device}")
    
    for epoch in range(max_epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        train_r2 = r2_score(train_targets, train_predictions)
        
        # Phase de validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_r2 = r2_score(val_targets, val_predictions)
        
        # Mise à jour historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        
        # Scheduler et early stopping
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, "
                  f"Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        if early_stopping(val_loss, model):
            print(f"  Early stopping à l'époque {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    print(f"Entraînement terminé en {training_time:.1f}s")
    print(f"Performance finale: R² = {val_r2:.4f}")
    
    return model, scaler, history, training_time

def evaluate_model(model, scaler, X_test, y_test):
    """
    Évalue un modèle sur l'ensemble de test.
    
    Args:
        model: Modèle entraîné
        scaler: Scaler utilisé
        X_test, y_test: Données de test
        
    Returns:
        dict: Métriques de performance
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Normalisation et prédiction
    X_test_scaled = scaler.transform(X_test)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_pred = model(X_tensor).squeeze().cpu().numpy()
    
    # Calcul des métriques
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'predictions': y_pred
    }
    
    return metrics

def save_complete_results(results, noise_levels, y_test):
    """Sauvegarde tous les résultats du test de robustesse."""

    print("\n=== SAUVEGARDE DES RÉSULTATS ===")

    # Résumé des performances
    summary = {
        'test_type': 'noise_robustness',
        'noise_levels': noise_levels,
        'n_test_samples': len(y_test),
        'results_by_noise': {}
    }

    # Tableau de performance
    performance_data = []

    for noise_level in noise_levels:
        result = results[noise_level]
        metrics = result['metrics']

        summary['results_by_noise'][noise_level] = {
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'training_time': result['training_time'],
            'epochs': result['final_epoch']
        }

        performance_data.append({
            'noise_level': noise_level,
            'r2_score': metrics['r2'],
            'rmse_um': metrics['rmse'],
            'mae_um': metrics['mae'],
            'mse': metrics['mse'],
            'training_time_s': result['training_time'],
            'epochs_to_convergence': result['final_epoch']
        })

    # Sauvegarder le résumé JSON
    with open('../results/noise_robustness_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Sauvegarder le tableau de performance
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv('../results/performance_by_noise_level.csv', index=False)

    # Sauvegarder les prédictions détaillées
    for noise_level in noise_levels:
        predictions = results[noise_level]['metrics']['predictions']
        pred_df = pd.DataFrame({
            'gap_true': y_test,
            'gap_predicted': predictions,
            'error': predictions - y_test,
            'absolute_error': np.abs(predictions - y_test)
        })
        pred_df.to_csv(f'../results/predictions_noise_{noise_level}percent.csv', index=False)

    print(f"Résultats sauvegardés:")
    print(f"  - Résumé: ../results/noise_robustness_summary.json")
    print(f"  - Performance: ../results/performance_by_noise_level.csv")
    print(f"  - Prédictions: ../results/predictions_noise_X%.csv")

def create_robustness_plots(results, noise_levels, y_test):
    """Crée les visualisations de robustesse au bruit."""

    print("\n=== GÉNÉRATION DES GRAPHIQUES ===")

    # 1. Courbe principale: R² vs Niveau de bruit
    plt.figure(figsize=(15, 12))

    # Subplot 1: R² vs Noise Level
    plt.subplot(2, 3, 1)
    r2_scores = [results[noise]['metrics']['r2'] for noise in noise_levels]
    plt.plot(noise_levels, r2_scores, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Objectif R² = 0.8')
    plt.xlabel('Niveau de bruit (%)')
    plt.ylabel('R² Score')
    plt.title('Performance vs Niveau de Bruit')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 2: RMSE vs Noise Level
    plt.subplot(2, 3, 2)
    rmse_scores = [results[noise]['metrics']['rmse'] for noise in noise_levels]
    plt.plot(noise_levels, rmse_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Niveau de bruit (%)')
    plt.ylabel('RMSE (µm)')
    plt.title('Erreur RMSE vs Niveau de Bruit')
    plt.grid(True, alpha=0.3)

    # Subplot 3: Temps d'entraînement vs Noise Level
    plt.subplot(2, 3, 3)
    training_times = [results[noise]['training_time'] for noise in noise_levels]
    plt.plot(noise_levels, training_times, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Niveau de bruit (%)')
    plt.ylabel('Temps d\'entraînement (s)')
    plt.title('Temps de Convergence vs Bruit')
    plt.grid(True, alpha=0.3)

    # Subplot 4: Époques de convergence
    plt.subplot(2, 3, 4)
    epochs = [results[noise]['final_epoch'] for noise in noise_levels]
    plt.plot(noise_levels, epochs, 'mo-', linewidth=2, markersize=8)
    plt.xlabel('Niveau de bruit (%)')
    plt.ylabel('Époques de convergence')
    plt.title('Convergence vs Niveau de Bruit')
    plt.grid(True, alpha=0.3)

    # Subplot 5: Comparaison R² train vs val (pour 0%, 5%, 20%)
    plt.subplot(2, 3, 5)
    selected_noise = [0, 5, 20]
    for noise in selected_noise:
        if noise in results:
            history = results[noise]['history']
            epochs_range = range(1, len(history['val_r2']) + 1)
            plt.plot(epochs_range, history['val_r2'],
                    label=f'{noise}% bruit', linewidth=2)
    plt.xlabel('Époque')
    plt.ylabel('R² Validation')
    plt.title('Convergence R² par Niveau de Bruit')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 6: Distribution des erreurs pour différents niveaux
    plt.subplot(2, 3, 6)
    selected_noise = [0, 5, 10, 20]
    for noise in selected_noise:
        if noise in results:
            predictions = results[noise]['metrics']['predictions']
            errors = predictions - y_test
            plt.hist(errors, bins=20, alpha=0.6, label=f'{noise}% bruit')
    plt.xlabel('Erreur de prédiction (µm)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des Erreurs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/noise_robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Graphique détaillé des prédictions pour chaque niveau
    create_predictions_comparison(results, noise_levels, y_test)

    print("Graphiques sauvegardés:")
    print("  - Vue d'ensemble: ../plots/noise_robustness_analysis.png")
    print("  - Prédictions détaillées: ../plots/predictions_by_noise.png")

def create_predictions_comparison(results, noise_levels, y_test):
    """Crée un graphique comparatif des prédictions pour chaque niveau de bruit."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, noise_level in enumerate(noise_levels):
        if i < len(axes):
            ax = axes[i]

            predictions = results[noise_level]['metrics']['predictions']

            # Scatter plot
            ax.scatter(y_test, predictions, alpha=0.6, s=20)

            # Ligne parfaite
            min_val = min(y_test.min(), predictions.min())
            max_val = max(y_test.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

            # Métriques
            r2 = results[noise_level]['metrics']['r2']
            rmse = results[noise_level]['metrics']['rmse']

            ax.set_xlabel('Gap réel (µm)')
            ax.set_ylabel('Gap prédit (µm)')
            ax.set_title(f'Bruit {noise_level}%\nR² = {r2:.3f}, RMSE = {rmse:.4f}')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/predictions_by_noise.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_robustness_results(results, noise_levels):
    """Analyse et résume les résultats de robustesse."""

    print("\n" + "="*60)
    print("ANALYSE DE LA ROBUSTESSE AU BRUIT")
    print("="*60)

    # Tableau de performance
    print(f"\n📊 PERFORMANCE PAR NIVEAU DE BRUIT:")
    print(f"{'Bruit':<8} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'Temps':<8} {'Époques':<8}")
    print("-" * 60)

    for noise_level in noise_levels:
        result = results[noise_level]
        metrics = result['metrics']

        print(f"{noise_level:>5}%   "
              f"{metrics['r2']:>6.3f}   "
              f"{metrics['rmse']:>8.4f}   "
              f"{metrics['mae']:>8.4f}   "
              f"{result['training_time']:>6.1f}s  "
              f"{result['final_epoch']:>6d}")

    # Analyse des seuils
    print(f"\n🎯 ANALYSE DES SEUILS:")

    # Trouver le seuil pour R² > 0.8
    threshold_80 = None
    for noise_level in noise_levels:
        if results[noise_level]['metrics']['r2'] >= 0.8:
            threshold_80 = noise_level
        else:
            break

    if threshold_80 is not None:
        print(f"  ✅ R² > 0.8 maintenu jusqu'à {threshold_80}% de bruit")
    else:
        print(f"  ❌ R² < 0.8 dès le premier niveau de bruit testé")

    # Dégradation relative
    r2_clean = results[0]['metrics']['r2']
    print(f"\n📉 DÉGRADATION RELATIVE (vs 0% bruit):")

    for noise_level in noise_levels[1:]:  # Skip 0%
        r2_noisy = results[noise_level]['metrics']['r2']
        degradation = (r2_clean - r2_noisy) / r2_clean * 100
        print(f"  {noise_level:>2}% bruit: -{degradation:>5.1f}% de performance")

    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")

    if threshold_80 and threshold_80 >= 5:
        print(f"  ✅ Modèle robuste - Tolérance jusqu'à {threshold_80}% de bruit")
        print(f"  ✅ Acquisition réelle: SNR > {100/threshold_80:.1f} recommandé")
    elif threshold_80 and threshold_80 >= 2:
        print(f"  ⚠️  Robustesse modérée - Attention aux conditions d'acquisition")
        print(f"  ⚠️  SNR > {100/threshold_80:.1f} requis pour performance optimale")
    else:
        print(f"  ❌ Robustesse insuffisante - Amélioration du modèle nécessaire")
        print(f"  ❌ Conditions d'acquisition très strictes requises")

    print(f"\n🔬 CONCLUSIONS:")
    print(f"  • Modèle testé sur {len(noise_levels)} niveaux de bruit")
    print(f"  • Performance de référence: R² = {r2_clean:.4f}")
    print(f"  • Seuil de tolérance identifié: {threshold_80}% de bruit")
    print(f"  • Recommandations pratiques établies")

if __name__ == "__main__":
    print("=== TEST DE ROBUSTESSE AU BRUIT ===")
    print("Évaluation de la performance du modèle face à différents niveaux de bruit\n")
    
    # Créer les dossiers de sortie
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../plots", exist_ok=True)
    os.makedirs("../results", exist_ok=True)
    
    # Niveaux de bruit à tester
    noise_levels = [0, 1, 2, 5, 10, 20]
    
    try:
        # 1. Charger les données
        X, y = load_dataset()
        
        # 2. Diviser les données
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(X, y)
        
        # 3. Tester chaque niveau de bruit
        results = {}
        
        for noise_level in noise_levels:
            print(f"\n{'='*60}")
            print(f"TEST AVEC {noise_level}% DE BRUIT")
            print(f"{'='*60}")
            
            # Entraîner le modèle
            model, scaler, history, training_time = train_model_with_noise(
                X_train, y_train, X_val, y_val, noise_level
            )
            
            # Évaluer sur le test
            metrics = evaluate_model(model, scaler, X_test, y_test)
            
            # Sauvegarder les résultats
            results[noise_level] = {
                'metrics': metrics,
                'history': history,
                'training_time': training_time,
                'final_epoch': len(history['train_loss'])
            }
            
            # Sauvegarder le modèle
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'noise_level': noise_level,
                'metrics': metrics
            }, f'../models/model_noise_{noise_level}percent.pth')
            
            print(f"Résultats {noise_level}% bruit: R² = {metrics['r2']:.4f}, "
                  f"RMSE = {metrics['rmse']:.4f} µm")
        
        # 4. Sauvegarder les résultats complets
        save_complete_results(results, noise_levels, y_test)

        # 5. Générer les visualisations
        create_robustness_plots(results, noise_levels, y_test)

        # 6. Analyser et résumer
        analyze_robustness_results(results, noise_levels)

        print(f"\n=== TEST DE ROBUSTESSE TERMINÉ ===")
        print(f"📊 Modèles sauvegardés: ../models/")
        print(f"📈 Graphiques générés: ../plots/")
        print(f"📋 Résultats détaillés: ../results/")

    except Exception as e:
        print(f"Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()


