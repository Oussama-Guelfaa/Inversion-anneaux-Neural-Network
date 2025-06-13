#!/usr/bin/env python3
"""
Réentraînement du réseau de neurones avec le nouveau dataset fusionné
Auteur: Oussama GUELFAA
Date: 12 - 06 - 2025

Script pour réentraîner le modèle existant avec les nouvelles données du dataset fusionné,
incluant l'augmentation par interpolation et le bruit synthétique de 5%.
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import json
import time
import warnings
warnings.filterwarnings('ignore')

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
    """Dataset PyTorch pour les profils d'intensité."""
    
    def __init__(self, intensity_profiles, gap_values):
        self.intensity_profiles = torch.FloatTensor(intensity_profiles)
        self.gap_values = torch.FloatTensor(gap_values)
    
    def __len__(self):
        return len(self.intensity_profiles)
    
    def __getitem__(self, idx):
        return self.intensity_profiles[idx], self.gap_values[idx]

class RobustGapPredictor(nn.Module):
    """Modèle robuste pour prédiction du gap - Architecture identique au modèle existant."""
    
    def __init__(self, input_size=600, dropout_rate=0.2):
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

def load_dataset_from_folder(dataset_path="../data_generation/dataset"):
    """
    Charge les données depuis le dossier dataset fusionné.

    Cette fonction lit le fichier labels.csv pour obtenir les métadonnées,
    puis charge chaque fichier .mat correspondant pour extraire les profils
    d'intensité (variable 'ratio'). Les profils sont normalisés à 600 points.

    Args:
        dataset_path (str): Chemin vers le dossier dataset contenant labels.csv
                           et les fichiers .mat individuels

    Returns:
        tuple: (intensity_profiles, gap_values) où:
               - intensity_profiles: array (n_samples, 600) des profils
               - gap_values: array (n_samples,) des valeurs de gap en µm
    """
    print("🔄 Chargement des données depuis le dataset fusionné...")
    
    dataset_path = Path(dataset_path)
    labels_path = dataset_path / "labels.csv"
    
    if not labels_path.exists():
        print(f"❌ Erreur: Fichier labels.csv non trouvé dans {dataset_path}")
        print(f"   Vérifiez que le chemin est correct et que le fichier existe")
        return None, None
    
    # Charger les métadonnées
    labels_df = pd.read_csv(labels_path)
    print(f"📊 {len(labels_df)} échantillons trouvés dans labels.csv")
    
    intensity_profiles = []
    gap_values = []
    
    for _, row in labels_df.iterrows():
        gap_value = row['gap_um']
        
        # Construire le nom du fichier .mat correspondant
        mat_filename = f"gap_{gap_value:.4f}um_L_10.000um.mat"
        mat_path = dataset_path / mat_filename
        
        if mat_path.exists():
            try:
                # Charger le fichier .mat
                mat_data = sio.loadmat(mat_path)
                
                # Extraire le profil d'intensité (variable 'ratio')
                if 'ratio' in mat_data:
                    profile = mat_data['ratio'].flatten()
                    
                    # Tronquer à 600 points si nécessaire
                    if len(profile) > 600:
                        profile = profile[:600]
                    elif len(profile) < 600:
                        # Interpoler si moins de 600 points
                        x_old = np.linspace(0, 1, len(profile))
                        x_new = np.linspace(0, 1, 600)
                        f = interp1d(x_old, profile, kind='linear')
                        profile = f(x_new)
                    
                    intensity_profiles.append(profile)
                    gap_values.append(gap_value)
                else:
                    print(f"⚠️ Variable 'ratio' non trouvée dans {mat_filename}")
                    
            except Exception as e:
                print(f"⚠️ Erreur lors du chargement de {mat_filename}: {e}")
        else:
            print(f"⚠️ Fichier {mat_filename} non trouvé")
    
    intensity_profiles = np.array(intensity_profiles)
    gap_values = np.array(gap_values)
    
    print(f"✅ {len(intensity_profiles)} échantillons chargés avec succès")
    print(f"   Forme des profils: {intensity_profiles.shape}")
    print(f"   Plage de gap: {gap_values.min():.3f} à {gap_values.max():.3f} µm")
    print(f"   Valeurs manquantes: {np.isnan(intensity_profiles).sum()} points")
    print(f"   Profils valides: {len(intensity_profiles)} / {len(labels_df)} fichiers")
    
    return intensity_profiles, gap_values

def augment_data_by_interpolation(X, y, factor=2):
    """
    Augmente les données par interpolation entre échantillons adjacents.
    
    Args:
        X (np.array): Profils d'intensité
        y (np.array): Valeurs de gap
        factor (int): Facteur d'augmentation (2 = doubler le dataset)
        
    Returns:
        tuple: (X_augmented, y_augmented)
    """
    print(f"🔄 Augmentation des données par interpolation (facteur {factor})...")
    
    # Trier par valeur de gap pour interpolation cohérente
    sort_indices = np.argsort(y)
    X_sorted = X[sort_indices]
    y_sorted = y[sort_indices]
    
    X_augmented = [X_sorted]
    y_augmented = [y_sorted]
    
    # Générer des échantillons interpolés
    for i in range(factor - 1):
        X_interp = []
        y_interp = []
        
        for j in range(len(X_sorted) - 1):
            # Interpolation linéaire entre échantillons adjacents
            alpha = (i + 1) / factor
            
            profile_interp = (1 - alpha) * X_sorted[j] + alpha * X_sorted[j + 1]
            gap_interp = (1 - alpha) * y_sorted[j] + alpha * y_sorted[j + 1]
            
            X_interp.append(profile_interp)
            y_interp.append(gap_interp)
        
        X_augmented.append(np.array(X_interp))
        y_augmented.append(np.array(y_interp))
    
    # Concaténer tous les échantillons
    X_final = np.concatenate(X_augmented, axis=0)
    y_final = np.concatenate(y_augmented, axis=0)
    
    print(f"✅ Augmentation terminée: {len(X)} → {len(X_final)} échantillons")
    print(f"   Facteur d'augmentation réalisé: {len(X_final)/len(X):.2f}x")
    print(f"   Échantillons interpolés générés: {len(X_final) - len(X)}")
    
    return X_final, y_final

def add_gaussian_noise(X, noise_level_percent=5):
    """
    Ajoute du bruit gaussien proportionnel au signal.
    
    Args:
        X (np.array): Données d'entrée
        noise_level_percent (float): Niveau de bruit en pourcentage
        
    Returns:
        np.array: Données avec bruit ajouté
    """
    if noise_level_percent == 0:
        return X.copy()
    
    # Calculer l'écart-type du signal pour chaque échantillon
    signal_std = np.std(X, axis=1, keepdims=True)
    
    # Générer le bruit proportionnel
    noise_std = (noise_level_percent / 100.0) * signal_std
    noise = np.random.normal(0, noise_std, X.shape)
    
    X_noisy = X + noise
    
    print(f"🔊 Bruit {noise_level_percent}% ajouté - SNR moyen: {1/(noise_level_percent/100):.1f}")
    print(f"   Écart-type du bruit: {np.mean(noise_std):.6f}")
    
    return X_noisy

def prepare_stratified_splits(X, y, test_size=0.2, random_state=42):
    """
    Divise les données en train/test avec stratification par plage de gap.
    
    Args:
        X (np.array): Profils d'intensité
        y (np.array): Valeurs de gap
        test_size (float): Proportion du jeu de test
        random_state (int): Graine aléatoire
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"📊 Division stratifiée des données (test: {test_size*100:.0f}%)...")
    
    # Créer des bins pour stratification
    n_bins = 10
    gap_bins = pd.cut(y, bins=n_bins, labels=False)
    
    # Division stratifiée
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=gap_bins
    )
    
    print(f"   Train: {len(X_train)} échantillons")
    print(f"   Test: {len(X_test)} échantillons")
    
    # Vérifier la distribution
    print(f"   Distribution train - Gap: {y_train.min():.3f} à {y_train.max():.3f} µm")
    print(f"   Distribution test - Gap: {y_test.min():.3f} à {y_test.max():.3f} µm")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_val, y_val, noise_level=5):
    """
    Entraîne le modèle avec les nouvelles données.

    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        noise_level: Niveau de bruit en pourcentage

    Returns:
        tuple: (model, scaler, history, training_time)
    """
    print(f"\n🚀 ENTRAÎNEMENT AVEC {noise_level}% DE BRUIT")

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

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Modèle et optimisation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustGapPredictor(input_size=X_train.shape[1]).to(device)

    learning_rate = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    early_stopping = EarlyStopping(patience=25)

    # Historique d'entraînement
    history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}

    max_epochs = 150
    print(f"📈 Entraînement sur {device}")

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
            print(f"   Epoch {epoch+1:3d}: Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, LR={optimizer.param_groups[0]['lr']:.2e}")

        if early_stopping(val_loss, model):
            print(f"   ⏹️ Early stopping à l'époque {epoch+1}")
            break

    training_time = time.time() - start_time

    # Sauvegarder le modèle réentraîné avec facteur 3
    model_path = "models/model_retrained_5percent_factor3.pth"
    torch.save(model.state_dict(), model_path)

    print(f"✅ Entraînement terminé en {training_time:.1f}s")
    print(f"   Performance finale: R² = {val_r2:.4f}")

    return model, scaler, history, training_time

def evaluate_model(model, scaler, X_test, y_test):
    """
    Évalue le modèle sur l'ensemble de test.

    Args:
        model: Modèle entraîné
        scaler: Normalisateur
        X_test, y_test: Données de test

    Returns:
        dict: Métriques d'évaluation
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

    # Analyse par plage de gap
    gap_ranges = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    range_metrics = {}

    for gap_min, gap_max in gap_ranges:
        mask = (y_test >= gap_min) & (y_test < gap_max)
        if np.sum(mask) > 0:
            r2_range = r2_score(y_test[mask], y_pred[mask])
            rmse_range = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            range_metrics[f"{gap_min}-{gap_max}µm"] = {
                'r2': r2_range,
                'rmse': rmse_range,
                'n_samples': np.sum(mask)
            }

    metrics = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'predictions': y_pred,
        'range_metrics': range_metrics
    }

    return metrics

def create_analysis_plots(y_test, y_pred, history, range_metrics):
    """
    Crée les graphiques d'analyse des résultats.

    Args:
        y_test: Valeurs réelles
        y_pred: Prédictions
        history: Historique d'entraînement
        range_metrics: Métriques par plage
    """
    print("📊 Génération des graphiques d'analyse...")

    plt.figure(figsize=(20, 12))

    # 1. Courbes d'entraînement
    plt.subplot(2, 4, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Courbes de Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 4, 2)
    plt.plot(history['train_r2'], label='Train R²', color='blue')
    plt.plot(history['val_r2'], label='Val R²', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Courbes de R²')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Scatter plot prédictions vs réalité
    plt.subplot(2, 4, 3)
    plt.scatter(y_test, y_pred, alpha=0.6, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Gap Réel (µm)')
    plt.ylabel('Gap Prédit (µm)')
    plt.title('Prédictions vs Réalité')
    plt.grid(True, alpha=0.3)

    # 3. Erreurs absolues
    plt.subplot(2, 4, 4)
    errors = np.abs(y_pred - y_test)
    plt.scatter(y_test, errors, alpha=0.6, s=20)
    plt.xlabel('Gap Réel (µm)')
    plt.ylabel('Erreur Absolue (µm)')
    plt.title('Erreurs Absolues')
    plt.grid(True, alpha=0.3)

    # 4. Distribution des erreurs
    plt.subplot(2, 4, 5)
    plt.hist(y_pred - y_test, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Erreur (µm)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des Erreurs')
    plt.grid(True, alpha=0.3)

    # 5. Performance par plage
    plt.subplot(2, 4, 6)
    ranges = list(range_metrics.keys())
    r2_values = [range_metrics[r]['r2'] for r in ranges]
    plt.bar(ranges, r2_values, alpha=0.7)
    plt.ylabel('R² Score')
    plt.title('Performance par Plage de Gap')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 6. RMSE par plage
    plt.subplot(2, 4, 7)
    rmse_values = [range_metrics[r]['rmse'] for r in ranges]
    plt.bar(ranges, rmse_values, alpha=0.7, color='orange')
    plt.ylabel('RMSE (µm)')
    plt.title('RMSE par Plage de Gap')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 7. Zone critique [1.75-2.00 µm]
    plt.subplot(2, 4, 8)
    critical_mask = (y_test >= 1.75) & (y_test <= 2.00)
    if np.sum(critical_mask) > 0:
        plt.scatter(y_test[critical_mask], y_pred[critical_mask], alpha=0.8, s=30, color='red')
        plt.plot([1.75, 2.00], [1.75, 2.00], 'k--', lw=2)
        plt.xlabel('Gap Réel (µm)')
        plt.ylabel('Gap Prédit (µm)')
        plt.title('Zone Critique [1.75-2.00 µm]')
        plt.grid(True, alpha=0.3)

        # Calculer R² pour la zone critique
        if np.sum(critical_mask) > 1:
            r2_critical = r2_score(y_test[critical_mask], y_pred[critical_mask])
            plt.text(0.05, 0.95, f'R² = {r2_critical:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('plots/retrained_model_analysis_factor3.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Graphiques sauvegardés dans plots/retrained_model_analysis_factor3.png")

def save_results(metrics, y_test, y_pred, training_time):
    """
    Sauvegarde les résultats du réentraînement.

    Args:
        metrics: Métriques d'évaluation
        y_test: Valeurs réelles
        y_pred: Prédictions
        training_time: Temps d'entraînement
    """
    print("💾 Sauvegarde des résultats...")

    # Résumé des performances (conversion en types Python natifs pour JSON)
    range_performance = {}
    for range_name, range_data in metrics['range_metrics'].items():
        range_performance[range_name] = {
            'r2': float(range_data['r2']),
            'rmse': float(range_data['rmse']),
            'n_samples': int(range_data['n_samples'])
        }

    summary = {
        'model_type': 'RobustGapPredictor_Retrained',
        'dataset': 'dataset_merged (0.005-3.000µm)',
        'augmentation': 'interpolation_factor_3',
        'noise_level': '5%',
        'training_time_s': float(training_time),
        'performance': {
            'r2_score': float(metrics['r2']),
            'rmse_um': float(metrics['rmse']),
            'mae_um': float(metrics['mae'])
        },
        'range_performance': range_performance
    }

    # Sauvegarder le résumé avec facteur 3
    with open('results/retrained_model_summary_factor3.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Sauvegarder les prédictions détaillées avec facteur 3
    predictions_df = pd.DataFrame({
        'gap_true': y_test,
        'gap_predicted': y_pred,
        'error': y_pred - y_test,
        'absolute_error': np.abs(y_pred - y_test)
    })
    predictions_df.to_csv('results/retrained_predictions_factor3.csv', index=False)

    print("✅ Résultats sauvegardés!")

def main():
    """
    Fonction principale de réentraînement.
    """
    print("🔬 RÉENTRAÎNEMENT AVEC NOUVEAU DATASET FUSIONNÉ")
    print("=" * 60)

    # 1. Charger les données du dataset fusionné
    intensity_profiles, gap_values = load_dataset_from_folder("../data_generation/dataset")

    if intensity_profiles is None:
        print("❌ Échec du chargement des données. Arrêt du programme.")
        return

    # 2. Augmentation par interpolation avec facteur 3
    print("\n📈 AUGMENTATION DES DONNÉES")
    X_augmented, y_augmented = augment_data_by_interpolation(
        intensity_profiles, gap_values, factor=3
    )

    # 3. Division stratifiée des données (80% train, 20% test)
    print("\n📊 DIVISION DES DONNÉES")
    X_train_full, X_test, y_train_full, y_test = prepare_stratified_splits(
        X_augmented, y_augmented, test_size=0.2, random_state=42
    )

    # 4. Division train/validation (80% train, 20% validation du train_full)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    print(f"   Division finale:")
    print(f"   Train: {len(X_train)} échantillons")
    print(f"   Validation: {len(X_val)} échantillons")
    print(f"   Test: {len(X_test)} échantillons")

    # 5. Entraînement avec bruit de 5%
    print("\n🚀 ENTRAÎNEMENT DU MODÈLE")
    model, scaler, history, training_time = train_model(
        X_train, y_train, X_val, y_val, noise_level=5
    )

    # 6. Évaluation sur le jeu de test
    print("\n📊 ÉVALUATION SUR LE JEU DE TEST")
    metrics = evaluate_model(model, scaler, X_test, y_test)

    print(f"\n🎯 RÉSULTATS FINAUX:")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f} µm")
    print(f"   MAE: {metrics['mae']:.4f} µm")
    print(f"   Temps d'entraînement: {training_time:.1f}s")
    print(f"   Échantillons de test: {len(y_test)}")

    print(f"\n📈 PERFORMANCE PAR PLAGE:")
    for range_name, range_metrics in metrics['range_metrics'].items():
        print(f"   {range_name}: R²={range_metrics['r2']:.4f}, "
              f"RMSE={range_metrics['rmse']:.4f} µm "
              f"({range_metrics['n_samples']} échantillons)")

    # 7. Analyse de la zone critique [1.75-2.00 µm]
    critical_mask = (y_test >= 1.75) & (y_test <= 2.00)
    if np.sum(critical_mask) > 1:
        y_pred = metrics['predictions']
        r2_critical = r2_score(y_test[critical_mask], y_pred[critical_mask])
        rmse_critical = np.sqrt(mean_squared_error(y_test[critical_mask], y_pred[critical_mask]))
        print(f"\n🎯 ZONE CRITIQUE [1.75-2.00 µm]:")
        print(f"   R² Score: {r2_critical:.4f}")
        print(f"   RMSE: {rmse_critical:.4f} µm")
        print(f"   Échantillons: {np.sum(critical_mask)}")

    # 8. Génération des graphiques
    print("\n📊 GÉNÉRATION DES ANALYSES VISUELLES")
    create_analysis_plots(y_test, metrics['predictions'], history, metrics['range_metrics'])

    # 9. Sauvegarde des résultats
    save_results(metrics, y_test, metrics['predictions'], training_time)

    # 10. Comparaison avec le modèle précédent
    print(f"\n📋 COMPARAISON AVEC LE MODÈLE PRÉCÉDENT:")
    print(f"   Nouveau dataset: {len(intensity_profiles)} échantillons originaux")
    print(f"   Après augmentation (facteur 3): {len(X_augmented)} échantillons")
    print(f"   Amélioration vs facteur 2: +{len(X_augmented) - (len(intensity_profiles)*2)} échantillons")
    print(f"   Plage étendue: 0.005 - 3.000 µm")
    print(f"   Focus zone critique: [1.75-2.00 µm]")

    if metrics['r2'] > 0.8:
        print(f"✅ OBJECTIF ATTEINT: R² = {metrics['r2']:.4f} > 0.8")
    else:
        print(f"⚠️ OBJECTIF NON ATTEINT: R² = {metrics['r2']:.4f} < 0.8")

    print("\n🏁 Réentraînement terminé avec succès!")

    return model, scaler, metrics

if __name__ == "__main__":
    # Créer les dossiers nécessaires
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Lancer le réentraînement
    model, scaler, metrics = main()
