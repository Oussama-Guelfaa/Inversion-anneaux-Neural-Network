#!/usr/bin/env python3
"""
Expérience avec données réduites - Impact de la quantité de données

Ce script réentraîne le même réseau de neurones avec moins de données pour évaluer
l'impact de la quantité de données sur la capacité de généralisation.

Configuration:
- 300 exemples pour l'entraînement
- 100 exemples pour le test
- Niveau de bruit fixé à 5%

Auteur: Oussama GUELFAA
Date: 11 - 06 - 2025
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
    """Dataset PyTorch pour les profils d'intensité."""
    
    def __init__(self, intensity_profiles, gap_values):
        self.intensity_profiles = torch.FloatTensor(intensity_profiles)
        self.gap_values = torch.FloatTensor(gap_values)
    
    def __len__(self):
        return len(self.intensity_profiles)
    
    def __getitem__(self, idx):
        return self.intensity_profiles[idx], self.gap_values[idx]

class RobustGapPredictor(nn.Module):
    """
    Modèle robuste pour prédiction du gap - Architecture identique à l'expérience originale.
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

def prepare_reduced_data(X, y, n_train=300, n_test=100):
    """
    Prépare les données avec la quantité réduite spécifiée.
    
    Args:
        X, y: Données complètes
        n_train: Nombre d'échantillons pour l'entraînement
        n_test: Nombre d'échantillons pour le test
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"\n=== PRÉPARATION DONNÉES RÉDUITES ===")
    print(f"Configuration demandée:")
    print(f"  - Entraînement: {n_train} échantillons")
    print(f"  - Test: {n_test} échantillons")
    print(f"  - Total utilisé: {n_train + n_test} / {len(X)} disponibles")
    
    # Vérifier qu'on a assez de données
    total_needed = n_train + n_test
    if total_needed > len(X):
        raise ValueError(f"Pas assez de données: {total_needed} demandés, {len(X)} disponibles")
    
    # Sélectionner aléatoirement les échantillons
    indices = np.random.choice(len(X), size=total_needed, replace=False)
    
    # Diviser en train/test
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    print(f"\nDonnées préparées:")
    print(f"  Train: X{X_train.shape}, y{y_train.shape}")
    print(f"  Test: X{X_test.shape}, y{y_test.shape}")
    print(f"  Gap range train: {y_train.min():.4f} - {y_train.max():.4f} µm")
    print(f"  Gap range test: {y_test.min():.4f} - {y_test.max():.4f} µm")
    
    return X_train, X_test, y_train, y_test

def train_model_reduced_data(X_train, y_train, X_test, y_test, noise_level=5):
    """
    Entraîne le modèle avec les données réduites et le niveau de bruit spécifié.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        noise_level: Niveau de bruit en pourcentage
        
    Returns:
        tuple: (model, scaler, history, predictions)
    """
    print(f"\n=== ENTRAÎNEMENT AVEC DONNÉES RÉDUITES ===")
    print(f"Niveau de bruit: {noise_level}%")
    
    start_time = time.time()
    
    # Ajouter du bruit aux données d'entraînement
    X_train_noisy = add_gaussian_noise(X_train, noise_level)
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_noisy)
    X_test_scaled = scaler.transform(X_test)  # Test sans bruit
    
    # Datasets et DataLoaders
    train_dataset = IntensityDataset(X_train_scaled, y_train)
    test_dataset = IntensityDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Modèle et optimisation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustGapPredictor(input_size=X_train.shape[1]).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=20)
    
    # Historique d'entraînement
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_r2': [],
        'test_r2': []
    }
    
    print(f"Entraînement sur {device}")
    print(f"Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
    
    max_epochs = 200
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
        
        # Phase de test
        model.eval()
        test_loss = 0.0
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                test_loss += loss.item()
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(batch_y.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_r2 = r2_score(test_targets, test_predictions)
        
        # Mise à jour historique
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_r2'].append(train_r2)
        history['test_r2'].append(test_r2)
        
        # Scheduler et early stopping
        scheduler.step(test_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, "
                  f"Train Loss={train_loss:.6f}, Test Loss={test_loss:.6f}")
        
        if early_stopping(test_loss, model):
            print(f"  Early stopping à l'époque {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    print(f"Entraînement terminé en {training_time:.1f}s")
    print(f"Performance finale: R² = {test_r2:.4f}")
    
    # Prédictions finales sur le test
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        final_predictions = model(X_test_tensor).squeeze().cpu().numpy()
    
    return model, scaler, history, final_predictions

def create_scatter_plot(y_true, y_pred, save_path="reduced_data_scatter_plot.png"):
    """
    Crée le scatter plot demandé : Gap réel vs Gap prédit.
    
    Args:
        y_true: Valeurs réelles de gap
        y_pred: Valeurs prédites de gap
        save_path: Chemin de sauvegarde
    """
    print(f"\n=== GÉNÉRATION DU SCATTER PLOT ===")
    
    # Calculer les métriques
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Créer le graphique
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.7, s=50, c='blue', edgecolors='black', linewidth=0.5)
    
    # Ligne parfaite y=x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
    
    # Mise en forme
    plt.xlabel('Gap réel (µm)', fontsize=12, fontweight='bold')
    plt.ylabel('Gap prédit (µm)', fontsize=12, fontweight='bold')
    plt.title('Impact de la Quantité de Données sur la Généralisation\n' + 
              f'300 échantillons d\'entraînement, 100 de test, Bruit 5%', 
              fontsize=14, fontweight='bold')
    
    # Ajouter les métriques
    metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f} µm\nMAE = {mae:.4f} µm'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11, fontweight='bold')
    
    # Grille et légende
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Égaliser les axes
    plt.axis('equal')
    
    # Ajuster les limites
    margin = (max_val - min_val) * 0.05
    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Scatter plot sauvegardé: {save_path}")
    
    return {
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

def analyze_results(y_true, y_pred, history):
    """
    Analyse les résultats et génère un rapport.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions
        history: Historique d'entraînement
    """
    print(f"\n=== ANALYSE DES RÉSULTATS ===")
    
    # Métriques de performance
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"📊 PERFORMANCE AVEC DONNÉES RÉDUITES:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f} µm")
    print(f"  MAE: {mae:.4f} µm")
    print(f"  MSE: {mse:.6f}")
    
    # Comparaison avec l'expérience originale (référence)
    print(f"\n📈 COMPARAISON AVEC EXPÉRIENCE ORIGINALE:")
    print(f"  Expérience originale (5% bruit): R² = 0.993, RMSE = 0.051 µm")
    print(f"  Données réduites (5% bruit):     R² = {r2:.3f}, RMSE = {rmse:.3f} µm")
    
    degradation_r2 = ((0.993 - r2) / 0.993) * 100
    degradation_rmse = ((rmse - 0.051) / 0.051) * 100
    
    print(f"  Dégradation R²: {degradation_r2:.1f}%")
    print(f"  Dégradation RMSE: {degradation_rmse:+.1f}%")
    
    # Analyse de la convergence
    final_epoch = len(history['train_loss'])
    print(f"\n🔄 CONVERGENCE:")
    print(f"  Époques d'entraînement: {final_epoch}")
    print(f"  R² final train: {history['train_r2'][-1]:.4f}")
    print(f"  R² final test: {history['test_r2'][-1]:.4f}")
    print(f"  Écart train/test: {abs(history['train_r2'][-1] - history['test_r2'][-1]):.4f}")
    
    # Évaluation de l'overfitting
    if abs(history['train_r2'][-1] - history['test_r2'][-1]) > 0.1:
        print(f"  ⚠️  Possible overfitting détecté")
    else:
        print(f"  ✅ Pas d'overfitting significatif")
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    if r2 < 0.8:
        print(f"  ❌ Performance insuffisante - Plus de données nécessaires")
        print(f"  📈 Recommandation: Augmenter à 500+ échantillons d'entraînement")
    elif r2 < 0.9:
        print(f"  ⚠️  Performance modérée - Amélioration possible")
        print(f"  📈 Recommandation: Augmenter à 400+ échantillons d'entraînement")
    else:
        print(f"  ✅ Performance acceptable avec données réduites")
        print(f"  📈 300 échantillons semblent suffisants pour cette tâche")

def main():
    """Fonction principale de l'expérience."""
    
    print("="*60)
    print("EXPÉRIENCE DONNÉES RÉDUITES - IMPACT SUR LA GÉNÉRALISATION")
    print("="*60)
    print("Configuration:")
    print("  • 300 exemples d'entraînement")
    print("  • 100 exemples de test")
    print("  • Niveau de bruit: 5%")
    print("="*60)
    
    # Créer les dossiers de sortie
    os.makedirs("../plots", exist_ok=True)
    os.makedirs("../results", exist_ok=True)
    
    try:
        # 1. Charger le dataset complet
        X, y = load_dataset()
        
        # 2. Préparer les données réduites
        X_train, X_test, y_train, y_test = prepare_reduced_data(X, y, n_train=300, n_test=100)
        
        # 3. Entraîner le modèle
        model, scaler, history, predictions = train_model_reduced_data(
            X_train, y_train, X_test, y_test, noise_level=5
        )
        
        # 4. Créer le scatter plot demandé
        metrics = create_scatter_plot(y_test, predictions, "../plots/reduced_data_scatter_plot.png")
        
        # 5. Analyser les résultats
        analyze_results(y_test, predictions, history)
        
        # 6. Sauvegarder les résultats
        results = {
            'experiment_config': {
                'n_train': 300,
                'n_test': 100,
                'noise_level': 5,
                'architecture': 'RobustGapPredictor'
            },
            'performance': metrics,
            'training_epochs': len(history['train_loss']),
            'final_train_r2': history['train_r2'][-1],
            'final_test_r2': history['test_r2'][-1]
        }
        
        with open('../results/reduced_data_experiment.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("EXPÉRIENCE TERMINÉE AVEC SUCCÈS")
        print(f"{'='*60}")
        print(f"📊 Scatter plot généré: ../plots/reduced_data_scatter_plot.png")
        print(f"📋 Résultats sauvegardés: ../results/reduced_data_experiment.json")
        
    except Exception as e:
        print(f"❌ Erreur durant l'expérience: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
