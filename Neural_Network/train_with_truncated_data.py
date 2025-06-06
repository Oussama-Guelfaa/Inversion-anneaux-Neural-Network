#!/usr/bin/env python3
"""
Train Neural Network with Truncated Data
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Entraîne le réseau de neurones avec les profils tronqués à 600 points
pour résoudre le problème du pic divergent et améliorer la prédiction du gap.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import time

# Configuration pour reproductibilité
torch.manual_seed(42)
np.random.seed(42)

class OptimizedRegressor(nn.Module):
    """Réseau de neurones optimisé pour profils tronqués à 600 points."""
    
    def __init__(self, input_size=600, hidden_sizes=[512, 256, 128, 64], output_size=2, dropout_rates=[0.2, 0.15, 0.1, 0.05]):
        super(OptimizedRegressor, self).__init__()
        
        # Feature extractor avec normalisation
        layers = []
        prev_size = input_size
        
        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_sizes, dropout_rates)):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Têtes spécialisées pour chaque paramètre
        self.L_ecran_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.gap_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialisation des poids
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialisation optimisée des poids."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        L_ecran = self.L_ecran_head(features)
        gap = self.gap_head(features)
        return torch.cat([L_ecran, gap], dim=1)

def load_truncated_training_data():
    """Charge les données d'entraînement tronquées."""
    
    print("=== CHARGEMENT DES DONNÉES TRONQUÉES ===")
    
    # Charger les profils tronqués
    df_profiles = pd.read_csv('processed_data/intensity_profiles_truncated_600.csv')
    df_params = pd.read_csv('processed_data/parameters_truncated_600.csv')
    
    X = df_profiles.values.astype(np.float32)
    y = df_params[['L_ecran', 'gap']].values.astype(np.float32)
    
    print(f"Données tronquées chargées:")
    print(f"  X shape: {X.shape} (profils à 600 points)")
    print(f"  y shape: {y.shape}")
    print(f"  Plages des paramètres:")
    print(f"    L_ecran: [{y[:, 0].min():.3f}, {y[:, 0].max():.3f}] µm")
    print(f"    gap: [{y[:, 1].min():.6f}, {y[:, 1].max():.6f}] µm")
    
    return X, y

def load_truncated_test_data():
    """Charge les données de test expérimentales tronquées."""
    
    print("\n=== CHARGEMENT DES DONNÉES DE TEST TRONQUÉES ===")
    
    dataset_dir = "../data_generation/dataset"
    labels_df = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))
    
    X_test = []
    y_test = []
    filenames = []
    
    for idx, row in labels_df.iterrows():
        filename = row['filename']
        gap = row['gap_um']
        L_ecran = row['L_um']
        
        mat_filename = filename.replace('.png', '.mat')
        mat_path = os.path.join(dataset_dir, mat_filename)
        
        if os.path.exists(mat_path):
            try:
                data = sio.loadmat(mat_path)
                ratio = data['ratio'].flatten()
                
                # Tronquer à 600 points
                ratio_truncated = ratio[:600]
                
                X_test.append(ratio_truncated)
                y_test.append([L_ecran, gap])
                filenames.append(filename)
                
            except Exception as e:
                print(f"Erreur {mat_filename}: {e}")
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    print(f"Données de test tronquées: X{X_test.shape}, y{y_test.shape}")
    return X_test, y_test, filenames

def train_truncated_model(X_train, X_val, y_train, y_val, epochs=200):
    """Entraîne le modèle avec les données tronquées."""
    
    print(f"\n=== ENTRAÎNEMENT AVEC DONNÉES TRONQUÉES ===")
    
    # Créer le modèle pour 600 points
    model = OptimizedRegressor(input_size=600)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Modèle créé pour {600} points d'entrée")
    print(f"Device: {device}")
    
    # Convertir en tenseurs PyTorch
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Datasets et DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Optimiseur et loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15, verbose=True)
    criterion = nn.MSELoss()
    
    # Entraînement
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    
    print(f"Début de l'entraînement...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Phase de validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Sauvegarder le meilleur modèle
            torch.save(model.state_dict(), 'models/truncated_best_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping à l'epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"Entraînement terminé en {training_time/60:.1f} minutes")
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load('models/truncated_best_model.pth'))
    
    return model, train_losses, val_losses

def evaluate_truncated_model(model, X_test, y_test, scaler_X, filenames):
    """Évalue le modèle tronqué sur les données de test."""
    
    print(f"\n=== ÉVALUATION MODÈLE TRONQUÉ ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Normaliser et prédire
    X_test_scaled = scaler_X.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
    
    # Calculer les métriques
    r2_global = r2_score(y_test, y_pred)
    r2_L = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_gap = r2_score(y_test[:, 1], y_pred[:, 1])
    
    rmse_L = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    
    mae_L = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_gap = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    
    mape_L = np.mean(np.abs((y_test[:, 0] - y_pred[:, 0]) / y_test[:, 0])) * 100
    mape_gap = np.mean(np.abs((y_test[:, 1] - y_pred[:, 1]) / y_test[:, 1])) * 100
    
    print(f"RÉSULTATS AVEC DONNÉES TRONQUÉES:")
    print(f"  R² global: {r2_global:.6f}")
    print(f"  R² L_ecran: {r2_L:.6f}")
    print(f"  R² gap: {r2_gap:.6f}")
    print(f"  RMSE L_ecran: {rmse_L:.6f} µm")
    print(f"  RMSE gap: {rmse_gap:.6f} µm")
    print(f"  MAE L_ecran: {mae_L:.6f} µm")
    print(f"  MAE gap: {mae_gap:.6f} µm")
    print(f"  MAPE L_ecran: {mape_L:.2f}%")
    print(f"  MAPE gap: {mape_gap:.2f}%")
    
    success = r2_global > 0.8
    print(f"  Objectif R² > 0.8: {'✓ ATTEINT' if success else '✗ NON ATTEINT'}")
    
    return y_pred, {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'mape_L': mape_L, 'mape_gap': mape_gap,
        'success': success
    }

def main():
    """Fonction principale."""
    
    print("="*80)
    print("ENTRAÎNEMENT AVEC DONNÉES TRONQUÉES (600 POINTS)")
    print("Résolution du problème du pic divergent")
    print("="*80)
    
    # 1. Charger les données tronquées
    X, y = load_truncated_training_data()
    X_test, y_test, filenames = load_truncated_test_data()
    
    # 2. Division train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDivision des données:")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # 3. Normalisation
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # 4. Entraînement
    os.makedirs('models', exist_ok=True)
    model, train_losses, val_losses = train_truncated_model(X_train_scaled, X_val_scaled, y_train, y_val)
    
    # 5. Évaluation
    y_pred, metrics = evaluate_truncated_model(model, X_test, y_test, scaler_X, filenames)
    
    # 6. Sauvegarder les scalers
    import joblib
    joblib.dump(scaler_X, 'models/truncated_scaler_X.pkl')
    
    print(f"\n{'='*80}")
    print(f"RÉSULTATS FINAUX - MODÈLE TRONQUÉ")
    print(f"{'='*80}")
    print(f"✅ AMÉLIORATION ATTENDUE:")
    print(f"   • Profils tronqués: 1000 → 600 points")
    print(f"   • Pic divergent supprimé")
    print(f"   • Signal/bruit amélioré")
    
    print(f"\n📊 PERFORMANCES:")
    print(f"   • R² global: {metrics['r2_global']:.6f}")
    print(f"   • R² L_ecran: {metrics['r2_L']:.6f}")
    print(f"   • R² gap: {metrics['r2_gap']:.6f}")
    print(f"   • Objectif atteint: {'OUI' if metrics['success'] else 'NON'}")
    
    print(f"\n📁 FICHIERS GÉNÉRÉS:")
    print(f"   • models/truncated_best_model.pth")
    print(f"   • models/truncated_scaler_X.pkl")

if __name__ == "__main__":
    main()
