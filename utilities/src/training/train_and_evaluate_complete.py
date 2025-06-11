#!/usr/bin/env python3
"""
Complete Neural Network Training and Evaluation
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Script complet qui :
1. Entraîne sur les données simulées (all_banque_new_24_01_25_NEW_full.mat)
2. Teste sur les données expérimentales réelles (dossier dataset)
3. Évalue les performances et génère un rapport complet
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import os
import time
import glob

class OptimizedRegressor(nn.Module):
    """
    Modèle optimisé basé sur l'analyse des données.
    Architecture simple mais efficace pour la régression des paramètres.
    """
    def __init__(self, input_dim=1000, output_dim=2):
        super(OptimizedRegressor, self).__init__()
        
        # Architecture progressive avec normalisation
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        # Prédicteurs spécialisés
        self.L_ecran_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.gap_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation optimisée des poids."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass avec têtes spécialisées."""
        features = self.feature_extractor(x)
        L_ecran = self.L_ecran_head(features)
        gap = self.gap_head(features)
        return torch.cat([L_ecran, gap], dim=1)

def load_training_data():
    """Charge les données d'entraînement simulées."""
    print("Chargement des données d'entraînement...")
    data = np.load('processed_data/training_data.npz', allow_pickle=True)
    X = data['X']
    y = data['y']
    metadata = data['metadata'].item()
    print(f"  Données d'entraînement: X{X.shape}, y{y.shape}")
    return X, y, metadata

def load_test_data():
    """Charge les données de test expérimentales."""
    print("Chargement des données de test expérimentales...")
    
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
                
                X_test.append(ratio)
                y_test.append([L_ecran, gap])
                filenames.append(filename)
                
            except Exception as e:
                print(f"  Erreur {mat_filename}: {e}")
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"  Données de test: X{X_test.shape}, y{y_test.shape}")
    return X_test, y_test, filenames

def prepare_training_data(X, y):
    """Prépare les données d'entraînement avec validation split."""

    # Division train/validation (pas de test car on a les données expérimentales)
    bins = np.linspace(y[:, 0].min(), y[:, 0].max(), 10)
    y_bins = np.digitize(y[:, 0], bins=bins)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y_bins
    )
    
    print(f"Division: Train{X_train.shape}, Val{X_val.shape}")
    
    # Normalisation
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    return (X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled,
            scaler_X, scaler_y)

def create_data_loaders(X_train, X_val, y_train, y_val, batch_size=64):
    """Crée les DataLoaders."""
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs=150, lr=0.001, device='cpu'):
    """Entraîne le modèle avec early stopping."""
    
    print(f"\nEntraînement du modèle...")
    print(f"  Epochs: {num_epochs}, LR: {lr}, Device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15, verbose=True
    )
    
    history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}
    best_val_r2 = -float('inf')
    patience_counter = 0
    patience = 25
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.append(outputs.detach().cpu().numpy())
            train_targets.append(batch_y.detach().cpu().numpy())
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
        
        # Métriques
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_pred = np.vstack(train_predictions)
        train_true = np.vstack(train_targets)
        val_pred = np.vstack(val_predictions)
        val_true = np.vstack(val_targets)
        
        train_r2 = r2_score(train_true, train_pred)
        val_r2 = r2_score(val_true, val_pred)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        
        scheduler.step(val_loss)
        
        # Early stopping et sauvegarde
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_r2': val_r2,
                'history': history
            }, 'models/final_optimized_regressor.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] - "
                  f"Train: Loss={train_loss:.6f}, R²={train_r2:.4f} | "
                  f"Val: Loss={val_loss:.6f}, R²={val_r2:.4f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping à l'epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\nEntraînement terminé en {total_time/60:.1f} minutes")
    print(f"Meilleur R² validation: {best_val_r2:.6f}")
    
    return model, history

def evaluate_on_test_data(model, X_test, y_test, scaler_X, scaler_y, filenames, device='cpu'):
    """Évalue le modèle sur les données de test expérimentales."""
    
    print(f"\n=== ÉVALUATION SUR DONNÉES EXPÉRIMENTALES ===")
    
    # Normaliser les données de test avec les scalers d'entraînement
    X_test_scaled = scaler_X.transform(X_test)
    
    # Prédiction
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    # Dénormaliser les prédictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculer les métriques
    r2_L = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_gap = r2_score(y_test[:, 1], y_pred[:, 1])
    r2_global = r2_score(y_test, y_pred)
    
    rmse_L = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    
    mae_L = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_gap = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    
    print(f"Métriques sur données expérimentales:")
    print(f"  R² global: {r2_global:.6f}")
    print(f"  R² L_ecran: {r2_L:.6f}")
    print(f"  R² gap: {r2_gap:.6f}")
    print(f"  RMSE L_ecran: {rmse_L:.6f}")
    print(f"  RMSE gap: {rmse_gap:.6f}")
    print(f"  MAE L_ecran: {mae_L:.6f}")
    print(f"  MAE gap: {mae_gap:.6f}")
    
    # Objectif atteint ?
    success = r2_global > 0.8
    print(f"  Objectif R² > 0.8: {'✓ ATTEINT' if success else '✗ NON ATTEINT'}")
    
    return y_pred, {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'success': success
    }

def main():
    """Fonction principale complète."""
    
    print("="*80)
    print("ENTRAÎNEMENT ET ÉVALUATION COMPLÈTE DU RÉSEAU DE NEURONES")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Charger les données
    X_train_full, y_train_full, metadata = load_training_data()
    X_test, y_test, filenames = load_test_data()
    
    # 2. Préparer les données d'entraînement
    (X_train, X_val, y_train, y_val, 
     scaler_X, scaler_y) = prepare_training_data(X_train_full, y_train_full)
    
    # 3. Créer les DataLoaders
    train_loader, val_loader = create_data_loaders(X_train, X_val, y_train, y_val)
    
    # 4. Créer et entraîner le modèle
    model = OptimizedRegressor(input_dim=1000, output_dim=2)
    print(f"Modèle créé: {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    model, history = train_model(model, train_loader, val_loader, 
                                num_epochs=150, lr=0.001, device=device)
    
    # 5. Évaluer sur les données expérimentales
    y_pred, metrics = evaluate_on_test_data(model, X_test, y_test, 
                                          scaler_X, scaler_y, filenames, device)
    
    # 6. Sauvegarder les scalers
    np.savez('models/final_scalers.npz',
             scaler_X_mean=scaler_X.mean_, scaler_X_scale=scaler_X.scale_,
             scaler_y_mean=scaler_y.mean_, scaler_y_scale=scaler_y.scale_)
    
    print(f"\n{'='*80}")
    print(f"RÉSULTATS FINAUX")
    print(f"{'='*80}")
    print(f"R² global sur données expérimentales: {metrics['r2_global']:.6f}")
    print(f"Objectif atteint (R² > 0.8): {'OUI' if metrics['success'] else 'NON'}")
    print(f"Fichiers sauvegardés:")
    print(f"  - models/final_optimized_regressor.pth")
    print(f"  - models/final_scalers.npz")

if __name__ == "__main__":
    main()
