#!/usr/bin/env python3
"""
Simple and Effective Neural Network Training
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Modèle simplifié et efficace basé sur l'analyse des données.
Utilise les corrélations découvertes pour créer un modèle optimal.
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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import time

class SimpleEffectiveRegressor(nn.Module):
    """
    Modèle simplifié basé sur l'analyse PCA et les corrélations.
    Architecture optimisée pour les données réelles.
    """
    def __init__(self, input_dim=1000, pca_components=10, output_dim=2):
        super(SimpleEffectiveRegressor, self).__init__()
        
        # Couche de réduction dimensionnelle inspirée de PCA
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, pca_components),
            nn.BatchNorm1d(pca_components),
            nn.ReLU()
        )
        
        # Prédicteur spécialisé pour chaque paramètre
        self.L_ecran_predictor = nn.Sequential(
            nn.Linear(pca_components, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.gap_predictor = nn.Sequential(
            nn.Linear(pca_components, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Initialisation des poids
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
        """Forward pass avec prédicteurs spécialisés."""
        # Extraction de features
        features = self.feature_extractor(x)
        
        # Prédictions spécialisées
        L_ecran = self.L_ecran_predictor(features)
        gap = self.gap_predictor(features)
        
        return torch.cat([L_ecran, gap], dim=1)

def load_and_prepare_data():
    """Charge et prépare les données avec normalisation optimisée."""
    
    print("Chargement et préparation des données...")
    
    # Charger les données
    data = np.load('processed_data/training_data.npz', allow_pickle=True)
    X = data['X']
    y = data['y']
    
    print(f"Données chargées: X{X.shape}, y{y.shape}")
    
    # Division stratifiée
    bins = np.linspace(y[:, 0].min(), y[:, 0].max(), 10)
    y_bins = np.digitize(y[:, 0], bins=bins)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42,
        stratify=y_bins
    )

    bins_temp = np.linspace(y_temp[:, 0].min(), y_temp[:, 0].max(), 8)
    y_temp_bins = np.digitize(y_temp[:, 0], bins=bins_temp)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,
        stratify=y_temp_bins
    )
    
    print(f"Division: Train{X_train.shape}, Val{X_val.shape}, Test{X_test.shape}")
    
    # Normalisation optimisée
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    
    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            scaler_X, scaler_y)

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """Crée les DataLoaders."""
    
    # Conversion en tenseurs
    datasets = []
    for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
        X_tensor = torch.FloatTensor(X.astype(np.float32))
        y_tensor = torch.FloatTensor(y.astype(np.float32))
        datasets.append(TensorDataset(X_tensor, y_tensor))
    
    # DataLoaders
    train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_simple_model(model, train_loader, val_loader, num_epochs=150, lr=0.001, device='cpu'):
    """Entraîne le modèle simplifié avec focus sur la convergence."""
    
    print(f"\nEntraînement du modèle simplifié...")
    print(f"Epochs: {num_epochs}, LR: {lr}, Device: {device}")
    
    model = model.to(device)
    
    # Loss function et optimiseur
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15, verbose=True
    )
    
    # Historique
    history = {
        'train_loss': [], 'val_loss': [],
        'train_r2': [], 'val_r2': [],
        'learning_rates': []
    }
    
    best_val_r2 = -float('inf')
    patience_counter = 0
    patience = 30
    
    print(f"\nDébut de l'entraînement...")
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
            
            # Gradient clipping
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
        
        # Calcul des métriques
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_pred = np.vstack(train_predictions)
        train_true = np.vstack(train_targets)
        val_pred = np.vstack(val_predictions)
        val_true = np.vstack(val_targets)
        
        train_r2 = r2_score(train_true, train_pred)
        val_r2 = r2_score(val_true, val_pred)
        
        # Sauvegarder l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Early stopping et sauvegarde
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            
            # Sauvegarder le meilleur modèle
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_r2': val_r2,
                'history': history
            }, 'models/simple_effective_regressor.pth')
        else:
            patience_counter += 1
        
        # Affichage
        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] - "
                  f"Train: Loss={train_loss:.6f}, R²={train_r2:.4f} | "
                  f"Val: Loss={val_loss:.6f}, R²={val_r2:.4f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping à l'epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\nEntraînement terminé en {total_time/60:.1f} minutes")
    print(f"Meilleur R² validation: {best_val_r2:.6f}")
    
    return model, history

def main():
    """Fonction principale."""
    
    print("="*60)
    print("ENTRAÎNEMENT MODÈLE SIMPLIFIÉ ET EFFICACE")
    print("="*60)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Préparer les données
    (X_train, X_val, X_test, y_train, y_val, y_test, 
     scaler_X, scaler_y) = load_and_prepare_data()
    
    # DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
    )
    
    # Créer le modèle
    model = SimpleEffectiveRegressor(
        input_dim=1000,
        pca_components=10,  # Basé sur l'analyse PCA
        output_dim=2
    )
    
    print(f"\nModèle créé:")
    print(f"  Paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    # Entraîner
    model, history = train_simple_model(
        model, train_loader, val_loader,
        num_epochs=150, lr=0.001, device=device
    )
    
    # Sauvegarder les scalers
    np.savez('models/simple_scalers.npz',
             scaler_X_mean=scaler_X.mean_, scaler_X_scale=scaler_X.scale_,
             scaler_y_mean=scaler_y.mean_, scaler_y_scale=scaler_y.scale_)
    
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*60}")
    
    final_r2 = history['val_r2'][-1] if history['val_r2'] else 0
    best_r2 = max(history['val_r2']) if history['val_r2'] else 0
    
    print(f"R² validation final: {final_r2:.6f}")
    print(f"Meilleur R² validation: {best_r2:.6f}")
    print(f"Objectif atteint (R² > 0.8): {'✓ OUI' if best_r2 > 0.8 else '✗ NON'}")

if __name__ == "__main__":
    main()
