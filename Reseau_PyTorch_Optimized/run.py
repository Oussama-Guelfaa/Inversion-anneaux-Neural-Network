#!/usr/bin/env python3
"""
PyTorch Optimized Network - Autonomous Training Script
Author: Oussama GUELFAA
Date: 10 - 01 - 2025

Script autonome pour le réseau PyTorch optimisé avec ResNet 1D.
Implémentation optimisée avec techniques PyTorch avancées.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import json
import time
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration pour reproductibilité
torch.manual_seed(42)
np.random.seed(42)

class ResidualBlock1D(nn.Module):
    """Bloc résiduel 1D optimisé."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        identity = self.skip_connection(identity)
        out += identity
        out = F.relu(out)
        
        return out

class OptimizedPyTorchRegressor(nn.Module):
    """Réseau PyTorch optimisé avec ResNet 1D."""
    
    def __init__(self, input_size=1000, num_classes=2):
        super(OptimizedPyTorchRegressor, self).__init__()
        
        # Couches convolutionnelles initiales
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Blocs résiduels
        self.res_block1 = ResidualBlock1D(128, 128)
        self.res_block2 = ResidualBlock1D(128, 128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.res_block3 = ResidualBlock1D(256, 256)
        self.res_block4 = ResidualBlock1D(256, 256)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Couches denses
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialisation optimisée
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Reshape pour Conv1D
        x = x.unsqueeze(1)  # (batch_size, 1, length)
        
        # Blocs convolutionnels
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Blocs résiduels 1
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Blocs résiduels 2
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling et classification
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class IntensityDataset(Dataset):
    """Dataset optimisé pour PyTorch."""
    
    def __init__(self, intensity_profiles, parameters, transform=None):
        self.intensity_profiles = torch.FloatTensor(intensity_profiles)
        self.parameters = torch.FloatTensor(parameters)
        self.transform = transform
    
    def __len__(self):
        return len(self.intensity_profiles)
    
    def __getitem__(self, idx):
        profile = self.intensity_profiles[idx]
        params = self.parameters[idx]
        
        if self.transform:
            profile = self.transform(profile)
        
        return profile, params

def load_config(config_path="config/pytorch_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_data_pytorch_optimized(config):
    """Extract data with PyTorch optimizations."""
    print("🔄 PyTorch optimized data extraction...")
    
    try:
        # Load MATLAB file
        mat_file = config['paths']['data_file']
        data = sio.loadmat(mat_file)
        
        # Extract variables
        L_ecran_vect = data['L_ecran_subs_vect'].flatten()
        gap_vect = data['gap_sphere_vect'].flatten()
        I_subs = data['I_subs']
        I_subs_inc = data['I_subs_inc']
        
        # Use full profiles if specified
        profile_length = config['data']['profile_length']
        
        intensity_profiles = []
        L_ecran_values = []
        gap_values = []
        
        for i in range(len(L_ecran_vect)):
            for j in range(len(gap_vect)):
                ratio = np.abs(I_subs[i, j, :profile_length])
                ratio_inc = np.abs(I_subs_inc[i, j, :profile_length])
                profile = ratio / ratio_inc
                
                intensity_profiles.append(profile)
                L_ecran_values.append(L_ecran_vect[i])
                gap_values.append(gap_vect[j])
        
        intensity_profiles = np.array(intensity_profiles)
        parameters = np.column_stack([L_ecran_values, gap_values])
        
        print(f"✅ PyTorch data extracted: {intensity_profiles.shape}")
        
        return intensity_profiles, parameters
        
    except Exception as e:
        print(f"❌ Error in PyTorch data extraction: {e}")
        return None, None

def train_pytorch_optimized(X_train, X_val, y_train, y_val, config):
    """Train PyTorch optimized model."""
    print("\n🚀 PYTORCH OPTIMIZED TRAINING")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Create datasets
    train_dataset = IntensityDataset(X_train, y_train)
    val_dataset = IntensityDataset(X_val, y_val)
    
    # Optimized data loaders
    batch_size = config['training']['batch_size']
    num_workers = config['advanced_features'].get('num_workers', 4)
    pin_memory = config['advanced_features'].get('pin_memory', True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Create optimized model
    model = OptimizedPyTorchRegressor(
        input_size=config['model']['input_size'],
        num_classes=config['model']['output_size']
    ).to(device)
    
    # Optimized optimizer
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Advanced scheduler
    if config['training']['lr_scheduler']['type'] == 'CosineAnnealingWarmRestarts':
        T_0 = config['training']['lr_scheduler']['T_0']
        T_mult = config['training']['lr_scheduler']['T_mult']
        eta_min = config['training']['lr_scheduler']['eta_min']
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    criterion = nn.MSELoss()
    
    # Training loop with optimizations
    history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': [], 'lr': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['training']['early_stopping']['patience']
    
    epochs = config['training']['epochs']
    print(f"🏃 Starting optimized training for {epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if config['training']['gradient_clipping']['enable']:
                max_norm = config['training']['gradient_clipping']['max_norm']
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.extend(output.detach().cpu().numpy())
            train_targets.extend(target.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                val_predictions.extend(output.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate R²
        train_predictions = np.array(train_predictions)
        train_targets = np.array(train_targets)
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        train_r2 = r2_score(train_targets, train_predictions)
        val_r2 = r2_score(val_targets, val_predictions)
        
        # Update scheduler
        if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['lr'].append(current_lr)
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/pytorch_optimized_best.pth')
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, "
                  f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, LR={current_lr:.2e}")
        
        if patience_counter >= patience:
            print(f"   ⏹️ Early stopping at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict(torch.load('models/pytorch_optimized_best.pth'))
    
    print(f"✅ PyTorch training completed in {training_time/60:.1f} minutes")
    
    return model, history, training_time

def evaluate_pytorch_model(model, X_test, y_test, config):
    """Evaluate PyTorch optimized model."""
    print("\n🔍 PYTORCH MODEL EVALUATION")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Create test dataset
    test_dataset = IntensityDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.numpy())
    
    y_pred = np.array(all_predictions)
    y_true = np.array(all_targets)
    
    # Calculate metrics
    r2_global = r2_score(y_true, y_pred)
    r2_L = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_gap = r2_score(y_true[:, 1], y_pred[:, 1])
    
    rmse_L = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    
    mae_L = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_gap = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    
    print(f"📊 PYTORCH OPTIMIZED RESULTS:")
    print(f"   R² global: {r2_global:.6f}")
    print(f"   R² L_ecran: {r2_L:.6f}")
    print(f"   R² gap: {r2_gap:.6f}")
    print(f"   RMSE L_ecran: {rmse_L:.6f}")
    print(f"   RMSE gap: {rmse_gap:.6f}")
    print(f"   MAE L_ecran: {mae_L:.6f}")
    print(f"   MAE gap: {mae_gap:.6f}")
    
    # Check targets
    target_r2_global = config['evaluation']['performance_targets']['r2_global']
    success = r2_global >= target_r2_global
    
    print(f"   Target R² > {target_r2_global}: {'✅' if success else '❌'}")
    
    return y_pred, {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'success': success
    }

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='PyTorch Optimized Training')
    parser.add_argument('--config', default='config/pytorch_config.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 PYTORCH OPTIMIZED NETWORK - ADVANCED EDITION")
    print("Réseau PyTorch avec ResNet 1D et optimisations avancées")
    print("="*80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Extract data
    intensity_profiles, parameters = extract_data_pytorch_optimized(config)
    
    if intensity_profiles is None:
        print("❌ Failed to extract data. Training aborted.")
        return
    
    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(intensity_profiles)
    y_scaled = scaler_y.fit_transform(parameters)
    
    # Split data
    train_split = config['data']['train_split']
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=1-train_split, random_state=42
    )
    
    print(f"\n📊 PyTorch data splits:")
    print(f"   Train: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    
    # Train model
    model, history, training_time = train_pytorch_optimized(X_train, X_val, y_train, y_val, config)
    
    # Evaluate model
    y_pred, metrics = evaluate_pytorch_model(model, X_val, y_val, config)
    
    # Save scalers
    joblib.dump(scaler_X, 'models/pytorch_scaler_X.pkl')
    joblib.dump(scaler_y, 'models/pytorch_scaler_y.pkl')
    
    # Save results
    history_df = pd.DataFrame(history)
    history_df.to_csv('results/pytorch_training_history.csv', index=False)
    
    with open('results/pytorch_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"🎯 PYTORCH OPTIMIZED RESULTS")
    print(f"{'='*80}")
    
    print(f"⚡ PYTORCH FEATURES:")
    print(f"   • ResNet 1D avec blocs résiduels")
    print(f"   • Scheduler CosineAnnealingWarmRestarts")
    print(f"   • Optimisations mémoire et parallélisation")
    print(f"   • Gradient clipping et early stopping")
    
    print(f"\n📊 PERFORMANCES:")
    print(f"   • R² global: {metrics['r2_global']:.6f}")
    print(f"   • R² L_ecran: {metrics['r2_L']:.6f}")
    print(f"   • R² gap: {metrics['r2_gap']:.6f}")
    print(f"   • Temps d'entraînement: {training_time/60:.1f} min")
    print(f"   • Succès: {'🎉 YES' if metrics['success'] else '⚠️ NO'}")
    
    print(f"\n📁 FICHIERS PYTORCH:")
    print(f"   • models/pytorch_optimized_best.pth")
    print(f"   • models/pytorch_scaler_*.pkl")
    print(f"   • results/pytorch_training_history.csv")
    
    print("\n🏁 PyTorch Optimized training completed!")

if __name__ == "__main__":
    main()
