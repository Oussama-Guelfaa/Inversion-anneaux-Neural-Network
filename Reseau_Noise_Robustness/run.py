#!/usr/bin/env python3
"""
Noise Robustness Test - Autonomous Testing Script
Author: Oussama GUELFAA
Date: 10 - 01 - 2025

Script autonome pour tester la robustesse au bruit du réseau de neurones.
Ce script évalue la performance du modèle face à différents niveaux de bruit gaussien
pour déterminer les conditions optimales de fonctionnement en environnement réel.
"""

import os
import sys
import yaml
import argparse
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
    """Dataset PyTorch pour les profils d'intensité avec support du bruit."""
    
    def __init__(self, intensity_profiles, gap_values):
        self.intensity_profiles = torch.FloatTensor(intensity_profiles)
        self.gap_values = torch.FloatTensor(gap_values)
    
    def __len__(self):
        return len(self.intensity_profiles)
    
    def __getitem__(self, idx):
        return self.intensity_profiles[idx], self.gap_values[idx]

class RobustGapPredictor(nn.Module):
    """Modèle robuste pour prédiction du gap avec régularisation."""
    
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

def load_config(config_path="config/noise_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_data_from_matlab():
    """Extract training data from MATLAB file."""
    print("🔄 Extracting data from MATLAB file...")
    
    try:
        import scipy.io as sio
        
        # Load MATLAB file
        mat_file = "../data_generation/all_banque_new_24_01_25_NEW_full.mat"
        data = sio.loadmat(mat_file)
        
        # Extract variables
        L_ecran_vect = data['L_ecran_subs_vect'].flatten()
        gap_vect = data['gap_sphere_vect'].flatten()
        I_subs = data['I_subs']
        I_subs_inc = data['I_subs_inc']
        
        # Calculate intensity ratios and truncate to 600 points
        intensity_profiles = []
        gap_values = []
        
        for i in range(len(L_ecran_vect)):
            for j in range(len(gap_vect)):
                ratio = np.abs(I_subs[i, j, :600])  # Truncate to 600 points
                ratio_inc = np.abs(I_subs_inc[i, j, :600])
                profile = ratio / ratio_inc
                
                intensity_profiles.append(profile)
                gap_values.append(gap_vect[j])
        
        intensity_profiles = np.array(intensity_profiles)
        gap_values = np.array(gap_values)
        
        print(f"✅ Extracted {len(intensity_profiles)} samples")
        print(f"   Profile shape: {intensity_profiles.shape}")
        print(f"   Gap range: {gap_values.min():.3f} to {gap_values.max():.3f}")
        
        return intensity_profiles, gap_values
        
    except Exception as e:
        print(f"❌ Error extracting data: {e}")
        return None, None

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
    
    print(f"🔊 Bruit {noise_level_percent}% ajouté - SNR moyen: {1/(noise_level_percent/100):.1f}")
    
    return X_noisy

def prepare_data_splits(X, y, config):
    """Divise les données en train/validation/test."""
    train_size = config['data_splits']['train']
    val_size = config['data_splits']['validation']
    test_size = config['data_splits']['test']
    
    # Première division: train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Deuxième division: train / val
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42
    )
    
    print(f"📊 Division des données:")
    print(f"   Train: {X_train.shape[0]} échantillons ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"   Validation: {X_val.shape[0]} échantillons ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"   Test: {X_test.shape[0]} échantillons ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model_with_noise(X_train, y_train, X_val, y_val, noise_level, config):
    """Entraîne un modèle avec un niveau de bruit spécifique."""
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
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Modèle et optimisation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustGapPredictor(input_size=X_train.shape[1]).to(device)
    
    learning_rate = config['training']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    patience = config['training']['early_stopping']['patience']
    early_stopping = EarlyStopping(patience=patience)
    
    # Historique d'entraînement
    history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}
    
    max_epochs = config['training']['epochs']
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
            print(f"   Epoch {epoch+1:3d}: Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
        
        if early_stopping(val_loss, model):
            print(f"   ⏹️ Early stopping à l'époque {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Sauvegarder le modèle
    model_path = f"models/model_noise_{noise_level}percent.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"✅ Entraînement terminé en {training_time:.1f}s")
    print(f"   Performance finale: R² = {val_r2:.4f}")
    
    return model, scaler, history, training_time, epoch + 1

def evaluate_model(model, scaler, X_test, y_test):
    """Évalue un modèle sur l'ensemble de test."""
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

def run_noise_robustness_test(config):
    """Execute complete noise robustness test."""
    print("🔬 NOISE ROBUSTNESS TEST - Comprehensive Analysis")
    print("=" * 60)
    
    # Extract data
    intensity_profiles, gap_values = extract_data_from_matlab()
    
    if intensity_profiles is None:
        print("❌ Failed to extract data. Test aborted.")
        return
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_splits(
        intensity_profiles, gap_values, config
    )
    
    # Test different noise levels
    noise_levels = config['noise_levels']
    results = {}
    
    for noise_level in noise_levels:
        noise_percent = int(noise_level * 100)
        
        # Train model with specific noise level
        model, scaler, history, training_time, final_epoch = train_model_with_noise(
            X_train, y_train, X_val, y_val, noise_percent, config
        )
        
        # Evaluate on test set
        metrics = evaluate_model(model, scaler, X_test, y_test)
        
        # Store results
        results[noise_percent] = {
            'metrics': metrics,
            'history': history,
            'training_time': training_time,
            'final_epoch': final_epoch
        }
        
        print(f"🎯 Noise {noise_percent}%: R² = {metrics['r2']:.4f}, "
              f"RMSE = {metrics['rmse']:.4f}")
    
    # Save results
    save_results(results, noise_levels, y_test, config)
    
    # Generate plots
    create_plots(results, noise_levels, config)
    
    print("\n🏁 Noise robustness test completed successfully!")
    return results

def save_results(results, noise_levels, y_test, config):
    """Save all test results."""
    print("\n💾 Saving results...")
    
    # Performance summary
    performance_data = []
    noise_percent_levels = [int(n * 100) for n in noise_levels]
    
    for noise_percent in noise_percent_levels:
        if noise_percent in results:
            result = results[noise_percent]
            metrics = result['metrics']
            
            performance_data.append({
                'noise_level': noise_percent,
                'r2_score': metrics['r2'],
                'rmse_um': metrics['rmse'],
                'mae_um': metrics['mae'],
                'training_time_s': result['training_time'],
                'epochs_to_convergence': result['final_epoch']
            })
    
    # Save performance table
    performance_df = pd.DataFrame(performance_data)
    performance_df.to_csv('results/performance_by_noise_level.csv', index=False)
    
    # Save detailed predictions
    for noise_percent in noise_percent_levels:
        if noise_percent in results:
            predictions = results[noise_percent]['metrics']['predictions']
            pred_df = pd.DataFrame({
                'gap_true': y_test,
                'gap_predicted': predictions,
                'error': predictions - y_test,
                'absolute_error': np.abs(predictions - y_test)
            })
            pred_df.to_csv(f'results/predictions_noise_{noise_percent}percent.csv', index=False)
    
    print("✅ Results saved successfully!")

def create_plots(results, noise_levels, config):
    """Create visualization plots."""
    print("\n📊 Generating plots...")
    
    noise_percent_levels = [int(n * 100) for n in noise_levels]
    
    # Main robustness plot
    plt.figure(figsize=(15, 10))
    
    # R² vs Noise Level
    plt.subplot(2, 3, 1)
    r2_scores = [results[noise]['metrics']['r2'] for noise in noise_percent_levels if noise in results]
    plt.plot(noise_percent_levels[:len(r2_scores)], r2_scores, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Target R² = 0.8')
    plt.xlabel('Noise Level (%)')
    plt.ylabel('R² Score')
    plt.title('Performance vs Noise Level')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # RMSE vs Noise Level
    plt.subplot(2, 3, 2)
    rmse_scores = [results[noise]['metrics']['rmse'] for noise in noise_percent_levels if noise in results]
    plt.plot(noise_percent_levels[:len(rmse_scores)], rmse_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Noise Level (%)')
    plt.ylabel('RMSE (µm)')
    plt.title('RMSE vs Noise Level')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/noise_robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Plots generated successfully!")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Noise Robustness Test')
    parser.add_argument('--config', default='config/noise_config.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run noise robustness test
    results = run_noise_robustness_test(config)
    
    print("🎉 Noise robustness analysis completed!")

if __name__ == "__main__":
    main()
