#!/usr/bin/env python3
"""
Ultra Specialized Regressor - Autonomous Training Script
Author: Oussama GUELFAA
Date: 10 - 01 - 2025

Script autonome pour le régresseur ultra-spécialisé avec ensemble training.
Architecture ultra-optimisée avec focus maximal sur le paramètre gap.
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
from torch.utils.data import Dataset, DataLoader, TensorDataset
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

class UltraWeightedLoss(nn.Module):
    """Loss ultra-pondérée avec focus extrême sur gap."""
    
    def __init__(self, gap_weight=50.0):
        super(UltraWeightedLoss, self).__init__()
        self.gap_weight = gap_weight
    
    def forward(self, pred, target):
        mse_L = (pred[:, 0] - target[:, 0]) ** 2
        mse_gap = (pred[:, 1] - target[:, 1]) ** 2
        
        # Loss ultra-pondérée
        total_loss = mse_L.mean() + self.gap_weight * mse_gap.mean()
        return total_loss

class UltraSpecializedRegressor(nn.Module):
    """Architecture ultra-spécialisée avec focus maximal sur gap."""
    
    def __init__(self, input_size=600):
        super(UltraSpecializedRegressor, self).__init__()
        
        # Feature extractor commun plus profond
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
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
            nn.Dropout(0.1)
        )
        
        # Tête L_ecran simple (signal fort)
        self.L_ecran_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Amélioration des features pour gap
        self.gap_feature_enhancer = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Double attention pour gap
        self.gap_attention_1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        self.gap_attention_2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        # Tête gap ultra-spécialisée
        self.gap_head = nn.Sequential(
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.02),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialisation optimisée
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Features communes
        features = self.feature_extractor(x)
        
        # L_ecran (simple)
        L_ecran = self.L_ecran_head(features)
        
        # Gap (ultra-spécialisé)
        gap_features = self.gap_feature_enhancer(features)
        
        # Double attention pour gap
        attention_1 = self.gap_attention_1(gap_features)
        attention_2 = self.gap_attention_2(gap_features)
        
        # Combinaison des attentions
        combined_attention = attention_1 * attention_2
        attended_features = gap_features * combined_attention
        
        gap = self.gap_head(attended_features)
        
        return torch.cat([L_ecran, gap], dim=1)

def load_config(config_path="config/ultra_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_and_preprocess_data(config):
    """Extract and preprocess data with ultra optimization."""
    print("🔄 Ultra data extraction and preprocessing...")
    
    try:
        # Load MATLAB file
        mat_file = config['paths']['data_file']
        data = sio.loadmat(mat_file)
        
        # Extract variables
        L_ecran_vect = data['L_ecran_subs_vect'].flatten()
        gap_vect = data['gap_sphere_vect'].flatten()
        I_subs = data['I_subs']
        I_subs_inc = data['I_subs_inc']
        
        # Calculate intensity ratios and truncate
        intensity_profiles = []
        L_ecran_values = []
        gap_values = []
        
        truncate_to = config['data_preprocessing']['profile_truncation']['truncate_to']
        
        for i in range(len(L_ecran_vect)):
            for j in range(len(gap_vect)):
                ratio = np.abs(I_subs[i, j, :truncate_to])
                ratio_inc = np.abs(I_subs_inc[i, j, :truncate_to])
                profile = ratio / ratio_inc
                
                intensity_profiles.append(profile)
                L_ecran_values.append(L_ecran_vect[i])
                gap_values.append(gap_vect[j])
        
        intensity_profiles = np.array(intensity_profiles)
        parameters = np.column_stack([L_ecran_values, gap_values])
        
        # Problem 1: Label rounding
        if config['data_preprocessing']['label_rounding']['enable']:
            decimals = config['data_preprocessing']['label_rounding']['decimals']
            parameters = np.round(parameters, decimals)
            print(f"🔢 Labels rounded to {decimals} decimals")
        
        # Problem 3: Experimental focus
        if config['data_preprocessing']['experimental_focus']['enable']:
            gap_range = config['data_preprocessing']['experimental_focus']['gap_range']
            mask = (parameters[:, 1] >= gap_range[0]) & (parameters[:, 1] <= gap_range[1])
            intensity_profiles = intensity_profiles[mask]
            parameters = parameters[mask]
            print(f"📊 Experimental focus: {len(intensity_profiles)} samples")
        
        # Problem 2: Separate normalization
        scaler_X = StandardScaler()
        scaler_L = StandardScaler()
        scaler_gap = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(intensity_profiles)
        y_L_scaled = scaler_L.fit_transform(parameters[:, 0:1])
        y_gap_scaled = scaler_gap.fit_transform(parameters[:, 1:2])
        y_scaled = np.hstack([y_L_scaled, y_gap_scaled])
        
        print(f"✅ Ultra preprocessing completed: {X_scaled.shape}")
        
        return X_scaled, y_scaled, parameters, scaler_X, scaler_L, scaler_gap
        
    except Exception as e:
        print(f"❌ Error in ultra preprocessing: {e}")
        return None, None, None, None, None, None

def train_ultra_ensemble(X_train, X_val, y_train, y_val, config):
    """Train ensemble of ultra-specialized models."""
    print("\n🚀 ULTRA ENSEMBLE TRAINING")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Ensemble training
    if config['ensemble_training']['enable']:
        n_models = config['ensemble_training']['n_models']
        gap_weights = config['ensemble_training']['gap_weights']
        models = []
        histories = []
        
        for i in range(n_models):
            print(f"\n🎯 Training ensemble model {i+1}/{n_models} (gap_weight={gap_weights[i]})")
            
            # Create model
            model = UltraSpecializedRegressor(input_size=config['model']['input_size']).to(device)
            
            # Ultra loss with specific weight
            criterion = UltraWeightedLoss(gap_weight=gap_weights[i])
            
            # Ultra optimizer
            learning_rate = config['training']['learning_rate']
            weight_decay = config['training']['weight_decay']
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            # Ultra scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.3, patience=15, min_lr=1e-6
            )
            
            # Training loop
            history = {'train_loss': [], 'val_loss': []}
            best_val_loss = float('inf')
            patience_counter = 0
            patience = config['training']['early_stopping']['patience']
            
            epochs = config['training']['epochs']
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Ultra gradient clipping
                    if config['advanced_features']['gradient_clipping']['enable']:
                        max_norm = config['advanced_features']['gradient_clipping']['max_norm']
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'models/ultra_model_{i}.pth')
                else:
                    patience_counter += 1
                
                if epoch % 20 == 0:
                    print(f"   Epoch {epoch:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}")
                
                if patience_counter >= patience:
                    print(f"   ⏹️ Early stopping at epoch {epoch}")
                    break
            
            # Load best model
            model.load_state_dict(torch.load(f'models/ultra_model_{i}.pth'))
            models.append(model)
            histories.append(history)
            
            print(f"   ✅ Model {i+1} completed with best val loss: {best_val_loss:.6f}")
        
        return models, histories
    
    else:
        # Single model training
        model = UltraSpecializedRegressor(input_size=config['model']['input_size']).to(device)
        # ... (similar training logic for single model)
        return [model], [{}]

def evaluate_ultra_ensemble(models, X_test, y_test_original, scaler_L, scaler_gap, config):
    """Evaluate ultra ensemble performance."""
    print("\n🔍 ULTRA ENSEMBLE EVALUATION")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Get predictions from all models
    all_predictions = []
    
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_test_tensor).cpu().numpy()
        
        # Denormalize
        pred_L = scaler_L.inverse_transform(pred_scaled[:, 0:1]).flatten()
        pred_gap = scaler_gap.inverse_transform(pred_scaled[:, 1:2]).flatten()
        pred = np.column_stack([pred_L, pred_gap])
        
        all_predictions.append(pred)
    
    # Ensemble prediction (weighted average)
    if len(models) > 1:
        # Simple average for now
        y_pred_ensemble = np.mean(all_predictions, axis=0)
    else:
        y_pred_ensemble = all_predictions[0]
    
    # Round predictions
    decimals = config['data_preprocessing']['label_rounding']['decimals']
    y_pred_rounded = np.round(y_pred_ensemble, decimals)
    
    # Calculate metrics
    r2_global = r2_score(y_test_original, y_pred_rounded)
    r2_L = r2_score(y_test_original[:, 0], y_pred_rounded[:, 0])
    r2_gap = r2_score(y_test_original[:, 1], y_pred_rounded[:, 1])
    
    rmse_L = np.sqrt(mean_squared_error(y_test_original[:, 0], y_pred_rounded[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_test_original[:, 1], y_pred_rounded[:, 1]))
    
    # Ultra tolerance evaluation
    if config['evaluation']['tolerance_evaluation']['enable']:
        tolerance_L = config['evaluation']['tolerance_evaluation']['tolerance_L']
        tolerance_gap = config['evaluation']['tolerance_evaluation']['tolerance_gap']
        
        tolerance_acc_L = np.mean(np.abs(y_test_original[:, 0] - y_pred_rounded[:, 0]) <= tolerance_L) * 100
        tolerance_acc_gap = np.mean(np.abs(y_test_original[:, 1] - y_pred_rounded[:, 1]) <= tolerance_gap) * 100
    else:
        tolerance_acc_L = tolerance_acc_gap = 0.0
    
    print(f"📊 ULTRA ENSEMBLE RESULTS:")
    print(f"   R² global: {r2_global:.6f}")
    print(f"   R² L_ecran: {r2_L:.6f}")
    print(f"   R² gap: {r2_gap:.6f}")
    print(f"   RMSE L_ecran: {rmse_L:.6f} µm")
    print(f"   RMSE gap: {rmse_gap:.6f} µm")
    print(f"   Ultra tolerance L_ecran: {tolerance_acc_L:.2f}%")
    print(f"   Ultra tolerance gap: {tolerance_acc_gap:.2f}%")
    
    # Check ultra targets
    target_r2_L = config['evaluation']['performance_targets']['r2_L_ecran']
    target_r2_gap = config['evaluation']['performance_targets']['r2_gap']
    
    success_L = r2_L >= target_r2_L
    success_gap = r2_gap >= target_r2_gap
    ultra_success = success_L and success_gap
    
    print(f"   Ultra target L_ecran > {target_r2_L}: {'✅' if success_L else '❌'}")
    print(f"   Ultra target gap > {target_r2_gap}: {'✅' if success_gap else '❌'}")
    print(f"   Ultra success: {'🎉 YES' if ultra_success else '⚠️ NO'}")
    
    return y_pred_rounded, {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'tolerance_acc_L': tolerance_acc_L, 'tolerance_acc_gap': tolerance_acc_gap,
        'ultra_success': ultra_success
    }

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Ultra Specialized Regressor Training')
    parser.add_argument('--config', default='config/ultra_config.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 ULTRA SPECIALIZED REGRESSOR - MAXIMUM PERFORMANCE EDITION")
    print("Architecture ultra-optimisée avec ensemble training")
    print("="*80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Ultra data preprocessing
    X_scaled, y_scaled, y_original, scaler_X, scaler_L, scaler_gap = extract_and_preprocess_data(config)
    
    if X_scaled is None:
        print("❌ Failed to extract data. Training aborted.")
        return
    
    # Split data
    train_split = config.get('data_splits', {}).get('train', 0.8)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=1-train_split, random_state=42
    )
    
    # Get original validation data for evaluation
    y_val_original = scaler_L.inverse_transform(y_val[:, 0:1])
    y_val_gap_original = scaler_gap.inverse_transform(y_val[:, 1:2])
    y_val_original = np.hstack([y_val_original, y_val_gap_original])
    
    print(f"\n📊 Ultra data splits:")
    print(f"   Train: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    
    # Ultra ensemble training
    models, histories = train_ultra_ensemble(X_train, X_val, y_train, y_val, config)
    
    # Ultra evaluation
    y_pred, metrics = evaluate_ultra_ensemble(models, X_val, y_val_original, scaler_L, scaler_gap, config)
    
    # Save ultra scalers
    joblib.dump(scaler_X, 'models/ultra_scaler_X.pkl')
    joblib.dump(scaler_L, 'models/ultra_scaler_L.pkl')
    joblib.dump(scaler_gap, 'models/ultra_scaler_gap.pkl')
    
    print(f"\n{'='*80}")
    print(f"🎯 ULTRA SPECIALIZED RESULTS")
    print(f"{'='*80}")
    
    print(f"🔥 ULTRA FEATURES:")
    print(f"   • Architecture ultra-profonde avec double attention")
    print(f"   • Loss ultra-pondérée (gap x{config['training']['gap_weight']})")
    print(f"   • Ensemble de {len(models)} modèles spécialisés")
    print(f"   • Gradient clipping ultra-strict")
    print(f"   • Tolérance ultra-précise")
    
    print(f"\n📊 ULTRA PERFORMANCES:")
    print(f"   • R² global: {metrics['r2_global']:.6f}")
    print(f"   • R² L_ecran: {metrics['r2_L']:.6f}")
    print(f"   • R² gap: {metrics['r2_gap']:.6f}")
    print(f"   • Ultra success: {'🎉 YES' if metrics['ultra_success'] else '⚠️ NO'}")
    
    print(f"\n📁 ULTRA FILES:")
    print(f"   • models/ultra_model_*.pth")
    print(f"   • models/ultra_scaler_*.pkl")
    
    print("\n🏁 Ultra Specialized training completed!")

if __name__ == "__main__":
    main()
