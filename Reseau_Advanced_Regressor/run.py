#!/usr/bin/env python3
"""
Advanced Regressor with Attention - Autonomous Training Script
Author: Oussama GUELFAA
Date: 10 - 01 - 2025

Script autonome pour le régresseur avancé avec mécanisme d'attention.
Ce script résout systématiquement les 5 problèmes identifiés :
1. Précision excessive des labels → Arrondissement à 3 décimales
2. Échelles déséquilibrées → Normalisation séparée par paramètre
3. Distribution déséquilibrée → Focus sur plage expérimentale
4. Loss function inadaptée → Loss pondérée pour gap
5. Signal gap faible → Architecture spécialisée avec attention
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

class WeightedMSELoss(nn.Module):
    """Loss pondérée pour donner plus d'importance au gap."""
    
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.FloatTensor(weights)
    
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        weighted_mse = mse * self.weights.to(pred.device)
        return weighted_mse.mean()

class AdvancedRegressor(nn.Module):
    """Architecture avancée avec têtes spécialisées et attention pour gap."""
    
    def __init__(self, input_size=600):
        super(AdvancedRegressor, self).__init__()
        
        # Feature extractor commun
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
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
        
        # Tête L_ecran (signal fort)
        self.L_ecran_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Tête gap spécialisée (signal faible)
        self.gap_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.01),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.01),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Mécanisme d'attention pour gap
        self.gap_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Sigmoid()
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
        # Extraction de features communes
        features = self.feature_extractor(x)
        
        # Prédiction L_ecran (directe)
        L_ecran = self.L_ecran_head(features)
        
        # Prédiction gap avec attention
        attention_weights = self.gap_attention(features)
        attended_features = features * attention_weights
        gap = self.gap_head(attended_features)
        
        return torch.cat([L_ecran, gap], dim=1)

def load_config(config_path="config/advanced_config.yaml"):
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
        L_ecran_values = []
        gap_values = []
        
        for i in range(len(L_ecran_vect)):
            for j in range(len(gap_vect)):
                ratio = np.abs(I_subs[i, j, :600])  # Truncate to 600 points
                ratio_inc = np.abs(I_subs_inc[i, j, :600])
                profile = ratio / ratio_inc
                
                intensity_profiles.append(profile)
                L_ecran_values.append(L_ecran_vect[i])
                gap_values.append(gap_vect[j])
        
        intensity_profiles = np.array(intensity_profiles)
        parameters = np.column_stack([L_ecran_values, gap_values])
        
        print(f"✅ Extracted {len(intensity_profiles)} samples")
        print(f"   Profile shape: {intensity_profiles.shape}")
        print(f"   L_ecran range: {parameters[:, 0].min():.3f} to {parameters[:, 0].max():.3f}")
        print(f"   Gap range: {parameters[:, 1].min():.3f} to {parameters[:, 1].max():.3f}")
        
        return intensity_profiles, parameters
        
    except Exception as e:
        print(f"❌ Error extracting data: {e}")
        return None, None

def solve_problem_1_precision(y_data, precision=3):
    """PROBLÈME 1: Résoudre la précision excessive des labels."""
    
    print(f"🔢 RÉSOLUTION PROBLÈME 1: Précision excessive")
    print(f"   Arrondissement à {precision} décimales")
    
    y_original = y_data.copy()
    y_rounded = np.round(y_data, precision)
    
    # Analyser l'impact
    diff_L = np.abs(y_original[:, 0] - y_rounded[:, 0]).max()
    diff_gap = np.abs(y_original[:, 1] - y_rounded[:, 1]).max()
    
    print(f"   Différence max L_ecran: {diff_L:.2e}")
    print(f"   Différence max gap: {diff_gap:.2e}")
    print(f"   ✅ Labels arrondis à {precision} décimales")
    
    return y_rounded

def solve_problem_2_scaling(X_data, y_data):
    """PROBLÈME 2: Résoudre les échelles déséquilibrées."""
    
    print(f"⚖️ RÉSOLUTION PROBLÈME 2: Échelles déséquilibrées")
    
    # Normalisation des features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_data)
    
    # Normalisation séparée pour chaque paramètre
    scaler_L = StandardScaler()
    scaler_gap = StandardScaler()
    
    y_L_scaled = scaler_L.fit_transform(y_data[:, 0:1])
    y_gap_scaled = scaler_gap.fit_transform(y_data[:, 1:2])
    
    y_scaled = np.hstack([y_L_scaled, y_gap_scaled])
    
    print(f"   X normalisé: mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f}")
    print(f"   L_ecran normalisé: mean={y_L_scaled.mean():.6f}, std={y_L_scaled.std():.6f}")
    print(f"   gap normalisé: mean={y_gap_scaled.mean():.6f}, std={y_gap_scaled.std():.6f}")
    print(f"   ✅ Normalisation séparée appliquée")
    
    return X_scaled, y_scaled, scaler_X, scaler_L, scaler_gap

def solve_problem_3_distribution(X_data, y_data, focus_on_experimental=True):
    """PROBLÈME 3: Résoudre la distribution déséquilibrée."""
    
    print(f"📊 RÉSOLUTION PROBLÈME 3: Distribution déséquilibrée")
    
    if focus_on_experimental:
        # Focus sur la plage expérimentale gap [0.025, 0.517]
        experimental_mask = (y_data[:, 1] >= 0.025) & (y_data[:, 1] <= 0.517)
        
        X_focused = X_data[experimental_mask]
        y_focused = y_data[experimental_mask]
        
        print(f"   Données originales: {len(X_data)} échantillons")
        print(f"   Données focalisées: {len(X_focused)} échantillons")
        print(f"   Plage gap: [{y_focused[:, 1].min():.6f}, {y_focused[:, 1].max():.6f}]")
        print(f"   ✅ Focus sur plage expérimentale")
        
        return X_focused, y_focused
    
    return X_data, y_data

def preprocess_data(intensity_profiles, parameters, config):
    """Preprocess data solving problems 1-3."""
    
    print("="*80)
    print("PRÉTRAITEMENT AVANCÉ - RÉSOLUTION DES PROBLÈMES")
    print("="*80)
    
    # PROBLÈME 1: Précision excessive
    precision = config['data_preprocessing']['label_rounding']['decimals']
    y_rounded = solve_problem_1_precision(parameters, precision=precision)
    
    # PROBLÈME 3: Distribution déséquilibrée
    if config['data_preprocessing']['experimental_focus']['enable']:
        gap_range = config['data_preprocessing']['experimental_focus']['gap_range']
        experimental_mask = (y_rounded[:, 1] >= gap_range[0]) & (y_rounded[:, 1] <= gap_range[1])
        X_focused = intensity_profiles[experimental_mask]
        y_focused = y_rounded[experimental_mask]
        print(f"📊 Focus expérimental: {len(X_focused)}/{len(intensity_profiles)} échantillons")
    else:
        X_focused, y_focused = intensity_profiles, y_rounded
    
    # PROBLÈME 2: Échelles déséquilibrées
    X_scaled, y_scaled, scaler_X, scaler_L, scaler_gap = solve_problem_2_scaling(X_focused, y_focused)
    
    return X_scaled, y_scaled, scaler_X, scaler_L, scaler_gap

def train_advanced_model(X_train, X_val, y_train, y_val, config):
    """PROBLÈME 4 & 5: Entraîne avec loss pondérée et architecture spécialisée."""
    
    print(f"\n🎛️ RÉSOLUTION PROBLÈME 4: Loss pondérée")
    print(f"🔍 RÉSOLUTION PROBLÈME 5: Architecture spécialisée pour gap")
    
    # Créer le modèle avancé
    model = AdvancedRegressor(input_size=config['model']['input_size'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"📈 Modèle avancé créé - Device: {device}")
    
    # Convertir en tenseurs
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Datasets et DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # PROBLÈME 4: Loss pondérée
    gap_weight = config['training']['gap_weight']
    L_ecran_weight = config['training']['L_ecran_weight']
    criterion = WeightedMSELoss(weights=[L_ecran_weight, gap_weight])
    
    # Optimiseur
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    print(f"   Loss pondérée: L_ecran={L_ecran_weight}, gap={gap_weight}")
    print(f"   Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    
    # Entraînement
    history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['training']['early_stopping']['patience']
    
    epochs = config['training']['epochs']
    print(f"🚀 Début de l'entraînement avancé...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            if config['advanced_features']['gradient_clipping']['enable']:
                max_norm = config['advanced_features']['gradient_clipping']['max_norm']
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            train_loss += loss.item()
            
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        
        # Phase de validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculer R²
        train_predictions = np.array(train_predictions)
        train_targets = np.array(train_targets)
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        train_r2 = r2_score(train_targets, train_predictions)
        val_r2 = r2_score(val_targets, val_predictions)
        
        # Stocker l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        
        # Scheduler et early stopping
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/advanced_regressor_best.pth')
        else:
            patience_counter += 1
        
        if epoch % config['monitoring']['log_frequency'] == 0:
            print(f"   Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, "
                  f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
        
        if patience_counter >= patience:
            print(f"   ⏹️ Early stopping à l'epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"✅ Entraînement terminé en {training_time/60:.1f} minutes")
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load('models/advanced_regressor_best.pth'))
    
    return model, history, training_time

def evaluate_model(model, X_test, y_test, scaler_L, scaler_gap, config):
    """Évalue le modèle avancé."""
    
    print(f"\n🔍 ÉVALUATION DU MODÈLE AVANCÉ")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Prédictions
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    # Dénormaliser les prédictions
    y_pred_L = scaler_L.inverse_transform(y_pred_scaled[:, 0:1]).flatten()
    y_pred_gap = scaler_gap.inverse_transform(y_pred_scaled[:, 1:2]).flatten()
    y_pred = np.column_stack([y_pred_L, y_pred_gap])
    
    # Arrondir les prédictions pour cohérence
    precision = config['data_preprocessing']['label_rounding']['decimals']
    y_pred_rounded = np.round(y_pred, precision)
    
    # Calculer les métriques
    r2_global = r2_score(y_test, y_pred_rounded)
    r2_L = r2_score(y_test[:, 0], y_pred_rounded[:, 0])
    r2_gap = r2_score(y_test[:, 1], y_pred_rounded[:, 1])
    
    rmse_L = np.sqrt(mean_squared_error(y_test[:, 0], y_pred_rounded[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_test[:, 1], y_pred_rounded[:, 1]))
    
    mae_L = mean_absolute_error(y_test[:, 0], y_pred_rounded[:, 0])
    mae_gap = mean_absolute_error(y_test[:, 1], y_pred_rounded[:, 1])
    
    # Évaluation avec tolérance
    if config['evaluation']['tolerance_evaluation']['enable']:
        tolerance_L = config['evaluation']['tolerance_evaluation']['tolerance_L']
        tolerance_gap = config['evaluation']['tolerance_evaluation']['tolerance_gap']
        
        tolerance_acc_L = np.mean(np.abs(y_test[:, 0] - y_pred_rounded[:, 0]) <= tolerance_L) * 100
        tolerance_acc_gap = np.mean(np.abs(y_test[:, 1] - y_pred_rounded[:, 1]) <= tolerance_gap) * 100
    else:
        tolerance_acc_L = tolerance_acc_gap = 0.0
    
    print(f"📊 RÉSULTATS ADVANCED REGRESSOR:")
    print(f"   R² global: {r2_global:.6f}")
    print(f"   R² L_ecran: {r2_L:.6f}")
    print(f"   R² gap: {r2_gap:.6f}")
    print(f"   RMSE L_ecran: {rmse_L:.6f} µm")
    print(f"   RMSE gap: {rmse_gap:.6f} µm")
    print(f"   MAE L_ecran: {mae_L:.6f} µm")
    print(f"   MAE gap: {mae_gap:.6f} µm")
    print(f"   Tolérance L_ecran: {tolerance_acc_L:.2f}%")
    print(f"   Tolérance gap: {tolerance_acc_gap:.2f}%")
    
    # Vérifier les objectifs
    target_r2_L = config['evaluation']['performance_targets']['r2_L_ecran']
    target_r2_gap = config['evaluation']['performance_targets']['r2_gap']
    
    success_L = r2_L >= target_r2_L
    success_gap = r2_gap >= target_r2_gap
    overall_success = success_L and success_gap
    
    print(f"   Objectif R² L_ecran > {target_r2_L}: {'✅' if success_L else '❌'}")
    print(f"   Objectif R² gap > {target_r2_gap}: {'✅' if success_gap else '❌'}")
    print(f"   Succès global: {'🎉 OUI' if overall_success else '⚠️ NON'}")
    
    return y_pred_rounded, {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'tolerance_acc_L': tolerance_acc_L, 'tolerance_acc_gap': tolerance_acc_gap,
        'success': overall_success
    }

def save_results(history, metrics, config):
    """Save training results and metrics."""
    print("\n💾 Saving results...")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('results/training_history.csv', index=False)
    
    # Save metrics
    with open('results/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save configuration
    with open('results/config_used.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Results saved successfully!")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Advanced Regressor Training')
    parser.add_argument('--config', default='config/advanced_config.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 ADVANCED REGRESSOR - PROBLEM-SOLVING EDITION")
    print("Résolution systématique des 5 problèmes identifiés")
    print("="*80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Extract data
    intensity_profiles, parameters = extract_data_from_matlab()
    
    if intensity_profiles is None:
        print("❌ Failed to extract data. Training aborted.")
        return
    
    # Preprocess data (solve problems 1-3)
    X_scaled, y_scaled, scaler_X, scaler_L, scaler_gap = preprocess_data(
        intensity_profiles, parameters, config
    )
    
    # Split data
    train_split = config['data_splits']['train']
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=1-train_split, random_state=42
    )
    
    print(f"\n📊 Division des données:")
    print(f"   Train: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    
    # Train model (solve problems 4-5)
    model, history, training_time = train_advanced_model(X_train, X_val, y_train, y_val, config)
    
    # Evaluate model
    # For evaluation, we'll use the validation set as test set
    y_val_original = scaler_L.inverse_transform(y_val[:, 0:1])
    y_val_gap_original = scaler_gap.inverse_transform(y_val[:, 1:2])
    y_val_original = np.hstack([y_val_original, y_val_gap_original])
    
    y_pred, metrics = evaluate_model(model, X_val, y_val_original, scaler_L, scaler_gap, config)
    
    # Save scalers
    joblib.dump(scaler_X, 'models/advanced_regressor_scaler_X.pkl')
    joblib.dump(scaler_L, 'models/advanced_regressor_scaler_L.pkl')
    joblib.dump(scaler_gap, 'models/advanced_regressor_scaler_gap.pkl')
    
    # Save results
    save_results(history, metrics, config)
    
    print(f"\n{'='*80}")
    print(f"🎯 RÉSULTATS FINAUX ADVANCED REGRESSOR")
    print(f"{'='*80}")
    
    print(f"✅ PROBLÈMES RÉSOLUS:")
    print(f"   1. 🔢 Labels arrondis à 3 décimales")
    print(f"   2. ⚖️ Normalisation séparée par paramètre")
    print(f"   3. 📊 Focus sur plage expérimentale")
    print(f"   4. 🎛️ Loss pondérée (gap x{config['training']['gap_weight']})")
    print(f"   5. 🔍 Architecture spécialisée + attention pour gap")
    
    print(f"\n📊 PERFORMANCES:")
    print(f"   • R² global: {metrics['r2_global']:.6f}")
    print(f"   • R² L_ecran: {metrics['r2_L']:.6f}")
    print(f"   • R² gap: {metrics['r2_gap']:.6f}")
    print(f"   • Objectif atteint: {'🎉 OUI' if metrics['success'] else '⚠️ NON'}")
    
    print(f"\n📁 FICHIERS GÉNÉRÉS:")
    print(f"   • models/advanced_regressor_best.pth")
    print(f"   • models/advanced_regressor_scaler_*.pkl")
    print(f"   • results/training_history.csv")
    print(f"   • results/evaluation_metrics.json")
    
    print("\n🏁 Advanced Regressor training completed!")

if __name__ == "__main__":
    main()
