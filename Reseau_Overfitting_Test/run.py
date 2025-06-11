#!/usr/bin/env python3
"""
Overfitting Validation Test - Autonomous Testing Script
Author: Oussama GUELFAA
Date: 10 - 01 - 2025

Script autonome pour tester la capacité d'overfitting du réseau de neurones.
Ce test valide que le modèle peut mémoriser parfaitement les données d'entraînement
en utilisant les mêmes données pour l'entraînement et la validation.
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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration pour reproductibilité
torch.manual_seed(42)
np.random.seed(42)

class IntensityDataset(Dataset):
    """Dataset PyTorch pour les profils d'intensité."""
    
    def __init__(self, intensity_profiles, gap_values):
        self.intensity_profiles = torch.FloatTensor(intensity_profiles)
        self.gap_values = torch.FloatTensor(gap_values)
    
    def __len__(self):
        return len(self.intensity_profiles)
    
    def __getitem__(self, idx):
        return self.intensity_profiles[idx], self.gap_values[idx]

class SimpleOverfittingModel(nn.Module):
    """Modèle simple sans régularisation pour test d'overfitting."""
    
    def __init__(self, input_size=600):
        super(SimpleOverfittingModel, self).__init__()
        
        # Architecture simple sans dropout ni batch norm
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
        # Initialisation Xavier pour stabilité
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_config(config_path="config/overfitting_config.yaml"):
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

def prepare_overfitting_data(intensity_profiles, gap_values, config):
    """Prepare data for overfitting test (same data for train and validation)."""
    print("📊 Preparing data for overfitting test...")
    
    # Normalisation
    scaler = StandardScaler()
    intensity_profiles_scaled = scaler.fit_transform(intensity_profiles)
    
    # Pour le test d'overfitting, on utilise les mêmes données pour train et validation
    X_train = intensity_profiles_scaled
    X_val = intensity_profiles_scaled  # Mêmes données !
    y_train = gap_values
    y_val = gap_values  # Mêmes données !
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)} (same as train)")
    print(f"   Using same data for train and validation: {np.array_equal(X_train, X_val)}")
    
    return X_train, X_val, y_train, y_val, scaler

def train_overfitting_model(X_train, y_train, X_val, y_val, config):
    """Train model for overfitting validation."""
    print("\n🚀 OVERFITTING VALIDATION TEST")
    print("=" * 50)
    
    start_time = time.time()
    
    # Create datasets (same data for train and validation)
    train_dataset = IntensityDataset(X_train, y_train)
    val_dataset = IntensityDataset(X_val, y_val)
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleOverfittingModel(input_size=X_train.shape[1]).to(device)
    
    # Optimizer without weight decay (no regularization)
    learning_rate = config['training']['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': [],
        'epochs': [], 'gradient_norms': []
    }
    
    max_epochs = config['training']['epochs']
    log_frequency = config['monitoring']['log_frequency']
    
    print(f"📈 Training on {device}")
    print(f"   Target: R² > {config['validation_criteria']['target_train_r2']}")
    print(f"   Target: Loss < {config['validation_criteria']['target_train_loss']}")
    
    best_train_r2 = -float('inf')
    best_val_r2 = -float('inf')
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        total_grad_norm = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Calculate gradient norm
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            total_grad_norm += grad_norm
            
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_targets.extend(batch_y.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        train_r2 = r2_score(train_targets, train_predictions)
        avg_grad_norm = total_grad_norm / len(train_loader)
        
        # Validation phase (same data)
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
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['epochs'].append(epoch + 1)
        history['gradient_norms'].append(avg_grad_norm)
        
        # Track best performance
        if train_r2 > best_train_r2:
            best_train_r2 = train_r2
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
        
        # Log progress
        if (epoch + 1) % log_frequency == 0:
            print(f"   Epoch {epoch+1:3d}: Train R²={train_r2:.6f}, Val R²={val_r2:.6f}, "
                  f"Train Loss={train_loss:.8f}, Val Loss={val_loss:.8f}")
        
        # Check if overfitting criteria are met
        target_r2 = config['validation_criteria']['target_train_r2']
        target_loss = config['validation_criteria']['target_train_loss']
        
        if train_r2 >= target_r2 and val_r2 >= target_r2 and train_loss <= target_loss and val_loss <= target_loss:
            print(f"\n🎯 OVERFITTING CRITERIA MET at epoch {epoch+1}!")
            print(f"   Train R²: {train_r2:.6f} >= {target_r2}")
            print(f"   Val R²: {val_r2:.6f} >= {target_r2}")
            print(f"   Train Loss: {train_loss:.8f} <= {target_loss}")
            print(f"   Val Loss: {val_loss:.8f} <= {target_loss}")
            break
    
    training_time = time.time() - start_time
    
    # Save model
    model_path = config['paths']['model_save']
    torch.save(model.state_dict(), model_path)
    
    print(f"\n✅ Training completed in {training_time:.1f}s")
    print(f"   Best Train R²: {best_train_r2:.6f}")
    print(f"   Best Val R²: {best_val_r2:.6f}")
    print(f"   Final Train Loss: {train_loss:.8f}")
    print(f"   Final Val Loss: {val_loss:.8f}")
    
    return model, history, training_time, epoch + 1

def evaluate_overfitting_success(history, config):
    """Evaluate if overfitting test was successful."""
    print("\n🔍 EVALUATING OVERFITTING SUCCESS")
    print("=" * 40)
    
    final_train_r2 = history['train_r2'][-1]
    final_val_r2 = history['val_r2'][-1]
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    target_r2 = config['validation_criteria']['target_train_r2']
    target_loss = config['validation_criteria']['target_train_loss']
    
    # Check criteria
    r2_success = final_train_r2 >= target_r2 and final_val_r2 >= target_r2
    loss_success = final_train_loss <= target_loss and final_val_loss <= target_loss
    
    # Check if train and val are similar (they should be identical)
    r2_diff = abs(final_train_r2 - final_val_r2)
    loss_diff = abs(final_train_loss - final_val_loss)
    
    similarity_success = r2_diff < 0.001 and loss_diff < 0.001
    
    overall_success = r2_success and loss_success and similarity_success
    
    results = {
        'overall_success': overall_success,
        'r2_success': r2_success,
        'loss_success': loss_success,
        'similarity_success': similarity_success,
        'final_metrics': {
            'train_r2': final_train_r2,
            'val_r2': final_val_r2,
            'train_loss': final_train_loss,
            'val_loss': final_val_loss,
            'r2_difference': r2_diff,
            'loss_difference': loss_diff
        },
        'criteria': {
            'target_r2': target_r2,
            'target_loss': target_loss
        }
    }
    
    print(f"   R² Success: {r2_success} (Train: {final_train_r2:.6f}, Val: {final_val_r2:.6f})")
    print(f"   Loss Success: {loss_success} (Train: {final_train_loss:.8f}, Val: {final_val_loss:.8f})")
    print(f"   Similarity Success: {similarity_success} (R² diff: {r2_diff:.6f}, Loss diff: {loss_diff:.8f})")
    print(f"   Overall Success: {overall_success}")
    
    if overall_success:
        print("\n🎉 OVERFITTING TEST PASSED!")
        print("   The model successfully memorized the training data.")
    else:
        print("\n❌ OVERFITTING TEST FAILED!")
        print("   The model could not perfectly memorize the training data.")
        print("   This indicates potential issues with architecture or training.")
    
    return results

def create_overfitting_plots(history, config):
    """Create visualization plots for overfitting analysis."""
    print("\n📊 Generating overfitting analysis plots...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(2, 3, 1)
    epochs = history['epochs']
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Training and Validation R²
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['train_r2'], 'b-', label='Train R²', linewidth=2)
    plt.plot(epochs, history['val_r2'], 'r--', label='Val R²', linewidth=2)
    plt.axhline(y=0.99, color='green', linestyle=':', label='Target R² = 0.99')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Training and Validation R²')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Gradient Norms
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['gradient_norms'], 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms During Training')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 4: Loss Convergence (last 50 epochs)
    plt.subplot(2, 3, 4)
    if len(epochs) > 50:
        recent_epochs = epochs[-50:]
        recent_train_loss = history['train_loss'][-50:]
        recent_val_loss = history['val_loss'][-50:]
    else:
        recent_epochs = epochs
        recent_train_loss = history['train_loss']
        recent_val_loss = history['val_loss']
    
    plt.plot(recent_epochs, recent_train_loss, 'b-', label='Train Loss', linewidth=2)
    plt.plot(recent_epochs, recent_val_loss, 'r--', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Convergence (Recent Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 5: R² Convergence (last 50 epochs)
    plt.subplot(2, 3, 5)
    if len(epochs) > 50:
        recent_r2_train = history['train_r2'][-50:]
        recent_r2_val = history['val_r2'][-50:]
    else:
        recent_r2_train = history['train_r2']
        recent_r2_val = history['val_r2']
    
    plt.plot(recent_epochs, recent_r2_train, 'b-', label='Train R²', linewidth=2)
    plt.plot(recent_epochs, recent_r2_val, 'r--', label='Val R²', linewidth=2)
    plt.axhline(y=0.99, color='green', linestyle=':', label='Target R² = 0.99')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('R² Convergence (Recent Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Train vs Val Difference
    plt.subplot(2, 3, 6)
    r2_diff = [abs(t - v) for t, v in zip(history['train_r2'], history['val_r2'])]
    loss_diff = [abs(t - v) for t, v in zip(history['train_loss'], history['val_loss'])]
    
    plt.plot(epochs, r2_diff, 'purple', label='R² Difference', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Absolute Difference')
    plt.title('Train vs Val Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Plots saved to plots/overfitting_analysis.png")

def save_overfitting_results(history, evaluation_results, training_time, final_epoch, config):
    """Save overfitting test results."""
    print("\n💾 Saving overfitting test results...")
    
    results = {
        'test_info': {
            'test_type': 'overfitting_validation',
            'timestamp': datetime.now().isoformat(),
            'training_time_seconds': training_time,
            'final_epoch': final_epoch,
            'total_epochs_run': len(history['epochs'])
        },
        'evaluation': evaluation_results,
        'training_history': {
            'epochs': history['epochs'],
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_r2': history['train_r2'],
            'val_r2': history['val_r2'],
            'gradient_norms': history['gradient_norms']
        },
        'configuration': config
    }
    
    # Save detailed results
    with open('results/overfitting_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary CSV
    summary_data = {
        'test_success': [evaluation_results['overall_success']],
        'final_train_r2': [evaluation_results['final_metrics']['train_r2']],
        'final_val_r2': [evaluation_results['final_metrics']['val_r2']],
        'final_train_loss': [evaluation_results['final_metrics']['train_loss']],
        'final_val_loss': [evaluation_results['final_metrics']['val_loss']],
        'training_time_s': [training_time],
        'epochs_to_completion': [final_epoch]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/overfitting_test_summary.csv', index=False)
    
    print("✅ Results saved successfully!")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Overfitting Validation Test')
    parser.add_argument('--config', default='config/overfitting_config.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    print("🧪 OVERFITTING VALIDATION TEST")
    print("=" * 60)
    print("Purpose: Validate model's ability to perfectly memorize training data")
    print("Method: Use same data for training and validation")
    print("Success: R² > 0.99 and Loss < 0.001 on both train and val")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Extract data
    intensity_profiles, gap_values = extract_data_from_matlab()
    
    if intensity_profiles is None:
        print("❌ Failed to extract data. Test aborted.")
        return
    
    # Prepare data for overfitting test
    X_train, X_val, y_train, y_val, scaler = prepare_overfitting_data(
        intensity_profiles, gap_values, config
    )
    
    # Train model for overfitting
    model, history, training_time, final_epoch = train_overfitting_model(
        X_train, y_train, X_val, y_val, config
    )
    
    # Evaluate overfitting success
    evaluation_results = evaluate_overfitting_success(history, config)
    
    # Create plots
    create_overfitting_plots(history, config)
    
    # Save results
    save_overfitting_results(history, evaluation_results, training_time, final_epoch, config)
    
    print("\n🏁 Overfitting validation test completed!")
    
    if evaluation_results['overall_success']:
        print("🎉 TEST PASSED: Model can successfully overfit!")
        print("   ✅ Architecture and training are working correctly")
        print("   ✅ Ready for generalization testing with separate data")
    else:
        print("❌ TEST FAILED: Model cannot overfit properly")
        print("   ⚠️  Check architecture, learning rate, or training parameters")
        print("   ⚠️  Model may have fundamental issues")

if __name__ == "__main__":
    main()
