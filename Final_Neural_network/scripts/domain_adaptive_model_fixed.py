#!/usr/bin/env python3
"""
Domain Adaptive Neural Network for Holographic Ring Analysis (Fixed Version)
Author: Oussama GUELFAA
Date: 01/08/2025

This is a simplified and more robust version of the domain adaptive model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for domain adaptation."""
    @staticmethod
    def forward(ctx, x, lambda_param=1.0):
        ctx.lambda_param = lambda_param
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_param * grad_output, None

class SimplifiedDomainAdaptiveModel(nn.Module):
    """Simplified Domain Adaptive Neural Network."""
    
    def __init__(self, input_size=750):
        super(SimplifiedDomainAdaptiveModel, self).__init__()
        
        # Feature Extractor with proper size calculation
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate conv output size: 750 -> 375 -> 187
        conv_output_size = 64 * (input_size // 4)
        
        self.feature_dense = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Regression Head
        self.regressor = nn.Linear(128, 2)
        
        # Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, lambda_param=1.0):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        features = self.feature_dense(x)
        
        # Regression output
        regression_output = self.regressor(features)
        
        # Domain classification with gradient reversal
        reversed_features = GradientReversalLayer.apply(features, lambda_param)
        domain_output = torch.sigmoid(self.domain_classifier(reversed_features))
        
        return {
            'regression': regression_output,
            'domain': domain_output,
            'features': features
        }

def load_and_preprocess_data():
    """Load and preprocess data with better error handling."""
    print("Loading and preprocessing data...")
    
    # Load simulation data
    sim_data = np.load('simulation_processed_with_labels.npz')
    X_sim = sim_data['X_data'].astype(np.float32)
    y_sim = sim_data['y_data'].astype(np.float32)
    
    # Load experimental data
    exp_data = np.load('experimental_processed_interp_to_sim_grid.npz')
    X_exp = exp_data['X_data'].astype(np.float32)
    
    print(f"Data loaded: X_sim {X_sim.shape}, y_sim {y_sim.shape}, X_exp {X_exp.shape}")
    
    # Normalize features
    X_scaler = StandardScaler()
    X_sim_norm = X_scaler.fit_transform(X_sim)
    X_exp_norm = X_scaler.transform(X_exp)
    
    # Normalize targets
    y_scaler = StandardScaler()
    y_sim_norm = y_scaler.fit_transform(y_sim)
    
    # Reshape for Conv1D
    X_sim_reshaped = X_sim_norm.reshape(-1, 1, 750)
    X_exp_reshaped = X_exp_norm.reshape(-1, 1, 750)
    
    print(f"Preprocessing completed. Shapes: X_sim {X_sim_reshaped.shape}, X_exp {X_exp_reshaped.shape}")
    
    return {
        'X_sim': X_sim_reshaped,
        'y_sim': y_sim_norm,
        'X_exp': X_exp_reshaped,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler
    }

def create_datasets_and_loaders(data_dict, batch_size=32, val_split=0.2):
    """Create datasets and data loaders."""
    print(f"Creating data loaders with batch_size={batch_size}...")
    
    # Split simulation data
    X_train, X_val, y_train, y_val = train_test_split(
        data_dict['X_sim'], data_dict['y_sim'], 
        test_size=val_split, random_state=42
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_exp_tensor = torch.FloatTensor(data_dict['X_exp'])
    
    # Create domain labels
    domain_train = torch.zeros(len(X_train), 1)  # 0 = simulation
    domain_val = torch.zeros(len(X_val), 1)
    domain_exp = torch.ones(len(data_dict['X_exp']), 1)  # 1 = experimental
    
    print(f"Data splits: Train {len(X_train)}, Val {len(X_val)}, Exp {len(data_dict['X_exp'])}")
    
    return {
        'X_train': X_train_tensor,
        'y_train': y_train_tensor,
        'domain_train': domain_train,
        'X_val': X_val_tensor,
        'y_val': y_val_tensor,
        'domain_val': domain_val,
        'X_exp': X_exp_tensor,
        'domain_exp': domain_exp
    }

def train_model_simplified(model, data_tensors, num_epochs=50, learning_rate=1e-3):
    """Simplified training loop with better stability."""
    print(f"Starting training for {num_epochs} epochs...")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Move data to device
    for key in data_tensors:
        data_tensors[key] = data_tensors[key].to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_reg_loss': [],
        'val_reg_loss': [],
        'domain_loss': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    batch_size = 32
    n_train = len(data_tensors['X_train'])
    
    for epoch in range(num_epochs):
        model.train()
        
        # Compute lambda with gradual increase
        lambda_param = min(0.1 * (epoch / 10), 0.5)  # Gradually increase to 0.5
        
        epoch_train_loss = 0
        epoch_reg_loss = 0
        epoch_domain_loss = 0
        n_batches = 0
        
        # Training loop
        for i in range(0, n_train, batch_size):
            end_idx = min(i + batch_size, n_train)
            
            # Simulation batch
            X_batch = data_tensors['X_train'][i:end_idx]
            y_batch = data_tensors['y_train'][i:end_idx]
            domain_batch = data_tensors['domain_train'][i:end_idx]
            
            # Experimental batch (sample randomly)
            exp_indices = torch.randint(0, len(data_tensors['X_exp']), (len(X_batch),))
            X_exp_batch = data_tensors['X_exp'][exp_indices]
            domain_exp_batch = data_tensors['domain_exp'][exp_indices]
            
            optimizer.zero_grad()
            
            # Forward pass on simulation data
            sim_outputs = model(X_batch, lambda_param)
            reg_loss = F.mse_loss(sim_outputs['regression'], y_batch)
            domain_loss_sim = F.binary_cross_entropy(sim_outputs['domain'], domain_batch)
            
            # Forward pass on experimental data
            exp_outputs = model(X_exp_batch, lambda_param)
            domain_loss_exp = F.binary_cross_entropy(exp_outputs['domain'], domain_exp_batch)
            
            # Combined loss
            domain_loss = (domain_loss_sim + domain_loss_exp) / 2
            total_loss = reg_loss + lambda_param * domain_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            # Accumulate losses
            epoch_train_loss += total_loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_domain_loss += domain_loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(data_tensors['X_val'], lambda_param)
            val_reg_loss = F.mse_loss(val_outputs['regression'], data_tensors['y_val'])
            val_domain_loss = F.binary_cross_entropy(val_outputs['domain'], data_tensors['domain_val'])
            val_total_loss = val_reg_loss + lambda_param * val_domain_loss
        
        # Update history
        history['train_loss'].append(epoch_train_loss / n_batches)
        history['val_loss'].append(val_total_loss.item())
        history['train_reg_loss'].append(epoch_reg_loss / n_batches)
        history['val_reg_loss'].append(val_reg_loss.item())
        history['domain_loss'].append(epoch_domain_loss / n_batches)
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:3d}: Train Loss {epoch_train_loss/n_batches:.4f}, "
                  f"Val Loss {val_total_loss:.4f}, Reg Loss {val_reg_loss:.4f}, "
                  f"Lambda {lambda_param:.3f}")
        
        # Early stopping
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate_and_predict(model, data_tensors, data_dict):
    """Evaluate model and make predictions."""
    print("Evaluating model and making predictions...")

    model.eval()

    with torch.no_grad():
        # Validation predictions
        val_outputs = model(data_tensors['X_val'])
        val_predictions = val_outputs['regression'].cpu().numpy()
        val_targets = data_tensors['y_val'].cpu().numpy()

        # Experimental predictions
        exp_outputs = model(data_tensors['X_exp'])
        exp_predictions = exp_outputs['regression'].cpu().numpy()

    # Inverse transform
    y_scaler = data_dict['y_scaler']
    val_predictions_orig = y_scaler.inverse_transform(val_predictions)
    val_targets_orig = y_scaler.inverse_transform(val_targets)
    exp_predictions_orig = y_scaler.inverse_transform(exp_predictions)

    # Compute metrics
    val_mse = np.mean((val_predictions_orig - val_targets_orig) ** 2, axis=0)
    val_mae = np.mean(np.abs(val_predictions_orig - val_targets_orig), axis=0)
    val_r2 = [r2_score(val_targets_orig[:, i], val_predictions_orig[:, i]) for i in range(2)]

    print(f"Validation Results:")
    print(f"  Gap - MSE: {val_mse[0]:.6f}, MAE: {val_mae[0]:.6f}, R²: {val_r2[0]:.3f}")
    print(f"  L_ecran - MSE: {val_mse[1]:.6f}, MAE: {val_mae[1]:.6f}, R²: {val_r2[1]:.3f}")

    print(f"Experimental Predictions:")
    print(f"  Gap range: {exp_predictions_orig[:, 0].min():.3f} to {exp_predictions_orig[:, 0].max():.3f} µm")
    print(f"  L_ecran range: {exp_predictions_orig[:, 1].min():.3f} to {exp_predictions_orig[:, 1].max():.3f} µm")

    return {
        'val_predictions': val_predictions_orig,
        'val_targets': val_targets_orig,
        'exp_predictions': exp_predictions_orig,
        'val_mse': val_mse,
        'val_mae': val_mae,
        'val_r2': val_r2
    }

def save_results_and_visualize(model, history, evaluation_results, data_dict):
    """Save results and create visualizations."""
    print("Saving results and creating visualizations...")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'X_scaler': data_dict['X_scaler'],
        'y_scaler': data_dict['y_scaler']
    }, 'domain_adaptive_model_fixed.pt')

    # Save experimental predictions
    exp_predictions = evaluation_results['exp_predictions']
    predictions_df = pd.DataFrame({
        'sample_id': range(len(exp_predictions)),
        'predicted_gap_um': exp_predictions[:, 0],
        'predicted_L_um': exp_predictions[:, 1]
    })
    predictions_df.to_csv('experimental_predictions_fixed.csv', index=False)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv('training_history_fixed.csv', index=False)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Training curves
    ax1 = axes[0, 0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.plot(epochs, history['train_reg_loss'], 'b--', label='Train Reg Loss', alpha=0.7)
    ax1.plot(epochs, history['val_reg_loss'], 'r--', label='Val Reg Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gap parameter validation
    ax2 = axes[0, 1]
    val_targets = evaluation_results['val_targets']
    val_predictions = evaluation_results['val_predictions']
    ax2.scatter(val_targets[:, 0], val_predictions[:, 0], alpha=0.6, s=20)
    min_gap = min(val_targets[:, 0].min(), val_predictions[:, 0].min())
    max_gap = max(val_targets[:, 0].max(), val_predictions[:, 0].max())
    ax2.plot([min_gap, max_gap], [min_gap, max_gap], 'r--', linewidth=2)
    ax2.set_xlabel('True Gap (µm)')
    ax2.set_ylabel('Predicted Gap (µm)')
    ax2.set_title(f'Gap Parameter (R² = {evaluation_results["val_r2"][0]:.3f})')
    ax2.grid(True, alpha=0.3)

    # L_ecran parameter validation
    ax3 = axes[1, 0]
    ax3.scatter(val_targets[:, 1], val_predictions[:, 1], alpha=0.6, s=20)
    min_L = min(val_targets[:, 1].min(), val_predictions[:, 1].min())
    max_L = max(val_targets[:, 1].max(), val_predictions[:, 1].max())
    ax3.plot([min_L, max_L], [min_L, max_L], 'r--', linewidth=2)
    ax3.set_xlabel('True L_ecran (µm)')
    ax3.set_ylabel('Predicted L_ecran (µm)')
    ax3.set_title(f'L_ecran Parameter (R² = {evaluation_results["val_r2"][1]:.3f})')
    ax3.grid(True, alpha=0.3)

    # Experimental predictions
    ax4 = axes[1, 1]
    ax4.scatter(exp_predictions[:, 0], exp_predictions[:, 1],
               alpha=0.8, s=50, color='red', label='Experimental')

    # Load original simulation data for comparison
    sim_data = np.load('simulation_processed_with_labels.npz')
    y_sim_orig = sim_data['y_data']
    ax4.scatter(y_sim_orig[::100, 0], y_sim_orig[::100, 1],
               alpha=0.3, s=1, color='blue', label='Simulation (subset)')

    ax4.set_xlabel('Gap Parameter (µm)')
    ax4.set_ylabel('L_ecran Parameter (µm)')
    ax4.set_title('Experimental vs Simulation Parameter Space')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('domain_adaptive_results_fixed.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✓ Results saved successfully!")

def main():
    """Main execution function."""
    print("=" * 80)
    print("DOMAIN ADAPTIVE NEURAL NETWORK (FIXED VERSION)")
    print("=" * 80)

    try:
        # Load and preprocess data
        data_dict = load_and_preprocess_data()

        # Create datasets and loaders
        data_tensors = create_datasets_and_loaders(data_dict, batch_size=32)

        # Initialize model
        model = SimplifiedDomainAdaptiveModel(input_size=750)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {total_params:,} parameters")

        # Train model
        trained_model, history = train_model_simplified(
            model, data_tensors, num_epochs=50, learning_rate=1e-3
        )

        # Evaluate and predict
        evaluation_results = evaluate_and_predict(trained_model, data_tensors, data_dict)

        # Save results and visualize
        save_results_and_visualize(trained_model, history, evaluation_results, data_dict)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("Files generated:")
        print("  ✓ domain_adaptive_model_fixed.pt")
        print("  ✓ experimental_predictions_fixed.csv")
        print("  ✓ training_history_fixed.csv")
        print("  ✓ domain_adaptive_results_fixed.png")
        print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
