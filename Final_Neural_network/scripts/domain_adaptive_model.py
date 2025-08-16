#!/usr/bin/env python3
"""
Domain Adaptive Neural Network for Holographic Ring Analysis
Author: Oussama GUELFAA
Date: 01/08/2025

This script implements a domain-adaptive neural network that learns from simulation data
and adapts to experimental data using gradient reversal for domain adaptation.

Architecture:
- Shared Feature Extractor (Conv1D layers)
- Regression Head (for gap and L_ecran prediction)
- Domain Classifier (with Gradient Reversal Layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for domain adaptation.
    Forward pass: identity function
    Backward pass: reverses gradients
    """
    @staticmethod
    def forward(ctx, x, lambda_param=1.0):
        ctx.lambda_param = lambda_param
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_param * grad_output, None

class DomainAdaptiveModel(nn.Module):
    """
    Domain Adaptive Neural Network with shared feature extractor,
    regression head, and domain classifier.
    """
    def __init__(self, input_size=750):
        super(DomainAdaptiveModel, self).__init__()
        
        # Shared Feature Extractor
        self.feature_extractor = nn.Sequential(
            # First Conv1D block
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            
            # Second Conv1D block
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            # Flatten and Dense layers
            nn.Flatten(),
        )
        
        # Calculate the size after conv layers
        # input_size=750 -> after 2 maxpool(2): 750/4 = 187.5 -> 187
        conv_output_size = 64 * (input_size // 4)
        
        self.feature_dense = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Regression Head (for gap and L_ecran prediction)
        self.regressor = nn.Sequential(
            nn.Linear(128, 2)  # outputs: [gap_um, L_um]
        )
        
        # Domain Classifier (with GRL)
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # domain: 0 = sim, 1 = exp
        )
    
    def forward(self, x, lambda_param=1.0, return_features=False):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (N, 1, 750)
            lambda_param: Lambda parameter for gradient reversal
            return_features: Whether to return intermediate features
        
        Returns:
            dict: Contains regression and domain predictions
        """
        # Extract features
        conv_features = self.feature_extractor(x)
        features = self.feature_dense(conv_features)
        
        # Regression prediction
        regression_output = self.regressor(features)
        
        # Domain prediction with gradient reversal
        reversed_features = GradientReversalLayer.apply(features, lambda_param)
        domain_output = self.domain_classifier(reversed_features)
        
        outputs = {
            'regression': regression_output,
            'domain': domain_output
        }
        
        if return_features:
            outputs['features'] = features
        
        return outputs

class HolographicDataset(Dataset):
    """
    Custom dataset for holographic ring data with domain labels.
    """
    def __init__(self, X_data, y_data=None, domain_labels=None):
        self.X_data = torch.FloatTensor(X_data)
        self.y_data = torch.FloatTensor(y_data) if y_data is not None else None
        self.domain_labels = torch.FloatTensor(domain_labels) if domain_labels is not None else None
        
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, idx):
        sample = {'X': self.X_data[idx]}
        
        if self.y_data is not None:
            sample['y'] = self.y_data[idx]
        
        if self.domain_labels is not None:
            sample['domain'] = self.domain_labels[idx]
        
        return sample

def load_and_preprocess_data():
    """
    Load and preprocess both simulation and experimental data.
    
    Returns:
        tuple: Processed datasets and scalers
    """
    print("=" * 70)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    
    # Load simulation data
    print("Loading simulation data...")
    sim_data = np.load('simulation_processed_with_labels.npz')
    X_sim = sim_data['X_data']  # (22540, 750)
    y_sim = sim_data['y_data']  # (22540, 2)
    
    print(f"Simulation data: X_sim {X_sim.shape}, y_sim {y_sim.shape}")
    
    # Load experimental data
    print("Loading experimental data...")
    exp_data = np.load('experimental_processed_interp_to_sim_grid.npz')
    X_exp = exp_data['X_data']  # (50, 750)
    
    print(f"Experimental data: X_exp {X_exp.shape}")
    
    # Normalize X_data using simulation statistics
    print("\nNormalizing input data...")
    X_scaler = StandardScaler()
    X_sim_normalized = X_scaler.fit_transform(X_sim)
    X_exp_normalized = X_scaler.transform(X_exp)
    
    print(f"X normalization - Sim mean: {X_sim_normalized.mean():.6f}, std: {X_sim_normalized.std():.6f}")
    print(f"X normalization - Exp mean: {X_exp_normalized.mean():.6f}, std: {X_exp_normalized.std():.6f}")
    
    # Normalize y_data (targets)
    print("Normalizing target data...")
    y_scaler = StandardScaler()
    y_sim_normalized = y_scaler.fit_transform(y_sim)
    
    print(f"Y normalization - Original range: Gap [{y_sim[:, 0].min():.3f}, {y_sim[:, 0].max():.3f}], L [{y_sim[:, 1].min():.3f}, {y_sim[:, 1].max():.3f}]")
    print(f"Y normalization - Normalized mean: {y_sim_normalized.mean():.6f}, std: {y_sim_normalized.std():.6f}")
    
    # Reshape for Conv1D: (N, 1, 750)
    X_sim_reshaped = X_sim_normalized.reshape(-1, 1, 750)
    X_exp_reshaped = X_exp_normalized.reshape(-1, 1, 750)
    
    print(f"\nReshaped for Conv1D:")
    print(f"X_sim: {X_sim_reshaped.shape}")
    print(f"X_exp: {X_exp_reshaped.shape}")
    
    # Create domain labels
    domain_sim = np.zeros(len(X_sim))  # 0 = simulation
    domain_exp = np.ones(len(X_exp))   # 1 = experimental
    
    print(f"\nDomain labels:")
    print(f"Simulation domain labels: {len(domain_sim)} samples (label=0)")
    print(f"Experimental domain labels: {len(domain_exp)} samples (label=1)")
    
    return {
        'X_sim': X_sim_reshaped,
        'y_sim': y_sim_normalized,
        'domain_sim': domain_sim,
        'X_exp': X_exp_reshaped,
        'domain_exp': domain_exp,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler
    }

def create_data_loaders(data_dict, batch_size=64, val_split=0.2):
    """
    Create PyTorch data loaders for training and validation.
    
    Args:
        data_dict: Dictionary containing processed data
        batch_size: Batch size for training
        val_split: Fraction of simulation data for validation
    
    Returns:
        dict: Data loaders for different splits
    """
    print(f"\nCreating data loaders (batch_size={batch_size}, val_split={val_split})...")
    
    # Split simulation data into train/validation
    X_sim_train, X_sim_val, y_sim_train, y_sim_val, domain_sim_train, domain_sim_val = train_test_split(
        data_dict['X_sim'], data_dict['y_sim'], data_dict['domain_sim'],
        test_size=val_split, random_state=42, stratify=None
    )
    
    print(f"Data splits:")
    print(f"  Training: {len(X_sim_train)} simulation samples")
    print(f"  Validation: {len(X_sim_val)} simulation samples")
    print(f"  Experimental: {len(data_dict['X_exp'])} samples")
    
    # Create datasets
    train_sim_dataset = HolographicDataset(X_sim_train, y_sim_train, domain_sim_train)
    val_sim_dataset = HolographicDataset(X_sim_val, y_sim_val, domain_sim_val)
    exp_dataset = HolographicDataset(data_dict['X_exp'], domain_labels=data_dict['domain_exp'])
    
    # Create data loaders
    train_sim_loader = DataLoader(train_sim_dataset, batch_size=batch_size, shuffle=True)
    val_sim_loader = DataLoader(val_sim_dataset, batch_size=batch_size, shuffle=False)
    exp_loader = DataLoader(exp_dataset, batch_size=batch_size, shuffle=True)
    
    return {
        'train_sim': train_sim_loader,
        'val_sim': val_sim_loader,
        'exp': exp_loader,
        'val_data': (X_sim_val, y_sim_val)  # For evaluation
    }

def compute_lambda_schedule(epoch, max_epochs, lambda_max=1.0):
    """
    Compute lambda parameter for gradient reversal using a schedule.
    
    Args:
        epoch: Current epoch
        max_epochs: Total number of epochs
        lambda_max: Maximum lambda value
    
    Returns:
        float: Lambda parameter
    """
    # Gradually increase lambda from 0 to lambda_max
    progress = epoch / max_epochs
    lambda_param = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
    return lambda_param * lambda_max

def train_epoch(model, train_sim_loader, exp_loader, optimizer, criterion_regression, criterion_domain, lambda_param):
    """
    Train the model for one epoch.
    
    Args:
        model: The domain adaptive model
        train_sim_loader: DataLoader for simulation training data
        exp_loader: DataLoader for experimental data
        optimizer: Optimizer
        criterion_regression: Regression loss function
        criterion_domain: Domain classification loss function
        lambda_param: Lambda parameter for domain loss weighting
    
    Returns:
        dict: Training losses
    """
    model.train()
    
    total_regression_loss = 0.0
    total_domain_loss = 0.0
    total_loss = 0.0
    num_batches = 0
    
    # Create iterators
    sim_iter = iter(train_sim_loader)
    exp_iter = iter(exp_loader)
    
    # Train on simulation data
    for sim_batch in train_sim_loader:
        X_sim = sim_batch['X'].to(device)
        y_sim = sim_batch['y'].to(device)
        domain_sim = sim_batch['domain'].to(device).unsqueeze(1)
        
        # Get experimental batch (cycle if needed)
        try:
            exp_batch = next(exp_iter)
        except StopIteration:
            exp_iter = iter(exp_loader)
            exp_batch = next(exp_iter)
        
        X_exp = exp_batch['X'].to(device)
        domain_exp = exp_batch['domain'].to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Forward pass on simulation data
        sim_outputs = model(X_sim, lambda_param)
        regression_loss = criterion_regression(sim_outputs['regression'], y_sim)
        domain_loss_sim = criterion_domain(sim_outputs['domain'], domain_sim)
        
        # Forward pass on experimental data (domain classification only)
        exp_outputs = model(X_exp, lambda_param)
        domain_loss_exp = criterion_domain(exp_outputs['domain'], domain_exp)
        
        # Combined domain loss
        domain_loss = (domain_loss_sim + domain_loss_exp) / 2
        
        # Total loss
        total_batch_loss = regression_loss + lambda_param * domain_loss
        
        # Backward pass
        total_batch_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_regression_loss += regression_loss.item()
        total_domain_loss += domain_loss.item()
        total_loss += total_batch_loss.item()
        num_batches += 1
    
    return {
        'regression_loss': total_regression_loss / num_batches,
        'domain_loss': total_domain_loss / num_batches,
        'total_loss': total_loss / num_batches
    }

def validate_model(model, val_loader, criterion_regression, criterion_domain, lambda_param):
    """
    Validate the model.

    Args:
        model: The domain adaptive model
        val_loader: DataLoader for validation data
        criterion_regression: Regression loss function
        criterion_domain: Domain classification loss function
        lambda_param: Lambda parameter for domain loss weighting

    Returns:
        dict: Validation losses
    """
    model.eval()

    total_regression_loss = 0.0
    total_domain_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            domain = batch['domain'].to(device).unsqueeze(1)

            # Forward pass
            outputs = model(X, lambda_param)

            # Compute losses
            regression_loss = criterion_regression(outputs['regression'], y)
            domain_loss = criterion_domain(outputs['domain'], domain)
            total_batch_loss = regression_loss + lambda_param * domain_loss

            # Accumulate losses
            total_regression_loss += regression_loss.item()
            total_domain_loss += domain_loss.item()
            total_loss += total_batch_loss.item()
            num_batches += 1

    return {
        'regression_loss': total_regression_loss / num_batches,
        'domain_loss': total_domain_loss / num_batches,
        'total_loss': total_loss / num_batches
    }

def train_domain_adaptive_model(model, data_loaders, num_epochs=100, learning_rate=1e-3, lambda_max=1.0):
    """
    Train the domain adaptive model.

    Args:
        model: The domain adaptive model
        data_loaders: Dictionary of data loaders
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        lambda_max: Maximum lambda value for domain adaptation

    Returns:
        dict: Training history and best model state
    """
    print(f"\n" + "=" * 70)
    print("TRAINING DOMAIN ADAPTIVE MODEL")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Lambda max: {lambda_max}")
    print(f"  Device: {device}")

    # Move model to device
    model = model.to(device)

    # Define optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_regression = nn.MSELoss()
    criterion_domain = nn.BCELoss()

    # Training history
    history = {
        'train_regression_loss': [],
        'train_domain_loss': [],
        'train_total_loss': [],
        'val_regression_loss': [],
        'val_domain_loss': [],
        'val_total_loss': [],
        'lambda_values': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None

    print(f"\nStarting training...")
    print(f"{'Epoch':<6} {'Lambda':<8} {'Train Loss':<12} {'Val Loss':<12} {'Reg Loss':<12} {'Dom Loss':<12}")
    print("-" * 70)

    for epoch in range(num_epochs):
        # Compute lambda parameter
        lambda_param = compute_lambda_schedule(epoch, num_epochs, lambda_max)

        # Train for one epoch
        train_losses = train_epoch(
            model, data_loaders['train_sim'], data_loaders['exp'],
            optimizer, criterion_regression, criterion_domain, lambda_param
        )

        # Validate
        val_losses = validate_model(
            model, data_loaders['val_sim'],
            criterion_regression, criterion_domain, lambda_param
        )

        # Update history
        history['train_regression_loss'].append(train_losses['regression_loss'])
        history['train_domain_loss'].append(train_losses['domain_loss'])
        history['train_total_loss'].append(train_losses['total_loss'])
        history['val_regression_loss'].append(val_losses['regression_loss'])
        history['val_domain_loss'].append(val_losses['domain_loss'])
        history['val_total_loss'].append(val_losses['total_loss'])
        history['lambda_values'].append(lambda_param)

        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"{epoch+1:<6} {lambda_param:<8.3f} {train_losses['total_loss']:<12.6f} "
                  f"{val_losses['total_loss']:<12.6f} {val_losses['regression_loss']:<12.6f} "
                  f"{val_losses['domain_loss']:<12.6f}")

        # Early stopping check
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
            break

    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        'model': model,
        'history': history,
        'best_val_loss': best_val_loss
    }

def evaluate_and_predict(model, data_loaders, data_dict):
    """
    Evaluate the model and make predictions on experimental data.

    Args:
        model: Trained domain adaptive model
        data_loaders: Dictionary of data loaders
        data_dict: Dictionary containing scalers and data

    Returns:
        dict: Evaluation results and predictions
    """
    print(f"\n" + "=" * 70)
    print("EVALUATION AND PREDICTION")
    print("=" * 70)

    model.eval()

    # Evaluate on validation data
    print("Evaluating on validation data...")
    X_val, y_val = data_loaders['val_data']
    X_val_tensor = torch.FloatTensor(X_val).to(device)

    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_predictions = val_outputs['regression'].cpu().numpy()

    # Inverse transform predictions and targets
    y_scaler = data_dict['y_scaler']
    val_predictions_original = y_scaler.inverse_transform(val_predictions)
    y_val_original = y_scaler.inverse_transform(y_val)

    # Compute validation metrics
    val_mse = np.mean((val_predictions_original - y_val_original) ** 2, axis=0)
    val_mae = np.mean(np.abs(val_predictions_original - y_val_original), axis=0)

    print(f"Validation Results:")
    print(f"  Gap parameter - MSE: {val_mse[0]:.6f}, MAE: {val_mae[0]:.6f}")
    print(f"  L_ecran parameter - MSE: {val_mse[1]:.6f}, MAE: {val_mae[1]:.6f}")

    # Predict on experimental data
    print("\nPredicting on experimental data...")
    exp_predictions_list = []

    with torch.no_grad():
        for batch in data_loaders['exp']:
            X_exp = batch['X'].to(device)
            exp_outputs = model(X_exp)
            exp_predictions_batch = exp_outputs['regression'].cpu().numpy()
            exp_predictions_list.append(exp_predictions_batch)

    # Combine all experimental predictions
    exp_predictions = np.vstack(exp_predictions_list)
    exp_predictions_original = y_scaler.inverse_transform(exp_predictions)

    print(f"Experimental Predictions:")
    print(f"  Number of samples: {len(exp_predictions_original)}")
    print(f"  Gap range: {exp_predictions_original[:, 0].min():.3f} to {exp_predictions_original[:, 0].max():.3f} µm")
    print(f"  L_ecran range: {exp_predictions_original[:, 1].min():.3f} to {exp_predictions_original[:, 1].max():.3f} µm")

    return {
        'val_predictions': val_predictions_original,
        'val_targets': y_val_original,
        'val_mse': val_mse,
        'val_mae': val_mae,
        'exp_predictions': exp_predictions_original
    }

def save_results(model, history, evaluation_results, data_dict):
    """
    Save model, training history, and prediction results.

    Args:
        model: Trained model
        history: Training history
        evaluation_results: Evaluation and prediction results
        data_dict: Data dictionary with scalers
    """
    print(f"\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save model weights
    model_path = 'domain_adaptive_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'X_scaler': data_dict['X_scaler'],
        'y_scaler': data_dict['y_scaler']
    }, model_path)
    print(f"✓ Model saved to: {model_path}")

    # Save experimental predictions to CSV
    exp_predictions = evaluation_results['exp_predictions']
    predictions_df = pd.DataFrame({
        'sample_id': range(len(exp_predictions)),
        'predicted_gap_um': exp_predictions[:, 0],
        'predicted_L_um': exp_predictions[:, 1]
    })

    csv_path = 'experimental_predictions.csv'
    predictions_df.to_csv(csv_path, index=False)
    print(f"✓ Experimental predictions saved to: {csv_path}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"✓ Training history saved to: {history_path}")

    print(f"✓ All results saved successfully!")

def create_visualizations(history, evaluation_results, data_dict):
    """
    Create and save visualization plots.

    Args:
        history: Training history
        evaluation_results: Evaluation results
        data_dict: Data dictionary
    """
    print(f"\nCreating visualizations...")

    # Create a comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Training and validation loss curves
    ax1 = axes[0, 0]
    epochs = range(1, len(history['train_total_loss']) + 1)
    ax1.plot(epochs, history['train_total_loss'], 'b-', label='Train Total Loss', linewidth=2)
    ax1.plot(epochs, history['val_total_loss'], 'r-', label='Val Total Loss', linewidth=2)
    ax1.plot(epochs, history['train_regression_loss'], 'b--', label='Train Regression Loss', alpha=0.7)
    ax1.plot(epochs, history['val_regression_loss'], 'r--', label='Val Regression Loss', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Domain loss and lambda schedule
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()

    ax2.plot(epochs, history['train_domain_loss'], 'g-', label='Train Domain Loss', linewidth=2)
    ax2.plot(epochs, history['val_domain_loss'], 'orange', label='Val Domain Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Domain Loss', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    ax2_twin.plot(epochs, history['lambda_values'], 'purple', label='Lambda', linewidth=2, linestyle='--')
    ax2_twin.set_ylabel('Lambda Parameter', color='purple')
    ax2_twin.tick_params(axis='y', labelcolor='purple')

    ax2.set_title('Domain Loss and Lambda Schedule')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Validation predictions vs targets (Gap)
    ax3 = axes[0, 2]
    val_targets = evaluation_results['val_targets']
    val_predictions = evaluation_results['val_predictions']

    ax3.scatter(val_targets[:, 0], val_predictions[:, 0], alpha=0.6, s=20)
    min_gap = min(val_targets[:, 0].min(), val_predictions[:, 0].min())
    max_gap = max(val_targets[:, 0].max(), val_predictions[:, 0].max())
    ax3.plot([min_gap, max_gap], [min_gap, max_gap], 'r--', linewidth=2)
    ax3.set_xlabel('True Gap (µm)')
    ax3.set_ylabel('Predicted Gap (µm)')
    ax3.set_title('Gap Parameter Validation')
    ax3.grid(True, alpha=0.3)

    # Add R² score
    from sklearn.metrics import r2_score
    r2_gap = r2_score(val_targets[:, 0], val_predictions[:, 0])
    ax3.text(0.05, 0.95, f'R² = {r2_gap:.3f}', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 4: Validation predictions vs targets (L_ecran)
    ax4 = axes[1, 0]
    ax4.scatter(val_targets[:, 1], val_predictions[:, 1], alpha=0.6, s=20)
    min_L = min(val_targets[:, 1].min(), val_predictions[:, 1].min())
    max_L = max(val_targets[:, 1].max(), val_predictions[:, 1].max())
    ax4.plot([min_L, max_L], [min_L, max_L], 'r--', linewidth=2)
    ax4.set_xlabel('True L_ecran (µm)')
    ax4.set_ylabel('Predicted L_ecran (µm)')
    ax4.set_title('L_ecran Parameter Validation')
    ax4.grid(True, alpha=0.3)

    # Add R² score
    r2_L = r2_score(val_targets[:, 1], val_predictions[:, 1])
    ax4.text(0.05, 0.95, f'R² = {r2_L:.3f}', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Plot 5: Experimental predictions distribution (Gap)
    ax5 = axes[1, 1]
    exp_predictions = evaluation_results['exp_predictions']

    # Load original simulation data for comparison
    sim_data = np.load('simulation_processed_with_labels.npz')
    y_sim_original = sim_data['y_data']

    ax5.hist(y_sim_original[:, 0], bins=50, alpha=0.6, label='Simulation (Training)',
             color='blue', density=True)
    ax5.hist(exp_predictions[:, 0], bins=20, alpha=0.8, label='Experimental (Predicted)',
             color='red', density=True)
    ax5.set_xlabel('Gap Parameter (µm)')
    ax5.set_ylabel('Density')
    ax5.set_title('Gap Parameter Distribution Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Experimental predictions distribution (L_ecran)
    ax6 = axes[1, 2]
    ax6.hist(y_sim_original[:, 1], bins=50, alpha=0.6, label='Simulation (Training)',
             color='blue', density=True)
    ax6.hist(exp_predictions[:, 1], bins=20, alpha=0.8, label='Experimental (Predicted)',
             color='red', density=True)
    ax6.set_xlabel('L_ecran Parameter (µm)')
    ax6.set_ylabel('Density')
    ax6.set_title('L_ecran Parameter Distribution Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the comprehensive plot
    plot_path = 'domain_adaptive_training_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comprehensive visualization saved to: {plot_path}")

    plt.close()

    # Create a separate plot for experimental predictions
    create_experimental_predictions_plot(exp_predictions, y_sim_original)

def create_experimental_predictions_plot(exp_predictions, sim_targets):
    """
    Create a focused plot for experimental predictions.

    Args:
        exp_predictions: Experimental predictions
        sim_targets: Original simulation targets for comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gap parameter comparison
    ax1 = axes[0]
    ax1.scatter(range(len(exp_predictions)), exp_predictions[:, 0],
               alpha=0.8, s=50, color='red', label='Experimental Predictions')
    ax1.axhline(y=sim_targets[:, 0].mean(), color='blue', linestyle='--',
               label=f'Simulation Mean: {sim_targets[:, 0].mean():.3f} µm')
    ax1.fill_between(range(len(exp_predictions)),
                     sim_targets[:, 0].min(), sim_targets[:, 0].max(),
                     alpha=0.2, color='blue', label='Simulation Range')
    ax1.set_xlabel('Experimental Sample')
    ax1.set_ylabel('Gap Parameter (µm)')
    ax1.set_title('Experimental Gap Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # L_ecran parameter comparison
    ax2 = axes[1]
    ax2.scatter(range(len(exp_predictions)), exp_predictions[:, 1],
               alpha=0.8, s=50, color='red', label='Experimental Predictions')
    ax2.axhline(y=sim_targets[:, 1].mean(), color='blue', linestyle='--',
               label=f'Simulation Mean: {sim_targets[:, 1].mean():.3f} µm')
    ax2.fill_between(range(len(exp_predictions)),
                     sim_targets[:, 1].min(), sim_targets[:, 1].max(),
                     alpha=0.2, color='blue', label='Simulation Range')
    ax2.set_xlabel('Experimental Sample')
    ax2.set_ylabel('L_ecran Parameter (µm)')
    ax2.set_title('Experimental L_ecran Predictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the experimental predictions plot
    exp_plot_path = 'experimental_predictions_analysis.png'
    plt.savefig(exp_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Experimental predictions plot saved to: {exp_plot_path}")

    plt.close()

def main():
    """
    Main function to run the complete domain adaptive training pipeline.
    """
    print("=" * 80)
    print("DOMAIN ADAPTIVE NEURAL NETWORK FOR HOLOGRAPHIC RING ANALYSIS")
    print("=" * 80)
    print("Author: Oussama GUELFAA")
    print("Date: 01/08/2025")
    print()

    # Configuration
    config = {
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'lambda_max': 1.0,
        'val_split': 0.2
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    try:
        # Step 1: Load and preprocess data
        data_dict = load_and_preprocess_data()

        # Step 2: Create data loaders
        data_loaders = create_data_loaders(
            data_dict,
            batch_size=config['batch_size'],
            val_split=config['val_split']
        )

        # Step 3: Initialize model
        print(f"\n" + "=" * 70)
        print("INITIALIZING MODEL")
        print("=" * 70)

        model = DomainAdaptiveModel(input_size=750)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model initialized:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model architecture:")
        print(f"    Feature Extractor: Conv1D layers + Dense layers")
        print(f"    Regression Head: 2 outputs (gap, L_ecran)")
        print(f"    Domain Classifier: 1 output (domain probability)")

        # Step 4: Train model
        training_results = train_domain_adaptive_model(
            model,
            data_loaders,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            lambda_max=config['lambda_max']
        )

        # Step 5: Evaluate and predict
        evaluation_results = evaluate_and_predict(
            training_results['model'],
            data_loaders,
            data_dict
        )

        # Step 6: Create visualizations
        create_visualizations(
            training_results['history'],
            evaluation_results,
            data_dict
        )

        # Step 7: Save results
        save_results(
            training_results['model'],
            training_results['history'],
            evaluation_results,
            data_dict
        )

        # Final summary
        print(f"\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        print(f"Final Results:")
        print(f"  Best validation loss: {training_results['best_val_loss']:.6f}")
        print(f"  Gap parameter validation MAE: {evaluation_results['val_mae'][0]:.6f} µm")
        print(f"  L_ecran parameter validation MAE: {evaluation_results['val_mae'][1]:.6f} µm")

        exp_predictions = evaluation_results['exp_predictions']
        print(f"\nExperimental Predictions Summary:")
        print(f"  Number of experimental samples: {len(exp_predictions)}")
        print(f"  Predicted gap range: {exp_predictions[:, 0].min():.3f} to {exp_predictions[:, 0].max():.3f} µm")
        print(f"  Predicted L_ecran range: {exp_predictions[:, 1].min():.3f} to {exp_predictions[:, 1].max():.3f} µm")

        print(f"\nFiles Generated:")
        print(f"  ✓ domain_adaptive_model.pt - Trained model weights")
        print(f"  ✓ experimental_predictions.csv - Experimental predictions")
        print(f"  ✓ training_history.csv - Training history")
        print(f"  ✓ domain_adaptive_training_results.png - Comprehensive plots")
        print(f"  ✓ experimental_predictions_analysis.png - Experimental analysis")

        print(f"\n" + "=" * 80)
        print("Domain adaptation training completed successfully!")
        print("The model is ready to predict gap and L_ecran parameters")
        print("from experimental holographic ring data.")
        print("=" * 80)

    except Exception as e:
        print(f"\n" + "=" * 80)
        print("ERROR OCCURRED DURING TRAINING")
        print("=" * 80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 80)

if __name__ == "__main__":
    main()
