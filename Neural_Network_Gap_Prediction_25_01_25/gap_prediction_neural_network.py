"""
Gap Parameter Prediction Neural Network
Author: Oussama GUELFAA
Date: 10 - 06 - 2025

This module implements a 1D Convolutional Neural Network with residual blocks
for predicting gap parameters from holographic intensity profiles.

The network architecture is specifically designed for 1D intensity profile data
and uses residual connections to enable deeper networks while maintaining
gradient flow during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class IntensityProfileDataset(Dataset):
    """
    Custom PyTorch Dataset for intensity profile data.
    
    This dataset class handles loading and preprocessing of intensity profiles
    and their corresponding gap parameters for neural network training.
    
    Args:
        intensity_profiles (np.ndarray): Array of intensity profiles (n_samples, n_features)
        gap_values (np.ndarray): Array of gap parameter values (n_samples,)
        scaler (StandardScaler, optional): Fitted scaler for intensity normalization
        
    Returns:
        tuple: (intensity_profile, gap_value) as PyTorch tensors
    """
    
    def __init__(self, intensity_profiles, gap_values, scaler=None):
        self.intensity_profiles = intensity_profiles
        self.gap_values = gap_values
        self.scaler = scaler
        
        # Apply scaling if scaler is provided
        if self.scaler is not None:
            self.intensity_profiles = self.scaler.transform(self.intensity_profiles)
    
    def __len__(self):
        return len(self.intensity_profiles)
    
    def __getitem__(self, idx):
        # Convert to PyTorch tensors with appropriate data types
        intensity = torch.FloatTensor(self.intensity_profiles[idx])
        gap = torch.FloatTensor([self.gap_values[idx]])
        
        return intensity, gap

class ResidualBlock(nn.Module):
    """
    1D Residual Block for Convolutional Neural Networks.
    
    Implements a residual connection that helps with gradient flow in deep networks.
    The block consists of two 1D convolutional layers with batch normalization
    and ReLU activation, followed by a skip connection.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        stride (int): Stride for the convolution
        padding (int): Padding for the convolution
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer with batch normalization
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolutional layer with batch normalization
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection - adjust dimensions if needed
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection applied
        """
        # Store input for skip connection
        identity = x
        
        # First conv block
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Second conv block (no activation yet)
        out = self.bn2(self.conv2(out))
        
        # Apply skip connection
        identity = self.skip_connection(identity)
        out += identity
        
        # Final activation
        out = F.relu(out)
        
        return out

class GapPredictionCNN(nn.Module):
    """
    1D Convolutional Neural Network for Gap Parameter Prediction.
    
    This network is specifically designed for predicting gap parameters from
    1D intensity profiles. It uses a combination of convolutional layers,
    residual blocks, and fully connected layers.
    
    Architecture:
    - Input: 1D intensity profile (1000 features)
    - Conv1D layers with increasing channels (64, 128, 256, 512)
    - Residual blocks for better gradient flow
    - Global Average Pooling to reduce overfitting
    - Fully connected layers for final prediction
    - Output: Single gap parameter value
    """
    
    def __init__(self, input_size=1000, num_classes=1):
        super(GapPredictionCNN, self).__init__()
        
        # Initial convolutional layer
        # Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        # First residual block
        self.res_block1 = ResidualBlock(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Third convolutional layer
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Second residual block
        self.res_block2 = ResidualBlock(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        
        # Output layer for gap prediction
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input intensity profile tensor (batch_size, 1000)
            
        Returns:
            torch.Tensor: Predicted gap parameter (batch_size, 1)
        """
        # Reshape input for 1D convolution: (batch_size, channels, length)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # First convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))
        
        # First residual block
        x = self.res_block1(x)
        
        # Third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Second residual block
        x = self.res_block2(x)
        
        # Fourth convolutional block
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x

def load_and_preprocess_data(intensity_path, parameters_path):
    """
    Load and preprocess the intensity profile and parameter data.
    
    This function loads the CSV files, extracts gap parameters, and prepares
    the data for neural network training with proper normalization.
    
    Args:
        intensity_path (str): Path to intensity profiles CSV file
        parameters_path (str): Path to parameters CSV file
        
    Returns:
        tuple: (intensity_profiles, gap_values, scaler) where:
            - intensity_profiles: numpy array of intensity data
            - gap_values: numpy array of gap parameter values
            - scaler: fitted StandardScaler for intensity normalization
    """
    print("Loading data...")
    
    # Load intensity profiles (no header)
    intensity_df = pd.read_csv(intensity_path, header=None)
    intensity_profiles = intensity_df.values
    
    # Load parameters
    params_df = pd.read_csv(parameters_path)
    gap_values = params_df['gap'].values
    
    print(f"Loaded {len(intensity_profiles)} intensity profiles")
    print(f"Intensity profile shape: {intensity_profiles.shape}")
    print(f"Gap value range: {gap_values.min():.3f} to {gap_values.max():.3f}")
    print(f"Number of unique gap values: {len(np.unique(gap_values))}")
    
    # Normalize intensity profiles using StandardScaler
    scaler = StandardScaler()
    intensity_profiles_scaled = scaler.fit_transform(intensity_profiles)
    
    print("Data preprocessing completed.")
    
    return intensity_profiles_scaled, gap_values, scaler

def tolerance_based_accuracy(y_true, y_pred, tolerance=0.01):
    """
    Calculate tolerance-based accuracy for gap parameter prediction.

    A prediction is considered correct if the absolute difference between
    predicted and true gap values is within the specified tolerance.

    Args:
        y_true (np.ndarray): True gap values
        y_pred (np.ndarray): Predicted gap values
        tolerance (float): Tolerance threshold (default: 0.01)

    Returns:
        float: Accuracy as percentage of predictions within tolerance
    """
    differences = np.abs(y_true - y_pred)
    correct_predictions = np.sum(differences <= tolerance)
    accuracy = (correct_predictions / len(y_true)) * 100
    return accuracy

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device='cpu'):
    """
    Train the gap prediction neural network.

    This function implements the complete training loop with validation monitoring,
    learning rate scheduling, and early stopping to prevent overfitting.

    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Maximum number of training epochs
        learning_rate (float): Initial learning rate for optimizer
        device (str): Device to run training on ('cpu' or 'cuda')

    Returns:
        dict: Training history containing losses and metrics
    """
    print(f"Training on device: {device}")
    model = model.to(device)

    # Define loss function (Mean Squared Error for regression)
    criterion = nn.MSELoss()

    # Define optimizer (Adam with weight decay for regularization)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Learning rate scheduler (reduces LR when validation loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training history storage
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': [],
        'train_tolerance_acc': [],
        'val_tolerance_acc': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20

    print("Starting training...")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_predictions = []
        train_targets = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Store predictions and targets for metrics calculation
            train_losses.append(loss.item())
            train_predictions.extend(output.detach().cpu().numpy().flatten())
            train_targets.extend(target.detach().cpu().numpy().flatten())

        # Calculate training metrics
        train_loss = np.mean(train_losses)
        train_r2 = r2_score(train_targets, train_predictions)
        train_tolerance_acc = tolerance_based_accuracy(
            np.array(train_targets), np.array(train_predictions)
        )

        # Validation phase
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                # Forward pass
                output = model(data)

                # Calculate loss
                loss = criterion(output, target)

                # Store predictions and targets
                val_losses.append(loss.item())
                val_predictions.extend(output.cpu().numpy().flatten())
                val_targets.extend(target.cpu().numpy().flatten())

        # Calculate validation metrics
        val_loss = np.mean(val_losses)
        val_r2 = r2_score(val_targets, val_predictions)
        val_tolerance_acc = tolerance_based_accuracy(
            np.array(val_targets), np.array(val_predictions)
        )

        # Update learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Store metrics in history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['train_tolerance_acc'].append(train_tolerance_acc)
        history['val_tolerance_acc'].append(val_tolerance_acc)
        history['learning_rates'].append(current_lr)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
            print(f"  Train Tolerance Acc: {train_tolerance_acc:.2f}%, Val Tolerance Acc: {val_tolerance_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            print("-" * 60)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print("Training completed!")
    return history

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate the trained model on test data.

    This function performs comprehensive evaluation including tolerance-based
    accuracy, R² score, MAE, and RMSE metrics.

    Args:
        model (nn.Module): Trained neural network model
        test_loader (DataLoader): Test data loader
        device (str): Device to run evaluation on

    Returns:
        dict: Dictionary containing all evaluation metrics and predictions
    """
    model.eval()
    model = model.to(device)

    predictions = []
    targets = []

    print("Evaluating model...")

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Store predictions and targets
            predictions.extend(output.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten())

    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate comprehensive metrics
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    tolerance_acc = tolerance_based_accuracy(targets, predictions, tolerance=0.01)

    # Calculate additional tolerance levels
    tolerance_acc_005 = tolerance_based_accuracy(targets, predictions, tolerance=0.005)
    tolerance_acc_02 = tolerance_based_accuracy(targets, predictions, tolerance=0.02)

    # Create results dictionary
    results = {
        'predictions': predictions,
        'targets': targets,
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse,
        'tolerance_accuracy_0.01': tolerance_acc,
        'tolerance_accuracy_0.005': tolerance_acc_005,
        'tolerance_accuracy_0.02': tolerance_acc_02,
        'mean_absolute_error_percentage': (mae / np.mean(targets)) * 100
    }

    # Print results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error Percentage: {results['mean_absolute_error_percentage']:.2f}%")
    print(f"Tolerance Accuracy (±0.005): {tolerance_acc_005:.2f}%")
    print(f"Tolerance Accuracy (±0.01): {tolerance_acc:.2f}%")
    print(f"Tolerance Accuracy (±0.02): {tolerance_acc_02:.2f}%")
    print("="*60)

    return results

def plot_training_history(history, save_path='Neural_Network_Gap_Prediction_25_01_25/plots'):
    """
    Plot training history including loss curves and metrics.

    Args:
        history (dict): Training history from train_model function
        save_path (str): Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)

    # Set up the plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('seaborn')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training History - Gap Prediction Neural Network', fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: R² Score
    axes[0, 1].plot(epochs, history['train_r2'], 'b-', label='Training R²', linewidth=2)
    axes[0, 1].plot(epochs, history['val_r2'], 'r-', label='Validation R²', linewidth=2)
    axes[0, 1].axhline(y=0.8, color='g', linestyle='--', label='Target R² = 0.8')
    axes[0, 1].set_title('R² Score Progress', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Tolerance-based Accuracy
    axes[1, 0].plot(epochs, history['train_tolerance_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1, 0].plot(epochs, history['val_tolerance_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1, 0].set_title('Tolerance-based Accuracy (±0.01)', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Learning Rate
    axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Training history plots saved to {save_path}/training_history.png")

def plot_evaluation_results(results, save_path='Neural_Network_Gap_Prediction_25_01_25/plots'):
    """
    Plot evaluation results including prediction vs actual and error analysis.

    Args:
        results (dict): Results from evaluate_model function
        save_path (str): Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)

    predictions = results['predictions']
    targets = results['targets']

    # Set up the plotting style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('seaborn')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Evaluation Results - Gap Parameter Prediction', fontsize=16, fontweight='bold')

    # Plot 1: Predictions vs Actual
    axes[0, 0].scatter(targets, predictions, alpha=0.6, s=30)
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Actual Gap Values')
    axes[0, 0].set_ylabel('Predicted Gap Values')
    axes[0, 0].set_title(f'Predictions vs Actual (R² = {results["r2_score"]:.4f})', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Add tolerance bands
    x_line = np.linspace(targets.min(), targets.max(), 100)
    axes[0, 0].fill_between(x_line, x_line - 0.01, x_line + 0.01, alpha=0.2, color='green',
                           label='±0.01 tolerance')
    axes[0, 0].legend()

    # Plot 2: Prediction errors
    errors = predictions - targets
    axes[0, 1].hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].axvline(x=0.01, color='green', linestyle='--', alpha=0.7, label='+0.01')
    axes[0, 1].axvline(x=-0.01, color='green', linestyle='--', alpha=0.7, label='-0.01')
    axes[0, 1].set_xlabel('Prediction Error (Predicted - Actual)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Error Distribution (MAE = {results["mae"]:.6f})', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Absolute errors vs actual values
    abs_errors = np.abs(errors)
    axes[1, 0].scatter(targets, abs_errors, alpha=0.6, s=30)
    axes[1, 0].axhline(y=0.01, color='red', linestyle='--', linewidth=2, label='0.01 tolerance')
    axes[1, 0].set_xlabel('Actual Gap Values')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Absolute Error vs Actual Values', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Tolerance accuracy summary
    tolerances = [0.005, 0.01, 0.02]
    accuracies = [
        results['tolerance_accuracy_0.005'],
        results['tolerance_accuracy_0.01'],
        results['tolerance_accuracy_0.02']
    ]

    bars = axes[1, 1].bar(tolerances, accuracies, color=['lightcoral', 'skyblue', 'lightgreen'],
                         alpha=0.8, edgecolor='black')
    axes[1, 1].set_xlabel('Tolerance Threshold')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Tolerance-based Accuracy', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_path}/evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Evaluation plots saved to {save_path}/evaluation_results.png")

def save_results_and_model(model, history, results, scaler, save_dir='Neural_Network_Gap_Prediction_25_01_25'):
    """
    Save trained model, training history, and evaluation results.

    Args:
        model (nn.Module): Trained neural network model
        history (dict): Training history
        results (dict): Evaluation results
        scaler (StandardScaler): Fitted data scaler
        save_dir (str): Directory to save all outputs
    """
    # Create directories
    os.makedirs(f'{save_dir}/models', exist_ok=True)
    os.makedirs(f'{save_dir}/results', exist_ok=True)

    # Save model state dict
    torch.save(model.state_dict(), f'{save_dir}/models/final_model.pth')

    # Save scaler
    import joblib
    joblib.dump(scaler, f'{save_dir}/models/scaler.pkl')

    # Save training history
    with open(f'{save_dir}/results/training_history.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in history.items():
            if isinstance(value, list):
                history_serializable[key] = value
            else:
                history_serializable[key] = value.tolist() if hasattr(value, 'tolist') else value
        json.dump(history_serializable, f, indent=4)

    # Save evaluation results
    with open(f'{save_dir}/results/evaluation_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                results_serializable[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                results_serializable[key] = int(value)
            else:
                results_serializable[key] = value
        json.dump(results_serializable, f, indent=4)

    # Save model architecture summary
    with open(f'{save_dir}/results/model_summary.txt', 'w') as f:
        f.write("Gap Prediction Neural Network Architecture\n")
        f.write("=" * 50 + "\n\n")
        f.write(str(model))
        f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
        f.write(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    print(f"Model and results saved to {save_dir}/")

def main():
    """
    Main function to execute the complete neural network training and evaluation pipeline.

    This function orchestrates the entire process from data loading to model evaluation,
    including data preprocessing, model training, evaluation, and results visualization.
    """
    print("="*80)
    print("GAP PARAMETER PREDICTION NEURAL NETWORK")
    print("Author: Oussama GUELFAA")
    print("Date: 25 - 01 - 2025")
    print("="*80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data paths (relative to project root)
    intensity_path = '../data/processed/intensity_profiles_full.csv'
    parameters_path = '../data/processed/parameters.csv'

    # Load and preprocess data
    intensity_profiles, gap_values, scaler = load_and_preprocess_data(intensity_path, parameters_path)

    # Create dataset
    dataset = IntensityProfileDataset(intensity_profiles, gap_values, scaler=None)  # Already scaled

    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = GapPredictionCNN(input_size=1000, num_classes=1)

    print(f"\nModel Architecture:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Train model
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=150,
        learning_rate=0.001,
        device=device
    )

    # Load best model for evaluation
    model.load_state_dict(torch.load('models/best_model.pth'))

    # Evaluate model
    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60)

    results = evaluate_model(model, val_loader, device)

    # Plot results
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    plot_training_history(history)
    plot_evaluation_results(results)

    # Save everything
    save_results_and_model(model, history, results, scaler)

    # Final summary
    print("\n" + "="*80)
    print("TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Final R² Score: {results['r2_score']:.4f}")
    print(f"Tolerance Accuracy (±0.01): {results['tolerance_accuracy_0.01']:.2f}%")
    print(f"Target R² > 0.8: {'✓ ACHIEVED' if results['r2_score'] > 0.8 else '✗ NOT ACHIEVED'}")
    print("="*80)

if __name__ == "__main__":
    main()
