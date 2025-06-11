"""
Improved Gap Parameter Prediction Neural Network
Author: Oussama GUELFAA
Date: 25 - 01 - 2025

This module implements an improved version of the gap prediction neural network
based on analysis of the initial training results. Key improvements include:
- Data truncation to 600 points (based on successful previous models)
- Simplified architecture to reduce overfitting
- Better hyperparameter configuration
- Enhanced data preprocessing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ImprovedIntensityDataset(Dataset):
    """
    Improved dataset class with data truncation and better preprocessing.
    
    Args:
        intensity_profiles (np.ndarray): Array of intensity profiles
        gap_values (np.ndarray): Array of gap parameter values
        truncate_to (int): Number of points to truncate profiles to (default: 600)
        scaler_type (str): Type of scaler to use ('standard' or 'minmax')
    """
    
    def __init__(self, intensity_profiles, gap_values, truncate_to=600, scaler_type='standard'):
        # Truncate intensity profiles
        if intensity_profiles.shape[1] > truncate_to:
            self.intensity_profiles = intensity_profiles[:, :truncate_to]
        else:
            self.intensity_profiles = intensity_profiles
            
        self.gap_values = gap_values
        
        # Apply scaling
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
            
        self.intensity_profiles = self.scaler.fit_transform(self.intensity_profiles)
        
        print(f"Dataset created with {len(self.intensity_profiles)} samples")
        print(f"Intensity profiles truncated to {self.intensity_profiles.shape[1]} points")
        print(f"Using {scaler_type} scaling")
    
    def __len__(self):
        return len(self.intensity_profiles)
    
    def __getitem__(self, idx):
        intensity = torch.FloatTensor(self.intensity_profiles[idx])
        gap = torch.FloatTensor([self.gap_values[idx]])
        return intensity, gap

class SimplifiedGapCNN(nn.Module):
    """
    Simplified 1D CNN for gap parameter prediction.
    
    This architecture is simpler than the original to reduce overfitting
    and improve learning on the available dataset.
    """
    
    def __init__(self, input_size=600, num_classes=1):
        super(SimplifiedGapCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(1, 32, kernel_size=11, stride=2, padding=5)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers with dropout
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer
        self.fc4 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # Reshape for 1D convolution
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        return x

def tolerance_based_accuracy(y_true, y_pred, tolerance=0.01):
    """Calculate tolerance-based accuracy."""
    differences = np.abs(y_true - y_pred)
    correct_predictions = np.sum(differences <= tolerance)
    accuracy = (correct_predictions / len(y_true)) * 100
    return accuracy

def train_improved_model(model, train_loader, val_loader, num_epochs=200, device='cpu'):
    """
    Train the improved model with better hyperparameters.
    """
    print(f"Training improved model on device: {device}")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    # More aggressive learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=15, verbose=True, min_lr=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': [],
        'train_tolerance_acc': [], 'val_tolerance_acc': [], 'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 30  # Increased patience
    
    print("Starting improved training...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_predictions = []
        train_targets = []
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
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
                output = model(data)
                loss = criterion(output, target)
                
                val_losses.append(loss.item())
                val_predictions.extend(output.cpu().numpy().flatten())
                val_targets.extend(target.cpu().numpy().flatten())
        
        # Calculate validation metrics
        val_loss = np.mean(val_losses)
        val_r2 = r2_score(val_targets, val_predictions)
        val_tolerance_acc = tolerance_based_accuracy(
            np.array(val_targets), np.array(val_predictions)
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['train_tolerance_acc'].append(train_tolerance_acc)
        history['val_tolerance_acc'].append(val_tolerance_acc)
        history['learning_rates'].append(current_lr)
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
            print(f"  Train Tolerance Acc: {train_tolerance_acc:.2f}%, Val Tolerance Acc: {val_tolerance_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.2e}")
            print("-" * 60)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/improved_best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("Improved training completed!")
    return history

def main_improved():
    """
    Main function for improved model training.
    """
    print("="*80)
    print("IMPROVED GAP PARAMETER PREDICTION NEURAL NETWORK")
    print("Author: Oussama GUELFAA")
    print("Date: 25 - 01 - 2025")
    print("="*80)
    
    # Load data
    print("Loading data...")
    intensity_df = pd.read_csv('../data/processed/intensity_profiles_full.csv', header=None)
    params_df = pd.read_csv('../data/processed/parameters.csv')
    
    intensity_profiles = intensity_df.values
    gap_values = params_df['gap'].values
    
    print(f"Original data shape: {intensity_profiles.shape}")
    
    # Create improved dataset with truncation
    dataset = ImprovedIntensityDataset(
        intensity_profiles, gap_values, 
        truncate_to=600, scaler_type='standard'
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch size
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize improved model
    model = SimplifiedGapCNN(input_size=600, num_classes=1)
    
    print(f"\nImproved Model Architecture:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = train_improved_model(model, train_loader, val_loader, num_epochs=200, device=device)
    
    print("\nImproved model training completed!")
    print("Check 'models/improved_best_model.pth' for the trained model.")

if __name__ == "__main__":
    main_improved()
