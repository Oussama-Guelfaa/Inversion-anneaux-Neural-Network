#!/usr/bin/env python3
"""
Gap Prediction CNN - Autonomous Training and Testing Script
Author: Oussama GUELFAA
Date: 10 - 01 - 2025

Script autonome pour entraîner et tester le réseau CNN de prédiction de gap.
Ce script gère automatiquement l'extraction des données, l'entraînement,
l'évaluation et la génération des visualisations.
"""

import sys
import os
import yaml
import argparse
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
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class IntensityProfileDataset(Dataset):
    """Custom PyTorch Dataset for intensity profile data."""
    
    def __init__(self, intensity_profiles, gap_values, scaler=None):
        self.intensity_profiles = intensity_profiles
        self.gap_values = gap_values
        self.scaler = scaler
        
        if self.scaler is not None:
            self.intensity_profiles = self.scaler.transform(self.intensity_profiles)
    
    def __len__(self):
        return len(self.intensity_profiles)
    
    def __getitem__(self, idx):
        intensity = torch.FloatTensor(self.intensity_profiles[idx])
        gap = torch.FloatTensor([self.gap_values[idx]])
        return intensity, gap

class ResidualBlock(nn.Module):
    """1D Residual Block for CNN."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
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

class GapPredictionCNN(nn.Module):
    """1D CNN for Gap Parameter Prediction."""
    
    def __init__(self, input_size=1000, num_classes=1):
        super(GapPredictionCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.res_block1 = ResidualBlock(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.res_block2 = ResidualBlock(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res_block1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res_block2(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def load_config(config_path="config/model_config.yaml"):
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
        
        # Calculate intensity ratios
        intensity_profiles = []
        gap_values = []
        
        for i in range(len(L_ecran_vect)):
            for j in range(len(gap_vect)):
                ratio = np.abs(I_subs[i, j, :]) / np.abs(I_subs_inc[i, j, :])
                intensity_profiles.append(ratio)
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

def train_model(model, train_loader, val_loader, config, device='cpu'):
    """Train the neural network model."""
    print(f"🚀 Training on device: {device}")
    model = model.to(device)
    
    # Training configuration
    num_epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    history = {
        'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config['training']['early_stopping']['patience']
    
    print("📈 Starting training...")
    
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
            optimizer.step()
            
            train_losses.append(loss.item())
            train_predictions.extend(output.detach().cpu().numpy().flatten())
            train_targets.extend(target.detach().cpu().numpy().flatten())
        
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
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_r2 = r2_score(train_targets, train_predictions)
        val_r2 = r2_score(val_targets, val_predictions)
        
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"⏹️ Early stopping after {epoch+1} epochs")
            break
    
    print("✅ Training completed!")
    return history

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Gap Prediction CNN Training')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both',
                        help='Mode: train, test, or both')
    parser.add_argument('--config', default='config/model_config.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    print("🔬 Gap Prediction CNN - Autonomous Training System")
    print("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    if args.mode in ['train', 'both']:
        # Extract data
        intensity_profiles, gap_values = extract_data_from_matlab()
        
        if intensity_profiles is not None:
            # Prepare data
            scaler = StandardScaler()
            intensity_profiles_scaled = scaler.fit_transform(intensity_profiles)
            
            # Create dataset
            dataset = IntensityProfileDataset(intensity_profiles_scaled, gap_values)
            
            # Split data
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            batch_size = config['training']['batch_size']
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            model = GapPredictionCNN(input_size=config['model']['input_size'])
            
            # Train model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            history = train_model(model, train_loader, val_loader, config, device)
            
            print("🎯 Training completed successfully!")
        else:
            print("❌ Failed to extract data. Training aborted.")
    
    if args.mode in ['test', 'both']:
        print("🧪 Testing functionality will be implemented in future versions.")
    
    print("🏁 Process completed!")

if __name__ == "__main__":
    main()
