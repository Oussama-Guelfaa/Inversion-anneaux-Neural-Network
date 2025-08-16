#!/usr/bin/env python3
"""
Neural Network Training Demo
Author: Oussama GUELFAA
Date: 01/08/2025

This script demonstrates how to load and use the complete dataset for neural network training.
It shows typical data loading, preprocessing, and train/validation split patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_complete_dataset():
    """
    Load the complete training dataset.
    
    Returns:
        tuple: (X_data, y_data, x_positions) - features, targets, and x-axis
    """
    dataset_file = "simulation_processed_with_labels.npz"
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found: {dataset_file}")
        return None, None, None
    
    print(f"Loading complete dataset from: {dataset_file}")
    
    data = np.load(dataset_file)
    X_data = data['X_data']          # Shape: (22540, 750)
    y_data = data['y_data']          # Shape: (22540, 2)
    x_positions = data['x_positions'] # Shape: (22540, 750)
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  Features (X_data): {X_data.shape}")
    print(f"  Targets (y_data): {y_data.shape}")
    print(f"  X-axis (x_positions): {x_positions.shape}")
    
    return X_data, y_data, x_positions

def analyze_dataset(X_data, y_data):
    """
    Analyze the dataset characteristics.
    
    Args:
        X_data (numpy.ndarray): Feature data
        y_data (numpy.ndarray): Target data
    """
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    # Feature analysis
    print(f"Feature Analysis (Intensity Profiles):")
    print(f"  Shape: {X_data.shape}")
    print(f"  Data type: {X_data.dtype}")
    print(f"  Range: {X_data.min():.6f} to {X_data.max():.6f}")
    print(f"  Mean: {X_data.mean():.6f}")
    print(f"  Std: {X_data.std():.6f}")
    
    # Target analysis
    gap_values = y_data[:, 0]
    L_values = y_data[:, 1]
    
    print(f"\nTarget Analysis:")
    print(f"  Shape: {y_data.shape}")
    print(f"  Data type: {y_data.dtype}")
    
    print(f"  Gap parameter (µm):")
    print(f"    Range: {gap_values.min():.6f} to {gap_values.max():.6f}")
    print(f"    Mean: {gap_values.mean():.6f} ± {gap_values.std():.6f}")
    print(f"    Unique values: {len(np.unique(gap_values))}")
    
    print(f"  L_ecran parameter (µm):")
    print(f"    Range: {L_values.min():.6f} to {L_values.max():.6f}")
    print(f"    Mean: {L_values.mean():.6f} ± {L_values.std():.6f}")
    print(f"    Unique values: {len(np.unique(L_values))}")

def create_train_validation_split(X_data, y_data, test_size=0.2, random_state=42):
    """
    Create train/validation split for neural network training.
    
    Args:
        X_data (numpy.ndarray): Feature data
        y_data (numpy.ndarray): Target data
        test_size (float): Fraction for validation set
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    print(f"\n" + "="*60)
    print("TRAIN/VALIDATION SPLIT")
    print("="*60)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  # Could implement stratification based on parameter ranges
    )
    
    print(f"Split configuration:")
    print(f"  Test size: {test_size} ({test_size*100:.1f}%)")
    print(f"  Random state: {random_state}")
    
    print(f"\nSplit results:")
    print(f"  Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/X_data.shape[0]*100:.1f}%)")
    print(f"  Validation set: {X_val.shape[0]:,} samples ({X_val.shape[0]/X_data.shape[0]*100:.1f}%)")
    
    # Analyze parameter distribution in splits
    print(f"\nParameter distribution analysis:")
    print(f"  Training set:")
    print(f"    Gap range: {y_train[:, 0].min():.3f} to {y_train[:, 0].max():.3f} µm")
    print(f"    L_ecran range: {y_train[:, 1].min():.3f} to {y_train[:, 1].max():.3f} µm")
    
    print(f"  Validation set:")
    print(f"    Gap range: {y_val[:, 0].min():.3f} to {y_val[:, 0].max():.3f} µm")
    print(f"    L_ecran range: {y_val[:, 1].min():.3f} to {y_val[:, 1].max():.3f} µm")
    
    return X_train, X_val, y_train, y_val

def demonstrate_data_preprocessing(X_train, X_val, y_train, y_val):
    """
    Demonstrate typical data preprocessing steps.
    
    Args:
        X_train, X_val: Training and validation features
        y_train, y_val: Training and validation targets
    
    Returns:
        tuple: Preprocessed data and scalers
    """
    print(f"\n" + "="*60)
    print("DATA PREPROCESSING DEMONSTRATION")
    print("="*60)
    
    # Feature scaling (StandardScaler)
    print("1. Feature Scaling (StandardScaler):")
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    
    print(f"  Original features - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")
    print(f"  Scaled features - Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")
    
    # Target scaling (optional, but often helpful)
    print("\n2. Target Scaling (StandardScaler):")
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    
    print(f"  Original targets:")
    print(f"    Gap - Mean: {y_train[:, 0].mean():.6f}, Std: {y_train[:, 0].std():.6f}")
    print(f"    L_ecran - Mean: {y_train[:, 1].mean():.6f}, Std: {y_train[:, 1].std():.6f}")
    
    print(f"  Scaled targets:")
    print(f"    Gap - Mean: {y_train_scaled[:, 0].mean():.6f}, Std: {y_train_scaled[:, 0].std():.6f}")
    print(f"    L_ecran - Mean: {y_train_scaled[:, 1].mean():.6f}, Std: {y_train_scaled[:, 1].std():.6f}")
    
    # Data type optimization
    print("\n3. Data Type Optimization:")
    print(f"  Original dtype: {X_train.dtype}")
    X_train_opt = X_train_scaled.astype(np.float32)
    X_val_opt = X_val_scaled.astype(np.float32)
    y_train_opt = y_train_scaled.astype(np.float32)
    y_val_opt = y_val_scaled.astype(np.float32)
    print(f"  Optimized dtype: {X_train_opt.dtype}")
    
    # Memory usage comparison
    original_memory = (X_train.nbytes + X_val.nbytes + y_train.nbytes + y_val.nbytes) / (1024**2)
    optimized_memory = (X_train_opt.nbytes + X_val_opt.nbytes + y_train_opt.nbytes + y_val_opt.nbytes) / (1024**2)
    
    print(f"  Memory usage: {original_memory:.1f} MB → {optimized_memory:.1f} MB (saved {original_memory-optimized_memory:.1f} MB)")
    
    return (X_train_opt, X_val_opt, y_train_opt, y_val_opt), (feature_scaler, target_scaler)

def create_sample_visualization(X_data, y_data, x_positions):
    """
    Create sample visualizations for the dataset.
    
    Args:
        X_data: Feature data
        y_data: Target data
        x_positions: X-axis data
    """
    print(f"\n" + "="*60)
    print("CREATING SAMPLE VISUALIZATIONS")
    print("="*60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Sample profiles colored by gap
    ax1 = axes[0, 0]
    n_samples = 20
    indices = np.linspace(0, X_data.shape[0]-1, n_samples, dtype=int)
    
    for idx in indices:
        gap_val = y_data[idx, 0]
        color_intensity = gap_val / y_data[:, 0].max()
        ax1.plot(x_positions[idx], X_data[idx], alpha=0.7, 
                color=plt.cm.viridis(color_intensity), linewidth=1)
    
    ax1.set_xlabel('Radial Position (µm)')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Sample Profiles (colored by Gap)')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=y_data[:, 0].min(), vmax=y_data[:, 0].max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label='Gap (µm)')
    
    # Plot 2: Parameter distribution
    ax2 = axes[0, 1]
    ax2.scatter(y_data[:, 0], y_data[:, 1], alpha=0.6, s=1)
    ax2.set_xlabel('Gap Parameter (µm)')
    ax2.set_ylabel('L_ecran Parameter (µm)')
    ax2.set_title('Parameter Space Coverage')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gap parameter histogram
    ax3 = axes[1, 0]
    ax3.hist(y_data[:, 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_xlabel('Gap Parameter (µm)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Gap Parameter Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: L_ecran parameter histogram
    ax4 = axes[1, 1]
    ax4.hist(y_data[:, 1], bins=50, alpha=0.7, color='red', edgecolor='black')
    ax4.set_xlabel('L_ecran Parameter (µm)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('L_ecran Parameter Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = "neural_network_demo_visualization.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Sample visualization saved as: {plot_file}")
    
    plt.close()

def main():
    """
    Main demonstration function.
    """
    print("=" * 70)
    print("NEURAL NETWORK TRAINING DATASET DEMONSTRATION")
    print("=" * 70)
    
    # Step 1: Load complete dataset
    X_data, y_data, x_positions = load_complete_dataset()
    
    if X_data is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Step 2: Analyze dataset
    analyze_dataset(X_data, y_data)
    
    # Step 3: Create train/validation split
    X_train, X_val, y_train, y_val = create_train_validation_split(X_data, y_data)
    
    # Step 4: Demonstrate preprocessing
    processed_data, scalers = demonstrate_data_preprocessing(X_train, X_val, y_train, y_val)
    
    # Step 5: Create visualizations
    create_sample_visualization(X_data, y_data, x_positions)
    
    print(f"\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED!")
    print("=" * 70)
    print("The dataset is ready for neural network training with frameworks like:")
    print("  • PyTorch")
    print("  • TensorFlow/Keras")
    print("  • Scikit-learn")
    print("=" * 70)

if __name__ == "__main__":
    main()
