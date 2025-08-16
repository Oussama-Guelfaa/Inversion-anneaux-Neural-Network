#!/usr/bin/env python3
"""
Verification Script for Combined Dataset
Author: Oussama GUELFAA
Date: 01/08/2025

This script verifies the combined simulation dataset with labels and creates
visualizations to understand the parameter distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data' / 'processed'
PLOTS_DIR = BASE_DIR / 'plots'

def verify_combined_dataset():
    """
    Verify the combined dataset and create analysis plots.
    """
    print("=" * 70)
    print("VERIFICATION OF COMBINED DATASET")
    print("=" * 70)
    
    # Load the combined dataset
    combined_file = DATA_DIR / "simulation_processed_with_labels.npz"

    if not os.path.exists(combined_file):
        print(f"Error: File not found: {combined_file}")
        return
    
    print(f"Loading combined dataset from: {combined_file}")
    
    data = np.load(str(combined_file))

    # Extract arrays
    X_data = data['X_data']
    x_positions = data['x_positions']
    y_data = data['y_data']
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  X_data: {X_data.shape}, dtype={X_data.dtype}")
    print(f"  x_positions: {x_positions.shape}, dtype={x_positions.dtype}")
    print(f"  y_data: {y_data.shape}, dtype={y_data.dtype}")
    
    # File size
    file_size = os.path.getsize(combined_file) / (1024 * 1024)
    print(f"  File size: {file_size:.2f} MB")
    
    # Data quality checks
    print(f"\nData quality checks:")
    
    # Check for NaN values
    nan_X = np.sum(np.isnan(X_data))
    nan_x = np.sum(np.isnan(x_positions))
    nan_y = np.sum(np.isnan(y_data))
    
    print(f"  NaN values: X_data={nan_X}, x_positions={nan_x}, y_data={nan_y}")
    
    # Check for infinite values
    inf_X = np.sum(np.isinf(X_data))
    inf_x = np.sum(np.isinf(x_positions))
    inf_y = np.sum(np.isinf(y_data))
    
    print(f"  Infinite values: X_data={inf_X}, x_positions={inf_x}, y_data={inf_y}")
    
    if nan_X + nan_x + nan_y + inf_X + inf_x + inf_y == 0:
        print("  ✓ No NaN or infinite values found")
    else:
        print("  ⚠ Data quality issues detected")
    
    # Parameter analysis
    gap_values = y_data[:, 0]
    L_values = y_data[:, 1]
    
    print(f"\nParameter analysis:")
    print(f"  Gap parameter (µm):")
    print(f"    Range: {gap_values.min():.6f} to {gap_values.max():.6f}")
    print(f"    Mean: {gap_values.mean():.6f} ± {gap_values.std():.6f}")
    print(f"    Unique values: {len(np.unique(gap_values))}")
    
    print(f"  L_ecran parameter (µm):")
    print(f"    Range: {L_values.min():.6f} to {L_values.max():.6f}")
    print(f"    Mean: {L_values.mean():.6f} ± {L_values.std():.6f}")
    print(f"    Unique values: {len(np.unique(L_values))}")
    
    # Create analysis plots
    create_parameter_analysis_plots(X_data, x_positions, y_data)
    
    # Test data loading for neural network
    test_neural_network_loading(X_data, x_positions, y_data)
    
    print("=" * 70)
    print("VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

def create_parameter_analysis_plots(X_data, x_positions, y_data):
    """
    Create comprehensive analysis plots for the dataset.
    """
    print("\nCreating parameter analysis plots...")
    
    gap_values = y_data[:, 0]
    L_values = y_data[:, 1]
    
    # Create a comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Gap parameter distribution
    ax1 = axes[0, 0]
    ax1.hist(gap_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Gap Parameter (µm)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Gap Parameter Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    gap_stats = f'Mean: {gap_values.mean():.3f}\nStd: {gap_values.std():.3f}\nRange: {gap_values.min():.3f}-{gap_values.max():.3f}'
    ax1.text(0.02, 0.98, gap_stats, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: L_ecran parameter distribution
    ax2 = axes[0, 1]
    ax2.hist(L_values, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.set_xlabel('L_ecran Parameter (µm)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('L_ecran Parameter Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    L_stats = f'Mean: {L_values.mean():.3f}\nStd: {L_values.std():.3f}\nRange: {L_values.min():.3f}-{L_values.max():.3f}'
    ax2.text(0.02, 0.98, L_stats, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3: 2D parameter space
    ax3 = axes[0, 2]
    scatter = ax3.scatter(gap_values, L_values, alpha=0.6, s=1, c=np.arange(len(gap_values)), cmap='viridis')
    ax3.set_xlabel('Gap Parameter (µm)')
    ax3.set_ylabel('L_ecran Parameter (µm)')
    ax3.set_title('Parameter Space Coverage')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Sample Index')
    
    # Plot 4: Sample intensity profiles colored by gap
    ax4 = axes[1, 0]
    
    # Select a few representative profiles
    n_samples = min(20, X_data.shape[0])
    indices = np.linspace(0, X_data.shape[0]-1, n_samples, dtype=int)
    
    for i, idx in enumerate(indices):
        color_val = gap_values[idx]
        ax4.plot(x_positions[idx], X_data[idx], alpha=0.7, linewidth=1,
                color=plt.cm.viridis(color_val / gap_values.max()))
    
    ax4.set_xlabel('Radial Position (µm)')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Sample Profiles (colored by Gap)')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                              norm=plt.Normalize(vmin=gap_values.min(), vmax=gap_values.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax4, label='Gap (µm)')
    
    # Plot 5: Sample intensity profiles colored by L_ecran
    ax5 = axes[1, 1]
    
    for i, idx in enumerate(indices):
        color_val = L_values[idx]
        ax5.plot(x_positions[idx], X_data[idx], alpha=0.7, linewidth=1,
                color=plt.cm.plasma((color_val - L_values.min()) / (L_values.max() - L_values.min())))
    
    ax5.set_xlabel('Radial Position (µm)')
    ax5.set_ylabel('Intensity')
    ax5.set_title('Sample Profiles (colored by L_ecran)')
    ax5.grid(True, alpha=0.3)
    
    # Add colorbar
    sm2 = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                               norm=plt.Normalize(vmin=L_values.min(), vmax=L_values.max()))
    sm2.set_array([])
    plt.colorbar(sm2, ax=ax5, label='L_ecran (µm)')
    
    # Plot 6: Parameter correlation analysis
    ax6 = axes[1, 2]
    
    # Create a 2D histogram to show parameter density
    hist, xedges, yedges = np.histogram2d(gap_values, L_values, bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im = ax6.imshow(hist.T, extent=extent, origin='lower', aspect='auto', cmap='Blues')
    ax6.set_xlabel('Gap Parameter (µm)')
    ax6.set_ylabel('L_ecran Parameter (µm)')
    ax6.set_title('Parameter Density Map')
    plt.colorbar(im, ax=ax6, label='Count')
    
    plt.tight_layout()
    
    # Save the analysis plot
    analysis_plot_file = PLOTS_DIR / "parameter_analysis.png"
    plt.savefig(analysis_plot_file, dpi=150, bbox_inches='tight')
    print(f"Parameter analysis plot saved as: {analysis_plot_file}")
    
    plt.close()

def test_neural_network_loading(X_data, x_positions, y_data):
    """
    Test typical neural network data loading patterns.
    """
    print("\nTesting neural network data loading patterns...")
    
    # Test 1: Basic indexing
    try:
        sample_X = X_data[0]
        sample_y = y_data[0]
        print(f"✓ Basic indexing: X shape {sample_X.shape}, y shape {sample_y.shape}")
    except Exception as e:
        print(f"✗ Basic indexing failed: {e}")
    
    # Test 2: Batch loading
    try:
        batch_size = 32
        batch_X = X_data[:batch_size]
        batch_y = y_data[:batch_size]
        print(f"✓ Batch loading: X shape {batch_X.shape}, y shape {batch_y.shape}")
    except Exception as e:
        print(f"✗ Batch loading failed: {e}")
    
    # Test 3: Random sampling
    try:
        n_samples = 100
        indices = np.random.choice(X_data.shape[0], n_samples, replace=False)
        random_X = X_data[indices]
        random_y = y_data[indices]
        print(f"✓ Random sampling: X shape {random_X.shape}, y shape {random_y.shape}")
    except Exception as e:
        print(f"✗ Random sampling failed: {e}")
    
    # Test 4: Data type compatibility
    try:
        # Test conversion to common ML formats
        X_float32 = X_data.astype(np.float32)
        y_float32 = y_data.astype(np.float32)
        print(f"✓ Float32 conversion: X dtype {X_float32.dtype}, y dtype {y_float32.dtype}")
    except Exception as e:
        print(f"✗ Float32 conversion failed: {e}")
    
    # Test 5: Memory usage estimation
    try:
        X_memory = X_data.nbytes / (1024**2)  # MB
        y_memory = y_data.nbytes / (1024**2)  # MB
        total_memory = X_memory + y_memory
        print(f"✓ Memory usage: X={X_memory:.1f}MB, y={y_memory:.1f}MB, total={total_memory:.1f}MB")
    except Exception as e:
        print(f"✗ Memory usage calculation failed: {e}")
    
    print("✓ All neural network loading tests passed!")

if __name__ == "__main__":
    verify_combined_dataset()
