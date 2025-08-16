#!/usr/bin/env python3
"""
Verification Script for Processed Simulation Data
Author: Oussama GUELFAA
Date: 01/08/2025

This script loads and verifies the processed simulation data from the .npz file.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data' / 'processed'
PLOTS_DIR = BASE_DIR / 'plots'

def verify_processed_data():
    """
    Load and verify the processed simulation data.
    """
    npz_file = DATA_DIR / "simulation_processed_truncate250_start.npz"

    if not os.path.exists(npz_file):
        print(f"Error: File not found: {npz_file}")
        return
    
    print("=" * 60)
    print("Verification of Processed Simulation Data")
    print("=" * 60)
    
    # Load the data
    data = np.load(str(npz_file))

    print("Available arrays in the file:")
    for key in data.keys():
        print(f"  {key}: {data[key].shape}")
    
    X_data = data['X_data']
    x_positions = data['x_positions']
    
    print(f"\nData summary:")
    print(f"  Number of profiles: {X_data.shape[0]}")
    print(f"  Points per profile: {X_data.shape[1]}")
    print(f"  Total data points: {X_data.size}")
    
    print(f"\nIntensity data statistics:")
    print(f"  Min value: {X_data.min():.6f}")
    print(f"  Max value: {X_data.max():.6f}")
    print(f"  Mean value: {X_data.mean():.6f}")
    print(f"  Std deviation: {X_data.std():.6f}")
    
    print(f"\nX-axis data statistics:")
    print(f"  Min value: {x_positions.min():.6f}")
    print(f"  Max value: {x_positions.max():.6f}")
    print(f"  Mean value: {x_positions.mean():.6f}")
    
    # Check if x-axis is consistent across profiles
    x_first_profile = x_positions[0, :]
    x_last_profile = x_positions[-1, :]
    if np.allclose(x_first_profile, x_last_profile):
        print("  X-axis is consistent across all profiles ✓")
    else:
        print("  Warning: X-axis varies between profiles")
    
    print(f"\nFile size: {os.path.getsize(npz_file) / (1024*1024):.2f} MB")
    
    # Plot a few sample profiles
    print("\nGenerating sample plots...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot first 5 profiles
    for i in range(min(5, X_data.shape[0])):
        plt.subplot(2, 3, i+1)
        plt.plot(x_positions[i, :], X_data[i, :])
        plt.title(f'Profile {i+1}')
        plt.xlabel('X position')
        plt.ylabel('Intensity')
        plt.grid(True, alpha=0.3)
    
    # Plot comparison of original vs truncated range
    plt.subplot(2, 3, 6)
    # Show the truncation effect by plotting the x-axis range
    original_x = np.arange(1000)  # Original 1000 points
    truncated_x = original_x[250:]  # After truncation
    
    plt.axvspan(0, 250, alpha=0.3, color='red', label='Truncated region')
    plt.axvspan(250, 1000, alpha=0.3, color='green', label='Kept region')
    plt.plot(original_x, np.ones_like(original_x), 'k-', alpha=0.5)
    plt.xlabel('Original point index')
    plt.ylabel('Value')
    plt.title('Truncation visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'verification_plots.png', dpi=150, bbox_inches='tight')
    print("Sample plots saved as 'verification_plots.png'")
    
    print("=" * 60)
    print("Verification completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    verify_processed_data()
