#!/usr/bin/env python3
"""
Experimental Data Processing Script
Author: Oussama GUELFAA
Date: 01/08/2025

This script loads experimental holographic data, interpolates it to match the simulation
x-grid, and creates comparison visualizations between simulation and experimental profiles.

Input: 
- profile_exp_PS_3um_z_positive.mat (experimental data)
- simulation_processed_truncate250_start.npz (simulation data)

Output: 
- experimental_processed_interp_to_sim_grid.npz (interpolated experimental data)
- sim_vs_exp_profiles.png (comparison plot)
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data' / 'processed'
PLOTS_DIR = BASE_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_experimental_data():
    """
    Load experimental data from MATLAB file.
    
    Returns:
        tuple: (I_profiles, r_exp) - intensity profiles and x-axis in meters
    """
    exp_file = "../Neural_Network_Gap_Lecran_Prediction/data/raw/Test/profile_exp_PS_3um_z_positive.mat"
    
    if not os.path.exists(exp_file):
        print(f"Error: Experimental file not found: {exp_file}")
        sys.exit(1)
    
    print(f"Loading experimental data from: {exp_file}")
    
    try:
        mat_data = sio.loadmat(exp_file)
        
        # Print available variables
        print("Available variables in experimental file:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                print(f"  {key}: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else type(mat_data[key])}")
        
        # Extract experimental data
        I_profiles = mat_data['I_profiles']  # Shape: (50, 184)
        r_exp = mat_data['r_exp'].flatten()  # Shape: (184,) - ensure 1D
        
        print(f"Experimental data loaded:")
        print(f"  I_profiles: {I_profiles.shape}")
        print(f"  r_exp: {r_exp.shape}")
        print(f"  r_exp range: {r_exp.min():.6f} to {r_exp.max():.6f} meters")
        
        return I_profiles, r_exp
        
    except Exception as e:
        print(f"Error loading experimental file: {e}")
        sys.exit(1)

def load_simulation_grid():
    """
    Load simulation x-grid from processed .npz file.
    
    Returns:
        numpy.ndarray: x_positions[0] - simulation x-grid in micrometers
    """
    sim_file = str(DATA_DIR / "simulation_processed_truncate250_start.npz")

    if not os.path.exists(sim_file):
        print(f"Error: Simulation file not found: {sim_file}")
        sys.exit(1)
    
    print(f"Loading simulation grid from: {sim_file}")
    
    try:
        sim_data = np.load(sim_file)
        x_positions = sim_data['x_positions']
        
        # Get the first profile's x-grid (all profiles use the same grid)
        sim_x_grid = x_positions[0]  # Shape: (750,)
        
        print(f"Simulation grid loaded:")
        print(f"  x_positions shape: {x_positions.shape}")
        print(f"  sim_x_grid shape: {sim_x_grid.shape}")
        print(f"  sim_x_grid range: {sim_x_grid.min():.6f} to {sim_x_grid.max():.6f} µm")
        
        return sim_x_grid
        
    except Exception as e:
        print(f"Error loading simulation file: {e}")
        sys.exit(1)

def interpolate_experimental_to_simulation_grid(I_profiles, r_exp, sim_x_grid):
    """
    Interpolate experimental profiles to simulation x-grid.
    
    Args:
        I_profiles (numpy.ndarray): Experimental intensity profiles (50, 184)
        r_exp (numpy.ndarray): Experimental x-axis in meters (184,)
        sim_x_grid (numpy.ndarray): Simulation x-grid in micrometers (750,)
    
    Returns:
        tuple: (X_data, x_clipped) - interpolated profiles and clipped x-grid
    """
    print("Processing experimental data...")
    
    # Convert r_exp from meters to micrometers
    r_exp_um = r_exp * 1e6
    print(f"r_exp converted to µm: {r_exp_um.min():.6f} to {r_exp_um.max():.6f} µm")
    
    # Clip simulation x-grid to experimental range
    x_min = r_exp_um.min()
    x_max = r_exp_um.max()
    
    # Find indices where simulation grid is within experimental range
    valid_indices = (sim_x_grid >= x_min) & (sim_x_grid <= x_max)
    x_clipped = sim_x_grid[valid_indices]
    
    print(f"Clipped simulation grid:")
    print(f"  Original sim_x_grid: {sim_x_grid.shape[0]} points")
    print(f"  Clipped x_grid: {x_clipped.shape[0]} points")
    print(f"  Clipped range: {x_clipped.min():.6f} to {x_clipped.max():.6f} µm")
    
    # Interpolate each experimental profile
    num_profiles = I_profiles.shape[0]
    X_data = np.zeros((num_profiles, len(x_clipped)), dtype=np.float32)
    
    for i in range(num_profiles):
        # Interpolate experimental profile onto clipped simulation grid
        interpolated = np.interp(x_clipped, r_exp_um, I_profiles[i])
        X_data[i] = interpolated.astype(np.float32)
        
        if i < 3:  # Print details for first few profiles
            print(f"  Profile {i+1}: interpolated from {len(r_exp_um)} to {len(x_clipped)} points")
    
    print(f"Interpolation completed:")
    print(f"  X_data shape: {X_data.shape}")
    print(f"  X_data range: {X_data.min():.6f} to {X_data.max():.6f}")
    
    return X_data, x_clipped

def save_processed_experimental_data(X_data, x_clipped):
    """
    Save processed experimental data to compressed .npz file.
    
    Args:
        X_data (numpy.ndarray): Interpolated experimental profiles
        x_clipped (numpy.ndarray): Clipped simulation x-grid
    """
    output_file = str(DATA_DIR / "experimental_processed_interp_to_sim_grid.npz")

    try:
        # Convert to float32 to reduce file size
        X_data_f32 = X_data.astype(np.float32)
        x_clipped_f32 = x_clipped.astype(np.float32)
        
        # Save to compressed .npz file
        np.savez_compressed(output_file,
                          X_data=X_data_f32,
                          x_positions=x_clipped_f32)
        
        print(f"Experimental data saved to: {output_file}")
        
        # Verify saved file
        loaded_data = np.load(output_file)
        print(f"Verification - saved arrays:")
        print(f"  X_data: {loaded_data['X_data'].shape}")
        print(f"  x_positions: {loaded_data['x_positions'].shape}")
        
        # Calculate file size
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"Error saving experimental data: {e}")
        sys.exit(1)

def create_comparison_plot():
    """
    Create comparison plot between simulation and experimental profiles.
    """
    print("Creating comparison visualization...")
    
    # Load simulation data
    sim_data = np.load(str(DATA_DIR / "simulation_processed_truncate250_start.npz"))
    sim_X_data = sim_data['X_data']
    sim_x_positions = sim_data['x_positions'][0]  # All profiles use same x-grid
    
    # Load experimental data
    exp_data = np.load(str(DATA_DIR / "experimental_processed_interp_to_sim_grid.npz"))
    exp_X_data = exp_data['X_data']
    exp_x_positions = exp_data['x_positions']
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot 3-5 random simulation profiles
    num_sim_profiles = min(5, sim_X_data.shape[0])
    sim_indices = np.random.choice(sim_X_data.shape[0], num_sim_profiles, replace=False)
    
    for i, idx in enumerate(sim_indices):
        plt.plot(sim_x_positions, sim_X_data[idx], 
                color='blue', alpha=0.7, linewidth=1.5,
                label='Simulation' if i == 0 else "")
    
    # Plot 3-5 experimental profiles
    num_exp_profiles = min(5, exp_X_data.shape[0])
    exp_indices = np.random.choice(exp_X_data.shape[0], num_exp_profiles, replace=False)
    
    for i, idx in enumerate(exp_indices):
        plt.plot(exp_x_positions, exp_X_data[idx], 
                color='red', alpha=0.8, linewidth=2,
                label='Experimental' if i == 0 else "")
    
    plt.xlabel('Radial Position (µm)', fontsize=12)
    plt.ylabel('Intensity', fontsize=12)
    plt.title('Comparison of Simulation vs Experimental Rings', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text box with information
    info_text = f"Simulation: {num_sim_profiles} profiles\nExperimental: {num_exp_profiles} profiles"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = str(PLOTS_DIR / "sim_vs_exp_profiles.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved as: {plot_file}")
    
    plt.close()

def main():
    """
    Main function to process experimental data and create comparisons.
    """
    print("=" * 70)
    print("Experimental Data Processing and Comparison")
    print("=" * 70)
    
    # Step 1: Load experimental data
    I_profiles, r_exp = load_experimental_data()
    
    # Step 2: Load simulation grid
    sim_x_grid = load_simulation_grid()
    
    # Step 3: Interpolate experimental data to simulation grid
    X_data, x_clipped = interpolate_experimental_to_simulation_grid(I_profiles, r_exp, sim_x_grid)
    
    # Step 4: Save processed experimental data
    save_processed_experimental_data(X_data, x_clipped)
    
    # Step 5: Create comparison visualization
    create_comparison_plot()
    
    print("=" * 70)
    print("Processing completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
