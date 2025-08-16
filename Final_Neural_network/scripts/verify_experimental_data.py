#!/usr/bin/env python3
"""
Verification Script for Processed Experimental Data
Author: Oussama GUELFAA
Date: 01/08/2025

This script verifies the processed experimental data and provides detailed analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def verify_experimental_processing():
    """
    Verify the experimental data processing results.
    """
    print("=" * 70)
    print("Verification of Experimental Data Processing")
    print("=" * 70)
    
    # Load experimental data
    exp_file = "experimental_processed_interp_to_sim_grid.npz"
    if not os.path.exists(exp_file):
        print(f"Error: File not found: {exp_file}")
        return
    
    exp_data = np.load(exp_file)
    exp_X_data = exp_data['X_data']
    exp_x_positions = exp_data['x_positions']
    
    # Load simulation data for comparison
    sim_file = "simulation_processed_truncate250_start.npz"
    if not os.path.exists(sim_file):
        print(f"Error: File not found: {sim_file}")
        return
    
    sim_data = np.load(sim_file)
    sim_X_data = sim_data['X_data']
    sim_x_positions = sim_data['x_positions'][0]  # All profiles use same x-grid
    
    print("Data loaded successfully!")
    print(f"\nExperimental data:")
    print(f"  X_data shape: {exp_X_data.shape}")
    print(f"  x_positions shape: {exp_x_positions.shape}")
    print(f"  File size: {os.path.getsize(exp_file) / (1024*1024):.2f} MB")
    
    print(f"\nSimulation data (for comparison):")
    print(f"  X_data shape: {sim_X_data.shape}")
    print(f"  x_positions shape: {sim_x_positions.shape}")
    print(f"  File size: {os.path.getsize(sim_file) / (1024*1024):.2f} MB")
    
    # Check x-axis alignment
    print(f"\nX-axis comparison:")
    print(f"  Experimental x-range: {exp_x_positions.min():.6f} to {exp_x_positions.max():.6f} µm")
    print(f"  Simulation x-range: {sim_x_positions.min():.6f} to {sim_x_positions.max():.6f} µm")
    
    if np.allclose(exp_x_positions, sim_x_positions):
        print("  ✓ X-axes are perfectly aligned!")
    else:
        print("  ⚠ X-axes are not perfectly aligned")
        print(f"    Max difference: {np.abs(exp_x_positions - sim_x_positions).max():.8f} µm")
    
    # Intensity statistics
    print(f"\nIntensity statistics:")
    print(f"  Experimental:")
    print(f"    Range: {exp_X_data.min():.6f} to {exp_X_data.max():.6f}")
    print(f"    Mean: {exp_X_data.mean():.6f}")
    print(f"    Std: {exp_X_data.std():.6f}")
    
    print(f"  Simulation:")
    print(f"    Range: {sim_X_data.min():.6f} to {sim_X_data.max():.6f}")
    print(f"    Mean: {sim_X_data.mean():.6f}")
    print(f"    Std: {sim_X_data.std():.6f}")
    
    # Create detailed comparison plots
    create_detailed_comparison_plots(exp_X_data, exp_x_positions, sim_X_data, sim_x_positions)
    
    print("=" * 70)
    print("Verification completed successfully!")
    print("=" * 70)

def create_detailed_comparison_plots(exp_X_data, exp_x_positions, sim_X_data, sim_x_positions):
    """
    Create detailed comparison plots.
    """
    print("\nCreating detailed comparison plots...")
    
    # Create a comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Sample profiles comparison
    ax1 = axes[0, 0]
    
    # Plot a few simulation profiles
    for i in range(min(3, sim_X_data.shape[0])):
        ax1.plot(sim_x_positions, sim_X_data[i], 'b-', alpha=0.6, linewidth=1, 
                label='Simulation' if i == 0 else "")
    
    # Plot experimental profiles
    for i in range(min(3, exp_X_data.shape[0])):
        ax1.plot(exp_x_positions, exp_X_data[i], 'r-', alpha=0.8, linewidth=2,
                label='Experimental' if i == 0 else "")
    
    ax1.set_xlabel('Radial Position (µm)')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Sample Profiles Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Intensity distribution histograms
    ax2 = axes[0, 1]
    
    ax2.hist(sim_X_data.flatten(), bins=50, alpha=0.6, color='blue', 
             label=f'Simulation (n={sim_X_data.size})', density=True)
    ax2.hist(exp_X_data.flatten(), bins=50, alpha=0.8, color='red', 
             label=f'Experimental (n={exp_X_data.size})', density=True)
    
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Density')
    ax2.set_title('Intensity Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean profiles
    ax3 = axes[1, 0]
    
    sim_mean = sim_X_data.mean(axis=0)
    sim_std = sim_X_data.std(axis=0)
    exp_mean = exp_X_data.mean(axis=0)
    exp_std = exp_X_data.std(axis=0)
    
    ax3.plot(sim_x_positions, sim_mean, 'b-', linewidth=2, label='Simulation Mean')
    ax3.fill_between(sim_x_positions, sim_mean - sim_std, sim_mean + sim_std, 
                     alpha=0.3, color='blue', label='Simulation ±1σ')
    
    ax3.plot(exp_x_positions, exp_mean, 'r-', linewidth=2, label='Experimental Mean')
    ax3.fill_between(exp_x_positions, exp_mean - exp_std, exp_mean + exp_std, 
                     alpha=0.3, color='red', label='Experimental ±1σ')
    
    ax3.set_xlabel('Radial Position (µm)')
    ax3.set_ylabel('Intensity')
    ax3.set_title('Mean Profiles with Standard Deviation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Profile statistics
    ax4 = axes[1, 1]
    
    # Calculate profile-wise statistics
    sim_profile_means = sim_X_data.mean(axis=1)
    sim_profile_stds = sim_X_data.std(axis=1)
    exp_profile_means = exp_X_data.mean(axis=1)
    exp_profile_stds = exp_X_data.std(axis=1)
    
    ax4.scatter(sim_profile_means, sim_profile_stds, alpha=0.6, color='blue', 
               s=20, label=f'Simulation ({len(sim_profile_means)} profiles)')
    ax4.scatter(exp_profile_means, exp_profile_stds, alpha=0.8, color='red', 
               s=40, label=f'Experimental ({len(exp_profile_means)} profiles)')
    
    ax4.set_xlabel('Profile Mean Intensity')
    ax4.set_ylabel('Profile Standard Deviation')
    ax4.set_title('Profile Statistics Scatter')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the detailed comparison
    detailed_plot_file = "detailed_sim_vs_exp_comparison.png"
    plt.savefig(detailed_plot_file, dpi=150, bbox_inches='tight')
    print(f"Detailed comparison plot saved as: {detailed_plot_file}")
    
    plt.close()
    
    # Create a focused ring structure comparison
    create_ring_structure_comparison(exp_X_data, exp_x_positions, sim_X_data, sim_x_positions)

def create_ring_structure_comparison(exp_X_data, exp_x_positions, sim_X_data, sim_x_positions):
    """
    Create a focused comparison of ring structures.
    """
    plt.figure(figsize=(14, 8))
    
    # Select one representative profile from each dataset
    exp_profile = exp_X_data[0]  # First experimental profile
    sim_profile = sim_X_data[0]  # First simulation profile
    
    plt.subplot(1, 2, 1)
    plt.plot(exp_x_positions, exp_profile, 'r-', linewidth=2, label='Experimental')
    plt.plot(sim_x_positions, sim_profile, 'b-', linewidth=2, alpha=0.7, label='Simulation')
    plt.xlabel('Radial Position (µm)')
    plt.ylabel('Intensity')
    plt.title('Single Profile Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom in on a specific region to see ring details
    plt.subplot(1, 2, 2)
    zoom_start, zoom_end = 2.0, 4.0  # µm
    zoom_mask_exp = (exp_x_positions >= zoom_start) & (exp_x_positions <= zoom_end)
    zoom_mask_sim = (sim_x_positions >= zoom_start) & (sim_x_positions <= zoom_end)
    
    plt.plot(exp_x_positions[zoom_mask_exp], exp_profile[zoom_mask_exp], 
             'r-', linewidth=2, label='Experimental', marker='o', markersize=3)
    plt.plot(sim_x_positions[zoom_mask_sim], sim_profile[zoom_mask_sim], 
             'b-', linewidth=2, alpha=0.7, label='Simulation', marker='s', markersize=2)
    
    plt.xlabel('Radial Position (µm)')
    plt.ylabel('Intensity')
    plt.title(f'Zoomed View ({zoom_start}-{zoom_end} µm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the ring structure comparison
    ring_plot_file = "ring_structure_comparison.png"
    plt.savefig(ring_plot_file, dpi=150, bbox_inches='tight')
    print(f"Ring structure comparison saved as: {ring_plot_file}")
    
    plt.close()

if __name__ == "__main__":
    verify_experimental_processing()
