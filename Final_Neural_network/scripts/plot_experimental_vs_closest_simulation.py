#!/usr/bin/env python3
"""
Plot Experimental Profiles vs Closest Simulation Matches
Author: Oussama GUELFAA
Date: 01/08/2025

This script plots experimental holographic profiles alongside the simulation profiles
that have the closest predicted gap and L_ecran parameters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

def load_all_data():
    """Load experimental data, predictions, and simulation data."""
    print("Loading all datasets...")
    
    # Load experimental predictions
    predictions = pd.read_csv('experimental_predictions_fixed.csv')
    print(f"Loaded {len(predictions)} experimental predictions")
    
    # Load experimental profiles
    exp_data = np.load('experimental_processed_interp_to_sim_grid.npz')
    X_exp = exp_data['X_data']  # (50, 750)
    x_exp = exp_data['x_positions']  # (750,)
    print(f"Loaded experimental profiles: {X_exp.shape}")
    
    # Load simulation data with labels
    sim_data = np.load('simulation_processed_with_labels.npz')
    X_sim = sim_data['X_data']  # (22540, 750)
    y_sim = sim_data['y_data']  # (22540, 2) - [gap_um, L_um]
    x_sim = sim_data['x_positions'][0]  # (750,) - all profiles use same x-axis
    print(f"Loaded simulation data: X_sim {X_sim.shape}, y_sim {y_sim.shape}")
    
    return {
        'predictions': predictions,
        'X_exp': X_exp,
        'x_exp': x_exp,
        'X_sim': X_sim,
        'y_sim': y_sim,
        'x_sim': x_sim
    }

def find_closest_simulation_profiles(predictions, y_sim, n_closest=5):
    """
    Find the closest simulation profiles for each experimental prediction.
    
    Args:
        predictions: DataFrame with experimental predictions
        y_sim: Simulation labels (gap, L_ecran)
        n_closest: Number of closest matches to find
    
    Returns:
        dict: Mapping of experimental indices to closest simulation indices
    """
    print(f"Finding {n_closest} closest simulation profiles for each experimental sample...")
    
    # Extract predicted parameters
    exp_params = predictions[['predicted_gap_um', 'predicted_L_um']].values
    
    # Normalize parameters for distance calculation (to handle different scales)
    gap_range = y_sim[:, 0].max() - y_sim[:, 0].min()
    L_range = y_sim[:, 1].max() - y_sim[:, 1].min()
    
    exp_params_norm = exp_params.copy()
    exp_params_norm[:, 0] = exp_params_norm[:, 0] / gap_range
    exp_params_norm[:, 1] = exp_params_norm[:, 1] / L_range
    
    y_sim_norm = y_sim.copy()
    y_sim_norm[:, 0] = y_sim_norm[:, 0] / gap_range
    y_sim_norm[:, 1] = y_sim_norm[:, 1] / L_range
    
    # Calculate distances
    distances = cdist(exp_params_norm, y_sim_norm, metric='euclidean')
    
    # Find closest matches
    closest_matches = {}
    for exp_idx in range(len(predictions)):
        # Get indices of n_closest simulation profiles
        closest_sim_indices = np.argsort(distances[exp_idx])[:n_closest]
        closest_distances = distances[exp_idx][closest_sim_indices]
        
        closest_matches[exp_idx] = {
            'sim_indices': closest_sim_indices,
            'distances': closest_distances,
            'exp_params': exp_params[exp_idx],
            'sim_params': y_sim[closest_sim_indices]
        }
        
        if exp_idx < 5:  # Print details for first few samples
            print(f"  Exp sample {exp_idx}: predicted gap={exp_params[exp_idx, 0]:.3f}, L={exp_params[exp_idx, 1]:.3f}")
            print(f"    Closest sim: gap={y_sim[closest_sim_indices[0], 0]:.3f}, L={y_sim[closest_sim_indices[0], 1]:.3f}, distance={closest_distances[0]:.4f}")
    
    return closest_matches

def create_comparison_plots(data, closest_matches, n_samples=12):
    """
    Create comparison plots showing experimental vs closest simulation profiles.
    
    Args:
        data: Dictionary containing all loaded data
        closest_matches: Dictionary of closest simulation matches
        n_samples: Number of experimental samples to plot
    """
    print(f"Creating comparison plots for {n_samples} experimental samples...")
    
    # Select samples to plot (evenly spaced)
    exp_indices = np.linspace(0, len(data['predictions']) - 1, n_samples, dtype=int)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, exp_idx in enumerate(exp_indices):
        ax = axes[i]
        
        # Get experimental profile
        exp_profile = data['X_exp'][exp_idx]
        x_axis = data['x_exp']
        
        # Get closest simulation profile
        closest_sim_idx = closest_matches[exp_idx]['sim_indices'][0]
        sim_profile = data['X_sim'][closest_sim_idx]
        
        # Get parameters
        exp_params = closest_matches[exp_idx]['exp_params']
        sim_params = closest_matches[exp_idx]['sim_params'][0]
        distance = closest_matches[exp_idx]['distances'][0]
        
        # Plot profiles
        ax.plot(x_axis, exp_profile, 'r-', linewidth=2, label='Experimental', alpha=0.8)
        ax.plot(data['x_sim'], sim_profile, 'b-', linewidth=2, label='Closest Simulation', alpha=0.7)
        
        # Add title with parameters
        title = f'Sample {exp_idx}\n'
        title += f'Exp: gap={exp_params[0]:.3f}, L={exp_params[1]:.3f}\n'
        title += f'Sim: gap={sim_params[0]:.3f}, L={sim_params[1]:.3f}\n'
        title += f'Distance: {distance:.4f}'
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Radial Position (µm)')
        ax.set_ylabel('Intensity')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set consistent y-axis limits
        all_intensities = np.concatenate([exp_profile, sim_profile])
        y_min, y_max = all_intensities.min(), all_intensities.max()
        y_margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = 'experimental_vs_closest_simulation_profiles.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {plot_path}")
    
    plt.close()

def create_parameter_space_visualization(data, closest_matches):
    """
    Create a visualization showing the parameter space coverage.
    
    Args:
        data: Dictionary containing all loaded data
        closest_matches: Dictionary of closest simulation matches
    """
    print("Creating parameter space visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Parameter space with connections
    ax1 = axes[0]
    
    # Plot simulation data (subset for clarity)
    sim_subset_indices = np.random.choice(len(data['y_sim']), 2000, replace=False)
    ax1.scatter(data['y_sim'][sim_subset_indices, 0], data['y_sim'][sim_subset_indices, 1], 
               alpha=0.3, s=1, color='lightblue', label='Simulation (subset)')
    
    # Plot experimental predictions
    exp_params = data['predictions'][['predicted_gap_um', 'predicted_L_um']].values
    ax1.scatter(exp_params[:, 0], exp_params[:, 1], 
               alpha=0.8, s=50, color='red', label='Experimental Predictions', zorder=5)
    
    # Plot closest simulation matches
    for exp_idx in range(len(data['predictions'])):
        closest_sim_idx = closest_matches[exp_idx]['sim_indices'][0]
        sim_params = data['y_sim'][closest_sim_idx]
        
        # Draw line connecting experimental prediction to closest simulation
        ax1.plot([exp_params[exp_idx, 0], sim_params[0]], 
                [exp_params[exp_idx, 1], sim_params[1]], 
                'gray', alpha=0.5, linewidth=0.5)
        
        # Plot closest simulation point
        ax1.scatter(sim_params[0], sim_params[1], 
                   alpha=0.8, s=30, color='blue', zorder=4)
    
    ax1.set_xlabel('Gap Parameter (µm)')
    ax1.set_ylabel('L_ecran Parameter (µm)')
    ax1.set_title('Parameter Space: Experimental Predictions vs Closest Simulation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distance distribution
    ax2 = axes[1]
    
    distances = [closest_matches[i]['distances'][0] for i in range(len(data['predictions']))]
    
    ax2.hist(distances, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Normalized Distance to Closest Simulation')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Distances to Closest Simulation Profiles')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    ax2.axvline(mean_dist, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_dist:.4f}')
    ax2.text(0.7, 0.8, f'Mean: {mean_dist:.4f}\nStd: {std_dist:.4f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    param_plot_path = 'parameter_space_analysis.png'
    plt.savefig(param_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Parameter space plot saved to: {param_plot_path}")
    
    plt.close()

def create_detailed_profile_comparison(data, closest_matches, sample_indices=[0, 10, 20, 30, 40]):
    """
    Create detailed comparison for selected samples.
    
    Args:
        data: Dictionary containing all loaded data
        closest_matches: Dictionary of closest simulation matches
        sample_indices: List of experimental sample indices to analyze in detail
    """
    print(f"Creating detailed comparison for samples: {sample_indices}")
    
    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(14, 3*len(sample_indices)))
    if len(sample_indices) == 1:
        axes = [axes]
    
    for i, exp_idx in enumerate(sample_indices):
        ax = axes[i]
        
        # Get experimental profile
        exp_profile = data['X_exp'][exp_idx]
        x_axis = data['x_exp']
        
        # Get multiple closest simulation profiles
        n_closest = min(3, len(closest_matches[exp_idx]['sim_indices']))
        
        # Plot experimental profile
        ax.plot(x_axis, exp_profile, 'r-', linewidth=3, label='Experimental', alpha=0.9)
        
        # Plot closest simulation profiles
        colors = ['blue', 'green', 'orange']
        for j in range(n_closest):
            sim_idx = closest_matches[exp_idx]['sim_indices'][j]
            sim_profile = data['X_sim'][sim_idx]
            sim_params = data['y_sim'][sim_idx]
            distance = closest_matches[exp_idx]['distances'][j]
            
            label = f'Sim {j+1}: gap={sim_params[0]:.3f}, L={sim_params[1]:.3f}, d={distance:.4f}'
            ax.plot(data['x_sim'], sim_profile, color=colors[j], linewidth=2, 
                   label=label, alpha=0.7, linestyle='--' if j > 0 else '-')
        
        # Get experimental parameters
        exp_params = closest_matches[exp_idx]['exp_params']
        
        # Add title and labels
        ax.set_title(f'Experimental Sample {exp_idx} - Predicted: gap={exp_params[0]:.3f} µm, L={exp_params[1]:.3f} µm', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Radial Position (µm)')
        ax.set_ylabel('Intensity')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add residual plot (inset)
        if i == 0:  # Only for first sample to avoid clutter
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
            
            closest_sim_idx = closest_matches[exp_idx]['sim_indices'][0]
            closest_sim_profile = data['X_sim'][closest_sim_idx]
            residual = exp_profile - closest_sim_profile
            
            axins.plot(x_axis, residual, 'purple', linewidth=1)
            axins.set_title('Residual', fontsize=8)
            axins.grid(True, alpha=0.3)
            axins.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    # Save the detailed plot
    detailed_plot_path = 'detailed_experimental_vs_simulation_comparison.png'
    plt.savefig(detailed_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Detailed comparison plot saved to: {detailed_plot_path}")
    
    plt.close()

def print_analysis_summary(data, closest_matches):
    """Print a summary of the analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENTAL VS SIMULATION ANALYSIS SUMMARY")
    print("=" * 70)
    
    distances = [closest_matches[i]['distances'][0] for i in range(len(data['predictions']))]
    
    print(f"Dataset Summary:")
    print(f"  Experimental samples: {len(data['predictions'])}")
    print(f"  Simulation samples: {len(data['y_sim'])}")
    
    print(f"\nClosest Match Analysis:")
    print(f"  Mean distance to closest simulation: {np.mean(distances):.4f}")
    print(f"  Std distance: {np.std(distances):.4f}")
    print(f"  Min distance: {np.min(distances):.4f}")
    print(f"  Max distance: {np.max(distances):.4f}")
    
    # Parameter range analysis
    exp_params = data['predictions'][['predicted_gap_um', 'predicted_L_um']].values
    
    print(f"\nParameter Range Comparison:")
    print(f"  Experimental gap range: {exp_params[:, 0].min():.3f} to {exp_params[:, 0].max():.3f} µm")
    print(f"  Simulation gap range: {data['y_sim'][:, 0].min():.3f} to {data['y_sim'][:, 0].max():.3f} µm")
    print(f"  Experimental L_ecran range: {exp_params[:, 1].min():.3f} to {exp_params[:, 1].max():.3f} µm")
    print(f"  Simulation L_ecran range: {data['y_sim'][:, 1].min():.3f} to {data['y_sim'][:, 1].max():.3f} µm")
    
    print(f"\nFiles Generated:")
    print(f"  ✓ experimental_vs_closest_simulation_profiles.png")
    print(f"  ✓ parameter_space_analysis.png")
    print(f"  ✓ detailed_experimental_vs_simulation_comparison.png")
    
    print("=" * 70)

def main():
    """Main function to create all comparison plots."""
    print("=" * 80)
    print("EXPERIMENTAL PROFILES VS CLOSEST SIMULATION MATCHES")
    print("=" * 80)
    
    # Load all data
    data = load_all_data()
    
    # Find closest simulation profiles
    closest_matches = find_closest_simulation_profiles(data['predictions'], data['y_sim'])
    
    # Create comparison plots
    create_comparison_plots(data, closest_matches, n_samples=12)
    
    # Create parameter space visualization
    create_parameter_space_visualization(data, closest_matches)
    
    # Create detailed comparison for selected samples
    create_detailed_profile_comparison(data, closest_matches)
    
    # Print analysis summary
    print_analysis_summary(data, closest_matches)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
