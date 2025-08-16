#!/usr/bin/env python3
"""
Combine Simulation Data with Labels
Author: Oussama GUELFAA
Date: 01/08/2025

This script combines the processed simulation data with the corresponding labels
from the CSV file to create a complete dataset ready for neural network training.

Input:
- simulation_processed_truncate250_start.npz (simulation data)
- labels.csv (gap and L_ecran parameters)

Output:
- simulation_processed_with_labels.npz (combined dataset)
"""

import numpy as np
import pandas as pd
import os
import sys

def load_simulation_data():
    """
    Load simulation data from .npz file.
    
    Returns:
        tuple: (X_data, x_positions) - intensity profiles and x-axis data
    """
    sim_file = "simulation_processed_truncate250_start.npz"
    
    if not os.path.exists(sim_file):
        print(f"Error: Simulation file not found: {sim_file}")
        sys.exit(1)
    
    print(f"Loading simulation data from: {sim_file}")
    
    try:
        sim_data = np.load(sim_file)
        
        print("Available arrays in simulation file:")
        for key in sim_data.keys():
            print(f"  {key}: {sim_data[key].shape}")
        
        X_data = sim_data['X_data']
        x_positions = sim_data['x_positions']
        
        print(f"Simulation data loaded:")
        print(f"  X_data: {X_data.shape}")
        print(f"  x_positions: {x_positions.shape}")
        print(f"  X_data range: {X_data.min():.6f} to {X_data.max():.6f}")
        print(f"  Data type: {X_data.dtype}")
        
        return X_data, x_positions
        
    except Exception as e:
        print(f"Error loading simulation file: {e}")
        sys.exit(1)

def load_labels_data():
    """
    Load labels from CSV file.
    
    Returns:
        numpy.ndarray: y_data with shape (N, 2) containing [gap_um, L_um]
    """
    labels_file = "../Neural_Network_Gap_Lecran_Prediction/data/raw/Train/labels.csv"
    
    if not os.path.exists(labels_file):
        print(f"Error: Labels file not found: {labels_file}")
        sys.exit(1)
    
    print(f"Loading labels from: {labels_file}")
    
    try:
        # Load CSV file
        labels_df = pd.read_csv(labels_file)
        
        print(f"Labels CSV loaded:")
        print(f"  Shape: {labels_df.shape}")
        print(f"  Columns: {list(labels_df.columns)}")
        
        # Display first few rows
        print(f"  First 5 rows:")
        print(labels_df.head())
        
        # Check for required columns
        required_columns = ['gap_um', 'L_um']
        missing_columns = [col for col in required_columns if col not in labels_df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            sys.exit(1)
        
        # Extract the two label columns
        y_data = labels_df[['gap_um', 'L_um']].values
        
        print(f"Labels extracted:")
        print(f"  y_data shape: {y_data.shape}")
        print(f"  gap_um range: {y_data[:, 0].min():.6f} to {y_data[:, 0].max():.6f}")
        print(f"  L_um range: {y_data[:, 1].min():.6f} to {y_data[:, 1].max():.6f}")
        print(f"  Data type: {y_data.dtype}")
        
        # Check for any missing values
        if np.any(np.isnan(y_data)):
            nan_count = np.sum(np.isnan(y_data))
            print(f"  Warning: {nan_count} NaN values found in labels")
        else:
            print(f"  ✓ No NaN values in labels")
        
        return y_data
        
    except Exception as e:
        print(f"Error loading labels file: {e}")
        sys.exit(1)

def verify_data_consistency(X_data, y_data):
    """
    Verify that simulation data and labels have consistent dimensions.
    
    Args:
        X_data (numpy.ndarray): Intensity profiles
        y_data (numpy.ndarray): Labels
    
    Returns:
        bool: True if data is consistent
    """
    print("Verifying data consistency...")
    
    # Check number of samples
    if X_data.shape[0] != y_data.shape[0]:
        print(f"Error: Sample count mismatch!")
        print(f"  X_data samples: {X_data.shape[0]}")
        print(f"  y_data samples: {y_data.shape[0]}")
        return False
    
    # Check expected dimensions
    expected_samples = 22540
    if X_data.shape[0] != expected_samples:
        print(f"Warning: Expected {expected_samples} samples, got {X_data.shape[0]}")
    
    # Check label dimensions
    if y_data.shape[1] != 2:
        print(f"Error: Expected 2 label columns, got {y_data.shape[1]}")
        return False
    
    print(f"✓ Data consistency verified:")
    print(f"  Number of samples: {X_data.shape[0]}")
    print(f"  X_data shape: {X_data.shape}")
    print(f"  y_data shape: {y_data.shape}")
    
    return True

def save_combined_data(X_data, x_positions, y_data):
    """
    Save combined simulation data and labels to .npz file.
    
    Args:
        X_data (numpy.ndarray): Intensity profiles
        x_positions (numpy.ndarray): X-axis data
        y_data (numpy.ndarray): Labels
    """
    output_file = "simulation_processed_with_labels.npz"
    
    print(f"Saving combined data to: {output_file}")
    
    try:
        # Convert to float32 to reduce file size
        X_data_f32 = X_data.astype(np.float32)
        x_positions_f32 = x_positions.astype(np.float32)
        y_data_f32 = y_data.astype(np.float32)
        
        print(f"Data types after conversion:")
        print(f"  X_data: {X_data_f32.dtype}")
        print(f"  x_positions: {x_positions_f32.dtype}")
        print(f"  y_data: {y_data_f32.dtype}")
        
        # Save to compressed .npz file
        np.savez_compressed(output_file,
                          X_data=X_data_f32,
                          x_positions=x_positions_f32,
                          y_data=y_data_f32)
        
        print(f"✓ Combined data saved successfully!")
        
        # Verify saved file
        print("Verifying saved file...")
        loaded_data = np.load(output_file)
        
        print(f"Verification - saved arrays:")
        for key in loaded_data.keys():
            array = loaded_data[key]
            print(f"  {key}: {array.shape}, dtype={array.dtype}")
        
        # Calculate file size
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.2f} MB")
        
        # Quick data integrity check
        loaded_X = loaded_data['X_data']
        loaded_y = loaded_data['y_data']
        
        print(f"Data integrity check:")
        print(f"  X_data range: {loaded_X.min():.6f} to {loaded_X.max():.6f}")
        print(f"  gap_um range: {loaded_y[:, 0].min():.6f} to {loaded_y[:, 0].max():.6f}")
        print(f"  L_um range: {loaded_y[:, 1].min():.6f} to {loaded_y[:, 1].max():.6f}")
        
        print("✓ File verification completed successfully!")
        
    except Exception as e:
        print(f"Error saving combined data: {e}")
        sys.exit(1)

def display_data_summary(X_data, x_positions, y_data):
    """
    Display a comprehensive summary of the combined dataset.
    
    Args:
        X_data (numpy.ndarray): Intensity profiles
        x_positions (numpy.ndarray): X-axis data
        y_data (numpy.ndarray): Labels
    """
    print("\n" + "="*70)
    print("COMBINED DATASET SUMMARY")
    print("="*70)
    
    print(f"Dataset dimensions:")
    print(f"  Number of samples: {X_data.shape[0]:,}")
    print(f"  Points per profile: {X_data.shape[1]:,}")
    print(f"  Total data points: {X_data.size:,}")
    print(f"  Number of target parameters: {y_data.shape[1]}")
    
    print(f"\nIntensity profiles (X_data):")
    print(f"  Shape: {X_data.shape}")
    print(f"  Range: {X_data.min():.6f} to {X_data.max():.6f}")
    print(f"  Mean: {X_data.mean():.6f} ± {X_data.std():.6f}")
    
    print(f"\nX-axis data (x_positions):")
    print(f"  Shape: {x_positions.shape}")
    print(f"  Range: {x_positions[0].min():.6f} to {x_positions[0].max():.6f} µm")
    
    print(f"\nTarget parameters (y_data):")
    print(f"  Shape: {y_data.shape}")
    print(f"  Gap parameter (µm):")
    print(f"    Range: {y_data[:, 0].min():.6f} to {y_data[:, 0].max():.6f}")
    print(f"    Mean: {y_data[:, 0].mean():.6f} ± {y_data[:, 0].std():.6f}")
    print(f"  L_ecran parameter (µm):")
    print(f"    Range: {y_data[:, 1].min():.6f} to {y_data[:, 1].max():.6f}")
    print(f"    Mean: {y_data[:, 1].mean():.6f} ± {y_data[:, 1].std():.6f}")
    
    print(f"\nDataset is ready for neural network training!")
    print("="*70)

def main():
    """
    Main function to combine simulation data with labels.
    """
    print("=" * 70)
    print("COMBINING SIMULATION DATA WITH LABELS")
    print("=" * 70)
    
    # Step 1: Load simulation data
    X_data, x_positions = load_simulation_data()
    
    print()
    
    # Step 2: Load labels
    y_data = load_labels_data()
    
    print()
    
    # Step 3: Verify data consistency
    if not verify_data_consistency(X_data, y_data):
        print("Data consistency check failed. Exiting.")
        sys.exit(1)
    
    print()
    
    # Step 4: Save combined data
    save_combined_data(X_data, x_positions, y_data)
    
    # Step 5: Display summary
    display_data_summary(X_data, x_positions, y_data)
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
