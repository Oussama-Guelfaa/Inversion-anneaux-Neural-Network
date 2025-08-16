#!/usr/bin/env python3
"""
Data Processing Script for Simulation Data
Author: Oussama GUELFAA
Date: 01/08/2025

This script loads simulation data from a MATLAB file, truncates the first 250 points
from each profile, and saves the processed data to a compressed .npz file.

Input: all_banque_new_04_07_25_NEW_full.mat
Output: simulation_processed_truncate250_start.npz
"""

import numpy as np
import scipy.io as sio
import os
import sys

def load_and_process_simulation_data():
    """
    Load simulation data from MATLAB file and process it by truncating first 250 points.
    
    Returns:
        tuple: (X_data, x_positions) - processed intensity profiles and x-axis values
    """
    # Define input file path
    input_file = "../Neural_Network_Gap_Lecran_Prediction/data/raw/Train/all_banque_new_04_07_25_NEW_full.mat"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"Loading data from: {input_file}")
    
    try:
        # Load MATLAB file
        mat_data = sio.loadmat(input_file)

        # Print available variables in the MATLAB file
        print("Available variables in MATLAB file:")
        for key in mat_data.keys():
            if not key.startswith('__'):
                print(f"  {key}: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else type(mat_data[key])}")

        # Look for intensity data - check multiple possible variables
        intensity_data = None
        x_axis_data = None

        # Check for I_subs which seems to contain the main intensity data
        if 'I_subs' in mat_data:
            I_subs = mat_data['I_subs']  # Shape: (161, 140, 1000)
            print(f"Found I_subs with shape: {I_subs.shape}")

            # Reshape to 2D: (num_profiles, num_points)
            # I_subs has shape (161, 140, 1000) -> reshape to (161*140, 1000)
            intensity_data = I_subs.reshape(-1, I_subs.shape[-1])
            print(f"Reshaped intensity data to: {intensity_data.shape}")

        elif 'ratio' in mat_data:
            ratio = mat_data['ratio']
            if ratio.shape[0] == 1000 and ratio.shape[1] == 1:
                # Single profile case - transpose to get (1, 1000)
                intensity_data = ratio.T
                print(f"Using single ratio profile, transposed to: {intensity_data.shape}")
            else:
                intensity_data = ratio
                print(f"Using ratio data with shape: {intensity_data.shape}")
        else:
            print("Error: No intensity data found (I_subs or ratio)")
            sys.exit(1)

        # Look for x-axis data
        if 'x' in mat_data:
            x_single = mat_data['x']  # Shape: (1, 1000)
            # Replicate x-axis for all profiles
            x_axis_data = np.tile(x_single, (intensity_data.shape[0], 1))
            print(f"Found x-axis data, replicated to shape: {x_axis_data.shape}")
        else:
            print("Warning: No x-axis variable found. Creating default x-axis...")
            # Create default x-axis (assuming uniform spacing)
            x_axis_data = np.tile(np.arange(intensity_data.shape[1]), (intensity_data.shape[0], 1))

        print(f"Final data shapes:")
        print(f"  intensity_data: {intensity_data.shape}")
        print(f"  x_axis_data: {x_axis_data.shape}")

        # Verify that each profile has 1000 points
        if intensity_data.shape[1] != 1000:
            print(f"Warning: Expected 1000 points per profile, got {intensity_data.shape[1]}")

        # Truncate first 250 points from each profile
        truncate_start = 250
        X_data = intensity_data[:, truncate_start:]  # Keep points from index 250 onwards
        x_positions = x_axis_data[:, truncate_start:]  # Keep corresponding x-axis values

        print(f"After truncation (removing first {truncate_start} points):")
        print(f"  X_data: {X_data.shape}")
        print(f"  x_positions: {x_positions.shape}")
        print(f"  Final profile length: {X_data.shape[1]} points")

        return X_data, x_positions
        
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        sys.exit(1)

def save_processed_data(X_data, x_positions, output_file):
    """
    Save processed data to compressed .npz file.
    
    Args:
        X_data (numpy.ndarray): Truncated intensity profiles
        x_positions (numpy.ndarray): Truncated x-axis values
        output_file (str): Output file path
    """
    try:
        # Save to compressed .npz file
        np.savez_compressed(output_file, 
                          X_data=X_data, 
                          x_positions=x_positions)
        
        print(f"Data saved successfully to: {output_file}")
        
        # Verify saved file
        loaded_data = np.load(output_file)
        print(f"Verification - saved arrays:")
        print(f"  X_data: {loaded_data['X_data'].shape}")
        print(f"  x_positions: {loaded_data['x_positions'].shape}")
        
        # Calculate file size
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"  File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)

def main():
    """
    Main function to process simulation data.
    """
    print("=" * 60)
    print("Simulation Data Processing")
    print("=" * 60)
    
    # Load and process data
    X_data, x_positions = load_and_process_simulation_data()
    
    # Define output file
    output_file = "simulation_processed_truncate250_start.npz"
    
    # Save processed data
    save_processed_data(X_data, x_positions, output_file)
    
    print("=" * 60)
    print("Processing completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
