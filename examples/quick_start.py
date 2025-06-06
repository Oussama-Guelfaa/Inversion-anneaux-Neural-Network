#!/usr/bin/env python3
"""
Quick Start Example for Inversion-anneaux-Neural-Network
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

This script demonstrates how to use the trained neural network models
for holographic parameter prediction.
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_sample_data():
    """Load sample intensity profile data."""
    
    # Check if processed data exists
    data_path = "../data/processed/training_data.npz"
    
    if os.path.exists(data_path):
        print("Loading sample data from processed dataset...")
        data = np.load(data_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        
        # Use first 5 samples as examples
        sample_X = X[:5]
        sample_y = y[:5]
        
        return sample_X, sample_y
    else:
        print("No processed data found. Generating synthetic sample...")
        # Generate synthetic intensity profile
        x = np.linspace(0, 10, 1000)
        
        # Simulate ring pattern with some parameters
        L_ecran = 8.0  # µm
        gap = 0.5      # µm
        
        # Simple ring pattern simulation
        intensity = 1.0 + 0.3 * np.cos(2 * np.pi * x / gap) * np.exp(-x / L_ecran)
        intensity += 0.1 * np.random.normal(0, 1, len(x))  # Add noise
        
        sample_X = intensity.reshape(1, -1)
        sample_y = np.array([[L_ecran, gap]])
        
        return sample_X, sample_y

def load_trained_model():
    """Load the trained PyTorch model."""
    
    model_path = "../models/pytorch/ring_regressor_new.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first using:")
        print("python src/training/train_pytorch.py")
        return None
    
    try:
        # Import model architecture
        from training.train_pytorch import RingProfileRegressor
        
        # Load model
        model = RingProfileRegressor(input_dim=1000, hidden_dims=[512, 256, 128], output_dim=2)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_parameters(model, intensity_profiles):
    """Predict L_ecran and gap from intensity profiles."""
    
    if model is None:
        return None
    
    with torch.no_grad():
        # Convert to tensor
        X_tensor = torch.FloatTensor(intensity_profiles)
        
        # Make predictions
        predictions = model(X_tensor)
        
        return predictions.numpy()

def visualize_results(intensity_profiles, true_params, predicted_params):
    """Visualize the intensity profiles and predictions."""
    
    n_samples = len(intensity_profiles)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_samples):
        # Plot intensity profile
        axes[0, i].plot(intensity_profiles[i])
        axes[0, i].set_title(f'Sample {i+1}: Intensity Profile')
        axes[0, i].set_xlabel('Radial Position')
        axes[0, i].set_ylabel('Intensity Ratio')
        axes[0, i].grid(True)
        
        # Plot parameter comparison
        params = ['L_ecran (µm)', 'gap (µm)']
        x_pos = np.arange(len(params))
        
        if true_params is not None:
            axes[1, i].bar(x_pos - 0.2, true_params[i], 0.4, label='True', alpha=0.7)
        
        if predicted_params is not None:
            axes[1, i].bar(x_pos + 0.2, predicted_params[i], 0.4, label='Predicted', alpha=0.7)
        
        axes[1, i].set_title(f'Sample {i+1}: Parameters')
        axes[1, i].set_xticks(x_pos)
        axes[1, i].set_xticklabels(params)
        axes[1, i].legend()
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.savefig('quick_start_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function demonstrating the neural network usage."""
    
    print("="*60)
    print("QUICK START: Holographic Parameter Prediction")
    print("="*60)
    
    # 1. Load sample data
    print("\n1. Loading sample data...")
    sample_X, sample_y = load_sample_data()
    print(f"Loaded {len(sample_X)} sample(s)")
    print(f"Intensity profile shape: {sample_X.shape}")
    print(f"Parameters shape: {sample_y.shape}")
    
    # 2. Load trained model
    print("\n2. Loading trained model...")
    model = load_trained_model()
    
    # 3. Make predictions
    print("\n3. Making predictions...")
    if model is not None:
        predicted_params = predict_parameters(model, sample_X)
        print("Predictions completed!")
        
        # Display results
        print("\nResults:")
        for i in range(len(sample_X)):
            print(f"Sample {i+1}:")
            if sample_y is not None:
                print(f"  True:      L_ecran={sample_y[i,0]:.3f} µm, gap={sample_y[i,1]:.3f} µm")
            print(f"  Predicted: L_ecran={predicted_params[i,0]:.3f} µm, gap={predicted_params[i,1]:.3f} µm")
            
            if sample_y is not None:
                error_L = abs(sample_y[i,0] - predicted_params[i,0])
                error_gap = abs(sample_y[i,1] - predicted_params[i,1])
                print(f"  Error:     L_ecran={error_L:.3f} µm, gap={error_gap:.3f} µm")
            print()
    else:
        predicted_params = None
        print("Could not make predictions (model not loaded)")
    
    # 4. Visualize results
    print("4. Generating visualization...")
    visualize_results(sample_X, sample_y, predicted_params)
    print("Results saved as 'quick_start_results.png'")
    
    print("\n" + "="*60)
    print("Quick start completed!")
    print("For more advanced usage, see the documentation in docs/")
    print("="*60)

if __name__ == "__main__":
    main()
