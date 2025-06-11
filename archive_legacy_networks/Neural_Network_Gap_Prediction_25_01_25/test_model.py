"""
Model Testing Script for Gap Parameter Prediction
Author: Oussama GUELFAA
Date: 10 - 06 - 2025

This script provides functionality to test the trained neural network model
on new intensity profile data and validate its performance.
"""

import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from gap_prediction_neural_network import GapPredictionCNN, tolerance_based_accuracy
import os

def load_trained_model(model_path='models/best_model.pth', scaler_path='models/scaler.pkl'):
    """
    Load the trained model and scaler for inference.
    
    Args:
        model_path (str): Path to the saved model weights
        scaler_path (str): Path to the saved scaler
        
    Returns:
        tuple: (model, scaler) - loaded model and scaler objects
    """
    # Initialize model architecture
    model = GapPredictionCNN(input_size=1000, num_classes=1)
    
    # Load trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully from {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    return model, scaler

def predict_gap(model, scaler, intensity_profiles):
    """
    Predict gap parameters for given intensity profiles.
    
    Args:
        model (nn.Module): Trained neural network model
        scaler (StandardScaler): Fitted data scaler
        intensity_profiles (np.ndarray): Array of intensity profiles
        
    Returns:
        np.ndarray: Predicted gap values
    """
    # Ensure input is 2D array
    if intensity_profiles.ndim == 1:
        intensity_profiles = intensity_profiles.reshape(1, -1)
    
    # Normalize input data
    intensity_profiles_scaled = scaler.transform(intensity_profiles)
    
    # Convert to PyTorch tensor
    input_tensor = torch.FloatTensor(intensity_profiles_scaled)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(input_tensor)
        predictions = predictions.cpu().numpy().flatten()
    
    return predictions

def test_on_validation_data():
    """
    Test the model on a subset of the original validation data.
    """
    print("Testing model on validation data...")
    
    # Load model and scaler
    model, scaler = load_trained_model()
    
    # Load original data
    intensity_df = pd.read_csv('../data/processed/intensity_profiles_full.csv', header=None)
    params_df = pd.read_csv('../data/processed/parameters.csv')
    
    # Take a random sample for testing
    np.random.seed(42)
    test_indices = np.random.choice(len(intensity_df), size=50, replace=False)
    
    test_intensities = intensity_df.iloc[test_indices].values
    test_gaps = params_df.iloc[test_indices]['gap'].values
    
    # Make predictions
    predicted_gaps = predict_gap(model, scaler, test_intensities)
    
    # Calculate metrics
    tolerance_acc = tolerance_based_accuracy(test_gaps, predicted_gaps, tolerance=0.01)
    mae = np.mean(np.abs(test_gaps - predicted_gaps))
    rmse = np.sqrt(np.mean((test_gaps - predicted_gaps)**2))
    
    print(f"\nTest Results on {len(test_indices)} samples:")
    print(f"Tolerance Accuracy (±0.01): {tolerance_acc:.2f}%")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Predictions vs Actual
    plt.subplot(1, 2, 1)
    plt.scatter(test_gaps, predicted_gaps, alpha=0.7)
    plt.plot([test_gaps.min(), test_gaps.max()], [test_gaps.min(), test_gaps.max()], 'r--')
    plt.xlabel('Actual Gap Values')
    plt.ylabel('Predicted Gap Values')
    plt.title('Test Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Prediction errors
    errors = predicted_gaps - test_gaps
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=15, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution (MAE: {mae:.6f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_gaps, predicted_gaps

def predict_single_profile(intensity_profile):
    """
    Predict gap parameter for a single intensity profile.
    
    Args:
        intensity_profile (np.ndarray): Single intensity profile (1000 features)
        
    Returns:
        float: Predicted gap value
    """
    # Load model and scaler
    model, scaler = load_trained_model()
    
    # Make prediction
    prediction = predict_gap(model, scaler, intensity_profile)
    
    return prediction[0]

def batch_prediction_example():
    """
    Example of batch prediction on multiple intensity profiles.
    """
    print("Running batch prediction example...")
    
    # Load model and scaler
    model, scaler = load_trained_model()
    
    # Load some sample data
    intensity_df = pd.read_csv('../data/processed/intensity_profiles_full.csv', header=None)
    params_df = pd.read_csv('../data/processed/parameters.csv')
    
    # Select first 10 samples
    sample_intensities = intensity_df.iloc[:10].values
    actual_gaps = params_df.iloc[:10]['gap'].values
    
    # Make predictions
    predicted_gaps = predict_gap(model, scaler, sample_intensities)
    
    # Display results
    print("\nBatch Prediction Results:")
    print("-" * 50)
    print(f"{'Index':<6} {'Actual':<10} {'Predicted':<12} {'Error':<10}")
    print("-" * 50)
    
    for i, (actual, predicted) in enumerate(zip(actual_gaps, predicted_gaps)):
        error = abs(actual - predicted)
        print(f"{i:<6} {actual:<10.6f} {predicted:<12.6f} {error:<10.6f}")
    
    print("-" * 50)
    tolerance_acc = tolerance_based_accuracy(actual_gaps, predicted_gaps, tolerance=0.01)
    print(f"Tolerance Accuracy (±0.01): {tolerance_acc:.2f}%")

def main():
    """
    Main function to run model testing.
    """
    print("="*60)
    print("GAP PARAMETER PREDICTION - MODEL TESTING")
    print("Author: Oussama GUELFAA")
    print("Date: 25 - 01 - 2025")
    print("="*60)
    
    try:
        # Test 1: Validation data testing
        print("\n1. Testing on validation data subset...")
        test_on_validation_data()
        
        # Test 2: Batch prediction example
        print("\n2. Batch prediction example...")
        batch_prediction_example()
        
        # Test 3: Single profile prediction
        print("\n3. Single profile prediction example...")
        intensity_df = pd.read_csv('../data/processed/intensity_profiles_full.csv', header=None)
        sample_profile = intensity_df.iloc[0].values
        predicted_gap = predict_single_profile(sample_profile)
        print(f"Predicted gap for sample profile: {predicted_gap:.6f}")
        
        print("\n" + "="*60)
        print("MODEL TESTING COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("Please ensure the model has been trained and saved properly.")

if __name__ == "__main__":
    main()
