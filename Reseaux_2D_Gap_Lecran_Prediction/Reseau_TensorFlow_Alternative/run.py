#!/usr/bin/env python3
"""
TensorFlow/Keras Alternative - Autonomous Training Script
Author: Oussama GUELFAA
Date: 10 - 01 - 2025

Script autonome pour l'alternative TensorFlow/Keras.
Architecture Dense avec callbacks Keras optimisés.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. Please install: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

import json
import time
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration pour reproductibilité
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(42)
np.random.seed(42)

def load_config(config_path="config/tensorflow_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_data_tensorflow(config):
    """Extract data for TensorFlow training."""
    print("🔄 TensorFlow data extraction...")
    
    try:
        # Load MATLAB file
        mat_file = config['paths']['data_file']
        data = sio.loadmat(mat_file)
        
        # Extract variables
        L_ecran_vect = data['L_ecran_subs_vect'].flatten()
        gap_vect = data['gap_sphere_vect'].flatten()
        I_subs = data['I_subs']
        I_subs_inc = data['I_subs_inc']
        
        # Use full profiles
        profile_length = config['model']['input_size']
        
        intensity_profiles = []
        L_ecran_values = []
        gap_values = []
        
        for i in range(len(L_ecran_vect)):
            for j in range(len(gap_vect)):
                ratio = np.abs(I_subs[i, j, :profile_length])
                ratio_inc = np.abs(I_subs_inc[i, j, :profile_length])
                profile = ratio / ratio_inc
                
                intensity_profiles.append(profile)
                L_ecran_values.append(L_ecran_vect[i])
                gap_values.append(gap_vect[j])
        
        intensity_profiles = np.array(intensity_profiles)
        parameters = np.column_stack([L_ecran_values, gap_values])
        
        print(f"✅ TensorFlow data extracted: {intensity_profiles.shape}")
        
        return intensity_profiles, parameters
        
    except Exception as e:
        print(f"❌ Error in TensorFlow data extraction: {e}")
        return None, None

def create_tensorflow_model(config):
    """Create TensorFlow/Keras model."""
    print("🏗️ Creating TensorFlow/Keras model...")
    
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow not available")
    
    input_size = config['model']['input_size']
    output_size = config['model']['output_size']
    
    # Architecture selon les spécifications mémoire: 512→256→128→64→2
    model = keras.Sequential([
        layers.Input(shape=(input_size,)),
        
        # Couche 1: 512 neurones
        layers.Dense(512, activation='relu', 
                    kernel_regularizer=regularizers.l2(config['architecture']['l2_lambda'])),
        layers.BatchNormalization() if config['architecture']['batch_normalization'] else layers.Lambda(lambda x: x),
        layers.Dropout(0.2),
        
        # Couche 2: 256 neurones
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(config['architecture']['l2_lambda'])),
        layers.BatchNormalization() if config['architecture']['batch_normalization'] else layers.Lambda(lambda x: x),
        layers.Dropout(0.2),
        
        # Couche 3: 128 neurones
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(config['architecture']['l2_lambda'])),
        layers.BatchNormalization() if config['architecture']['batch_normalization'] else layers.Lambda(lambda x: x),
        layers.Dropout(0.2),
        
        # Couche 4: 64 neurones
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(config['architecture']['l2_lambda'])),
        layers.BatchNormalization() if config['architecture']['batch_normalization'] else layers.Lambda(lambda x: x),
        layers.Dropout(0.2),
        
        # Couche de sortie: 2 neurones (L_ecran, gap)
        layers.Dense(output_size, activation='linear')
    ])
    
    # Compilation du modèle
    optimizer = keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss=config['training']['loss_function'],
        metrics=config['training']['metrics']
    )
    
    print("✅ TensorFlow model created and compiled")
    print(f"   Total parameters: {model.count_params():,}")
    
    return model

def create_keras_callbacks(config):
    """Create Keras callbacks."""
    print("⚙️ Setting up Keras callbacks...")
    
    callback_list = []
    
    # Early Stopping
    if config['training']['callbacks']['early_stopping']['enable']:
        early_stopping = callbacks.EarlyStopping(
            monitor=config['training']['callbacks']['early_stopping']['monitor'],
            patience=config['training']['callbacks']['early_stopping']['patience'],
            restore_best_weights=config['training']['callbacks']['early_stopping']['restore_best_weights'],
            verbose=1
        )
        callback_list.append(early_stopping)
        print("   ✅ Early Stopping callback added")
    
    # Reduce Learning Rate
    if config['training']['callbacks']['reduce_lr']['enable']:
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor=config['training']['callbacks']['reduce_lr']['monitor'],
            factor=config['training']['callbacks']['reduce_lr']['factor'],
            patience=config['training']['callbacks']['reduce_lr']['patience'],
            min_lr=config['training']['callbacks']['reduce_lr']['min_lr'],
            verbose=1
        )
        callback_list.append(reduce_lr)
        print("   ✅ ReduceLROnPlateau callback added")
    
    # Model Checkpoint
    if config['training']['callbacks']['model_checkpoint']['enable']:
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=config['training']['callbacks']['model_checkpoint']['filepath'],
            monitor=config['training']['callbacks']['model_checkpoint']['monitor'],
            save_best_only=config['training']['callbacks']['model_checkpoint']['save_best_only'],
            verbose=1
        )
        callback_list.append(model_checkpoint)
        print("   ✅ ModelCheckpoint callback added")
    
    # TensorBoard (optionnel)
    if config['tensorflow_specific']['tensorboard']['enable']:
        tensorboard = callbacks.TensorBoard(
            log_dir=config['tensorflow_specific']['tensorboard']['log_dir'],
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callback_list.append(tensorboard)
        print("   ✅ TensorBoard callback added")
    
    return callback_list

def train_tensorflow_model(model, X_train, X_val, y_train, y_val, config):
    """Train TensorFlow model."""
    print("\n🚀 TENSORFLOW/KERAS TRAINING")
    print("=" * 50)
    
    # Create callbacks
    keras_callbacks = create_keras_callbacks(config)
    
    # Training parameters
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    
    print(f"🏃 Starting TensorFlow training...")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {epochs}")
    print(f"   Callbacks: {len(keras_callbacks)}")
    
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=keras_callbacks,
        verbose=1,
        shuffle=config['data']['shuffle'],
        use_multiprocessing=config['tensorflow_specific']['use_multiprocessing'],
        workers=config['tensorflow_specific']['workers']
    )
    
    training_time = time.time() - start_time
    
    print(f"✅ TensorFlow training completed in {training_time/60:.1f} minutes")
    
    return history, training_time

def evaluate_tensorflow_model(model, X_test, y_test, config):
    """Evaluate TensorFlow model."""
    print("\n🔍 TENSORFLOW MODEL EVALUATION")
    print("=" * 40)
    
    # Get predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate metrics
    r2_global = r2_score(y_test, y_pred)
    r2_L = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_gap = r2_score(y_test[:, 1], y_pred[:, 1])
    
    rmse_L = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    
    mae_L = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_gap = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    
    # TensorFlow native evaluation
    tf_loss, tf_mae, tf_mse = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"📊 TENSORFLOW RESULTS:")
    print(f"   R² global: {r2_global:.6f}")
    print(f"   R² L_ecran: {r2_L:.6f}")
    print(f"   R² gap: {r2_gap:.6f}")
    print(f"   RMSE L_ecran: {rmse_L:.6f}")
    print(f"   RMSE gap: {rmse_gap:.6f}")
    print(f"   MAE L_ecran: {mae_L:.6f}")
    print(f"   MAE gap: {mae_gap:.6f}")
    print(f"   TensorFlow Loss: {tf_loss:.6f}")
    print(f"   TensorFlow MAE: {tf_mae:.6f}")
    print(f"   TensorFlow MSE: {tf_mse:.6f}")
    
    # Check targets
    target_r2_global = config['evaluation']['performance_targets']['r2_global']
    success = r2_global >= target_r2_global
    
    print(f"   Target R² > {target_r2_global}: {'✅' if success else '❌'}")
    
    return y_pred, {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'tf_loss': tf_loss, 'tf_mae': tf_mae, 'tf_mse': tf_mse,
        'success': success
    }

def create_tensorflow_plots(history, y_true, y_pred, config):
    """Create TensorFlow-specific plots."""
    print("\n📊 Generating TensorFlow plots...")
    
    # Training history plot
    plt.figure(figsize=(15, 10))
    
    # Loss curves
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE curves
    plt.subplot(2, 3, 2)
    plt.plot(history.history['mae'], label='Train MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate (if available)
    plt.subplot(2, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], linewidth=2)
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                ha='center', va='center', transform=plt.gca().transAxes)
    plt.grid(True, alpha=0.3)
    
    # Predictions scatter plots
    plt.subplot(2, 3, 4)
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6)
    plt.plot([y_true[:, 0].min(), y_true[:, 0].max()], 
             [y_true[:, 0].min(), y_true[:, 0].max()], 'r--', linewidth=2)
    plt.xlabel('True L_ecran')
    plt.ylabel('Predicted L_ecran')
    plt.title('L_ecran Predictions')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.6)
    plt.plot([y_true[:, 1].min(), y_true[:, 1].max()], 
             [y_true[:, 1].min(), y_true[:, 1].max()], 'r--', linewidth=2)
    plt.xlabel('True Gap')
    plt.ylabel('Predicted Gap')
    plt.title('Gap Predictions')
    plt.grid(True, alpha=0.3)
    
    # Residuals
    plt.subplot(2, 3, 6)
    residuals_L = y_pred[:, 0] - y_true[:, 0]
    residuals_gap = y_pred[:, 1] - y_true[:, 1]
    plt.hist(residuals_L, bins=30, alpha=0.7, label='L_ecran', density=True)
    plt.hist(residuals_gap, bins=30, alpha=0.7, label='Gap', density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residuals Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/tensorflow_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ TensorFlow plots saved to plots/tensorflow_analysis.png")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='TensorFlow/Keras Alternative Training')
    parser.add_argument('--config', default='config/tensorflow_config.yaml',
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 TENSORFLOW/KERAS ALTERNATIVE - DENSE ARCHITECTURE")
    print("Architecture Dense 512→256→128→64→2 avec callbacks Keras")
    print("="*80)
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Please install: pip install tensorflow")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Extract data
    intensity_profiles, parameters = extract_data_tensorflow(config)
    
    if intensity_profiles is None:
        print("❌ Failed to extract data. Training aborted.")
        return
    
    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(intensity_profiles)
    y_scaled = scaler_y.fit_transform(parameters)
    
    # Split data
    validation_split = config['data']['validation_split']
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=validation_split, random_state=42
    )
    
    print(f"\n📊 TensorFlow data splits:")
    print(f"   Train: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    
    # Create model
    model = create_tensorflow_model(config)
    
    # Train model
    history, training_time = train_tensorflow_model(model, X_train, X_val, y_train, y_val, config)
    
    # Evaluate model
    y_pred, metrics = evaluate_tensorflow_model(model, X_val, y_val, config)
    
    # Create plots
    create_tensorflow_plots(history, y_val, y_pred, config)
    
    # Save scalers
    joblib.dump(scaler_X, 'models/tensorflow_scaler_X.pkl')
    joblib.dump(scaler_y, 'models/tensorflow_scaler_y.pkl')
    
    # Save results
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('results/tensorflow_training_history.csv', index=False)
    
    with open('results/tensorflow_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model summary
    with open('results/tensorflow_model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"\n{'='*80}")
    print(f"🎯 TENSORFLOW/KERAS RESULTS")
    print(f"{'='*80}")
    
    print(f"🔧 TENSORFLOW FEATURES:")
    print(f"   • Architecture Dense 512→256→128→64→2")
    print(f"   • Callbacks Keras automatiques")
    print(f"   • Early stopping et ReduceLROnPlateau")
    print(f"   • Sauvegarde native .h5")
    
    print(f"\n📊 PERFORMANCES:")
    print(f"   • R² global: {metrics['r2_global']:.6f}")
    print(f"   • R² L_ecran: {metrics['r2_L']:.6f}")
    print(f"   • R² gap: {metrics['r2_gap']:.6f}")
    print(f"   • Temps d'entraînement: {training_time/60:.1f} min")
    print(f"   • Succès: {'🎉 YES' if metrics['success'] else '⚠️ NO'}")
    
    print(f"\n📁 FICHIERS TENSORFLOW:")
    print(f"   • models/tensorflow_best_model.h5")
    print(f"   • models/tensorflow_scaler_*.pkl")
    print(f"   • results/tensorflow_training_history.csv")
    print(f"   • results/tensorflow_model_summary.txt")
    
    print("\n🏁 TensorFlow/Keras training completed!")

if __name__ == "__main__":
    main()
