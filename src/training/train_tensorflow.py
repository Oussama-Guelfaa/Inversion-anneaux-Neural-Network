#!/usr/bin/env python3
"""
TensorFlow/Keras Neural Network Implementation
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Implémentation d'un réseau de neurones avec TensorFlow/Keras pour prédire
les paramètres gap et L_ecran à partir de profils d'intensité.
Basé sur l'architecture suggérée avec optimisations.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import time

# Configuration pour reproductibilité
tf.random.set_seed(42)
np.random.seed(42)

def load_training_data():
    """Charge les données d'entraînement depuis les fichiers CSV."""
    
    print("=== CHARGEMENT DES DONNÉES D'ENTRAÎNEMENT ===")
    
    # Charger les fichiers CSV
    df_params = pd.read_csv('processed_data/parameters.csv')
    df_profiles = pd.read_csv('processed_data/intensity_profiles_full.csv')
    
    print(f"Paramètres shape: {df_params.shape}")
    print(f"Profils shape: {df_profiles.shape}")
    
    # Aligner les dimensions (prendre le minimum)
    min_samples = min(len(df_params), len(df_profiles))
    df_params = df_params.iloc[:min_samples].reset_index(drop=True)
    df_profiles = df_profiles.iloc[:min_samples].reset_index(drop=True)
    
    print(f"Données alignées: {min_samples} échantillons")
    
    # Préparer X (profils) et Y (paramètres)
    X = df_profiles.values.astype('float32')  # shape = (N, 1000)
    Y = df_params[['L_ecran', 'gap']].values.astype('float32')  # shape = (N, 2)
    
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Plages des paramètres:")
    print(f"  L_ecran: [{Y[:, 0].min():.3f}, {Y[:, 0].max():.3f}]")
    print(f"  gap: [{Y[:, 1].min():.6f}, {Y[:, 1].max():.6f}]")
    
    return X, Y

def load_test_data():
    """Charge les données de test expérimentales."""
    
    print("\n=== CHARGEMENT DES DONNÉES DE TEST EXPÉRIMENTALES ===")
    
    dataset_dir = "../data_generation/dataset"
    labels_df = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))
    
    X_test = []
    y_test = []
    filenames = []
    
    for idx, row in labels_df.iterrows():
        filename = row['filename']
        gap = row['gap_um']
        L_ecran = row['L_um']
        
        mat_filename = filename.replace('.png', '.mat')
        mat_path = os.path.join(dataset_dir, mat_filename)
        
        if os.path.exists(mat_path):
            try:
                data = sio.loadmat(mat_path)
                ratio = data['ratio'].flatten()
                
                X_test.append(ratio)
                y_test.append([L_ecran, gap])  # Même ordre que l'entraînement
                filenames.append(filename)
                
            except Exception as e:
                print(f"Erreur {mat_filename}: {e}")
    
    X_test = np.array(X_test, dtype='float32')
    y_test = np.array(y_test, dtype='float32')
    
    print(f"Données de test chargées: X{X_test.shape}, y{y_test.shape}")
    return X_test, y_test, filenames

def create_tensorflow_model(input_dim=1000, output_dim=2):
    """Crée le modèle TensorFlow/Keras optimisé."""
    
    print("\n=== CRÉATION DU MODÈLE TENSORFLOW/KERAS ===")
    
    model = Sequential([
        # Couche d'entrée avec normalisation
        Dense(512, activation='relu', input_dim=input_dim, name='dense_1'),
        BatchNormalization(name='batch_norm_1'),
        Dropout(0.2, name='dropout_1'),
        
        # Couches cachées avec normalisation progressive
        Dense(256, activation='relu', name='dense_2'),
        BatchNormalization(name='batch_norm_2'),
        Dropout(0.2, name='dropout_2'),
        
        Dense(128, activation='relu', name='dense_3'),
        BatchNormalization(name='batch_norm_3'),
        Dropout(0.15, name='dropout_3'),
        
        Dense(64, activation='relu', name='dense_4'),
        BatchNormalization(name='batch_norm_4'),
        Dropout(0.1, name='dropout_4'),
        
        # Couche de sortie pour régression
        Dense(output_dim, activation='linear', name='output')
    ])
    
    # Compilation avec optimiseur Adam optimisé
    model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss='mse',  # Mean Squared Error pour régression
        metrics=['mae']  # Mean Absolute Error pour suivi
    )
    
    print("Architecture du modèle:")
    model.summary()
    
    return model

def train_tensorflow_model(model, X_train, X_val, y_train, y_val, epochs=500):
    """Entraîne le modèle TensorFlow avec callbacks optimisés."""
    
    print("\n=== ENTRAÎNEMENT DU MODÈLE TENSORFLOW ===")
    
    # Callbacks pour optimiser l'entraînement
    callbacks = [
        # Early stopping avec patience
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Réduction du learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=15,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Sauvegarde du meilleur modèle (format Keras natif)
        ModelCheckpoint(
            'models/tensorflow_best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print(f"Début de l'entraînement...")
    print(f"  Epochs max: {epochs}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    start_time = time.time()
    
    # Entraînement
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nEntraînement terminé en {training_time/60:.1f} minutes")
    
    return history

def evaluate_tensorflow_model(model, X_test, y_test, scaler_X, scaler_y, filenames):
    """Évalue le modèle TensorFlow sur les données de test."""
    
    print("\n=== ÉVALUATION SUR DONNÉES EXPÉRIMENTALES ===")
    
    # Normaliser les données de test
    X_test_scaled = scaler_X.transform(X_test)
    
    # Prédictions
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    
    # Dénormaliser si nécessaire
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    else:
        y_pred = y_pred_scaled
    
    # Calculer les métriques
    r2_global = r2_score(y_test, y_pred)
    r2_L = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_gap = r2_score(y_test[:, 1], y_pred[:, 1])
    
    rmse_L = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
    
    mae_L = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_gap = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    
    # Erreurs relatives
    mape_L = np.mean(np.abs((y_test[:, 0] - y_pred[:, 0]) / y_test[:, 0])) * 100
    mape_gap = np.mean(np.abs((y_test[:, 1] - y_pred[:, 1]) / y_test[:, 1])) * 100
    
    print(f"Métriques de performance TensorFlow:")
    print(f"  R² global: {r2_global:.6f}")
    print(f"  R² L_ecran: {r2_L:.6f}")
    print(f"  R² gap: {r2_gap:.6f}")
    print(f"  RMSE L_ecran: {rmse_L:.6f} µm")
    print(f"  RMSE gap: {rmse_gap:.6f} µm")
    print(f"  MAE L_ecran: {mae_L:.6f} µm")
    print(f"  MAE gap: {mae_gap:.6f} µm")
    print(f"  MAPE L_ecran: {mape_L:.2f}%")
    print(f"  MAPE gap: {mape_gap:.2f}%")
    
    success = r2_global > 0.8
    print(f"  Objectif R² > 0.8: {'✓ ATTEINT' if success else '✗ NON ATTEINT'}")
    
    return y_pred, {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'mape_L': mape_L, 'mape_gap': mape_gap,
        'success': success
    }

def plot_training_history(history):
    """Visualise l'historique d'entraînement."""
    
    print("\n=== GÉNÉRATION DES GRAPHIQUES D'ENTRAÎNEMENT ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Loss Evolution')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # MAE curves
    ax2.plot(history.history['mae'], label='Train MAE', linewidth=2)
    ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_title('MAE Evolution')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate (si disponible)
    if 'lr' in history.history:
        ax3.plot(history.history['lr'], linewidth=2, color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Learning Rate\nNot Recorded', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Learning Rate')
    
    # Résumé des performances
    ax4.axis('off')
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    epochs_trained = len(history.history['loss'])
    
    summary_text = f"""
RÉSUMÉ DE L'ENTRAÎNEMENT

Epochs entraînés: {epochs_trained}

Loss finale:
  Train: {final_train_loss:.6f}
  Validation: {final_val_loss:.6f}

MAE finale:
  Train: {final_train_mae:.6f}
  Validation: {final_val_mae:.6f}

Modèle sauvegardé:
  tensorflow_best_model.h5
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/tensorflow_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Graphiques sauvegardés: plots/tensorflow_training_history.png")

def main():
    """Fonction principale d'entraînement TensorFlow."""
    
    print("="*80)
    print("ENTRAÎNEMENT RÉSEAU DE NEURONES TENSORFLOW/KERAS")
    print("="*80)
    
    # Vérifier TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
    
    # 1. Charger les données d'entraînement
    X, y = load_training_data()
    
    # 2. Charger les données de test
    X_test, y_test, filenames = load_test_data()
    
    # 3. Division train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDivision des données:")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # 4. Normalisation des features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # Option: normaliser aussi les targets (décommenter si nécessaire)
    scaler_y = None
    # scaler_y = StandardScaler()
    # y_train_scaled = scaler_y.fit_transform(y_train)
    # y_val_scaled = scaler_y.transform(y_val)
    
    # 5. Créer le modèle
    model = create_tensorflow_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
    
    # 6. Entraîner le modèle
    os.makedirs('models', exist_ok=True)
    history = train_tensorflow_model(model, X_train_scaled, X_val_scaled, y_train, y_val)
    
    # 7. Visualiser l'entraînement
    plot_training_history(history)
    
    # 8. Charger le meilleur modèle et évaluer
    best_model = tf.keras.models.load_model('models/tensorflow_best_model.keras')
    y_pred, metrics = evaluate_tensorflow_model(best_model, X_test, y_test, scaler_X, scaler_y, filenames)
    
    # 9. Sauvegarder les scalers
    import joblib
    joblib.dump(scaler_X, 'models/tensorflow_scaler_X.pkl')
    if scaler_y is not None:
        joblib.dump(scaler_y, 'models/tensorflow_scaler_y.pkl')
    
    print(f"\n{'='*80}")
    print(f"RÉSULTATS FINAUX TENSORFLOW")
    print(f"{'='*80}")
    print(f"R² global: {metrics['r2_global']:.6f}")
    print(f"R² L_ecran: {metrics['r2_L']:.6f}")
    print(f"R² gap: {metrics['r2_gap']:.6f}")
    print(f"Objectif atteint: {'OUI' if metrics['success'] else 'NON'}")
    
    print(f"\nFichiers générés:")
    print(f"  • models/tensorflow_best_model.keras")
    print(f"  • models/tensorflow_scaler_X.pkl")
    print(f"  • plots/tensorflow_training_history.png")

if __name__ == "__main__":
    main()
