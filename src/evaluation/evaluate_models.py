#!/usr/bin/env python3
"""
Model Evaluation and Testing
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Ce script évalue les performances du modèle entraîné sur les nouvelles données
et génère des visualisations détaillées des résultats.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os
from train_new_dataset import RingProfileRegressor, load_training_data, prepare_data_for_training, create_data_loaders

def load_trained_model(model_path="models/ring_regressor_new.pth", scalers_path="models/scalers_new.npz"):
    """
    Charge le modèle entraîné et les scalers.
    
    Args:
        model_path (str): Chemin vers le modèle sauvegardé
        scalers_path (str): Chemin vers les scalers
    
    Returns:
        tuple: (model, scaler_X, scaler_y) - Modèle et scalers chargés
    """
    print(f"Chargement du modèle depuis {model_path}")
    
    # Charger le modèle
    model = RingProfileRegressor(input_dim=1000, hidden_dims=[512, 256, 128], output_dim=2)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Charger les scalers
    scalers_data = np.load(scalers_path)
    
    # Reconstruire les scalers (approximation)
    class MockScaler:
        def __init__(self, mean, scale):
            self.mean_ = mean
            self.scale_ = scale
        
        def transform(self, X):
            return (X - self.mean_) / self.scale_
        
        def inverse_transform(self, X):
            return X * self.scale_ + self.mean_
    
    scaler_X = MockScaler(scalers_data['scaler_X_mean'], scalers_data['scaler_X_scale'])
    scaler_y = MockScaler(scalers_data['scaler_y_mean'], scalers_data['scaler_y_scale'])
    
    print(f"Modèle et scalers chargés avec succès")
    return model, scaler_X, scaler_y

def evaluate_model(model, test_loader, scaler_y, device='cpu'):
    """
    Évalue le modèle sur le jeu de test.
    
    Args:
        model: Modèle PyTorch
        test_loader: DataLoader de test
        scaler_y: Scaler pour dénormaliser les prédictions
        device (str): Device de calcul
    
    Returns:
        tuple: (y_true, y_pred) - Vraies valeurs et prédictions
    """
    print(f"\nÉvaluation du modèle sur le jeu de test...")
    
    model = model.to(device)
    model.eval()
    
    y_true_scaled = []
    y_pred_scaled = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Prédictions
            outputs = model(batch_X)
            
            y_true_scaled.append(batch_y.cpu().numpy())
            y_pred_scaled.append(outputs.cpu().numpy())
    
    # Concaténer tous les résultats
    y_true_scaled = np.vstack(y_true_scaled)
    y_pred_scaled = np.vstack(y_pred_scaled)
    
    # Dénormaliser
    y_true = scaler_y.inverse_transform(y_true_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    print(f"Évaluation terminée sur {len(y_true)} échantillons")
    return y_true, y_pred

def calculate_metrics(y_true, y_pred, parameter_names=['L_ecran', 'gap']):
    """
    Calcule les métriques de performance.
    
    Args:
        y_true (np.ndarray): Vraies valeurs
        y_pred (np.ndarray): Prédictions
        parameter_names (list): Noms des paramètres
    
    Returns:
        dict: Métriques calculées
    """
    metrics = {}
    
    for i, param_name in enumerate(parameter_names):
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        metrics[param_name] = {
            'MSE': mean_squared_error(true_vals, pred_vals),
            'RMSE': np.sqrt(mean_squared_error(true_vals, pred_vals)),
            'MAE': mean_absolute_error(true_vals, pred_vals),
            'R2': r2_score(true_vals, pred_vals),
            'MAPE': np.mean(np.abs((true_vals - pred_vals) / true_vals)) * 100
        }
    
    # Métriques globales
    metrics['Global'] = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    
    return metrics

def plot_results(y_true, y_pred, metrics, save_path="plots/evaluation_results.png"):
    """
    Génère des visualisations des résultats.
    
    Args:
        y_true (np.ndarray): Vraies valeurs
        y_pred (np.ndarray): Prédictions
        metrics (dict): Métriques calculées
        save_path (str): Chemin de sauvegarde
    """
    print(f"\nGénération des visualisations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    parameter_names = ['L_ecran', 'gap']
    
    for i, param_name in enumerate(parameter_names):
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]
        
        # 1. Scatter plot prédictions vs vraies valeurs
        axes[i, 0].scatter(true_vals, pred_vals, alpha=0.6, s=20)
        
        # Ligne de référence y=x
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        axes[i, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        axes[i, 0].set_xlabel(f'{param_name} (vraie valeur)')
        axes[i, 0].set_ylabel(f'{param_name} (prédiction)')
        axes[i, 0].set_title(f'{param_name} - R² = {metrics[param_name]["R2"]:.4f}')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 2. Histogramme des erreurs
        errors = pred_vals - true_vals
        axes[i, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[i, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[i, 1].set_xlabel(f'Erreur {param_name}')
        axes[i, 1].set_ylabel('Fréquence')
        axes[i, 1].set_title(f'Distribution des erreurs - MAE = {metrics[param_name]["MAE"]:.4f}')
        axes[i, 1].grid(True, alpha=0.3)
        
        # 3. Erreurs relatives
        relative_errors = np.abs(errors / true_vals) * 100
        axes[i, 2].scatter(true_vals, relative_errors, alpha=0.6, s=20)
        axes[i, 2].set_xlabel(f'{param_name} (vraie valeur)')
        axes[i, 2].set_ylabel('Erreur relative (%)')
        axes[i, 2].set_title(f'Erreurs relatives - MAPE = {metrics[param_name]["MAPE"]:.2f}%')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualisations sauvegardées: {save_path}")

def print_metrics_summary(metrics):
    """
    Affiche un résumé des métriques.
    
    Args:
        metrics (dict): Métriques calculées
    """
    print(f"\n{'='*60}")
    print(f"{'RÉSUMÉ DES PERFORMANCES':^60}")
    print(f"{'='*60}")
    
    for param_name in ['L_ecran', 'gap']:
        print(f"\n{param_name.upper()}:")
        print(f"  R² Score:      {metrics[param_name]['R2']:.6f}")
        print(f"  RMSE:          {metrics[param_name]['RMSE']:.6f}")
        print(f"  MAE:           {metrics[param_name]['MAE']:.6f}")
        print(f"  MAPE:          {metrics[param_name]['MAPE']:.2f}%")
    
    print(f"\nGLOBAL:")
    print(f"  R² Score:      {metrics['Global']['R2']:.6f}")
    print(f"  RMSE:          {metrics['Global']['RMSE']:.6f}")
    print(f"  MAE:           {metrics['Global']['MAE']:.6f}")
    
    print(f"\n{'='*60}")

def test_on_sample_profiles(model, scaler_X, scaler_y, X_test, y_test, n_samples=5):
    """
    Teste le modèle sur quelques profils d'exemple.
    
    Args:
        model: Modèle entraîné
        scaler_X, scaler_y: Scalers
        X_test, y_test: Données de test
        n_samples (int): Nombre d'échantillons à tester
    """
    print(f"\nTest sur {n_samples} profils d'exemple:")
    
    # Sélectionner des échantillons aléatoires
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Préparer l'entrée
            profile = X_test[idx:idx+1]  # Garder la dimension batch
            profile_scaled = scaler_X.transform(profile)
            profile_tensor = torch.FloatTensor(profile_scaled)
            
            # Prédiction
            pred_scaled = model(profile_tensor).numpy()
            pred = scaler_y.inverse_transform(pred_scaled)[0]
            
            # Vraie valeur
            true = y_test[idx]
            
            print(f"\nÉchantillon {i+1}:")
            print(f"  L_ecran - Vrai: {true[0]:.4f}, Prédit: {pred[0]:.4f}, Erreur: {abs(true[0]-pred[0]):.4f}")
            print(f"  gap     - Vrai: {true[1]:.4f}, Prédit: {pred[1]:.4f}, Erreur: {abs(true[1]-pred[1]):.4f}")

def main():
    """Fonction principale d'évaluation."""
    
    print("=== ÉVALUATION DU MODÈLE ENTRAÎNÉ ===")
    
    # 1. Charger les données
    X, y, metadata = load_training_data()
    
    # 2. Préparer les données (même division que l'entraînement)
    (X_train, X_val, X_test, y_train, y_val, y_test, 
     scaler_X, scaler_y) = prepare_data_for_training(X, y, random_state=42)
    
    # 3. Créer le DataLoader de test
    _, _, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32
    )
    
    # 4. Charger le modèle entraîné
    model, scaler_X_loaded, scaler_y_loaded = load_trained_model()
    
    # 5. Évaluer le modèle
    y_true, y_pred = evaluate_model(model, test_loader, scaler_y_loaded)
    
    # 6. Calculer les métriques
    metrics = calculate_metrics(y_true, y_pred)
    
    # 7. Afficher les résultats
    print_metrics_summary(metrics)
    
    # 8. Générer les visualisations
    plot_results(y_true, y_pred, metrics)
    
    # 9. Test sur des échantillons
    test_on_sample_profiles(model, scaler_X_loaded, scaler_y_loaded, X_test, y_test)
    
    # 10. Sauvegarder les résultats
    results_df = pd.DataFrame({
        'L_ecran_true': y_true[:, 0],
        'L_ecran_pred': y_pred[:, 0],
        'gap_true': y_true[:, 1],
        'gap_pred': y_pred[:, 1]
    })
    results_df.to_csv('plots/predictions_results.csv', index=False)
    
    print(f"\n=== ÉVALUATION TERMINÉE ===")
    print(f"Résultats sauvegardés dans le dossier 'plots/'")

if __name__ == "__main__":
    main()
