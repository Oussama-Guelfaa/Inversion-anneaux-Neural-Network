#!/usr/bin/env python3
"""
Test d'overfitting pour validation du modèle de prédiction du gap

Ce script implémente un test d'overfitting intentionnel pour vérifier que le modèle
peut parfaitement apprendre la relation entre profils d'intensité et gap dans un cas idéal.

Auteur: Oussama GUELFAA
Date: 10 - 01 - 2025
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Configuration pour reproductibilité
torch.manual_seed(42)
np.random.seed(42)

class IntensityDataset(Dataset):
    """
    Dataset PyTorch pour les profils d'intensité.
    
    Args:
        intensity_profiles (np.ndarray): Profils d'intensité (n_samples, n_features)
        gap_values (np.ndarray): Valeurs de gap correspondantes (n_samples,)
    """
    
    def __init__(self, intensity_profiles, gap_values):
        self.intensity_profiles = torch.FloatTensor(intensity_profiles)
        self.gap_values = torch.FloatTensor(gap_values)
    
    def __len__(self):
        return len(self.intensity_profiles)
    
    def __getitem__(self, idx):
        return self.intensity_profiles[idx], self.gap_values[idx]

class SimpleGapPredictor(nn.Module):
    """
    Modèle simple pour prédiction du gap - conçu pour l'overfitting.
    
    Architecture volontairement simple sans régularisation pour favoriser l'overfitting.
    """
    
    def __init__(self, input_size=1000):
        super(SimpleGapPredictor, self).__init__()
        
        # Architecture simple: 4 couches denses
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)  # Une seule sortie: gap
        
        # Pas de dropout ni de régularisation pour favoriser l'overfitting
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Sortie linéaire
        return x

def load_dataset():
    """
    Charge le dataset complet depuis dataset_small_particle.
    
    Returns:
        tuple: (X, y) où X sont les profils d'intensité et y les valeurs de gap
    """
    print("=== CHARGEMENT DU DATASET ===")
    
    dataset_dir = "../../data_generation/dataset_small_particle"
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Le dossier {dataset_dir} n'existe pas")
    
    # Lister tous les fichiers .mat
    mat_files = [f for f in os.listdir(dataset_dir) if f.endswith('.mat') and f.startswith('gap_')]
    mat_files.sort()
    
    print(f"Nombre de fichiers trouvés: {len(mat_files)}")
    
    X = []  # Profils d'intensité
    y = []  # Valeurs de gap
    
    for i, filename in enumerate(mat_files):
        mat_path = os.path.join(dataset_dir, filename)
        
        try:
            data = sio.loadmat(mat_path)
            
            # Extraire le profil d'intensité (ratio)
            ratio = data['ratio'].flatten()  # (1000,)
            
            # Extraire la valeur du gap
            gap_value = float(data['gap'][0, 0])
            
            X.append(ratio)
            y.append(gap_value)
            
            if (i + 1) % 50 == 0:
                print(f"  Chargé {i + 1}/{len(mat_files)} fichiers...")
                
        except Exception as e:
            print(f"Erreur avec {filename}: {e}")
    
    X = np.array(X)  # (400, 1000)
    y = np.array(y)  # (400,)
    
    print(f"\nDonnées chargées:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Gap range: {y.min():.4f} - {y.max():.4f} µm")
    print(f"  Intensité range: {X.min():.4f} - {X.max():.4f}")
    
    return X, y

def prepare_data_for_overfitting(X, y):
    """
    Prépare les données pour le test d'overfitting.
    
    IMPORTANT: Utilise les MÊMES données pour train et validation pour forcer l'overfitting.
    
    Args:
        X (np.ndarray): Profils d'intensité
        y (np.ndarray): Valeurs de gap
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, scaler)
    """
    print("\n=== PRÉPARATION POUR OVERFITTING ===")
    
    # Normalisation des profils d'intensité
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # IMPORTANT: Pour le test d'overfitting, on utilise les MÊMES données
    # pour l'entraînement et la validation
    X_train = X_scaled.copy()
    y_train = y.copy()
    X_val = X_scaled.copy()  # MÊMES données !
    y_val = y.copy()        # MÊMES données !
    
    print(f"Données d'entraînement: X{X_train.shape}, y{y_train.shape}")
    print(f"Données de validation: X{X_val.shape}, y{y_val.shape}")
    print("⚠️  ATTENTION: Mêmes données utilisées pour train et validation (overfitting intentionnel)")
    
    return X_train, y_train, X_val, y_val, scaler

def train_model(X_train, y_train, X_val, y_val):
    """
    Entraîne le modèle avec paramètres optimisés pour l'overfitting.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation (identiques pour overfitting)
        
    Returns:
        tuple: (model, train_losses, val_losses)
    """
    print("\n=== ENTRAÎNEMENT DU MODÈLE ===")
    
    # Paramètres d'entraînement pour favoriser l'overfitting
    batch_size = 8          # Petit batch size
    learning_rate = 0.0001  # Learning rate faible
    num_epochs = 200        # Nombreuses époques
    
    # Créer les datasets et dataloaders
    train_dataset = IntensityDataset(X_train, y_train)
    val_dataset = IntensityDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialiser le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleGapPredictor(input_size=X_train.shape[1]).to(device)
    
    print(f"Modèle créé sur: {device}")
    print(f"Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer et loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Listes pour stocker les losses
    train_losses = []
    val_losses = []
    
    print(f"\nDébut de l'entraînement:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    for epoch in range(num_epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Phase de validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Affichage périodique
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    print(f"\nEntraînement terminé!")
    print(f"Loss finale - Train: {train_losses[-1]:.6f}, Val: {val_losses[-1]:.6f}")
    
    return model, train_losses, val_losses

def evaluate_overfitting(model, X_val, y_val, scaler):
    """
    Évalue les performances du modèle pour vérifier l'overfitting parfait.
    
    Args:
        model: Modèle entraîné
        X_val, y_val: Données de validation
        scaler: Scaler utilisé pour la normalisation
        
    Returns:
        dict: Métriques de performance
    """
    print("\n=== ÉVALUATION DE L'OVERFITTING ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Prédictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_val).to(device)
        y_pred = model(X_tensor).squeeze().cpu().numpy()
    
    # Calcul des métriques
    r2 = r2_score(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }
    
    print(f"Métriques de performance:")
    print(f"  R² Score: {r2:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    
    # Vérification de l'overfitting parfait
    print(f"\n=== VÉRIFICATION OVERFITTING PARFAIT ===")
    if r2 > 0.99:
        print("✅ EXCELLENT: R² > 0.99 - Overfitting parfait atteint!")
    elif r2 > 0.95:
        print("✅ BON: R² > 0.95 - Overfitting très satisfaisant")
    elif r2 > 0.90:
        print("⚠️  MOYEN: R² > 0.90 - Overfitting partiel")
    else:
        print("❌ PROBLÈME: R² < 0.90 - Overfitting insuffisant")
    
    if mse < 1e-4:
        print("✅ EXCELLENT: MSE très faible - Prédictions quasi-parfaites")
    elif mse < 1e-3:
        print("✅ BON: MSE faible - Bonnes prédictions")
    else:
        print("⚠️  À améliorer: MSE élevée")
    
    return metrics, y_pred

def plot_training_curves(train_losses, val_losses):
    """
    Trace les courbes de loss pendant l'entraînement.

    Args:
        train_losses (list): Losses d'entraînement
        val_losses (list): Losses de validation
    """
    plt.figure(figsize=(12, 5))

    # Courbes de loss
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Évolution de la Loss pendant l\'Entraînement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Échelle logarithmique pour mieux voir la décroissance

    # Zoom sur les dernières époques
    plt.subplot(1, 2, 2)
    last_50 = max(1, len(train_losses) - 50)
    plt.plot(epochs[last_50:], train_losses[last_50:], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs[last_50:], val_losses[last_50:], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss - Dernières 50 Époques')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Courbes d'entraînement sauvegardées: ../plots/training_curves.png")

def plot_predictions_vs_actual(y_true, y_pred):
    """
    Trace les prédictions vs valeurs réelles.

    Args:
        y_true (np.ndarray): Valeurs réelles
        y_pred (np.ndarray): Prédictions du modèle
    """
    plt.figure(figsize=(15, 5))

    # Scatter plot prédictions vs réelles
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, s=20)

    # Ligne parfaite y=x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')

    plt.xlabel('Gap réel (µm)')
    plt.ylabel('Gap prédit (µm)')
    plt.title('Prédictions vs Valeurs Réelles')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Histogramme des erreurs
    plt.subplot(1, 3, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Erreur de prédiction (µm)')
    plt.ylabel('Fréquence')
    plt.title('Distribution des Erreurs')
    plt.axvline(0, color='red', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)

    # Erreurs en fonction de la valeur réelle
    plt.subplot(1, 3, 3)
    plt.scatter(y_true, errors, alpha=0.6, s=20)
    plt.xlabel('Gap réel (µm)')
    plt.ylabel('Erreur de prédiction (µm)')
    plt.title('Erreurs vs Valeurs Réelles')
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Analyse des prédictions sauvegardée: ../plots/predictions_analysis.png")

def save_results(metrics, y_true, y_pred, train_losses, val_losses):
    """
    Sauvegarde tous les résultats du test d'overfitting.

    Args:
        metrics (dict): Métriques de performance
        y_true, y_pred: Valeurs réelles et prédites
        train_losses, val_losses: Historique des losses
    """
    print("\n=== SAUVEGARDE DES RÉSULTATS ===")

    # 1. Sauvegarder les métriques
    results_summary = {
        'test_type': 'overfitting_validation',
        'dataset': 'dataset_small_particle',
        'n_samples': len(y_true),
        'gap_range': f"{y_true.min():.4f} - {y_true.max():.4f} µm",
        'model_architecture': 'SimpleGapPredictor (512-256-128-1)',
        'training_epochs': len(train_losses),
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        **metrics
    }

    # Sauvegarder en JSON
    import json
    with open('../results/overfitting_test_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)

    # 2. Sauvegarder les prédictions détaillées
    predictions_df = pd.DataFrame({
        'gap_true': y_true,
        'gap_predicted': y_pred,
        'error': y_pred - y_true,
        'absolute_error': np.abs(y_pred - y_true),
        'relative_error_percent': 100 * np.abs(y_pred - y_true) / y_true
    })
    predictions_df.to_csv('../results/detailed_predictions.csv', index=False)

    # 3. Sauvegarder l'historique d'entraînement
    training_history = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    training_history.to_csv('../results/training_history.csv', index=False)

    print(f"Résultats sauvegardés:")
    print(f"  - Résumé: ../results/overfitting_test_summary.json")
    print(f"  - Prédictions: ../results/detailed_predictions.csv")
    print(f"  - Historique: ../results/training_history.csv")

if __name__ == "__main__":
    print("=== TEST D'OVERFITTING POUR VALIDATION DU MODÈLE ===")
    print("Objectif: Vérifier que le modèle peut parfaitement apprendre")
    print("la relation profil d'intensité → gap dans un cas idéal\n")

    # Créer les dossiers de sortie
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../plots", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    try:
        # 1. Charger les données
        X, y = load_dataset()

        # 2. Préparer pour overfitting
        X_train, y_train, X_val, y_val, scaler = prepare_data_for_overfitting(X, y)

        # 3. Entraîner le modèle
        model, train_losses, val_losses = train_model(X_train, y_train, X_val, y_val)

        # 4. Évaluer l'overfitting
        metrics, y_pred = evaluate_overfitting(model, X_val, y_val, scaler)

        # 5. Visualiser les résultats
        plot_training_curves(train_losses, val_losses)
        plot_predictions_vs_actual(y_val, y_pred)

        # 6. Sauvegarder tous les résultats
        save_results(metrics, y_val, y_pred, train_losses, val_losses)

        # 7. Sauvegarder le modèle
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'metrics': metrics,
            'architecture': 'SimpleGapPredictor'
        }, '../models/overfitting_test_model.pth')

        print(f"\n=== TEST D'OVERFITTING TERMINÉ ===")
        print(f"✅ Modèle sauvegardé: ../models/overfitting_test_model.pth")
        print(f"📊 Graphiques dans: ../plots/")
        print(f"📋 Résultats dans: ../results/")

        # Résumé final
        print(f"\n=== RÉSUMÉ FINAL ===")
        print(f"R² Score: {metrics['r2']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f} µm")

        if metrics['r2'] > 0.99:
            print("🎉 SUCCÈS: Overfitting parfait atteint! Le modèle peut apprendre la relation.")
        else:
            print("⚠️  Le modèle nécessite des ajustements pour atteindre l'overfitting parfait.")

    except Exception as e:
        print(f"Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()
