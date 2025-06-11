#!/usr/bin/env python3
"""
Test Rounded Labels Impact
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Teste l'impact de l'arrondissement des labels gap à 3 décimales
sur les performances du réseau de neurones.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import scipy.io as sio
import os
import time

# Configuration pour reproductibilité
torch.manual_seed(42)
np.random.seed(42)

class OptimizedRegressor(nn.Module):
    """Réseau de neurones optimisé."""
    
    def __init__(self, input_size=600, hidden_sizes=[512, 256, 128, 64], output_size=2, dropout_rates=[0.2, 0.15, 0.1, 0.05]):
        super(OptimizedRegressor, self).__init__()
        
        # Feature extractor avec normalisation
        layers = []
        prev_size = input_size
        
        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_sizes, dropout_rates)):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Têtes spécialisées pour chaque paramètre
        self.L_ecran_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.gap_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialisation des poids
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialisation optimisée des poids."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        L_ecran = self.L_ecran_head(features)
        gap = self.gap_head(features)
        return torch.cat([L_ecran, gap], dim=1)

def analyze_precision_problem():
    """Analyse le problème de précision des labels."""
    
    print("="*80)
    print("ANALYSE DU PROBLÈME DE PRÉCISION DES LABELS")
    print("="*80)
    
    # Charger les labels originaux
    dataset_dir = "../data_generation/dataset"
    labels_df = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))
    
    print(f"Analyse des labels originaux:")
    print(f"  Nombre d'échantillons: {len(labels_df)}")
    
    # Analyser la précision des gaps
    gap_values = labels_df['gap_um'].values
    
    print(f"\nAnalyse de la précision des gaps:")
    print(f"  Valeurs uniques: {len(np.unique(gap_values))}")
    print(f"  Min: {gap_values.min()}")
    print(f"  Max: {gap_values.max()}")
    
    print(f"\nExemples de précision excessive:")
    for i in range(min(5, len(gap_values))):
        original = gap_values[i]
        rounded_3 = round(original, 3)
        rounded_6 = round(original, 6)
        print(f"  Original: {original:.15f}")
        print(f"  3 décimales: {rounded_3:.3f}")
        print(f"  6 décimales: {rounded_6:.6f}")
        print(f"  Différence 3 déc: {abs(original - rounded_3):.2e}")
        print(f"  Différence 6 déc: {abs(original - rounded_6):.2e}")
        print()
    
    # Créer les versions arrondies
    gap_rounded_3 = np.round(gap_values, 3)
    gap_rounded_6 = np.round(gap_values, 6)
    
    print(f"Impact de l'arrondissement:")
    print(f"  Valeurs uniques originales: {len(np.unique(gap_values))}")
    print(f"  Valeurs uniques 3 décimales: {len(np.unique(gap_rounded_3))}")
    print(f"  Valeurs uniques 6 décimales: {len(np.unique(gap_rounded_6))}")
    
    print(f"\nValeurs uniques après arrondissement à 3 décimales:")
    unique_gaps_3 = np.unique(gap_rounded_3)
    print(f"  {unique_gaps_3}")
    
    return gap_values, gap_rounded_3, gap_rounded_6

def create_rounded_test_data(precision=3):
    """Crée les données de test avec labels arrondis."""
    
    print(f"\n=== CRÉATION DONNÉES TEST ARRONDIES ({precision} décimales) ===")
    
    dataset_dir = "../data_generation/dataset"
    labels_df = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))
    
    X_test = []
    y_test = []
    filenames = []
    
    for idx, row in labels_df.iterrows():
        filename = row['filename']
        gap = round(row['gap_um'], precision)  # Arrondir ici !
        L_ecran = round(row['L_um'], precision)  # Arrondir aussi L_ecran pour cohérence
        
        mat_filename = filename.replace('.png', '.mat')
        mat_path = os.path.join(dataset_dir, mat_filename)
        
        if os.path.exists(mat_path):
            try:
                data = sio.loadmat(mat_path)
                ratio = data['ratio'].flatten()
                ratio_truncated = ratio[:600]  # Utiliser les données tronquées
                
                X_test.append(ratio_truncated)
                y_test.append([L_ecran, gap])
                filenames.append(filename)
                
            except Exception as e:
                print(f"Erreur {mat_filename}: {e}")
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    print(f"Données de test arrondies:")
    print(f"  X shape: {X_test.shape}")
    print(f"  y shape: {y_test.shape}")
    print(f"  Plages arrondies:")
    print(f"    L_ecran: [{y_test[:, 0].min():.3f}, {y_test[:, 0].max():.3f}]")
    print(f"    gap: [{y_test[:, 1].min():.3f}, {y_test[:, 1].max():.3f}]")
    print(f"  Valeurs gap uniques: {sorted(np.unique(y_test[:, 1]))}")
    
    return X_test, y_test, filenames

def train_with_rounded_labels(precision=3):
    """Entraîne le modèle avec des labels arrondis."""
    
    print(f"\n=== ENTRAÎNEMENT AVEC LABELS ARRONDIS ({precision} décimales) ===")
    
    # Charger les données d'entraînement
    df_profiles = pd.read_csv('processed_data/intensity_profiles_truncated_600.csv')
    df_params = pd.read_csv('processed_data/parameters_truncated_600.csv')
    
    X = df_profiles.values.astype(np.float32)
    
    # Arrondir les labels d'entraînement !
    y_original = df_params[['L_ecran', 'gap']].values.astype(np.float32)
    y_rounded = np.round(y_original, precision)
    
    print(f"Labels d'entraînement arrondis:")
    print(f"  L_ecran: {len(np.unique(y_rounded[:, 0]))} valeurs uniques")
    print(f"  gap: {len(np.unique(y_rounded[:, 1]))} valeurs uniques")
    print(f"  Plages gap: [{y_rounded[:, 1].min():.3f}, {y_rounded[:, 1].max():.3f}]")
    
    # Division train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y_rounded, test_size=0.2, random_state=42)
    
    # Normalisation
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    
    # Créer et entraîner le modèle
    model = OptimizedRegressor(input_size=600)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Convertir en tenseurs
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Datasets et DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Optimiseur et loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, verbose=False)
    criterion = nn.MSELoss()
    
    # Entraînement rapide
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    epochs = 100
    
    print(f"Entraînement rapide ({epochs} epochs max)...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
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
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Scheduler et early stopping
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'models/rounded_{precision}dec_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:2d}: Train = {train_loss:.4f}, Val = {val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping à l'epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"Entraînement terminé en {training_time:.1f}s")
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load(f'models/rounded_{precision}dec_model.pth'))
    
    return model, scaler_X

def evaluate_rounded_model(model, X_test, y_test, scaler_X, precision):
    """Évalue le modèle avec labels arrondis."""
    
    print(f"\n=== ÉVALUATION MODÈLE LABELS ARRONDIS ({precision} décimales) ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Normaliser et prédire
    X_test_scaled = scaler_X.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
    
    # Arrondir aussi les prédictions pour cohérence
    y_pred_rounded = np.round(y_pred, precision)
    
    # Calculer les métriques
    r2_global = r2_score(y_test, y_pred_rounded)
    r2_L = r2_score(y_test[:, 0], y_pred_rounded[:, 0])
    r2_gap = r2_score(y_test[:, 1], y_pred_rounded[:, 1])
    
    rmse_L = np.sqrt(mean_squared_error(y_test[:, 0], y_pred_rounded[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_test[:, 1], y_pred_rounded[:, 1]))
    
    mae_L = mean_absolute_error(y_test[:, 0], y_pred_rounded[:, 0])
    mae_gap = mean_absolute_error(y_test[:, 1], y_pred_rounded[:, 1])
    
    # MAPE avec protection contre division par zéro
    mape_L = np.mean(np.abs((y_test[:, 0] - y_pred_rounded[:, 0]) / np.maximum(y_test[:, 0], 1e-8))) * 100
    mape_gap = np.mean(np.abs((y_test[:, 1] - y_pred_rounded[:, 1]) / np.maximum(y_test[:, 1], 1e-8))) * 100
    
    print(f"RÉSULTATS AVEC LABELS ARRONDIS ({precision} décimales):")
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
    
    return {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'mape_L': mape_L, 'mape_gap': mape_gap,
        'success': success
    }

def main():
    """Fonction principale."""
    
    print("="*80)
    print("TEST DE L'IMPACT DE L'ARRONDISSEMENT DES LABELS")
    print("="*80)
    
    # 1. Analyser le problème de précision
    gap_original, gap_3dec, gap_6dec = analyze_precision_problem()
    
    # 2. Tester différentes précisions
    os.makedirs('models', exist_ok=True)
    results = {}
    
    for precision in [3, 6]:
        print(f"\n{'='*60}")
        print(f"TEST AVEC {precision} DÉCIMALES")
        print(f"{'='*60}")
        
        # Créer les données de test arrondies
        X_test, y_test, filenames = create_rounded_test_data(precision)
        
        # Entraîner le modèle
        model, scaler_X = train_with_rounded_labels(precision)
        
        # Évaluer
        metrics = evaluate_rounded_model(model, X_test, y_test, scaler_X, precision)
        results[precision] = metrics
    
    # 3. Comparaison finale
    print(f"\n{'='*80}")
    print(f"COMPARAISON DES RÉSULTATS")
    print(f"{'='*80}")
    
    print(f"| Métrique | 3 décimales | 6 décimales | Amélioration |")
    print(f"|----------|-------------|-------------|--------------|")
    
    for metric in ['r2_global', 'r2_L', 'r2_gap']:
        val_3 = results[3][metric]
        val_6 = results[6][metric]
        diff = val_3 - val_6
        print(f"| {metric:8s} | {val_3:11.6f} | {val_6:11.6f} | {diff:+11.6f} |")
    
    print(f"\n🎯 CONCLUSION:")
    if results[3]['r2_global'] > results[6]['r2_global']:
        print(f"✅ L'arrondissement à 3 décimales AMÉLIORE les performances !")
        print(f"   R² global: {results[6]['r2_global']:.6f} → {results[3]['r2_global']:.6f}")
    else:
        print(f"❌ L'arrondissement à 3 décimales n'améliore pas significativement")
    
    if results[3]['success']:
        print(f"🎉 OBJECTIF ATTEINT avec 3 décimales: R² > 0.8 !")
    else:
        print(f"⚠️  Objectif non atteint, mais amélioration possible")

if __name__ == "__main__":
    main()
