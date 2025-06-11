#!/usr/bin/env python3
"""
Neural Network 06-06-25 - Problem-Solving Edition
Author: Oussama GUELFAA
Date: 06 - 06 - 2025

Réseau de neurones qui résout systématiquement les 5 problèmes identifiés :
1. 🔢 Précision excessive des labels → Arrondissement à 3 décimales
2. ⚖️ Échelles déséquilibrées → Normalisation séparée par paramètre
3. 📊 Distribution déséquilibrée → Focus sur plage expérimentale
4. 🎛️ Loss function inadaptée → Loss pondérée pour gap
5. 🔍 Signal gap faible → Architecture spécialisée pour gap
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
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import time

# Configuration pour reproductibilité
torch.manual_seed(42)
np.random.seed(42)

class WeightedMSELoss(nn.Module):
    """Loss pondérée pour donner plus d'importance au gap."""
    
    def __init__(self, weights=[1.0, 10.0]):  # [L_ecran, gap]
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        weighted_mse = mse * self.weights.to(pred.device)
        return weighted_mse.mean()

class AdvancedRegressor(nn.Module):
    """Architecture avancée avec têtes spécialisées et attention pour gap."""
    
    def __init__(self, input_size=600):
        super(AdvancedRegressor, self).__init__()
        
        # Feature extractor commun
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Tête L_ecran (signal fort)
        self.L_ecran_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Tête gap spécialisée (signal faible) - Plus de neurones
        self.gap_head = nn.Sequential(
            nn.Linear(128, 128),  # Plus de capacité
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.01),     # Moins de dropout
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.01),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Mécanisme d'attention pour gap
        self.gap_attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Sigmoid()
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
        # Extraction de features communes
        features = self.feature_extractor(x)
        
        # Prédiction L_ecran (directe)
        L_ecran = self.L_ecran_head(features)
        
        # Prédiction gap avec attention
        attention_weights = self.gap_attention(features)
        attended_features = features * attention_weights
        gap = self.gap_head(attended_features)
        
        return torch.cat([L_ecran, gap], dim=1)

def solve_problem_1_precision(y_data, precision=3):
    """PROBLÈME 1: Résoudre la précision excessive des labels."""
    
    print(f"🔢 RÉSOLUTION PROBLÈME 1: Précision excessive")
    print(f"   Arrondissement à {precision} décimales")
    
    y_original = y_data.copy()
    y_rounded = np.round(y_data, precision)
    
    # Analyser l'impact
    diff_L = np.abs(y_original[:, 0] - y_rounded[:, 0]).max()
    diff_gap = np.abs(y_original[:, 1] - y_rounded[:, 1]).max()
    
    print(f"   Différence max L_ecran: {diff_L:.2e}")
    print(f"   Différence max gap: {diff_gap:.2e}")
    print(f"   ✅ Labels arrondis à {precision} décimales")
    
    return y_rounded

def solve_problem_2_scaling(X_data, y_data):
    """PROBLÈME 2: Résoudre les échelles déséquilibrées."""
    
    print(f"⚖️ RÉSOLUTION PROBLÈME 2: Échelles déséquilibrées")
    
    # Normalisation des features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_data)
    
    # Normalisation séparée pour chaque paramètre
    scaler_L = StandardScaler()
    scaler_gap = StandardScaler()
    
    y_L_scaled = scaler_L.fit_transform(y_data[:, 0:1])
    y_gap_scaled = scaler_gap.fit_transform(y_data[:, 1:2])
    
    y_scaled = np.hstack([y_L_scaled, y_gap_scaled])
    
    print(f"   X normalisé: mean={X_scaled.mean():.6f}, std={X_scaled.std():.6f}")
    print(f"   L_ecran normalisé: mean={y_L_scaled.mean():.6f}, std={y_L_scaled.std():.6f}")
    print(f"   gap normalisé: mean={y_gap_scaled.mean():.6f}, std={y_gap_scaled.std():.6f}")
    print(f"   ✅ Normalisation séparée appliquée")
    
    return X_scaled, y_scaled, scaler_X, scaler_L, scaler_gap

def solve_problem_3_distribution(X_data, y_data, focus_on_experimental=True):
    """PROBLÈME 3: Résoudre la distribution déséquilibrée."""
    
    print(f"📊 RÉSOLUTION PROBLÈME 3: Distribution déséquilibrée")
    
    if focus_on_experimental:
        # Focus sur la plage expérimentale gap [0.025, 0.517]
        experimental_mask = (y_data[:, 1] >= 0.025) & (y_data[:, 1] <= 0.517)
        
        X_focused = X_data[experimental_mask]
        y_focused = y_data[experimental_mask]
        
        print(f"   Données originales: {len(X_data)} échantillons")
        print(f"   Données focalisées: {len(X_focused)} échantillons")
        print(f"   Plage gap: [{y_focused[:, 1].min():.6f}, {y_focused[:, 1].max():.6f}]")
        print(f"   ✅ Focus sur plage expérimentale")
        
        return X_focused, y_focused
    
    return X_data, y_data

def load_and_preprocess_data():
    """Charge et prétraite les données en résolvant les problèmes 1-3."""
    
    print("="*80)
    print("CHARGEMENT ET PRÉTRAITEMENT AVANCÉ")
    print("="*80)
    
    # Charger les données tronquées
    df_profiles = pd.read_csv('processed_data/intensity_profiles_truncated_600.csv')
    df_params = pd.read_csv('processed_data/parameters_truncated_600.csv')
    
    X_raw = df_profiles.values.astype(np.float32)
    y_raw = df_params[['L_ecran', 'gap']].values.astype(np.float32)
    
    print(f"Données brutes: X{X_raw.shape}, y{y_raw.shape}")
    
    # PROBLÈME 1: Précision excessive
    y_rounded = solve_problem_1_precision(y_raw, precision=3)
    
    # PROBLÈME 3: Distribution déséquilibrée
    X_focused, y_focused = solve_problem_3_distribution(X_raw, y_rounded, focus_on_experimental=True)
    
    # PROBLÈME 2: Échelles déséquilibrées
    X_scaled, y_scaled, scaler_X, scaler_L, scaler_gap = solve_problem_2_scaling(X_focused, y_focused)
    
    return X_scaled, y_scaled, scaler_X, scaler_L, scaler_gap

def load_test_data_advanced(scaler_X, scaler_L, scaler_gap):
    """Charge les données de test avec le même prétraitement."""
    
    print(f"\n=== CHARGEMENT DONNÉES DE TEST AVANCÉ ===")
    
    dataset_dir = "../data_generation/dataset"
    labels_df = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))
    
    X_test = []
    y_test = []
    filenames = []
    
    for idx, row in labels_df.iterrows():
        filename = row['filename']
        gap = round(row['gap_um'], 3)  # PROBLÈME 1: Arrondir
        L_ecran = round(row['L_um'], 3)
        
        mat_filename = filename.replace('.png', '.mat')
        mat_path = os.path.join(dataset_dir, mat_filename)
        
        if os.path.exists(mat_path):
            try:
                data = sio.loadmat(mat_path)
                ratio = data['ratio'].flatten()[:600]  # Tronquer
                
                X_test.append(ratio)
                y_test.append([L_ecran, gap])
                filenames.append(filename)
                
            except Exception as e:
                print(f"Erreur {mat_filename}: {e}")
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    # Appliquer les mêmes transformations
    X_test_scaled = scaler_X.transform(X_test)
    
    # Normaliser les targets de test séparément
    y_test_L_scaled = scaler_L.transform(y_test[:, 0:1])
    y_test_gap_scaled = scaler_gap.transform(y_test[:, 1:2])
    y_test_scaled = np.hstack([y_test_L_scaled, y_test_gap_scaled])
    
    print(f"Données de test prétraitées: X{X_test_scaled.shape}, y{y_test_scaled.shape}")
    
    return X_test_scaled, y_test_scaled, y_test, filenames

def train_advanced_model(X_train, X_val, y_train, y_val, epochs=150):
    """PROBLÈME 4 & 5: Entraîne avec loss pondérée et architecture spécialisée."""
    
    print(f"\n🎛️ RÉSOLUTION PROBLÈME 4: Loss pondérée")
    print(f"🔍 RÉSOLUTION PROBLÈME 5: Architecture spécialisée pour gap")
    
    # Créer le modèle avancé
    model = AdvancedRegressor(input_size=600)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Modèle avancé créé - Device: {device}")
    
    # Convertir en tenseurs
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Datasets et DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # PROBLÈME 4: Loss pondérée (10x plus de poids sur gap)
    criterion = WeightedMSELoss(weights=[1.0, 10.0])
    
    # Optimiseur avec learning rates différentiels
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-3},
        {'params': model.L_ecran_head.parameters(), 'lr': 1e-3},
        {'params': model.gap_head.parameters(), 'lr': 5e-4},  # Plus lent pour gap
        {'params': model.gap_attention.parameters(), 'lr': 5e-4}
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15, verbose=False)
    
    print(f"   Loss pondérée: L_ecran=1.0, gap=10.0")
    print(f"   Learning rates: features=1e-3, gap=5e-4")
    
    # Entraînement
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    
    print(f"Début de l'entraînement avancé...")
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
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Scheduler et early stopping
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/neural_network_06_06_25.pth')
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: Train = {train_loss:.6f}, Val = {val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping à l'epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    print(f"Entraînement terminé en {training_time/60:.1f} minutes")
    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load('models/neural_network_06_06_25.pth'))
    
    return model, train_losses, val_losses

def evaluate_advanced_model(model, X_test, y_test_scaled, y_test_original, scaler_L, scaler_gap):
    """Évalue le modèle avancé."""
    
    print(f"\n=== ÉVALUATION MODÈLE NEURAL NETWORK 06-06-25 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Prédictions
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()
    
    # Dénormaliser les prédictions
    y_pred_L = scaler_L.inverse_transform(y_pred_scaled[:, 0:1]).flatten()
    y_pred_gap = scaler_gap.inverse_transform(y_pred_scaled[:, 1:2]).flatten()
    y_pred = np.column_stack([y_pred_L, y_pred_gap])
    
    # Arrondir les prédictions à 3 décimales pour cohérence
    y_pred_rounded = np.round(y_pred, 3)
    
    # Calculer les métriques
    r2_global = r2_score(y_test_original, y_pred_rounded)
    r2_L = r2_score(y_test_original[:, 0], y_pred_rounded[:, 0])
    r2_gap = r2_score(y_test_original[:, 1], y_pred_rounded[:, 1])
    
    rmse_L = np.sqrt(mean_squared_error(y_test_original[:, 0], y_pred_rounded[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_test_original[:, 1], y_pred_rounded[:, 1]))
    
    mae_L = mean_absolute_error(y_test_original[:, 0], y_pred_rounded[:, 0])
    mae_gap = mean_absolute_error(y_test_original[:, 1], y_pred_rounded[:, 1])
    
    mape_L = np.mean(np.abs((y_test_original[:, 0] - y_pred_rounded[:, 0]) / np.maximum(y_test_original[:, 0], 1e-8))) * 100
    mape_gap = np.mean(np.abs((y_test_original[:, 1] - y_pred_rounded[:, 1]) / np.maximum(y_test_original[:, 1], 1e-8))) * 100
    
    print(f"RÉSULTATS NEURAL NETWORK 06-06-25:")
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
    print(f"  Objectif R² > 0.8: {'✅ ATTEINT' if success else '❌ NON ATTEINT'}")
    
    return y_pred_rounded, {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'mape_L': mape_L, 'mape_gap': mape_gap,
        'success': success
    }

def main():
    """Fonction principale du Neural Network 06-06-25."""
    
    print("="*80)
    print("🚀 NEURAL NETWORK 06-06-25 - PROBLEM-SOLVING EDITION")
    print("Résolution systématique des 5 problèmes identifiés")
    print("="*80)
    
    # 1. Charger et prétraiter les données (Problèmes 1-3)
    X_scaled, y_scaled, scaler_X, scaler_L, scaler_gap = load_and_preprocess_data()
    
    # 2. Charger les données de test
    X_test, y_test_scaled, y_test_original, filenames = load_test_data_advanced(scaler_X, scaler_L, scaler_gap)
    
    # 3. Division train/validation
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    print(f"\nDivision des données:")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # 4. Entraînement avancé (Problèmes 4-5)
    os.makedirs('models', exist_ok=True)
    model, train_losses, val_losses = train_advanced_model(X_train, X_val, y_train, y_val)
    
    # 5. Évaluation finale
    y_pred, metrics = evaluate_advanced_model(model, X_test, y_test_scaled, y_test_original, scaler_L, scaler_gap)
    
    # 6. Sauvegarder les scalers
    import joblib
    joblib.dump(scaler_X, 'models/neural_network_06_06_25_scaler_X.pkl')
    joblib.dump(scaler_L, 'models/neural_network_06_06_25_scaler_L.pkl')
    joblib.dump(scaler_gap, 'models/neural_network_06_06_25_scaler_gap.pkl')
    
    print(f"\n{'='*80}")
    print(f"🎯 RÉSULTATS FINAUX NEURAL NETWORK 06-06-25")
    print(f"{'='*80}")
    
    print(f"✅ PROBLÈMES RÉSOLUS:")
    print(f"   1. 🔢 Labels arrondis à 3 décimales")
    print(f"   2. ⚖️ Normalisation séparée par paramètre")
    print(f"   3. 📊 Focus sur plage expérimentale")
    print(f"   4. 🎛️ Loss pondérée (gap x10)")
    print(f"   5. 🔍 Architecture spécialisée + attention pour gap")
    
    print(f"\n📊 PERFORMANCES:")
    print(f"   • R² global: {metrics['r2_global']:.6f}")
    print(f"   • R² L_ecran: {metrics['r2_L']:.6f}")
    print(f"   • R² gap: {metrics['r2_gap']:.6f}")
    print(f"   • Objectif atteint: {'🎉 OUI' if metrics['success'] else '⚠️ NON'}")
    
    print(f"\n📁 FICHIERS GÉNÉRÉS:")
    print(f"   • models/neural_network_06_06_25.pth")
    print(f"   • models/neural_network_06_06_25_scaler_*.pkl")

if __name__ == "__main__":
    main()
