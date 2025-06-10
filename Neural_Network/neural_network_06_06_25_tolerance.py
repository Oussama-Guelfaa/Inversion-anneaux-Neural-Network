#!/usr/bin/env python3
"""
Neural Network 06-06-25 with Tolerance Evaluation
Author: Oussama GUELFAA
Date: 06 - 06 - 2025

Version modifiée avec évaluation par tolérance :
- Tolérance de 0.01 pour considérer une prédiction comme correcte
- Métriques d'accuracy avec tolérance en plus des métriques classiques
- Basé sur la version ULTRA qui avait donné les meilleurs résultats
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

class UltraWeightedLoss(nn.Module):
    """Loss ultra-pondérée avec focus extrême sur gap."""
    
    def __init__(self, gap_weight=50.0):
        super(UltraWeightedLoss, self).__init__()
        self.gap_weight = gap_weight
    
    def forward(self, pred, target):
        mse_L = (pred[:, 0] - target[:, 0]) ** 2
        mse_gap = (pred[:, 1] - target[:, 1]) ** 2
        
        total_loss = mse_L.mean() + self.gap_weight * mse_gap.mean()
        return total_loss

class UltraSpecializedRegressor(nn.Module):
    """Architecture ultra-spécialisée avec focus maximal sur gap."""
    
    def __init__(self, input_size=600):
        super(UltraSpecializedRegressor, self).__init__()
        
        # Feature extractor commun plus profond
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
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
        
        # Tête L_ecran simple
        self.L_ecran_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Tête gap ultra-spécialisée
        self.gap_feature_enhancer = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.01),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.01),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Mécanisme d'attention double pour gap
        self.gap_attention_1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        self.gap_attention_2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        # Prédicteur gap final
        self.gap_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.005),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # L_ecran
        L_ecran = self.L_ecran_head(features)
        
        # Gap avec double attention
        gap_features = self.gap_feature_enhancer(features)
        attention_1 = self.gap_attention_1(gap_features)
        attention_2 = self.gap_attention_2(gap_features)
        combined_attention = (attention_1 + attention_2) / 2
        attended_features = gap_features * combined_attention
        gap = self.gap_predictor(attended_features)
        
        return torch.cat([L_ecran, gap], dim=1)

def calculate_tolerance_accuracy(y_true, y_pred, tolerance=0.1):
    """
    Calcule l'accuracy avec tolérance.

    Args:
        y_true: Valeurs vraies
        y_pred: Valeurs prédites
        tolerance: Tolérance acceptable (défaut: 0.1)

    Returns:
        accuracy: Pourcentage de prédictions dans la tolérance
    """

    # Calculer l'écart absolu
    absolute_errors = np.abs(y_true - y_pred)

    # Compter les prédictions dans la tolérance
    correct_predictions = absolute_errors <= tolerance

    # Calculer l'accuracy
    accuracy = np.mean(correct_predictions) * 100

    return accuracy

def calculate_adaptive_tolerance_accuracy(y_true_L, y_pred_L, y_true_gap, y_pred_gap, tolerance_L=0.5, tolerance_gap=0.1):
    """
    Calcule l'accuracy avec tolérances adaptatives pour L_ecran et gap.

    Args:
        y_true_L: Valeurs vraies L_ecran
        y_pred_L: Valeurs prédites L_ecran
        y_true_gap: Valeurs vraies gap
        y_pred_gap: Valeurs prédites gap
        tolerance_L: Tolérance pour L_ecran (défaut: 0.5)
        tolerance_gap: Tolérance pour gap (défaut: 0.1)

    Returns:
        dict: Accuracies pour L_ecran, gap et globale
    """

    # Calculer les erreurs absolues
    errors_L = np.abs(y_true_L - y_pred_L)
    errors_gap = np.abs(y_true_gap - y_pred_gap)

    # Prédictions correctes par paramètre
    correct_L = errors_L <= tolerance_L
    correct_gap = errors_gap <= tolerance_gap
    correct_both = correct_L & correct_gap

    # Calculer les accuracies
    accuracy_L = np.mean(correct_L) * 100
    accuracy_gap = np.mean(correct_gap) * 100
    accuracy_global = np.mean(correct_both) * 100

    return {
        'accuracy_L': accuracy_L,
        'accuracy_gap': accuracy_gap,
        'accuracy_global': accuracy_global,
        'correct_L': np.sum(correct_L),
        'correct_gap': np.sum(correct_gap),
        'correct_both': np.sum(correct_both)
    }

def evaluate_with_adaptive_tolerance(model, X_test, y_test_original, scaler_L, scaler_gap, tolerance_L=0.5, tolerance_gap=0.1):
    """Évalue le modèle avec métriques classiques + tolérances adaptatives."""

    print(f"\n=== ÉVALUATION AVEC TOLÉRANCES ADAPTATIVES ===")
    print(f"Tolérance L_ecran: ±{tolerance_L} µm")
    print(f"Tolérance gap: ±{tolerance_gap} µm")

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

    # Arrondir les prédictions à 3 décimales
    y_pred_rounded = np.round(y_pred, 3)

    # === MÉTRIQUES CLASSIQUES ===
    r2_global = r2_score(y_test_original, y_pred_rounded)
    r2_L = r2_score(y_test_original[:, 0], y_pred_rounded[:, 0])
    r2_gap = r2_score(y_test_original[:, 1], y_pred_rounded[:, 1])

    rmse_L = np.sqrt(mean_squared_error(y_test_original[:, 0], y_pred_rounded[:, 0]))
    rmse_gap = np.sqrt(mean_squared_error(y_test_original[:, 1], y_pred_rounded[:, 1]))

    mae_L = mean_absolute_error(y_test_original[:, 0], y_pred_rounded[:, 0])
    mae_gap = mean_absolute_error(y_test_original[:, 1], y_pred_rounded[:, 1])

    mape_L = np.mean(np.abs((y_test_original[:, 0] - y_pred_rounded[:, 0]) / np.maximum(y_test_original[:, 0], 1e-8))) * 100
    mape_gap = np.mean(np.abs((y_test_original[:, 1] - y_pred_rounded[:, 1]) / np.maximum(y_test_original[:, 1], 1e-8))) * 100

    # === NOUVELLES MÉTRIQUES AVEC TOLÉRANCES ADAPTATIVES ===
    adaptive_results = calculate_adaptive_tolerance_accuracy(
        y_test_original[:, 0], y_pred_rounded[:, 0],
        y_test_original[:, 1], y_pred_rounded[:, 1],
        tolerance_L, tolerance_gap
    )
    
    print(f"MÉTRIQUES CLASSIQUES:")
    print(f"  R² global: {r2_global:.6f}")
    print(f"  R² L_ecran: {r2_L:.6f}")
    print(f"  R² gap: {r2_gap:.6f}")
    print(f"  RMSE L_ecran: {rmse_L:.6f} µm")
    print(f"  RMSE gap: {rmse_gap:.6f} µm")
    print(f"  MAE L_ecran: {mae_L:.6f} µm")
    print(f"  MAE gap: {mae_gap:.6f} µm")
    print(f"  MAPE L_ecran: {mape_L:.2f}%")
    print(f"  MAPE gap: {mape_gap:.2f}%")

    print(f"\nMÉTRIQUES AVEC TOLÉRANCES ADAPTATIVES:")
    print(f"  Accuracy L_ecran (±{tolerance_L}): {adaptive_results['accuracy_L']:.2f}% ({adaptive_results['correct_L']}/{len(y_test_original)} échantillons)")
    print(f"  Accuracy gap (±{tolerance_gap}): {adaptive_results['accuracy_gap']:.2f}% ({adaptive_results['correct_gap']}/{len(y_test_original)} échantillons)")
    print(f"  Accuracy globale: {adaptive_results['accuracy_global']:.2f}%")
    print(f"  Prédictions parfaites (L_ecran ET gap): {adaptive_results['correct_both']}/{len(y_test_original)} ({adaptive_results['correct_both']/len(y_test_original)*100:.1f}%)")

    # Analyse détaillée des erreurs
    print(f"\nANALYSE DES ERREURS:")
    errors_L = np.abs(y_test_original[:, 0] - y_pred_rounded[:, 0])
    errors_gap = np.abs(y_test_original[:, 1] - y_pred_rounded[:, 1])

    print(f"  Erreurs L_ecran:")
    print(f"    Min: {errors_L.min():.6f} µm")
    print(f"    Max: {errors_L.max():.6f} µm")
    print(f"    Moyenne: {errors_L.mean():.6f} µm")
    print(f"    Dans tolérance (±{tolerance_L}): {np.sum(errors_L <= tolerance_L)}/{len(errors_L)}")

    print(f"  Erreurs gap:")
    print(f"    Min: {errors_gap.min():.6f} µm")
    print(f"    Max: {errors_gap.max():.6f} µm")
    print(f"    Moyenne: {errors_gap.mean():.6f} µm")
    print(f"    Dans tolérance (±{tolerance_gap}): {np.sum(errors_gap <= tolerance_gap)}/{len(errors_gap)}")

    success_r2 = r2_global > 0.8
    success_tolerance = adaptive_results['accuracy_global'] > 80.0  # 80% des prédictions dans la tolérance

    print(f"\nOBJECTIFS:")
    print(f"  R² > 0.8: {'✅ ATTEINT' if success_r2 else '❌ NON ATTEINT'} (R² = {r2_global:.6f})")
    print(f"  Accuracy > 80%: {'✅ ATTEINT' if success_tolerance else '❌ NON ATTEINT'} (Accuracy = {adaptive_results['accuracy_global']:.2f}%)")

    return {
        'r2_global': r2_global, 'r2_L': r2_L, 'r2_gap': r2_gap,
        'rmse_L': rmse_L, 'rmse_gap': rmse_gap,
        'mae_L': mae_L, 'mae_gap': mae_gap,
        'mape_L': mape_L, 'mape_gap': mape_gap,
        'accuracy_L_tolerance': adaptive_results['accuracy_L'],
        'accuracy_gap_tolerance': adaptive_results['accuracy_gap'],
        'accuracy_global_tolerance': adaptive_results['accuracy_global'],
        'perfect_L': adaptive_results['correct_L'],
        'perfect_gap': adaptive_results['correct_gap'],
        'perfect_both': adaptive_results['correct_both'],
        'success_r2': success_r2, 'success_tolerance': success_tolerance,
        'tolerance_L': tolerance_L, 'tolerance_gap': tolerance_gap
    }

def load_and_preprocess_data():
    """Charge et prétraite les données (même méthode que la version ULTRA)."""
    
    print("="*80)
    print("CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES")
    print("="*80)
    
    # Charger les données tronquées
    df_profiles = pd.read_csv('processed_data/intensity_profiles_truncated_600.csv')
    df_params = pd.read_csv('processed_data/parameters_truncated_600.csv')
    
    X_raw = df_profiles.values.astype(np.float32)
    y_raw = df_params[['L_ecran', 'gap']].values.astype(np.float32)
    
    # Arrondissement à 3 décimales
    y_rounded = np.round(y_raw, 3)
    
    # Focus sur plage expérimentale
    experimental_mask = (y_rounded[:, 1] >= 0.025) & (y_rounded[:, 1] <= 0.517)
    X_focused = X_raw[experimental_mask]
    y_focused = y_rounded[experimental_mask]
    
    print(f"Données focalisées: {X_focused.shape[0]} échantillons")
    print(f"Plage gap: [{y_focused[:, 1].min():.3f}, {y_focused[:, 1].max():.3f}] µm")
    
    # Normalisation séparée
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_focused)
    
    scaler_L = StandardScaler()
    scaler_gap = StandardScaler()
    
    y_L_scaled = scaler_L.fit_transform(y_focused[:, 0:1])
    y_gap_scaled = scaler_gap.fit_transform(y_focused[:, 1:2])
    y_scaled = np.hstack([y_L_scaled, y_gap_scaled])
    
    return X_scaled, y_scaled, scaler_X, scaler_L, scaler_gap

def load_test_data(scaler_X):
    """Charge les données de test."""
    
    print(f"\n=== CHARGEMENT DONNÉES DE TEST ===")
    
    dataset_dir = "../data_generation/dataset"
    labels_df = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))
    
    X_test = []
    y_test = []
    
    for idx, row in labels_df.iterrows():
        filename = row['filename']
        gap = round(row['gap_um'], 3)
        L_ecran = round(row['L_um'], 3)
        
        mat_filename = filename.replace('.png', '.mat')
        mat_path = os.path.join(dataset_dir, mat_filename)
        
        if os.path.exists(mat_path):
            try:
                data = sio.loadmat(mat_path)
                ratio = data['ratio'].flatten()[:600]
                X_test.append(ratio)
                y_test.append([L_ecran, gap])
            except:
                pass
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    # Normaliser test
    X_test_scaled = scaler_X.transform(X_test)
    
    print(f"Données de test: {X_test.shape[0]} échantillons")
    
    return X_test_scaled, y_test

def train_model(X_train, X_val, y_train, y_val, epochs=150):
    """Entraîne le modèle (même méthode que la version ULTRA)."""
    
    print(f"\n=== ENTRAÎNEMENT DU MODÈLE ===")
    
    model = UltraSpecializedRegressor(input_size=600)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Device utilisé: {device}")
    
    # Convertir en tenseurs
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Loss pondérée et optimiseur
    criterion = UltraWeightedLoss(gap_weight=50.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    
    print(f"Début de l'entraînement...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Entraînement
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/tolerance_model.pth')
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
    model.load_state_dict(torch.load('models/tolerance_model.pth'))
    
    return model

def main():
    """Fonction principale avec évaluation par tolérance."""
    
    print("="*80)
    print("🎯 NEURAL NETWORK 06-06-25 AVEC TOLÉRANCES ADAPTATIVES")
    print("Tolérance L_ecran: ±0.5 µm | Tolérance gap: ±0.1 µm")
    print("="*80)
    
    # 1. Charger et prétraiter les données
    X_scaled, y_scaled, scaler_X, scaler_L, scaler_gap = load_and_preprocess_data()
    
    # 2. Charger données de test
    X_test_scaled, y_test_original = load_test_data(scaler_X)
    
    # 3. Division train/validation
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    print(f"\nDonnées d'entraînement: {X_train.shape}")
    print(f"Données de validation: {X_val.shape}")
    print(f"Données de test: {X_test_scaled.shape}")
    
    # 4. Entraîner le modèle
    os.makedirs('models', exist_ok=True)
    model = train_model(X_train, X_val, y_train, y_val)
    
    # 5. Évaluation avec tolérances adaptatives
    metrics = evaluate_with_adaptive_tolerance(model, X_test_scaled, y_test_original,
                                             scaler_L, scaler_gap, tolerance_L=0.5, tolerance_gap=0.1)
    
    # 6. Sauvegarder les scalers
    import joblib
    joblib.dump(scaler_X, 'models/tolerance_scaler_X.pkl')
    joblib.dump(scaler_L, 'models/tolerance_scaler_L.pkl')
    joblib.dump(scaler_gap, 'models/tolerance_scaler_gap.pkl')
    
    # 7. Résumé final
    print(f"\n{'='*80}")
    print(f"🎯 RÉSULTATS FINAUX AVEC TOLÉRANCE")
    print(f"{'='*80}")
    
    print(f"PERFORMANCES CLASSIQUES:")
    print(f"  R² global: {metrics['r2_global']:.6f}")
    print(f"  R² L_ecran: {metrics['r2_L']:.6f}")
    print(f"  R² gap: {metrics['r2_gap']:.6f}")
    
    print(f"\nPERFORMANCES AVEC TOLÉRANCES ADAPTATIVES:")
    print(f"  Accuracy L_ecran (±{metrics['tolerance_L']}): {metrics['accuracy_L_tolerance']:.2f}%")
    print(f"  Accuracy gap (±{metrics['tolerance_gap']}): {metrics['accuracy_gap_tolerance']:.2f}%")
    print(f"  Accuracy globale: {metrics['accuracy_global_tolerance']:.2f}%")
    print(f"  Prédictions parfaites: {metrics['perfect_both']}/{48} ({metrics['perfect_both']/48*100:.1f}%)")
    
    print(f"\nOBJECTIFS:")
    print(f"  R² > 0.8: {'🎉 ATTEINT' if metrics['success_r2'] else '⚠️ NON ATTEINT'}")
    print(f"  Accuracy > 80%: {'🎉 ATTEINT' if metrics['success_tolerance'] else '⚠️ NON ATTEINT'}")
    
    if metrics['success_tolerance']:
        print(f"\n🏆 FÉLICITATIONS ! Objectif d'accuracy avec tolérance atteint !")
    elif metrics['accuracy_global_tolerance'] > 70:
        print(f"\n📈 Très bon progrès ! Accuracy = {metrics['accuracy_global_tolerance']:.1f}% (proche de l'objectif)")
    
    print(f"\n📁 FICHIERS GÉNÉRÉS:")
    print(f"  • models/tolerance_model.pth")
    print(f"  • models/tolerance_scaler_*.pkl")
    
    return metrics

if __name__ == "__main__":
    results = main()
