#!/usr/bin/env python3
"""
Script d'Entraînement avec K-Fold Cross-Validation

Auteur: Oussama GUELFAA
Date: 25/06/2025

Améliorations selon les recommandations du tuteur :
1. Split aléatoire au lieu de stratifié
2. K-Fold Cross-Validation pour meilleure évaluation
3. Évaluation robuste de la généralisation
4. Visualisations des résultats par fold
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HologramDataset(Dataset):
    """Dataset PyTorch pour les profils holographiques."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImprovedDualParameterNet(nn.Module):
    """
    Réseau amélioré pour précision gap optimisée.
    Architecture identique au modèle amélioré.
    """
    
    def __init__(self, input_size=600, dropout_rate=0.15):
        super(ImprovedDualParameterNet, self).__init__()
        
        # Architecture plus profonde pour meilleure capacité
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(1024, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(768, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(dropout_rate * 0.8)
        
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(dropout_rate * 0.6)
        
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.dropout6 = nn.Dropout(dropout_rate * 0.4)
        
        # Couches spécialisées pour chaque paramètre
        self.gap_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.L_ecran_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.2),
            nn.Linear(32, 1)
        )
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise les poids avec Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass avec têtes spécialisées."""
        # Backbone partagé
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        x = torch.relu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)
        
        features = torch.relu(self.bn6(self.fc6(x)))
        features = self.dropout6(features)
        
        # Têtes spécialisées
        gap_pred = self.gap_head(features)
        L_ecran_pred = self.L_ecran_head(features)
        
        return torch.cat([gap_pred, L_ecran_pred], dim=1)

class WeightedDualLoss(nn.Module):
    """Loss pondérée privilégiant la précision du gap."""
    
    def __init__(self, gap_weight=15.0, L_ecran_weight=1.0):
        super(WeightedDualLoss, self).__init__()
        self.gap_weight = gap_weight
        self.L_ecran_weight = L_ecran_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        gap_loss = self.mse(predictions[:, 0], targets[:, 0])
        L_ecran_loss = self.mse(predictions[:, 1], targets[:, 1])
        
        total_loss = (self.gap_weight * gap_loss + 
                     self.L_ecran_weight * L_ecran_loss)
        
        return total_loss, gap_loss, L_ecran_loss

class KFoldDualParameterTrainer:
    """Entraîneur avec K-Fold Cross-Validation."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🔧 Trainer K-Fold initialisé sur {self.device}")
    
    def load_dataset(self, dataset_path, max_files=None):
        """Charge le dataset avec préprocessing."""
        logger.info(f"📊 Chargement dataset: {dataset_path}")
        
        # Charger labels.csv
        labels_path = Path(dataset_path) / "labels.csv"
        labels_df = pd.read_csv(labels_path)
        
        if max_files:
            labels_df = labels_df.head(max_files)
        
        X_data = []
        y_data = []
        truncate_to = self.config.get('truncate_to', 600)
        
        for i, (_, row) in enumerate(labels_df.iterrows()):
            if i % 2000 == 0:
                logger.info(f"   Progression: {i}/{len(labels_df)} fichiers...")
            
            filename = row['filename']
            gap = row['gap_um']
            L_ecran = row['L_um']
            
            mat_file_path = Path(dataset_path) / filename
            
            try:
                # Charger le fichier .mat
                data = loadmat(str(mat_file_path))
                
                # Extraire le profil d'intensité
                if 'ratio' in data:
                    ratio = data['ratio'].flatten()
                elif 'I_ratio' in data:
                    ratio = data['I_ratio'].flatten()
                else:
                    possible_keys = [k for k in data.keys() if not k.startswith('__')]
                    if possible_keys:
                        ratio = data[possible_keys[0]].flatten()
                    else:
                        continue
                
                # Préprocessing
                if len(ratio) > truncate_to:
                    ratio = ratio[:truncate_to]
                elif len(ratio) < truncate_to:
                    ratio = np.pad(ratio, (0, truncate_to - len(ratio)), 'edge')
                
                # Filtrage léger pour réduire le bruit
                from scipy.ndimage import gaussian_filter1d
                ratio = gaussian_filter1d(ratio, sigma=0.5)
                
                X_data.append(ratio)
                y_data.append([gap, L_ecran])
                
            except Exception as e:
                logger.warning(f"   ⚠️  Erreur avec {filename}: {e}")
                continue
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        logger.info(f"✅ Dataset chargé: X{X.shape}, y{y.shape}")
        logger.info(f"   Gap range: {np.min(y[:, 0]):.4f} - {np.max(y[:, 0]):.4f} µm")
        logger.info(f"   L_ecran range: {np.min(y[:, 1]):.1f} - {np.max(y[:, 1]):.1f} µm")
        
        return X, y
    
    def train_single_fold(self, X_train, y_train, X_val, y_val, fold_num, epochs=200):
        """Entraîne le modèle pour un fold donné."""
        logger.info(f"\n🔄 FOLD {fold_num + 1} - Entraînement")
        
        # Scalers séparés pour ce fold
        input_scaler = StandardScaler()
        gap_scaler = MinMaxScaler(feature_range=(-1, 1))
        L_ecran_scaler = StandardScaler()
        
        # Normalisation des entrées
        X_train_scaled = input_scaler.fit_transform(X_train)
        X_val_scaled = input_scaler.transform(X_val)
        
        # Normalisation séparée des sorties
        gap_train = y_train[:, 0].reshape(-1, 1)
        gap_val = y_val[:, 0].reshape(-1, 1)
        L_ecran_train = y_train[:, 1].reshape(-1, 1)
        L_ecran_val = y_val[:, 1].reshape(-1, 1)
        
        gap_train_scaled = gap_scaler.fit_transform(gap_train)
        gap_val_scaled = gap_scaler.transform(gap_val)
        L_ecran_train_scaled = L_ecran_scaler.fit_transform(L_ecran_train)
        L_ecran_val_scaled = L_ecran_scaler.transform(L_ecran_val)
        
        # Recombiner
        y_train_scaled = np.hstack([gap_train_scaled, L_ecran_train_scaled])
        y_val_scaled = np.hstack([gap_val_scaled, L_ecran_val_scaled])
        
        # Créer les datasets PyTorch
        train_dataset = HologramDataset(X_train_scaled, y_train_scaled)
        val_dataset = HologramDataset(X_val_scaled, y_val_scaled)
        
        # Créer les DataLoaders
        batch_size = self.config.get('batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Créer le modèle
        model = ImprovedDualParameterNet(input_size=600).to(self.device)
        criterion = WeightedDualLoss(gap_weight=self.config.get('gap_weight', 15.0))
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 0.0005),
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        
        # Entraînement
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        fold_history = {
            'train_loss': [], 'val_loss': [],
            'gap_r2': [], 'L_ecran_r2': [],
            'gap_mae': [], 'L_ecran_mae': []
        }
        
        for epoch in range(epochs):
            # Phase d'entraînement
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss, _, _ = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Phase de validation
            model.eval()
            val_loss = 0.0
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    predictions = model(batch_X)
                    loss, _, _ = criterion(predictions, batch_y)
                    
                    val_loss += loss.item()
                    all_predictions.append(predictions.cpu().numpy())
                    all_targets.append(batch_y.cpu().numpy())
            
            # Calculer les métriques
            predictions = np.vstack(all_predictions)
            targets = np.vstack(all_targets)
            
            # Dénormaliser
            gap_pred_denorm = gap_scaler.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
            gap_target_denorm = gap_scaler.inverse_transform(targets[:, 0].reshape(-1, 1)).flatten()
            L_ecran_pred_denorm = L_ecran_scaler.inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()
            L_ecran_target_denorm = L_ecran_scaler.inverse_transform(targets[:, 1].reshape(-1, 1)).flatten()
            
            gap_r2 = r2_score(gap_target_denorm, gap_pred_denorm)
            L_ecran_r2 = r2_score(L_ecran_target_denorm, L_ecran_pred_denorm)
            gap_mae = mean_absolute_error(gap_target_denorm, gap_pred_denorm)
            L_ecran_mae = mean_absolute_error(L_ecran_target_denorm, L_ecran_pred_denorm)
            
            # Mise à jour historique
            fold_history['train_loss'].append(train_loss / len(train_loader))
            fold_history['val_loss'].append(val_loss / len(val_loader))
            fold_history['gap_r2'].append(gap_r2)
            fold_history['L_ecran_r2'].append(L_ecran_r2)
            fold_history['gap_mae'].append(gap_mae)
            fold_history['L_ecran_mae'].append(L_ecran_mae)
            
            # Scheduler et early stopping
            current_val_loss = val_loss / len(val_loader)
            scheduler.step(current_val_loss)
            
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Affichage périodique
            if (epoch + 1) % 20 == 0:
                logger.info(f"   Epoch {epoch+1:3d}: Gap MAE={gap_mae:.4f}µm, "
                           f"Gap R²={gap_r2:.3f}, L_ecran R²={L_ecran_r2:.3f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"   Early stopping à l'epoch {epoch+1}")
                break
        
        # Restaurer le meilleur modèle
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
        
        # Résultats finaux du fold
        final_metrics = {
            'gap_r2': fold_history['gap_r2'][-1],
            'L_ecran_r2': fold_history['L_ecran_r2'][-1],
            'gap_mae': fold_history['gap_mae'][-1],
            'L_ecran_mae': fold_history['L_ecran_mae'][-1],
            'val_loss': fold_history['val_loss'][-1]
        }
        
        logger.info(f"✅ Fold {fold_num + 1} terminé: Gap R²={final_metrics['gap_r2']:.3f}, "
                   f"Gap MAE={final_metrics['gap_mae']:.4f}µm")
        
        return model, fold_history, final_metrics, (input_scaler, gap_scaler, L_ecran_scaler)

    def run_kfold_validation(self, X, y, k_folds=5, epochs=200):
        """Exécute la validation croisée K-Fold."""
        logger.info(f"\n🔄 DÉBUT K-FOLD CROSS-VALIDATION (K={k_folds})")
        logger.info("="*60)

        # Mélange aléatoire des données (comme demandé par le tuteur)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        logger.info(f"📊 Données mélangées aléatoirement: {len(X_shuffled)} échantillons")

        # Initialiser K-Fold
        kfold = KFold(n_splits=k_folds, shuffle=False, random_state=None)  # Pas de shuffle car déjà fait

        # Stocker les résultats de chaque fold
        fold_results = []
        fold_histories = []
        fold_models = []
        fold_scalers = []

        # Métriques globales
        all_gap_r2 = []
        all_L_ecran_r2 = []
        all_gap_mae = []
        all_L_ecran_mae = []

        start_time = time.time()

        # Exécuter chaque fold
        for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X_shuffled)):
            logger.info(f"\n📋 FOLD {fold_num + 1}/{k_folds}")
            logger.info(f"   Train: {len(train_idx)} échantillons")
            logger.info(f"   Validation: {len(val_idx)} échantillons")

            # Diviser les données
            X_train_fold = X_shuffled[train_idx]
            y_train_fold = y_shuffled[train_idx]
            X_val_fold = X_shuffled[val_idx]
            y_val_fold = y_shuffled[val_idx]

            # Entraîner le modèle pour ce fold
            model, history, metrics, scalers = self.train_single_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                fold_num, epochs
            )

            # Stocker les résultats
            fold_results.append(metrics)
            fold_histories.append(history)
            fold_models.append(model)
            fold_scalers.append(scalers)

            # Accumuler les métriques
            all_gap_r2.append(metrics['gap_r2'])
            all_L_ecran_r2.append(metrics['L_ecran_r2'])
            all_gap_mae.append(metrics['gap_mae'])
            all_L_ecran_mae.append(metrics['L_ecran_mae'])

        total_time = time.time() - start_time

        # Calculer les statistiques globales
        gap_r2_mean = np.mean(all_gap_r2)
        gap_r2_std = np.std(all_gap_r2)
        L_ecran_r2_mean = np.mean(all_L_ecran_r2)
        L_ecran_r2_std = np.std(all_L_ecran_r2)
        gap_mae_mean = np.mean(all_gap_mae)
        gap_mae_std = np.std(all_gap_mae)
        L_ecran_mae_mean = np.mean(all_L_ecran_mae)
        L_ecran_mae_std = np.std(all_L_ecran_mae)

        logger.info(f"\n📊 RÉSULTATS K-FOLD CROSS-VALIDATION")
        logger.info("="*50)
        logger.info(f"Temps total: {total_time:.1f}s")
        logger.info(f"\nGAP:")
        logger.info(f"   R²: {gap_r2_mean:.3f} ± {gap_r2_std:.3f}")
        logger.info(f"   MAE: {gap_mae_mean:.4f} ± {gap_mae_std:.4f} µm")
        logger.info(f"\nL_ÉCRAN:")
        logger.info(f"   R²: {L_ecran_r2_mean:.3f} ± {L_ecran_r2_std:.3f}")
        logger.info(f"   MAE: {L_ecran_mae_mean:.1f} ± {L_ecran_mae_std:.1f} µm")

        # Résultats par fold
        logger.info(f"\n📋 DÉTAIL PAR FOLD:")
        for i, result in enumerate(fold_results):
            logger.info(f"   Fold {i+1}: Gap R²={result['gap_r2']:.3f}, "
                       f"Gap MAE={result['gap_mae']:.4f}µm, "
                       f"L_écran R²={result['L_ecran_r2']:.3f}")

        # Identifier le meilleur fold
        best_fold_idx = np.argmin(all_gap_mae)  # Meilleur = plus petit MAE Gap
        logger.info(f"\n🏆 MEILLEUR FOLD: {best_fold_idx + 1} "
                   f"(Gap MAE={all_gap_mae[best_fold_idx]:.4f}µm)")

        # Résumé final
        kfold_summary = {
            'gap_r2_mean': gap_r2_mean,
            'gap_r2_std': gap_r2_std,
            'L_ecran_r2_mean': L_ecran_r2_mean,
            'L_ecran_r2_std': L_ecran_r2_std,
            'gap_mae_mean': gap_mae_mean,
            'gap_mae_std': gap_mae_std,
            'L_ecran_mae_mean': L_ecran_mae_mean,
            'L_ecran_mae_std': L_ecran_mae_std,
            'best_fold_idx': best_fold_idx,
            'total_time': total_time
        }

        return fold_results, fold_histories, fold_models, fold_scalers, kfold_summary

    def save_best_model(self, fold_models, fold_scalers, best_fold_idx, model_path):
        """Sauvegarde le meilleur modèle du K-Fold."""
        best_model = fold_models[best_fold_idx]
        best_scalers = fold_scalers[best_fold_idx]

        # Sauvegarder le modèle
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'model_config': {
                'input_size': 600,
                'dropout_rate': 0.15
            },
            'fold_number': best_fold_idx + 1,
            'training_method': 'kfold_cross_validation'
        }, model_path)

        # Sauvegarder les scalers
        model_dir = Path(model_path).parent
        model_dir.mkdir(exist_ok=True)

        input_scaler, gap_scaler, L_ecran_scaler = best_scalers
        joblib.dump(input_scaler, model_dir / 'input_scaler_kfold.pkl')
        joblib.dump(gap_scaler, model_dir / 'gap_scaler_kfold.pkl')
        joblib.dump(L_ecran_scaler, model_dir / 'L_ecran_scaler_kfold.pkl')

        logger.info(f"💾 Meilleur modèle sauvegardé: {model_path}")
        logger.info(f"💾 Scalers sauvegardés: {model_dir}")

    def create_kfold_visualizations(self, fold_histories, kfold_summary, output_dir="../../plots"):
        """Crée des visualisations des résultats K-Fold."""
        logger.info(f"\n📊 CRÉATION DES VISUALISATIONS K-FOLD")

        # Créer le dossier de sortie
        Path(output_dir).mkdir(exist_ok=True)

        # Configuration matplotlib
        plt.style.use('default')
        sns.set_palette("husl")

        # Figure 1: Évolution des métriques par fold
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Gap R²
        for i, history in enumerate(fold_histories):
            axes[0, 0].plot(history['gap_r2'], label=f'Fold {i+1}', alpha=0.7)
        axes[0, 0].set_title('Évolution Gap R² par Fold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Gap R²')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # L_écran R²
        for i, history in enumerate(fold_histories):
            axes[0, 1].plot(history['L_ecran_r2'], label=f'Fold {i+1}', alpha=0.7)
        axes[0, 1].set_title('Évolution L_écran R² par Fold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('L_écran R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Gap MAE
        for i, history in enumerate(fold_histories):
            axes[1, 0].plot(history['gap_mae'], label=f'Fold {i+1}', alpha=0.7)
        axes[1, 0].set_title('Évolution Gap MAE par Fold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Gap MAE (µm)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # L_écran MAE
        for i, history in enumerate(fold_histories):
            axes[1, 1].plot(history['L_ecran_mae'], label=f'Fold {i+1}', alpha=0.7)
        axes[1, 1].set_title('Évolution L_écran MAE par Fold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('L_écran MAE (µm)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/kfold_evolution_metriques.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 2: Boxplots des performances finales
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Extraire les métriques finales de chaque fold
        gap_r2_final = [history['gap_r2'][-1] for history in fold_histories]
        gap_mae_final = [history['gap_mae'][-1] for history in fold_histories]
        L_ecran_r2_final = [history['L_ecran_r2'][-1] for history in fold_histories]
        L_ecran_mae_final = [history['L_ecran_mae'][-1] for history in fold_histories]

        # Boxplot Gap
        gap_data = [gap_r2_final, gap_mae_final]
        gap_labels = ['R²', 'MAE (µm)']
        axes[0].boxplot(gap_data, labels=gap_labels)
        axes[0].set_title('Distribution des Performances Gap')
        axes[0].grid(True, alpha=0.3)

        # Boxplot L_écran
        L_ecran_data = [L_ecran_r2_final, L_ecran_mae_final]
        L_ecran_labels = ['R²', 'MAE (µm)']
        axes[1].boxplot(L_ecran_data, labels=L_ecran_labels)
        axes[1].set_title('Distribution des Performances L_écran')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/kfold_boxplots_performances.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 3: Comparaison des folds
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        fold_numbers = list(range(1, len(fold_histories) + 1))

        ax.bar([x - 0.2 for x in fold_numbers], gap_r2_final,
               width=0.4, label='Gap R²', alpha=0.7)
        ax.bar([x + 0.2 for x in fold_numbers], L_ecran_r2_final,
               width=0.4, label='L_écran R²', alpha=0.7)

        ax.set_xlabel('Fold')
        ax.set_ylabel('R²')
        ax.set_title('Comparaison R² par Fold')
        ax.set_xticks(fold_numbers)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Ajouter les moyennes
        ax.axhline(y=kfold_summary['gap_r2_mean'], color='blue',
                  linestyle='--', alpha=0.7, label=f'Gap R² moyen: {kfold_summary["gap_r2_mean"]:.3f}')
        ax.axhline(y=kfold_summary['L_ecran_r2_mean'], color='orange',
                  linestyle='--', alpha=0.7, label=f'L_écran R² moyen: {kfold_summary["L_ecran_r2_mean"]:.3f}')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/kfold_comparaison_folds.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ Visualisations K-Fold sauvegardées dans {output_dir}/")
        logger.info(f"   - kfold_evolution_metriques.png")
        logger.info(f"   - kfold_boxplots_performances.png")
        logger.info(f"   - kfold_comparaison_folds.png")


def save_kfold_results_csv(fold_results, kfold_summary, output_file="../../results/kfold_results.csv"):
    """Sauvegarde les résultats K-Fold dans un CSV."""

    # Créer le DataFrame des résultats par fold
    df_folds = pd.DataFrame(fold_results)
    df_folds['fold_number'] = range(1, len(fold_results) + 1)

    # Réorganiser les colonnes
    df_folds = df_folds[['fold_number', 'gap_r2', 'gap_mae', 'L_ecran_r2', 'L_ecran_mae', 'val_loss']]

    # Ajouter une ligne de résumé
    summary_row = {
        'fold_number': 'MOYENNE',
        'gap_r2': kfold_summary['gap_r2_mean'],
        'gap_mae': kfold_summary['gap_mae_mean'],
        'L_ecran_r2': kfold_summary['L_ecran_r2_mean'],
        'L_ecran_mae': kfold_summary['L_ecran_mae_mean'],
        'val_loss': 'N/A'
    }

    std_row = {
        'fold_number': 'ECART_TYPE',
        'gap_r2': kfold_summary['gap_r2_std'],
        'gap_mae': kfold_summary['gap_mae_std'],
        'L_ecran_r2': kfold_summary['L_ecran_r2_std'],
        'L_ecran_mae': kfold_summary['L_ecran_mae_std'],
        'val_loss': 'N/A'
    }

    # Ajouter les lignes de résumé
    df_final = pd.concat([df_folds, pd.DataFrame([summary_row, std_row])], ignore_index=True)

    # Créer le dossier de sortie
    Path(output_file).parent.mkdir(exist_ok=True)

    # Sauvegarder
    df_final.to_csv(output_file, index=False, float_format='%.4f')

    logger.info(f"💾 Résultats K-Fold sauvegardés: {output_file}")

    return df_final


def main():
    """Fonction principale pour K-Fold Cross-Validation."""
    parser = argparse.ArgumentParser(description='Entraînement K-Fold Cross-Validation')
    parser.add_argument('--dataset', type=str, default='../../data_generation/dataset_2D_Train_Augmented')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--gap_weight', type=float, default=15.0)
    parser.add_argument('--model_path', type=str, default='../../models/dual_parameter_model_kfold.pt')
    parser.add_argument('--max_samples', type=int, default=None, help='Limite le nombre d\'échantillons pour test rapide')

    args = parser.parse_args()

    config = {
        'truncate_to': 600,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'gap_weight': args.gap_weight,
        'k_folds': args.k_folds
    }

    logger.info("🚀 ENTRAÎNEMENT K-FOLD CROSS-VALIDATION")
    logger.info("="*70)
    logger.info(f"📊 Dataset: {args.dataset}")
    logger.info(f"🔄 K-Folds: {args.k_folds}")
    logger.info(f"🎯 Epochs par fold: {args.epochs}")
    logger.info(f"📦 Batch size: {args.batch_size}")
    logger.info(f"📈 Learning rate: {args.learning_rate}")
    logger.info(f"⚖️ Gap weight: {args.gap_weight}x")
    if args.max_samples:
        logger.info(f"⚠️ Mode test: Limité à {args.max_samples} échantillons")

    # Créer l'entraîneur K-Fold
    trainer = KFoldDualParameterTrainer(config)

    # Charger les données
    X, y = trainer.load_dataset(args.dataset, max_files=args.max_samples)

    # Exécuter la validation croisée K-Fold
    fold_results, fold_histories, fold_models, fold_scalers, kfold_summary = trainer.run_kfold_validation(
        X, y, k_folds=args.k_folds, epochs=args.epochs
    )

    # Sauvegarder le meilleur modèle
    trainer.save_best_model(fold_models, fold_scalers, kfold_summary['best_fold_idx'], args.model_path)

    # Créer les visualisations
    trainer.create_kfold_visualizations(fold_histories, kfold_summary)

    # Sauvegarder les résultats en CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"../../results/kfold_results_{timestamp}.csv"
    save_kfold_results_csv(fold_results, kfold_summary, csv_file)

    # Afficher le résumé final
    logger.info(f"\n🎉 K-FOLD CROSS-VALIDATION TERMINÉE")
    logger.info("="*50)
    logger.info(f"📊 Résultats moyens ({args.k_folds} folds):")
    logger.info(f"   Gap R²: {kfold_summary['gap_r2_mean']:.3f} ± {kfold_summary['gap_r2_std']:.3f}")
    logger.info(f"   Gap MAE: {kfold_summary['gap_mae_mean']:.4f} ± {kfold_summary['gap_mae_std']:.4f} µm")
    logger.info(f"   L_écran R²: {kfold_summary['L_ecran_r2_mean']:.3f} ± {kfold_summary['L_ecran_r2_std']:.3f}")
    logger.info(f"   L_écran MAE: {kfold_summary['L_ecran_mae_mean']:.1f} ± {kfold_summary['L_ecran_mae_std']:.1f} µm")
    logger.info(f"\n💾 Fichiers générés:")
    logger.info(f"   Modèle: {args.model_path}")
    logger.info(f"   Résultats CSV: {csv_file}")
    logger.info(f"   Visualisations: ../../plots/kfold_*.png")

    # Évaluation de la qualité
    if kfold_summary['gap_r2_mean'] > 0.7 and kfold_summary['L_ecran_r2_mean'] > 0.95:
        logger.info(f"\n🎉 MODÈLE EXCELLENT ! Généralisation robuste validée.")
    elif kfold_summary['gap_r2_mean'] > 0.5 and kfold_summary['L_ecran_r2_mean'] > 0.9:
        logger.info(f"\n✅ MODÈLE DE BONNE QUALITÉ. Généralisation satisfaisante.")
    else:
        logger.info(f"\n⚠️ MODÈLE ACCEPTABLE. Amélioration possible.")

    # Recommandations basées sur la variance
    if kfold_summary['gap_r2_std'] > 0.1:
        logger.info(f"💡 Recommandation: Variance Gap R² élevée ({kfold_summary['gap_r2_std']:.3f}) - "
                   f"Considérer plus de données ou régularisation.")

    if kfold_summary['gap_mae_std'] > 0.01:
        logger.info(f"💡 Recommandation: Variance Gap MAE élevée ({kfold_summary['gap_mae_std']:.4f}) - "
                   f"Modèle instable, ajuster hyperparamètres.")


if __name__ == "__main__":
    main()
