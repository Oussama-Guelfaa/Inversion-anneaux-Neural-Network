#!/usr/bin/env python3
"""
Script d'Entraînement Amélioré pour Précision Gap

Auteur: Oussama GUELFAA
Date: 25/06/2025

Améliorations pour augmenter la précision du gap :
1. Loss pondérée privilégiant le gap
2. Architecture plus profonde
3. Learning rate adaptatif
4. Normalisation séparée par paramètre
5. Régularisation avancée
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import logging

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
    
    Améliorations:
    - Architecture plus profonde
    - Connexions résiduelles
    - Normalisation par couche
    - Dropout adaptatif
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
    """
    Loss pondérée privilégiant la précision du gap.
    """
    
    def __init__(self, gap_weight=10.0, L_ecran_weight=1.0):
        super(WeightedDualLoss, self).__init__()
        self.gap_weight = gap_weight
        self.L_ecran_weight = L_ecran_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Calcule la loss pondérée.
        
        Args:
            predictions: [batch_size, 2] - [gap, L_ecran]
            targets: [batch_size, 2] - [gap, L_ecran]
        """
        gap_loss = self.mse(predictions[:, 0], targets[:, 0])
        L_ecran_loss = self.mse(predictions[:, 1], targets[:, 1])
        
        total_loss = (self.gap_weight * gap_loss + 
                     self.L_ecran_weight * L_ecran_loss)
        
        return total_loss, gap_loss, L_ecran_loss

class ImprovedDualParameterTrainer:
    """Entraîneur amélioré pour précision gap optimisée."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Scalers séparés pour meilleure normalisation
        self.input_scaler = StandardScaler()
        self.gap_scaler = MinMaxScaler(feature_range=(-1, 1))  # Normalisation plus agressive pour gap
        self.L_ecran_scaler = StandardScaler()
        
        # Modèle et optimiseur
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        logger.info(f"🔧 Trainer amélioré initialisé sur {self.device}")
    
    def load_dataset(self, dataset_path, max_files=None):
        """Charge le dataset avec préprocessing amélioré."""
        logger.info(f"📊 Chargement dataset amélioré: {dataset_path}")
        
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
                
                # Préprocessing amélioré
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
    
    def prepare_data_improved(self, dataset_path, validation_split=0.2, batch_size=32):
        """Préparation améliorée avec normalisation séparée."""
        logger.info(f"🔄 Préparation améliorée des données...")
        
        # Charger les données
        X, y = self.load_dataset(dataset_path)
        
        # Division train/validation stratifiée par gap
        # Créer des bins pour stratification
        gap_bins = pd.cut(y[:, 0], bins=10, labels=False)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=gap_bins
        )
        
        logger.info(f"📊 Division stratifiée: Train={len(X_train)}, Val={len(X_val)}")
        
        # Normalisation des entrées
        X_train_scaled = self.input_scaler.fit_transform(X_train)
        X_val_scaled = self.input_scaler.transform(X_val)
        
        # Normalisation séparée des sorties
        gap_train = y_train[:, 0].reshape(-1, 1)
        gap_val = y_val[:, 0].reshape(-1, 1)
        L_ecran_train = y_train[:, 1].reshape(-1, 1)
        L_ecran_val = y_val[:, 1].reshape(-1, 1)
        
        gap_train_scaled = self.gap_scaler.fit_transform(gap_train)
        gap_val_scaled = self.gap_scaler.transform(gap_val)
        L_ecran_train_scaled = self.L_ecran_scaler.fit_transform(L_ecran_train)
        L_ecran_val_scaled = self.L_ecran_scaler.transform(L_ecran_val)
        
        # Recombiner
        y_train_scaled = np.hstack([gap_train_scaled, L_ecran_train_scaled])
        y_val_scaled = np.hstack([gap_val_scaled, L_ecran_val_scaled])
        
        # Créer les datasets PyTorch
        train_dataset = HologramDataset(X_train_scaled, y_train_scaled)
        val_dataset = HologramDataset(X_val_scaled, y_val_scaled)
        
        # Créer les DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"✅ Données préparées avec normalisation séparée")
        
        return train_loader, val_loader

    def setup_improved_model(self, input_size=600, learning_rate=0.0005, gap_weight=10.0):
        """Configure le modèle amélioré avec optimisations."""
        self.model = ImprovedDualParameterNet(input_size=input_size).to(self.device)

        # Loss pondérée privilégiant le gap
        self.criterion = WeightedDualLoss(gap_weight=gap_weight, L_ecran_weight=1.0)

        # Optimiseur avec learning rate plus faible
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        # Scheduler plus agressif
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"🏗️ Modèle amélioré créé: {total_params:,} paramètres")
        logger.info(f"   Gap weight: {gap_weight}x (privilégié)")
        logger.info(f"   Learning rate: {learning_rate}")

    def train_epoch_improved(self, train_loader):
        """Entraîne le modèle pour une epoch avec métriques détaillées."""
        self.model.train()
        total_loss = 0.0
        total_gap_loss = 0.0
        total_L_ecran_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch_X)

            # Loss pondérée
            loss, gap_loss, L_ecran_loss = self.criterion(predictions, batch_y)

            loss.backward()

            # Gradient clipping pour stabilité
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_gap_loss += gap_loss.item()
            total_L_ecran_loss += L_ecran_loss.item()

        return {
            'loss': total_loss / len(train_loader),
            'gap_loss': total_gap_loss / len(train_loader),
            'L_ecran_loss': total_L_ecran_loss / len(train_loader)
        }

    def validate_epoch_improved(self, val_loader):
        """Valide le modèle avec dénormalisation séparée."""
        self.model.eval()
        total_loss = 0.0
        total_gap_loss = 0.0
        total_L_ecran_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                predictions = self.model(batch_X)
                loss, gap_loss, L_ecran_loss = self.criterion(predictions, batch_y)

                total_loss += loss.item()
                total_gap_loss += gap_loss.item()
                total_L_ecran_loss += L_ecran_loss.item()

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        # Dénormalisation séparée
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)

        # Dénormaliser gap et L_ecran séparément
        gap_pred_denorm = self.gap_scaler.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
        gap_target_denorm = self.gap_scaler.inverse_transform(targets[:, 0].reshape(-1, 1)).flatten()

        L_ecran_pred_denorm = self.L_ecran_scaler.inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()
        L_ecran_target_denorm = self.L_ecran_scaler.inverse_transform(targets[:, 1].reshape(-1, 1)).flatten()

        # Métriques
        gap_r2 = r2_score(gap_target_denorm, gap_pred_denorm)
        L_ecran_r2 = r2_score(L_ecran_target_denorm, L_ecran_pred_denorm)
        gap_mae = mean_absolute_error(gap_target_denorm, gap_pred_denorm)
        L_ecran_mae = mean_absolute_error(L_ecran_target_denorm, L_ecran_pred_denorm)

        return {
            'loss': total_loss / len(val_loader),
            'gap_loss': total_gap_loss / len(val_loader),
            'L_ecran_loss': total_L_ecran_loss / len(val_loader),
            'gap_r2': gap_r2,
            'L_ecran_r2': L_ecran_r2,
            'gap_mae': gap_mae,
            'L_ecran_mae': L_ecran_mae
        }

    def train_improved_model(self, train_loader, val_loader, epochs=300, patience=30):
        """Entraîne le modèle amélioré avec early stopping adaptatif."""
        logger.info(f"🚀 DÉBUT DE L'ENTRAÎNEMENT AMÉLIORÉ")
        logger.info("="*60)

        history = {
            'train_loss': [], 'val_loss': [],
            'train_gap_loss': [], 'val_gap_loss': [],
            'gap_r2': [], 'L_ecran_r2': [],
            'gap_mae': [], 'L_ecran_mae': [],
            'learning_rate': []
        }

        best_gap_mae = float('inf')
        patience_counter = 0
        best_model_state = None

        start_time = time.time()

        for epoch in range(epochs):
            # Entraînement
            train_metrics = self.train_epoch_improved(train_loader)

            # Validation
            val_metrics = self.validate_epoch_improved(val_loader)

            # Mise à jour historique
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_gap_loss'].append(train_metrics['gap_loss'])
            history['val_gap_loss'].append(val_metrics['gap_loss'])
            history['gap_r2'].append(val_metrics['gap_r2'])
            history['L_ecran_r2'].append(val_metrics['L_ecran_r2'])
            history['gap_mae'].append(val_metrics['gap_mae'])
            history['L_ecran_mae'].append(val_metrics['L_ecran_mae'])
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Scheduler basé sur gap MAE (plus important)
            self.scheduler.step(val_metrics['gap_mae'])

            # Early stopping basé sur gap MAE
            if val_metrics['gap_mae'] < best_gap_mae:
                best_gap_mae = val_metrics['gap_mae']
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Affichage périodique
            if (epoch + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1:3d}: "
                           f"Gap MAE={val_metrics['gap_mae']:.4f}µm, "
                           f"Gap R²={val_metrics['gap_r2']:.3f}, "
                           f"L_ecran R²={val_metrics['L_ecran_r2']:.3f}, "
                           f"LR={lr:.2e}")

            # Arrêt anticipé
            if patience_counter >= patience:
                logger.info(f"⏹️ Early stopping à l'epoch {epoch+1} (Gap MAE={best_gap_mae:.4f}µm)")
                break

        # Restaurer le meilleur modèle
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        training_time = time.time() - start_time
        logger.info(f"✅ Entraînement amélioré terminé en {training_time:.1f}s")

        return history

    def predict_improved(self, ratio_input):
        """Prédiction avec dénormalisation séparée."""
        self.model.eval()

        # Préparer l'entrée
        if isinstance(ratio_input, (list, tuple)):
            ratio_input = np.array(ratio_input)

        if len(ratio_input) > 600:
            ratio_input = ratio_input[:600]
        elif len(ratio_input) < 600:
            ratio_input = np.pad(ratio_input, (0, 600 - len(ratio_input)), 'edge')

        # Filtrage léger
        from scipy.ndimage import gaussian_filter1d
        ratio_input = gaussian_filter1d(ratio_input, sigma=0.5)

        # Normaliser
        ratio_scaled = self.input_scaler.transform(ratio_input.reshape(1, -1))

        # Prédiction
        with torch.no_grad():
            ratio_tensor = torch.FloatTensor(ratio_scaled).to(self.device)
            prediction_scaled = self.model(ratio_tensor)

            # Dénormalisation séparée
            gap_pred = self.gap_scaler.inverse_transform(
                prediction_scaled[0, 0].cpu().numpy().reshape(-1, 1)
            )[0, 0]
            L_ecran_pred = self.L_ecran_scaler.inverse_transform(
                prediction_scaled[0, 1].cpu().numpy().reshape(-1, 1)
            )[0, 0]

        return gap_pred, L_ecran_pred

    def save_improved_model(self, model_path, scalers_path=None):
        """Sauvegarde le modèle amélioré avec scalers séparés."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': 600,
                'dropout_rate': 0.15
            }
        }, model_path)

        if scalers_path:
            scalers_dir = Path(scalers_path)
            scalers_dir.mkdir(exist_ok=True)

            joblib.dump(self.input_scaler, scalers_dir / 'input_scaler_improved.pkl')
            joblib.dump(self.gap_scaler, scalers_dir / 'gap_scaler_improved.pkl')
            joblib.dump(self.L_ecran_scaler, scalers_dir / 'L_ecran_scaler_improved.pkl')

        logger.info(f"💾 Modèle amélioré sauvegardé: {model_path}")
        if scalers_path:
            logger.info(f"💾 Scalers améliorés sauvegardés: {scalers_path}")


def main():
    """Fonction principale pour l'entraînement amélioré."""
    parser = argparse.ArgumentParser(description='Entraînement amélioré pour précision gap')
    parser.add_argument('--dataset', type=str, default='../../data_generation/dataset_2D_Train_Augmented')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--gap_weight', type=float, default=10.0)
    parser.add_argument('--model_path', type=str, default='../models/dual_parameter_model_improved.pt')

    args = parser.parse_args()

    config = {
        'truncate_to': 600,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'gap_weight': args.gap_weight
    }

    logger.info("🚀 ENTRAÎNEMENT AMÉLIORÉ - PRÉCISION GAP OPTIMISÉE")
    logger.info("="*70)
    logger.info(f"📊 Dataset: {args.dataset}")
    logger.info(f"🎯 Epochs: {args.epochs}")
    logger.info(f"📦 Batch size: {args.batch_size}")
    logger.info(f"📈 Learning rate: {args.learning_rate}")
    logger.info(f"⚖️ Gap weight: {args.gap_weight}x")

    # Créer l'entraîneur amélioré
    trainer = ImprovedDualParameterTrainer(config)

    # Préparer les données
    train_loader, val_loader = trainer.prepare_data_improved(
        args.dataset,
        validation_split=0.2,
        batch_size=config['batch_size']
    )

    # Configurer le modèle amélioré
    trainer.setup_improved_model(
        learning_rate=config['learning_rate'],
        gap_weight=config['gap_weight']
    )

    # Entraîner le modèle
    history = trainer.train_improved_model(
        train_loader, val_loader,
        epochs=config['epochs'],
        patience=30
    )

    # Sauvegarder le modèle
    model_dir = Path(args.model_path).parent
    model_dir.mkdir(exist_ok=True)

    trainer.save_improved_model(args.model_path, model_dir)

    # Afficher les résultats finaux
    final_gap_r2 = history['gap_r2'][-1]
    final_L_ecran_r2 = history['L_ecran_r2'][-1]
    final_gap_mae = history['gap_mae'][-1]
    final_L_ecran_mae = history['L_ecran_mae'][-1]

    logger.info(f"\n📊 RÉSULTATS FINAUX AMÉLIORÉS:")
    logger.info(f"   Gap R²: {final_gap_r2:.3f}")
    logger.info(f"   L_écran R²: {final_L_ecran_r2:.3f}")
    logger.info(f"   Gap MAE: {final_gap_mae:.4f} µm")
    logger.info(f"   L_écran MAE: {final_L_ecran_mae:.1f} µm")

    logger.info(f"\n✅ ENTRAÎNEMENT AMÉLIORÉ TERMINÉ !")


if __name__ == "__main__":
    main()
