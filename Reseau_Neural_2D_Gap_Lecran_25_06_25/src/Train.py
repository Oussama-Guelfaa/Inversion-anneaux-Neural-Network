#!/usr/bin/env python3
"""
Script d'Entraînement pour Réseau Neural 2D Gap + L_écran

Auteur: Oussama GUELFAA
Date: 25/06/2025

Ce script entraîne un réseau de neurones pour prédire simultanément
les paramètres gap et L_écran à partir de profils d'intensité 2D.

Usage:
    python Train.py --ratio <profil_intensité> [options]
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
import yaml
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

class DualParameterNet(nn.Module):
    """
    Réseau de neurones pour prédiction dual Gap + L_écran.
    
    Architecture: 600 → 512 → 256 → 128 → 64 → 2
    """
    
    def __init__(self, input_size=600, dropout_rate=0.2):
        super(DualParameterNet, self).__init__()
        
        # Architecture dense progressive
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout_rate * 0.5)
        
        # Sortie dual: [gap, L_écran]
        self.fc_out = nn.Linear(64, 2)
        
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
        """
        Forward pass.
        
        Args:
            x: Profils d'intensité [batch_size, 600]
        
        Returns:
            Prédictions [batch_size, 2] où [:, 0] = gap, [:, 1] = L_écran
        """
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        # Sortie linéaire pour régression
        x = self.fc_out(x)
        
        return x

class DualParameterTrainer:
    """Classe principale pour l'entraînement du modèle dual."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Scalers
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
        # Modèle et optimiseur
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        logger.info(f"🔧 Trainer initialisé sur {self.device}")
    
    def load_dataset(self, dataset_path, max_files=None):
        """
        Charge le dataset depuis les fichiers .mat et labels.csv.
        
        Args:
            dataset_path: Chemin vers le dataset
            max_files: Limite le nombre de fichiers (None = tous)
        
        Returns:
            tuple: (X, y) données chargées
        """
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
            if i % 1000 == 0:
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
                    # Chercher d'autres clés possibles
                    possible_keys = [k for k in data.keys() if not k.startswith('__')]
                    if possible_keys:
                        ratio = data[possible_keys[0]].flatten()
                    else:
                        continue
                
                # Tronquer/padding à la taille désirée
                if len(ratio) > truncate_to:
                    ratio = ratio[:truncate_to]
                elif len(ratio) < truncate_to:
                    ratio = np.pad(ratio, (0, truncate_to - len(ratio)), 'edge')
                
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
    
    def prepare_data(self, dataset_path, validation_split=0.2, batch_size=32):
        """
        Prépare les données pour l'entraînement.
        
        Args:
            dataset_path: Chemin vers le dataset
            validation_split: Proportion pour validation
            batch_size: Taille des batches
        
        Returns:
            tuple: (train_loader, val_loader)
        """
        logger.info(f"🔄 Préparation des données...")
        
        # Charger les données
        X, y = self.load_dataset(dataset_path)
        
        # Division train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        logger.info(f"📊 Division: Train={len(X_train)}, Val={len(X_val)}")
        
        # Normalisation
        X_train_scaled = self.input_scaler.fit_transform(X_train)
        X_val_scaled = self.input_scaler.transform(X_val)
        
        y_train_scaled = self.output_scaler.fit_transform(y_train)
        y_val_scaled = self.output_scaler.transform(y_val)
        
        # Créer les datasets PyTorch
        train_dataset = HologramDataset(X_train_scaled, y_train_scaled)
        val_dataset = HologramDataset(X_val_scaled, y_val_scaled)
        
        # Créer les DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"✅ Données préparées (batch_size={batch_size})")
        
        return train_loader, val_loader

    def setup_model(self, input_size=600, learning_rate=0.001):
        """
        Configure le modèle et l'optimiseur.

        Args:
            input_size: Taille d'entrée
            learning_rate: Taux d'apprentissage
        """
        self.model = DualParameterNet(input_size=input_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"🏗️ Modèle créé: {total_params:,} paramètres")

    def train_epoch(self, train_loader):
        """Entraîne le modèle pour une epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate_epoch(self, val_loader):
        """Valide le modèle pour une epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)

                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        # Calculer les métriques
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)

        # Dénormaliser pour calcul des métriques
        pred_denorm = self.output_scaler.inverse_transform(predictions)
        target_denorm = self.output_scaler.inverse_transform(targets)

        # R² pour chaque paramètre
        gap_r2 = r2_score(target_denorm[:, 0], pred_denorm[:, 0])
        L_ecran_r2 = r2_score(target_denorm[:, 1], pred_denorm[:, 1])

        # MAE pour chaque paramètre
        gap_mae = mean_absolute_error(target_denorm[:, 0], pred_denorm[:, 0])
        L_ecran_mae = mean_absolute_error(target_denorm[:, 1], pred_denorm[:, 1])

        return {
            'loss': total_loss / len(val_loader),
            'gap_r2': gap_r2,
            'L_ecran_r2': L_ecran_r2,
            'gap_mae': gap_mae,
            'L_ecran_mae': L_ecran_mae
        }

    def train_model(self, train_loader, val_loader, epochs=200, patience=20):
        """
        Entraîne le modèle complet avec early stopping.

        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            epochs: Nombre maximum d'epochs
            patience: Patience pour early stopping

        Returns:
            dict: Historique d'entraînement
        """
        logger.info(f"🚀 DÉBUT DE L'ENTRAÎNEMENT")
        logger.info("="*50)

        history = {
            'train_loss': [], 'val_loss': [],
            'gap_r2': [], 'L_ecran_r2': [],
            'gap_mae': [], 'L_ecran_mae': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        start_time = time.time()

        for epoch in range(epochs):
            # Entraînement
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_metrics = self.validate_epoch(val_loader)

            # Mise à jour historique
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['gap_r2'].append(val_metrics['gap_r2'])
            history['L_ecran_r2'].append(val_metrics['L_ecran_r2'])
            history['gap_mae'].append(val_metrics['gap_mae'])
            history['L_ecran_mae'].append(val_metrics['L_ecran_mae'])

            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Affichage périodique
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d}: "
                           f"Val Loss={val_metrics['loss']:.4f}, "
                           f"Gap R²={val_metrics['gap_r2']:.3f}, "
                           f"L_ecran R²={val_metrics['L_ecran_r2']:.3f}")

            # Arrêt anticipé
            if patience_counter >= patience:
                logger.info(f"⏹️ Early stopping à l'epoch {epoch+1}")
                break

        # Restaurer le meilleur modèle
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        training_time = time.time() - start_time
        logger.info(f"✅ Entraînement terminé en {training_time:.1f}s")

        return history

    def save_model(self, model_path, scalers_path=None):
        """
        Sauvegarde le modèle et les scalers.

        Args:
            model_path: Chemin pour sauvegarder le modèle
            scalers_path: Chemin pour sauvegarder les scalers
        """
        # Sauvegarder le modèle
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': 600,
                'dropout_rate': 0.2
            }
        }, model_path)

        # Sauvegarder les scalers
        if scalers_path:
            scalers_dir = Path(scalers_path)
            scalers_dir.mkdir(exist_ok=True)

            joblib.dump(self.input_scaler, scalers_dir / 'input_scaler.pkl')
            joblib.dump(self.output_scaler, scalers_dir / 'output_scaler.pkl')

        logger.info(f"💾 Modèle sauvegardé: {model_path}")
        if scalers_path:
            logger.info(f"💾 Scalers sauvegardés: {scalers_path}")

    def predict(self, ratio_input):
        """
        Fait une prédiction sur un profil d'intensité.

        Args:
            ratio_input: Profil d'intensité (array ou liste)

        Returns:
            tuple: (gap_predicted, L_ecran_predicted)
        """
        self.model.eval()

        # Préparer l'entrée
        if isinstance(ratio_input, (list, tuple)):
            ratio_input = np.array(ratio_input)

        # Tronquer/padding si nécessaire
        if len(ratio_input) > 600:
            ratio_input = ratio_input[:600]
        elif len(ratio_input) < 600:
            ratio_input = np.pad(ratio_input, (0, 600 - len(ratio_input)), 'edge')

        # Normaliser
        ratio_scaled = self.input_scaler.transform(ratio_input.reshape(1, -1))

        # Prédiction
        with torch.no_grad():
            ratio_tensor = torch.FloatTensor(ratio_scaled).to(self.device)
            prediction_scaled = self.model(ratio_tensor)
            prediction = self.output_scaler.inverse_transform(prediction_scaled.cpu().numpy())

        gap_pred = prediction[0, 0]
        L_ecran_pred = prediction[0, 1]

        return gap_pred, L_ecran_pred


def load_config():
    """Charge la configuration par défaut."""
    return {
        'truncate_to': 600,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 200,
        'patience': 20,
        'validation_split': 0.2
    }


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description='Entraînement du réseau neural dual Gap + L_écran'
    )

    parser.add_argument(
        '--ratio',
        type=str,
        help='Profil d\'intensité pour prédiction (liste de valeurs séparées par des virgules)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='../data_generation/dataset_2D_Train_Augmented',
        help='Chemin vers le dataset d\'entraînement'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Nombre maximum d\'epochs'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Taille des batches'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Taux d\'apprentissage'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default='../models/dual_parameter_model.pt',
        help='Chemin pour sauvegarder le modèle'
    )

    parser.add_argument(
        '--predict_only',
        action='store_true',
        help='Mode prédiction uniquement (sans entraînement)'
    )

    return parser.parse_args()


def main():
    """
    Fonction principale du script d'entraînement.
    """
    # Parser les arguments
    args = parse_arguments()

    # Charger la configuration
    config = load_config()
    config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs
    })

    logger.info("🚀 SCRIPT D'ENTRAÎNEMENT RÉSEAU NEURAL 2D")
    logger.info("="*60)
    logger.info(f"📊 Dataset: {args.dataset}")
    logger.info(f"🎯 Epochs: {args.epochs}")
    logger.info(f"📦 Batch size: {args.batch_size}")
    logger.info(f"📈 Learning rate: {args.learning_rate}")

    # Créer l'entraîneur
    trainer = DualParameterTrainer(config)

    # Mode prédiction uniquement
    if args.predict_only and args.ratio:
        logger.info("\n🔮 MODE PRÉDICTION")

        # Charger le modèle existant
        if Path(args.model_path).exists():
            # Charger le modèle
            trainer.setup_model()
            checkpoint = torch.load(args.model_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])

            # Charger les scalers
            scalers_dir = Path(args.model_path).parent
            trainer.input_scaler = joblib.load(scalers_dir / 'input_scaler.pkl')
            trainer.output_scaler = joblib.load(scalers_dir / 'output_scaler.pkl')

            # Parser le profil d'intensité
            ratio_values = [float(x.strip()) for x in args.ratio.split(',')]

            # Faire la prédiction
            gap_pred, L_ecran_pred = trainer.predict(ratio_values)

            logger.info(f"✅ PRÉDICTION TERMINÉE:")
            logger.info(f"   Gap prédit: {gap_pred:.4f} µm")
            logger.info(f"   L_écran prédit: {L_ecran_pred:.1f} µm")

            return gap_pred, L_ecran_pred
        else:
            logger.error(f"❌ Modèle non trouvé: {args.model_path}")
            return None, None

    # Mode entraînement
    logger.info("\n🏋️ MODE ENTRAÎNEMENT")

    # Préparer les données
    train_loader, val_loader = trainer.prepare_data(
        args.dataset,
        validation_split=config['validation_split'],
        batch_size=config['batch_size']
    )

    # Configurer le modèle
    trainer.setup_model(learning_rate=config['learning_rate'])

    # Entraîner le modèle
    history = trainer.train_model(
        train_loader, val_loader,
        epochs=config['epochs'],
        patience=config['patience']
    )

    # Sauvegarder le modèle
    model_dir = Path(args.model_path).parent
    model_dir.mkdir(exist_ok=True)

    trainer.save_model(args.model_path, model_dir)

    # Afficher les résultats finaux
    final_gap_r2 = history['gap_r2'][-1]
    final_L_ecran_r2 = history['L_ecran_r2'][-1]
    final_gap_mae = history['gap_mae'][-1]
    final_L_ecran_mae = history['L_ecran_mae'][-1]

    logger.info(f"\n� RÉSULTATS FINAUX:")
    logger.info(f"   Gap R²: {final_gap_r2:.3f}")
    logger.info(f"   L_écran R²: {final_L_ecran_r2:.3f}")
    logger.info(f"   Gap MAE: {final_gap_mae:.4f} µm")
    logger.info(f"   L_écran MAE: {final_L_ecran_mae:.1f} µm")

    # Test de prédiction si un ratio est fourni
    if args.ratio:
        logger.info(f"\n🔮 TEST DE PRÉDICTION:")
        ratio_values = [float(x.strip()) for x in args.ratio.split(',')]
        gap_pred, L_ecran_pred = trainer.predict(ratio_values)

        logger.info(f"   Gap prédit: {gap_pred:.4f} µm")
        logger.info(f"   L_écran prédit: {L_ecran_pred:.1f} µm")

        return gap_pred, L_ecran_pred

    logger.info(f"\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    return None, None


if __name__ == "__main__":
    main()
