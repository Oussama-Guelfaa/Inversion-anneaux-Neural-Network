#!/usr/bin/env python3
"""
Entraînement Rapide avec OptimizedDataLoader
Auteur: Oussama GUELFAA
Date: 15/07/2025

Script d'entraînement utilisant le nouveau data loader optimisé
basé sur Reseau_Neural_2D_Gap_Lecran_25_06_25 avec troncature 200-800.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
import argparse
from pathlib import Path

from optimized_data_loader import OptimizedDataLoader

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class FastNeuralNetwork(nn.Module):
    """
    Réseau de neurones rapide et efficace pour la prédiction Gap/L_écran.
    """
    
    def __init__(self, input_size=601, hidden_sizes=[512, 256, 128, 64], output_size=2, dropout=0.2):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Couche de sortie
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise les poids du réseau."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class FastTrainer:
    """
    Entraîneur rapide et efficace.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Créer le dossier de résultats
        self.results_dir = Path("results") / config['experiment_name']
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 FastTrainer initialisé")
        logger.info(f"   🖥️ Device: {self.device}")
        logger.info(f"   📁 Résultats: {self.results_dir}")
    
    def setup_data(self):
        """Configure les données."""
        logger.info("📂 Configuration des données...")
        
        # Créer le data loader optimisé
        self.data_loader = OptimizedDataLoader(self.config)
        
        # Charger les données
        X, y = self.data_loader.load_dataset_from_directory(
            self.config['data']['train_dir'],
            sample_ratio=self.config['training']['sample_ratio']
        )
        
        # Créer les splits
        splits = self.data_loader.create_train_val_test_split(X, y)
        
        # Normaliser
        normalized_splits = self.data_loader.normalize_data(splits)
        
        # Créer les DataLoaders PyTorch
        self.data_loaders = self.data_loader.create_data_loaders(
            normalized_splits, 
            batch_size=self.config['training']['batch_size']
        )
        
        # Sauvegarder les scalers
        scaler_path = self.results_dir / "scalers.joblib"
        self.data_loader.save_scalers(str(scaler_path))
        
        logger.info("✅ Données configurées")
    
    def setup_model(self):
        """Configure le modèle."""
        logger.info("🧠 Configuration du modèle...")
        
        # Créer le modèle
        self.model = FastNeuralNetwork(
            input_size=self.config['preprocessing']['expected_points'],
            hidden_sizes=self.config['model']['hidden_sizes'],
            output_size=2,
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # Loss function avec pondération
        self.criterion = nn.MSELoss()
        
        # Optimiseur
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"   📊 Paramètres: {total_params:,}")
        
        # Historique
        self.history = {
            'train_loss': [], 'val_loss': [],
            'epochs': []
        }
        
        logger.info("✅ Modèle configuré")
    
    def train_epoch(self):
        """Entraîne pour une époque."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for data, target in self.data_loaders['train']:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self):
        """Valide pour une époque."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in self.data_loaders['val']:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self):
        """Entraînement complet."""
        logger.info(f"🚀 Début de l'entraînement: {self.config['training']['epochs']} époques")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['training']['early_stopping_patience']
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            epoch_start = time.time()
            
            # Entraînement et validation
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            # Mise à jour du scheduler
            self.scheduler.step(val_loss)
            
            # Historique
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epochs'].append(epoch + 1)
            
            # Affichage
            epoch_time = time.time() - epoch_start
            logger.info(f"Époque {epoch+1}/{self.config['training']['epochs']} ({epoch_time:.1f}s)")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss: {val_loss:.6f}")
            
            # Sauvegarde du meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, self.results_dir / 'best_model.pt')
                
                logger.info(f"  ✅ Nouveau meilleur modèle sauvegardé!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"⏹️ Early stopping après {epoch+1} époques")
                break
        
        total_time = time.time() - start_time
        logger.info(f"✅ Entraînement terminé en {total_time/60:.1f} minutes")
        
        # Sauvegarder l'historique et générer les graphiques
        self.save_results()
    
    def save_results(self):
        """Sauvegarde les résultats et génère les graphiques."""
        logger.info("💾 Sauvegarde des résultats...")
        
        # Sauvegarder l'historique
        np.savez(
            self.results_dir / 'training_history.npz',
            **self.history
        )
        
        # Générer les graphiques
        self.plot_training_curves()
        
        logger.info(f"✅ Résultats sauvegardés dans {self.results_dir}")
    
    def plot_training_curves(self):
        """Génère les courbes d'entraînement."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = self.history['epochs']
        
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax.set_title('Courbes d\'Entraînement')
        ax.set_xlabel('Époque')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("   📈 Courbes d'entraînement sauvegardées")

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Entraînement Rapide Neural Network')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='Ratio d\'échantillonnage')
    parser.add_argument('--epochs', type=int, default=100, help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille des batches')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--experiment_name', type=str, default='fast_training', help='Nom de l\'expérience')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'data': {
            'train_dir': 'Train',
            'test_dir': 'Test'
        },
        'preprocessing': {
            'truncate_start': 200,
            'truncate_end': 800,
            'expected_points': 601
        },
        'model': {
            'hidden_sizes': [512, 256, 128, 64],
            'dropout': 0.2
        },
        'training': {
            'sample_ratio': args.sample_ratio,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': 1e-4,
            'early_stopping_patience': 15,
            'test_size': 0.2,
            'val_size': 0.15,
            'random_state': 42
        },
        'experiment_name': args.experiment_name
    }
    
    logger.info("🧠 Fast Training avec OptimizedDataLoader")
    logger.info("=" * 60)
    
    # Créer l'entraîneur
    trainer = FastTrainer(config)
    
    # Configurer les données
    trainer.setup_data()
    
    # Configurer le modèle
    trainer.setup_model()
    
    # Entraîner
    trainer.train()
    
    logger.info("🎉 Entraînement terminé avec succès !")

if __name__ == "__main__":
    main()
