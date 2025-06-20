#!/usr/bin/env python3
"""
Entraîneur pour Réseau Dual Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Ce module gère l'entraînement complet du réseau de neurones
pour prédiction conjointe gap + L_ecran.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

from dual_parameter_model import (
    DualParameterPredictor, DualParameterLoss, 
    DualParameterMetrics, DualParameterVisualizer
)
from data_loader import DualDataLoader

class EarlyStopping:
    """
    Early stopping pour éviter l'overfitting.
    """
    
    def __init__(self, patience=30, min_delta=0.0001, restore_best_weights=True):
        """
        Initialise l'early stopping.
        
        Args:
            patience (int): Nombre d'epochs sans amélioration
            min_delta (float): Amélioration minimale considérée
            restore_best_weights (bool): Restaurer les meilleurs poids
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        """
        Vérifie si l'entraînement doit s'arrêter.
        
        Args:
            val_loss (float): Loss de validation actuelle
            model (nn.Module): Modèle à surveiller
        
        Returns:
            bool: True si l'entraînement doit s'arrêter
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False

class DualParameterTrainer:
    """
    Classe principale pour l'entraînement du modèle dual.
    """
    
    def __init__(self, config_path="../config/dual_prediction_config.yaml"):
        """
        Initialise l'entraîneur.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration
        """
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Créer les dossiers de sortie
        self.create_output_directories()
        
        # Initialiser les composants
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.metrics_calculator = None
        self.visualizer = None
        
        print(f"🔧 DualParameterTrainer initialisé")
        print(f"💻 Device: {self.device}")
        print(f"📁 Config: {config_path}")
    
    def load_config(self, config_path):
        """Charge la configuration depuis le fichier YAML."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def create_output_directories(self):
        """Crée les dossiers de sortie nécessaires."""
        for dir_name in ['models', 'plots', 'results', 'logs', 'data']:
            Path(dir_name).mkdir(exist_ok=True)
    
    def setup_model_and_training(self):
        """Configure le modèle et les composants d'entraînement."""
        print(f"\n🏗️ Configuration du modèle et de l'entraînement...")
        
        # Modèle
        input_size = self.config['model']['input_size']
        # Extraire le dropout rate de la première couche dense
        first_layer = self.config['architecture']['dense_layers'][0]
        dropout_rate = first_layer.get('dropout', 0.2)
        
        self.model = DualParameterPredictor(
            input_size=input_size, 
            dropout_rate=dropout_rate
        ).to(self.device)
        
        # Loss function
        loss_weights = self.config.get('dual_output', {}).get('loss_weights', {})
        gap_weight = loss_weights.get('gap_weight', 1.0)
        L_ecran_weight = loss_weights.get('L_ecran_weight', 1.0)
        
        self.criterion = DualParameterLoss(gap_weight, L_ecran_weight)
        
        # Optimiseur
        learning_rate = float(self.config['training']['learning_rate'])
        weight_decay = float(self.config['training'].get('weight_decay', 1e-4))
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Scheduler
        scheduler_config = self.config['training']['scheduler']
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=scheduler_config['mode'],
            factor=float(scheduler_config['factor']),
            patience=int(scheduler_config['patience']),
            min_lr=float(scheduler_config['min_lr'])
        )
        
        # Métriques et visualisation
        tolerance_config = self.config['evaluation']['tolerance']
        self.metrics_calculator = DualParameterMetrics(
            gap_tolerance=tolerance_config['gap_tolerance'],
            L_ecran_tolerance=tolerance_config['L_ecran_tolerance']
        )
        
        self.visualizer = DualParameterVisualizer(save_dir="../plots/")
        
        print(f"✅ Configuration terminée")
        print(f"   Modèle: {sum(p.numel() for p in self.model.parameters()):,} paramètres")
        print(f"   Optimiseur: Adam (lr={learning_rate}, wd={weight_decay})")
        print(f"   Loss weights: Gap={gap_weight}, L_ecran={L_ecran_weight}")
    
    def train_epoch(self, train_loader):
        """
        Entraîne le modèle pour une epoch.
        
        Args:
            train_loader: DataLoader d'entraînement
        
        Returns:
            dict: Métriques de l'epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_gap_loss = 0.0
        total_L_ecran_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_X)
            
            # Calculer la loss
            total_loss_batch, gap_loss, L_ecran_loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()
            
            # Accumuler les métriques
            total_loss += total_loss_batch.item()
            total_gap_loss += gap_loss.item()
            total_L_ecran_loss += L_ecran_loss.item()
            
            # Stocker pour métriques
            all_predictions.append(predictions.detach().cpu().numpy())
            all_targets.append(batch_y.detach().cpu().numpy())
        
        # Calculer les métriques moyennes
        avg_loss = total_loss / len(train_loader)
        avg_gap_loss = total_gap_loss / len(train_loader)
        avg_L_ecran_loss = total_L_ecran_loss / len(train_loader)
        
        # Métriques détaillées
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        detailed_metrics = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        
        return {
            'loss': avg_loss,
            'gap_loss': avg_gap_loss,
            'L_ecran_loss': avg_L_ecran_loss,
            'gap_r2': detailed_metrics['gap_r2'],
            'L_ecran_r2': detailed_metrics['L_ecran_r2'],
            'combined_r2': detailed_metrics['combined_r2']
        }
    
    def validate_epoch(self, val_loader):
        """
        Valide le modèle pour une epoch.
        
        Args:
            val_loader: DataLoader de validation
        
        Returns:
            dict: Métriques de validation
        """
        self.model.eval()
        
        total_loss = 0.0
        total_gap_loss = 0.0
        total_L_ecran_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_X)
                
                # Calculer la loss
                total_loss_batch, gap_loss, L_ecran_loss = self.criterion(predictions, batch_y)
                
                # Accumuler les métriques
                total_loss += total_loss_batch.item()
                total_gap_loss += gap_loss.item()
                total_L_ecran_loss += L_ecran_loss.item()
                
                # Stocker pour métriques
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        # Calculer les métriques moyennes
        avg_loss = total_loss / len(val_loader)
        avg_gap_loss = total_gap_loss / len(val_loader)
        avg_L_ecran_loss = total_L_ecran_loss / len(val_loader)
        
        # Métriques détaillées
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        detailed_metrics = self.metrics_calculator.calculate_metrics(all_predictions, all_targets)
        
        return {
            'loss': avg_loss,
            'gap_loss': avg_gap_loss,
            'L_ecran_loss': avg_L_ecran_loss,
            'gap_r2': detailed_metrics['gap_r2'],
            'L_ecran_r2': detailed_metrics['L_ecran_r2'],
            'combined_r2': detailed_metrics['combined_r2'],
            'predictions': all_predictions,
            'targets': all_targets,
            'detailed_metrics': detailed_metrics
        }
    
    def train_model(self, train_loader, val_loader):
        """
        Entraîne le modèle complet.
        
        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
        
        Returns:
            dict: Historique d'entraînement
        """
        print(f"\n🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print("="*50)
        
        # Configuration early stopping
        early_stopping_config = self.config['training']['early_stopping']
        early_stopping = EarlyStopping(
            patience=early_stopping_config['patience'],
            min_delta=early_stopping_config['min_delta'],
            restore_best_weights=early_stopping_config['restore_best_weights']
        )
        
        # Historique
        history = {
            'train_loss': [], 'val_loss': [],
            'train_gap_r2': [], 'val_gap_r2': [],
            'train_L_ecran_r2': [], 'val_L_ecran_r2': [],
            'learning_rate': []
        }
        
        max_epochs = self.config['training']['epochs']
        start_time = time.time()
        
        print(f"📈 Entraînement sur {self.device}")
        print(f"🎯 Objectifs: Gap Acc > 90%, L_ecran Acc > 90%, R² > 80%")
        
        for epoch in range(max_epochs):
            epoch_start = time.time()
            
            # Entraînement
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Mise à jour historique
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_gap_r2'].append(train_metrics['gap_r2'])
            history['val_gap_r2'].append(val_metrics['gap_r2'])
            history['train_L_ecran_r2'].append(train_metrics['L_ecran_r2'])
            history['val_L_ecran_r2'].append(val_metrics['L_ecran_r2'])
            history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Affichage périodique
            if (epoch + 1) % 10 == 0:
                epoch_time = time.time() - epoch_start
                print(f"   Epoch {epoch+1:3d}: "
                      f"Val Loss={val_metrics['loss']:.4f}, "
                      f"Gap R²={val_metrics['gap_r2']:.3f}, "
                      f"L_ecran R²={val_metrics['L_ecran_r2']:.3f} "
                      f"({epoch_time:.1f}s)")
            
            # Early stopping
            if early_stopping(val_metrics['loss'], self.model):
                print(f"   ⏹️ Early stopping à l'époque {epoch+1}")
                break
        
        training_time = time.time() - start_time
        final_epoch = epoch + 1
        
        print(f"\n✅ Entraînement terminé en {training_time:.1f}s ({final_epoch} epochs)")
        
        return history, final_epoch, training_time


def test_trainer():
    """
    Test rapide de l'entraîneur.
    """
    print("🧪 Test du DualParameterTrainer")
    print("="*40)
    
    try:
        trainer = DualParameterTrainer()
        trainer.setup_model_and_training()
        print("✅ Trainer initialisé avec succès")
    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    test_trainer()
