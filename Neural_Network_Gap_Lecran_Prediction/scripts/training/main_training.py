#!/usr/bin/env python3
"""
Script principal d'entraînement ultra-sophistiqué
Auteur: Oussama GUELFAA
Date: 15/07/2025

Ce script orchestre l'entraînement complet du réseau de neurones hybride
en combinant tous les composants avancés développés.

Fonctionnalités:
- Chargement et augmentation des données
- Architecture hybride multi-échelle
- Loss pondérée avec priorité sur gap
- Optimisation avancée avec schedulers
- Monitoring et visualisation en temps réel
- Sauvegarde automatique des meilleurs modèles
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Imports des modules développés
from data_loader import AdvancedDataLoader
from data_augmentation import AdvancedDataAugmentation
from advanced_neural_network import AdvancedHybridNetwork, create_model_variants
from advanced_training import (WeightedMSELoss, AdvancedMetrics, AdvancedOptimizer, 
                              TrainingConfig, create_loss_function)
from visualization_monitoring import AdvancedVisualizer

class UltraSophisticatedTrainer:
    """Entraîneur ultra-sophistiqué pour le réseau hybride"""
    
    def __init__(self, config: TrainingConfig, experiment_name: str = None):
        """
        Initialise l'entraîneur
        
        Args:
            config: Configuration d'entraînement
            experiment_name: Nom de l'expérience
        """
        self.config = config
        self.experiment_name = experiment_name or f"hybrid_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device utilisé: {self.device}")
        
        # Composants
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_function = None
        self.visualizer = None
        
        # Données
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.scaler_X = None
        self.scaler_y = None
        
        # Historique
        self.best_val_loss = float('inf')
        self.best_gap_r2 = 0.0
        self.patience_counter = 0
        
        print(f"🧠 UltraSophisticatedTrainer initialisé: {self.experiment_name}")
    
    def setup_data(self, sample_ratio: float = 0.1, use_augmentation: bool = True):
        """Configure les données d'entraînement"""
        print("📂 Configuration des données...")
        
        # Chargeur de données (chargement direct des fichiers .mat)
        data_loader = AdvancedDataLoader(
            train_dir="Train",
            preprocessed_data_path=None  # Pas de fichier preprocessed_data.npz requis
        )
        
        # Charger les données (échantillon pour test rapide)
        X_data, y_data, filenames = data_loader.load_all_training_data(sample_ratio=sample_ratio)
        
        # Division train/val/test
        datasets = data_loader.create_train_val_test_split()
        
        # Augmentation des données si demandée
        if use_augmentation:
            print("🔄 Augmentation des données d'entraînement...")
            augmenter = AdvancedDataAugmentation(augmentation_factor=3, noise_level=0.02)
            
            # Créer le vecteur radial
            r_radial = np.linspace(data_loader.r_min, data_loader.r_max, data_loader.n_points)
            
            # Augmenter seulement les données d'entraînement
            X_train_aug, y_train_aug = augmenter.augment_dataset(
                datasets['X_train'], datasets['y_train'], r_radial
            )
            datasets['X_train'] = X_train_aug
            datasets['y_train'] = y_train_aug
        
        # Normalisation
        normalized_data = data_loader.normalize_data()
        
        # Stocker les scalers
        self.scaler_X = data_loader.scaler_X
        self.scaler_y = data_loader.scaler_y
        
        # Créer les DataLoaders PyTorch
        train_dataset = TensorDataset(
            torch.FloatTensor(normalized_data['X_train_norm']),
            torch.FloatTensor(normalized_data['y_train_norm'])
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(normalized_data['X_val_norm']),
            torch.FloatTensor(normalized_data['y_val_norm'])
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(normalized_data['X_test_norm']),
            torch.FloatTensor(normalized_data['y_test_norm'])
        )
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, 
                                     shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, 
                                   shuffle=False, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, 
                                    shuffle=False, num_workers=2)
        
        print(f"✅ Données configurées:")
        print(f"   📊 Train: {len(train_dataset)} échantillons")
        print(f"   📊 Validation: {len(val_dataset)} échantillons")
        print(f"   📊 Test: {len(test_dataset)} échantillons")
    
    def setup_model(self, model_variant: str = 'standard'):
        """Configure le modèle et les composants d'entraînement"""
        print(f"🧠 Configuration du modèle '{model_variant}'...")
        
        # Créer le modèle
        models = create_model_variants()
        if model_variant not in models:
            raise ValueError(f"Variante de modèle non supportée: {model_variant}")
        
        self.model = models[model_variant].to(self.device)
        
        # Fonction de loss
        self.loss_function = create_loss_function(
            self.config.loss_type, 
            self.config.gap_weight, 
            self.config.L_ecran_weight
        )
        
        # Optimiseur
        self.optimizer = AdvancedOptimizer.create_optimizer(
            self.model, 
            self.config.optimizer_name,
            self.config.learning_rate,
            self.config.weight_decay
        )
        
        # Scheduler
        self.scheduler = AdvancedOptimizer.create_scheduler(
            self.optimizer,
            self.config.scheduler_name,
            self.config.epochs
        )
        
        # Visualiseur
        self.visualizer = AdvancedVisualizer(
            save_dir="results",
            experiment_name=self.experiment_name
        )
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Modèle configuré:")
        print(f"   🔧 Paramètres totaux: {total_params:,}")
        print(f"   🔧 Paramètres entraînables: {trainable_params:,}")
    
    def train_epoch(self) -> dict:
        """Entraîne le modèle pour une époque"""
        self.model.train()
        
        total_loss = 0.0
        total_gap_loss = 0.0
        total_L_ecran_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Calcul de la loss
            if self.config.loss_type == 'weighted_mse':
                loss, gap_loss, L_ecran_loss = self.loss_function(output, target)
            else:
                loss, gap_loss, L_ecran_loss = self.loss_function(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             self.config.gradient_clip_norm)
            
            self.optimizer.step()
            
            # Accumulation des métriques
            total_loss += loss.item()
            total_gap_loss += gap_loss.item()
            total_L_ecran_loss += L_ecran_loss.item()
            
            # Stocker pour calcul des métriques
            all_predictions.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
        
        # Calcul des métriques finales
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Dénormaliser pour calcul des métriques réelles
        predictions_denorm = self.scaler_y.inverse_transform(predictions)
        targets_denorm = self.scaler_y.inverse_transform(targets)
        
        metrics = AdvancedMetrics.calculate_metrics(predictions_denorm, targets_denorm)
        metrics['loss'] = total_loss / len(self.train_loader)
        metrics['gap_loss'] = total_gap_loss / len(self.train_loader)
        metrics['L_ecran_loss'] = total_L_ecran_loss / len(self.train_loader)
        
        return metrics
    
    def validate_epoch(self) -> dict:
        """Valide le modèle pour une époque"""
        self.model.eval()
        
        total_loss = 0.0
        total_gap_loss = 0.0
        total_L_ecran_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                
                if self.config.loss_type == 'weighted_mse':
                    loss, gap_loss, L_ecran_loss = self.loss_function(output, target)
                else:
                    loss, gap_loss, L_ecran_loss = self.loss_function(output, target)
                
                total_loss += loss.item()
                total_gap_loss += gap_loss.item()
                total_L_ecran_loss += L_ecran_loss.item()
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Calcul des métriques
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        predictions_denorm = self.scaler_y.inverse_transform(predictions)
        targets_denorm = self.scaler_y.inverse_transform(targets)
        
        metrics = AdvancedMetrics.calculate_metrics(predictions_denorm, targets_denorm)
        metrics['loss'] = total_loss / len(self.val_loader)
        metrics['gap_loss'] = total_gap_loss / len(self.val_loader)
        metrics['L_ecran_loss'] = total_L_ecran_loss / len(self.val_loader)
        
        return metrics
    
    def train(self):
        """Entraînement complet du modèle"""
        print(f"🚀 Début de l'entraînement: {self.config.epochs} époques")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Entraînement
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Mise à jour du scheduler
            if self.config.scheduler_name == 'plateau':
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Learning rate actuel
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Mise à jour du visualiseur
            self.visualizer.update_training_metrics(epoch, train_metrics, val_metrics, current_lr)
            
            # Affichage des métriques
            epoch_time = time.time() - epoch_start
            print(f"Époque {epoch+1}/{self.config.epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.6f}, Gap R²: {train_metrics['gap_r2']:.4f}, L_écran R²: {train_metrics['L_ecran_r2']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.6f}, Gap R²: {val_metrics['gap_r2']:.4f}, L_écran R²: {val_metrics['L_ecran_r2']:.4f}")
            print(f"  Gap Tolérance ±0.007µm: {val_metrics['gap_tolerance_0.007um']:.1f}%")
            
            # Sauvegarde du meilleur modèle
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_gap_r2 = val_metrics['gap_r2']
                self.patience_counter = 0
                
                if self.config.save_best_model:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'val_metrics': val_metrics,
                        'config': self.config.to_dict()
                    }, os.path.join(self.visualizer.experiment_dir, 'best_model.pt'))
                    
                print(f"  ✅ Nouveau meilleur modèle sauvegardé!")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"⏹️ Early stopping après {epoch+1} époques")
                break
            
            # Visualisation périodique
            if (epoch + 1) % self.config.plot_frequency == 0:
                self.visualizer.plot_training_curves(save=True, show=False)
        
        total_time = time.time() - start_time
        print(f"✅ Entraînement terminé en {total_time/3600:.2f}h")
        
        # Génération des visualisations finales
        self.visualizer.plot_training_curves(save=True, show=False)
        
        # Test final
        test_metrics = self.test_model()
        
        # Rapport final
        model_info = {
            'architecture': 'AdvancedHybridNetwork',
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'training_time_hours': total_time / 3600,
            'best_epoch': len(self.visualizer.epochs) - self.patience_counter
        }
        
        self.visualizer.generate_training_report(test_metrics, model_info)
        
        return test_metrics
    
    def test_model(self) -> dict:
        """Test final du modèle"""
        print("🧪 Test final du modèle...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        predictions_denorm = self.scaler_y.inverse_transform(predictions)
        targets_denorm = self.scaler_y.inverse_transform(targets)
        
        test_metrics = AdvancedMetrics.calculate_metrics(predictions_denorm, targets_denorm)
        
        # Visualisation des prédictions
        self.visualizer.plot_predictions_scatter(predictions_denorm, targets_denorm, 
                                               "Test", save=True, show=False)
        
        print(f"📊 Résultats finaux:")
        print(f"   Gap R²: {test_metrics['gap_r2']:.4f}")
        print(f"   L_écran R²: {test_metrics['L_ecran_r2']:.4f}")
        print(f"   Gap Tolérance ±0.007µm: {test_metrics['gap_tolerance_0.007um']:.1f}%")
        print(f"   L_écran Tolérance ±0.5µm: {test_metrics['L_ecran_tolerance_0.5um']:.1f}%")
        
        return test_metrics

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Entraînement ultra-sophistiqué du réseau hybride')
    parser.add_argument('--model', type=str, default='standard', 
                       choices=['lightweight', 'standard', 'heavy', 'ultra_deep'],
                       help='Variante du modèle à utiliser')
    parser.add_argument('--epochs', type=int, default=100, help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille du batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--sample_ratio', type=float, default=0.1, 
                       help='Ratio d\'échantillonnage des données (0.1 = 10%)')
    parser.add_argument('--no_augmentation', action='store_true', 
                       help='Désactiver l\'augmentation de données')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Nom de l\'expérience')
    
    args = parser.parse_args()
    
    print("🧠 Neural_Network_Gap_Lecran_Prediction - Entraînement Ultra-Sophistiqué")
    print("=" * 80)
    
    # Configuration
    config = TrainingConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    # Entraîneur
    trainer = UltraSophisticatedTrainer(config, args.experiment_name)
    
    # Configuration des données
    trainer.setup_data(sample_ratio=args.sample_ratio, 
                      use_augmentation=not args.no_augmentation)
    
    # Configuration du modèle
    trainer.setup_model(args.model)
    
    # Entraînement
    final_metrics = trainer.train()
    
    print(f"\n🎉 Entraînement terminé avec succès!")
    print(f"📁 Résultats sauvegardés dans: {trainer.visualizer.experiment_dir}")

if __name__ == "__main__":
    main()
