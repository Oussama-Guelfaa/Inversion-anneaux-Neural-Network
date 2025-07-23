#!/usr/bin/env python3
"""
Entraînement simplifié mais efficace du réseau ultra-profond
Auteur: Oussama GUELFAA
Date: 15/07/2025

Version optimisée pour éviter les problèmes de mémoire tout en gardant
la sophistication de l'architecture.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import json

# Imports des modules
from data_loader import AdvancedDataLoader
from advanced_neural_network import AdvancedHybridNetwork
from advanced_training import WeightedMSELoss, AdvancedMetrics

class SimplifiedUltraTrainer:
    """Entraîneur simplifié mais ultra-efficace"""
    
    def __init__(self, experiment_name="ultra_deep_training"):
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Créer le dossier de résultats
        self.results_dir = f"results/{experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🧠 SimplifiedUltraTrainer initialisé")
        print(f"   🖥️ Device: {self.device}")
        print(f"   📁 Résultats: {self.results_dir}")
    
    def load_data(self, sample_ratio=1.0):
        """Charge et prépare les données"""
        print(f"📂 Chargement de TOUTES les données ({sample_ratio*100:.0f}% = 22,542 profils)...")

        # Chargeur de données
        loader = AdvancedDataLoader(
            train_dir="Train",
            preprocessed_data_path="preprocessed_data.npz"
        )

        # Charger TOUTES les données
        X_data, y_data, filenames = loader.load_all_training_data(sample_ratio=sample_ratio)
        
        # Division
        datasets = loader.create_train_val_test_split()
        
        # Normalisation
        normalized_data = loader.normalize_data()
        
        # Stocker les scalers
        self.scaler_X = loader.scaler_X
        self.scaler_y = loader.scaler_y
        
        # Créer les DataLoaders
        batch_size = 32  # Taille optimisée pour l'entraînement complet
        
        self.train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(normalized_data['X_train_norm']),
                torch.FloatTensor(normalized_data['y_train_norm'])
            ),
            batch_size=batch_size, shuffle=True
        )
        
        self.val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(normalized_data['X_val_norm']),
                torch.FloatTensor(normalized_data['y_val_norm'])
            ),
            batch_size=batch_size, shuffle=False
        )
        
        self.test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(normalized_data['X_test_norm']),
                torch.FloatTensor(normalized_data['y_test_norm'])
            ),
            batch_size=batch_size, shuffle=False
        )
        
        print(f"✅ Données chargées:")
        print(f"   📊 Train: {len(self.train_loader.dataset)} échantillons")
        print(f"   📊 Validation: {len(self.val_loader.dataset)} échantillons")
        print(f"   📊 Test: {len(self.test_loader.dataset)} échantillons")
    
    def create_model(self):
        """Crée le modèle ultra-profond optimisé"""
        print("🧠 Création du modèle ultra-profond...")
        
        # Modèle ULTRA-PROFOND pour performance maximale
        self.model = AdvancedHybridNetwork(
            input_size=601,
            output_size=2,
            base_channels=96,  # Maximum pour performance ultime
            num_encoder_blocks=12,  # ULTRA-PROFOND
            num_heads=12,  # Attention multi-têtes maximale
            dropout=0.2,  # Régularisation forte
            use_positional_encoding=True
        ).to(self.device)
        
        # Loss pondérée avec priorité sur gap
        self.loss_function = WeightedMSELoss(gap_weight=3.0, L_ecran_weight=1.0)
        
        # Optimiseur AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-4,  # Learning rate optimisé
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Scheduler cosine annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-6
        )
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Modèle créé: {total_params:,} paramètres")
        
        # Historique
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_gap_r2': [], 'val_gap_r2': [],
            'train_L_ecran_r2': [], 'val_L_ecran_r2': [],
            'val_gap_tolerance_007um': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self):
        """Entraîne pour une époque"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            loss, gap_loss, L_ecran_loss = self.loss_function(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.append(output.detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
        
        # Calcul des métriques
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Dénormaliser
        predictions_denorm = self.scaler_y.inverse_transform(predictions)
        targets_denorm = self.scaler_y.inverse_transform(targets)
        
        metrics = AdvancedMetrics.calculate_metrics(predictions_denorm, targets_denorm)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate_epoch(self):
        """Valide pour une époque"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss, gap_loss, L_ecran_loss = self.loss_function(output, target)
                total_loss += loss.item()
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        predictions_denorm = self.scaler_y.inverse_transform(predictions)
        targets_denorm = self.scaler_y.inverse_transform(targets)
        
        metrics = AdvancedMetrics.calculate_metrics(predictions_denorm, targets_denorm)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(self, epochs=200):
        """Entraînement complet"""
        print(f"🚀 Début de l'entraînement: {epochs} époques")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Entraînement et validation
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            
            # Mise à jour du scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Mise à jour de l'historique
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_gap_r2'].append(train_metrics['gap_r2'])
            self.history['val_gap_r2'].append(val_metrics['gap_r2'])
            self.history['train_L_ecran_r2'].append(train_metrics['L_ecran_r2'])
            self.history['val_L_ecran_r2'].append(val_metrics['L_ecran_r2'])
            self.history['val_gap_tolerance_007um'].append(val_metrics['gap_tolerance_0.007um'])
            self.history['learning_rates'].append(current_lr)
            
            # Affichage
            epoch_time = time.time() - epoch_start
            print(f"Époque {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.6f}, Gap R²: {train_metrics['gap_r2']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.6f}, Gap R²: {val_metrics['gap_r2']:.4f}")
            print(f"  Gap ±0.007µm: {val_metrics['gap_tolerance_0.007um']:.1f}%, LR: {current_lr:.2e}")
            
            # Sauvegarde du meilleur modèle
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_metrics': val_metrics
                }, os.path.join(self.results_dir, 'best_model.pt'))
                
                print(f"  ✅ Nouveau meilleur modèle sauvegardé!")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= 30:
                print(f"⏹️ Early stopping après {epoch+1} époques")
                break
            
            # Visualisation périodique
            if (epoch + 1) % 20 == 0:
                self.plot_training_curves()
        
        total_time = time.time() - start_time
        print(f"✅ Entraînement terminé en {total_time/3600:.2f}h")
        
        # Test final
        self.test_model()
        self.plot_training_curves()
    
    def test_model(self):
        """Test final"""
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
        
        print(f"📊 Résultats finaux:")
        print(f"   Gap R²: {test_metrics['gap_r2']:.4f}")
        print(f"   L_écran R²: {test_metrics['L_ecran_r2']:.4f}")
        print(f"   Gap ±0.007µm: {test_metrics['gap_tolerance_0.007um']:.1f}%")
        print(f"   L_écran ±0.5µm: {test_metrics['L_ecran_tolerance_0.5um']:.1f}%")
        
        # Sauvegarder les résultats
        with open(os.path.join(self.results_dir, 'final_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        return test_metrics
    
    def plot_training_curves(self):
        """Génère les courbes d'entraînement"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Époque')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Gap R²
        axes[0, 1].plot(epochs, self.history['train_gap_r2'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['val_gap_r2'], 'r-', label='Validation')
        axes[0, 1].set_title('Gap R²')
        axes[0, 1].set_xlabel('Époque')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].axhline(y=0.8, color='g', linestyle='--', alpha=0.7)
        
        # L_écran R²
        axes[1, 0].plot(epochs, self.history['train_L_ecran_r2'], 'b-', label='Train')
        axes[1, 0].plot(epochs, self.history['val_L_ecran_r2'], 'r-', label='Validation')
        axes[1, 0].set_title('L_écran R²')
        axes[1, 0].set_xlabel('Époque')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].axhline(y=0.8, color='g', linestyle='--', alpha=0.7)
        
        # Gap Tolérance
        axes[1, 1].plot(epochs, self.history['val_gap_tolerance_007um'], 'g-', linewidth=2)
        axes[1, 1].set_title('Gap Précision ±0.007µm')
        axes[1, 1].set_xlabel('Époque')
        axes[1, 1].set_ylabel('Précision (%)')
        axes[1, 1].grid(True)
        axes[1, 1].axhline(y=80, color='r', linestyle='--', alpha=0.7)
        
        plt.suptitle(f'Entraînement Ultra-Profond - {self.experiment_name}', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Courbes sauvegardées: {save_path}")
        plt.close()

def main():
    """Fonction principale"""
    print("🧠 Entraînement Ultra-Profond Simplifié")
    print("=" * 50)
    
    # Créer l'entraîneur
    trainer = SimplifiedUltraTrainer("ultra_deep_precision_training")
    
    # Charger TOUTES les données (100% = 22,542 profils)
    trainer.load_data(sample_ratio=1.0)
    
    # Créer le modèle
    trainer.create_model()
    
    # Entraîner avec MAXIMUM d'époques pour performance ultime
    trainer.train(epochs=500)
    
    print(f"\n🎉 Entraînement ultra-profond terminé!")
    print(f"📁 Résultats dans: {trainer.results_dir}")

if __name__ == "__main__":
    main()
