#!/usr/bin/env python3
"""
Entraînement Ultra-Rapide avec TOUS les 22,540 profils
Auteur: Oussama GUELFAA
Date: 15/07/2025

Script d'entraînement utilisant le data loader ultra-rapide
avec TOUTES les données du fichier unique.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
from pathlib import Path
import json

import sys
sys.path.append('../../utils/data_loaders')
from ultra_fast_data_loader import UltraFastDataLoader

class ResidualBlock(nn.Module):
    """Bloc résiduel avec connexions skip."""

    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()

        self.main_path = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )

        # Connexion résiduelle
        if in_features != out_features:
            self.skip_connection = nn.Linear(in_features, out_features)
        else:
            self.skip_connection = nn.Identity()

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.skip_connection(x)
        out = self.main_path(x)
        out = out + residual
        out = self.activation(out)
        out = self.dropout(out)
        return out

class UltraDeepNetwork(nn.Module):
    """
    Réseau de neurones ULTRA-PROFOND avec architecture sophistiquée.
    """

    def __init__(self, input_size=601, output_size=2, dropout=0.3):
        super().__init__()

        # Architecture ULTRA-PROFONDE avec connexions résiduelles
        self.input_layer = nn.Linear(input_size, 1024)
        self.input_bn = nn.BatchNorm1d(1024)

        # Blocs résiduels profonds
        self.deep_blocks = nn.ModuleList([
            self._make_residual_block(1024, 1024, dropout),
            self._make_residual_block(1024, 512, dropout),
            self._make_residual_block(512, 512, dropout),
            self._make_residual_block(512, 256, dropout),
            self._make_residual_block(256, 256, dropout),
            self._make_residual_block(256, 128, dropout),
            self._make_residual_block(128, 128, dropout),
            self._make_residual_block(128, 64, dropout),
        ])

        # Couches finales avec attention
        self.attention = nn.MultiheadAttention(64, num_heads=8, dropout=dropout, batch_first=True)
        self.final_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, output_size)
        )

        # Initialisation des poids
        self._initialize_weights()

    def _make_residual_block(self, in_features, out_features, dropout):
        """Crée un bloc résiduel."""
        return ResidualBlock(in_features, out_features, dropout)

    def _initialize_weights(self):
        """Initialise les poids du réseau."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _initialize_weights(self):
        """Initialise les poids du réseau."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Couche d'entrée
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = torch.relu(x)

        # Blocs résiduels profonds
        for block in self.deep_blocks:
            x = block(x)

        # Attention mechanism
        x_unsqueezed = x.unsqueeze(1)  # (batch, 1, features)
        attn_out, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        x = attn_out.squeeze(1)  # (batch, features)

        # Couches finales
        x = self.final_layers(x)

        return x

class WeightedMSELoss(nn.Module):
    """Loss pondérée pour prioriser la prédiction du gap."""
    
    def __init__(self, gap_weight=3.0, L_ecran_weight=1.0):
        super().__init__()
        self.gap_weight = gap_weight
        self.L_ecran_weight = L_ecran_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # Séparer gap et L_écran
        gap_pred, L_ecran_pred = predictions[:, 0], predictions[:, 1]
        gap_true, L_ecran_true = targets[:, 0], targets[:, 1]
        
        # Calculer les losses séparément
        gap_loss = self.mse(gap_pred, gap_true)
        L_ecran_loss = self.mse(L_ecran_pred, L_ecran_true)
        
        # Loss pondérée
        total_loss = self.gap_weight * gap_loss + self.L_ecran_weight * L_ecran_loss
        
        return total_loss, gap_loss, L_ecran_loss

class UltraFastTrainer:
    """
    Entraîneur ultra-rapide pour TOUS les profils.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Créer le dossier de résultats
        self.results_dir = Path("results") / config['experiment_name']
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"⚡ UltraFastTrainer initialisé")
        print(f"   🖥️ Device: {self.device}")
        print(f"   📁 Résultats: {self.results_dir}")
    
    def setup_data(self):
        """Configure les données ultra-rapidement."""
        print("⚡ Configuration ultra-rapide des données...")
        
        # Créer le data loader ultra-rapide avec le bon chemin
        self.data_loader = UltraFastDataLoader("../../data/processed/extracted_data_full.npz")
        
        # Pipeline complet ultra-rapide
        self.data_loaders, self.normalized_splits = self.data_loader.get_full_pipeline(
            sample_ratio=self.config['training']['sample_ratio'],
            batch_size=self.config['training']['batch_size'],
            test_size=self.config['training']['test_size'],
            val_size=self.config['training']['val_size']
        )
        
        print("✅ Données configurées ultra-rapidement !")
    
    def setup_model(self):
        """Configure le modèle."""
        print("🧠 Configuration du modèle...")
        
        # Créer le modèle ULTRA-PROFOND
        self.model = UltraDeepNetwork(
            input_size=601,  # Points après troncature
            output_size=2,
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # Loss function pondérée
        self.criterion = WeightedMSELoss(
            gap_weight=self.config['training']['gap_weight'],
            L_ecran_weight=self.config['training']['L_ecran_weight']
        )
        
        # Optimiseur
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['training']['epochs'], 
            eta_min=1e-6
        )
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   📊 Paramètres: {total_params:,}")
        
        # Historique
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_gap_loss': [], 'val_gap_loss': [],
            'train_L_ecran_loss': [], 'val_L_ecran_loss': [],
            'epochs': []
        }
        
        print("✅ Modèle configuré")
    
    def calculate_r2_score(self, predictions, targets):
        """Calcule le score R²."""
        # Pas de dénormalisation car on utilise les données brutes
        pred_raw = predictions.cpu().numpy()
        target_raw = targets.cpu().numpy()

        # R² pour gap et L_écran
        from sklearn.metrics import r2_score
        gap_r2 = r2_score(target_raw[:, 0], pred_raw[:, 0])
        L_ecran_r2 = r2_score(target_raw[:, 1], pred_raw[:, 1])

        return gap_r2, L_ecran_r2
    
    def train_epoch(self):
        """Entraîne pour une époque."""
        self.model.train()
        total_loss = 0.0
        total_gap_loss = 0.0
        total_L_ecran_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        for data, target in self.data_loaders['train']:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            loss, gap_loss, L_ecran_loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_gap_loss += gap_loss.item()
            total_L_ecran_loss += L_ecran_loss.item()
            num_batches += 1
            
            all_predictions.append(output.detach())
            all_targets.append(target.detach())
        
        # Calculer R²
        all_pred = torch.cat(all_predictions, dim=0)
        all_targ = torch.cat(all_targets, dim=0)
        gap_r2, L_ecran_r2 = self.calculate_r2_score(all_pred, all_targ)
        
        return {
            'loss': total_loss / num_batches,
            'gap_loss': total_gap_loss / num_batches,
            'L_ecran_loss': total_L_ecran_loss / num_batches,
            'gap_r2': gap_r2,
            'L_ecran_r2': L_ecran_r2
        }
    
    def validate_epoch(self):
        """Valide pour une époque."""
        self.model.eval()
        total_loss = 0.0
        total_gap_loss = 0.0
        total_L_ecran_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.data_loaders['val']:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss, gap_loss, L_ecran_loss = self.criterion(output, target)
                
                total_loss += loss.item()
                total_gap_loss += gap_loss.item()
                total_L_ecran_loss += L_ecran_loss.item()
                num_batches += 1
                
                all_predictions.append(output)
                all_targets.append(target)
        
        # Calculer R²
        all_pred = torch.cat(all_predictions, dim=0)
        all_targ = torch.cat(all_targets, dim=0)
        gap_r2, L_ecran_r2 = self.calculate_r2_score(all_pred, all_targ)
        
        return {
            'loss': total_loss / num_batches,
            'gap_loss': total_gap_loss / num_batches,
            'L_ecran_loss': total_L_ecran_loss / num_batches,
            'gap_r2': gap_r2,
            'L_ecran_r2': L_ecran_r2
        }
    
    def train(self):
        """Entraînement complet."""
        print(f"🚀 Début de l'entraînement: {self.config['training']['epochs']} époques")
        print(f"   📊 Données d'entraînement: {len(self.data_loaders['train'].dataset)} profils")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['training']['early_stopping_patience']
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            epoch_start = time.time()
            
            # Entraînement et validation
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            
            # Mise à jour du scheduler
            self.scheduler.step()
            
            # Historique
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_gap_loss'].append(train_metrics['gap_loss'])
            self.history['val_gap_loss'].append(val_metrics['gap_loss'])
            self.history['train_L_ecran_loss'].append(train_metrics['L_ecran_loss'])
            self.history['val_L_ecran_loss'].append(val_metrics['L_ecran_loss'])
            self.history['epochs'].append(epoch + 1)
            
            # Affichage
            epoch_time = time.time() - epoch_start
            print(f"Époque {epoch+1}/{self.config['training']['epochs']} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.6f}, Gap R²: {train_metrics['gap_r2']:.4f}, L_écran R²: {train_metrics['L_ecran_r2']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.6f}, Gap R²: {val_metrics['gap_r2']:.4f}, L_écran R²: {val_metrics['L_ecran_r2']:.4f}")
            
            # Sauvegarde du meilleur modèle
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_metrics': val_metrics,
                    'config': self.config
                }, self.results_dir / 'best_model.pt')
                
                print(f"  ✅ Nouveau meilleur modèle sauvegardé!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏹️ Early stopping après {epoch+1} époques")
                break
        
        total_time = time.time() - start_time
        print(f"✅ Entraînement terminé en {total_time/60:.1f} minutes")
        
        # Test final et sauvegarde
        self.test_final()
        self.save_results()
    
    def test_final(self):
        """Test final sur le test set."""
        print("🧪 Test final...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.data_loaders['test']:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                all_predictions.append(output)
                all_targets.append(target)
        
        all_pred = torch.cat(all_predictions, dim=0)
        all_targ = torch.cat(all_targets, dim=0)
        
        gap_r2, L_ecran_r2 = self.calculate_r2_score(all_pred, all_targ)
        
        test_results = {
            'gap_r2': gap_r2,
            'L_ecran_r2': L_ecran_r2,
            'overall_r2': (gap_r2 + L_ecran_r2) / 2
        }
        
        print(f"📊 Résultats finaux:")
        print(f"   Gap R²: {gap_r2:.4f}")
        print(f"   L_écran R²: {L_ecran_r2:.4f}")
        print(f"   R² Global: {test_results['overall_r2']:.4f}")
        
        # Sauvegarder les résultats
        with open(self.results_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return test_results
    
    def save_results(self):
        """Sauvegarde les résultats."""
        print("💾 Sauvegarde des résultats...")
        
        # Sauvegarder l'historique
        np.savez(
            self.results_dir / 'training_history.npz',
            **self.history
        )
        
        # Générer les graphiques
        self.plot_training_curves()
        
        print(f"✅ Résultats sauvegardés dans {self.results_dir}")
    
    def plot_training_curves(self):
        """Génère les courbes d'entraînement."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = self.history['epochs']
        
        # Loss totale
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss Totale')
        axes[0, 0].set_xlabel('Époque')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Gap Loss
        axes[0, 1].plot(epochs, self.history['train_gap_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_gap_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Gap Loss')
        axes[0, 1].set_xlabel('Époque')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # L_écran Loss
        axes[1, 0].plot(epochs, self.history['train_L_ecran_loss'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_L_ecran_loss'], 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('L_écran Loss')
        axes[1, 0].set_xlabel('Époque')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Placeholder pour futurs graphiques
        axes[1, 1].text(0.5, 0.5, 'Entraînement\nUltra-Rapide\nTerminé!', 
                       ha='center', va='center', fontsize=16, fontweight='bold')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Entraînement Ultra-Rapide - {self.config["experiment_name"]}', fontsize=16)
        plt.tight_layout()
        
        plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   📈 Courbes d'entraînement sauvegardées")

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Entraînement Ultra-Rapide')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='Ratio d\'échantillonnage (1.0 = tous)')
    parser.add_argument('--epochs', type=int, default=200, help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=64, help='Taille des batches')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--experiment_name', type=str, default='ultra_fast_full_training', help='Nom de l\'expérience')
    
    args = parser.parse_args()
    
    # Configuration pour modèle ULTRA-PROFOND
    config = {
        'model': {
            'dropout': 0.3  # Plus de régularisation pour modèle profond
        },
        'training': {
            'sample_ratio': args.sample_ratio,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': 1e-4,
            'gap_weight': 3.0,  # Priorité sur gap
            'L_ecran_weight': 1.0,
            'early_stopping_patience': 20,
            'test_size': 0.2,
            'val_size': 0.15
        },
        'experiment_name': args.experiment_name
    }
    
    print("⚡ Entraînement Ultra-Rapide avec TOUS les Profils")
    print("=" * 60)
    print(f"📊 Configuration:")
    print(f"   🔢 Échantillon: {args.sample_ratio*100:.0f}% des données")
    print(f"   🔢 Époques: {args.epochs}")
    print(f"   🔢 Batch size: {args.batch_size}")
    print(f"   🔢 Learning rate: {args.lr}")
    
    # Créer l'entraîneur
    trainer = UltraFastTrainer(config)
    
    # Configurer les données
    trainer.setup_data()
    
    # Configurer le modèle
    trainer.setup_model()
    
    # Entraîner
    trainer.train()
    
    print("🎉 Entraînement ultra-rapide terminé avec succès !")

if __name__ == "__main__":
    main()
