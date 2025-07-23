#!/usr/bin/env python3
"""
Entraînement avec Architecture Spécialisée pour le Gap
Auteur: Oussama GUELFAA
Date: 18/07/2025

Réseau dual avec branches séparées pour atteindre R² = 0.9 sur le gap.
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

class GapSpecializedBranch(nn.Module):
    """Branche spécialisée pour la prédiction du gap."""
    
    def __init__(self, input_size=601, hidden_sizes=[1024, 512, 256, 128, 64]):
        super().__init__()
        
        # Couches spécialisées pour le gap
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)  # Dropout plus faible pour préserver l'information
            ])
            prev_size = hidden_size
        
        # Couche finale pour le gap
        layers.append(nn.Linear(prev_size, 1))
        
        self.gap_network = nn.Sequential(*layers)
        
        # Initialisation spéciale pour le gap
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation spéciale pour la prédiction du gap."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier pour les couches cachées
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialisation spéciale pour la dernière couche (gap)
        final_layer = self.gap_network[-1]
        nn.init.normal_(final_layer.weight, mean=0, std=0.01)
        nn.init.constant_(final_layer.bias, 0.115)  # Biais vers gap expérimental
    
    def forward(self, x):
        return self.gap_network(x)

class LEcranSpecializedBranch(nn.Module):
    """Branche spécialisée pour la prédiction de L_écran."""
    
    def __init__(self, input_size=601, hidden_sizes=[512, 256, 128, 64]):
        super().__init__()
        
        # Couches pour L_écran (plus simple car plus facile)
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        # Couche finale pour L_écran
        layers.append(nn.Linear(prev_size, 1))
        
        self.L_ecran_network = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation pour L_écran."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Biais vers L_écran expérimental
        final_layer = self.L_ecran_network[-1]
        nn.init.constant_(final_layer.bias, 10.3)

    def forward(self, x):
        return self.L_ecran_network(x)

class DualSpecializedNetwork(nn.Module):
    """Réseau dual avec branches spécialisées gap et L_écran."""
    
    def __init__(self, input_size=601):
        super().__init__()
        
        # Couche d'extraction de features communes
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Branches spécialisées
        self.gap_branch = GapSpecializedBranch(512, [512, 256, 128, 64, 32])
        self.L_ecran_branch = LEcranSpecializedBranch(512, [256, 128, 64])
        
        print("🧠 Réseau Dual Spécialisé initialisé")
        print(f"   🎯 Branche Gap: 5 couches (512→32→1)")
        print(f"   🎯 Branche L_écran: 3 couches (256→64→1)")
    
    def forward(self, x):
        # Extraction de features communes
        features = self.feature_extractor(x)
        
        # Prédictions spécialisées
        gap_pred = self.gap_branch(features)
        L_ecran_pred = self.L_ecran_branch(features)
        
        # Combiner les sorties
        return torch.cat([gap_pred, L_ecran_pred], dim=1)

class SpecializedLoss(nn.Module):
    """Loss function spécialisée avec pondération forte pour le gap."""
    
    def __init__(self, gap_weight=50.0, L_ecran_weight=1.0):
        super().__init__()
        self.gap_weight = gap_weight
        self.L_ecran_weight = L_ecran_weight
        self.mse = nn.MSELoss()
        
        print(f"🎯 Loss Spécialisée: Gap x{gap_weight}, L_écran x{L_ecran_weight}")
    
    def forward(self, predictions, targets):
        gap_pred, L_ecran_pred = predictions[:, 0], predictions[:, 1]
        gap_true, L_ecran_true = targets[:, 0], targets[:, 1]
        
        # Losses séparées
        gap_loss = self.mse(gap_pred, gap_true)
        L_ecran_loss = self.mse(L_ecran_pred, L_ecran_true)
        
        # Loss pondérée
        total_loss = (self.gap_weight * gap_loss + 
                     self.L_ecran_weight * L_ecran_loss)
        
        return total_loss, gap_loss, L_ecran_loss

class SpecializedGapTrainer:
    """Entraîneur spécialisé pour le gap."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Créer le dossier de résultats
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("🎯 ENTRAÎNEUR SPÉCIALISÉ POUR LE GAP")
        print("=" * 50)
        print(f"   🖥️ Device: {self.device}")
        print(f"   📁 Résultats: {self.results_dir}")
        print(f"   🎯 Objectif: R² Gap = 0.9, précision ±0.01 µm")
    
    def setup_data(self):
        """Configure les données."""
        print("⚡ Configuration des données...")
        
        self.data_loader = UltraFastDataLoader("../../data/processed/extracted_data_full.npz")
        
        # Pipeline avec augmentation ciblée pour le gap
        self.data_loaders, self.normalized_splits = self.data_loader.get_full_pipeline(
            sample_ratio=self.config['training']['sample_ratio'],
            batch_size=self.config['training']['batch_size'],
            test_size=0.15,  # Plus de données pour validation
            val_size=0.15
        )
        
        print("✅ Données configurées !")
    
    def setup_model(self):
        """Configure le modèle spécialisé."""
        print("🧠 Configuration du modèle spécialisé...")
        
        self.model = DualSpecializedNetwork().to(self.device)
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        gap_params = sum(p.numel() for p in self.model.gap_branch.parameters())
        L_ecran_params = sum(p.numel() for p in self.model.L_ecran_branch.parameters())
        
        print(f"   📊 Paramètres totaux: {total_params:,}")
        print(f"   🎯 Paramètres Gap: {gap_params:,}")
        print(f"   🎯 Paramètres L_écran: {L_ecran_params:,}")
        
        # Loss function spécialisée
        self.criterion = SpecializedLoss(
            gap_weight=self.config['training']['gap_weight'],
            L_ecran_weight=self.config['training']['L_ecran_weight']
        )
        
        # Optimiseur avec learning rates différentiels
        gap_params = list(self.model.gap_branch.parameters())
        L_ecran_params = list(self.model.L_ecran_branch.parameters())
        feature_params = list(self.model.feature_extractor.parameters())
        
        self.optimizer = optim.Adam([
            {'params': gap_params, 'lr': self.config['training']['gap_lr']},
            {'params': L_ecran_params, 'lr': self.config['training']['L_ecran_lr']},
            {'params': feature_params, 'lr': self.config['training']['feature_lr']}
        ], weight_decay=1e-6)
        
        # Scheduler adaptatif
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-7
        )
        
        print("✅ Modèle spécialisé configuré")
    
    def calculate_r2_score(self, predictions, targets):
        """Calcule le score R² pour gap et L_écran."""
        pred_raw = predictions.cpu().numpy()
        target_raw = targets.cpu().numpy()
        
        from sklearn.metrics import r2_score
        gap_r2 = r2_score(target_raw[:, 0], pred_raw[:, 0])
        L_ecran_r2 = r2_score(target_raw[:, 1], pred_raw[:, 1])
        
        return gap_r2, L_ecran_r2
    
    def calculate_precision(self, predictions, targets, tolerance=0.01):
        """Calcule la précision (% dans la tolérance)."""
        pred_raw = predictions.cpu().numpy()
        target_raw = targets.cpu().numpy()
        
        gap_errors = np.abs(pred_raw[:, 0] - target_raw[:, 0])
        L_ecran_errors = np.abs(pred_raw[:, 1] - target_raw[:, 1])
        
        gap_precision = np.mean(gap_errors <= tolerance) * 100
        L_ecran_precision = np.mean(L_ecran_errors <= tolerance) * 100
        
        return gap_precision, L_ecran_precision
    
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
            
            # Gradient clipping plus agressif pour le gap
            torch.nn.utils.clip_grad_norm_(self.model.gap_branch.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.model.L_ecran_branch.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_gap_loss += gap_loss.item()
            total_L_ecran_loss += L_ecran_loss.item()
            num_batches += 1
            
            all_predictions.append(output.detach())
            all_targets.append(target.detach())
        
        # Calculer R² et précision
        all_pred = torch.cat(all_predictions, dim=0)
        all_targ = torch.cat(all_targets, dim=0)
        gap_r2, L_ecran_r2 = self.calculate_r2_score(all_pred, all_targ)
        gap_prec, L_ecran_prec = self.calculate_precision(all_pred, all_targ)
        
        return {
            'loss': total_loss / num_batches,
            'gap_loss': total_gap_loss / num_batches,
            'L_ecran_loss': total_L_ecran_loss / num_batches,
            'gap_r2': gap_r2,
            'L_ecran_r2': L_ecran_r2,
            'gap_precision': gap_prec,
            'L_ecran_precision': L_ecran_prec
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
        
        # Calculer R² et précision
        all_pred = torch.cat(all_predictions, dim=0)
        all_targ = torch.cat(all_targets, dim=0)
        gap_r2, L_ecran_r2 = self.calculate_r2_score(all_pred, all_targ)
        gap_prec, L_ecran_prec = self.calculate_precision(all_pred, all_targ)
        
        return {
            'loss': total_loss / num_batches,
            'gap_loss': total_gap_loss / num_batches,
            'L_ecran_loss': total_L_ecran_loss / num_batches,
            'gap_r2': gap_r2,
            'L_ecran_r2': L_ecran_r2,
            'gap_precision': gap_prec,
            'L_ecran_precision': L_ecran_prec
        }
    
    def train(self):
        """Entraînement complet spécialisé."""
        print(f"🚀 Début de l'entraînement spécialisé: {self.config['training']['epochs']} époques")
        
        best_gap_r2 = -float('inf')
        best_gap_precision = 0.0
        patience_counter = 0
        patience = self.config['training']['early_stopping_patience']
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_gap_r2': [], 'val_gap_r2': [],
            'train_L_ecran_r2': [], 'val_L_ecran_r2': [],
            'train_gap_precision': [], 'val_gap_precision': [],
            'train_L_ecran_precision': [], 'val_L_ecran_precision': []
        }
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['epochs']):
            epoch_start = time.time()
            
            # Entraînement
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Sauvegarder l'historique
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_gap_r2'].append(train_metrics['gap_r2'])
            history['val_gap_r2'].append(val_metrics['gap_r2'])
            history['train_L_ecran_r2'].append(train_metrics['L_ecran_r2'])
            history['val_L_ecran_r2'].append(val_metrics['L_ecran_r2'])
            history['train_gap_precision'].append(train_metrics['gap_precision'])
            history['val_gap_precision'].append(val_metrics['gap_precision'])
            history['train_L_ecran_precision'].append(train_metrics['L_ecran_precision'])
            history['val_L_ecran_precision'].append(val_metrics['L_ecran_precision'])
            
            epoch_time = time.time() - epoch_start
            
            # Affichage détaillé
            print(f"Époque {epoch+1}/{self.config['training']['epochs']} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.6f}, Gap R²: {train_metrics['gap_r2']:.4f}, L_écran R²: {train_metrics['L_ecran_r2']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.6f}, Gap R²: {val_metrics['gap_r2']:.4f}, L_écran R²: {val_metrics['L_ecran_r2']:.4f}")
            print(f"  Précision Gap: {val_metrics['gap_precision']:.1f}%, L_écran: {val_metrics['L_ecran_precision']:.1f}%")
            
            # Critère d'amélioration: priorité au gap
            gap_improvement = val_metrics['gap_r2'] > best_gap_r2
            precision_improvement = val_metrics['gap_precision'] > best_gap_precision
            
            if gap_improvement or (val_metrics['gap_r2'] > 0.8 and precision_improvement):
                best_gap_r2 = val_metrics['gap_r2']
                best_gap_precision = val_metrics['gap_precision']
                patience_counter = 0
                
                # Sauvegarder le meilleur modèle
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'gap_r2': val_metrics['gap_r2'],
                    'L_ecran_r2': val_metrics['L_ecran_r2'],
                    'gap_precision': val_metrics['gap_precision']
                }, self.results_dir / 'best_specialized_model.pt')
                
                print(f"  ✅ Nouveau meilleur modèle sauvegardé! (Gap R²: {val_metrics['gap_r2']:.4f})")
            else:
                patience_counter += 1
            
            # Objectif atteint ?
            if val_metrics['gap_r2'] >= 0.9 and val_metrics['gap_precision'] >= 90:
                print(f"  🎯 OBJECTIF ATTEINT! Gap R²: {val_metrics['gap_r2']:.4f}, Précision: {val_metrics['gap_precision']:.1f}%")
                break
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  ⏹️  Early stopping (patience: {patience})")
                break
        
        total_time = time.time() - start_time
        print(f"✅ Entraînement terminé en {total_time/60:.1f} minutes")
        
        return history

def main():
    """Fonction principale."""
    
    # Configuration spécialisée
    config = {
        'training': {
            'epochs': 200,
            'batch_size': 32,  # Plus petit pour plus de stabilité
            'sample_ratio': 1.0,
            'gap_weight': 100.0,  # Poids très élevé pour le gap
            'L_ecran_weight': 1.0,
            'gap_lr': 0.001,      # LR plus élevé pour le gap
            'L_ecran_lr': 0.0005, # LR plus faible pour L_écran
            'feature_lr': 0.0005, # LR modéré pour les features
            'early_stopping_patience': 30
        },
        'results_dir': 'results/specialized_gap_training'
    }
    
    print("🎯 ENTRAÎNEMENT SPÉCIALISÉ POUR LE GAP")
    print("=" * 60)
    print("🎯 Objectif: R² Gap = 0.9, précision ±0.01 µm")
    
    trainer = SpecializedGapTrainer(config)
    
    # Setup
    trainer.setup_data()
    trainer.setup_model()
    
    # Entraînement
    history = trainer.train()
    
    print("🎉 Entraînement spécialisé terminé!")

if __name__ == "__main__":
    main()
