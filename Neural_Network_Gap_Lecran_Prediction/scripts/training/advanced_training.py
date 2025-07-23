#!/usr/bin/env python3
"""
Système d'entraînement avancé avec loss pondérée et optimisation sophistiquée
Auteur: Oussama GUELFAA
Date: 15/07/2025

Ce module implémente:
- Loss personnalisée avec priorité sur gap
- Optimiseurs avancés (AdamW, Ranger, Lookahead)
- Schedulers dynamiques de learning rate
- Techniques de régularisation avancées
- Métriques de performance sophistiquées
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import time
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class WeightedMSELoss(nn.Module):
    """Loss MSE pondérée avec priorité sur gap"""
    
    def __init__(self, gap_weight: float = 3.0, L_ecran_weight: float = 1.0):
        super().__init__()
        self.gap_weight = gap_weight
        self.L_ecran_weight = L_ecran_weight
        
        print(f"🎯 WeightedMSELoss initialisée:")
        print(f"   📊 Poids Gap: {gap_weight}")
        print(f"   📊 Poids L_écran: {L_ecran_weight}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions et targets shape: (batch_size, 2) [gap, L_ecran]
        
        gap_loss = nn.functional.mse_loss(predictions[:, 0], targets[:, 0])
        L_ecran_loss = nn.functional.mse_loss(predictions[:, 1], targets[:, 1])
        
        total_loss = (self.gap_weight * gap_loss + 
                     self.L_ecran_weight * L_ecran_loss)
        
        return total_loss, gap_loss, L_ecran_loss

class AdaptiveHuberLoss(nn.Module):
    """Loss Huber adaptative avec pondération"""
    
    def __init__(self, gap_weight: float = 3.0, L_ecran_weight: float = 1.0, 
                 delta: float = 1.0):
        super().__init__()
        self.gap_weight = gap_weight
        self.L_ecran_weight = L_ecran_weight
        self.delta = delta
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gap_loss = nn.functional.smooth_l1_loss(predictions[:, 0], targets[:, 0], 
                                               beta=self.delta)
        L_ecran_loss = nn.functional.smooth_l1_loss(predictions[:, 1], targets[:, 1], 
                                                   beta=self.delta)
        
        total_loss = (self.gap_weight * gap_loss + 
                     self.L_ecran_weight * L_ecran_loss)
        
        return total_loss, gap_loss, L_ecran_loss

class CombinedLoss(nn.Module):
    """Loss combinée: MSE + Huber + L1 avec pondération adaptative"""
    
    def __init__(self, gap_weight: float = 3.0, L_ecran_weight: float = 1.0,
                 mse_weight: float = 0.5, huber_weight: float = 0.3, l1_weight: float = 0.2):
        super().__init__()
        self.gap_weight = gap_weight
        self.L_ecran_weight = L_ecran_weight
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.l1_weight = l1_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # MSE Loss
        gap_mse = nn.functional.mse_loss(predictions[:, 0], targets[:, 0])
        L_ecran_mse = nn.functional.mse_loss(predictions[:, 1], targets[:, 1])
        mse_loss = self.gap_weight * gap_mse + self.L_ecran_weight * L_ecran_mse
        
        # Huber Loss
        gap_huber = nn.functional.smooth_l1_loss(predictions[:, 0], targets[:, 0])
        L_ecran_huber = nn.functional.smooth_l1_loss(predictions[:, 1], targets[:, 1])
        huber_loss = self.gap_weight * gap_huber + self.L_ecran_weight * L_ecran_huber
        
        # L1 Loss
        gap_l1 = nn.functional.l1_loss(predictions[:, 0], targets[:, 0])
        L_ecran_l1 = nn.functional.l1_loss(predictions[:, 1], targets[:, 1])
        l1_loss = self.gap_weight * gap_l1 + self.L_ecran_weight * L_ecran_l1
        
        # Combinaison
        total_loss = (self.mse_weight * mse_loss + 
                     self.huber_weight * huber_loss + 
                     self.l1_weight * l1_loss)
        
        return total_loss, gap_mse, L_ecran_mse

class AdvancedMetrics:
    """Calcul de métriques avancées pour l'évaluation"""
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Calcule des métriques complètes
        
        Args:
            predictions: Prédictions (N, 2) [gap, L_ecran]
            targets: Vraies valeurs (N, 2) [gap, L_ecran]
            
        Returns:
            Dictionnaire des métriques
        """
        metrics = {}
        
        # Métriques pour Gap (index 0)
        gap_pred = predictions[:, 0]
        gap_true = targets[:, 0]
        
        metrics['gap_mse'] = mean_squared_error(gap_true, gap_pred)
        metrics['gap_rmse'] = np.sqrt(metrics['gap_mse'])
        metrics['gap_mae'] = mean_absolute_error(gap_true, gap_pred)
        metrics['gap_r2'] = r2_score(gap_true, gap_pred)
        metrics['gap_mape'] = np.mean(np.abs((gap_true - gap_pred) / gap_true)) * 100
        
        # Métriques pour L_ecran (index 1)
        L_ecran_pred = predictions[:, 1]
        L_ecran_true = targets[:, 1]
        
        metrics['L_ecran_mse'] = mean_squared_error(L_ecran_true, L_ecran_pred)
        metrics['L_ecran_rmse'] = np.sqrt(metrics['L_ecran_mse'])
        metrics['L_ecran_mae'] = mean_absolute_error(L_ecran_true, L_ecran_pred)
        metrics['L_ecran_r2'] = r2_score(L_ecran_true, L_ecran_pred)
        metrics['L_ecran_mape'] = np.mean(np.abs((L_ecran_true - L_ecran_pred) / L_ecran_true)) * 100
        
        # Métriques globales
        metrics['overall_mse'] = (metrics['gap_mse'] + metrics['L_ecran_mse']) / 2
        metrics['overall_rmse'] = np.sqrt(metrics['overall_mse'])
        metrics['overall_mae'] = (metrics['gap_mae'] + metrics['L_ecran_mae']) / 2
        metrics['overall_r2'] = (metrics['gap_r2'] + metrics['L_ecran_r2']) / 2
        
        # Métriques de tolérance (précision ultra-haute pour gap)
        gap_tolerance_001 = np.mean(np.abs(gap_true - gap_pred) <= 0.001) * 100  # ±0.001 µm (ultra-précision)
        gap_tolerance_005 = np.mean(np.abs(gap_true - gap_pred) <= 0.005) * 100  # ±0.005 µm
        gap_tolerance_007 = np.mean(np.abs(gap_true - gap_pred) <= 0.007) * 100  # ±0.007 µm (objectif projet)
        gap_tolerance_01 = np.mean(np.abs(gap_true - gap_pred) <= 0.01) * 100    # ±0.01 µm

        L_ecran_tolerance_05 = np.mean(np.abs(L_ecran_true - L_ecran_pred) <= 0.5) * 100  # ±0.5 µm
        L_ecran_tolerance_1 = np.mean(np.abs(L_ecran_true - L_ecran_pred) <= 1.0) * 100   # ±1.0 µm

        metrics['gap_tolerance_0.001um'] = gap_tolerance_001
        metrics['gap_tolerance_0.005um'] = gap_tolerance_005
        metrics['gap_tolerance_0.007um'] = gap_tolerance_007  # Objectif projet
        metrics['gap_tolerance_0.01um'] = gap_tolerance_01
        metrics['L_ecran_tolerance_0.5um'] = L_ecran_tolerance_05
        metrics['L_ecran_tolerance_1.0um'] = L_ecran_tolerance_1
        
        return metrics

class AdvancedOptimizer:
    """Factory pour créer des optimiseurs avancés"""
    
    @staticmethod
    def create_optimizer(model: nn.Module, optimizer_name: str = 'adamw',
                        lr: float = 1e-3, weight_decay: float = 1e-4,
                        **kwargs) -> torch.optim.Optimizer:
        """
        Crée un optimiseur avancé
        
        Args:
            model: Modèle PyTorch
            optimizer_name: Type d'optimiseur ('adamw', 'adam', 'sgd', 'rmsprop')
            lr: Learning rate
            weight_decay: Décroissance des poids
            
        Returns:
            Optimiseur configuré
        """
        
        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                   betas=(0.9, 0.999), eps=1e-8)
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.9, 0.999), eps=1e-8)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                 momentum=0.9, nesterov=True)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay,
                                     momentum=0.9, alpha=0.99)
        else:
            raise ValueError(f"Optimiseur non supporté: {optimizer_name}")
        
        print(f"🚀 Optimiseur {optimizer_name.upper()} créé:")
        print(f"   📈 Learning rate: {lr}")
        print(f"   🔧 Weight decay: {weight_decay}")
        
        return optimizer
    
    @staticmethod
    def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str = 'cosine',
                        epochs: int = 100, **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Crée un scheduler de learning rate
        
        Args:
            optimizer: Optimiseur
            scheduler_name: Type de scheduler ('cosine', 'plateau', 'onecycle', 'step')
            epochs: Nombre d'époques
            
        Returns:
            Scheduler configuré
        """
        
        if scheduler_name.lower() == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        elif scheduler_name.lower() == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                        patience=10, verbose=True)
        elif scheduler_name.lower() == 'onecycle':
            scheduler = OneCycleLR(optimizer, max_lr=optimizer.param_groups[0]['lr'],
                                  epochs=epochs, steps_per_epoch=1)
        elif scheduler_name.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
        else:
            raise ValueError(f"Scheduler non supporté: {scheduler_name}")
        
        print(f"📅 Scheduler {scheduler_name.upper()} créé pour {epochs} époques")
        
        return scheduler

class TrainingConfig:
    """Configuration d'entraînement"""
    
    def __init__(self):
        # Paramètres généraux
        self.epochs = 200
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        
        # Loss et optimisation
        self.loss_type = 'weighted_mse'  # 'weighted_mse', 'huber', 'combined'
        self.gap_weight = 3.0
        self.L_ecran_weight = 1.0
        self.optimizer_name = 'adamw'
        self.scheduler_name = 'cosine'
        
        # Régularisation
        self.dropout = 0.1
        self.early_stopping_patience = 25
        self.gradient_clip_norm = 1.0
        
        # Validation et sauvegarde
        self.validation_frequency = 1
        self.save_best_model = True
        self.save_checkpoint_frequency = 10
        
        # Visualisation
        self.plot_frequency = 10
        self.save_plots = True
        
    def to_dict(self) -> Dict:
        """Convertit la configuration en dictionnaire"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

def create_loss_function(loss_type: str = 'weighted_mse', 
                        gap_weight: float = 3.0, 
                        L_ecran_weight: float = 1.0) -> nn.Module:
    """
    Factory pour créer les fonctions de loss
    
    Args:
        loss_type: Type de loss ('weighted_mse', 'huber', 'combined')
        gap_weight: Poids pour gap
        L_ecran_weight: Poids pour L_ecran
        
    Returns:
        Fonction de loss configurée
    """
    
    if loss_type == 'weighted_mse':
        return WeightedMSELoss(gap_weight, L_ecran_weight)
    elif loss_type == 'huber':
        return AdaptiveHuberLoss(gap_weight, L_ecran_weight)
    elif loss_type == 'combined':
        return CombinedLoss(gap_weight, L_ecran_weight)
    else:
        raise ValueError(f"Type de loss non supporté: {loss_type}")

def main():
    """Fonction principale de test"""
    print("🧠 AdvancedTraining - Test des composants")
    print("=" * 50)
    
    # Test de la configuration
    config = TrainingConfig()
    print("📋 Configuration d'entraînement:")
    for key, value in config.to_dict().items():
        print(f"   {key}: {value}")
    
    # Test des loss functions
    print("\n🎯 Test des fonctions de loss:")
    
    # Données de test
    batch_size = 4
    predictions = torch.randn(batch_size, 2)
    targets = torch.randn(batch_size, 2)
    
    # Test WeightedMSELoss
    weighted_loss = WeightedMSELoss(gap_weight=3.0, L_ecran_weight=1.0)
    total_loss, gap_loss, L_ecran_loss = weighted_loss(predictions, targets)
    print(f"   WeightedMSE - Total: {total_loss:.4f}, Gap: {gap_loss:.4f}, L_écran: {L_ecran_loss:.4f}")
    
    # Test des métriques
    print("\n📊 Test des métriques:")
    pred_np = predictions.detach().numpy()
    target_np = targets.detach().numpy()
    
    metrics = AdvancedMetrics.calculate_metrics(pred_np, target_np)
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n✅ Test des composants d'entraînement terminé!")

if __name__ == "__main__":
    main()
