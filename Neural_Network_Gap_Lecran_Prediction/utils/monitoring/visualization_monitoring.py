#!/usr/bin/env python3
"""
Système de visualisation et monitoring avancé
Auteur: Oussama GUELFAA
Date: 15/07/2025

Ce module implémente un système complet de visualisation et monitoring pour:
- Courbes de loss et métriques en temps réel
- Scatter plots des prédictions vs vraies valeurs
- Visualisation des poids d'attention
- Monitoring des gradients et activations
- Génération automatique de rapports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration matplotlib pour de beaux graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedVisualizer:
    """Système de visualisation avancé pour le monitoring d'entraînement"""
    
    def __init__(self, save_dir: str = "plots", experiment_name: str = None):
        """
        Initialise le visualiseur
        
        Args:
            save_dir: Dossier de sauvegarde des graphiques
            experiment_name: Nom de l'expérience
        """
        self.save_dir = save_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Créer le dossier de sauvegarde
        self.experiment_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Historique des métriques
        self.train_history = {
            'loss': [], 'gap_loss': [], 'L_ecran_loss': [],
            'gap_r2': [], 'L_ecran_r2': [], 'overall_r2': [],
            'gap_mae': [], 'L_ecran_mae': [],
            'gap_tolerance_007um': [], 'L_ecran_tolerance_05um': []
        }
        
        self.val_history = {
            'loss': [], 'gap_loss': [], 'L_ecran_loss': [],
            'gap_r2': [], 'L_ecran_r2': [], 'overall_r2': [],
            'gap_mae': [], 'L_ecran_mae': [],
            'gap_tolerance_007um': [], 'L_ecran_tolerance_05um': []
        }
        
        self.learning_rates = []
        self.epochs = []
        
        print(f"📊 AdvancedVisualizer initialisé:")
        print(f"   📁 Dossier: {self.experiment_dir}")
        print(f"   🧪 Expérience: {self.experiment_name}")
    
    def update_training_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict, 
                               learning_rate: float):
        """Met à jour l'historique des métriques d'entraînement"""
        self.epochs.append(epoch)
        self.learning_rates.append(learning_rate)
        
        # Métriques d'entraînement
        for key in self.train_history.keys():
            if key in train_metrics:
                self.train_history[key].append(train_metrics[key])
            else:
                self.train_history[key].append(0.0)
        
        # Métriques de validation
        for key in self.val_history.keys():
            if key in val_metrics:
                self.val_history[key].append(val_metrics[key])
            else:
                self.val_history[key].append(0.0)
    
    def plot_training_curves(self, save: bool = True, show: bool = False):
        """Génère les courbes d'entraînement complètes"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # 1. Loss curves
        axes[0, 0].plot(self.epochs, self.train_history['loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(self.epochs, self.val_history['loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss Totale', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Époque')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Gap Loss
        axes[0, 1].plot(self.epochs, self.train_history['gap_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(self.epochs, self.val_history['gap_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Loss Gap', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Époque')
        axes[0, 1].set_ylabel('Gap Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. L_ecran Loss
        axes[0, 2].plot(self.epochs, self.train_history['L_ecran_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 2].plot(self.epochs, self.val_history['L_ecran_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 2].set_title('Loss L_écran', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Époque')
        axes[0, 2].set_ylabel('L_écran Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. R² Scores
        axes[1, 0].plot(self.epochs, self.train_history['gap_r2'], 'b-', label='Gap Train', linewidth=2)
        axes[1, 0].plot(self.epochs, self.val_history['gap_r2'], 'r-', label='Gap Val', linewidth=2)
        axes[1, 0].plot(self.epochs, self.train_history['L_ecran_r2'], 'b--', label='L_écran Train', linewidth=2)
        axes[1, 0].plot(self.epochs, self.val_history['L_ecran_r2'], 'r--', label='L_écran Val', linewidth=2)
        axes[1, 0].set_title('Scores R²', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Époque')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.8, color='g', linestyle=':', alpha=0.7, label='Objectif 80%')
        
        # 5. MAE
        axes[1, 1].plot(self.epochs, self.train_history['gap_mae'], 'b-', label='Gap Train', linewidth=2)
        axes[1, 1].plot(self.epochs, self.val_history['gap_mae'], 'r-', label='Gap Val', linewidth=2)
        axes[1, 1].set_title('Mean Absolute Error - Gap', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Époque')
        axes[1, 1].set_ylabel('MAE (µm)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Tolérance Gap (±0.007 µm)
        axes[1, 2].plot(self.epochs, self.train_history['gap_tolerance_007um'], 'b-', label='Train', linewidth=2)
        axes[1, 2].plot(self.epochs, self.val_history['gap_tolerance_007um'], 'r-', label='Validation', linewidth=2)
        axes[1, 2].set_title('Précision Gap (±0.007µm)', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Époque')
        axes[1, 2].set_ylabel('Précision (%)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=80, color='g', linestyle=':', alpha=0.7, label='Objectif 80%')
        
        # 7. Learning Rate
        axes[2, 0].plot(self.epochs, self.learning_rates, 'g-', linewidth=2)
        axes[2, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Époque')
        axes[2, 0].set_ylabel('Learning Rate')
        axes[2, 0].set_yscale('log')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Overall R²
        axes[2, 1].plot(self.epochs, self.train_history['overall_r2'], 'b-', label='Train', linewidth=2)
        axes[2, 1].plot(self.epochs, self.val_history['overall_r2'], 'r-', label='Validation', linewidth=2)
        axes[2, 1].set_title('R² Global', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Époque')
        axes[2, 1].set_ylabel('R² Global')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].axhline(y=0.8, color='g', linestyle=':', alpha=0.7, label='Objectif 80%')
        
        # 9. Tolérance L_écran
        axes[2, 2].plot(self.epochs, self.train_history['L_ecran_tolerance_05um'], 'b-', label='Train', linewidth=2)
        axes[2, 2].plot(self.epochs, self.val_history['L_ecran_tolerance_05um'], 'r-', label='Validation', linewidth=2)
        axes[2, 2].set_title('Précision L_écran (±0.5µm)', fontsize=14, fontweight='bold')
        axes[2, 2].set_xlabel('Époque')
        axes[2, 2].set_ylabel('Précision (%)')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Courbes d\'Entraînement - {self.experiment_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.experiment_dir, 'training_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Courbes sauvegardées: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_predictions_scatter(self, predictions: np.ndarray, targets: np.ndarray,
                               dataset_name: str = "Test", save: bool = True, show: bool = False):
        """Génère les scatter plots des prédictions vs vraies valeurs"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Gap predictions
        gap_pred = predictions[:, 0]
        gap_true = targets[:, 0]
        
        axes[0].scatter(gap_true, gap_pred, alpha=0.6, s=30)
        axes[0].plot([gap_true.min(), gap_true.max()], [gap_true.min(), gap_true.max()], 
                    'r--', linewidth=2, label='Parfait')
        
        # Zones de tolérance
        gap_min, gap_max = gap_true.min(), gap_true.max()
        x_range = np.linspace(gap_min, gap_max, 100)
        axes[0].fill_between(x_range, x_range - 0.007, x_range + 0.007, 
                           alpha=0.2, color='green', label='±0.007µm')
        axes[0].fill_between(x_range, x_range - 0.01, x_range + 0.01, 
                           alpha=0.1, color='orange', label='±0.01µm')
        
        axes[0].set_xlabel('Gap Réel (µm)', fontsize=12)
        axes[0].set_ylabel('Gap Prédit (µm)', fontsize=12)
        axes[0].set_title(f'Prédictions Gap - {dataset_name}', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Calculer R²
        from sklearn.metrics import r2_score
        gap_r2 = r2_score(gap_true, gap_pred)
        axes[0].text(0.05, 0.95, f'R² = {gap_r2:.4f}', transform=axes[0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. L_ecran predictions
        L_ecran_pred = predictions[:, 1]
        L_ecran_true = targets[:, 1]
        
        axes[1].scatter(L_ecran_true, L_ecran_pred, alpha=0.6, s=30, color='orange')
        axes[1].plot([L_ecran_true.min(), L_ecran_true.max()], 
                    [L_ecran_true.min(), L_ecran_true.max()], 
                    'r--', linewidth=2, label='Parfait')
        
        # Zone de tolérance
        L_ecran_min, L_ecran_max = L_ecran_true.min(), L_ecran_true.max()
        x_range = np.linspace(L_ecran_min, L_ecran_max, 100)
        axes[1].fill_between(x_range, x_range - 0.5, x_range + 0.5, 
                           alpha=0.2, color='green', label='±0.5µm')
        
        axes[1].set_xlabel('L_écran Réel (µm)', fontsize=12)
        axes[1].set_ylabel('L_écran Prédit (µm)', fontsize=12)
        axes[1].set_title(f'Prédictions L_écran - {dataset_name}', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        L_ecran_r2 = r2_score(L_ecran_true, L_ecran_pred)
        axes[1].text(0.05, 0.95, f'R² = {L_ecran_r2:.4f}', transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 3. Erreurs relatives
        gap_errors = np.abs(gap_true - gap_pred)
        L_ecran_errors = np.abs(L_ecran_true - L_ecran_pred)
        
        axes[2].hist(gap_errors, bins=30, alpha=0.7, label='Gap', density=True)
        axes[2].hist(L_ecran_errors, bins=30, alpha=0.7, label='L_écran', density=True)
        axes[2].axvline(x=0.007, color='red', linestyle='--', label='Objectif Gap (0.007µm)')
        axes[2].axvline(x=0.5, color='orange', linestyle='--', label='Objectif L_écran (0.5µm)')
        axes[2].set_xlabel('Erreur Absolue (µm)', fontsize=12)
        axes[2].set_ylabel('Densité', fontsize=12)
        axes[2].set_title('Distribution des Erreurs', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.experiment_dir, f'predictions_scatter_{dataset_name.lower()}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Scatter plots sauvegardés: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_attention_weights(self, attention_weights: List[torch.Tensor], 
                              input_sequence: np.ndarray, save: bool = True, show: bool = False):
        """Visualise les poids d'attention"""
        n_layers = len(attention_weights)
        fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(4 * n_layers, 8))
        
        if n_layers == 1:
            axes = [axes]
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        for i, weights in enumerate(attention_weights):
            row = i // ((n_layers + 1) // 2)
            col = i % ((n_layers + 1) // 2)
            
            # Moyenner sur les têtes d'attention et le batch
            avg_weights = weights.mean(dim=(0, 1)).cpu().numpy()  # (seq_len, seq_len)
            
            im = axes[row, col].imshow(avg_weights, cmap='Blues', aspect='auto')
            axes[row, col].set_title(f'Attention Layer {i+1}', fontsize=12, fontweight='bold')
            axes[row, col].set_xlabel('Position Source')
            axes[row, col].set_ylabel('Position Cible')
            plt.colorbar(im, ax=axes[row, col])
        
        plt.suptitle('Poids d\'Attention par Couche', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.experiment_dir, 'attention_weights.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ Poids d'attention sauvegardés: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_training_report(self, final_metrics: Dict, model_info: Dict):
        """Génère un rapport complet d'entraînement"""
        report = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'model_info': model_info,
            'final_metrics': final_metrics,
            'training_summary': {
                'total_epochs': len(self.epochs),
                'best_val_loss': min(self.val_history['loss']) if self.val_history['loss'] else None,
                'best_gap_r2': max(self.val_history['gap_r2']) if self.val_history['gap_r2'] else None,
                'best_L_ecran_r2': max(self.val_history['L_ecran_r2']) if self.val_history['L_ecran_r2'] else None,
                'final_gap_tolerance_007um': self.val_history['gap_tolerance_007um'][-1] if self.val_history['gap_tolerance_007um'] else None
            }
        }
        
        # Sauvegarder le rapport JSON
        report_path = os.path.join(self.experiment_dir, 'training_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Rapport d'entraînement sauvegardé: {report_path}")
        
        return report

def main():
    """Fonction principale de test"""
    print("📊 AdvancedVisualizer - Test du système de visualisation")
    print("=" * 60)
    
    # Créer un visualiseur de test
    visualizer = AdvancedVisualizer(save_dir="test_plots", experiment_name="test_visualization")
    
    # Simuler des données d'entraînement
    n_epochs = 50
    for epoch in range(n_epochs):
        # Simuler des métriques qui s'améliorent
        train_metrics = {
            'loss': 1.0 - 0.8 * epoch / n_epochs + 0.1 * np.random.random(),
            'gap_loss': 0.5 - 0.4 * epoch / n_epochs + 0.05 * np.random.random(),
            'L_ecran_loss': 0.3 - 0.2 * epoch / n_epochs + 0.03 * np.random.random(),
            'gap_r2': 0.2 + 0.7 * epoch / n_epochs + 0.05 * np.random.random(),
            'L_ecran_r2': 0.3 + 0.6 * epoch / n_epochs + 0.05 * np.random.random(),
            'gap_tolerance_007um': 20 + 60 * epoch / n_epochs + 5 * np.random.random()
        }
        
        val_metrics = {
            'loss': 1.1 - 0.7 * epoch / n_epochs + 0.15 * np.random.random(),
            'gap_loss': 0.6 - 0.35 * epoch / n_epochs + 0.08 * np.random.random(),
            'L_ecran_loss': 0.35 - 0.15 * epoch / n_epochs + 0.05 * np.random.random(),
            'gap_r2': 0.15 + 0.65 * epoch / n_epochs + 0.08 * np.random.random(),
            'L_ecran_r2': 0.25 + 0.55 * epoch / n_epochs + 0.08 * np.random.random(),
            'gap_tolerance_007um': 15 + 55 * epoch / n_epochs + 8 * np.random.random()
        }
        
        # Calculer les métriques dérivées
        train_metrics['overall_r2'] = (train_metrics['gap_r2'] + train_metrics['L_ecran_r2']) / 2
        val_metrics['overall_r2'] = (val_metrics['gap_r2'] + val_metrics['L_ecran_r2']) / 2
        
        train_metrics['gap_mae'] = 0.02 - 0.015 * epoch / n_epochs + 0.002 * np.random.random()
        val_metrics['gap_mae'] = 0.025 - 0.018 * epoch / n_epochs + 0.003 * np.random.random()
        
        train_metrics['L_ecran_mae'] = 1.0 - 0.7 * epoch / n_epochs + 0.1 * np.random.random()
        val_metrics['L_ecran_mae'] = 1.2 - 0.8 * epoch / n_epochs + 0.15 * np.random.random()
        
        train_metrics['L_ecran_tolerance_05um'] = 60 + 30 * epoch / n_epochs + 5 * np.random.random()
        val_metrics['L_ecran_tolerance_05um'] = 55 + 35 * epoch / n_epochs + 8 * np.random.random()
        
        lr = 1e-3 * (0.1 ** (epoch / (n_epochs / 3)))
        
        visualizer.update_training_metrics(epoch, train_metrics, val_metrics, lr)
    
    # Générer les courbes d'entraînement
    visualizer.plot_training_curves(save=True, show=False)
    
    # Simuler des prédictions pour scatter plots
    n_samples = 1000
    predictions = np.random.randn(n_samples, 2) * 0.1 + np.array([0.15, 10.0])
    targets = np.random.randn(n_samples, 2) * 0.05 + np.array([0.15, 10.0])
    
    visualizer.plot_predictions_scatter(predictions, targets, "Test", save=True, show=False)
    
    # Générer un rapport
    final_metrics = {'gap_r2': 0.85, 'L_ecran_r2': 0.82, 'gap_tolerance_007um': 78.5}
    model_info = {'architecture': 'AdvancedHybridNetwork', 'parameters': 2500000}
    
    visualizer.generate_training_report(final_metrics, model_info)
    
    print("\n✅ Test du système de visualisation terminé!")
    print(f"📁 Fichiers générés dans: {visualizer.experiment_dir}")

if __name__ == "__main__":
    main()
