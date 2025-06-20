#!/usr/bin/env python3
"""
Modèle de Réseau de Neurones pour Prédiction Conjointe Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Ce module implémente un réseau de neurones robuste pour prédire
simultanément les paramètres gap et L_ecran à partir de profils
d'intensité holographiques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

class DualParameterPredictor(nn.Module):
    """
    Réseau de neurones pour prédiction conjointe gap + L_ecran.
    
    Architecture inspirée du modèle robuste gap-only mais adaptée
    pour sortie dual avec techniques de régularisation avancées.
    """
    
    def __init__(self, input_size=600, dropout_rate=0.2):
        """
        Initialise le modèle dual.
        
        Args:
            input_size (int): Taille d'entrée (600 points recommandé)
            dropout_rate (float): Taux de dropout pour régularisation
        """
        super(DualParameterPredictor, self).__init__()
        
        # Couche d'entrée - 512 neurones
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Couche cachée 1 - 256 neurones
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Couche cachée 2 - 128 neurones
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Couche cachée 3 - 64 neurones
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(dropout_rate * 0.5)  # Dropout réduit
        
        # Couche de sortie - 2 paramètres [gap, L_ecran]
        self.fc_out = nn.Linear(64, 2)
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialise les poids avec Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass du modèle.
        
        Args:
            x (torch.Tensor): Profils d'intensité [batch_size, 600]
        
        Returns:
            torch.Tensor: Prédictions [batch_size, 2] où [:, 0] = gap, [:, 1] = L_ecran
        """
        # Couche 1: 600 → 512
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Couche 2: 512 → 256
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Couche 3: 256 → 128
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Couche 4: 128 → 64
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        # Sortie: 64 → 2 (gap, L_ecran)
        x = self.fc_out(x)
        
        return x

class DualParameterLoss(nn.Module):
    """
    Loss function personnalisée pour prédiction dual avec pondération.
    """
    
    def __init__(self, gap_weight=1.0, L_ecran_weight=1.0):
        """
        Initialise la loss function dual.
        
        Args:
            gap_weight (float): Poids pour la loss du gap
            L_ecran_weight (float): Poids pour la loss de L_ecran
        """
        super(DualParameterLoss, self).__init__()
        self.gap_weight = gap_weight
        self.L_ecran_weight = L_ecran_weight
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        """
        Calcule la loss pondérée.
        
        Args:
            predictions (torch.Tensor): Prédictions [batch_size, 2]
            targets (torch.Tensor): Vraies valeurs [batch_size, 2]
        
        Returns:
            torch.Tensor: Loss totale pondérée
        """
        # Séparer gap et L_ecran
        pred_gap = predictions[:, 0]
        pred_L_ecran = predictions[:, 1]
        true_gap = targets[:, 0]
        true_L_ecran = targets[:, 1]
        
        # Calculer les losses séparées
        loss_gap = self.mse(pred_gap, true_gap)
        loss_L_ecran = self.mse(pred_L_ecran, true_L_ecran)
        
        # Loss totale pondérée
        total_loss = (self.gap_weight * loss_gap + 
                     self.L_ecran_weight * loss_L_ecran)
        
        return total_loss, loss_gap, loss_L_ecran

class DualParameterMetrics:
    """
    Classe pour calculer les métriques de performance dual.
    """
    
    def __init__(self, gap_tolerance=0.01, L_ecran_tolerance=0.1):
        """
        Initialise les métriques.
        
        Args:
            gap_tolerance (float): Tolérance pour accuracy gap (µm)
            L_ecran_tolerance (float): Tolérance pour accuracy L_ecran (µm)
        """
        self.gap_tolerance = gap_tolerance
        self.L_ecran_tolerance = L_ecran_tolerance
    
    def calculate_metrics(self, predictions, targets):
        """
        Calcule toutes les métriques de performance.
        
        Args:
            predictions (np.array): Prédictions [n_samples, 2]
            targets (np.array): Vraies valeurs [n_samples, 2]
        
        Returns:
            dict: Dictionnaire des métriques
        """
        # Séparer les paramètres
        pred_gap = predictions[:, 0]
        pred_L_ecran = predictions[:, 1]
        true_gap = targets[:, 0]
        true_L_ecran = targets[:, 1]
        
        # Métriques pour gap
        gap_r2 = r2_score(true_gap, pred_gap)
        gap_mae = mean_absolute_error(true_gap, pred_gap)
        gap_mse = mean_squared_error(true_gap, pred_gap)
        gap_rmse = np.sqrt(gap_mse)
        gap_accuracy = np.mean(np.abs(pred_gap - true_gap) <= self.gap_tolerance)
        
        # Métriques pour L_ecran
        L_ecran_r2 = r2_score(true_L_ecran, pred_L_ecran)
        L_ecran_mae = mean_absolute_error(true_L_ecran, pred_L_ecran)
        L_ecran_mse = mean_squared_error(true_L_ecran, pred_L_ecran)
        L_ecran_rmse = np.sqrt(L_ecran_mse)
        L_ecran_accuracy = np.mean(np.abs(pred_L_ecran - true_L_ecran) <= self.L_ecran_tolerance)
        
        # Métriques combinées
        combined_r2 = (gap_r2 + L_ecran_r2) / 2
        combined_accuracy = (gap_accuracy + L_ecran_accuracy) / 2
        
        return {
            # Métriques gap
            'gap_r2': gap_r2,
            'gap_mae': gap_mae,
            'gap_mse': gap_mse,
            'gap_rmse': gap_rmse,
            'gap_accuracy': gap_accuracy,
            
            # Métriques L_ecran
            'L_ecran_r2': L_ecran_r2,
            'L_ecran_mae': L_ecran_mae,
            'L_ecran_mse': L_ecran_mse,
            'L_ecran_rmse': L_ecran_rmse,
            'L_ecran_accuracy': L_ecran_accuracy,
            
            # Métriques combinées
            'combined_r2': combined_r2,
            'combined_accuracy': combined_accuracy
        }
    
    def print_metrics(self, metrics, title="Performance Metrics"):
        """
        Affiche les métriques de façon formatée.
        
        Args:
            metrics (dict): Métriques calculées
            title (str): Titre de l'affichage
        """
        print(f"\n📊 {title}")
        print("="*50)
        
        print(f"🎯 GAP METRICS:")
        print(f"   R² Score: {metrics['gap_r2']:.4f}")
        print(f"   MAE: {metrics['gap_mae']:.4f} µm")
        print(f"   RMSE: {metrics['gap_rmse']:.4f} µm")
        print(f"   Accuracy (±{self.gap_tolerance}µm): {metrics['gap_accuracy']:.1%}")
        
        print(f"\n🎯 L_ECRAN METRICS:")
        print(f"   R² Score: {metrics['L_ecran_r2']:.4f}")
        print(f"   MAE: {metrics['L_ecran_mae']:.4f} µm")
        print(f"   RMSE: {metrics['L_ecran_rmse']:.4f} µm")
        print(f"   Accuracy (±{self.L_ecran_tolerance}µm): {metrics['L_ecran_accuracy']:.1%}")
        
        print(f"\n🎯 COMBINED METRICS:")
        print(f"   Combined R²: {metrics['combined_r2']:.4f}")
        print(f"   Combined Accuracy: {metrics['combined_accuracy']:.1%}")
        
        # Vérification des objectifs
        gap_success = metrics['gap_accuracy'] >= 0.90
        L_ecran_success = metrics['L_ecran_accuracy'] >= 0.90
        r2_success = metrics['combined_r2'] >= 0.80
        
        print(f"\n✅ OBJECTIFS:")
        print(f"   Gap Accuracy > 90%: {'✅' if gap_success else '❌'}")
        print(f"   L_ecran Accuracy > 90%: {'✅' if L_ecran_success else '❌'}")
        print(f"   Combined R² > 80%: {'✅' if r2_success else '❌'}")

class DualParameterVisualizer:
    """
    Classe pour visualiser les résultats du modèle dual.
    """
    
    def __init__(self, save_dir="plots/"):
        """
        Initialise le visualiseur.
        
        Args:
            save_dir (str): Dossier de sauvegarde des plots
        """
        self.save_dir = save_dir
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # Utiliser le style par défaut
        sns.set_palette("husl")
    
    def plot_training_curves(self, history, save_name="training_curves.png"):
        """
        Trace les courbes d'entraînement.
        
        Args:
            history (dict): Historique d'entraînement
            save_name (str): Nom du fichier de sauvegarde
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Courbes d\'Entraînement - Prédiction Dual Gap + L_ecran', 
                    fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss totale
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss Totale')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # R² Gap
        axes[0, 1].plot(epochs, history['train_gap_r2'], 'b-', label='Train R² Gap')
        axes[0, 1].plot(epochs, history['val_gap_r2'], 'r-', label='Val R² Gap')
        axes[0, 1].axhline(y=0.8, color='green', linestyle='--', label='Target R²=0.8')
        axes[0, 1].set_title('R² Score - Gap')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # R² L_ecran
        axes[1, 0].plot(epochs, history['train_L_ecran_r2'], 'b-', label='Train R² L_ecran')
        axes[1, 0].plot(epochs, history['val_L_ecran_r2'], 'r-', label='Val R² L_ecran')
        axes[1, 0].axhline(y=0.8, color='green', linestyle='--', label='Target R²=0.8')
        axes[1, 0].set_title('R² Score - L_ecran')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'learning_rate' in history:
            axes[1, 1].plot(epochs, history['learning_rate'], 'g-')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions_scatter(self, predictions, targets, save_name="predictions_scatter.png"):
        """
        Trace les scatter plots des prédictions vs vraies valeurs.
        
        Args:
            predictions (np.array): Prédictions [n_samples, 2]
            targets (np.array): Vraies valeurs [n_samples, 2]
            save_name (str): Nom du fichier de sauvegarde
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Prédictions vs Vraies Valeurs', fontsize=16, fontweight='bold')
        
        # Scatter plot Gap
        axes[0].scatter(targets[:, 0], predictions[:, 0], alpha=0.6, s=20)
        axes[0].plot([targets[:, 0].min(), targets[:, 0].max()], 
                    [targets[:, 0].min(), targets[:, 0].max()], 'r--', linewidth=2)
        axes[0].set_xlabel('Gap Vrai (µm)')
        axes[0].set_ylabel('Gap Prédit (µm)')
        axes[0].set_title('Prédiction Gap')
        axes[0].grid(True, alpha=0.3)
        
        # Calculer R² pour gap
        gap_r2 = r2_score(targets[:, 0], predictions[:, 0])
        axes[0].text(0.05, 0.95, f'R² = {gap_r2:.4f}', 
                    transform=axes[0].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Scatter plot L_ecran
        axes[1].scatter(targets[:, 1], predictions[:, 1], alpha=0.6, s=20)
        axes[1].plot([targets[:, 1].min(), targets[:, 1].max()], 
                    [targets[:, 1].min(), targets[:, 1].max()], 'r--', linewidth=2)
        axes[1].set_xlabel('L_ecran Vrai (µm)')
        axes[1].set_ylabel('L_ecran Prédit (µm)')
        axes[1].set_title('Prédiction L_ecran')
        axes[1].grid(True, alpha=0.3)
        
        # Calculer R² pour L_ecran
        L_ecran_r2 = r2_score(targets[:, 1], predictions[:, 1])
        axes[1].text(0.05, 0.95, f'R² = {L_ecran_r2:.4f}', 
                    transform=axes[1].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=300, bbox_inches='tight')
        plt.close()


def test_model_architecture():
    """
    Test rapide de l'architecture du modèle.
    """
    print("🧪 Test de l'architecture DualParameterPredictor")
    print("="*50)
    
    # Créer le modèle
    model = DualParameterPredictor(input_size=600)
    
    # Test avec données factices
    batch_size = 32
    input_size = 600
    
    # Données d'entrée factices
    x = torch.randn(batch_size, input_size)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(x)
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {predictions.shape}")
    print(f"✅ Expected output shape: ({batch_size}, 2)")
    
    # Vérifier que la sortie a la bonne forme
    assert predictions.shape == (batch_size, 2), f"Erreur: forme attendue ({batch_size}, 2), obtenue {predictions.shape}"
    
    print(f"✅ Test réussi ! Le modèle fonctionne correctement.")
    
    # Afficher le nombre de paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Paramètres totaux: {total_params:,}")
    print(f"📊 Paramètres entraînables: {trainable_params:,}")


if __name__ == "__main__":
    test_model_architecture()
