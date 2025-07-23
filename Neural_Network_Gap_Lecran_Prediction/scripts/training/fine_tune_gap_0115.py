#!/usr/bin/env python3
"""
Fine-tuning spécialisé sur données proches de gap=0.115 µm
Auteur: Oussama GUELFAA
Date: 18/07/2025

Fine-tuning du modèle spécialisé sur des données simulées
proches du gap expérimental (0.115 µm) pour maximiser la précision.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys
sys.path.append('../../utils/data_loaders')
from ultra_fast_data_loader import UltraFastDataLoader
sys.path.append('../training')
from specialized_gap_training import DualSpecializedNetwork, SpecializedLoss

class GapFineTuner:
    """Fine-tuner spécialisé pour gap=0.115 µm."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🎯 FINE-TUNING SPÉCIALISÉ GAP = 0.115 µm")
        print("=" * 50)
        print(f"   🖥️ Device: {self.device}")
        print(f"   🎯 Objectif: Optimiser pour gap ≈ 0.115 µm")
    
    def load_specialized_model(self):
        """Charge le modèle spécialisé pré-entraîné."""
        
        print("📂 Chargement du modèle spécialisé...")
        
        model_path = "results/specialized_gap_training/best_specialized_model.pt"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        # Charger le checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Créer et charger le modèle
        self.model = DualSpecializedNetwork().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"   ✅ Modèle chargé (époque {checkpoint['epoch']})")
        print(f"   📊 Gap R² initial: {checkpoint['gap_r2']:.4f}")
        
        return checkpoint
    
    def extract_target_data(self, gap_target=0.115, tolerance=0.02):
        """Extrait les données proches du gap cible."""
        
        print(f"📊 Extraction des données proches de gap={gap_target:.3f} µm (±{tolerance:.3f})...")
        
        # Charger toutes les données
        data_loader = UltraFastDataLoader("../../data/processed/extracted_data_full.npz")
        X_all, y_all = data_loader.load_data(sample_ratio=1.0)
        
        gaps_all = y_all[:, 0]
        
        # Filtrer par gap
        mask = (gaps_all >= gap_target - tolerance) & (gaps_all <= gap_target + tolerance)
        indices = np.where(mask)[0]
        
        X_target = X_all[indices]
        y_target = y_all[indices]
        
        print(f"   ✅ {len(X_target)} échantillons extraits")
        print(f"   📊 Gap range: [{y_target[:, 0].min():.6f}, {y_target[:, 0].max():.6f}] µm")
        print(f"   📊 L_écran range: [{y_target[:, 1].min():.3f}, {y_target[:, 1].max():.3f}] µm")
        
        return X_target, y_target
    
    def create_fine_tune_loaders(self, X_target, y_target, batch_size=16):
        """Crée les data loaders pour le fine-tuning."""
        
        print(f"🔄 Création des data loaders (batch_size={batch_size})...")
        
        # Division train/val pour le fine-tuning
        n_samples = len(X_target)
        n_train = int(0.8 * n_samples)
        
        # Mélanger les indices
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train = X_target[train_indices]
        y_train = y_target[train_indices]
        X_val = X_target[val_indices]
        y_val = y_target[val_indices]
        
        # Créer les datasets PyTorch
        from torch.utils.data import TensorDataset, DataLoader
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"   ✅ Train: {len(X_train)} échantillons")
        print(f"   ✅ Val: {len(X_val)} échantillons")
        
        return train_loader, val_loader
    
    def fine_tune(self, train_loader, val_loader, epochs=100, lr=1e-5):
        """Fine-tuning du modèle."""
        
        print(f"🚀 Début du fine-tuning ({epochs} époques, lr={lr})...")
        
        # Loss function ultra-agressive pour le gap
        criterion = SpecializedLoss(gap_weight=500.0, L_ecran_weight=1.0)
        
        # Optimiseur avec learning rate très faible
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-7)
        
        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-8
        )
        
        best_gap_r2 = -float('inf')
        patience_counter = 0
        patience = 25
        
        history = {
            'train_loss': [], 'val_loss': [],
            'train_gap_r2': [], 'val_gap_r2': [],
            'train_L_ecran_r2': [], 'val_L_ecran_r2': []
        }
        
        for epoch in range(epochs):
            # Entraînement
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                loss, gap_loss, L_ecran_loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping agressif
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_predictions.append(output.detach().cpu().numpy())
                train_targets.append(target.detach().cpu().numpy())
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    
                    loss, gap_loss, L_ecran_loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    val_predictions.append(output.cpu().numpy())
                    val_targets.append(target.cpu().numpy())
            
            # Calculer R²
            train_pred = np.vstack(train_predictions)
            train_true = np.vstack(train_targets)
            val_pred = np.vstack(val_predictions)
            val_true = np.vstack(val_targets)
            
            from sklearn.metrics import r2_score
            train_gap_r2 = r2_score(train_true[:, 0], train_pred[:, 0])
            train_L_ecran_r2 = r2_score(train_true[:, 1], train_pred[:, 1])
            val_gap_r2 = r2_score(val_true[:, 0], val_pred[:, 0])
            val_L_ecran_r2 = r2_score(val_true[:, 1], val_pred[:, 1])
            
            # Sauvegarder l'historique
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss / len(val_loader))
            history['train_gap_r2'].append(train_gap_r2)
            history['val_gap_r2'].append(val_gap_r2)
            history['train_L_ecran_r2'].append(train_L_ecran_r2)
            history['val_L_ecran_r2'].append(val_L_ecran_r2)
            
            # Scheduler
            scheduler.step(val_loss / len(val_loader))
            
            # Affichage
            if epoch % 10 == 0 or epoch < 10:
                print(f"Époque {epoch:3d}: Train Gap R²={train_gap_r2:.4f}, Val Gap R²={val_gap_r2:.4f}")
            
            # Sauvegarde du meilleur modèle
            if val_gap_r2 > best_gap_r2:
                best_gap_r2 = val_gap_r2
                patience_counter = 0
                
                # Sauvegarder
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'gap_r2': val_gap_r2,
                    'L_ecran_r2': val_L_ecran_r2,
                    'fine_tuned_for': 'gap_0115'
                }, 'results/fine_tuned_gap_0115_model.pt')
                
                if epoch % 10 == 0:
                    print(f"   ✅ Nouveau meilleur modèle sauvegardé! (Gap R²: {val_gap_r2:.4f})")
            else:
                patience_counter += 1
            
            # Objectif atteint ?
            if val_gap_r2 >= 0.95:
                print(f"   🎯 OBJECTIF DÉPASSÉ! Gap R²: {val_gap_r2:.4f}")
                break
            
            # Early stopping
            if patience_counter >= patience:
                print(f"   ⏹️  Early stopping à l'époque {epoch}")
                break
        
        print(f"✅ Fine-tuning terminé! Meilleur Gap R²: {best_gap_r2:.4f}")
        
        return history, best_gap_r2
    
    def run_fine_tuning(self):
        """Lance le fine-tuning complet."""
        
        try:
            # 1. Charger le modèle
            checkpoint = self.load_specialized_model()
            
            # 2. Extraire les données cibles
            X_target, y_target = self.extract_target_data(gap_target=0.115, tolerance=0.02)
            
            # 3. Créer les data loaders
            train_loader, val_loader = self.create_fine_tune_loaders(X_target, y_target)
            
            # 4. Fine-tuning
            history, best_gap_r2 = self.fine_tune(train_loader, val_loader, epochs=150, lr=5e-6)
            
            print(f"\n✅ FINE-TUNING TERMINÉ!")
            print(f"   🎯 Meilleur Gap R²: {best_gap_r2:.4f}")
            
            return best_gap_r2
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            raise

def main():
    """Fonction principale."""
    
    fine_tuner = GapFineTuner()
    best_r2 = fine_tuner.run_fine_tuning()
    
    print(f"\n🎉 FINE-TUNING SPÉCIALISÉ TERMINÉ!")
    print(f"   🎯 Gap R² final: {best_r2:.4f}")

if __name__ == "__main__":
    main()
