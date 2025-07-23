#!/usr/bin/env python3
"""
Entraînement optimisé CPU pour Neural_Network_Gap_Lecran_Prediction
Auteur: Oussama GUELFAA
Date: 15/07/2025

Version ultra-optimisée pour CPU avec architecture légère mais efficace.
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Architecture légère pour CPU
class LightweightNetwork(nn.Module):
    """Réseau léger optimisé pour CPU"""
    
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class CPUTrainer:
    """Entraîneur optimisé pour CPU"""
    
    def __init__(self, experiment_name="cpu_training"):
        self.experiment_name = experiment_name
        self.device = torch.device('cpu')
        
        # Créer le dossier de résultats
        self.results_dir = f"results/{experiment_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🧠 CPUTrainer initialisé")
        print(f"   🖥️ Device: {self.device}")
        print(f"   📁 Résultats: {self.results_dir}")
    
    def load_sample_data(self, n_samples=10000):
        """Charge un échantillon de données ÉTENDU pour performance maximale"""
        print(f"📂 Chargement d'un échantillon ÉTENDU de {n_samples} profils...")

        try:
            # Essayer de charger les données prétraitées
            data = np.load('preprocessed_data.npz')
            r_test = data['r_test']
            n_points = len(r_test)
            print(f"   ✅ Données prétraitées trouvées: {n_points} points par profil")
        except:
            print("   ⚠️ Données prétraitées non trouvées, génération de données synthétiques AMÉLIORÉES...")
            n_points = 601
            r_test = np.linspace(1.384585, 5.538338, n_points)
        
        # Générer des données synthétiques réalistes
        np.random.seed(42)
        
        X_data = []
        y_data = []
        
        for i in range(n_samples):
            # Paramètres aléatoires
            gap = np.random.uniform(0.01, 0.5)  # Gap entre 0.01 et 0.5 µm
            L_ecran = np.random.uniform(8.0, 12.0)  # L_écran entre 8 et 12 µm
            
            # Profil d'intensité synthétique (simulation d'anneaux holographiques)
            r_center = np.random.uniform(2.0, 4.0)
            width = np.random.uniform(0.3, 0.8)
            
            # Fonction gaussienne avec oscillations (simulation d'anneaux)
            intensity = np.exp(-0.5 * ((r_test - r_center) / width)**2)
            
            # Ajouter des oscillations pour simuler les anneaux
            freq = 10.0 / gap  # Fréquence inversement proportionnelle au gap
            oscillations = 0.3 * np.sin(freq * r_test) * intensity
            intensity += oscillations
            
            # Ajouter du bruit
            noise = 0.05 * np.random.randn(n_points)
            intensity += noise
            
            # Normaliser
            intensity = np.maximum(intensity, 0)  # Pas de valeurs négatives
            
            X_data.append(intensity)
            y_data.append([gap, L_ecran])
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"✅ Données générées:")
        print(f"   📊 Forme X: {X_data.shape}")
        print(f"   📊 Forme y: {y_data.shape}")
        print(f"   📈 Plage Gap: [{y_data[:, 0].min():.4f}, {y_data[:, 0].max():.4f}] µm")
        print(f"   📈 Plage L_écran: [{y_data[:, 1].min():.3f}, {y_data[:, 1].max():.3f}] µm")
        
        return X_data, y_data
    
    def create_datasets(self, X_data, y_data, train_ratio=0.7, val_ratio=0.15):
        """Crée les datasets train/val/test"""
        n_samples = len(X_data)
        
        # Indices aléatoires
        indices = np.random.permutation(n_samples)
        
        # Calcul des tailles
        train_size = int(train_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        
        # Division
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Normalisation
        from sklearn.preprocessing import StandardScaler
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train = self.scaler_X.fit_transform(X_data[train_indices])
        X_val = self.scaler_X.transform(X_data[val_indices])
        X_test = self.scaler_X.transform(X_data[test_indices])
        
        y_train = self.scaler_y.fit_transform(y_data[train_indices])
        y_val = self.scaler_y.transform(y_data[val_indices])
        y_test = self.scaler_y.transform(y_data[test_indices])
        
        # DataLoaders
        batch_size = 16  # Petit batch pour CPU
        
        self.train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=batch_size, shuffle=True
        )
        
        self.val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
            batch_size=batch_size, shuffle=False
        )
        
        self.test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
            batch_size=batch_size, shuffle=False
        )
        
        print(f"✅ Datasets créés:")
        print(f"   📊 Train: {len(self.train_loader.dataset)} échantillons")
        print(f"   📊 Validation: {len(self.val_loader.dataset)} échantillons")
        print(f"   📊 Test: {len(self.test_loader.dataset)} échantillons")
    
    def create_model(self, input_size=601):
        """Crée le modèle léger"""
        print("🧠 Création du modèle léger pour CPU...")
        
        self.model = LightweightNetwork(
            input_size=input_size,
            hidden_sizes=[512, 256, 128, 64, 32],  # Architecture PLUS PROFONDE
            output_size=2,
            dropout=0.3  # Plus de régularisation
        )
        
        # Loss pondérée
        self.criterion = nn.MSELoss()
        
        # Optimiseur AMÉLIORÉ
        self.optimizer = torch.optim.AdamW(  # AdamW pour meilleure généralisation
            self.model.parameters(),
            lr=5e-4,  # Learning rate plus petit pour stabilité
            weight_decay=1e-3  # Plus de weight decay
        )

        # Scheduler AMÉLIORÉ
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=1e-6  # Cosine annealing
        )
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Modèle créé: {total_params:,} paramètres")
        
        # Historique
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_r2': [], 'val_r2': [],
            'epochs': []
        }
    
    def calculate_metrics(self, predictions, targets):
        """Calcule les métriques"""
        # Dénormaliser
        pred_denorm = self.scaler_y.inverse_transform(predictions)
        target_denorm = self.scaler_y.inverse_transform(targets)
        
        # R² pour chaque paramètre
        gap_r2 = r2_score(target_denorm[:, 0], pred_denorm[:, 0])
        L_ecran_r2 = r2_score(target_denorm[:, 1], pred_denorm[:, 1])
        overall_r2 = (gap_r2 + L_ecran_r2) / 2
        
        # MAE
        gap_mae = mean_absolute_error(target_denorm[:, 0], pred_denorm[:, 0])
        L_ecran_mae = mean_absolute_error(target_denorm[:, 1], pred_denorm[:, 1])
        
        return {
            'gap_r2': gap_r2,
            'L_ecran_r2': L_ecran_r2,
            'overall_r2': overall_r2,
            'gap_mae': gap_mae,
            'L_ecran_mae': L_ecran_mae
        }
    
    def train_epoch(self):
        """Entraîne pour une époque"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for data, target in self.train_loader:
            self.optimizer.zero_grad()
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_predictions.append(output.detach().numpy())
            all_targets.append(target.detach().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        metrics = self.calculate_metrics(predictions, targets)
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
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                all_predictions.append(output.numpy())
                all_targets.append(target.numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        metrics = self.calculate_metrics(predictions, targets)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(self, epochs=100):
        """Entraînement complet"""
        print(f"🚀 Début de l'entraînement: {epochs} époques")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 25  # Plus de patience pour convergence
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Entraînement et validation
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            
            # Mise à jour du scheduler
            self.scheduler.step()  # CosineAnnealingLR ne prend pas de métrique
            
            # Historique
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_r2'].append(train_metrics['overall_r2'])
            self.history['val_r2'].append(val_metrics['overall_r2'])
            self.history['epochs'].append(epoch + 1)
            
            # Affichage
            epoch_time = time.time() - epoch_start
            print(f"Époque {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.6f}, R²: {train_metrics['overall_r2']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.6f}, R²: {val_metrics['overall_r2']:.4f}")
            print(f"  Gap R²: {val_metrics['gap_r2']:.4f}, L_écran R²: {val_metrics['L_ecran_r2']:.4f}")
            
            # Sauvegarde du meilleur modèle
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_metrics': val_metrics
                }, os.path.join(self.results_dir, 'best_model.pt'))
                
                print(f"  ✅ Nouveau meilleur modèle sauvegardé!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏹️ Early stopping après {epoch+1} époques")
                break
        
        total_time = time.time() - start_time
        print(f"✅ Entraînement terminé en {total_time/60:.1f} minutes")
        
        # Test final
        self.test_model()
        self.plot_results()
    
    def test_model(self):
        """Test final"""
        print("🧪 Test final du modèle...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                all_predictions.append(output.numpy())
                all_targets.append(target.numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        test_metrics = self.calculate_metrics(predictions, targets)
        
        print(f"📊 Résultats finaux:")
        print(f"   Gap R²: {test_metrics['gap_r2']:.4f}")
        print(f"   L_écran R²: {test_metrics['L_ecran_r2']:.4f}")
        print(f"   R² Global: {test_metrics['overall_r2']:.4f}")
        print(f"   Gap MAE: {test_metrics['gap_mae']:.6f} µm")
        print(f"   L_écran MAE: {test_metrics['L_ecran_mae']:.6f} µm")
        
        # Sauvegarder les résultats
        with open(os.path.join(self.results_dir, 'final_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        return test_metrics
    
    def plot_results(self):
        """Génère les graphiques de résultats"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = self.history['epochs']
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0].set_title('Loss d\'Entraînement')
        axes[0].set_xlabel('Époque')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # R²
        axes[1].plot(epochs, self.history['train_r2'], 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.history['val_r2'], 'r-', label='Validation', linewidth=2)
        axes[1].set_title('Score R²')
        axes[1].set_xlabel('Époque')
        axes[1].set_ylabel('R²')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='Objectif 80%')
        
        plt.suptitle(f'Entraînement CPU - {self.experiment_name}', fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'training_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Graphiques sauvegardés: {save_path}")
        plt.close()

def main():
    """Fonction principale"""
    print("🧠 Entraînement CPU Optimisé")
    print("=" * 40)
    
    # Créer l'entraîneur AMÉLIORÉ
    trainer = CPUTrainer("cpu_enhanced_training_10k_samples")
    
    # Charger PLUS de données pour performance maximale
    X_data, y_data = trainer.load_sample_data(n_samples=10000)  # 10,000 échantillons !

    # Créer les datasets
    trainer.create_datasets(X_data, y_data)

    # Créer le modèle
    trainer.create_model(input_size=X_data.shape[1])

    # Entraîner PLUS LONGTEMPS
    trainer.train(epochs=200)  # 200 époques pour convergence optimale
    
    print(f"\n🎉 Entraînement CPU terminé!")
    print(f"📁 Résultats dans: {trainer.results_dir}")

if __name__ == "__main__":
    main()
