#!/usr/bin/env python3
"""
Optimized Neural Network Training with New Dataset
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Ce script entraîne un réseau de neurones optimisé sur les données extraites du fichier
all_banque_new_24_01_25_NEW_full.mat pour prédire les paramètres L_ecran et gap
à partir des profils radiaux d'intensité normalisés I_subs/I_subs_inc.

Optimisations implémentées:
- Architecture améliorée avec attention et meilleure initialisation
- Fonctions d'activation avancées (Swish/SiLU)
- Scheduler de learning rate sophistiqué (Cosine annealing)
- Early stopping avec restauration du meilleur modèle
- Gradient clipping et régularisation adaptative
- Loss function robuste (Huber loss)
- Métriques de performance complètes

Architecture:
- Entrée: Profils radiaux 1D (1000 points)
- Sortie: Paramètres [L_ecran, gap]
- Réseau: Dense avec blocs résiduels, attention et normalisation avancée
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import time
import warnings
from pathlib import Path

# Supprimer les warnings pour un affichage plus propre
warnings.filterwarnings('ignore')

class SelfAttention(nn.Module):
    """
    Mécanisme d'attention pour améliorer l'apprentissage des features importantes.
    Permet au modèle de se concentrer sur les parties les plus informatives du profil.
    """
    def __init__(self, input_dim, attention_dim=64):
        super(SelfAttention, self).__init__()
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.output = nn.Linear(attention_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Forward pass avec mécanisme d'attention."""
        batch_size = x.size(0)

        # Calculer Q, K, V
        Q = self.query(x)  # [batch_size, attention_dim]
        K = self.key(x)    # [batch_size, attention_dim]
        V = self.value(x)  # [batch_size, attention_dim]

        # Calculer les scores d'attention
        attention_scores = torch.matmul(Q.unsqueeze(1), K.unsqueeze(2)).squeeze()
        attention_weights = F.softmax(attention_scores / np.sqrt(self.attention_dim), dim=-1)

        # Appliquer l'attention
        attended = attention_weights.unsqueeze(1) * V
        output = self.output(attended)
        output = self.dropout(output)

        return output + x  # Connexion résiduelle

class OptimizedResidualBlock(nn.Module):
    """
    Bloc résiduel optimisé avec activation Swish, normalisation améliorée et attention.
    Utilise des techniques avancées pour améliorer la convergence et les performances.
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.15, use_attention=True):
        super(OptimizedResidualBlock, self).__init__()

        # Couches principales avec activation Swish (SiLU)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

        # Normalisation par couches (plus stable que BatchNorm pour les petits batches)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

        # Dropout adaptatif
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)  # Moins de dropout sur la sortie

        # Activation Swish (plus lisse que ReLU)
        self.activation = nn.SiLU()

        # Mécanisme d'attention optionnel
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(input_dim)

        # Initialisation He pour Swish/SiLU
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialisation optimisée des poids pour Swish/SiLU."""
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x):
        """Forward pass optimisé avec connexion résiduelle."""
        residual = x

        # Premier bloc avec normalisation et activation
        out = self.linear1(x)
        out = self.layer_norm1(out)
        out = self.activation(out)
        out = self.dropout1(out)

        # Deuxième bloc
        out = self.linear2(out)
        out = self.layer_norm2(out)
        out = self.dropout2(out)

        # Connexion résiduelle
        out = out + residual

        # Attention optionnelle
        if self.use_attention:
            out = self.attention(out)

        # Activation finale
        out = self.activation(out)

        return out

class OptimizedRingProfileRegressor(nn.Module):
    """
    Réseau de neurones optimisé pour la régression des paramètres à partir de profils radiaux.

    Améliorations:
    - Architecture plus profonde et plus large
    - Blocs résiduels optimisés avec attention
    - Normalisation par couches (LayerNorm)
    - Activation Swish/SiLU plus performante
    - Initialisation He optimisée
    - Régularisation adaptative

    Architecture:
    - Couche d'entrée: 1000 → 768
    - Blocs résiduels optimisés: 768 → 768 (avec attention)
    - Couches de sortie: 768 → 512 → 256 → 128 → 64 → 2
    """
    def __init__(self, input_dim=1000, hidden_dims=[768, 512, 256, 128, 64], output_dim=2,
                 n_residual_blocks=4, dropout_rate=0.15, use_attention=True):
        super(OptimizedRingProfileRegressor, self).__init__()

        # Couche d'entrée optimisée
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),  # LayerNorm plus stable
            nn.SiLU(),  # Activation Swish/SiLU
            nn.Dropout(dropout_rate * 0.5)  # Dropout réduit en entrée
        )

        # Blocs résiduels optimisés avec attention
        self.residual_blocks = nn.ModuleList([
            OptimizedResidualBlock(
                hidden_dims[0],
                hidden_dims[0] * 2,
                dropout_rate,
                use_attention=(use_attention and i < 2)  # Attention sur les premiers blocs
            )
            for i in range(n_residual_blocks)
        ])

        # Couches de sortie avec architecture progressive
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.SiLU(),
                nn.Dropout(dropout_rate * (0.8 - i * 0.1))  # Dropout dégressif
            ])

        # Couche finale avec initialisation spéciale
        final_layer = nn.Linear(hidden_dims[-1], output_dim)
        # Initialisation Xavier pour la couche de sortie
        nn.init.xavier_normal_(final_layer.weight)
        nn.init.constant_(final_layer.bias, 0)
        layers.append(final_layer)

        self.output_layers = nn.Sequential(*layers)

        # Initialisation globale des poids
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialisation optimisée de tous les poids du réseau."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization pour SiLU/Swish
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass optimisé du réseau."""
        # Couche d'entrée
        x = self.input_layer(x)

        # Blocs résiduels avec connexions profondes
        for block in self.residual_blocks:
            x = block(x)

        # Couches de sortie
        x = self.output_layers(x)

        return x

def load_training_data(data_path="processed_data/training_data.npz"):
    """
    Charge les données d'entraînement préparées.
    
    Args:
        data_path (str): Chemin vers le fichier de données
    
    Returns:
        tuple: (X, y, metadata) - Features, targets et métadonnées
    """
    print(f"Chargement des données depuis {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    metadata = data['metadata'].item()
    
    print(f"Données chargées:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Nombre d'échantillons: {metadata['n_samples']}")
    print(f"  Plage L_ecran: {metadata['L_ecran_range']}")
    print(f"  Plage gap: {metadata['gap_range']}")
    
    return X, y, metadata

def prepare_data_for_training(X, y, test_size=0.2, val_size=0.1, random_state=42,
                             use_robust_scaling=True, add_noise=False):
    """
    Prépare les données pour l'entraînement avec normalisation robuste et augmentation.

    Args:
        X (np.ndarray): Features (profils d'intensité)
        y (np.ndarray): Targets (paramètres)
        test_size (float): Proportion du jeu de test
        val_size (float): Proportion du jeu de validation
        random_state (int): Graine aléatoire
        use_robust_scaling (bool): Utiliser RobustScaler au lieu de StandardScaler
        add_noise (bool): Ajouter du bruit pour l'augmentation de données

    Returns:
        tuple: Données préparées et normalisées
    """
    print(f"\nPréparation optimisée des données pour l'entraînement...")
    print(f"  Robust scaling: {use_robust_scaling}")
    print(f"  Data augmentation: {add_noise}")

    # Vérification de la qualité des données
    print(f"\nAnalyse des données d'entrée:")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  X range: [{X.min():.6f}, {X.max():.6f}]")
    print(f"  y range: L_ecran [{y[:, 0].min():.3f}, {y[:, 0].max():.3f}], gap [{y[:, 1].min():.6f}, {y[:, 1].max():.6f}]")
    print(f"  NaN values: X={np.isnan(X).sum()}, y={np.isnan(y).sum()}")

    # Division train/test stratifiée basée sur les paramètres
    # Créer des bins pour la stratification
    y_bins = np.digitize(y[:, 0], bins=np.linspace(y[:, 0].min(), y[:, 0].max(), 10))

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y_bins
    )

    # Division train/validation
    val_size_adjusted = val_size / (1 - test_size)
    y_train_val_bins = np.digitize(y_train_val[:, 0], bins=np.linspace(y_train_val[:, 0].min(), y_train_val[:, 0].max(), 8))
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted,
        random_state=random_state, stratify=y_train_val_bins
    )

    print(f"\nDivision stratifiée des données:")
    print(f"  Train: {X_train.shape[0]} échantillons ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Validation: {X_val.shape[0]} échantillons ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test: {X_test.shape[0]} échantillons ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Augmentation de données (optionnelle)
    if add_noise:
        print(f"\nAugmentation des données d'entraînement...")
        noise_factor = 0.01
        X_train_noise = X_train + np.random.normal(0, noise_factor, X_train.shape)
        X_train = np.vstack([X_train, X_train_noise])
        y_train = np.vstack([y_train, y_train])
        print(f"  Nouvelles données d'entraînement: {X_train.shape[0]} échantillons")

    # Normalisation robuste des features
    if use_robust_scaling:
        scaler_X = RobustScaler()  # Plus robuste aux outliers
    else:
        scaler_X = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Normalisation des targets avec StandardScaler (meilleur pour la régression)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    print(f"\nNormalisation appliquée:")
    if hasattr(scaler_X, 'center_'):
        print(f"  Features (Robust) - center: {scaler_X.center_[:5]}, scale: {scaler_X.scale_[:5]}")
    else:
        print(f"  Features (Standard) - mean: {scaler_X.mean_[:5]}, std: {scaler_X.scale_[:5]}")
    print(f"  Targets - mean: {scaler_y.mean_}, std: {scaler_y.scale_}")

    # Vérification post-normalisation
    print(f"\nVérification post-normalisation:")
    print(f"  X_train_scaled range: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
    print(f"  y_train_scaled range: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            scaler_X, scaler_y)

def create_optimized_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test,
                                 batch_size=64, num_workers=0):
    """
    Crée les DataLoaders PyTorch optimisés pour l'entraînement.

    Args:
        X_*, y_*: Données d'entraînement, validation et test
        batch_size (int): Taille des batches (augmentée pour la stabilité)
        num_workers (int): Nombre de workers pour le chargement des données

    Returns:
        tuple: DataLoaders pour train, validation et test
    """
    print(f"\nCréation des DataLoaders optimisés...")

    # Conversion en tenseurs PyTorch avec vérification
    try:
        X_train_tensor = torch.FloatTensor(X_train.astype(np.float32))
        X_val_tensor = torch.FloatTensor(X_val.astype(np.float32))
        X_test_tensor = torch.FloatTensor(X_test.astype(np.float32))
        y_train_tensor = torch.FloatTensor(y_train.astype(np.float32))
        y_val_tensor = torch.FloatTensor(y_val.astype(np.float32))
        y_test_tensor = torch.FloatTensor(y_test.astype(np.float32))

        print(f"  Tenseurs créés avec succès")
        print(f"  X_train_tensor shape: {X_train_tensor.shape}")
        print(f"  y_train_tensor shape: {y_train_tensor.shape}")

    except Exception as e:
        print(f"  Erreur lors de la création des tenseurs: {e}")
        raise

    # Création des datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Création des DataLoaders avec optimisations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Pour éviter les problèmes avec BatchNorm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Pin memory: {torch.cuda.is_available()}")

    return train_loader, val_loader, test_loader

def train_optimized_model(model, train_loader, val_loader, num_epochs=200,
                         initial_lr=0.001, device='cpu', save_path="models/optimized_ring_regressor.pth"):
    """
    Entraîne le modèle avec des optimisations avancées pour maximiser les performances.

    Optimisations:
    - Loss function robuste (Huber loss)
    - Scheduler cosine annealing avec warm restarts
    - Early stopping avec patience et restauration
    - Gradient clipping pour la stabilité
    - Métriques de performance complètes (R², RMSE, MAE)
    - Monitoring en temps réel

    Args:
        model: Modèle PyTorch optimisé
        train_loader, val_loader: DataLoaders
        num_epochs (int): Nombre d'époques maximum
        initial_lr (float): Taux d'apprentissage initial
        device (str): Device ('cpu' ou 'cuda')
        save_path (str): Chemin de sauvegarde du modèle

    Returns:
        tuple: (model, history) - Modèle entraîné et historique détaillé
    """
    print(f"\n{'='*60}")
    print(f"{'ENTRAÎNEMENT OPTIMISÉ DU MODÈLE':^60}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs max: {num_epochs}")
    print(f"Learning rate initial: {initial_lr}")
    print(f"Modèle: {model.__class__.__name__}")
    print(f"Paramètres: {sum(p.numel() for p in model.parameters()):,}")

    model = model.to(device)

    # Loss function robuste (Huber loss moins sensible aux outliers)
    criterion = nn.HuberLoss(delta=1.0)

    # Optimiseur Adam avec paramètres optimisés
    optimizer = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=1e-4,  # Régularisation L2
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Scheduler cosine annealing avec warm restarts plus lent
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,  # Redémarrage tous les 50 epochs pour plus de stabilité
        T_mult=1,  # Garder la même période
        eta_min=1e-7  # LR minimum plus bas
    )

    # Variables de suivi
    history = {
        'train_loss': [], 'val_loss': [],
        'train_r2': [], 'val_r2': [],
        'train_rmse': [], 'val_rmse': [],
        'train_mae': [], 'val_mae': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    patience_counter = 0
    patience = 50  # Early stopping patience augmentée

    print(f"\nDébut de l'entraînement...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Phase d'entraînement
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Gradient clipping pour éviter l'explosion des gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_predictions.append(outputs.detach().cpu().numpy())
            train_targets.append(batch_y.detach().cpu().numpy())

        # Phase de validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        # Calcul des métriques
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Concaténer les prédictions pour calculer les métriques
        train_pred = np.vstack(train_predictions)
        train_true = np.vstack(train_targets)
        val_pred = np.vstack(val_predictions)
        val_true = np.vstack(val_targets)

        # Métriques de performance
        train_r2 = r2_score(train_true, train_pred)
        val_r2 = r2_score(val_true, val_pred)
        train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
        val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
        train_mae = mean_absolute_error(train_true, train_pred)
        val_mae = mean_absolute_error(val_true, val_pred)

        # Sauvegarder l'historique
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Scheduler step
        scheduler.step()

        # Early stopping et sauvegarde du meilleur modèle
        if val_r2 > best_val_r2:  # Utiliser R² comme métrique principale
            best_val_r2 = val_r2
            best_val_loss = val_loss
            patience_counter = 0

            # Sauvegarder le meilleur modèle
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_r2': val_r2,
                'history': history
            }, save_path)
        else:
            patience_counter += 1

        # Affichage périodique
        if (epoch + 1) % 10 == 0 or epoch < 5:
            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] ({epoch_time:.1f}s) - "
                  f"Train: Loss={train_loss:.6f}, R²={train_r2:.4f} | "
                  f"Val: Loss={val_loss:.6f}, R²={val_r2:.4f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping à l'epoch {epoch+1} (patience={patience})")
            break

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*60}")
    print(f"Temps total: {total_time/60:.1f} minutes")
    print(f"Meilleur R² validation: {best_val_r2:.6f}")
    print(f"Meilleur loss validation: {best_val_loss:.6f}")
    print(f"Modèle sauvegardé: {save_path}")

    return model, history

def plot_training_history(history, save_path="plots/training_history.png"):
    """
    Génère des graphiques détaillés de l'historique d'entraînement.

    Args:
        history (dict): Historique d'entraînement
        save_path (str): Chemin de sauvegarde
    """
    print(f"\nGénération des graphiques d'entraînement...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss Evolution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Huber Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # R² Score curves
    axes[0, 1].plot(epochs, history['train_r2'], 'b-', label='Train R²', linewidth=2)
    axes[0, 1].plot(epochs, history['val_r2'], 'r-', label='Validation R²', linewidth=2)
    axes[0, 1].axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='Target R² = 0.8')
    axes[0, 1].set_title('R² Score Evolution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # RMSE curves
    axes[1, 0].plot(epochs, history['train_rmse'], 'b-', label='Train RMSE', linewidth=2)
    axes[1, 0].plot(epochs, history['val_rmse'], 'r-', label='Validation RMSE', linewidth=2)
    axes[1, 0].set_title('RMSE Evolution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Graphiques sauvegardés: {save_path}")

def main():
    """Fonction principale optimisée pour l'entraînement."""

    print(f"\n{'='*80}")
    print(f"{'ENTRAÎNEMENT OPTIMISÉ DU RÉSEAU DE NEURONES':^80}")
    print(f"{'Prédiction des paramètres L_ecran et gap':^80}")
    print(f"{'='*80}")

    # Configuration optimisée
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # 1. Charger les données
    print(f"\n{'='*50}")
    print(f"{'CHARGEMENT DES DONNÉES':^50}")
    print(f"{'='*50}")
    X, y, metadata = load_training_data()

    # 2. Préparer les données avec optimisations
    print(f"\n{'='*50}")
    print(f"{'PRÉPARATION DES DONNÉES':^50}")
    print(f"{'='*50}")
    (X_train, X_val, X_test, y_train, y_val, y_test,
     scaler_X, scaler_y) = prepare_data_for_training(
        X, y,
        use_robust_scaling=True,  # Utiliser RobustScaler
        add_noise=True   # Activer l'augmentation de données
    )

    # 3. Créer les DataLoaders optimisés
    print(f"\n{'='*50}")
    print(f"{'CRÉATION DES DATALOADERS':^50}")
    print(f"{'='*50}")
    train_loader, val_loader, test_loader = create_optimized_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        batch_size=64,  # Batch size plus grand pour la stabilité
        num_workers=0   # 0 pour éviter les problèmes sur certains systèmes
    )

    # 4. Créer le modèle optimisé
    print(f"\n{'='*50}")
    print(f"{'CRÉATION DU MODÈLE OPTIMISÉ':^50}")
    print(f"{'='*50}")
    model = OptimizedRingProfileRegressor(
        input_dim=1000,
        hidden_dims=[1024, 768, 512, 256, 128],  # Architecture encore plus large
        output_dim=2,
        n_residual_blocks=6,  # Plus de blocs résiduels pour plus de capacité
        dropout_rate=0.1,     # Dropout réduit pour permettre plus d'apprentissage
        use_attention=True    # Mécanisme d'attention
    )

    print(f"Modèle créé: {model.__class__.__name__}")
    print(f"  Paramètres totaux: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Paramètres entraînables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Taille mémoire estimée: {sum(p.numel() * 4 for p in model.parameters()) / 1024**2:.1f} MB")

    # 5. Entraîner le modèle avec optimisations avancées
    print(f"\n{'='*50}")
    print(f"{'ENTRAÎNEMENT OPTIMISÉ':^50}")
    print(f"{'='*50}")
    model, history = train_optimized_model(
        model, train_loader, val_loader,
        num_epochs=500,           # Beaucoup plus d'époques
        initial_lr=0.0005,        # Learning rate plus petit pour convergence stable
        device=device,
        save_path="models/optimized_ring_regressor.pth"
    )

    # 6. Sauvegarder les scalers avec informations complètes
    print(f"\n{'='*50}")
    print(f"{'SAUVEGARDE DES COMPOSANTS':^50}")
    print(f"{'='*50}")

    scaler_data = {
        'scaler_X_mean': getattr(scaler_X, 'mean_', getattr(scaler_X, 'center_', None)),
        'scaler_X_scale': scaler_X.scale_,
        'scaler_y_mean': scaler_y.mean_,
        'scaler_y_scale': scaler_y.scale_,
        'scaler_type': type(scaler_X).__name__,
        'input_dim': 1000,
        'output_dim': 2,
        'metadata': metadata
    }

    os.makedirs('models', exist_ok=True)
    np.savez('models/optimized_scalers.npz', **scaler_data)

    # 7. Générer les graphiques d'entraînement
    print(f"\n{'='*50}")
    print(f"{'GÉNÉRATION DES VISUALISATIONS':^50}")
    print(f"{'='*50}")
    plot_training_history(history)

    # 8. Résumé final
    print(f"\n{'='*80}")
    print(f"{'ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS':^80}")
    print(f"{'='*80}")

    final_val_r2 = history['val_r2'][-1] if history['val_r2'] else 0
    best_val_r2 = max(history['val_r2']) if history['val_r2'] else 0

    print(f"Performances finales:")
    print(f"  R² validation final: {final_val_r2:.6f}")
    print(f"  Meilleur R² validation: {best_val_r2:.6f}")
    print(f"  Objectif atteint (R² > 0.8): {'✓ OUI' if best_val_r2 > 0.8 else '✗ NON'}")
    print(f"\nFichiers sauvegardés:")
    print(f"  Modèle: models/optimized_ring_regressor.pth")
    print(f"  Scalers: models/optimized_scalers.npz")
    print(f"  Graphiques: plots/training_history.png")
    print(f"\nPour évaluer le modèle, exécutez:")
    print(f"  python evaluate_new_model.py")

if __name__ == "__main__":
    main()
