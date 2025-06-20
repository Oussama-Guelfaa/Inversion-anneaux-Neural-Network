#!/usr/bin/env python3
"""
Architecture de Modèle Robuste pour Prédiction Dual Gap + L_ecran

Auteur: Oussama GUELFAA
Date: 19 - 06 - 2025
Objectif: Modèle résistant au bruit avec architecture optimisée
"""

import torch
import torch.nn as nn
import numpy as np

class RobustLoss(nn.Module):
    """
    Fonction de perte robuste combinant MSE et Huber Loss
    """
    
    def __init__(self, alpha=0.7, beta=0.3, huber_delta=1.0):
        """
        Initialise la fonction de perte robuste
        
        Args:
            alpha: Poids pour MSE Loss
            beta: Poids pour Huber Loss
            huber_delta: Paramètre delta pour Huber Loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss(beta=huber_delta)
        
    def forward(self, predictions, targets):
        """
        Calcule la perte robuste
        
        Args:
            predictions: Prédictions du modèle
            targets: Valeurs cibles
        
        Returns:
            Perte combinée
        """
        mse_loss = self.mse(predictions, targets)
        huber_loss = self.huber(predictions, targets)
        
        return self.alpha * mse_loss + self.beta * huber_loss

class NoiseResistantBlock(nn.Module):
    """
    Bloc résistant au bruit avec normalisation et régularisation
    """
    
    def __init__(self, input_size, output_size, dropout_rate=0.3, use_residual=True):
        """
        Initialise le bloc résistant au bruit
        
        Args:
            input_size: Taille d'entrée
            output_size: Taille de sortie
            dropout_rate: Taux de dropout
            use_residual: Utiliser connexion résiduelle
        """
        super().__init__()
        self.use_residual = use_residual and (input_size == output_size)
        
        self.linear = nn.Linear(input_size, output_size)
        self.batch_norm = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        
        # Connexion résiduelle si applicable
        if self.use_residual:
            self.residual_connection = nn.Identity()
        
    def forward(self, x):
        """
        Passage avant du bloc
        
        Args:
            x: Entrée du bloc
        
        Returns:
            Sortie du bloc
        """
        identity = x if self.use_residual else None
        
        out = self.linear(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Ajouter la connexion résiduelle si applicable
        if self.use_residual and identity is not None:
            out = out + identity
        
        return out

class RobustDualParameterModel(nn.Module):
    """
    Modèle robuste pour prédiction dual Gap + L_ecran
    """
    
    def __init__(self, input_size=600, dropout_rate=0.3, use_attention=True):
        """
        Initialise le modèle robuste
        
        Args:
            input_size: Taille d'entrée (600 points)
            dropout_rate: Taux de dropout
            use_attention: Utiliser mécanisme d'attention
        """
        super().__init__()
        
        self.input_size = input_size
        self.use_attention = use_attention
        
        # Normalisation d'entrée robuste
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Encodeur principal avec blocs résistants au bruit
        self.encoder = nn.Sequential(
            NoiseResistantBlock(input_size, 1024, dropout_rate, use_residual=False),
            NoiseResistantBlock(1024, 512, dropout_rate, use_residual=False),
            NoiseResistantBlock(512, 256, dropout_rate, use_residual=False),
            NoiseResistantBlock(256, 128, dropout_rate, use_residual=False),
        )
        
        # Mécanisme d'attention multi-têtes si activé
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=128,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(128)
        
        # Têtes de prédiction spécialisées
        self.gap_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1)
        )
        
        self.lecran_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1)
        )
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialise les poids du modèle
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Passage avant du modèle
        
        Args:
            x: Profils d'intensité [batch_size, input_size]
        
        Returns:
            Prédictions [batch_size, 2] où [:, 0] = gap, [:, 1] = L_ecran
        """
        # Normalisation d'entrée
        x = self.input_norm(x)
        
        # Encodage avec blocs résistants au bruit
        features = self.encoder(x)
        
        # Mécanisme d'attention si activé
        if self.use_attention:
            # Reshape pour attention (batch_size, seq_len=1, embed_dim)
            features_reshaped = features.unsqueeze(1)
            
            # Attention multi-têtes
            attended_features, _ = self.attention(
                features_reshaped, features_reshaped, features_reshaped
            )
            
            # Normalisation et connexion résiduelle
            features = self.attention_norm(
                features_reshaped + attended_features
            ).squeeze(1)
        
        # Prédictions spécialisées
        gap_pred = self.gap_head(features)
        lecran_pred = self.lecran_head(features)
        
        return torch.cat([gap_pred, lecran_pred], dim=1)
    
    def count_parameters(self):
        """
        Compte le nombre de paramètres du modèle
        
        Returns:
            Nombre total de paramètres
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_robust_model(input_size=600, dropout_rate=0.3, use_attention=True):
    """
    Crée un modèle robuste avec les paramètres optimaux
    
    Args:
        input_size: Taille d'entrée
        dropout_rate: Taux de dropout
        use_attention: Utiliser l'attention
    
    Returns:
        Modèle robuste initialisé
    """
    model = RobustDualParameterModel(
        input_size=input_size,
        dropout_rate=dropout_rate,
        use_attention=use_attention
    )
    
    print(f"🛡️ Modèle robuste créé:")
    print(f"   Paramètres: {model.count_parameters():,}")
    print(f"   Dropout: {dropout_rate}")
    print(f"   Attention: {use_attention}")
    
    return model

def main():
    """
    Test de l'architecture robuste
    """
    print("🧪 Test de l'architecture robuste")
    
    # Créer le modèle
    model = create_robust_model()
    
    # Test avec données factices
    batch_size = 32
    input_size = 600
    x = torch.randn(batch_size, input_size)
    
    # Forward pass
    predictions = model(x)
    
    print(f"\n✅ Test réussi:")
    print(f"   Entrée: {x.shape}")
    print(f"   Sortie: {predictions.shape}")
    print(f"   Paramètres: {model.count_parameters():,}")
    
    # Test de la fonction de perte
    targets = torch.randn(batch_size, 2)
    loss_fn = RobustLoss()
    loss = loss_fn(predictions, targets)
    
    print(f"   Perte test: {loss.item():.6f}")
    
    print(f"\n🎉 Architecture robuste validée !")

if __name__ == "__main__":
    main()
