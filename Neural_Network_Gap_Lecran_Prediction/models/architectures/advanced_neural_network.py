#!/usr/bin/env python3
"""
Architecture de réseau de neurones hybride ultra-sophistiquée
Auteur: Oussama GUELFAA
Date: 15/07/2025

Ce module implémente une architecture hybride combinant:
- CNN 1D multi-échelle avec kernels de différentes tailles
- Blocs résiduels (ResNet-like) pour éviter le vanishing gradient
- Mécanismes d'attention (self-attention) pour capturer les dépendances longues
- Encoder-Decoder 1D pour modéliser finement la structure du signal
- Techniques de régularisation avancées
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Dict, List

class PositionalEncoding(nn.Module):
    """Encodage positionnel pour les mécanismes d'attention"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiScaleConv1D(nn.Module):
    """Module CNN 1D multi-échelle avec kernels de différentes tailles"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_sizes: List[int] = [3, 5, 7, 11]):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(kernel_sizes), 
                     kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # Appliquer chaque convolution
        conv_outputs = [conv(x) for conv in self.convs]
        
        # Concaténer les sorties
        x = torch.cat(conv_outputs, dim=1)
        
        # Normalisation et activation
        x = self.bn(x)
        x = self.activation(x)
        
        return x

class ResidualBlock1D(nn.Module):
    """Bloc résiduel 1D avec normalisation et dropout adaptatif"""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        
        # Première convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Deuxième convolution
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Connexion résiduelle
        out += residual
        out = self.activation(out)
        
        return out

class SelfAttention1D(nn.Module):
    """Mécanisme d'auto-attention pour capturer les dépendances longues"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.size()
        
        # Connexion résiduelle
        residual = x
        
        # Calcul des Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Application de l'attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Projection finale
        output = self.out_linear(attended)
        
        # Connexion résiduelle et normalisation
        output = self.layer_norm(output + residual)
        
        return output

class EncoderBlock(nn.Module):
    """Bloc d'encodeur combinant convolution, résiduel et attention"""
    
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.multi_scale_conv = MultiScaleConv1D(channels, channels)
        self.residual_block = ResidualBlock1D(channels, dropout=dropout)
        self.attention = SelfAttention1D(channels, num_heads, dropout)
        
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        
        # Convolution multi-échelle
        x = self.multi_scale_conv(x)
        
        # Bloc résiduel
        x = self.residual_block(x)
        
        # Préparer pour l'attention (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Auto-attention
        x = self.attention(x)
        
        # Retour au format convolutionnel
        x = x.transpose(1, 2)
        
        return x

class AdvancedHybridNetwork(nn.Module):
    """
    Architecture hybride ultra-sophistiquée pour prédiction Gap/L_écran
    
    Combine:
    - CNN 1D multi-échelle
    - Blocs résiduels
    - Mécanismes d'attention
    - Encoder-Decoder
    - Techniques de régularisation avancées
    """
    
    def __init__(self, input_size: int = 601, output_size: int = 2,
                 base_channels: int = 64, num_encoder_blocks: int = 6,
                 num_heads: int = 8, dropout: float = 0.1,
                 use_positional_encoding: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.use_positional_encoding = use_positional_encoding
        
        # Embedding initial
        self.input_embedding = nn.Conv1d(1, base_channels, kernel_size=7, padding=3)
        self.input_norm = nn.BatchNorm1d(base_channels)
        
        # Encodage positionnel
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(base_channels, input_size)
        
        # Encodeur multi-blocs
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(base_channels * (2**i), num_heads, dropout)
            for i in range(num_encoder_blocks)
        ])
        
        # Couches de downsampling entre les blocs
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(base_channels * (2**i), base_channels * (2**(i+1)), 
                         kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(base_channels * (2**(i+1))),
                nn.GELU()
            )
            for i in range(num_encoder_blocks - 1)
        ])
        
        # Calcul de la taille après downsampling
        final_seq_len = input_size
        for _ in range(num_encoder_blocks - 1):
            final_seq_len = (final_seq_len + 1) // 2
        
        final_channels = base_channels * (2**(num_encoder_blocks - 1))
        
        # Couches de classification/régression
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, final_channels // 2),
            nn.LayerNorm(final_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(final_channels // 2, final_channels // 4),
            nn.LayerNorm(final_channels // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(final_channels // 4, final_channels // 8),
            nn.LayerNorm(final_channels // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(final_channels // 8, output_size)
        )
        
        # Initialisation des poids
        self._initialize_weights()
        
        print(f"🧠 AdvancedHybridNetwork initialisé:")
        print(f"   📊 Entrée: {input_size} points")
        print(f"   📊 Sortie: {output_size} paramètres")
        print(f"   🏗️ Blocs encodeur: {num_encoder_blocks}")
        print(f"   🧠 Têtes d'attention: {num_heads}")
        print(f"   📈 Canaux de base: {base_channels}")
        print(f"   🔧 Dropout: {dropout}")
        
    def _initialize_weights(self):
        """Initialisation des poids avec Xavier/He"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len) ou (batch_size, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Ajouter dimension des canaux
        
        # Embedding initial
        x = self.input_embedding(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # Passage à travers les blocs encodeur
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            
            # Downsampling (sauf pour le dernier bloc)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)
        
        # Pooling global et classification
        x = self.global_pool(x)  # (batch_size, channels, 1)
        x = x.squeeze(-1)  # (batch_size, channels)
        
        # Classification finale
        output = self.classifier(x)
        
        return output
    
    def get_attention_weights(self, x):
        """Extrait les poids d'attention pour visualisation"""
        attention_weights = []
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_embedding(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        for i, encoder_block in enumerate(self.encoder_blocks):
            # Récupérer les poids d'attention du bloc
            x_temp = encoder_block.multi_scale_conv(x)
            x_temp = encoder_block.residual_block(x_temp)
            x_temp = x_temp.transpose(1, 2)
            
            # Calculer l'attention manuellement pour récupérer les poids
            embed_dim = x_temp.size(-1)
            num_heads = encoder_block.attention.num_heads
            head_dim = embed_dim // num_heads
            
            Q = encoder_block.attention.q_linear(x_temp).view(x_temp.size(0), x_temp.size(1), num_heads, head_dim).transpose(1, 2)
            K = encoder_block.attention.k_linear(x_temp).view(x_temp.size(0), x_temp.size(1), num_heads, head_dim).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
            weights = F.softmax(scores, dim=-1)
            
            attention_weights.append(weights.detach())
            
            # Continuer le forward
            x = encoder_block(x)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)
        
        return attention_weights

def create_model_variants() -> Dict[str, AdvancedHybridNetwork]:
    """Crée différentes variantes du modèle pour expérimentation"""
    
    variants = {
        'lightweight': AdvancedHybridNetwork(
            base_channels=32, num_encoder_blocks=4, num_heads=4, dropout=0.1
        ),
        'standard': AdvancedHybridNetwork(
            base_channels=64, num_encoder_blocks=6, num_heads=8, dropout=0.1
        ),
        'heavy': AdvancedHybridNetwork(
            base_channels=128, num_encoder_blocks=8, num_heads=16, dropout=0.15
        ),
        'ultra_deep': AdvancedHybridNetwork(
            base_channels=96, num_encoder_blocks=12, num_heads=12, dropout=0.2
        )
    }
    
    return variants

def main():
    """Fonction principale de test"""
    print("🧠 AdvancedHybridNetwork - Test d'architecture")
    print("=" * 50)
    
    # Créer les variantes du modèle
    models = create_model_variants()
    
    # Tester chaque variante
    batch_size = 4
    input_size = 601
    x_test = torch.randn(batch_size, input_size)
    
    for name, model in models.items():
        print(f"\n🔍 Test du modèle '{name}':")
        
        # Forward pass
        with torch.no_grad():
            output = model(x_test)
            print(f"   📊 Sortie: {output.shape}")
            print(f"   📈 Prédictions exemple: {output[0].numpy()}")
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   🔧 Paramètres totaux: {total_params:,}")
        print(f"   🔧 Paramètres entraînables: {trainable_params:,}")
    
    print("\n✅ Test des architectures terminé!")

if __name__ == "__main__":
    main()
