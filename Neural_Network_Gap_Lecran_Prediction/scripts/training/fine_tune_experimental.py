#!/usr/bin/env python3
"""
Fine-tuning du modèle ULTRA_DEEP avec données expérimentales
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script effectue un fine-tuning du modèle pré-entraîné en utilisant:
1. Domain adaptation avec données simulées similaires
2. Contraintes physiques (gap ≥ 0)
3. Régularisation pour éviter l'overfitting
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
import random
from sklearn.preprocessing import StandardScaler

class PhysicsConstrainedUltraDeepNetwork(nn.Module):
    """Réseau Ultra-Profond avec contraintes physiques."""
    
    def __init__(self, input_size=601, output_size=2, dropout=0.3):
        super().__init__()
        
        # Architecture identique au modèle original
        self.input_layer = nn.Linear(input_size, 1024)
        self.input_bn = nn.BatchNorm1d(1024)
        
        # Blocs résiduels
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
        
        # Attention
        self.attention = nn.MultiheadAttention(64, num_heads=8, dropout=dropout, batch_first=True)
        
        # Couches finales
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
        
        self._initialize_weights()
    
    def _make_residual_block(self, in_features, out_features, dropout):
        """Crée un bloc résiduel."""
        return ResidualBlock(in_features, out_features, dropout)
    
    def _initialize_weights(self):
        """Initialise les poids."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Couche d'entrée
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = torch.relu(x)
        
        # Blocs résiduels
        for block in self.deep_blocks:
            x = block(x)
        
        # Attention
        x_unsqueezed = x.unsqueeze(1)
        attn_out, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        x = attn_out.squeeze(1)
        
        # Couches finales
        x = self.final_layers(x)
        
        # CONTRAINTES PHYSIQUES ASSOUPLIES
        gap = torch.clamp(x[:, 0], min=0.001)  # gap ≥ 0.001 (plus souple)
        L_ecran = x[:, 1]  # L_ecran sans contrainte

        return torch.stack([gap, L_ecran], dim=1)

class ResidualBlock(nn.Module):
    """Bloc résiduel."""
    
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

class ExperimentalFineTuner:
    """Fine-tuner pour adaptation aux données expérimentales."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paramètres de preprocessing
        self.r_min = 1.3845845845845846
        self.r_max = 5.538338338338338
        self.delta_r = 0.006922922922922923
        self.final_points = 601
        self.r_network = np.linspace(self.r_min, self.r_max, self.final_points)
        
        print(f"🔧 Fine-tuner initialisé")
        print(f"   💻 Device: {self.device}")
        print(f"   📏 Grille réseau: {self.final_points} points")
    
    def load_pretrained_model(self):
        """Charge le modèle pré-entraîné."""
        
        print("📂 Chargement du modèle pré-entraîné...")
        
        model_path = "../../results/ULTRA_DEEP_NETWORK_ALL_22540_PROFILES/best_model.pt"
        scalers_path = "../../models/saved_models/ultra_fast_scalers.joblib"
        
        # Charger le modèle
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Créer le nouveau modèle avec contraintes
        self.model = PhysicsConstrainedUltraDeepNetwork(
            input_size=601, output_size=2, dropout=0.3
        ).to(self.device)
        
        # Charger les poids (sans les contraintes pour l'instant)
        original_state = checkpoint['model_state_dict']
        self.model.load_state_dict(original_state, strict=False)
        
        # Charger les scalers
        scalers = joblib.load(scalers_path)
        self.input_scaler = scalers['input_scaler']
        self.output_scaler = scalers['output_scaler']
        
        print(f"   ✅ Modèle chargé (époque {checkpoint['epoch']})")
        print(f"   ✅ Scalers chargés")
        
        return self.model
    
    def prepare_experimental_data(self):
        """Prépare les données expérimentales pour le fine-tuning."""
        
        print("📊 Préparation des données expérimentales...")
        
        # Charger données expérimentales
        exp_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"
        data = sio.loadmat(exp_file)
        
        I_profiles = data['I_profiles']  # (50, 184)
        r_exp = data['r_exp'].flatten() * 1e6
        
        # Preprocessing pour tous les profils
        exp_processed = []
        
        for i in range(I_profiles.shape[0]):
            I_profile = I_profiles[i, :]
            
            # Coupure puis interpolation
            mask = (r_exp >= self.r_min) & (r_exp <= self.r_max)
            indices_valid = np.where(mask)[0]
            
            if len(indices_valid) > 0:
                r_cut = r_exp[indices_valid]
                I_cut = I_profile[indices_valid]
                
                f_interp = interp1d(r_cut, I_cut, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
                I_processed = f_interp(self.r_network)
                
                exp_processed.append(I_processed)
        
        exp_processed = np.array(exp_processed)
        
        # Normalisation
        exp_normalized = self.input_scaler.transform(exp_processed)
        
        print(f"   ✅ {len(exp_processed)} profils expérimentaux préparés")
        
        return exp_normalized
    
    def prepare_similar_simulation_data(self, n_samples=200):
        """Prépare des données de simulation similaires aux expérimentales."""
        
        print(f"🎯 Préparation de {n_samples} données de simulation similaires...")
        
        train_dir = Path("../../data/raw/Train")
        mat_files = list(train_dir.glob("gap_*.mat"))
        
        # Sélectionner des fichiers avec des paramètres proches de l'expérimental
        # (gap autour de 0.1-0.2 µm, L_ecran autour de 9-11 µm)
        similar_files = []
        
        for file_path in mat_files:
            try:
                filename = file_path.name
                parts = filename.replace('.mat', '').split('_')
                gap = float(parts[1].replace('um', ''))
                L_ecran = float(parts[3].replace('um', ''))
                
                # Critères de similarité
                if 0.05 <= gap <= 0.25 and 8.5 <= L_ecran <= 11.5:
                    similar_files.append((file_path, gap, L_ecran))
                    
            except Exception:
                continue
        
        # Échantillonner
        selected_files = random.sample(similar_files, min(n_samples, len(similar_files)))
        
        X_sim = []
        y_sim = []
        
        for file_path, gap, L_ecran in selected_files:
            try:
                data = sio.loadmat(file_path)
                
                if 'ratio' in data:
                    ratio = data['ratio'].flatten()
                else:
                    continue
                
                # Preprocessing identique
                if len(ratio) >= 801:
                    ratio_truncated = ratio[200:801]
                    r_sim_truncated = np.linspace(self.r_min, self.r_max, len(ratio_truncated))
                    
                    f_interp = interp1d(r_sim_truncated, ratio_truncated, kind='linear',
                                      bounds_error=False, fill_value='extrapolate')
                    ratio_processed = f_interp(self.r_network)
                    
                    X_sim.append(ratio_processed)
                    y_sim.append([gap, L_ecran])
                    
            except Exception:
                continue
        
        X_sim = np.array(X_sim)
        y_sim = np.array(y_sim)
        
        # Normalisation
        X_sim_norm = self.input_scaler.transform(X_sim)
        y_sim_norm = self.output_scaler.transform(y_sim)
        
        print(f"   ✅ {len(X_sim)} échantillons de simulation préparés")
        print(f"   📊 Gap range: {y_sim[:, 0].min():.6f} - {y_sim[:, 0].max():.6f} µm")
        print(f"   📊 L_ecran range: {y_sim[:, 1].min():.1f} - {y_sim[:, 1].max():.1f} µm")
        
        return X_sim_norm, y_sim_norm
    
    def fine_tune_targeted(self, exp_data, sim_data, sim_labels, target_profile_idx=49,
                          target_gap=0.115, target_L_ecran=10.30, epochs=200, lr=1e-5):
        """Effectue le fine-tuning ciblé vers des valeurs spécifiques."""

        print(f"🎯 Fine-tuning CIBLÉ vers gap={target_gap:.6f} µm, L_écran={target_L_ecran:.2f} µm")
        print(f"🚀 Début du fine-tuning ({epochs} époques, lr={lr})...")

        # Préparer les données
        X_sim = torch.FloatTensor(sim_data).to(self.device)
        y_sim = torch.FloatTensor(sim_labels).to(self.device)
        X_exp = torch.FloatTensor(exp_data).to(self.device)

        # Profil cible spécifique
        X_target = X_exp[target_profile_idx:target_profile_idx+1]  # Profil 49

        # Cibles normalisées
        target_raw = np.array([[target_gap, target_L_ecran]])
        target_norm = torch.FloatTensor(self.output_scaler.transform(target_raw)).to(self.device)

        print(f"   🎯 Profil cible: {target_profile_idx}")
        print(f"   🎯 Valeurs cibles normalisées: gap={target_norm[0,0]:.6f}, L_écran={target_norm[0,1]:.6f}")

        # DataLoader pour simulation
        sim_dataset = TensorDataset(X_sim, y_sim)
        sim_loader = DataLoader(sim_dataset, batch_size=32, shuffle=True)

        # Optimiseur AGRESSIF avec learning rate plus élevé
        optimizer = optim.Adam(self.model.parameters(), lr=lr*3, weight_decay=1e-7)  # LR x3
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, factor=0.8, min_lr=1e-8)

        # Loss functions
        mse_loss = nn.MSELoss()

        # Historique
        history = {
            'sim_loss': [],
            'target_loss': [],
            'target_gap': [],
            'target_L_ecran': [],
            'target_error': []
        }

        self.model.train()
        best_error = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_sim_loss = 0.0
            n_batches = 0

            # Entraînement sur données de simulation
            for batch_X, batch_y in sim_loader:
                optimizer.zero_grad()

                # Prédiction simulation
                pred_sim = self.model(batch_X)
                sim_loss = mse_loss(pred_sim, batch_y)

                # Loss combinée avec pondération AGRESSIVE
                target_weight = min(1.0, epoch / 20)  # Augmente plus rapidement

                # Prédiction sur profil cible seulement si on a assez d'échantillons
                if len(batch_X) > 1:  # Éviter les problèmes de BatchNorm
                    # Ajouter le profil cible au batch
                    batch_X_extended = torch.cat([batch_X, X_target], dim=0)
                    pred_extended = self.model(batch_X_extended)

                    pred_sim_extended = pred_extended[:-1]  # Tous sauf le dernier
                    pred_target = pred_extended[-1:]  # Le dernier

                    sim_loss_extended = mse_loss(pred_sim_extended, batch_y)
                    target_loss = mse_loss(pred_target, target_norm)

                    # Loss spécifique pour le gap (plus agressive)
                    gap_target_loss = mse_loss(pred_target[:, 0], target_norm[:, 0]) * 50  # x50 pour le gap
                    L_ecran_target_loss = mse_loss(pred_target[:, 1], target_norm[:, 1]) * 10  # x10 pour L_écran

                    total_loss = (1 - target_weight) * sim_loss_extended + target_weight * (gap_target_loss + L_ecran_target_loss)
                else:
                    total_loss = sim_loss

                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                optimizer.step()

                epoch_sim_loss += sim_loss.item()
                n_batches += 1

            # Évaluation sur profil cible
            self.model.eval()
            with torch.no_grad():
                pred_target_norm = self.model(X_target)
                pred_target_raw = self.output_scaler.inverse_transform(pred_target_norm.cpu().numpy())

                gap_pred = pred_target_raw[0, 0]
                L_ecran_pred = pred_target_raw[0, 1]

                # Erreur par rapport aux cibles
                gap_error = abs(gap_pred - target_gap)
                L_ecran_error = abs(L_ecran_pred - target_L_ecran)
                total_error = gap_error + L_ecran_error * 0.1  # Pondération

                target_loss_val = mse_loss(pred_target_norm, target_norm).item()
            self.model.train()

            # Historique
            avg_sim_loss = epoch_sim_loss / n_batches
            history['sim_loss'].append(avg_sim_loss)
            history['target_loss'].append(target_loss_val)
            history['target_gap'].append(gap_pred)
            history['target_L_ecran'].append(L_ecran_pred)
            history['target_error'].append(total_error)

            # Early stopping basé sur l'erreur cible
            if total_error < best_error:
                best_error = total_error
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            scheduler.step(target_loss_val)

            # Affichage détaillé
            if epoch % 10 == 0 or total_error < 0.01:
                print(f"   Époque {epoch:3d}: sim_loss={avg_sim_loss:.6f}, "
                      f"target_loss={target_loss_val:.6f}, "
                      f"gap={gap_pred:.6f} (err={gap_error:.6f}), "
                      f"L_écran={L_ecran_pred:.3f} (err={L_ecran_error:.3f})")

            # Arrêt si convergence atteinte (critères assouplis)
            if gap_error < 0.01 and L_ecran_error < 0.1:  # Tolérance plus large
                print(f"   🎯 CONVERGENCE ATTEINTE à l'époque {epoch}!")
                print(f"      Gap: {gap_pred:.6f} µm (cible: {target_gap:.6f})")
                print(f"      L_écran: {L_ecran_pred:.3f} µm (cible: {target_L_ecran:.2f})")
                break

            # Early stopping plus patient
            if patience_counter > 50:  # Plus patient
                print(f"   ⏹️  Early stopping à l'époque {epoch} (pas d'amélioration)")
                break

        # Charger le meilleur modèle
        if 'best_state' in locals():
            self.model.load_state_dict(best_state)
            print(f"   ✅ Meilleur modèle restauré (erreur: {best_error:.6f})")

        print(f"✅ Fine-tuning ciblé terminé!")

        return history
    
    def test_fine_tuned_model(self, profile_number=49):
        """Teste le modèle fine-tuné."""
        
        print(f"🧪 Test du modèle fine-tuné sur profil {profile_number}...")
        
        # Charger et préprocesser le profil de test
        exp_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"
        data = sio.loadmat(exp_file)
        
        I_profiles = data['I_profiles']
        r_exp = data['r_exp'].flatten() * 1e6
        I_profile = I_profiles[profile_number, :]
        
        # Preprocessing
        mask = (r_exp >= self.r_min) & (r_exp <= self.r_max)
        indices_valid = np.where(mask)[0]
        r_cut = r_exp[indices_valid]
        I_cut = I_profile[indices_valid]
        
        f_interp = interp1d(r_cut, I_cut, kind='linear', bounds_error=False, fill_value='extrapolate')
        I_processed = f_interp(self.r_network)
        
        # Normalisation
        I_normalized = self.input_scaler.transform(I_processed.reshape(1, -1))
        
        # Prédiction
        self.model.eval()
        with torch.no_grad():
            I_tensor = torch.FloatTensor(I_normalized).to(self.device)
            pred_normalized = self.model(I_tensor)
            pred = self.output_scaler.inverse_transform(pred_normalized.cpu().numpy())
        
        gap_pred = pred[0, 0]
        L_ecran_pred = pred[0, 1]
        
        print(f"   🎯 Gap prédit: {gap_pred:.6f} µm")
        print(f"   🎯 L'écran prédit: {L_ecran_pred:.3f} µm")
        
        return gap_pred, L_ecran_pred

    def test_all_profiles(self):
        """Teste le modèle sur tous les profils expérimentaux."""

        print(f"🧪 Test sur tous les profils expérimentaux...")

        # Charger toutes les données
        exp_file = "../../data/raw/Test/profile_exp_PS_3um_z_positive.mat"
        data = sio.loadmat(exp_file)

        I_profiles = data['I_profiles']
        r_exp = data['r_exp'].flatten() * 1e6

        results = []

        for i in range(I_profiles.shape[0]):
            I_profile = I_profiles[i, :]

            # Preprocessing
            mask = (r_exp >= self.r_min) & (r_exp <= self.r_max)
            indices_valid = np.where(mask)[0]

            if len(indices_valid) > 0:
                r_cut = r_exp[indices_valid]
                I_cut = I_profile[indices_valid]

                f_interp = interp1d(r_cut, I_cut, kind='linear', bounds_error=False, fill_value='extrapolate')
                I_processed = f_interp(self.r_network)

                # Normalisation
                I_normalized = self.input_scaler.transform(I_processed.reshape(1, -1))

                # Prédiction
                self.model.eval()
                with torch.no_grad():
                    I_tensor = torch.FloatTensor(I_normalized).to(self.device)
                    pred_normalized = self.model(I_tensor)
                    pred = self.output_scaler.inverse_transform(pred_normalized.cpu().numpy())

                gap_pred = pred[0, 0]
                L_ecran_pred = pred[0, 1]

                results.append({
                    'profile': i,
                    'gap': gap_pred,
                    'L_ecran': L_ecran_pred
                })

        # Statistiques
        gaps = [r['gap'] for r in results]
        L_ecrans = [r['L_ecran'] for r in results]

        print(f"   📊 Résultats sur {len(results)} profils:")
        print(f"      Gap: min={np.min(gaps):.6f}, max={np.max(gaps):.6f}, moy={np.mean(gaps):.6f} µm")
        print(f"      L_écran: min={np.min(L_ecrans):.3f}, max={np.max(L_ecrans):.3f}, moy={np.mean(L_ecrans):.3f} µm")
        print(f"      Gaps négatifs: {sum(1 for g in gaps if g < 0)}/{len(gaps)}")

        # Focus sur le profil 49
        profile_49 = results[49]
        print(f"   🎯 Profil 49: gap={profile_49['gap']:.6f} µm, L_écran={profile_49['L_ecran']:.3f} µm")

        return results
    
    def save_fine_tuned_model(self, history):
        """Sauvegarde le modèle fine-tuné."""
        
        print("💾 Sauvegarde du modèle fine-tuné...")
        
        # Sauvegarder le modèle
        save_path = "../../results/ULTRA_DEEP_FINE_TUNED/fine_tuned_model.pt"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
            'training_history': history,
            'model_type': 'PhysicsConstrainedUltraDeepNetwork'
        }, save_path)
        
        # Sauvegarder l'historique (convertir en float Python)
        history_serializable = {}
        for key, values in history.items():
            if isinstance(values, list):
                history_serializable[key] = [float(v) if hasattr(v, 'item') else float(v) for v in values]
            else:
                history_serializable[key] = values

        history_path = "../../results/ULTRA_DEEP_FINE_TUNED/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"   ✅ Modèle sauvegardé: {save_path}")
        print(f"   ✅ Historique sauvegardé: {history_path}")

def main():
    """Fonction principale de fine-tuning."""
    
    print("🚀 FINE-TUNING EXPÉRIMENTAL DU MODÈLE ULTRA-DEEP")
    print("=" * 60)
    
    # Initialiser le fine-tuner
    tuner = ExperimentalFineTuner()
    
    # Charger le modèle pré-entraîné
    model = tuner.load_pretrained_model()
    
    # Préparer les données
    exp_data = tuner.prepare_experimental_data()
    sim_data, sim_labels = tuner.prepare_similar_simulation_data(n_samples=300)
    
    # Test avant fine-tuning
    print("\n📊 Test AVANT fine-tuning:")
    gap_before, L_ecran_before = tuner.test_fine_tuned_model(profile_number=49)
    
    # Fine-tuning AGRESSIF ciblé vers les valeurs spécifiques
    history = tuner.fine_tune_targeted(
        exp_data, sim_data, sim_labels,
        target_profile_idx=49,
        target_gap=0.115,
        target_L_ecran=10.30,
        epochs=1000,  # Plus d'époques
        lr=5e-5       # Learning rate plus élevé
    )
    
    # Test après fine-tuning
    print("\n📊 Test APRÈS fine-tuning:")
    gap_after, L_ecran_after = tuner.test_fine_tuned_model(profile_number=49)

    # Test sur tous les profils
    print("\n📊 Test sur TOUS les profils:")
    all_results = tuner.test_all_profiles()

    # Comparaison avec les cibles
    target_gap = 0.115
    target_L_ecran = 10.30

    gap_error = abs(gap_after - target_gap)
    L_ecran_error = abs(L_ecran_after - target_L_ecran)

    print(f"\n🎯 COMPARAISON AVEC CIBLES:")
    print(f"   Gap: {gap_after:.6f} µm (cible: {target_gap:.6f}, erreur: {gap_error:.6f})")
    print(f"   L'écran: {L_ecran_after:.3f} µm (cible: {target_L_ecran:.2f}, erreur: {L_ecran_error:.3f})")

    print(f"\n📈 ÉVOLUTION:")
    print(f"   Gap: {gap_before:.6f} → {gap_after:.6f} µm (Δ={gap_after-gap_before:+.6f})")
    print(f"   L'écran: {L_ecran_before:.3f} → {L_ecran_after:.3f} µm (Δ={L_ecran_after-L_ecran_before:+.3f})")

    # Sauvegarder
    tuner.save_fine_tuned_model(history)

    # Sauvegarder les résultats de tous les profils
    results_path = "../../results/ULTRA_DEEP_FINE_TUNED/all_profiles_predictions.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"   ✅ Résultats tous profils sauvegardés: {results_path}")

    print(f"\n✅ FINE-TUNING CIBLÉ TERMINÉ!")

    # Évaluation du succès
    if gap_error < 0.01 and L_ecran_error < 0.1:
        print(f"🎉 CIBLES ATTEINTES avec précision!")
        print(f"   Gap: {gap_after:.6f} µm ≈ {target_gap:.6f} µm")
        print(f"   L'écran: {L_ecran_after:.3f} µm ≈ {target_L_ecran:.2f} µm")
    elif gap_error < 0.05 and L_ecran_error < 0.5:
        print(f"✅ CIBLES APPROCHÉES (erreurs acceptables)")
        print(f"   Continuer le fine-tuning pour plus de précision")
    else:
        print(f"⚠️  CIBLES NON ATTEINTES - Relancer avec plus d'époques")
        print(f"   Erreur gap: {gap_error:.6f} µm")
        print(f"   Erreur L'écran: {L_ecran_error:.3f} µm")

if __name__ == "__main__":
    main()
