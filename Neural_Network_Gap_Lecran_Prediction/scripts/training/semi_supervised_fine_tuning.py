#!/usr/bin/env python3
"""
Semi-Supervised Fine-Tuning du modèle Ultra-Profond
Auteur: Oussama GUELFAA
Date: 15/07/2025

Fine-tuning semi-supervisé pour adapter le modèle ultra-profond
aux données expérimentales sans labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import time
import json
from pathlib import Path
from sklearn.metrics import r2_score
import scipy.io

from ultra_fast_training import UltraDeepNetwork
from test_ultra_deep_on_experimental import ExperimentalDataProcessor

class SemiSupervisedTrainer:
    """
    Entraîneur semi-supervisé pour adaptation de domaine.
    """
    
    def __init__(self, model_path="results/ULTRA_DEEP_NETWORK_ALL_22540_PROFILES/best_model.pt"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path("results/semi_supervised_light_optimizations")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧠 SemiSupervisedTrainer initialisé")
        print(f"   🖥️ Device: {self.device}")
        print(f"   📁 Résultats: {self.results_dir}")
    
    def load_pretrained_model(self):
        """Charge le modèle ultra-profond pré-entraîné."""
        print("📂 Chargement du modèle ultra-profond pré-entraîné...")
        
        # Charger le checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Créer le modèle
        self.model = UltraDeepNetwork(
            input_size=601,
            output_size=2,
            dropout=0.3
        ).to(self.device)
        
        # Charger les poids pré-entraînés
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Charger les scalers
        scalers = joblib.load("ultra_fast_scalers.joblib")
        self.input_scaler = scalers['input_scaler']
        self.output_scaler = scalers['output_scaler']
        # Alias pour compatibilité
        self.scaler_X = self.input_scaler
        self.scaler_y = self.output_scaler
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   ✅ Modèle chargé: {total_params:,} paramètres")
        print(f"   📊 Performance initiale: {checkpoint['val_loss']:.6f} loss")
    
    def load_simulation_data(self, n_samples=5000):
        """Charge un échantillon des données de simulation pour le fine-tuning."""
        print(f"📂 Chargement de {n_samples} échantillons de simulation...")

        # Charger les données extraites
        data = np.load('extracted_data_full.npz')
        X_sim = data['X_data']  # (22540, 601)
        y_sim = data['y_data']  # (22540, 2)

        # Si on demande toutes les données, utiliser un échantillonnage stratifié
        if n_samples >= len(X_sim):
            print("   🎯 Utilisation de TOUTES les données avec optimisation mémoire")
            self.X_sim = X_sim
            self.y_sim = y_sim
            self.use_full_dataset = True
        else:
            # Échantillonnage aléatoire
            np.random.seed(42)
            indices = np.random.choice(len(X_sim), n_samples, replace=False)
            self.X_sim = X_sim[indices]
            self.y_sim = y_sim[indices]
            self.use_full_dataset = False

        print(f"   ✅ Données simulation: {self.X_sim.shape}")
        print(f"   📈 Gap range: [{self.y_sim[:, 0].min():.6f}, {self.y_sim[:, 0].max():.6f}] µm")
    
    def load_experimental_data(self):
        """Charge et prétraite toutes les données expérimentales."""
        print("📂 Chargement des données expérimentales...")
        
        # Utiliser le processeur existant
        processor = ExperimentalDataProcessor()
        processor.extract_train_parameters()
        
        # Charger tous les profils expérimentaux
        exp_profiles = []
        
        for profile_num in range(1, 50):  # Profils 1-49
            try:
                r_exp, I_profile = processor.load_experimental_data(profile_num)
                processed_profile = processor.preprocess_experimental_profile(r_exp, I_profile)
                exp_profiles.append(processed_profile)
            except Exception as e:
                print(f"   ⚠️ Erreur profil {profile_num}: {e}")
        
        self.X_exp = np.array(exp_profiles)
        
        print(f"   ✅ Données expérimentales: {self.X_exp.shape}")
        print(f"   📊 Profils chargés: {len(exp_profiles)}")
    
    def compute_mmd_loss(self, features_sim, features_exp):
        """
        Calcule la Maximum Mean Discrepancy (MMD) loss pour l'alignement de features.
        """
        def gaussian_kernel(x, y, sigma=1.0):
            """Noyau gaussien pour MMD."""
            dist = torch.cdist(x, y, p=2)
            return torch.exp(-dist**2 / (2 * sigma**2))
        
        # Calculer les moyennes des noyaux
        K_xx = gaussian_kernel(features_sim, features_sim).mean()
        K_yy = gaussian_kernel(features_exp, features_exp).mean()
        K_xy = gaussian_kernel(features_sim, features_exp).mean()
        
        # MMD²
        mmd_loss = K_xx + K_yy - 2 * K_xy
        
        return mmd_loss
    
    def get_feature_extractor(self):
        """Extrait la partie feature extraction du modèle."""
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.input_layer = model.input_layer
                self.input_bn = model.input_bn
                self.deep_blocks = model.deep_blocks
                self.attention = model.attention
            
            def forward(self, x):
                # Couche d'entrée
                x = self.input_layer(x)
                x = self.input_bn(x)
                x = torch.relu(x)
                
                # Blocs résiduels profonds
                for block in self.deep_blocks:
                    x = block(x)
                
                # Attention mechanism
                x_unsqueezed = x.unsqueeze(1)
                attn_out, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
                x = attn_out.squeeze(1)
                
                return x
        
        return FeatureExtractor(self.model)
    
    def get_confident_predictions(self, X_exp, confidence_threshold=0.85):
        """
        Génère des pseudo-labels pour les prédictions les plus confiantes.
        """
        self.model.eval()
        
        # Normaliser les données expérimentales
        X_exp_norm = self.input_scaler.transform(X_exp)
        X_exp_tensor = torch.FloatTensor(X_exp_norm).to(self.device)
        
        with torch.no_grad():
            # Prédictions
            predictions_norm = self.model(X_exp_tensor)
            predictions = self.output_scaler.inverse_transform(predictions_norm.cpu().numpy())
            
            # Calculer la "confiance" basée sur la cohérence des prédictions
            # (ici, on utilise une heuristique simple)
            gaps = predictions[:, 0]
            L_ecrans = predictions[:, 1]
            
            # Critères de confiance :
            # 1. Gap positif (physiquement réaliste)
            # 2. L_écran dans une plage raisonnable
            # 3. Pas de valeurs extrêmes
            
            confidence_mask = (
                (gaps > 0.001) &  # Gap positif minimum
                (gaps < 1.0) &    # Gap maximum raisonnable
                (L_ecrans > 7.0) & (L_ecrans < 13.0)  # L_écran dans plage train
            )
            
            confident_indices = np.where(confidence_mask)[0]
            confident_X = X_exp[confident_indices]
            confident_y = predictions[confident_indices]
            
            print(f"   📊 Prédictions confiantes: {len(confident_indices)}/{len(X_exp)} ({len(confident_indices)/len(X_exp)*100:.1f}%)")
            
            return confident_X, confident_y, confident_indices
    
    def semi_supervised_fine_tuning(self, epochs=120, lambda_alignment=0.15, lambda_self_training=0.6):
        """
        Fine-tuning semi-supervisé principal.
        """
        print(f"🚀 Début du fine-tuning semi-supervisé: {epochs} époques")
        print(f"   λ_alignment: {lambda_alignment}")
        print(f"   λ_self_training: {lambda_self_training}")
        
        # Optimiseur avec learning rate légèrement optimisé pour fine-tuning
        optimizer = optim.AdamW(self.model.parameters(), lr=8e-5, weight_decay=2e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # Feature extractor
        feature_extractor = self.get_feature_extractor()
        
        # Loss functions
        mse_loss = nn.MSELoss()
        
        # Historique
        history = {
            'total_loss': [], 'supervised_loss': [], 'alignment_loss': [], 'self_training_loss': [],
            'epochs': []
        }
        
        best_alignment_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            self.model.train()
            feature_extractor.train()
            
            # Batch des données de simulation (légèrement augmenté)
            batch_size = 40
            sim_indices = np.random.choice(len(self.X_sim), batch_size, replace=False)
            X_sim_batch = torch.FloatTensor(self.input_scaler.transform(self.X_sim[sim_indices])).to(self.device)
            y_sim_batch = torch.FloatTensor(self.output_scaler.transform(self.y_sim[sim_indices])).to(self.device)
            
            # Batch des données expérimentales
            exp_indices = np.random.choice(len(self.X_exp), min(batch_size, len(self.X_exp)), replace=False)
            X_exp_batch = torch.FloatTensor(self.input_scaler.transform(self.X_exp[exp_indices])).to(self.device)
            
            optimizer.zero_grad()
            
            # 1. Loss supervisée sur simulation
            sim_predictions = self.model(X_sim_batch)
            supervised_loss = mse_loss(sim_predictions, y_sim_batch)
            
            # 2. Loss d'alignement de features
            features_sim = feature_extractor(X_sim_batch)
            features_exp = feature_extractor(X_exp_batch)
            alignment_loss = self.compute_mmd_loss(features_sim, features_exp)
            
            # 3. Self-training loss (tous les 8 epochs pour plus de fréquence)
            self_training_loss = torch.tensor(0.0).to(self.device)
            if epoch % 8 == 0:
                confident_X, confident_y, _ = self.get_confident_predictions(self.X_exp)
                if len(confident_X) > 0:
                    confident_X_norm = torch.FloatTensor(self.input_scaler.transform(confident_X)).to(self.device)
                    confident_y_norm = torch.FloatTensor(self.output_scaler.transform(confident_y)).to(self.device)
                    confident_predictions = self.model(confident_X_norm)
                    self_training_loss = mse_loss(confident_predictions, confident_y_norm)
            
            # Loss totale
            total_loss = supervised_loss + lambda_alignment * alignment_loss + lambda_self_training * self_training_loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Historique
            history['total_loss'].append(total_loss.item())
            history['supervised_loss'].append(supervised_loss.item())
            history['alignment_loss'].append(alignment_loss.item())
            history['self_training_loss'].append(self_training_loss.item())
            history['epochs'].append(epoch + 1)
            
            # Affichage
            epoch_time = time.time() - epoch_start
            if (epoch + 1) % 10 == 0:
                print(f"Époque {epoch+1}/{epochs} ({epoch_time:.1f}s)")
                print(f"  Total: {total_loss.item():.6f}")
                print(f"  Supervised: {supervised_loss.item():.6f}")
                print(f"  Alignment: {alignment_loss.item():.6f}")
                print(f"  Self-training: {self_training_loss.item():.6f}")
            
            # Sauvegarde du meilleur modèle (basé sur alignment loss)
            if alignment_loss.item() < best_alignment_loss:
                best_alignment_loss = alignment_loss.item()
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'alignment_loss': alignment_loss.item(),
                    'total_loss': total_loss.item()
                }, self.results_dir / 'best_semi_supervised_model.pt')
        
        print(f"✅ Fine-tuning terminé")
        
        # Sauvegarder l'historique
        with open(self.results_dir / 'fine_tuning_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return history

    def semi_supervised_fine_tuning_full_data(self, epochs=150):
        """Fine-tuning semi-supervisé avec TOUTES les données de simulation."""
        print(f"\n🚀 Début du fine-tuning avec {len(self.X_sim)} données de simulation")

        # Optimiseur avec learning rate plus conservateur pour plus de données
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.8)

        # Paramètres adaptés pour dataset complet
        lambda_alignment = 0.12
        lambda_self_training = 0.55
        confidence_threshold = 0.85
        # Optimisation mémoire pour dataset complet
        if hasattr(self, 'use_full_dataset') and self.use_full_dataset:
            batch_size = 32  # Batch plus petit pour gérer la mémoire
            print("   💾 Mode optimisation mémoire activé (batch_size=32)")
        else:
            batch_size = 64  # Plus grand batch pour efficacité

        history = {'loss': [], 'supervised_loss': [], 'alignment_loss': [], 'self_training_loss': []}

        # Feature extractor
        feature_extractor = self.get_feature_extractor()

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {'total': [], 'supervised': [], 'alignment': [], 'self_training': []}

            # Nombre de batches basé sur les données de simulation
            n_batches = len(self.X_sim) // batch_size

            for batch_idx in range(n_batches):
                # Batch de données simulées
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(self.X_sim))

                X_sim_batch = torch.FloatTensor(self.X_sim[start_idx:end_idx])
                y_sim_batch = torch.FloatTensor(self.y_sim[start_idx:end_idx])

                # Batch de données expérimentales (cyclique si moins de données)
                exp_indices = np.random.choice(len(self.X_exp),
                                             min(batch_size, len(self.X_exp)),
                                             replace=len(self.X_exp) < batch_size)
                X_exp_batch = torch.FloatTensor(self.X_exp[exp_indices])

                optimizer.zero_grad()

                # 1. Loss supervisée sur simulation
                sim_predictions = self.model(X_sim_batch)
                supervised_loss = F.mse_loss(sim_predictions, y_sim_batch)

                # 2. Loss d'alignement de features
                features_sim = feature_extractor(X_sim_batch)
                features_exp = feature_extractor(X_exp_batch)
                alignment_loss = self.compute_mmd_loss(features_sim, features_exp)

                # 3. Self-training loss avec prédictions confiantes
                with torch.no_grad():
                    exp_predictions = self.model(X_exp_batch)
                    exp_predictions_np = exp_predictions.cpu().numpy()

                    # Dénormaliser pour vérifier la confiance
                    exp_pred_denorm = self.scaler_y.inverse_transform(exp_predictions_np)

                    # Critères de confiance plus stricts pour dataset complet
                    confident_mask = (
                        (exp_pred_denorm[:, 0] > 0) &  # Gap positif
                        (exp_pred_denorm[:, 0] < 0.15) &  # Gap raisonnable
                        (exp_pred_denorm[:, 1] > 6) &  # L_écran minimum
                        (exp_pred_denorm[:, 1] < 15) &  # L_écran maximum
                        (np.abs(exp_pred_denorm[:, 0] - np.mean(exp_pred_denorm[:, 0])) < 2 * np.std(exp_pred_denorm[:, 0])) &
                        (np.abs(exp_pred_denorm[:, 1] - np.mean(exp_pred_denorm[:, 1])) < 2 * np.std(exp_pred_denorm[:, 1]))
                    )

                if confident_mask.sum() > 0:
                    confident_predictions = exp_predictions[confident_mask]
                    confident_y_norm = exp_predictions[confident_mask].detach()
                    self_training_loss = F.mse_loss(confident_predictions, confident_y_norm)
                else:
                    self_training_loss = torch.tensor(0.0)

                # Loss totale
                total_loss = (supervised_loss +
                            lambda_alignment * alignment_loss +
                            lambda_self_training * self_training_loss)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                # Enregistrer les losses
                epoch_losses['total'].append(total_loss.item())
                epoch_losses['supervised'].append(supervised_loss.item())
                epoch_losses['alignment'].append(alignment_loss.item())
                epoch_losses['self_training'].append(self_training_loss.item() if isinstance(self_training_loss, torch.Tensor) else 0.0)

            # Moyennes des losses pour l'époque
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            history['loss'].append(avg_losses['total'])
            history['supervised_loss'].append(avg_losses['supervised'])
            history['alignment_loss'].append(avg_losses['alignment'])
            history['self_training_loss'].append(avg_losses['self_training'])

            scheduler.step(avg_losses['total'])

            if (epoch + 1) % 10 == 0:
                print(f"Époque {epoch+1}/{epochs}")
                print(f"  📊 Loss totale: {avg_losses['total']:.6f}")
                print(f"  🎯 Loss supervisée: {avg_losses['supervised']:.6f}")
                print(f"  🔄 Loss alignement: {avg_losses['alignment']:.6f}")
                print(f"  🧠 Loss self-training: {avg_losses['self_training']:.6f}")
                print(f"  📈 LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Sauvegarder le modèle fine-tuné
        model_path = self.results_dir / "model_fine_tuned_FULL_DATA.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'history': history,
            'training_samples': len(self.X_sim)
        }, model_path)

        print(f"\n✅ Modèle fine-tuné sauvegardé: {model_path}")

        # Sauvegarder l'historique d'entraînement
        history_path = self.results_dir / "training_history_FULL_DATA.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        return history

    def test_on_experimental_data(self):
        """Teste le modèle fine-tuné sur les données expérimentales."""
        print("🧪 Test du modèle fine-tuné sur données expérimentales...")
        
        self.model.eval()
        
        # Normaliser les données expérimentales
        X_exp_norm = self.input_scaler.transform(self.X_exp)
        X_exp_tensor = torch.FloatTensor(X_exp_norm).to(self.device)
        
        with torch.no_grad():
            predictions_norm = self.model(X_exp_tensor)
            predictions = self.output_scaler.inverse_transform(predictions_norm.cpu().numpy())
        
        # Analyser les résultats
        gaps = predictions[:, 0]
        L_ecrans = predictions[:, 1]
        
        negative_gaps = np.sum(gaps < 0)
        negative_percentage = negative_gaps / len(gaps) * 100
        
        results = {
            'total_profiles': len(predictions),
            'gap_min': float(gaps.min()),
            'gap_max': float(gaps.max()),
            'gap_mean': float(gaps.mean()),
            'gap_std': float(gaps.std()),
            'L_ecran_min': float(L_ecrans.min()),
            'L_ecran_max': float(L_ecrans.max()),
            'L_ecran_mean': float(L_ecrans.mean()),
            'L_ecran_std': float(L_ecrans.std()),
            'negative_gaps_count': int(negative_gaps),
            'negative_gaps_percentage': float(negative_percentage)
        }
        
        print(f"   📊 Résultats après fine-tuning:")
        print(f"      Gap range: [{gaps.min():.6f}, {gaps.max():.6f}] µm")
        print(f"      L_écran range: [{L_ecrans.min():.3f}, {L_ecrans.max():.3f}] µm")
        print(f"      Gaps négatifs: {negative_gaps}/{len(gaps)} ({negative_percentage:.1f}%)")
        
        # Sauvegarder les prédictions
        df = pd.DataFrame({
            'profile_number': range(1, len(predictions) + 1),
            'gap_predicted_um': gaps,
            'L_ecran_predicted_um': L_ecrans
        })
        
        df.to_csv(self.results_dir / 'experimental_predictions_fine_tuned.csv', index=False)
        
        return results, predictions
    
    def compare_results(self, results_before_file="all_profiles_test_results.csv"):
        """Compare les résultats avant et après fine-tuning."""
        print("📊 Comparaison avant/après fine-tuning...")
        
        try:
            # Charger les résultats avant fine-tuning
            df_before = pd.read_csv(results_before_file)
            gaps_before = df_before['gap_predicted_um'].values
            
            # Charger les résultats après fine-tuning
            df_after = pd.read_csv(self.results_dir / 'experimental_predictions_fine_tuned.csv')
            gaps_after = df_after['gap_predicted_um'].values
            
            # Comparaison
            negative_before = np.sum(gaps_before < 0)
            negative_after = np.sum(gaps_after < 0)
            
            print(f"   📈 Amélioration:")
            print(f"      Avant: {negative_before}/{len(gaps_before)} gaps négatifs ({negative_before/len(gaps_before)*100:.1f}%)")
            print(f"      Après: {negative_after}/{len(gaps_after)} gaps négatifs ({negative_after/len(gaps_after)*100:.1f}%)")
            print(f"      Réduction: {negative_before - negative_after} gaps négatifs")
            
            # Graphique de comparaison
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].hist(gaps_before, bins=20, alpha=0.7, color='red', label='Avant fine-tuning')
            axes[0].axvline(0, color='black', linestyle='--', alpha=0.7)
            axes[0].set_title('Avant Fine-Tuning')
            axes[0].set_xlabel('Gap Prédit (µm)')
            axes[0].set_ylabel('Fréquence')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].hist(gaps_after, bins=20, alpha=0.7, color='green', label='Après fine-tuning')
            axes[1].axvline(0, color='black', linestyle='--', alpha=0.7)
            axes[1].set_title('Après Fine-Tuning Semi-Supervisé')
            axes[1].set_xlabel('Gap Prédit (µm)')
            axes[1].set_ylabel('Fréquence')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle('Comparaison Avant/Après Fine-Tuning Semi-Supervisé')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'comparison_before_after.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Graphique de comparaison sauvegardé")
            
        except Exception as e:
            print(f"   ⚠️ Erreur de comparaison: {e}")

def main_full_data():
    """Fonction principale avec TOUTES les données de simulation."""
    print("🧠 Semi-Supervised Fine-Tuning avec TOUTES les Données de Simulation")
    print("=" * 80)

    # Créer l'entraîneur semi-supervisé avec dossier spécifique
    trainer = SemiSupervisedTrainer()
    trainer.results_dir = Path("results/semi_supervised_FULL_DATA_22540")
    trainer.results_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Résultats seront sauvegardés dans: {trainer.results_dir}")

    # 1. Charger le modèle pré-entraîné
    trainer.load_pretrained_model()

    # 2. Charger TOUTES les données de simulation
    print("🚀 Chargement de TOUTES les données de simulation (22,540 profils)")
    trainer.load_simulation_data(n_samples=22540)  # TOUTES les données
    trainer.load_experimental_data()

    # 3. Fine-tuning semi-supervisé avec paramètres adaptés pour plus de données
    print("🎯 Configuration pour dataset complet:")
    print("   📊 Données simulation: 22,540 profils (100%)")
    print("   📈 Époques: 150 (plus d'époques pour convergence)")
    print("   📦 Batch size: 64 (plus grand pour efficacité)")
    print("   🧠 Learning rate: 5e-5 (plus conservateur)")
    print("   📊 Lambda alignment: 0.12")
    print("   🔄 Lambda self-training: 0.55")

    history = trainer.semi_supervised_fine_tuning_full_data(epochs=150)

    # 4. Test sur données expérimentales
    results, predictions = trainer.test_on_experimental_data()

    # 5. Comparaison des résultats
    trainer.compare_results()

    # 6. Analyse détaillée des résultats
    trainer.detailed_analysis()

    print("\n🎉 Fine-tuning avec dataset complet terminé avec succès !")
    print(f"📁 Résultats dans: {trainer.results_dir}")

def main():
    """Fonction principale."""
    print("🧠 Semi-Supervised Fine-Tuning du Modèle Ultra-Profond")
    print("=" * 80)

    # Créer l'entraîneur semi-supervisé
    trainer = SemiSupervisedTrainer()

    # 1. Charger le modèle pré-entraîné
    trainer.load_pretrained_model()

    # 2. Charger les données (légèrement plus d'échantillons)
    trainer.load_simulation_data(n_samples=6000)
    trainer.load_experimental_data()

    # 3. Fine-tuning semi-supervisé avec optimisations légères
    print("🎯 Optimisations légères appliquées:")
    print("   📈 Époques: 50 → 120")
    print("   📊 Lambda alignment: 0.1 → 0.15")
    print("   🔄 Lambda self-training: 0.5 → 0.6")
    print("   🎯 Confidence threshold: 0.8 → 0.85")
    print("   📦 Batch size: 32 → 40")
    print("   🧠 Learning rate: 1e-4 → 8e-5")

    history = trainer.semi_supervised_fine_tuning(epochs=120)

    # 4. Test sur données expérimentales
    results, predictions = trainer.test_on_experimental_data()

    # 5. Comparaison des résultats
    trainer.compare_results()

    print("\n🎉 Fine-tuning semi-supervisé terminé avec succès !")
    print(f"📁 Résultats dans: {trainer.results_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--full-data":
        main_full_data()
    else:
        main()
