#!/usr/bin/env python3
"""
Test du modèle ULTRA_DEEP_NETWORK sur données de simulation
Auteur: Oussama GUELFAA
Date: 15/07/2025

Test du modèle ultra-profond sur 2000 échantillons aléatoires des données
de simulation pour vérifier la cohérence des prédictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
import time
from sklearn.metrics import r2_score, mean_absolute_error
import json

from ultra_fast_training import UltraDeepNetwork

class SimulationDataTester:
    """
    Testeur pour les données de simulation.
    """
    
    def __init__(self, model_path="results/ULTRA_DEEP_NETWORK_ALL_22540_PROFILES/best_model.pt"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🧠 SimulationDataTester initialisé")
        print(f"   🖥️ Device: {self.device}")
        print(f"   📄 Modèle: {model_path}")
    
    def load_model_and_scalers(self):
        """Charge le modèle et les scalers."""
        print("📂 Chargement du modèle et des scalers...")
        
        # Charger le modèle
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.model = UltraDeepNetwork(
            input_size=601,
            output_size=2,
            dropout=0.3
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Charger les scalers
        scalers = joblib.load("ultra_fast_scalers.joblib")
        self.input_scaler = scalers['input_scaler']
        self.output_scaler = scalers['output_scaler']
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   ✅ Modèle chargé: {total_params:,} paramètres")
        print(f"   ✅ Scalers chargés")
    
    def load_simulation_data(self, n_samples=2000):
        """
        Charge un échantillon aléatoire des données de simulation.
        
        Args:
            n_samples (int): Nombre d'échantillons à charger
        
        Returns:
            tuple: (X_sample, y_sample)
        """
        print(f"📂 Chargement de {n_samples} échantillons aléatoires des données de simulation...")
        
        # Charger les données extraites
        data = np.load('extracted_data_full.npz')
        X_data = data['X_data']  # (22540, 601)
        y_data = data['y_data']  # (22540, 2)
        
        print(f"   📊 Données disponibles: {X_data.shape[0]} profils")
        
        # Échantillonnage aléatoire
        np.random.seed(42)  # Pour reproductibilité
        indices = np.random.choice(len(X_data), n_samples, replace=False)
        
        X_sample = X_data[indices]
        y_sample = y_data[indices]
        
        print(f"   ✅ Échantillon sélectionné: {X_sample.shape}")
        print(f"   📈 Gap range: [{y_sample[:, 0].min():.6f}, {y_sample[:, 0].max():.6f}] µm")
        print(f"   📈 L_écran range: [{y_sample[:, 1].min():.1f}, {y_sample[:, 1].max():.1f}] µm")
        
        return X_sample, y_sample
    
    def predict_on_simulation_data(self, X_data, y_true):
        """
        Fait des prédictions sur les données de simulation.
        
        Args:
            X_data (array): Données d'entrée
            y_true (array): Vraies valeurs
        
        Returns:
            array: Prédictions
        """
        print("🔮 Prédictions sur les données de simulation...")
        
        start_time = time.time()
        
        # Normaliser les données d'entrée
        X_normalized = self.input_scaler.transform(X_data)
        
        # Convertir en tensor
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        # Prédictions par batches pour éviter les problèmes de mémoire
        batch_size = 64
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                pred_batch = self.model(batch)
                predictions.append(pred_batch.cpu().numpy())
        
        # Concaténer toutes les prédictions
        predictions_normalized = np.vstack(predictions)
        
        # Dénormaliser les prédictions
        predictions = self.output_scaler.inverse_transform(predictions_normalized)
        
        prediction_time = time.time() - start_time
        
        print(f"   ✅ Prédictions terminées en {prediction_time:.2f} secondes")
        print(f"   📊 Prédictions shape: {predictions.shape}")
        print(f"   📈 Gap prédit range: [{predictions[:, 0].min():.6f}, {predictions[:, 0].max():.6f}] µm")
        print(f"   📈 L_écran prédit range: [{predictions[:, 1].min():.1f}, {predictions[:, 1].max():.1f}] µm")
        
        return predictions
    
    def calculate_metrics(self, y_true, y_pred):
        """Calcule les métriques de performance."""
        print("📊 Calcul des métriques de performance...")
        
        # Séparer gap et L_écran
        gap_true = y_true[:, 0]
        gap_pred = y_pred[:, 0]
        L_ecran_true = y_true[:, 1]
        L_ecran_pred = y_pred[:, 1]
        
        # Métriques pour gap
        gap_r2 = r2_score(gap_true, gap_pred)
        gap_mae = mean_absolute_error(gap_true, gap_pred)
        gap_rmse = np.sqrt(np.mean((gap_true - gap_pred)**2))
        
        # Métriques pour L_écran
        L_ecran_r2 = r2_score(L_ecran_true, L_ecran_pred)
        L_ecran_mae = mean_absolute_error(L_ecran_true, L_ecran_pred)
        L_ecran_rmse = np.sqrt(np.mean((L_ecran_true - L_ecran_pred)**2))
        
        # Analyser les gaps négatifs
        negative_gaps = np.sum(gap_pred < 0)
        negative_gap_percentage = negative_gaps / len(gap_pred) * 100
        
        metrics = {
            'gap_r2': float(gap_r2),
            'gap_mae': float(gap_mae),
            'gap_rmse': float(gap_rmse),
            'L_ecran_r2': float(L_ecran_r2),
            'L_ecran_mae': float(L_ecran_mae),
            'L_ecran_rmse': float(L_ecran_rmse),
            'negative_gaps_count': int(negative_gaps),
            'negative_gaps_percentage': float(negative_gap_percentage),
            'total_samples': int(len(y_true))
        }
        
        print(f"   📈 Gap R²: {gap_r2:.4f}")
        print(f"   📈 Gap MAE: {gap_mae:.6f} µm")
        print(f"   📈 Gap RMSE: {gap_rmse:.6f} µm")
        print(f"   📈 L_écran R²: {L_ecran_r2:.4f}")
        print(f"   📈 L_écran MAE: {L_ecran_mae:.3f} µm")
        print(f"   📈 L_écran RMSE: {L_ecran_rmse:.3f} µm")
        print(f"   ⚠️ Gaps négatifs: {negative_gaps}/{len(gap_pred)} ({negative_gap_percentage:.1f}%)")
        
        return metrics
    
    def save_predictions(self, y_true, y_pred, filename="simulation_predictions.csv"):
        """Sauvegarde les prédictions dans un fichier CSV."""
        print(f"💾 Sauvegarde des prédictions...")
        
        # Créer le DataFrame
        df = pd.DataFrame({
            'gap_true_um': y_true[:, 0],
            'gap_predicted_um': y_pred[:, 0],
            'L_ecran_true_um': y_true[:, 1],
            'L_ecran_predicted_um': y_pred[:, 1],
            'gap_error_um': y_pred[:, 0] - y_true[:, 0],
            'L_ecran_error_um': y_pred[:, 1] - y_true[:, 1],
            'gap_error_percentage': (y_pred[:, 0] - y_true[:, 0]) / y_true[:, 0] * 100,
            'L_ecran_error_percentage': (y_pred[:, 1] - y_true[:, 1]) / y_true[:, 1] * 100
        })
        
        # Sauvegarder
        df.to_csv(filename, index=False)
        
        print(f"   ✅ Prédictions sauvegardées: {filename}")
        print(f"   📊 Colonnes: {list(df.columns)}")
        
        return df
    
    def plot_predictions(self, y_true, y_pred, save_path="simulation_predictions_plots.png"):
        """Génère les graphiques de prédictions."""
        print("📈 Génération des graphiques...")
        
        # Créer une figure avec 2 sous-graphiques
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Gap prédit vs Gap vrai
        gap_true = y_true[:, 0]
        gap_pred = y_pred[:, 0]
        
        axes[0].scatter(gap_true, gap_pred, alpha=0.6, s=20, c='blue', edgecolors='none')
        
        # Ligne parfaite (y=x)
        gap_min = min(gap_true.min(), gap_pred.min())
        gap_max = max(gap_true.max(), gap_pred.max())
        axes[0].plot([gap_min, gap_max], [gap_min, gap_max], 'r--', linewidth=2, label='Prédiction parfaite')
        
        # Ligne y=0 pour montrer les gaps négatifs
        axes[0].axhline(y=0, color='orange', linestyle=':', alpha=0.7, label='Gap = 0 (limite physique)')
        
        axes[0].set_xlabel('Gap Vrai (µm)')
        axes[0].set_ylabel('Gap Prédit (µm)')
        axes[0].set_title(f'Gap Prédit vs Gap Vrai\nR² = {r2_score(gap_true, gap_pred):.4f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. L_écran prédit vs L_écran vrai
        L_ecran_true = y_true[:, 1]
        L_ecran_pred = y_pred[:, 1]
        
        axes[1].scatter(L_ecran_true, L_ecran_pred, alpha=0.6, s=20, c='green', edgecolors='none')
        
        # Ligne parfaite (y=x)
        L_ecran_min = min(L_ecran_true.min(), L_ecran_pred.min())
        L_ecran_max = max(L_ecran_true.max(), L_ecran_pred.max())
        axes[1].plot([L_ecran_min, L_ecran_max], [L_ecran_min, L_ecran_max], 'r--', linewidth=2, label='Prédiction parfaite')
        
        axes[1].set_xlabel('L_écran Vrai (µm)')
        axes[1].set_ylabel('L_écran Prédit (µm)')
        axes[1].set_title(f'L_écran Prédit vs L_écran Vrai\nR² = {r2_score(L_ecran_true, L_ecran_pred):.4f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Test du Modèle Ultra-Profond sur Données de Simulation', fontsize=16)
        plt.tight_layout()
        
        # Sauvegarder
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Graphiques sauvegardés: {save_path}")
    
    def run_complete_test(self, n_samples=2000):
        """Exécute le test complet."""
        print("🧪 Test Complet du Modèle Ultra-Profond sur Données de Simulation")
        print("=" * 80)
        
        start_time = time.time()
        
        # 1. Charger le modèle et les scalers
        self.load_model_and_scalers()
        
        # 2. Charger les données de simulation
        X_data, y_true = self.load_simulation_data(n_samples)
        
        # 3. Faire les prédictions
        y_pred = self.predict_on_simulation_data(X_data, y_true)
        
        # 4. Calculer les métriques
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # 5. Sauvegarder les prédictions
        df = self.save_predictions(y_true, y_pred, "simulation_predictions_2000_samples.csv")
        
        # 6. Générer les graphiques
        self.plot_predictions(y_true, y_pred, "simulation_predictions_plots.png")
        
        # 7. Sauvegarder les métriques
        with open("simulation_test_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        total_time = time.time() - start_time
        
        print("=" * 80)
        print(f"✅ Test complet terminé en {total_time/60:.1f} minutes")
        print(f"📁 Fichiers générés:")
        print(f"   - simulation_predictions_2000_samples.csv")
        print(f"   - simulation_predictions_plots.png")
        print(f"   - simulation_test_metrics.json")
        
        return metrics, df

def main():
    """Fonction principale."""
    print("🧠 Test du Modèle Ultra-Profond sur Données de Simulation")
    print("=" * 70)
    
    # Créer le testeur
    tester = SimulationDataTester()
    
    # Exécuter le test complet
    metrics, df = tester.run_complete_test(n_samples=2000)
    
    print("\n🎉 Test terminé avec succès !")
    print(f"📊 Résumé des performances:")
    print(f"   Gap R²: {metrics['gap_r2']:.4f}")
    print(f"   L_écran R²: {metrics['L_ecran_r2']:.4f}")
    print(f"   Gaps négatifs: {metrics['negative_gaps_count']}/{metrics['total_samples']} ({metrics['negative_gaps_percentage']:.1f}%)")

if __name__ == "__main__":
    main()
