#!/usr/bin/env python3
"""
Test du modèle spécialisé sur 5000 échantillons simulés aléatoires
Auteur: Oussama GUELFAA
Date: 18/07/2025

Ce script teste le modèle spécialisé pour le gap sur des données simulées
pour évaluer ses performances réelles.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import random
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import sys
sys.path.append('../../utils/data_loaders')
from ultra_fast_data_loader import UltraFastDataLoader

# Import des classes du modèle spécialisé
sys.path.append('../training')
from specialized_gap_training import DualSpecializedNetwork

class SpecializedModelTester:
    """Testeur pour le modèle spécialisé."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🧪 TESTEUR DU MODÈLE SPÉCIALISÉ")
        print("=" * 50)
        print(f"   🖥️ Device: {self.device}")
        print(f"   🎯 Test sur 5000 échantillons simulés aléatoires")
    
    def load_specialized_model(self):
        """Charge le modèle spécialisé entraîné."""
        
        print("📂 Chargement du modèle spécialisé...")
        
        model_path = "../training/results/specialized_gap_training/best_specialized_model.pt"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        # Charger le checkpoint (avec weights_only=False pour compatibilité)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Créer le modèle
        self.model = DualSpecializedNetwork().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Informations du modèle
        epoch = checkpoint['epoch']
        gap_r2 = checkpoint['gap_r2']
        L_ecran_r2 = checkpoint['L_ecran_r2']
        gap_precision = checkpoint['gap_precision']
        
        print(f"   ✅ Modèle chargé (époque {epoch})")
        print(f"   📊 Gap R² d'entraînement: {gap_r2:.4f}")
        print(f"   📊 L_écran R² d'entraînement: {L_ecran_r2:.4f}")
        print(f"   📊 Précision Gap d'entraînement: {gap_precision:.1f}%")
        
        return checkpoint
    
    def load_test_data(self, n_samples=5000):
        """Charge 5000 échantillons simulés aléatoires."""
        
        print(f"📊 Chargement de {n_samples} échantillons simulés aléatoires...")
        
        # Charger toutes les données
        data_loader = UltraFastDataLoader("../../data/processed/extracted_data_full.npz")
        X_all, y_all = data_loader.load_data(sample_ratio=1.0)
        
        print(f"   📊 Données totales disponibles: {len(X_all)}")
        
        # Sélection aléatoire
        total_samples = len(X_all)
        if n_samples > total_samples:
            print(f"   ⚠️  Demande {n_samples} mais seulement {total_samples} disponibles")
            n_samples = total_samples
        
        # Indices aléatoires
        random.seed(42)  # Pour la reproductibilité
        random_indices = random.sample(range(total_samples), n_samples)
        random_indices.sort()  # Trier pour l'efficacité
        
        # Extraire les échantillons
        X_test = X_all[random_indices]
        y_test = y_all[random_indices]
        
        gaps_test = y_test[:, 0]
        L_ecrans_test = y_test[:, 1]
        
        print(f"   ✅ {len(X_test)} échantillons sélectionnés aléatoirement")
        print(f"   📊 Gap range: [{gaps_test.min():.6f}, {gaps_test.max():.6f}] µm")
        print(f"   📊 L_écran range: [{L_ecrans_test.min():.3f}, {L_ecrans_test.max():.3f}] µm")
        
        return X_test, y_test, random_indices
    
    def predict_on_test_data(self, X_test, y_test):
        """Fait les prédictions sur les données de test."""
        
        print("🔮 Prédictions sur les données de test...")
        
        # Convertir en tenseurs
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Prédictions par batches pour éviter les problèmes de mémoire
        batch_size = 64
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                pred_batch = self.model(batch)
                predictions.append(pred_batch.cpu().numpy())
        
        # Concaténer toutes les prédictions
        y_pred = np.vstack(predictions)
        
        print(f"   ✅ {len(y_pred)} prédictions effectuées")
        print(f"   📊 Gap prédit range: [{y_pred[:, 0].min():.6f}, {y_pred[:, 0].max():.6f}] µm")
        print(f"   📊 L_écran prédit range: [{y_pred[:, 1].min():.3f}, {y_pred[:, 1].max():.3f}] µm")
        
        return y_pred
    
    def calculate_detailed_metrics(self, y_true, y_pred):
        """Calcule des métriques détaillées."""
        
        print("📊 Calcul des métriques détaillées...")
        
        gaps_true = y_true[:, 0]
        L_ecrans_true = y_true[:, 1]
        gaps_pred = y_pred[:, 0]
        L_ecrans_pred = y_pred[:, 1]
        
        # Métriques pour le gap
        gap_r2 = r2_score(gaps_true, gaps_pred)
        gap_mae = mean_absolute_error(gaps_true, gaps_pred)
        gap_rmse = np.sqrt(mean_squared_error(gaps_true, gaps_pred))
        gap_mape = np.mean(np.abs((gaps_true - gaps_pred) / gaps_true)) * 100
        
        # Métriques pour L_écran
        L_ecran_r2 = r2_score(L_ecrans_true, L_ecrans_pred)
        L_ecran_mae = mean_absolute_error(L_ecrans_true, L_ecrans_pred)
        L_ecran_rmse = np.sqrt(mean_squared_error(L_ecrans_true, L_ecrans_pred))
        L_ecran_mape = np.mean(np.abs((L_ecrans_true - L_ecrans_pred) / L_ecrans_true)) * 100
        
        # Précision (% dans la tolérance)
        gap_errors = np.abs(gaps_true - gaps_pred)
        L_ecran_errors = np.abs(L_ecrans_true - L_ecrans_pred)
        
        gap_precision_001 = np.mean(gap_errors <= 0.001) * 100  # ±0.001 µm
        gap_precision_01 = np.mean(gap_errors <= 0.01) * 100    # ±0.01 µm
        gap_precision_05 = np.mean(gap_errors <= 0.05) * 100    # ±0.05 µm
        
        L_ecran_precision_01 = np.mean(L_ecran_errors <= 0.01) * 100  # ±0.01 µm
        L_ecran_precision_1 = np.mean(L_ecran_errors <= 0.1) * 100    # ±0.1 µm
        L_ecran_precision_5 = np.mean(L_ecran_errors <= 0.5) * 100    # ±0.5 µm
        
        metrics = {
            'gap': {
                'r2': gap_r2,
                'mae': gap_mae,
                'rmse': gap_rmse,
                'mape': gap_mape,
                'precision_0.001': gap_precision_001,
                'precision_0.01': gap_precision_01,
                'precision_0.05': gap_precision_05
            },
            'L_ecran': {
                'r2': L_ecran_r2,
                'mae': L_ecran_mae,
                'rmse': L_ecran_rmse,
                'mape': L_ecran_mape,
                'precision_0.01': L_ecran_precision_01,
                'precision_0.1': L_ecran_precision_1,
                'precision_0.5': L_ecran_precision_5
            }
        }
        
        # Affichage des résultats
        print(f"\n📊 RÉSULTATS DÉTAILLÉS:")
        print(f"   🎯 GAP:")
        print(f"      R² = {gap_r2:.4f}")
        print(f"      MAE = {gap_mae:.6f} µm")
        print(f"      RMSE = {gap_rmse:.6f} µm")
        print(f"      MAPE = {gap_mape:.2f}%")
        print(f"      Précision ±0.001 µm: {gap_precision_001:.1f}%")
        print(f"      Précision ±0.01 µm: {gap_precision_01:.1f}%")
        print(f"      Précision ±0.05 µm: {gap_precision_05:.1f}%")
        
        print(f"   🎯 L_ÉCRAN:")
        print(f"      R² = {L_ecran_r2:.4f}")
        print(f"      MAE = {L_ecran_mae:.6f} µm")
        print(f"      RMSE = {L_ecran_rmse:.6f} µm")
        print(f"      MAPE = {L_ecran_mape:.2f}%")
        print(f"      Précision ±0.01 µm: {L_ecran_precision_01:.1f}%")
        print(f"      Précision ±0.1 µm: {L_ecran_precision_1:.1f}%")
        print(f"      Précision ±0.5 µm: {L_ecran_precision_5:.1f}%")
        
        return metrics
    
    def create_detailed_plots(self, y_true, y_pred, metrics):
        """Crée des graphiques détaillés des résultats."""
        
        print("📈 Création des graphiques détaillés...")
        
        gaps_true = y_true[:, 0]
        L_ecrans_true = y_true[:, 1]
        gaps_pred = y_pred[:, 0]
        L_ecrans_pred = y_pred[:, 1]
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('TEST DU MODÈLE SPÉCIALISÉ - 5000 ÉCHANTILLONS SIMULÉS ALÉATOIRES', 
                     fontsize=16, fontweight='bold')
        
        # 1. Scatter plot Gap
        ax1 = axes[0, 0]
        ax1.scatter(gaps_true, gaps_pred, alpha=0.6, s=10)
        ax1.plot([gaps_true.min(), gaps_true.max()], [gaps_true.min(), gaps_true.max()], 
                'r--', linewidth=2, label='Parfait')
        ax1.set_xlabel('Gap Réel (µm)')
        ax1.set_ylabel('Gap Prédit (µm)')
        ax1.set_title(f'Gap: R² = {metrics["gap"]["r2"]:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot L_écran
        ax2 = axes[0, 1]
        ax2.scatter(L_ecrans_true, L_ecrans_pred, alpha=0.6, s=10)
        ax2.plot([L_ecrans_true.min(), L_ecrans_true.max()], 
                [L_ecrans_true.min(), L_ecrans_true.max()], 
                'r--', linewidth=2, label='Parfait')
        ax2.set_xlabel('L_écran Réel (µm)')
        ax2.set_ylabel('L_écran Prédit (µm)')
        ax2.set_title(f'L_écran: R² = {metrics["L_ecran"]["r2"]:.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution des erreurs Gap
        ax3 = axes[0, 2]
        gap_errors = gaps_pred - gaps_true
        ax3.hist(gap_errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.axvline(np.mean(gap_errors), color='orange', linestyle='-', linewidth=2, 
                   label=f'Moyenne: {np.mean(gap_errors):.6f}')
        ax3.set_xlabel('Erreur Gap (µm)')
        ax3.set_ylabel('Fréquence')
        ax3.set_title(f'Distribution Erreurs Gap\nMAE: {metrics["gap"]["mae"]:.6f} µm')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribution des erreurs L_écran
        ax4 = axes[1, 0]
        L_ecran_errors = L_ecrans_pred - L_ecrans_true
        ax4.hist(L_ecran_errors, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2)
        ax4.axvline(np.mean(L_ecran_errors), color='orange', linestyle='-', linewidth=2,
                   label=f'Moyenne: {np.mean(L_ecran_errors):.6f}')
        ax4.set_xlabel('Erreur L_écran (µm)')
        ax4.set_ylabel('Fréquence')
        ax4.set_title(f'Distribution Erreurs L_écran\nMAE: {metrics["L_ecran"]["mae"]:.6f} µm')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Précision par tolérance
        ax5 = axes[1, 1]
        tolerances_gap = [0.001, 0.01, 0.05]
        precisions_gap = [metrics['gap']['precision_0.001'], 
                         metrics['gap']['precision_0.01'], 
                         metrics['gap']['precision_0.05']]
        
        bars = ax5.bar(range(len(tolerances_gap)), precisions_gap, 
                      alpha=0.7, color='blue')
        ax5.set_xticks(range(len(tolerances_gap)))
        ax5.set_xticklabels([f'±{t:.3f}' for t in tolerances_gap])
        ax5.set_xlabel('Tolérance (µm)')
        ax5.set_ylabel('Précision (%)')
        ax5.set_title('Précision Gap par Tolérance')
        ax5.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, prec in zip(bars, precisions_gap):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prec:.1f}%', ha='center', va='bottom')
        
        # 6. Tableau de résumé
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"""RÉSUMÉ DES PERFORMANCES
        
🎯 GAP:
   R² = {metrics['gap']['r2']:.4f}
   MAE = {metrics['gap']['mae']:.6f} µm
   RMSE = {metrics['gap']['rmse']:.6f} µm
   Précision ±0.01 µm: {metrics['gap']['precision_0.01']:.1f}%
   
🎯 L_ÉCRAN:
   R² = {metrics['L_ecran']['r2']:.4f}
   MAE = {metrics['L_ecran']['mae']:.6f} µm
   RMSE = {metrics['L_ecran']['rmse']:.6f} µm
   Précision ±0.1 µm: {metrics['L_ecran']['precision_0.1']:.1f}%
   
📊 ÉCHANTILLONS: 5000 (aléatoires)
📊 MODÈLE: Architecture Spécialisée
📊 DONNÉES: Simulées (sans normalisation)
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarder
        output_file = "../../visualizations/plots/specialized_model_test_results.png"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Graphiques sauvegardés: {output_file}")
        
        plt.show()
    
    def save_detailed_results(self, y_true, y_pred, metrics, random_indices):
        """Sauvegarde les résultats détaillés."""
        
        print("💾 Sauvegarde des résultats détaillés...")
        
        # Créer un DataFrame avec tous les résultats
        results_df = pd.DataFrame({
            'Index': random_indices,
            'Gap_Reel': y_true[:, 0],
            'Gap_Predit': y_pred[:, 0],
            'Gap_Erreur': y_pred[:, 0] - y_true[:, 0],
            'Gap_Erreur_Abs': np.abs(y_pred[:, 0] - y_true[:, 0]),
            'L_ecran_Reel': y_true[:, 1],
            'L_ecran_Predit': y_pred[:, 1],
            'L_ecran_Erreur': y_pred[:, 1] - y_true[:, 1],
            'L_ecran_Erreur_Abs': np.abs(y_pred[:, 1] - y_true[:, 1])
        })
        
        # Sauvegarder le CSV
        csv_file = "../../results/specialized_model_test_5000_samples.csv"
        Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_file, index=False)
        
        # Sauvegarder les métriques
        metrics_file = "../../results/specialized_model_metrics.json"
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   ✅ Résultats CSV: {csv_file}")
        print(f"   ✅ Métriques JSON: {metrics_file}")
        print(f"   📊 {len(results_df)} échantillons sauvegardés")
    
    def run_complete_test(self):
        """Lance le test complet du modèle spécialisé."""
        
        try:
            # 1. Charger le modèle
            checkpoint = self.load_specialized_model()
            
            # 2. Charger les données de test
            X_test, y_test, random_indices = self.load_test_data(n_samples=5000)
            
            # 3. Faire les prédictions
            y_pred = self.predict_on_test_data(X_test, y_test)
            
            # 4. Calculer les métriques
            metrics = self.calculate_detailed_metrics(y_test, y_pred)
            
            # 5. Créer les graphiques
            self.create_detailed_plots(y_test, y_pred, metrics)
            
            # 6. Sauvegarder les résultats
            self.save_detailed_results(y_test, y_pred, metrics, random_indices)
            
            print(f"\n✅ TEST COMPLET TERMINÉ!")
            print(f"   🎯 Gap R²: {metrics['gap']['r2']:.4f}")
            print(f"   🎯 L_écran R²: {metrics['L_ecran']['r2']:.4f}")
            print(f"   📊 5000 échantillons testés")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            raise

def main():
    """Fonction principale."""
    
    tester = SpecializedModelTester()
    metrics = tester.run_complete_test()
    
    print(f"\n🎉 TEST DU MODÈLE SPÉCIALISÉ TERMINÉ!")

if __name__ == "__main__":
    main()
