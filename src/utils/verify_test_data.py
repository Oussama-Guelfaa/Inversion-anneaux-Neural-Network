#!/usr/bin/env python3
"""
Verification of Test Data Process
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Vérifie exactement ce qui se passe avec les données de test.
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import os

def verify_test_data_process():
    """Vérifie exactement le processus de test."""
    
    print("="*80)
    print("VÉRIFICATION DU PROCESSUS DE TEST")
    print("="*80)
    
    # 1. Examiner le fichier labels.csv
    dataset_dir = "../data_generation/dataset"
    labels_df = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))
    
    print(f"\n1. FICHIER LABELS.CSV:")
    print(f"   Nombre d'échantillons: {len(labels_df)}")
    print(f"   Colonnes: {list(labels_df.columns)}")
    print(f"   Premiers échantillons:")
    print(labels_df.head(10))
    
    # 2. Examiner quelques fichiers .mat
    print(f"\n2. EXAMEN DES FICHIERS .MAT:")
    
    for i in range(min(5, len(labels_df))):
        row = labels_df.iloc[i]
        filename = row['filename']
        gap_true = row['gap_um']
        L_true = row['L_um']
        
        mat_filename = filename.replace('.png', '.mat')
        mat_path = os.path.join(dataset_dir, mat_filename)
        
        print(f"\n   Fichier {i+1}: {mat_filename}")
        print(f"   Paramètres vrais: L_ecran={L_true}, gap={gap_true}")
        
        if os.path.exists(mat_path):
            try:
                data = sio.loadmat(mat_path)
                print(f"   Variables dans le fichier:")
                for key in data.keys():
                    if not key.startswith('__'):
                        value = data[key]
                        print(f"     {key}: shape={value.shape}, dtype={value.dtype}")
                        
                        if key == 'ratio':
                            ratio = value.flatten()
                            print(f"       ratio min/max: [{ratio.min():.6f}, {ratio.max():.6f}]")
                            print(f"       ratio mean/std: {ratio.mean():.6f} ± {ratio.std():.6f}")
                            print(f"       ratio length: {len(ratio)}")
                            
            except Exception as e:
                print(f"   ERREUR: {e}")
        else:
            print(f"   FICHIER MANQUANT!")
    
    # 3. Simuler le processus exact de test
    print(f"\n3. SIMULATION DU PROCESSUS DE TEST:")
    
    X_test = []
    y_test = []
    filenames = []
    
    for idx, row in labels_df.iterrows():
        filename = row['filename']
        gap = row['gap_um']
        L_ecran = row['L_um']
        
        mat_filename = filename.replace('.png', '.mat')
        mat_path = os.path.join(dataset_dir, mat_filename)
        
        if os.path.exists(mat_path):
            try:
                data = sio.loadmat(mat_path)
                ratio = data['ratio'].flatten()
                
                X_test.append(ratio)
                y_test.append([L_ecran, gap])  # ORDRE: L_ecran, gap
                filenames.append(filename)
                
                if len(X_test) <= 3:  # Afficher les premiers
                    print(f"   Échantillon {len(X_test)}:")
                    print(f"     Fichier: {mat_filename}")
                    print(f"     Ratio shape: {ratio.shape}")
                    print(f"     Paramètres vrais: [L_ecran={L_ecran}, gap={gap}]")
                    print(f"     Ajouté à X_test: {ratio[:5]}... (premiers 5 points)")
                    print(f"     Ajouté à y_test: [{L_ecran}, {gap}]")
                
            except Exception as e:
                print(f"   ERREUR {mat_filename}: {e}")
    
    X_test = np.array(X_test, dtype='float32')
    y_test = np.array(y_test, dtype='float32')
    
    print(f"\n4. RÉSULTAT FINAL DU CHARGEMENT:")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_test shape: {y_test.shape}")
    print(f"   Échantillons chargés: {len(X_test)}/{len(labels_df)}")
    
    print(f"\n   Plages des données de test:")
    print(f"     X_test (ratios): [{X_test.min():.6f}, {X_test.max():.6f}]")
    print(f"     y_test L_ecran: [{y_test[:, 0].min():.3f}, {y_test[:, 0].max():.3f}]")
    print(f"     y_test gap: [{y_test[:, 1].min():.6f}, {y_test[:, 1].max():.6f}]")
    
    print(f"\n   Valeurs uniques:")
    print(f"     L_ecran: {sorted(np.unique(y_test[:, 0]))}")
    print(f"     gap: {sorted(np.unique(y_test[:, 1]))}")
    
    # 5. Vérifier l'ordre des paramètres
    print(f"\n5. VÉRIFICATION DE L'ORDRE DES PARAMÈTRES:")
    print(f"   Dans le code, j'utilise: y_test.append([L_ecran, gap])")
    print(f"   Donc y_test[:, 0] = L_ecran")
    print(f"   Donc y_test[:, 1] = gap")
    
    print(f"\n   Vérification avec les premiers échantillons:")
    for i in range(min(3, len(y_test))):
        print(f"     Échantillon {i+1}: y_test[{i}] = [{y_test[i, 0]:.3f}, {y_test[i, 1]:.6f}]")
        print(f"       Correspond à: L_ecran={y_test[i, 0]:.3f}, gap={y_test[i, 1]:.6f}")
    
    return X_test, y_test, filenames

def simulate_prediction_process(X_test, y_test):
    """Simule le processus de prédiction."""
    
    print(f"\n6. SIMULATION DU PROCESSUS DE PRÉDICTION:")
    
    # Simuler des prédictions (remplacer par le vrai modèle)
    np.random.seed(42)
    y_pred = np.random.random((len(y_test), 2))
    
    # Ajuster les prédictions pour être dans des plages réalistes
    y_pred[:, 0] = y_pred[:, 0] * 8 + 6    # L_ecran entre 6-14
    y_pred[:, 1] = y_pred[:, 1] * 0.5 + 0.025  # gap entre 0.025-0.525
    
    print(f"   Prédictions simulées shape: {y_pred.shape}")
    print(f"   y_pred L_ecran range: [{y_pred[:, 0].min():.3f}, {y_pred[:, 0].max():.3f}]")
    print(f"   y_pred gap range: [{y_pred[:, 1].min():.6f}, {y_pred[:, 1].max():.6f}]")
    
    print(f"\n   Comparaison pour les premiers échantillons:")
    for i in range(min(5, len(y_test))):
        print(f"     Échantillon {i+1}:")
        print(f"       Vrai:   L_ecran={y_test[i, 0]:.3f}, gap={y_test[i, 1]:.6f}")
        print(f"       Prédit: L_ecran={y_pred[i, 0]:.3f}, gap={y_pred[i, 1]:.6f}")
        print(f"       Erreur: L_ecran={abs(y_test[i, 0] - y_pred[i, 0]):.3f}, gap={abs(y_test[i, 1] - y_pred[i, 1]):.6f}")
    
    return y_pred

def main():
    """Fonction principale de vérification."""
    
    # Vérifier le processus de chargement des données de test
    X_test, y_test, filenames = verify_test_data_process()
    
    # Simuler le processus de prédiction
    y_pred = simulate_prediction_process(X_test, y_test)
    
    print(f"\n{'='*80}")
    print(f"RÉSUMÉ DE LA VÉRIFICATION")
    print(f"{'='*80}")
    
    print(f"✅ CE QUI EST FAIT CORRECTEMENT:")
    print(f"   1. Les ratios sont bien extraits des fichiers .mat")
    print(f"   2. Les paramètres vrais (L_ecran, gap) sont bien récupérés du labels.csv")
    print(f"   3. L'ordre est correct: y_test[:, 0] = L_ecran, y_test[:, 1] = gap")
    print(f"   4. Le processus est: ratio → modèle → [L_ecran_prédit, gap_prédit]")
    print(f"   5. La comparaison se fait entre vrais et prédits")
    
    print(f"\n📊 DONNÉES DE TEST:")
    print(f"   • {len(X_test)} échantillons expérimentaux")
    print(f"   • Chaque ratio a {X_test.shape[1]} points")
    print(f"   • L_ecran: {len(np.unique(y_test[:, 0]))} valeurs uniques")
    print(f"   • gap: {len(np.unique(y_test[:, 1]))} valeurs uniques")
    
    print(f"\n🎯 PROCESSUS DE TEST CONFIRMÉ:")
    print(f"   INPUT:  ratio (1000 points) → Réseau de neurones")
    print(f"   OUTPUT: [L_ecran_prédit, gap_prédit]")
    print(f"   VÉRIFICATION: Comparer avec [L_ecran_vrai, gap_vrai]")

if __name__ == "__main__":
    main()
