#!/usr/bin/env python3
"""
Test Data Loader
Author: Oussama GUELFAA
Date: 05 - 06 - 2025

Script pour examiner et charger les données de test expérimentales
du dossier dataset.
"""

import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def inspect_test_data_structure():
    """Examine la structure des données de test expérimentales."""
    
    print("=== INSPECTION DES DONNÉES DE TEST EXPÉRIMENTALES ===")
    
    dataset_dir = "../data_generation/dataset"
    
    # Lire le fichier labels
    labels_file = os.path.join(dataset_dir, "labels.csv")
    if os.path.exists(labels_file):
        labels_df = pd.read_csv(labels_file)
        print(f"\nFichier labels.csv trouvé:")
        print(f"  Nombre d'échantillons: {len(labels_df)}")
        print(f"  Colonnes: {list(labels_df.columns)}")
        print(f"  Plages de valeurs:")
        print(f"    gap_um: {labels_df['gap_um'].min():.6f} à {labels_df['gap_um'].max():.6f}")
        print(f"    L_um: {labels_df['L_um'].min():.3f} à {labels_df['L_um'].max():.3f}")
        print(f"  Valeurs uniques gap: {sorted(labels_df['gap_um'].unique())}")
        print(f"  Valeurs uniques L: {sorted(labels_df['L_um'].unique())}")
    else:
        print(f"ERREUR: Fichier labels non trouvé: {labels_file}")
        return None
    
    # Examiner quelques fichiers .mat
    mat_files = glob.glob(os.path.join(dataset_dir, "*.mat"))
    print(f"\nFichiers .mat trouvés: {len(mat_files)}")
    
    if len(mat_files) > 0:
        # Examiner le premier fichier
        first_file = mat_files[0]
        print(f"\nExamen du fichier: {os.path.basename(first_file)}")
        
        try:
            data = sio.loadmat(first_file)
            print(f"Variables dans le fichier:")
            for key in data.keys():
                if not key.startswith('__'):
                    value = data[key]
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    
                    if hasattr(value, 'flatten') and value.size > 0:
                        flat = value.flatten()
                        if np.issubdtype(flat.dtype, np.number):
                            print(f"    min/max: {flat.min():.6f} / {flat.max():.6f}")
                            if value.size <= 10:
                                print(f"    valeurs: {flat}")
                            else:
                                print(f"    premiers/derniers: [{flat[0]:.6f}, {flat[1]:.6f}, ..., {flat[-2]:.6f}, {flat[-1]:.6f}]")
        except Exception as e:
            print(f"Erreur lors de la lecture: {e}")
    
    return labels_df, mat_files

def load_test_dataset():
    """Charge toutes les données de test expérimentales."""
    
    print(f"\n=== CHARGEMENT DU DATASET DE TEST ===")
    
    dataset_dir = "../data_generation/dataset"
    
    # Charger les labels
    labels_df = pd.read_csv(os.path.join(dataset_dir, "labels.csv"))
    
    # Préparer les listes pour stocker les données
    X_test = []
    y_test = []
    filenames = []
    
    print(f"Chargement de {len(labels_df)} échantillons...")
    
    for idx, row in labels_df.iterrows():
        filename = row['filename']
        gap = row['gap_um']
        L_ecran = row['L_um']
        
        # Construire le nom du fichier .mat correspondant
        mat_filename = filename.replace('.png', '.mat')
        mat_path = os.path.join(dataset_dir, mat_filename)
        
        if os.path.exists(mat_path):
            try:
                # Charger le fichier .mat
                data = sio.loadmat(mat_path)
                
                # Extraire les données (adapter selon la structure réelle)
                if 'ratio' in data:
                    ratio = data['ratio'].flatten()
                elif 'x' in data and 'ratio' in data:
                    ratio = data['ratio'].flatten()
                else:
                    # Examiner toutes les variables numériques
                    ratio = None
                    for key, value in data.items():
                        if not key.startswith('__') and hasattr(value, 'flatten'):
                            if value.size > 100:  # Probablement le profil
                                ratio = value.flatten()
                                break
                
                if ratio is not None:
                    X_test.append(ratio)
                    y_test.append([L_ecran, gap])
                    filenames.append(filename)
                    
                    if len(X_test) <= 3:  # Afficher les premiers
                        print(f"  {mat_filename}: ratio shape={ratio.shape}, L={L_ecran}, gap={gap:.6f}")
                else:
                    print(f"  ATTENTION: Impossible d'extraire le ratio de {mat_filename}")
                    
            except Exception as e:
                print(f"  ERREUR: {mat_filename} - {e}")
        else:
            print(f"  MANQUANT: {mat_filename}")
    
    if len(X_test) > 0:
        # Convertir en arrays numpy
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"\nDataset de test chargé:")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        print(f"  Échantillons réussis: {len(X_test)}/{len(labels_df)}")
        
        # Vérifier la cohérence des tailles
        if X_test.shape[1] != 1000:
            print(f"  ATTENTION: Taille des profils différente de 1000: {X_test.shape[1]}")
        
        return X_test, y_test, filenames
    else:
        print(f"ERREUR: Aucune donnée de test chargée!")
        return None, None, None

def compare_train_test_data():
    """Compare les données d'entraînement et de test."""
    
    print(f"\n=== COMPARAISON DONNÉES ENTRAÎNEMENT vs TEST ===")
    
    # Charger les données d'entraînement
    train_data = np.load('processed_data/training_data.npz', allow_pickle=True)
    X_train = train_data['X']
    y_train = train_data['y']
    
    # Charger les données de test
    X_test, y_test, filenames = load_test_dataset()
    
    if X_test is not None:
        print(f"\nComparaison des dimensions:")
        print(f"  Entraînement: X{X_train.shape}, y{y_train.shape}")
        print(f"  Test: X{X_test.shape}, y{y_test.shape}")
        
        print(f"\nComparaison des plages de valeurs:")
        print(f"  X_train: [{X_train.min():.6f}, {X_train.max():.6f}]")
        print(f"  X_test: [{X_test.min():.6f}, {X_test.max():.6f}]")
        
        print(f"\nComparaison des paramètres:")
        print(f"  L_ecran train: [{y_train[:, 0].min():.3f}, {y_train[:, 0].max():.3f}]")
        print(f"  L_ecran test: [{y_test[:, 0].min():.3f}, {y_test[:, 0].max():.3f}]")
        print(f"  gap train: [{y_train[:, 1].min():.6f}, {y_train[:, 1].max():.6f}]")
        print(f"  gap test: [{y_test[:, 1].min():.6f}, {y_test[:, 1].max():.6f}]")
        
        # Visualisation comparative
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Profils d'intensité
        for i in range(min(5, len(X_train))):
            axes[0, 0].plot(X_train[i], alpha=0.6, linewidth=0.8)
        axes[0, 0].set_title('Profils d\'entraînement (échantillon)')
        axes[0, 0].set_xlabel('Position radiale')
        axes[0, 0].set_ylabel('Intensité normalisée')
        
        for i in range(min(5, len(X_test))):
            axes[0, 1].plot(X_test[i], alpha=0.6, linewidth=0.8)
        axes[0, 1].set_title('Profils de test (échantillon)')
        axes[0, 1].set_xlabel('Position radiale')
        axes[0, 1].set_ylabel('Intensité normalisée')
        
        # Distribution des paramètres
        axes[1, 0].scatter(y_train[:, 0], y_train[:, 1], alpha=0.6, s=20, label='Entraînement')
        axes[1, 0].scatter(y_test[:, 0], y_test[:, 1], alpha=0.8, s=40, label='Test', marker='x')
        axes[1, 0].set_xlabel('L_ecran')
        axes[1, 0].set_ylabel('gap')
        axes[1, 0].set_title('Distribution des paramètres')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogrammes des intensités
        axes[1, 1].hist(X_train.flatten(), bins=50, alpha=0.6, label='Entraînement', density=True)
        axes[1, 1].hist(X_test.flatten(), bins=50, alpha=0.6, label='Test', density=True)
        axes[1, 1].set_xlabel('Intensité')
        axes[1, 1].set_ylabel('Densité')
        axes[1, 1].set_title('Distribution des intensités')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('plots/train_vs_test_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return X_test, y_test, filenames
    else:
        return None, None, None

if __name__ == "__main__":
    # Créer le dossier plots s'il n'existe pas
    os.makedirs('plots', exist_ok=True)
    
    # Inspecter la structure
    labels_df, mat_files = inspect_test_data_structure()
    
    # Charger et comparer
    X_test, y_test, filenames = compare_train_test_data()
    
    if X_test is not None:
        print(f"\n=== RÉSUMÉ ===")
        print(f"Données de test chargées avec succès!")
        print(f"Prêt pour l'évaluation du modèle sur les données expérimentales.")
    else:
        print(f"\n=== PROBLÈME ===")
        print(f"Impossible de charger les données de test.")
        print(f"Vérifiez la structure des fichiers .mat dans le dossier dataset.")
