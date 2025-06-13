#!/usr/bin/env python3
"""
Test d'accès aux données du dataset 2D

Auteur: Oussama GUELFAA
Date: 06 - 01 - 2025

Ce script teste l'accès aux données et valide leur format pour l'entraînement.
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
import matplotlib.pyplot as plt
import random

def test_data_loading():
    """
    Teste le chargement de quelques fichiers du dataset.
    """
    print("🧪 TEST DE CHARGEMENT DES DONNÉES")
    print("="*40)
    
    dataset_path = Path("data_generation/dataset_2D")
    
    if not dataset_path.exists():
        print("❌ Dossier dataset_2D non trouvé")
        return False
    
    # Trouver tous les fichiers .mat
    mat_files = list(dataset_path.glob("*.mat"))
    mat_files = [f for f in mat_files if f.name != "labels.mat"]
    
    print(f"📁 Fichiers trouvés: {len(mat_files)}")
    
    # Tester quelques fichiers aléatoires
    test_files = random.sample(mat_files, min(5, len(mat_files)))
    
    success_count = 0
    for i, file_path in enumerate(test_files):
        print(f"\n🔍 Test {i+1}/5: {file_path.name}")
        
        try:
            # Charger le fichier
            data = loadmat(str(file_path))
            
            # Vérifier les clés attendues
            expected_keys = ['ratio', 'x', 'gap', 'L_ecran_subs']
            missing_keys = [key for key in expected_keys if key not in data]
            
            if missing_keys:
                print(f"   ⚠️  Clés manquantes: {missing_keys}")
            else:
                print(f"   ✅ Toutes les clés présentes")
            
            # Vérifier les dimensions
            ratio = data['ratio']
            x = data['x']
            
            print(f"   📊 Ratio shape: {ratio.shape}")
            print(f"   📊 X shape: {x.shape}")
            print(f"   📊 Gap: {data['gap'][0,0]:.4f}")
            print(f"   📊 L_ecran: {data['L_ecran_subs'][0,0]:.3f}")
            
            # Vérifier les valeurs
            ratio_flat = ratio.flatten()
            print(f"   📈 Ratio min/max: {ratio_flat.min():.4f} / {ratio_flat.max():.4f}")
            print(f"   📈 Ratio mean: {ratio_flat.mean():.4f}")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    print(f"\n✅ Succès: {success_count}/{len(test_files)} fichiers")
    return success_count == len(test_files)

def test_data_consistency():
    """
    Teste la cohérence des données entre fichiers.
    """
    print("\n🔍 TEST DE COHÉRENCE DES DONNÉES")
    print("="*40)
    
    dataset_path = Path("data_generation/dataset_2D")
    mat_files = list(dataset_path.glob("*.mat"))
    mat_files = [f for f in mat_files if f.name != "labels.mat"]
    
    # Tester un échantillon
    test_files = random.sample(mat_files, min(10, len(mat_files)))
    
    shapes = []
    gaps = []
    L_ecrans = []
    
    for file_path in test_files:
        try:
            data = loadmat(str(file_path))
            shapes.append(data['ratio'].shape)
            gaps.append(data['gap'][0,0])
            L_ecrans.append(data['L_ecran_subs'][0,0])
        except:
            continue
    
    # Vérifier la cohérence des shapes
    unique_shapes = list(set(shapes))
    print(f"📐 Shapes uniques trouvées: {unique_shapes}")
    
    if len(unique_shapes) == 1:
        print("   ✅ Toutes les données ont la même dimension")
    else:
        print("   ⚠️  Dimensions incohérentes détectées")
    
    # Vérifier les plages de paramètres
    print(f"📊 Plage gaps: {min(gaps):.4f} - {max(gaps):.4f}")
    print(f"📊 Plage L_ecran: {min(L_ecrans):.3f} - {max(L_ecrans):.3f}")
    
    return len(unique_shapes) == 1

def test_neural_network_format():
    """
    Teste le format des données pour l'entraînement de réseaux de neurones.
    """
    print("\n🧠 TEST FORMAT RÉSEAUX DE NEURONES")
    print("="*40)
    
    dataset_path = Path("data_generation/dataset_2D")
    mat_files = list(dataset_path.glob("*.mat"))
    mat_files = [f for f in mat_files if f.name != "labels.mat"]
    
    # Charger quelques échantillons
    sample_files = random.sample(mat_files, min(3, len(mat_files)))
    
    X_samples = []
    y_samples = []
    
    for file_path in sample_files:
        try:
            data = loadmat(str(file_path))
            
            # Préparer X (features) - ratio d'intensité
            ratio = data['ratio'].flatten()
            
            # Tronquer à 600 points comme recommandé
            if len(ratio) > 600:
                ratio = ratio[:600]
            
            X_samples.append(ratio)
            
            # Préparer y (targets) - gap et L_ecran
            gap = data['gap'][0,0]
            L_ecran = data['L_ecran_subs'][0,0]
            y_samples.append([gap, L_ecran])
            
        except Exception as e:
            print(f"   ❌ Erreur lors du traitement: {e}")
    
    if X_samples and y_samples:
        X_array = np.array(X_samples)
        y_array = np.array(y_samples)
        
        print(f"📊 Format X (features): {X_array.shape}")
        print(f"📊 Format y (targets): {y_array.shape}")
        print(f"📊 Type X: {X_array.dtype}")
        print(f"📊 Type y: {y_array.dtype}")
        
        # Statistiques sur X
        print(f"📈 X min/max: {X_array.min():.4f} / {X_array.max():.4f}")
        print(f"📈 X mean/std: {X_array.mean():.4f} / {X_array.std():.4f}")
        
        # Statistiques sur y
        print(f"📈 Gap min/max: {y_array[:,0].min():.4f} / {y_array[:,0].max():.4f}")
        print(f"📈 L_ecran min/max: {y_array[:,1].min():.3f} / {y_array[:,1].max():.3f}")
        
        print("✅ Format compatible avec les réseaux de neurones")
        return True
    else:
        print("❌ Impossible de préparer les données")
        return False

def create_sample_visualization():
    """
    Crée une visualisation rapide d'un échantillon.
    """
    print("\n🎨 CRÉATION D'UNE VISUALISATION ÉCHANTILLON")
    print("="*40)
    
    dataset_path = Path("data_generation/dataset_2D")
    mat_files = list(dataset_path.glob("*.mat"))
    mat_files = [f for f in mat_files if f.name != "labels.mat"]
    
    # Prendre un fichier aléatoire
    sample_file = random.choice(mat_files)
    
    try:
        data = loadmat(str(sample_file))
        
        ratio = data['ratio'].flatten()
        x = data['x'].flatten() if 'x' in data else np.arange(len(ratio))
        gap = data['gap'][0,0]
        L_ecran = data['L_ecran_subs'][0,0]
        
        # Créer le graphique
        plt.figure(figsize=(10, 6))
        plt.plot(x, ratio, 'b-', linewidth=1.5, alpha=0.8)
        plt.title(f'Échantillon: Gap={gap:.4f}µm, L_ecran={L_ecran:.1f}µm\n{sample_file.name}')
        plt.xlabel('Position (µm)')
        plt.ylabel('Ratio I/I₀')
        plt.grid(True, alpha=0.3)
        
        # Sauvegarder
        output_path = Path("analysis_scripts/dataset_2D_analysis/outputs_analysis_2D")
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path / "test_sample_visualization.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualisation sauvegardée: test_sample_visualization.png")
        print(f"📊 Fichier testé: {sample_file.name}")
        print(f"📊 Points de données: {len(ratio)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la visualisation: {e}")
        return False

def main():
    """
    Fonction principale de test.
    """
    print("🧪 TEST D'ACCÈS AU DATASET 2D")
    print("Auteur: Oussama GUELFAA")
    print("Date: 06-01-2025")
    print("="*50)
    
    # Tests séquentiels
    tests_results = []
    
    # Test 1: Chargement des données
    tests_results.append(test_data_loading())
    
    # Test 2: Cohérence des données
    tests_results.append(test_data_consistency())
    
    # Test 3: Format pour réseaux de neurones
    tests_results.append(test_neural_network_format())
    
    # Test 4: Visualisation échantillon
    tests_results.append(create_sample_visualization())
    
    # Résumé final
    print("\n" + "="*50)
    print("📋 RÉSUMÉ DES TESTS")
    print("-"*30)
    
    test_names = [
        "Chargement des données",
        "Cohérence des données", 
        "Format réseaux de neurones",
        "Visualisation échantillon"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, tests_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {name}: {status}")
    
    success_rate = sum(tests_results) / len(tests_results) * 100
    print(f"\n🎯 Taux de succès: {success_rate:.0f}%")
    
    if success_rate == 100:
        print("🚀 Tous les tests passés ! Dataset prêt pour l'utilisation.")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les données.")
    
    print("="*50)

if __name__ == "__main__":
    main()
