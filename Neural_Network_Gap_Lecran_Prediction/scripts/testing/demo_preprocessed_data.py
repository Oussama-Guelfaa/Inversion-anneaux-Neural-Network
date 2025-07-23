#!/usr/bin/env python3
"""
Démonstration d'utilisation des données prétraitées
Auteur: Oussama GUELFAA
Date: 15/07/2025

Ce script montre comment charger et utiliser les données prétraitées
pour l'entraînement d'un réseau de neurones.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_preprocessed_data():
    """
    Charge les données prétraitées depuis les fichiers .npz
    
    Returns:
        tuple: (preprocessed_data, train_examples)
    """
    print("📂 Chargement des données prétraitées...")
    
    # Vérifier l'existence des fichiers
    if not os.path.exists('preprocessed_data.npz'):
        raise FileNotFoundError("Fichier preprocessed_data.npz non trouvé. Exécutez d'abord preprocess_data.py")
    
    if not os.path.exists('train_examples.npz'):
        raise FileNotFoundError("Fichier train_examples.npz non trouvé. Exécutez d'abord preprocess_data.py")
    
    # Charger les données
    preprocessed_data = np.load('preprocessed_data.npz')
    train_examples = np.load('train_examples.npz')
    
    print("✅ Données chargées avec succès!")
    print(f"   - Plage radiale : [{preprocessed_data['r_min']:.6f}, {preprocessed_data['r_max']:.6f}] µm")
    print(f"   - Espacement : {preprocessed_data['delta_r']:.6f} µm")
    print(f"   - Points par profil : {preprocessed_data['points_per_profile']}")
    print(f"   - Profils de test : {preprocessed_data['test_profiles_count']}")
    print(f"   - Fichiers d'entraînement : {preprocessed_data['train_files_count']}")
    print(f"   - Exemples d'entraînement chargés : {train_examples['examples_count']}")
    
    return preprocessed_data, train_examples

def visualize_data_distribution(train_examples):
    """
    Visualise la distribution des paramètres d'entraînement
    
    Args:
        train_examples: Données d'exemples d'entraînement
    """
    print("\n📊 Visualisation de la distribution des données...")
    
    gap_values = train_examples['gap_values']
    L_ecran_values = train_examples['L_ecran_values']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribution des gaps
    ax1.hist(gap_values, bins=10, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Gap (µm)')
    ax1.set_ylabel('Fréquence')
    ax1.set_title('Distribution des valeurs de Gap')
    ax1.grid(True, alpha=0.3)
    
    # Distribution des L_ecran
    ax2.hist(L_ecran_values, bins=10, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('L_écran (µm)')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution des valeurs de L_écran')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("   ✅ Graphique sauvegardé : data_distribution.png")
    plt.show()

def visualize_profile_examples(preprocessed_data, train_examples):
    """
    Visualise quelques exemples de profils d'entraînement et de test
    
    Args:
        preprocessed_data: Données prétraitées principales
        train_examples: Exemples d'entraînement
    """
    print("\n🔍 Visualisation d'exemples de profils...")
    
    r_train = preprocessed_data['r_train']
    r_test = preprocessed_data['r_test']
    I_test = preprocessed_data['I_profiles_test_interpolated']
    ratio_examples = train_examples['ratio_examples']
    gap_values = train_examples['gap_values']
    L_ecran_values = train_examples['L_ecran_values']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Profils d'entraînement (3 premiers exemples)
    for i in range(3):
        ax = axes[0, i]
        ax.plot(r_train, ratio_examples[i], 'b-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Rayon r (µm)')
        ax.set_ylabel('Ratio')
        ax.set_title(f'Train - Gap: {gap_values[i]:.4f}µm\nL_écran: {L_ecran_values[i]:.3f}µm')
        ax.grid(True, alpha=0.3)
    
    # Profils de test (3 premiers exemples)
    for i in range(3):
        ax = axes[1, i]
        ax.plot(r_test, I_test[i], 'r-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Rayon r (µm)')
        ax.set_ylabel('Intensité')
        ax.set_title(f'Test - Profil expérimental {i+1}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Exemples de profils d\'entraînement et de test', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('profile_examples.png', dpi=300, bbox_inches='tight')
    print("   ✅ Graphique sauvegardé : profile_examples.png")
    plt.show()

def compare_train_test_ranges(preprocessed_data):
    """
    Compare les plages de valeurs entre train et test
    
    Args:
        preprocessed_data: Données prétraitées principales
    """
    print("\n⚖️ Comparaison des plages train vs test...")
    
    r_train = preprocessed_data['r_train']
    r_test = preprocessed_data['r_test']
    I_test = preprocessed_data['I_profiles_test_interpolated']
    
    print(f"📏 Plages radiales :")
    print(f"   Train : [{r_train.min():.6f}, {r_train.max():.6f}] µm (longueur: {len(r_train)})")
    print(f"   Test  : [{r_test.min():.6f}, {r_test.max():.6f}] µm (longueur: {len(r_test)})")

    # Vérifier si les longueurs sont identiques
    if len(r_train) == len(r_test):
        correspondance = np.allclose(r_train, r_test, rtol=1e-10)
        print(f"   Correspondance : {'✅' if correspondance else '❌'}")
    else:
        print(f"   Correspondance : ❌ (longueurs différentes: {len(r_train)} vs {len(r_test)})")
        # Comparer les plages min/max
        train_range_match = abs(r_train.min() - r_test.min()) < 1e-6 and abs(r_train.max() - r_test.max()) < 1e-6
        print(f"   Plages min/max identiques : {'✅' if train_range_match else '❌'}")
    
    print(f"\n📊 Statistiques des intensités :")
    print(f"   Test - Min : {I_test.min():.6f}")
    print(f"   Test - Max : {I_test.max():.6f}")
    print(f"   Test - Moyenne : {I_test.mean():.6f}")
    print(f"   Test - Écart-type : {I_test.std():.6f}")

def generate_summary_report(preprocessed_data, train_examples):
    """
    Génère un rapport de synthèse
    
    Args:
        preprocessed_data: Données prétraitées principales
        train_examples: Exemples d'entraînement
    """
    print("\n📋 Génération du rapport de synthèse...")
    
    report = f"""
# Rapport de Prétraitement des Données
## Neural_Network_Gap_Lecran_Prediction

### Paramètres de prétraitement
- **Plage radiale** : [{preprocessed_data['r_min']:.6f}, {preprocessed_data['r_max']:.6f}] µm
- **Espacement** : {preprocessed_data['delta_r']:.6f} µm
- **Points par profil** : {preprocessed_data['points_per_profile']}

### Données d'entraînement
- **Nombre de fichiers** : {preprocessed_data['train_files_count']}
- **Troncature appliquée** : indices 200 à 800 (sur 1000 points originaux)
- **Format** : fichiers .mat avec variables 'x' et 'ratio'

### Données de test
- **Nombre de profils** : {preprocessed_data['test_profiles_count']}
- **Source** : profile_exp_PS_3um_z_positive.mat
- **Interpolation** : {preprocessed_data['test_profiles_count']} profils interpolés à {preprocessed_data['points_per_profile']} points

### Exemples analysés
- **Gap min** : {min(train_examples['gap_values']):.6f} µm
- **Gap max** : {max(train_examples['gap_values']):.6f} µm
- **L_écran min** : {min(train_examples['L_ecran_values']):.3f} µm
- **L_écran max** : {max(train_examples['L_ecran_values']):.3f} µm

### Fichiers générés
- `preprocessed_data.npz` : Données principales prétraitées
- `train_examples.npz` : Exemples d'entraînement
- `comparison_train_test_preprocessed.png` : Visualisation comparative
- `data_distribution.png` : Distribution des paramètres
- `profile_examples.png` : Exemples de profils

### Prochaines étapes
1. Charger toutes les données d'entraînement avec les mêmes paramètres
2. Créer les datasets d'entraînement/validation/test
3. Entraîner le réseau de neurones
4. Évaluer les performances sur les données expérimentales
"""
    
    with open('preprocessing_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("   ✅ Rapport sauvegardé : preprocessing_report.md")

def main():
    """Fonction principale de démonstration"""
    print("🧠 Démonstration des données prétraitées")
    print("=" * 50)
    
    try:
        # Charger les données
        preprocessed_data, train_examples = load_preprocessed_data()
        
        # Visualisations
        visualize_data_distribution(train_examples)
        visualize_profile_examples(preprocessed_data, train_examples)
        
        # Comparaisons
        compare_train_test_ranges(preprocessed_data)
        
        # Rapport
        generate_summary_report(preprocessed_data, train_examples)
        
        print("\n✅ Démonstration terminée avec succès!")
        print("📁 Fichiers générés :")
        print("   - data_distribution.png")
        print("   - profile_examples.png") 
        print("   - preprocessing_report.md")
        
    except Exception as e:
        print(f"❌ Erreur lors de la démonstration : {e}")
        raise

if __name__ == "__main__":
    main()
