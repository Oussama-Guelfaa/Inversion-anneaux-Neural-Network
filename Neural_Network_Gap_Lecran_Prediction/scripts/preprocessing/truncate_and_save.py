#!/usr/bin/env python3
"""
Script de troncature et sauvegarde des fichiers .mat
Auteur: Oussama GUELFAA
Date: 15/07/2025

Ce script :
1. Charge tous les fichiers .mat du dossier Train/
2. Applique la troncature (indices 200-800) sur les données 'ratio'
3. Sauvegarde les nouveaux fichiers .mat dans Train_Truncated/
"""

import os
import glob
import scipy.io
import numpy as np
from tqdm import tqdm
import time

def truncate_and_save_mat_files(input_dir="Train", output_dir="Train_Truncated", 
                               start_idx=200, end_idx=800):
    """
    Tronque les fichiers .mat et les sauvegarde dans un nouveau dossier
    
    Args:
        input_dir: Dossier source avec les fichiers .mat originaux
        output_dir: Dossier de destination pour les fichiers tronqués
        start_idx: Index de début pour la troncature (200)
        end_idx: Index de fin pour la troncature (800)
    """
    
    print("🔧 Script de Troncature et Sauvegarde des Fichiers .mat")
    print("=" * 60)
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Dossier de sortie créé: {output_dir}")
    
    # Obtenir la liste des fichiers .mat
    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))
    mat_files = [f for f in mat_files if not f.endswith('labels.mat')]
    
    print(f"📂 Fichiers trouvés: {len(mat_files)}")
    print(f"🔪 Troncature: indices {start_idx} à {end_idx} (soit {end_idx-start_idx+1} points)")
    
    # Statistiques
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    # Traiter chaque fichier avec barre de progression
    for i, mat_file in enumerate(tqdm(mat_files, desc="Troncature en cours")):
        try:
            # Nom du fichier de sortie
            filename = os.path.basename(mat_file)
            output_file = os.path.join(output_dir, filename)
            
            # Charger le fichier .mat original
            data = scipy.io.loadmat(mat_file)
            
            # Extraire les données ratio
            if 'ratio' not in data:
                print(f"⚠️ Pas de 'ratio' dans {filename}")
                error_count += 1
                continue
                
            ratio_original = data['ratio'].flatten()
            
            # Vérifier la taille
            if len(ratio_original) <= end_idx:
                print(f"⚠️ Fichier trop petit {filename}: {len(ratio_original)} points")
                error_count += 1
                continue
            
            # Appliquer la troncature
            ratio_truncated = ratio_original[start_idx:end_idx+1]
            
            # Créer le nouveau dictionnaire de données
            new_data = {}
            
            # Copier toutes les métadonnées importantes
            for key, value in data.items():
                if not key.startswith('__'):  # Ignorer les métadonnées MATLAB
                    if key == 'ratio':
                        # Remplacer par la version tronquée
                        new_data[key] = ratio_truncated.reshape(-1, 1)  # Format colonne
                    elif key == 'x' and 'x' in data:
                        # Tronquer aussi x si présent (pour cohérence)
                        x_original = data['x'].flatten()
                        if len(x_original) > end_idx:
                            x_truncated = x_original[start_idx:end_idx+1]
                            new_data[key] = x_truncated.reshape(-1, 1)
                        else:
                            new_data[key] = value
                    else:
                        # Copier tel quel (gap, L_ecran_subs, etc.)
                        new_data[key] = value
            
            # Sauvegarder le nouveau fichier .mat
            scipy.io.savemat(output_file, new_data)
            success_count += 1
            
            # Affichage périodique
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(mat_files) - i - 1) / rate
                print(f"   📈 {i+1}/{len(mat_files)} - {rate:.1f} fichiers/s - Reste: {remaining/60:.1f}min")
                
        except Exception as e:
            print(f"❌ Erreur avec {filename}: {e}")
            error_count += 1
            continue
    
    # Statistiques finales
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ TRONCATURE TERMINÉE !")
    print(f"📊 Statistiques:")
    print(f"   ✅ Fichiers traités avec succès: {success_count}")
    print(f"   ❌ Fichiers en erreur: {error_count}")
    print(f"   📁 Total fichiers: {len(mat_files)}")
    print(f"   ⏱️ Temps total: {total_time/60:.1f} minutes")
    print(f"   🚀 Vitesse moyenne: {len(mat_files)/total_time:.1f} fichiers/seconde")
    print(f"   📁 Fichiers sauvegardés dans: {output_dir}")
    
    return success_count, error_count

def verify_truncated_files(output_dir="Train_Truncated", expected_points=601):
    """
    Vérifie que les fichiers tronqués ont la bonne taille
    """
    print(f"\n🔍 Vérification des fichiers tronqués...")
    
    mat_files = glob.glob(os.path.join(output_dir, "*.mat"))
    
    if not mat_files:
        print(f"❌ Aucun fichier trouvé dans {output_dir}")
        return
    
    # Vérifier quelques fichiers
    sample_files = mat_files[:5]  # Vérifier les 5 premiers
    
    for mat_file in sample_files:
        try:
            data = scipy.io.loadmat(mat_file)
            ratio = data['ratio'].flatten()
            filename = os.path.basename(mat_file)
            
            print(f"   📄 {filename}: {len(ratio)} points", end="")
            
            if len(ratio) == expected_points:
                print(" ✅")
            else:
                print(f" ❌ (attendu: {expected_points})")
                
        except Exception as e:
            print(f"   ❌ Erreur avec {os.path.basename(mat_file)}: {e}")
    
    print(f"✅ Vérification terminée sur {len(sample_files)} fichiers échantillons")

def compare_original_vs_truncated(original_file, truncated_file):
    """
    Compare un fichier original avec sa version tronquée
    """
    print(f"\n🔍 Comparaison: {os.path.basename(original_file)}")
    
    try:
        # Charger les deux fichiers
        orig_data = scipy.io.loadmat(original_file)
        trunc_data = scipy.io.loadmat(truncated_file)
        
        # Comparer les ratios
        orig_ratio = orig_data['ratio'].flatten()
        trunc_ratio = trunc_data['ratio'].flatten()
        
        print(f"   📊 Original: {len(orig_ratio)} points")
        print(f"   📊 Tronqué: {len(trunc_ratio)} points")
        
        # Vérifier que la troncature est correcte
        expected_trunc = orig_ratio[200:801]
        
        if np.allclose(trunc_ratio, expected_trunc):
            print("   ✅ Troncature correcte !")
        else:
            print("   ❌ Problème de troncature !")
            
        # Vérifier les labels
        if 'gap' in orig_data and 'gap' in trunc_data:
            orig_gap = float(orig_data['gap'][0, 0])
            trunc_gap = float(trunc_data['gap'][0, 0])
            print(f"   🎯 Gap: {orig_gap} → {trunc_gap} {'✅' if orig_gap == trunc_gap else '❌'}")
            
        if 'L_ecran_subs' in orig_data and 'L_ecran_subs' in trunc_data:
            orig_L = float(orig_data['L_ecran_subs'][0, 0])
            trunc_L = float(trunc_data['L_ecran_subs'][0, 0])
            print(f"   🎯 L_écran: {orig_L} → {trunc_L} {'✅' if orig_L == trunc_L else '❌'}")
            
    except Exception as e:
        print(f"   ❌ Erreur de comparaison: {e}")

def main():
    """Fonction principale"""
    print("🧠 Neural_Network_Gap_Lecran_Prediction - Troncature des Données")
    print("=" * 70)
    
    # Paramètres
    input_dir = "Train"
    output_dir = "Train_Truncated"
    start_idx = 200
    end_idx = 800
    
    # Vérifier que le dossier source existe
    if not os.path.exists(input_dir):
        print(f"❌ Dossier source '{input_dir}' introuvable !")
        return
    
    # Effectuer la troncature
    success_count, error_count = truncate_and_save_mat_files(
        input_dir=input_dir,
        output_dir=output_dir,
        start_idx=start_idx,
        end_idx=end_idx
    )
    
    if success_count > 0:
        # Vérifier les fichiers tronqués
        verify_truncated_files(output_dir, expected_points=601)
        
        # Comparer un fichier exemple
        original_files = glob.glob(os.path.join(input_dir, "*.mat"))
        original_files = [f for f in original_files if not f.endswith('labels.mat')]
        
        if original_files:
            original_file = original_files[0]
            truncated_file = os.path.join(output_dir, os.path.basename(original_file))
            
            if os.path.exists(truncated_file):
                compare_original_vs_truncated(original_file, truncated_file)
    
    print(f"\n🎉 Script terminé ! Fichiers tronqués disponibles dans '{output_dir}/'")

if __name__ == "__main__":
    main()
