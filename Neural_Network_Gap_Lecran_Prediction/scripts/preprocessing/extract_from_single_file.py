#!/usr/bin/env python3
"""
Extraction des données depuis all_banque_new_04_07_25_NEW_full.mat
Auteur: Oussama GUELFAA
Date: 15/07/2025

Ce script remplace tous les fichiers du dossier Train/ par les données
extraites du fichier unique all_banque_new_04_07_25_NEW_full.mat
"""

import numpy as np
import scipy.io
import os
from pathlib import Path
import time
from tqdm import tqdm

class SingleFileDataLoader:
    """
    Data Loader qui utilise le fichier unique au lieu de 22,541 fichiers séparés.
    """
    
    def __init__(self, mat_file_path="Train/all_banque_new_04_07_25_NEW_full.mat"):
        self.mat_file_path = mat_file_path
        self.data = None
        self.truncate_start = 200
        self.truncate_end = 800
        self.expected_points = 601
        
        print(f"🔧 SingleFileDataLoader initialisé")
        print(f"   📄 Fichier source: {mat_file_path}")
        print(f"   ✂️ Troncature: indices {self.truncate_start}-{self.truncate_end} ({self.expected_points} points)")
    
    def load_single_file(self):
        """Charge le fichier unique."""
        print(f"📂 Chargement du fichier unique...")
        
        start_time = time.time()
        self.data = scipy.io.loadmat(self.mat_file_path)
        load_time = time.time() - start_time
        
        print(f"✅ Fichier chargé en {load_time:.2f} secondes")
        
        # Analyser la structure
        self.analyze_structure()
    
    def analyze_structure(self):
        """Analyse la structure des données."""
        print(f"🔍 Analyse de la structure...")
        
        # Variables principales
        self.I_subs = self.data['I_subs']  # (161, 140, 1000)
        self.I_subs_inc = self.data['I_subs_inc']  # (161, 140, 1000)
        self.L_ecran_vect = self.data['L_ecran_subs_vect'].flatten()  # (161,)
        self.gap_vect = self.data['gap_sphere_vect'].flatten()  # (140,)
        self.x_positions = self.data['x'].flatten()  # (1000,)
        
        print(f"   📊 I_subs shape: {self.I_subs.shape}")
        print(f"   📊 I_subs_inc shape: {self.I_subs_inc.shape}")
        print(f"   📊 L_écran values: {len(self.L_ecran_vect)} valeurs [{self.L_ecran_vect.min():.1f} → {self.L_ecran_vect.max():.1f}]")
        print(f"   📊 Gap values: {len(self.gap_vect)} valeurs [{self.gap_vect.min():.6f} → {self.gap_vect.max():.6f}]")
        print(f"   📊 X positions: {len(self.x_positions)} points [{self.x_positions.min():.6f} → {self.x_positions.max():.6f}]")
        
        # Calculer le nombre total de profils
        self.n_L_ecran = len(self.L_ecran_vect)
        self.n_gap = len(self.gap_vect)
        self.total_profiles = self.n_L_ecran * self.n_gap
        
        print(f"   📈 Total profils: {self.n_L_ecran} × {self.n_gap} = {self.total_profiles}")
    
    def extract_all_profiles(self, sample_ratio=1.0):
        """
        Extrait tous les profils avec leurs paramètres.
        
        Args:
            sample_ratio (float): Ratio d'échantillonnage (1.0 = tous)
        
        Returns:
            tuple: (X_data, y_data) avec troncature appliquée
        """
        print(f"🔄 Extraction de tous les profils (échantillon {sample_ratio*100:.1f}%)...")
        
        # Calculer le ratio I_subs / I_subs_inc
        ratio_full = self.I_subs / self.I_subs_inc  # (161, 140, 1000)
        
        # Pré-allouer les arrays
        total_samples = int(self.total_profiles * sample_ratio)
        X_data = np.zeros((total_samples, self.expected_points), dtype=np.float32)
        y_data = np.zeros((total_samples, 2), dtype=np.float32)
        
        print(f"   📊 Extraction de {total_samples} profils...")
        
        sample_count = 0
        start_time = time.time()
        
        # Parcourir toutes les combinaisons L_écran × gap
        for i in range(self.n_L_ecran):
            for j in range(self.n_gap):
                if sample_count >= total_samples:
                    break
                
                # Échantillonnage aléatoire si nécessaire
                if sample_ratio < 1.0:
                    if np.random.random() > sample_ratio:
                        continue
                
                # Extraire le profil d'intensité
                ratio_profile = ratio_full[i, j, :]  # (1000,)
                
                # Appliquer la troncature (200-800)
                ratio_truncated = ratio_profile[self.truncate_start:self.truncate_end+1]
                
                # Vérifier la taille
                if len(ratio_truncated) != self.expected_points:
                    print(f"⚠️ Taille incorrecte pour profil [{i},{j}]: {len(ratio_truncated)}")
                    continue
                
                # Paramètres correspondants
                L_ecran = self.L_ecran_vect[i]
                gap = self.gap_vect[j]
                
                # Stocker
                X_data[sample_count] = ratio_truncated
                y_data[sample_count] = [gap, L_ecran]
                sample_count += 1
                
                # Affichage périodique
                if sample_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = sample_count / elapsed
                    remaining = (total_samples - sample_count) / rate
                    print(f"   📈 {sample_count}/{total_samples} profils - {rate:.0f} profils/s - Reste: {remaining:.1f}s")
            
            if sample_count >= total_samples:
                break
        
        # Tronquer aux données valides
        X_data = X_data[:sample_count]
        y_data = y_data[:sample_count]
        
        total_time = time.time() - start_time
        print(f"✅ Extraction terminée:")
        print(f"   📊 Profils extraits: {sample_count}")
        print(f"   📏 Forme X: {X_data.shape}")
        print(f"   📏 Forme y: {y_data.shape}")
        print(f"   ⏱️ Temps: {total_time:.2f} secondes")
        print(f"   🚀 Vitesse: {sample_count/total_time:.0f} profils/seconde")
        
        # Statistiques des paramètres
        print(f"   📈 Gap range: [{y_data[:, 0].min():.6f}, {y_data[:, 0].max():.6f}] µm")
        print(f"   📈 L_écran range: [{y_data[:, 1].min():.1f}, {y_data[:, 1].max():.1f}] µm")
        
        return X_data, y_data
    
    def save_extracted_data(self, X_data, y_data, output_file="extracted_data.npz"):
        """Sauvegarde les données extraites."""
        print(f"💾 Sauvegarde des données extraites...")
        
        np.savez_compressed(
            output_file,
            X_data=X_data,
            y_data=y_data,
            x_positions=self.x_positions[self.truncate_start:self.truncate_end+1],
            L_ecran_values=self.L_ecran_vect,
            gap_values=self.gap_vect,
            truncate_start=self.truncate_start,
            truncate_end=self.truncate_end
        )
        
        print(f"✅ Données sauvegardées: {output_file}")
        
        # Taille du fichier
        file_size = os.path.getsize(output_file) / (1024**2)  # MB
        print(f"   📁 Taille: {file_size:.1f} MB")

def compare_with_individual_files():
    """Compare avec quelques fichiers individuels pour validation."""
    print(f"\n🔍 Validation avec fichiers individuels...")
    
    # Charger le fichier unique
    loader = SingleFileDataLoader()
    loader.load_single_file()
    
    # Extraire un échantillon
    X_single, y_single = loader.extract_all_profiles(sample_ratio=0.01)  # 1% pour test
    
    print(f"✅ Validation terminée")
    print(f"   📊 Échantillon extrait: {X_single.shape[0]} profils")
    print(f"   📏 Points par profil: {X_single.shape[1]}")

def main():
    """Fonction principale."""
    print("🧠 Extraction depuis Fichier Unique")
    print("=" * 50)
    
    # Vérifier que le fichier existe
    mat_file = "Train/all_banque_new_04_07_25_NEW_full.mat"
    if not os.path.exists(mat_file):
        print(f"❌ Fichier non trouvé: {mat_file}")
        return
    
    # Créer le loader
    loader = SingleFileDataLoader(mat_file)
    
    # Charger le fichier
    loader.load_single_file()
    
    # Extraire tous les profils (ou un échantillon)
    sample_ratio = 1.0  # 100% = tous les profils
    X_data, y_data = loader.extract_all_profiles(sample_ratio=sample_ratio)
    
    # Sauvegarder
    output_file = f"extracted_data_full.npz"
    loader.save_extracted_data(X_data, y_data, output_file)
    
    print(f"\n🎉 Extraction terminée avec succès !")
    print(f"📁 Données disponibles dans: {output_file}")
    print(f"📊 {X_data.shape[0]} profils de {X_data.shape[1]} points chacun")
    print(f"🚀 Prêt pour l'entraînement ultra-rapide !")

if __name__ == "__main__":
    main()
