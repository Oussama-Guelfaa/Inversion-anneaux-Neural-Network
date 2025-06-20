# 📖 Guide d'Utilisation - Réseau Neural Robuste

**Auteur:** Oussama GUELFAA  
**Date:** 19 - 06 - 2025  
**Projet:** Réseau Neural Robuste au Bruit 3%

---

## 🚀 **DÉMARRAGE RAPIDE**

### **1. Préparation de l'Environnement**
```bash
cd Reseau_Neural_Robust_3pct_Noise_19_06_25/src
pip install torch numpy matplotlib scikit-learn joblib
```

### **2. Entraînement du Modèle**
```bash
# Entraînement standard
python robust_trainer.py

# Entraînement avec 5% de bruit
python robust_trainer_5pct_noise.py
```

### **3. Test du Modèle**
```bash
# Test standard
python test_robust_model.py

# Test avec 5% de bruit
python test_robust_5pct_noise.py
```

### **4. Exécution Complète**
```bash
python run.py
```

---

## 📁 **STRUCTURE DES FICHIERS**

### **Scripts Principaux**
- `robust_trainer.py` : Entraînement robuste
- `test_robust_model.py` : Test de robustesse
- `robust_model_architecture.py` : Architecture du modèle
- `run.py` : Exécution complète

### **Données et Modèles**
- `data/` : Datasets augmentés
- `models/` : Modèles entraînés et scalers
- `results/` : Résultats JSON
- `plots/` : Graphiques de performance

---

## 🎯 **PARAMÈTRES CONFIGURABLES**

### **Entraînement**
- `noise_level` : Niveau de bruit (0.03 = 3%)
- `epochs` : Nombre d'epochs (200)
- `batch_size` : Taille des batches (64)
- `learning_rate` : Taux d'apprentissage (0.001)

### **Architecture**
- `input_size` : Taille d'entrée (600)
- `dropout_rate` : Taux de dropout (0.3)
- `use_attention` : Mécanisme d'attention (True)

---

## 📊 **INTERPRÉTATION DES RÉSULTATS**

### **Métriques Clés**
- **R²** : Coefficient de détermination (>0.8 = bon)
- **Accuracy** : Précision dans tolérance (>80% = objectif)
- **MAE** : Erreur absolue moyenne
- **RMSE** : Erreur quadratique moyenne

### **Tolérances**
- **Gap** : ±0.01 µm
- **L_ecran** : ±0.1 µm

---

**🎯 Objectif** : Guide pratique pour utiliser le réseau neural robuste.
