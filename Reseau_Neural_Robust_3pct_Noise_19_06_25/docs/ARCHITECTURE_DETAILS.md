# 🏗️ Architecture Détaillée du Modèle Robuste

**Auteur:** Oussama GUELFAA  
**Date:** 19 - 06 - 2025  
**Projet:** Réseau Neural Robuste au Bruit 3%

---

## 🛡️ **ARCHITECTURE ROBUSTE**

### **Composants Principaux**
1. **NoiseResistantBlock** : Blocs résistants au bruit
2. **RobustDualParameterModel** : Modèle dual Gap + L_ecran
3. **RobustLoss** : Fonction de perte robuste (MSE + Huber)
4. **Attention Multi-têtes** : Mécanisme d'attention (8 heads)

### **Paramètres du Modèle**
- **Total** : 1,396,914 paramètres
- **Entrée** : 600 points (profils d'intensité)
- **Sortie** : 2 paramètres (Gap, L_ecran)
- **Dropout** : 0.3 (adaptatif)

---

## 🔧 **INNOVATIONS TECHNIQUES**

### **1. Blocs Résistants au Bruit**
- Normalisation par batch
- Connexions résiduelles
- Dropout adaptatif
- Activation ReLU

### **2. Fonction de Perte Robuste**
- 70% MSE Loss (précision)
- 30% Huber Loss (robustesse)
- Gradient clipping (max_norm=1.0)

### **3. Mécanisme d'Attention**
- 8 têtes d'attention
- Normalisation par couches
- Connexions résiduelles

---

## 📊 **PERFORMANCES**

### **Robustesse au Bruit**
- **L_ecran** : Excellente (R² > 0.97)
- **Gap** : En amélioration (+10,000% vs original)
- **Stabilité** : Convergence rapide (11.6 min)

---

**🎯 Objectif** : Architecture optimisée pour la robustesse au bruit gaussien.
