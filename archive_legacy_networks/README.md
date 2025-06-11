# 📦 Archive Legacy Networks

**Author:** Oussama GUELFAA  
**Date:** 10 - 01 - 2025

## 📖 Description

Ce dossier contient les **anciens réseaux de neurones** qui ont été remplacés par la nouvelle structure modulaire. Ces réseaux sont conservés pour référence historique et comparaison.

## 📁 Contenu

### Neural_Network/
- **Description**: Version originale du régresseur avancé
- **Fichiers**: Scripts Python originaux, modèles, résultats
- **Status**: Remplacé par `Reseau_Advanced_Regressor`

### Neural_Network_Gap_Prediction_25_01_25/
- **Description**: Version originale de prédiction gap
- **Fichiers**: CNN pour prédiction gap, analyses
- **Status**: Remplacé par `Reseau_Gap_Prediction_CNN`

### Neural_Network_Noise_Robustness_Test_10_01_25/
- **Description**: Version originale des tests de robustesse
- **Fichiers**: Tests de bruit, analyses de robustesse
- **Status**: Remplacé par `Reseau_Noise_Robustness`

### Neural_Network_Overfitting_Test_10_01_25/
- **Description**: Version originale des tests d'overfitting
- **Fichiers**: Tests de validation, analyses diagnostiques
- **Status**: Remplacé par `Reseau_Overfitting_Test`

## 🔄 Migration

Ces réseaux ont été **restructurés et améliorés** dans la nouvelle architecture modulaire :

| Ancien Réseau | Nouveau Réseau | Améliorations |
|---------------|----------------|---------------|
| Neural_Network | Reseau_Advanced_Regressor | Structure modulaire, config YAML |
| Neural_Network_Gap_Prediction_25_01_25 | Reseau_Gap_Prediction_CNN | Script autonome, documentation |
| Neural_Network_Noise_Robustness_Test_10_01_25 | Reseau_Noise_Robustness | Configuration flexible, plots auto |
| Neural_Network_Overfitting_Test_10_01_25 | Reseau_Overfitting_Test | Critères clairs, monitoring |

## ⚠️ Statut

- **Archivé**: Ces réseaux ne sont plus maintenus
- **Référence**: Conservés pour comparaison historique
- **Utilisation**: Utiliser les nouveaux réseaux modulaires à la place

## 🚀 Nouveaux Réseaux

Pour utiliser les versions modernes et améliorées :

```bash
# Au lieu de l'ancien Neural_Network
cd ../Reseau_Advanced_Regressor
python run.py

# Au lieu de l'ancien Gap Prediction
cd ../Reseau_Gap_Prediction_CNN
python run.py

# Au lieu de l'ancien Noise Robustness
cd ../Reseau_Noise_Robustness
python run.py

# Au lieu de l'ancien Overfitting Test
cd ../Reseau_Overfitting_Test
python run.py
```

**Ces archives sont conservées uniquement pour référence historique.** ✅
