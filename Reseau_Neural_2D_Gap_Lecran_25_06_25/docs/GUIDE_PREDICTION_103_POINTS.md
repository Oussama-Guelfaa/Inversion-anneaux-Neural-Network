# 📊 Guide de Prédiction avec 103 Points

**Auteur:** Oussama GUELFAA  
**Date:** 25/06/2025  
**Objectif:** Utiliser le modèle entraîné sur 600 points avec des données de 103 points

## 🎯 **Problématique**

Le modèle de réseau de neurones a été entraîné sur des profils d'intensité de **600 points**, mais vous disposez de données expérimentales avec seulement **103 points**. Il faut adapter ces données pour les rendre compatibles avec le modèle.

## 🔧 **Solutions Implémentées**

### **5 Méthodes d'Adaptation Disponibles :**

#### **1. Interpolation Linéaire (`linear`)**
- **Principe :** Interpolation linéaire simple entre les points
- **Avantages :** Rapide, préserve les tendances générales
- **Inconvénients :** Peut créer des artefacts angulaires
- **Usage :** Données avec peu de bruit, tendances simples

#### **2. Interpolation Cubique (`cubic`)**
- **Principe :** Spline cubique pour interpolation lisse
- **Avantages :** Courbes lisses, préserve les courbures
- **Inconvénients :** Peut créer des oscillations
- **Usage :** **Recommandé par défaut** pour la plupart des cas

#### **3. Spline avec Lissage (`spline`)**
- **Principe :** Spline avec facteur de lissage pour réduire le bruit
- **Avantages :** Réduit le bruit, courbes très lisses
- **Inconvénients :** Peut perdre des détails fins
- **Usage :** Données bruitées

#### **4. Padding + Interpolation (`padding`)**
- **Principe :** Interpolation à 300 points puis padding symétrique
- **Avantages :** Préserve mieux la structure originale
- **Inconvénients :** Peut créer des discontinuités aux bords
- **Usage :** Données avec structure importante aux extrémités

#### **5. Reconstruction Fourier (`fourier`)**
- **Principe :** FFT → Zero-padding → IFFT
- **Avantages :** Préserve les fréquences importantes
- **Inconvénients :** Peut créer des artefacts de Gibbs
- **Usage :** Signaux périodiques ou avec composantes fréquentielles

## 🚀 **Utilisation du Script**

### **Commandes de Base :**

```bash
# Test avec toutes les méthodes + visualisation
python predict_103_points.py --method all --visualize

# Utilisation avec méthode spécifique (recommandée)
python predict_103_points.py --method cubic

# Avec vos propres données (fichier)
python predict_103_points.py --method cubic --data mon_profil_103pts.txt

# Avec données en ligne de commande
python predict_103_points.py --method cubic --data "1.0,0.95,0.9,0.85,..."
```

### **Paramètres Disponibles :**

- `--method` : Méthode d'adaptation (`linear`, `cubic`, `spline`, `padding`, `fourier`, `all`)
- `--data` : Données (fichier ou valeurs séparées par virgules)
- `--visualize` : Créer des graphiques de comparaison

## 📊 **Format des Données d'Entrée**

### **Fichier Texte :**
```
1.025
1.024
1.023
...
(103 valeurs au total)
```

### **Ligne de Commande :**
```bash
--data "1.025,1.024,1.023,1.022,..."
```

### **Contraintes :**
- **Exactement 103 points** (ajustement automatique si différent)
- **Valeurs numériques** (ratios d'intensité)
- **Plage recommandée :** 0.5 - 1.5 (comme dans le dataset d'entraînement)

## 🎯 **Recommandations d'Usage**

### **Choix de la Méthode :**

#### **Pour des données expérimentales typiques :**
```bash
python predict_103_points.py --method cubic --data votre_fichier.txt
```

#### **Pour des données bruitées :**
```bash
python predict_103_points.py --method spline --data votre_fichier.txt
```

#### **Pour comparer toutes les méthodes :**
```bash
python predict_103_points.py --method all --data votre_fichier.txt --visualize
```

### **Validation des Résultats :**

1. **Vérifiez la cohérence** : Les prédictions doivent être dans les plages d'entraînement
   - **Gap :** 0.005 - 0.400 µm
   - **L_écran :** 4.0 - 8.0 µm

2. **Comparez les méthodes** : Si les résultats varient beaucoup, vos données peuvent être problématiques

3. **Visualisez l'adaptation** : Utilisez `--visualize` pour voir comment vos 103 points sont transformés

## ⚠️ **Limitations et Précautions**

### **Limitations :**
- **Perte d'information** : 103 → 600 points introduit de l'interpolation
- **Domaine d'entraînement** : Le modèle n'a vu que des profils du dataset d'entraînement
- **Extrapolation** : Données très différentes du dataset peuvent donner des résultats aberrants

### **Précautions :**
- **Vérifiez la plage** : Vos données doivent ressembler aux profils d'entraînement
- **Testez plusieurs méthodes** : Comparez les résultats pour validation
- **Analysez la cohérence** : Des prédictions très différentes entre méthodes indiquent un problème

## 📈 **Exemple Complet**

### **1. Préparer vos données (103 points) :**
```bash
# Créer un fichier avec vos 103 mesures
echo "1.025
1.024
1.023
..." > mon_profil.txt
```

### **2. Tester toutes les méthodes :**
```bash
python predict_103_points.py --method all --data mon_profil.txt --visualize
```

### **3. Analyser les résultats :**
```
📋 Méthode: LINEAR
   Gap prédit: 0.1234 µm
   L_écran prédit: 5.6 µm

📋 Méthode: CUBIC
   Gap prédit: 0.1245 µm
   L_écran prédit: 5.7 µm
```

### **4. Choisir la meilleure méthode :**
- **Résultats cohérents** → Confiance élevée
- **Résultats dispersés** → Vérifier les données

### **5. Utiliser la méthode retenue :**
```bash
python predict_103_points.py --method cubic --data mon_profil.txt
```

## 🔍 **Diagnostic des Problèmes**

### **Prédictions Aberrantes :**
- **Gap négatif ou > 0.4µm** → Données hors domaine d'entraînement
- **L_écran < 4µm ou > 8µm** → Profil non représentatif

### **Solutions :**
1. **Vérifier les données** : Normalisation, unités, plage de valeurs
2. **Tester d'autres méthodes** : Certaines peuvent mieux s'adapter
3. **Comparer avec profils connus** : Utiliser des données de référence

## 📁 **Fichiers Générés**

- **`adaptation_methods_103_to_600.png`** : Visualisation des méthodes d'adaptation
- **Logs détaillés** : Statistiques et résultats de chaque méthode

## 🎯 **Résumé Pratique**

### **Usage Recommandé :**
```bash
# 1. Test initial avec toutes les méthodes
python predict_103_points.py --method all --data vos_donnees.txt --visualize

# 2. Utilisation avec la meilleure méthode (généralement cubic)
python predict_103_points.py --method cubic --data vos_donnees.txt
```

### **Critères de Validation :**
- ✅ **Gap :** 0.005 - 0.400 µm
- ✅ **L_écran :** 4.0 - 8.0 µm  
- ✅ **Cohérence** entre méthodes
- ✅ **Profil adapté** ressemble aux données d'entraînement

**Votre modèle est maintenant prêt à traiter des données de 103 points ! 🚀**
