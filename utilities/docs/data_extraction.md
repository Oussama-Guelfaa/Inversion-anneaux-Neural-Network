# Explication Détaillée : extract_training_data.py

**Auteur:** Oussama GUELFAA  
**Date:** 05 - 06 - 2025

## 🎯 Objectif du Script

Le script `extract_training_data.py` extrait et organise les données du fichier MATLAB `all_banque_new_24_01_25_NEW_full.mat` pour l'entraînement du réseau de neurones.

## 📊 Structure des Données d'Entrée

### Variables dans le fichier .mat :
```matlab
L_ecran_subs_vect : [1×33] - Valeurs de L_ecran (6.0 à 14.0 µm)
gap_sphere_vect   : [1×30] - Valeurs de gap (0.025 à 1.5 µm)
I_subs           : [33×30×1000] - Intensités diffusées
I_subs_inc       : [33×30×1000] - Intensités incidentes
```

### Organisation 3D :
- **Dimension 1 (33)** : Différentes valeurs de L_ecran
- **Dimension 2 (30)** : Différentes valeurs de gap  
- **Dimension 3 (1000)** : Points radiaux du profil

## ⚡ Calcul du Module au Carré des Intensités

### 🔬 Principe Physique

En holographie, l'intensité mesurée est proportionnelle au **module au carré du champ électrique total** :

```
I ∝ |E_total|² = |E_incident + E_diffusé|²
```

### 📐 Calcul dans le Script

```python
# 1. Éviter la division par zéro
I_subs_inc_safe = np.where(I_subs_inc == 0, 1e-10, I_subs_inc)

# 2. Calcul du ratio normalisé
intensity_ratio = I_subs / I_subs_inc_safe
```

### 🧮 Explication Mathématique

Le ratio `I_subs/I_subs_inc` représente :

```
Ratio = |E_total|² / |E_incident|²
      = |E_incident + E_diffusé|² / |E_incident|²
      = |1 + E_diffusé/E_incident|²
```

**Pourquoi ce ratio ?**
1. **Normalisation** : Élimine les variations d'intensité de la source
2. **Contraste** : Met en évidence les effets de diffusion
3. **Invariance** : Indépendant de la puissance du laser

### 📈 Interprétation Physique

- **Ratio = 1** : Pas de diffusion (champ incident seul)
- **Ratio > 1** : Interférence constructive
- **Ratio < 1** : Interférence destructive
- **Oscillations** : Anneaux de diffraction caractéristiques

## 🔄 Transformation des Données

### Étape 1 : Reshape 3D → 2D
```python
# De (33, 30, 1000) vers (990, 1000)
X = intensity_ratio.reshape(n_L * n_g, n_radial)
```

### Étape 2 : Création des Grilles de Paramètres
```python
# Grille 2D des paramètres
L_grid, gap_grid = np.meshgrid(L_ecran_vect, gap_vect, indexing='ij')

# Aplatissement en vecteurs 1D
L_ecran_final = L_grid.flatten()  # 990 valeurs
gap_final = gap_grid.flatten()    # 990 valeurs
```

### Étape 3 : Organisation Finale
```python
X = intensity_ratio.reshape(990, 1000)  # Features
y = np.column_stack([L_ecran_final, gap_final])  # Targets
```

## 📋 Correspondance Échantillon ↔ Paramètres

Chaque ligne `i` du dataset correspond à :
```python
L_ecran = L_ecran_vect[i // 30]  # Division entière
gap = gap_vect[i % 30]           # Modulo
```

**Exemple :**
- Échantillon 0 : L_ecran[0], gap[0] = (6.0, 0.025)
- Échantillon 30 : L_ecran[1], gap[0] = (6.25, 0.025)
- Échantillon 989 : L_ecran[32], gap[29] = (14.0, 1.5)

## 💾 Fichiers de Sortie

### 1. `training_data.npz`
```python
{
    'X': array(990, 1000),      # Profils d'intensité
    'y': array(990, 2),         # Paramètres [L_ecran, gap]
    'metadata': dict            # Informations sur le dataset
}
```

### 2. `intensity_profiles_full.csv`
- **990 lignes** × **1000 colonnes**
- Format CSV pour inspection manuelle
- Chaque ligne = un profil radial complet

### 3. `parameters.csv`
```csv
L_ecran,gap
6.000000,0.025000
6.000000,0.075862
...
14.000000,1.500000
```

## 🎯 Avantages de cette Organisation

1. **Efficacité** : Format optimisé pour PyTorch/NumPy
2. **Traçabilité** : Correspondance claire échantillon ↔ paramètres
3. **Flexibilité** : Plusieurs formats de sortie
4. **Validation** : Visualisations automatiques pour vérification

## 🔍 Points Clés à Retenir

- **990 échantillons** = 33 × 30 combinaisons de paramètres
- **Ratio I_subs/I_subs_inc** = intensité physiquement normalisée
- **Profils 1D** = plus efficace que les images 2D pour les réseaux
- **Organisation matricielle** = correspondance directe paramètres ↔ profils
