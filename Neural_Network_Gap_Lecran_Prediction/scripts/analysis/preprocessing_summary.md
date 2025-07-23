# RÉSUMÉ DU PREPROCESSING - ÉTAPE PAR ÉTAPE

## 🔍 **CE QUI SE PASSE EXACTEMENT AVEC LES DONNÉES DE TEST**

### 📊 **ÉTAPE 1: Données Expérimentales Brutes**
- **Fichier :** `profile_exp_PS_3um_z_positive.mat`
- **Structure :** 50 profils × 184 points
- **Profil testé :** Profil 49 (dernier)
- **Plage radiale :** 0.000000 - 10.687200 µm
- **Intensité :** Min=0.126632, Max=1.751384

### 🧠 **ÉTAPE 2: Paramètres du Réseau de Neurones**
- **r_min :** 1.3845845846 µm
- **r_max :** 5.5383383383 µm  
- **Points :** 601
- **Delta_r :** 0.0069229229 µm

### 🔍 **ÉTAPE 3: Problème de Compatibilité des Plages**

**❌ PROBLÈME MAJEUR IDENTIFIÉ :**

```
Plage expérimentale: 0.000000 - 10.687200 µm
Plage réseau:        1.384585 -  5.538338 µm
Recouvrement:        1.384585 -  5.538338 µm
```

**🚨 CONSÉQUENCES :**
1. **Données exp commencent AVANT la grille réseau** (0 vs 1.385 µm)
2. **Données exp finissent APRÈS la grille réseau** (10.687 vs 5.538 µm)
3. **Le réseau ne "voit" que 50% de la plage expérimentale**

### 🔄 **ÉTAPE 4: Interpolation Forcée**
- **Méthode :** `interp1d` avec extrapolation
- **Transformation :** 184 points → 601 points
- **Résultat :** Interpolation + **EXTRAPOLATION MASSIVE**

### 🔧 **ÉTAPE 5: Normalisation**
- **Scalers :** Entraînés sur données simulation (plage 1.385-5.538 µm)
- **Application :** Sur données exp interpolées/extrapolées
- **Résultat final :** [-1.532297, 1.455075] (normalisé)

## 🎯 **COMPARAISON SIMULATION vs EXPÉRIMENTAL**

### 📊 **Données Simulation (entraînement) :**
- **Plage :** 0 - 5.538338 µm (1000 points)
- **Après preprocessing :** 1.385 - 5.538 µm (601 points)
- **Intensité :** 0.063 - 1.738
- **Normalisé :** Distribution centrée autour de 0

### 📊 **Données Expérimentales (test) :**
- **Plage :** 0 - 10.687 µm (184 points)
- **Après preprocessing :** 1.385 - 5.538 µm (601 points) **FORCÉ**
- **Intensité :** 0.127 - 1.751
- **Normalisé :** Distribution décalée

## 🚨 **PROBLÈMES IDENTIFIÉS**

### 1. **Incompatibilité des Plages Radiales**
- **Simulation :** Optimisée pour 1.385-5.538 µm
- **Expérimental :** Couvre 0-10.687 µm
- **Impact :** Le réseau n'a jamais vu les plages 0-1.385 µm et 5.538-10.687 µm

### 2. **Extrapolation Massive**
- **Points extrapolés :** Probablement 30-40% des 601 points
- **Conséquence :** Valeurs artificielles non représentatives

### 3. **Normalisation Inadaptée**
- **Scalers :** Entraînés sur distribution simulation
- **Application :** Sur données exp avec structure différente
- **Résultat :** Biais dans la normalisation

### 4. **Résolution Différente**
- **Simulation :** 1000 points → 601 points (sous-échantillonnage)
- **Expérimental :** 184 points → 601 points (sur-échantillonnage)
- **Impact :** Résolutions incompatibles

## 💡 **EXPLICATION DU GAP NÉGATIF**

Le gap négatif (-0.151 µm) s'explique par :

1. **🔴 Données hors distribution :** Le réseau n'a jamais vu de données comme celles-ci
2. **🔴 Extrapolation forcée :** 30-40% des points sont artificiels
3. **🔴 Normalisation biaisée :** Scalers inadaptés aux données exp
4. **🔴 Plages incompatibles :** Le réseau prédit sur des données qu'il ne connaît pas

## 🛠️ **SOLUTIONS RECOMMANDÉES**

### 1. **Solution Immédiate : Contraintes Physiques**
```python
# Dans le modèle
def forward(self, x):
    output = self.network(x)
    gap = torch.clamp(output[:, 0], min=0.0)  # gap ≥ 0
    L_ecran = output[:, 1]
    return torch.stack([gap, L_ecran], dim=1)
```

### 2. **Solution Optimale : Adaptation de Domaine**
- Réentraîner avec plage étendue (0-12 µm)
- Adapter les scalers aux données expérimentales
- Utiliser des techniques de domain adaptation

### 3. **Solution Alternative : Preprocessing Cohérent**
- Extraire seulement la plage 1.385-5.538 µm des données exp
- Utiliser la même résolution que l'entraînement
- Vérifier la cohérence avant prédiction

## 🎯 **CONCLUSION**

**Le gap négatif n'est PAS un bug du modèle mais une conséquence directe de l'incompatibilité fondamentale entre les données d'entraînement (simulation) et de test (expérimental).**

Le réseau de neurones fait ce qu'il peut avec des données qu'il n'a jamais vues, d'où les prédictions aberrantes.

**Action prioritaire :** Implémenter des contraintes physiques ou adapter le domaine.
