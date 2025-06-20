# 🔍 GUIDE DÉTAILLÉ DU WORKFLOW DE PRÉDICTION

**Auteur:** Oussama GUELFAA  
**Date:** 19-06-2025  
**Test:** 2392 échantillons

---

## 📋 **WORKFLOW COMPLET ÉTAPE PAR ÉTAPE**

### **🔧 1. CHARGEMENT ET PRÉPARATION DES DONNÉES**

#### **A. Chargement du Modèle depuis le fichier .pth**

```python
# Ligne 69: Chargement du checkpoint
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

# Ligne 71-73: Reconstruction du modèle
model = DualParameterPredictor(input_size=600)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

**📊 Contenu du fichier .pth :**
- `model_state_dict`: Poids du réseau neuronal (1,318,882 paramètres)
- `config`: Configuration d'entraînement
- `training_info`: Informations d'entraînement (300 epochs)
- `test_metrics`: Métriques de performance (Gap R²: 0.9953, L_ecran R²: 0.9891)

#### **B. Division des Données (Splits 70/16/14)**

```python
# Ligne 82-89: Configuration des splits
config = {
    'data_processing': {
        'data_splits': {'train': 0.70, 'validation': 0.16, 'test': 0.14}
    }
}
```

**📈 Résultat de la division :**
- **Dataset total :** 17,080 échantillons (augmentés)
- **Train :** 11,955 échantillons (70%)
- **Validation :** 2,733 échantillons (16%)
- **Test :** 2,392 échantillons (14%) ← **Notre cible ~2400**

#### **C. Rôle du DualDataLoader et Normalisation**

```python
# Ligne 78: Initialisation
data_loader = DualDataLoader()

# Ligne 91: Pipeline complet
pipeline_result = data_loader.get_complete_pipeline(config)

# Ligne 94-95: Récupération des données NON normalisées
X_test = pipeline_result['raw_data'][2]  # Profils [2392, 600]
y_test = pipeline_result['raw_data'][5]  # Labels [2392, 2]
```

**🔧 Fonctions du DualDataLoader :**
1. **Chargement :** Cache `augmented_dataset_advanced.npz`
2. **Séparation stricte :** Aucun chevauchement entre train/val/test
3. **Normalisation :** StandardScaler pour profils et scaling séparé pour gap/L_ecran
4. **Scalers sauvegardés :** `input_scaler`, `gap_scaler`, `L_ecran_scaler`

---

### **⚙️ 2. PROCESSUS DE PRÉDICTION**

#### **A. Traitement d'un Échantillon [600 points]**

```python
# Ligne 129-130: Pour chaque échantillon
profile = X_test[i]  # Shape: [600] - Profil d'intensité radial
gap_pred, L_ecran_pred = predict_sample(model, profile, data_loader)
```

**📊 Exemple concret d'un profil :**
- **Input :** `profile = [0.234, 0.456, 0.789, ..., 0.123]` (600 valeurs)
- **Représente :** Intensité holographique en fonction du rayon (0-6 µm)

#### **B. Étapes Normalisation → Prédiction → Dénormalisation**

```python
# Ligne 40: ÉTAPE 1 - Normalisation
profile_scaled = data_loader.input_scaler.transform(profile.reshape(1, -1))
# Exemple: [0.234, 0.456, ...] → [-1.23, 0.45, ...] (standardisé)

# Ligne 43-45: ÉTAPE 2 - Prédiction
with torch.no_grad():
    input_tensor = torch.FloatTensor(profile_scaled)  # [1, 600]
    prediction_scaled = model(input_tensor).numpy()  # [1, 2]

# Ligne 48: ÉTAPE 3 - Dénormalisation
prediction_original = data_loader.inverse_transform_predictions(prediction_scaled)
```

**🔢 Exemple de transformation :**
```
Input brut:     [0.234, 0.456, 0.789, ..., 0.123]
↓ Normalisation
Input normalisé: [-1.23, 0.45, 1.67, ..., -0.89]
↓ Réseau neuronal
Output normalisé: [-0.567, 1.234]
↓ Dénormalisation
Output final:    [0.0847, 10.45] → Gap: 0.0847µm, L_ecran: 10.45µm
```

#### **C. Extraction des Prédictions Gap et L_ecran**

```python
# Ligne 50-51: Extraction des valeurs
gap_pred = prediction_original[0, 0]      # Premier élément: Gap en µm
L_ecran_pred = prediction_original[0, 1]  # Deuxième élément: L_ecran en µm
```

---

### **📊 3. CALCUL DES MÉTRIQUES**

#### **A. Métriques R², MAE, RMSE**

```python
# Ligne 146-154: Calcul des métriques
gap_r2 = r2_score(true_gap, predictions_gap)
L_ecran_r2 = r2_score(true_L_ecran, predictions_L_ecran)
combined_r2 = (gap_r2 + L_ecran_r2) / 2

gap_mae = mean_absolute_error(true_gap, predictions_gap)
gap_rmse = np.sqrt(mean_squared_error(true_gap, predictions_gap))
```

**📈 Formules utilisées :**
- **R² :** `1 - (SS_res / SS_tot)` où SS_res = Σ(y_true - y_pred)², SS_tot = Σ(y_true - y_mean)²
- **MAE :** `Σ|y_true - y_pred| / n`
- **RMSE :** `√(Σ(y_true - y_pred)² / n)`

**🎯 Résultats obtenus :**
- **Gap R² :** 0.9944 (99.44% de variance expliquée)
- **Gap MAE :** 0.0035 µm (erreur moyenne absolue)
- **Gap RMSE :** 0.0043 µm (erreur quadratique moyenne)

#### **B. Analyse de Tolérance**

```python
# Ligne 157-164: Analyse de tolérance
gap_tolerance = 0.01  # µm
L_ecran_tolerance = 0.1  # µm

gap_within_tolerance = np.sum(np.abs(predictions_gap - true_gap) <= gap_tolerance)
gap_accuracy = gap_within_tolerance / len(X_test)
```

**🎯 Signification des tolérances :**
- **Gap ±0.01µm :** Précision requise pour applications industrielles
- **L_ecran ±0.1µm :** Tolérance acceptable pour distance écran-objet

**📊 Résultats de tolérance :**
- **Gap :** 2368/2392 (99.0%) dans ±0.01µm
- **L_ecran :** 2245/2392 (93.9%) dans ±0.1µm

#### **C. Statistiques d'Erreurs (Max, Min, Médiane)**

```python
# Ligne 185-190: Statistiques d'erreurs
gap_errors = np.abs(predictions_gap - true_gap)
L_ecran_errors = np.abs(predictions_L_ecran - true_L_ecran)

print(f"Gap - Max: {gap_errors.max():.4f} µm, Min: {gap_errors.min():.4f} µm")
print(f"Médiane: {np.median(gap_errors):.4f} µm")
```

**📈 Signification des statistiques :**
- **Max :** Pire erreur observée (Gap: 0.0110µm, L_ecran: 0.1644µm)
- **Min :** Meilleure prédiction (0.0000µm pour les deux)
- **Médiane :** Erreur typique (Gap: 0.0031µm, L_ecran: 0.0245µm)

---

## 🆕 **COMMENT TESTER SUR DE NOUVELLES DONNÉES**

### **📋 Étapes pour Nouvelles Données :**

#### **1. Préparation des Données**
```python
# Vos nouvelles données doivent être au format:
# X_new: [n_samples, 600] - Profils d'intensité radiaux
# y_new: [n_samples, 2] - Labels [gap, L_ecran] (optionnel pour validation)

# Exemple:
X_new = np.load('mes_nouveaux_profils.npy')  # Shape: [100, 600]
y_new = np.load('mes_nouveaux_labels.npy')   # Shape: [100, 2] (optionnel)
```

#### **2. Chargement du Modèle et DataLoader**
```python
# Charger le modèle entraîné
model, _, _, data_loader = load_model_and_data()
```

#### **3. Prédictions sur Nouvelles Données**
```python
predictions_gap = []
predictions_L_ecran = []

for i in range(len(X_new)):
    profile = X_new[i]
    gap_pred, L_ecran_pred = predict_sample(model, profile, data_loader)
    predictions_gap.append(gap_pred)
    predictions_L_ecran.append(L_ecran_pred)

# Conversion en arrays
predictions_gap = np.array(predictions_gap)
predictions_L_ecran = np.array(predictions_L_ecran)
```

#### **4. Évaluation (si labels disponibles)**
```python
if y_new is not None:
    gap_r2 = r2_score(y_new[:, 0], predictions_gap)
    L_ecran_r2 = r2_score(y_new[:, 1], predictions_L_ecran)
    print(f"Gap R²: {gap_r2:.4f}, L_ecran R²: {L_ecran_r2:.4f}")
```

#### **5. Sauvegarde des Résultats**
```python
results = {
    'predictions': {
        'gap': predictions_gap.tolist(),
        'L_ecran': predictions_L_ecran.tolist()
    }
}

with open('mes_predictions.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## ✅ **POINTS CLÉS À RETENIR**

1. **Normalisation Cruciale :** Toujours utiliser les mêmes scalers d'entraînement
2. **Format des Données :** Profils de 600 points (intensité radiale 0-6µm)
3. **Mode Évaluation :** `model.eval()` et `torch.no_grad()` pour l'inférence
4. **Dénormalisation :** Obligatoire pour obtenir les valeurs physiques réelles
5. **Tolérances :** ±0.01µm pour gap, ±0.1µm pour L_ecran selon spécifications

Le workflow est **robuste** et **reproductible** avec des performances **exceptionnelles** validées sur 2392 échantillons ! 🎉
