
# Rapport de Prétraitement des Données
## Neural_Network_Gap_Lecran_Prediction

### Paramètres de prétraitement
- **Plage radiale** : [1.384585, 5.538338] µm
- **Espacement** : 0.006923 µm
- **Points par profil** : 602

### Données d'entraînement
- **Nombre de fichiers** : 22542
- **Troncature appliquée** : indices 200 à 800 (sur 1000 points originaux)
- **Format** : fichiers .mat avec variables 'x' et 'ratio'

### Données de test
- **Nombre de profils** : 50
- **Source** : profile_exp_PS_3um_z_positive.mat
- **Interpolation** : 50 profils interpolés à 602 points

### Exemples analysés
- **Gap min** : 0.215000 µm
- **Gap max** : 0.650000 µm
- **L_écran min** : 8.075 µm
- **L_écran max** : 11.300 µm

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
