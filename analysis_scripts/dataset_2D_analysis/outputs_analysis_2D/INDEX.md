# Index des Outputs - Analyse Dataset 2D

**Généré le:** 13/06/2025 à 13:25
**Auteur:** Oussama GUELFAA

## 📊 Visualisations (visualizations/)

### Distributions et Densités
- **parameter_distributions.png** - Histogrammes et heatmap des paramètres (gap, L_ecran)
- **parameter_density_2D.png** - Densité hexagonale dans l'espace des paramètres

### Couverture du Dataset
- **coverage_matrix.png** - Matrice de couverture pour identifier les zones manquantes

### Échantillons d'Anneaux
- **ring_samples_grid.png** - Grille de 36 échantillons représentatifs d'anneaux
- **rings_by_L_ecran.png** - Évolution des profils d'anneaux par L_ecran fixe

## 📈 Statistiques (statistics/)

- **dataset_statistics.csv** - Statistiques générales du dataset
- **detailed_statistics.csv** - Statistiques détaillées par variable

## 📄 Rapports (reports/)

- **analysis_report.txt** - Rapport complet de l'analyse avec recommandations

## 🎯 Résultats Clés

- **2440 fichiers** analysés (100% de complétude)
- **40 gaps** de 0.005 à 0.2 µm
- **61 L_ecran** de 10.0 à 11.5 µm
- **30.9 MB** de données au total
- **Qualité excellente** (ratios cohérents)

## 🚀 Utilisation

### Pour l'Entraînement de Réseaux
- Train: 1708 échantillons (70%)
- Validation: 366 échantillons (15%) 
- Test: 366 échantillons (15%)

### Préprocessing Recommandé
1. Tronquer les profils à 600 points
2. Normalisation StandardScaler
3. Validation croisée stratifiée

## 📞 Contact

Pour questions sur cette analyse: Oussama GUELFAA
