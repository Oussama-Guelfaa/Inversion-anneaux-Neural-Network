
# 🧪 Rapport de Test - Tous les Profils Expérimentaux

## 📊 Résumé Exécutif
- **Modèle testé**: ULTRA_DEEP_NETWORK_ALL_22540_PROFILES
- **Profils testés**: 49
- **Succès**: 49 (100.0%)
- **Échecs**: 0 (0.0%)

## 🎯 Prédictions Gap
- **Minimum**: -0.245424 µm
- **Maximum**: -0.085451 µm
- **Moyenne**: -0.188709 µm
- **Médiane**: -0.190371 µm
- **Écart-type**: 0.034040 µm
- **Gaps négatifs**: 49/49 (100.0%)

## 🎯 Prédictions L_écran
- **Minimum**: 9.046 µm
- **Maximum**: 9.577 µm
- **Moyenne**: 9.424 µm
- **Médiane**: 9.477 µm
- **Écart-type**: 0.138 µm

## 📈 Analyse de Cohérence
- **Gap CV**: 18.0%
- **L_écran CV**: 1.5%

## ⚠️ Observations
1. **Gaps négatifs**: 49 profils avec gaps négatifs (physiquement impossibles)
2. **Cohérence L_écran**: Bonne
3. **Adaptation de domaine**: Nécessaire entre données simulées et expérimentales

## 🎯 Recommandations
1. Analyser les profils avec gaps négatifs
2. Appliquer une adaptation de domaine
3. Comparer avec le modèle simple
4. Investiguer les différences simulation vs expérimental
