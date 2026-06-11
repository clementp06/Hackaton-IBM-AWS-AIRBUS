# 🚀 Stratégie d'optimisation rapide (30 minutes)

## 📊 Problème identifié

**MSE Leaderboard : 0.24** vs **MSE Test local : 0.108**

Cet écart suggère :
1. Distribution différente entre train et test
2. Possible overfitting sur le validation set local
3. Feature selection trop agressive (42 features peut-être insuffisant)

## 💡 Solution implémentée

### 1. Utiliser TOUTES les features (127)
**Rationale** : La feature selection à 42 features peut avoir éliminé des informations importantes pour le test set réel.

### 2. Ensemble de 3 modèles avec stratégies différentes

#### Modèle 1 : LightGBM Conservateur
```python
learning_rate: 0.01
num_leaves: 31
max_depth: 6
min_child_samples: 100
reg_alpha: 0.5
reg_lambda: 1.0
```
**Objectif** : Minimiser l'overfitting, meilleure généralisation

#### Modèle 2 : LightGBM Agressif
```python
learning_rate: 0.02
num_leaves: 63
max_depth: 8
min_child_samples: 50
reg_alpha: 0.1
reg_lambda: 0.5
```
**Objectif** : Capturer des patterns complexes

#### Modèle 3 : XGBoost
```python
learning_rate: 0.01
max_depth: 7
min_child_weight: 80
reg_alpha: 0.3
reg_lambda: 0.7
```
**Objectif** : Diversité algorithmique

### 3. Optimisation des poids
Recherche des meilleurs poids pour minimiser le MSE sur validation set.

### 4. Clipping des prédictions
```python
predictions = np.clip(predictions, 0, 1)
```
Garantit que les probabilités sont dans [0, 1].

## 🎯 Améliorations attendues

### Par rapport à la première soumission (MSE 0.24)

1. **Plus de features** : 127 vs 42 (+200%)
   - Gain attendu : -0.03 à -0.05 MSE

2. **Ensemble de 3 modèles** : Diversité
   - Gain attendu : -0.02 à -0.04 MSE

3. **Régularisation plus forte** : Moins d'overfitting
   - Gain attendu : -0.01 à -0.02 MSE

**MSE cible** : **0.15-0.18** (amélioration de 25-40%)

## 📈 Comparaison des approches

| Approche | Features | Modèles | MSE attendu |
|----------|----------|---------|-------------|
| **V1 (soumise)** | 42 | LightGBM seul | 0.24 |
| **V2 (optimisée)** | 127 | Ensemble (3) | 0.15-0.18 |

## ⏱️ Temps d'exécution

- Feature engineering : ~8 min
- Training modèle 1 : ~5 min
- Training modèle 2 : ~5 min
- Training modèle 3 : ~5 min
- Optimisation poids : ~1 min
- Prédiction : ~2 min

**Total** : ~25-28 minutes

## 🔍 Analyse post-soumission

### Si le MSE s'améliore significativement
✅ La stratégie "plus de features + ensemble" était correcte
✅ La feature selection était trop agressive

### Si le MSE reste élevé
Causes possibles :
1. Distribution très différente train/test
2. Besoin de features spécifiques au test set
3. Problème de calibration des probabilités

### Prochaines étapes si temps restant
1. **Calibration** : Isotonic regression ou Platt scaling
2. **Stacking** : Meta-learner sur les prédictions
3. **Features temporelles** : Tendances spécifiques par année

## 📝 Fichiers générés

- `submission_optimized.csv` : Nouvelle soumission
- Statistiques affichées dans le terminal

## 🎓 Leçons apprises

1. **Feature selection** : Peut être trop agressive pour des distributions différentes
2. **Ensemble** : Toujours meilleur qu'un modèle seul
3. **Régularisation** : Critique pour la généralisation
4. **Validation** : Le validation set local ne garantit pas la performance sur le leaderboard

## 🚀 Commande

```bash
cd Hackaton-IBM-AWS-AIRBUS/LightGBM
uv run python optimize_fast.py
```

**En cours d'exécution...** ⏳