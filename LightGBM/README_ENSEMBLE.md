# Ensemble Training Pipeline

## Architecture

Cette pipeline implémente un système d'ensemble avancé pour la prédiction de corrosion d'avions.

### Structure des données

**Split stratifié par avion :**
- **Train set** : 60% des avions
- **Validation set** : 20% des avions  
- **Test set** : 20% des avions

Le split respecte les groupes (aircraft_id) pour éviter le data leakage.

### Modèles

L'ensemble combine 3 algorithmes de gradient boosting :

1. **LightGBM** (poids par défaut : 0.4)
   - Rapide et efficace
   - Excellente gestion des features catégorielles
   - Régularisation L1/L2

2. **XGBoost** (poids par défaut : 0.3)
   - Robuste et stable
   - Bonne gestion du déséquilibre de classes
   - Régularisation avancée

3. **CatBoost** (poids par défaut : 0.3)
   - Gestion native des features catégorielles
   - Résistant à l'overfitting
   - Ordered boosting

### Feature Engineering

**Pipeline multi-niveaux :**
- **Bronze** : Features de base (température, humidité, etc.)
- **Silver** : Agrégations temporelles (moyennes, écarts-types)
- **Gold** : Features complexes (interactions, ratios)
- **Platinum** : Features avancées (streaks, patterns)
- **Scientific** : Features basées sur la science de la corrosion (ISO 9223, TOW, etc.)

**Feature Selection intelligente :**
- Sélection par gain (threshold configurable)
- Élimination des features corrélées (>0.95)
- Réduction de 127 à ~42 features
- Conservation de 98.9% du gain

## Utilisation

### 1. Training de base avec ensemble

```bash
cd Hackaton-IBM-AWS-AIRBUS/LightGBM
uv run python train_ensemble.py --use-feature-selection
```

### 2. Training avec optimisation des hyperparamètres

```bash
uv run python train_ensemble.py \
  --use-feature-selection \
  --optimize-hyperparams \
  --n-trials 100 \
  --save-params best_params.json
```

Cette commande :
- Optimise les hyperparamètres de chaque modèle (100 trials par modèle)
- Sauvegarde les meilleurs paramètres
- Peut prendre 2-3 heures

### 3. Training avec paramètres optimisés

```bash
uv run python train_ensemble.py \
  --use-feature-selection \
  --load-params best_params.json \
  --optimize-weights
```

Cette commande :
- Charge les hyperparamètres optimisés
- Optimise les poids de l'ensemble
- Plus rapide (~30 minutes)

### 4. Configuration personnalisée

```bash
uv run python train_ensemble.py \
  --use-feature-selection \
  --top-n-features 60 \
  --min-gain 500 \
  --corr-threshold 0.90 \
  --weights 0.5 0.3 0.2 \
  --train-size 0.7 \
  --valid-size 0.15 \
  --test-size 0.15
```

## Paramètres

### Feature Selection
- `--use-feature-selection` : Activer la sélection de features
- `--top-n-features` : Nombre max de features (default: 50)
- `--min-gain` : Gain minimum requis (default: 800)
- `--corr-threshold` : Seuil de corrélation (default: 0.95)

### Hyperparameter Optimization
- `--optimize-hyperparams` : Optimiser avec Optuna
- `--n-trials` : Nombre de trials (default: 100)
- `--load-params` : Charger paramètres depuis JSON
- `--save-params` : Sauvegarder paramètres optimisés

### Ensemble
- `--optimize-weights` : Optimiser les poids de l'ensemble
- `--weights` : Poids manuels [LGB, XGB, CB] (default: [0.4, 0.3, 0.3])

### Data Split
- `--train-size` : Proportion train (default: 0.6)
- `--valid-size` : Proportion validation (default: 0.2)
- `--test-size` : Proportion test (default: 0.2)

## Résultats attendus

### Performance baseline (sans optimisation)
- **Ensemble AUC** : ~0.93
- **LightGBM AUC** : ~0.928
- **XGBoost AUC** : ~0.925
- **CatBoost AUC** : ~0.923

### Performance optimisée (avec Optuna)
- **Ensemble AUC** : ~0.94-0.95
- Gain de +1-2% sur chaque modèle
- Meilleure généralisation

### Overfitting
- **Train-Valid** : <0.05 (bien contrôlé)
- **Valid-Test** : <0.01 (excellente généralisation)

## Fichiers générés

- `ensemble_results.json` : Résultats détaillés de l'ensemble
- `best_params.json` : Meilleurs hyperparamètres (si optimisés)
- `lightgbm_history_feature_importance.csv` : Importance des features

## Modules

### `data_split.py`
Gestion du split 60/20/20 avec stratification par avion.

### `ensemble.py`
Classe `EnsembleModel` pour combiner les 3 modèles.

### `hyperopt.py`
Optimisation des hyperparamètres avec Optuna.

### `train_ensemble.py`
Script principal d'entraînement.

## Workflow recommandé

1. **Baseline** : Training rapide sans optimisation
   ```bash
   uv run python train_ensemble.py --use-feature-selection
   ```

2. **Optimisation** : Recherche des meilleurs hyperparamètres (long)
   ```bash
   uv run python train_ensemble.py \
     --use-feature-selection \
     --optimize-hyperparams \
     --n-trials 100 \
     --save-params best_params.json
   ```

3. **Fine-tuning** : Optimisation des poids avec les meilleurs paramètres
   ```bash
   uv run python train_ensemble.py \
     --use-feature-selection \
     --load-params best_params.json \
     --optimize-weights
   ```

4. **Production** : Training final avec configuration optimale
   ```bash
   uv run python train_ensemble.py \
     --use-feature-selection \
     --load-params best_params.json \
     --weights <optimized_weights>
   ```

## Temps d'exécution estimés

- **Baseline** : ~15-20 minutes
- **Avec optimisation hyperparamètres** : ~2-3 heures
- **Avec optimisation poids** : ~30 minutes
- **Optimisation complète** : ~3-4 heures

## Notes

- Le split 60/20/20 garantit une évaluation robuste
- L'ensemble améliore la stabilité et réduit la variance
- L'optimisation des hyperparamètres peut donner +1-2% AUC
- L'optimisation des poids donne généralement +0.1-0.3% AUC
- Le test set ne doit JAMAIS être utilisé pour l'optimisation