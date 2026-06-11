# Pipeline LightGBM

Pipeline de baseline LightGBM pour prédire `corrosion_risk` à partir de
`environment_training.csv` et `prediction_training.csv`.

## Lancer un smoke test

```powershell
python .\LightGBM\train.py --cv-splits 2 --n-estimators 30 --early-stopping-rounds 5 --no-save
```

## Lancer l'entraînement complet

```powershell
python .\LightGBM\train.py
```

## Structure

- `config.py`: chemins et hyperparamètres par défaut.
- `data.py`: chargement des CSV et jointure labels/features.
- `features.py`: features historiques par avion.
- `training.py`: split par avion, cross-validation, entraînement LightGBM.
- `train.py`: point d'entrée CLI.

Les features utilisent tout l'historique disponible pour un avion jusqu'au mois
considéré: valeurs courantes, agrégats cumulés, fenêtres glissantes, lags et
deltas.
