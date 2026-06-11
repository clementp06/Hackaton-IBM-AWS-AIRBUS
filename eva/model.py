import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

env    = pd.read_csv('data/environment_training.csv')
labels = pd.read_csv('generer_predictions/prediction_training.csv')

# dans le tableau prediction_training, la colonne id est enfaite 
# aircraft_id + date, on split pour faire le lien avec le tableau env
labels[['aircraft_id', 'date']] = labels['id'].str.split('_', expand=True)

env['date']    = pd.to_datetime(env['month_start_date']).dt.to_period('M').astype(str)
labels['date'] = pd.to_datetime(labels['date']).dt.to_period('M').astype(str)

df = env.merge(labels[['aircraft_id', 'date', 'corrosion_risk']], on=['aircraft_id', 'date'])

df = df.sort_values(['aircraft_id', 'date']).reset_index(drop=True)

meta_cols = ['aircraft_id', 'year_month', 'month_start_date', 'date', 'corrosion_risk']
base_cols = [c for c in df.select_dtypes('number').columns if c not in meta_cols]

# ============================================================================
# 🆕 FEATURE MARINE: Indice de Corrosion Marine
# ============================================================================

# 1. Salinité totale (somme des 3 tailles de particules de sel)
df['salinite_totale'] = (
    df['sea_salt_aerosol_003_05_mixing_ratio'] +
    df['sea_salt_aerosol_05_5_mixing_ratio'] +
    df['sea_salt_aerosol_5_20_mixing_ratio']
)

# 2. Indice de corrosion marine (sel × humidité × temps d'exposition)
# Normalisé par 1000 pour avoir des valeurs raisonnables
df['marine_corrosion_index'] = (
    df['salinite_totale'] *
    df['metar_relative_humidity'] *
    df['total_parking_minutes'] / 1000
)

# Ajouter les nouvelles features aux colonnes de base
base_cols = list(base_cols) + ['salinite_totale', 'marine_corrosion_index']

print(f"✅ Feature marine ajoutée: marine_corrosion_index")
print(f"📊 Nombre total de features de base: {len(base_cols)}")

g = df.groupby('aircraft_id', sort=False)

# Partie environnement
feature_parts = [df[base_cols]]
feature_parts.append(g[base_cols].cumsum().add_suffix('_cumsum'))
feature_parts.append(g[base_cols].cumsum().div(g.cumcount() + 1, axis=0).add_suffix('_cummean'))
feature_parts.append(g[base_cols].cummax().add_suffix('_cummax'))

roll3  = g[base_cols].rolling(3,  min_periods=1).mean().reset_index(level=0, drop=True)
roll12 = g[base_cols].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
feature_parts.append(roll3.add_suffix('_roll3'))
feature_parts.append(roll12.add_suffix('_roll12'))
feature_parts.append(g[base_cols].shift(1).add_suffix('_lag1'))


age = g.cumcount().rename('age_months')

X = pd.concat(feature_parts + [age], axis=1)
y = df['corrosion_risk']
groups = df['aircraft_id']

# 3. Split par avion : train / validation (early stopping) / test
gss = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
trainval_idx, test_idx = next(gss.split(X, y, groups))
train_idx, val_idx = next(gss.split(
    X.iloc[trainval_idx], y.iloc[trainval_idx], groups.iloc[trainval_idx]))
train_idx, val_idx = trainval_idx[train_idx], trainval_idx[val_idx]

X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

# 4. Modèle : arbres peu profonds + régularisation + early stopping
model = xgb.XGBClassifier(
    n_estimators=2000, max_depth=6, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=5, gamma=1.0, reg_lambda=5.0,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric='auc', early_stopping_rounds=50,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 5. Évaluation
proba_train = model.predict_proba(X_train)[:, 1]
proba_test = model.predict_proba(X_test)[:, 1]

auc_train = roc_auc_score(y_train, proba_train)
auc_test = roc_auc_score(y_test, proba_test)

brier_train = brier_score_loss(y_train, proba_train)
brier_test = brier_score_loss(y_test, proba_test)

print("Nb features    :", X.shape[1])
print("Arbres retenus :", model.best_iteration + 1)
print("\n=== AUC (higher is better) ===")
print("AUC Train      :", round(auc_train, 4))
print("AUC Test       :", round(auc_test, 4))
print("\n=== Brier Score (lower is better) ===")
print("Brier Train    :", round(brier_train, 4))
print("Brier Test     :", round(brier_test, 4))
print("Baseline (0.5) : 0.2500")

# Top features les plus utiles
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 15 features :")
print(imp.head(15).round(4))
