import pandas as pd
import xgboost as xgb
import numpy as np
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
# 🆕 NOUVELLES FEATURES: INDICE DE CORROSIVITÉ MARINE
# ============================================================================

print("\n" + "="*70)
print("🌊 CRÉATION DES INDICES DE CORROSIVITÉ MARINE")
print("="*70)

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

# 3. Sel fin (plus dangereux car pénètre mieux)
df['salt_fine_ratio'] = (
    df['sea_salt_aerosol_003_05_mixing_ratio'] / 
    (df['sea_salt_aerosol_5_20_mixing_ratio'] + 1e-12)
)

# 4. Interaction humidité × sel (accélérateur de corrosion)
df['humidity_salt_interaction'] = (
    df['metar_relative_humidity'] * df['salinite_totale']
)

# 5. Condensation risk (température × humidité)
df['condensation_risk'] = (
    df['metar_temperature_c'] * df['metar_relative_humidity']
)

# 6. Temps d'exposition en conditions critiques (humidité > 80%)
df['high_humidity_exposure'] = (
    df['total_parking_minutes'] * (df['metar_relative_humidity'] > 80).astype(int)
)

print(f"✅ 6 nouvelles features créées:")
print(f"   - salinite_totale")
print(f"   - marine_corrosion_index (🎯 PRINCIPALE)")
print(f"   - salt_fine_ratio")
print(f"   - humidity_salt_interaction")
print(f"   - condensation_risk")
print(f"   - high_humidity_exposure")

# Ajouter les nouvelles features aux colonnes de base
new_features = [
    'marine_corrosion_index'
]
base_cols = base_cols + new_features

print("="*70)

# ============================================================================
# Feature Engineering (comme avant)
# ============================================================================

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

print(f"\n📊 Nombre total de features: {X.shape[1]}")
print(f"   (dont {len(new_features)} nouvelles features × 7 transformations)")

# 3. Split par avion : train / validation (early stopping) / test
gss = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
trainval_idx, test_idx = next(gss.split(X, y, groups))
train_idx, val_idx = next(gss.split(
    X.iloc[trainval_idx], y.iloc[trainval_idx], groups.iloc[trainval_idx]))
train_idx, val_idx = trainval_idx[train_idx], trainval_idx[val_idx]

X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

# 4. Modèle : arbres peu profonds + régularisation + early stopping
print("\n🚀 Entraînement du modèle avec nouvelles features...\n")

model = xgb.XGBClassifier(
    n_estimators=2000, max_depth=6, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8,
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

print("="*70)
print("📊 RÉSULTATS AVEC INDICES DE CORROSIVITÉ MARINE")
print("="*70)
print(f"Nb features    : {X.shape[1]}")
print(f"Arbres retenus : {model.best_iteration + 1}")
print("\n=== AUC (higher is better) ===")
print(f"AUC Train      : {round(auc_train, 4)}")
print(f"AUC Test       : {round(auc_test, 4)}")
print("\n=== Brier Score (lower is better) ===")
print(f"Brier Train    : {round(brier_train, 4)}")
print(f"Brier Test     : {round(brier_test, 4)} 🎯")
print(f"Baseline (0.5) : 0.2500")
print("="*70)

# ============================================================================
# 🔍 ANALYSE DE L'IMPACT DES NOUVELLES FEATURES
# ============================================================================

print("\n" + "="*70)
print("🔬 IMPORTANCE DES NOUVELLES FEATURES")
print("="*70)

# Récupérer toutes les importances
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Filtrer les features liées aux nouvelles créations
marine_features = []
for new_feat in new_features:
    # Trouver toutes les variantes (cumsum, cummean, etc.)
    related = [col for col in imp.index if new_feat in col]
    marine_features.extend(related)

marine_imp = imp[marine_features].sort_values(ascending=False)

print(f"\n🌊 Top 15 features marines (sur {len(marine_features)} créées):")
for i, (feat, importance) in enumerate(marine_imp.head(15).items(), 1):
    print(f"   {i:2d}. {feat:50s} {importance:.4f}")

# Importance totale des features marines
total_marine_importance = marine_imp.sum()
total_importance = imp.sum()
marine_percentage = (total_marine_importance / total_importance) * 100

print(f"\n📊 Contribution totale des features marines:")
print(f"   {total_marine_importance:.4f} / {total_importance:.4f} = {marine_percentage:.1f}%")

# Top features globales
print(f"\n🏆 Top 20 features globales (toutes catégories):")
for i, (feat, importance) in enumerate(imp.head(20).items(), 1):
    is_marine = any(nf in feat for nf in new_features)
    marker = "🌊" if is_marine else "  "
    print(f"   {marker} {i:2d}. {feat:50s} {importance:.4f}")

print("="*70)

# ============================================================================
# 📈 COMPARAISON AVEC MODÈLE DE BASE
# ============================================================================

print("\n" + "="*70)
print("📈 COMPARAISON: Avec vs Sans Features Marines")
print("="*70)

# Entraîner modèle sans les nouvelles features
print("\n⏳ Entraînement du modèle de référence (sans features marines)...")

# Retirer les features marines
base_features = [col for col in X.columns if not any(nf in col for nf in new_features)]
X_train_base = X_train[base_features]
X_val_base = X_val[base_features]
X_test_base = X_test[base_features]

model_base = xgb.XGBClassifier(
    n_estimators=2000, max_depth=6, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, gamma=1.0, reg_lambda=5.0,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric='auc', early_stopping_rounds=50,
)
model_base.fit(X_train_base, y_train, eval_set=[(X_val_base, y_val)], verbose=False)

proba_test_base = model_base.predict_proba(X_test_base)[:, 1]
brier_test_base = brier_score_loss(y_test, proba_test_base)

print("\n📊 RÉSULTATS COMPARATIFS:")
print(f"{'Modèle':<30} {'Brier Test':<15} {'Amélioration'}")
print("-" * 70)
print(f"{'Sans features marines':<30} {brier_test_base:.4f}")
print(f"{'Avec features marines':<30} {brier_test:.4f}         {(brier_test_base - brier_test):.4f} {'✅' if brier_test < brier_test_base else '❌'}")

improvement_pct = ((brier_test_base - brier_test) / brier_test_base) * 100
print(f"\n{'Amélioration relative:':<30} {improvement_pct:+.2f}%")

if brier_test < brier_test_base:
    print("\n✅ Les features marines AMÉLIORENT le modèle!")
else:
    print("\n⚠️  Les features marines n'améliorent pas le modèle.")
    print("   Possibles raisons:")
    print("   - Information déjà capturée par features existantes")
    print("   - Besoin d'ajuster les formules")
    print("   - Tester d'autres combinaisons")

print("="*70)

# Made with Bob
