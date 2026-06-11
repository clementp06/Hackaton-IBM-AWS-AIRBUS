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

g = df.groupby('aircraft_id', sort=False)

# ============================================================================
# 🆕 EXPOSITION CUMULATIVE PONDÉRÉE (Weighted Cumulative Exposure)
# ============================================================================

print("\n" + "="*70)
print("⏳ CRÉATION DES EXPOSITIONS CUMULATIVES PONDÉRÉES")
print("="*70)

# Concept: Les expositions récentes ont plus d'impact que les anciennes
# Facteur de décroissance: 0.95 = 5% de "perte de mémoire" par mois

DECAY_FACTOR = 0.95

print(f"\n📊 Paramètre: decay_factor = {DECAY_FACTOR}")
print(f"   → Après 12 mois: {DECAY_FACTOR**12:.1%} de l'impact initial")
print(f"   → Après 24 mois: {DECAY_FACTOR**24:.1%} de l'impact initial")

# Variables clés pour l'exposition pondérée
key_vars = [
    'metar_temperature_c',
    'metar_relative_humidity',
    'sea_salt_aerosol_003_05_mixing_ratio',
    'sea_salt_aerosol_05_5_mixing_ratio',
    'sea_salt_aerosol_5_20_mixing_ratio',
    'sulphur_dioxide_mass_mixing_ratio',
    'h2o2',
    'formaldehyde',
    'hno3',
]

print(f"\n🎯 Variables avec exposition pondérée: {len(key_vars)}")

# Créer les expositions pondérées
weighted_features = []

for var in key_vars:
    # Pour chaque avion, calculer l'exposition pondérée
    # Plus récent = plus de poids
    def weighted_cumsum(series):
        n = len(series)
        weights = DECAY_FACTOR ** np.arange(n)[::-1]  # Plus récent = poids plus élevé
        return (series.values * weights).cumsum()
    
    weighted_col = f'{var}_weighted_exp'
    df[weighted_col] = g[var].transform(weighted_cumsum)
    weighted_features.append(weighted_col)

print(f"✅ {len(weighted_features)} features d'exposition pondérée créées")

# Créer aussi des indices composites pondérés
print("\n🔬 Création d'indices composites pondérés...")

# 1. Salinité totale pondérée
df['salt_total_weighted'] = (
    df['sea_salt_aerosol_003_05_mixing_ratio_weighted_exp'] +
    df['sea_salt_aerosol_05_5_mixing_ratio_weighted_exp'] +
    df['sea_salt_aerosol_5_20_mixing_ratio_weighted_exp']
)
weighted_features.append('salt_total_weighted')

# 2. Acidité totale pondérée
df['acidity_total_weighted'] = (
    df['h2o2_weighted_exp'] +
    df['formaldehyde_weighted_exp'] +
    df['hno3_weighted_exp'] +
    df['sulphur_dioxide_mass_mixing_ratio_weighted_exp']
)
weighted_features.append('acidity_total_weighted')

# 3. Indice de corrosion pondéré (sel × humidité)
df['corrosion_index_weighted'] = (
    df['salt_total_weighted'] * 
    df['metar_relative_humidity_weighted_exp']
)
weighted_features.append('corrosion_index_weighted')

print(f"✅ 3 indices composites pondérés créés")
print(f"\n📊 Total nouvelles features: {len(weighted_features)}")

# Ajouter aux colonnes de base
base_cols = list(base_cols) + weighted_features

print("="*70)

# ============================================================================
# Feature Engineering (comme avant)
# ============================================================================

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
print(f"   (dont {len(weighted_features)} nouvelles features × 7 transformations)")

# 3. Split par avion : train / validation (early stopping) / test
gss = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
trainval_idx, test_idx = next(gss.split(X, y, groups))
train_idx, val_idx = next(gss.split(
    X.iloc[trainval_idx], y.iloc[trainval_idx], groups.iloc[trainval_idx]))
train_idx, val_idx = trainval_idx[train_idx], trainval_idx[val_idx]

X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

# ============================================================================
# 4. MODÈLE AVEC EXPOSITION PONDÉRÉE
# ============================================================================

print("\n🚀 Entraînement du modèle avec exposition pondérée...\n")

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
print("📊 RÉSULTATS AVEC EXPOSITION PONDÉRÉE")
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
# 🔍 ANALYSE DE L'IMPACT DES FEATURES PONDÉRÉES
# ============================================================================

print("\n" + "="*70)
print("🔬 IMPORTANCE DES FEATURES D'EXPOSITION PONDÉRÉE")
print("="*70)

# Récupérer toutes les importances
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Filtrer les features pondérées
weighted_cols = [col for col in imp.index if 'weighted' in col.lower()]
weighted_imp = imp[weighted_cols].sort_values(ascending=False)

print(f"\n⏳ Top 20 features pondérées (sur {len(weighted_cols)} créées):")
for i, (feat, importance) in enumerate(weighted_imp.head(20).items(), 1):
    print(f"   {i:2d}. {feat:55s} {importance:.4f}")

# Importance totale des features pondérées
total_weighted_importance = weighted_imp.sum()
total_importance = imp.sum()
weighted_percentage = (total_weighted_importance / total_importance) * 100

print(f"\n📊 Contribution totale des features pondérées:")
print(f"   {total_weighted_importance:.4f} / {total_importance:.4f} = {weighted_percentage:.1f}%")

# Top features globales
print(f"\n🏆 Top 25 features globales (toutes catégories):")
for i, (feat, importance) in enumerate(imp.head(25).items(), 1):
    is_weighted = 'weighted' in feat.lower()
    marker = "⏳" if is_weighted else "  "
    print(f"   {marker} {i:2d}. {feat:55s} {importance:.4f}")

print("="*70)

# ============================================================================
# 📈 COMPARAISON AVEC MODÈLE DE BASE
# ============================================================================

print("\n" + "="*70)
print("📈 COMPARAISON: Avec vs Sans Exposition Pondérée")
print("="*70)

# Entraîner modèle sans les features pondérées
print("\n⏳ Entraînement du modèle de référence (sans exposition pondérée)...")

# Retirer les features pondérées
base_features = [col for col in X.columns if 'weighted' not in col.lower()]
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
print(f"{'Modèle':<40} {'Brier Test':<15} {'Amélioration'}")
print("-" * 70)
print(f"{'Sans exposition pondérée':<40} {brier_test_base:.4f}")
print(f"{'Avec exposition pondérée':<40} {brier_test:.4f}         {(brier_test_base - brier_test):.4f} {'✅' if brier_test < brier_test_base else '❌'}")

improvement_pct = ((brier_test_base - brier_test) / brier_test_base) * 100
print(f"\n{'Amélioration relative:':<40} {improvement_pct:+.2f}%")

if brier_test < brier_test_base:
    print("\n✅ L'exposition pondérée AMÉLIORE le modèle!")
    print(f"   Gain: {(brier_test_base - brier_test):.4f} points de Brier")
    print("\n💡 Interprétation:")
    print("   → Les expositions récentes ont plus d'impact sur la corrosion")
    print("   → Le matériau a une 'mémoire' des agressions récentes")
    print("   → Decay factor de 0.95 semble approprié")
else:
    print("\n⚠️  L'exposition pondérée n'améliore pas le modèle.")
    print("   Possibles raisons:")
    print("   - Les cumsum classiques capturent déjà cette information")
    print("   - Le decay factor (0.95) n'est peut-être pas optimal")
    print("   - Tester d'autres valeurs: 0.90 (mémoire courte) ou 0.98 (mémoire longue)")

print("="*70)

# ============================================================================
# 🔬 TEST DE DIFFÉRENTS DECAY FACTORS
# ============================================================================

print("\n" + "="*70)
print("🔬 EXPÉRIMENTATION: Impact du Decay Factor")
print("="*70)

print("\nTest de différents facteurs de décroissance...")
print("(Modèles rapides pour comparaison)")

for decay in [0.90, 0.93, 0.95, 0.97, 0.99]:
    print(f"\n⏳ Test avec decay_factor = {decay}...")
    
    # Recréer les features avec ce decay
    df_test = df.copy()
    g_test = df_test.groupby('aircraft_id', sort=False)
    
    test_features = []
    for var in ['metar_relative_humidity', 'sea_salt_aerosol_003_05_mixing_ratio']:
        def weighted_cumsum_test(series):
            n = len(series)
            weights = decay ** np.arange(n)[::-1]
            return (series.values * weights).cumsum()
        
        col_name = f'{var}_test_{decay}'
        df_test[col_name] = g_test[var].transform(weighted_cumsum_test)
        test_features.append(col_name)
    
    # Ajouter aux features existantes
    X_test_decay = pd.concat([X, df_test[test_features]], axis=1)
    
    # Split
    X_train_d = X_test_decay.iloc[train_idx]
    X_val_d = X_test_decay.iloc[val_idx]
    X_test_d = X_test_decay.iloc[test_idx]
    
    # Modèle rapide
    model_d = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='auc', early_stopping_rounds=30,
    )
    model_d.fit(X_train_d, y_train, eval_set=[(X_val_d, y_val)], verbose=False)
    
    proba_d = model_d.predict_proba(X_test_d)[:, 1]
    brier_d = brier_score_loss(y_test, proba_d)
    
    print(f"   Decay {decay} → Brier Test: {brier_d:.4f}")

print("\n" + "="*70)

# Made with Bob
