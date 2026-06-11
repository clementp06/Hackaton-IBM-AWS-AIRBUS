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
# 🆕 NOUVELLES FEATURES: INDICES D'ACIDITÉ
# ============================================================================

print("\n" + "="*70)
print("🧪 CRÉATION DES INDICES D'ACIDITÉ")
print("="*70)

# 1. Acidité totale (somme des composés acides)
df['acidite_totale'] = (
    df['h2o2'] +                                    # Peroxyde d'hydrogène
    df['formaldehyde'] +                            # Formaldéhyde
    df['hno3'] +                                    # Acide nitrique
    df['sulphur_dioxide_mass_mixing_ratio']         # Dioxyde de soufre (SO2)
)

# 2. Pluies acides (précipitation × acidité)
df['acid_rain_risk'] = (
    df['metar_hour_precipitation'] * df['acidite_totale']
)

# 3. Acidité en milieu humide (humidité × acidité)
# L'humidité active les acides
df['humid_acidity'] = (
    df['metar_relative_humidity'] * df['acidite_totale']
)

# 4. Ratio acides forts / acides faibles
# HNO3 et SO2 sont plus corrosifs que H2O2 et formaldéhyde
df['strong_acid_ratio'] = (
    (df['hno3'] + df['sulphur_dioxide_mass_mixing_ratio']) /
    (df['h2o2'] + df['formaldehyde'] + 1e-12)
)

# 5. Exposition acide cumulée (temps × acidité)
df['acid_exposure'] = (
    df['total_parking_minutes'] * df['acidite_totale'] / 1000
)

# 6. Acidité + Sel (combinaison très corrosive)
df['acid_salt_synergy'] = (
    df['acidite_totale'] * 
    (df['sea_salt_aerosol_003_05_mixing_ratio'] + 
     df['sea_salt_aerosol_05_5_mixing_ratio'] + 
     df['sea_salt_aerosol_5_20_mixing_ratio'])
)

print(f"✅ 6 nouvelles features créées:")
print(f"   - acidite_totale (🎯 PRINCIPALE)")
print(f"   - acid_rain_risk")
print(f"   - humid_acidity")
print(f"   - strong_acid_ratio")
print(f"   - acid_exposure")
print(f"   - acid_salt_synergy")

# Statistiques descriptives
print(f"\n📊 Statistiques de l'acidité totale:")
print(f"   Min:    {df['acidite_totale'].min():.2e}")
print(f"   Médiane: {df['acidite_totale'].median():.2e}")
print(f"   Max:    {df['acidite_totale'].max():.2e}")
print(f"   Std:    {df['acidite_totale'].std():.2e}")

# Ajouter les nouvelles features aux colonnes de base
new_features = [
    'acid_rain_risk', 'humid_acidity',
    'strong_acid_ratio', 'acid_exposure', 'acid_salt_synergy'
]
base_cols = list(base_cols) + new_features

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

# ============================================================================
# 4. MODÈLE AVEC FEATURES ACIDITÉ
# ============================================================================

print("\n🚀 Entraînement du modèle avec indices d'acidité...\n")

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
print("📊 RÉSULTATS AVEC INDICES D'ACIDITÉ")
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
# 🔍 ANALYSE DE L'IMPACT DES FEATURES ACIDITÉ
# ============================================================================

print("\n" + "="*70)
print("🔬 IMPORTANCE DES FEATURES ACIDITÉ")
print("="*70)

# Récupérer toutes les importances
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Filtrer les features liées à l'acidité
acid_features = [col for col in imp.index if any(nf in col for nf in new_features)]
acid_imp = imp[acid_features].sort_values(ascending=False)

print(f"\n🧪 Top 15 features acidité (sur {len(acid_features)} créées):")
for i, (feat, importance) in enumerate(acid_imp.head(15).items(), 1):
    print(f"   {i:2d}. {feat:50s} {importance:.4f}")

# Importance totale des features acidité
total_acid_importance = acid_imp.sum()
total_importance = imp.sum()
acid_percentage = (total_acid_importance / total_importance) * 100

print(f"\n📊 Contribution totale des features acidité:")
print(f"   {total_acid_importance:.4f} / {total_importance:.4f} = {acid_percentage:.1f}%")

# Top features globales
print(f"\n🏆 Top 20 features globales (toutes catégories):")
for i, (feat, importance) in enumerate(imp.head(20).items(), 1):
    is_acid = any(nf in feat for nf in new_features)
    marker = "🧪" if is_acid else "  "
    print(f"   {marker} {i:2d}. {feat:50s} {importance:.4f}")

print("="*70)

# ============================================================================
# 📈 COMPARAISON AVEC MODÈLE DE BASE
# ============================================================================

print("\n" + "="*70)
print("📈 COMPARAISON: Avec vs Sans Features Acidité")
print("="*70)

# Entraîner modèle sans les nouvelles features
print("\n⏳ Entraînement du modèle de référence (sans features acidité)...")

# Retirer les features acidité
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
print(f"{'Modèle':<35} {'Brier Test':<15} {'Amélioration'}")
print("-" * 70)
print(f"{'Sans features acidité':<35} {brier_test_base:.4f}")
print(f"{'Avec features acidité':<35} {brier_test:.4f}         {(brier_test_base - brier_test):.4f} {'✅' if brier_test < brier_test_base else '❌'}")

improvement_pct = ((brier_test_base - brier_test) / brier_test_base) * 100
print(f"\n{'Amélioration relative:':<35} {improvement_pct:+.2f}%")

if brier_test < brier_test_base:
    print("\n✅ Les features acidité AMÉLIORENT le modèle!")
    print(f"   Gain: {(brier_test_base - brier_test):.4f} points de Brier")
else:
    print("\n⚠️  Les features acidité n'améliorent pas le modèle.")
    print("   Possibles raisons:")
    print("   - Information déjà capturée par features existantes")
    print("   - Acidité moins importante que sel pour la corrosion aviation")
    print("   - Besoin d'ajuster les formules ou combinaisons")

print("="*70)

# ============================================================================
# 💡 INSIGHTS SUPPLÉMENTAIRES
# ============================================================================

print("\n" + "="*70)
print("💡 ANALYSE COMPLÉMENTAIRE")
print("="*70)

# Corrélation entre acidité et corrosion
corr_acid_corrosion = df[['acidite_totale', 'corrosion_risk']].corr().iloc[0, 1]
print(f"\n📊 Corrélation acidité ↔ corrosion: {corr_acid_corrosion:.4f}")

# Distribution de l'acidité selon le risque de corrosion
print(f"\n📈 Acidité moyenne selon le risque:")
print(f"   Sans corrosion (0): {df[df['corrosion_risk']==0]['acidite_totale'].mean():.2e}")
print(f"   Avec corrosion (1): {df[df['corrosion_risk']==1]['acidite_totale'].mean():.2e}")

ratio = df[df['corrosion_risk']==1]['acidite_totale'].mean() / df[df['corrosion_risk']==0]['acidite_totale'].mean()
print(f"   Ratio: {ratio:.2f}x")

print("="*70)

# Made with Bob
