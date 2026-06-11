import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit

print("\n" + "="*70)
print("🚀 MODÈLE OPTIMISÉ AVEC TOUTES LES FEATURES + SÉLECTION AUTO")
print("="*70)

env    = pd.read_csv('data/environment_training.csv')
labels = pd.read_csv('generer_predictions/prediction_training.csv')

labels[['aircraft_id', 'date']] = labels['id'].str.split('_', expand=True)

env['date']    = pd.to_datetime(env['month_start_date']).dt.to_period('M').astype(str)
labels['date'] = pd.to_datetime(labels['date']).dt.to_period('M').astype(str)

df = env.merge(labels[['aircraft_id', 'date', 'corrosion_risk']], on=['aircraft_id', 'date'])
df = df.sort_values(['aircraft_id', 'date']).reset_index(drop=True)

meta_cols = ['aircraft_id', 'year_month', 'month_start_date', 'date', 'corrosion_risk']
base_cols = [c for c in df.select_dtypes('number').columns if c not in meta_cols]

print(f"\n📊 Variables de base: {len(base_cols)}")

# ============================================================================
# 🆕 CRÉATION DE TOUTES LES NOUVELLES FEATURES
# ============================================================================

print("\n" + "="*70)
print("🔬 CRÉATION DES FEATURES AVANCÉES")
print("="*70)

g = df.groupby('aircraft_id', sort=False)

new_features = []

# ============================================================================
# 1. INDICES DE CORROSIVITÉ MARINE (6 features)
# ============================================================================
print("\n1️⃣ Indices de corrosivité marine...")

df['salinite_totale'] = (
    df['sea_salt_aerosol_003_05_mixing_ratio'] + 
    df['sea_salt_aerosol_05_5_mixing_ratio'] + 
    df['sea_salt_aerosol_5_20_mixing_ratio']
)
new_features.append('salinite_totale')

df['marine_corrosion_index'] = (
    df['salinite_totale'] * 
    df['metar_relative_humidity'] * 
    df['total_parking_minutes'] / 1000
)
new_features.append('marine_corrosion_index')

df['salt_fine_ratio'] = (
    df['sea_salt_aerosol_003_05_mixing_ratio'] / 
    (df['sea_salt_aerosol_5_20_mixing_ratio'] + 1e-12)
)
new_features.append('salt_fine_ratio')

df['humidity_salt_interaction'] = (
    df['metar_relative_humidity'] * df['salinite_totale']
)
new_features.append('humidity_salt_interaction')

df['condensation_risk'] = (
    df['metar_temperature_c'] * df['metar_relative_humidity']
)
new_features.append('condensation_risk')

df['high_humidity_exposure'] = (
    df['total_parking_minutes'] * (df['metar_relative_humidity'] > 80).astype(int)
)
new_features.append('high_humidity_exposure')

print(f"   ✅ {6} features créées")

# ============================================================================
# 2. INDICES D'ACIDITÉ (6 features)
# ============================================================================
print("\n2️⃣ Indices d'acidité...")

df['acidite_totale'] = (
    df['h2o2'] + df['formaldehyde'] + 
    df['hno3'] + df['sulphur_dioxide_mass_mixing_ratio']
)
new_features.append('acidite_totale')

df['acid_rain_risk'] = (
    df['metar_hour_precipitation'] * df['acidite_totale']
)
new_features.append('acid_rain_risk')

df['humid_acidity'] = (
    df['metar_relative_humidity'] * df['acidite_totale']
)
new_features.append('humid_acidity')

df['strong_acid_ratio'] = (
    (df['hno3'] + df['sulphur_dioxide_mass_mixing_ratio']) /
    (df['h2o2'] + df['formaldehyde'] + 1e-12)
)
new_features.append('strong_acid_ratio')

df['acid_exposure'] = (
    df['total_parking_minutes'] * df['acidite_totale'] / 1000
)
new_features.append('acid_exposure')

df['acid_salt_synergy'] = (
    df['acidite_totale'] * df['salinite_totale']
)
new_features.append('acid_salt_synergy')

print(f"   ✅ {6} features créées")

# ============================================================================
# 3. EXPOSITION PONDÉRÉE (12 features)
# ============================================================================
print("\n3️⃣ Exposition cumulative pondérée...")

DECAY_FACTOR = 0.95

key_vars_weighted = [
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

for var in key_vars_weighted:
    def weighted_cumsum(series):
        n = len(series)
        weights = DECAY_FACTOR ** np.arange(n)[::-1]
        return (series.values * weights).cumsum()
    
    weighted_col = f'{var}_weighted_exp'
    df[weighted_col] = g[var].transform(weighted_cumsum)
    new_features.append(weighted_col)

df['salt_total_weighted'] = (
    df['sea_salt_aerosol_003_05_mixing_ratio_weighted_exp'] +
    df['sea_salt_aerosol_05_5_mixing_ratio_weighted_exp'] +
    df['sea_salt_aerosol_5_20_mixing_ratio_weighted_exp']
)
new_features.append('salt_total_weighted')

df['acidity_total_weighted'] = (
    df['h2o2_weighted_exp'] +
    df['formaldehyde_weighted_exp'] +
    df['hno3_weighted_exp'] +
    df['sulphur_dioxide_mass_mixing_ratio_weighted_exp']
)
new_features.append('acidity_total_weighted')

df['corrosion_index_weighted'] = (
    df['salt_total_weighted'] * 
    df['metar_relative_humidity_weighted_exp']
)
new_features.append('corrosion_index_weighted')

print(f"   ✅ {12} features créées")

# ============================================================================
# 4. POLLUTION INDUSTRIELLE (5 features)
# ============================================================================
print("\n4️⃣ Indices de pollution industrielle...")

df['pollution_index'] = (
    df['carbon_monoxide_mass_mixing_ratio'] +
    df['nitrogen_dioxide_mass_mixing_ratio'] +
    df['sulphate_aerosol_mixing_ratio']
)
new_features.append('pollution_index')

df['nox_ratio'] = (
    df['nitrogen_dioxide_mass_mixing_ratio'] / 
    (df['nitrogen_monoxide_mass_mixing_ratio'] + 1e-12)
)
new_features.append('nox_ratio')

df['ozone_pollution'] = (
    df['ozone_mass_mixing_ratio'] * df['pollution_index']
)
new_features.append('ozone_pollution')

df['urban_pollution'] = (
    df['carbon_monoxide_mass_mixing_ratio'] * 
    df['nitrogen_dioxide_mass_mixing_ratio']
)
new_features.append('urban_pollution')

df['industrial_exposure'] = (
    df['total_parking_minutes'] * df['pollution_index'] / 1000
)
new_features.append('industrial_exposure')

print(f"   ✅ {5} features créées")

# ============================================================================
# 5. AÉROSOLS ET POUSSIÈRES (6 features)
# ============================================================================
print("\n5️⃣ Indices d'aérosols et poussières...")

df['dust_total'] = (
    df['dust_aerosol_003_055_mixing_ratio'] +
    df['dust_aerosol_055_09_mixing_ratio'] +
    df['dust_aerosol_09_20_mixing_ratio']
)
new_features.append('dust_total')

df['organic_aerosol_total'] = (
    df['hydrophilic_organic_matter_aerosol_mixing_ratio'] +
    df['hydrophobic_organic_matter_aerosol_mixing_ratio']
)
new_features.append('organic_aerosol_total')

df['black_carbon_total'] = (
    df['hydrophilic_black_carbon_aerosol_mixing_ratio'] +
    df['hydrophobic_black_carbon_aerosol_mixing_ratio']
)
new_features.append('black_carbon_total')

df['hydrophilic_ratio'] = (
    df['hydrophilic_organic_matter_aerosol_mixing_ratio'] /
    (df['hydrophobic_organic_matter_aerosol_mixing_ratio'] + 1e-12)
)
new_features.append('hydrophilic_ratio')

df['dust_salt_mix'] = df['dust_total'] * df['salinite_totale']
new_features.append('dust_salt_mix')

df['aerosol_exposure'] = (
    df['total_parking_minutes'] * 
    (df['dust_total'] + df['organic_aerosol_total']) / 1000
)
new_features.append('aerosol_exposure')

print(f"   ✅ {6} features créées")

# ============================================================================
# 6. STRESS THERMIQUE ET CYCLES (5 features)
# ============================================================================
print("\n6️⃣ Stress thermique et cycles...")

df['temp_variance'] = g['metar_temperature_c'].transform(
    lambda x: x.rolling(3, min_periods=1).std()
)
new_features.append('temp_variance')

df['thermal_stress'] = g['metar_temperature_c'].transform(
    lambda x: x.diff().abs()
)
new_features.append('thermal_stress')

df['humidity_variance'] = g['metar_relative_humidity'].transform(
    lambda x: x.rolling(3, min_periods=1).std()
)
new_features.append('humidity_variance')

df['temp_humidity_product'] = (
    df['metar_temperature_c'] * df['metar_relative_humidity']
)
new_features.append('temp_humidity_product')

df['extreme_temp'] = (
    (df['metar_temperature_c'] > 35) | 
    (df['metar_temperature_c'] < 5)
).astype(int)
new_features.append('extreme_temp')

print(f"   ✅ {5} features créées")

# ============================================================================
# 7. SEUILS CRITIQUES ET CONDITIONS DANGEREUSES (6 features)
# ============================================================================
print("\n7️⃣ Seuils critiques et conditions dangereuses...")

df['high_salt_zone'] = (
    df['salinite_totale'] > df['salinite_totale'].quantile(0.75)
).astype(int)
new_features.append('high_salt_zone')

df['high_humidity_zone'] = (
    df['metar_relative_humidity'] > 80
).astype(int)
new_features.append('high_humidity_zone')

df['danger_zone'] = (
    df['high_salt_zone'] * df['high_humidity_zone']
)
new_features.append('danger_zone')

df['corrosive_conditions'] = (
    (df['salinite_totale'] > df['salinite_totale'].quantile(0.75)) &
    (df['metar_relative_humidity'] > 70) &
    (df['metar_temperature_c'] > 20)
).astype(int)
new_features.append('corrosive_conditions')

df['acid_attack_risk'] = (
    (df['acidite_totale'] > df['acidite_totale'].quantile(0.75)) &
    (df['metar_hour_precipitation'] > 0)
).astype(int)
new_features.append('acid_attack_risk')

df['combined_risk_score'] = (
    df['high_salt_zone'] + 
    df['high_humidity_zone'] + 
    (df['acidite_totale'] > df['acidite_totale'].quantile(0.75)).astype(int)
)
new_features.append('combined_risk_score')

print(f"   ✅ {6} features créées")

# ============================================================================
# 8. RATIOS ET INTERACTIONS CHIMIQUES (5 features)
# ============================================================================
print("\n8️⃣ Ratios et interactions chimiques...")

df['salt_acid_ratio'] = (
    df['salinite_totale'] / (df['acidite_totale'] + 1e-12)
)
new_features.append('salt_acid_ratio')

df['organic_inorganic_ratio'] = (
    df['organic_aerosol_total'] / 
    (df['sulphate_aerosol_mixing_ratio'] + 1e-12)
)
new_features.append('organic_inorganic_ratio')

df['oxidation_potential'] = (
    df['ozone_mass_mixing_ratio'] + 
    df['h2o2'] + 
    df['oh']
)
new_features.append('oxidation_potential')

df['corrosion_accelerator'] = (
    df['salinite_totale'] * df['acidite_totale'] * df['metar_relative_humidity']
)
new_features.append('corrosion_accelerator')

df['chemical_aggressiveness'] = (
    df['acidite_totale'] + df['salinite_totale'] + df['pollution_index']
)
new_features.append('chemical_aggressiveness')

print(f"   ✅ {5} features créées")

print(f"\n📊 TOTAL NOUVELLES FEATURES: {len(new_features)}")

# Ajouter aux colonnes de base
all_base_cols = list(base_cols) + new_features

# ============================================================================
# Feature Engineering Standard
# ============================================================================

print("\n" + "="*70)
print("⚙️  FEATURE ENGINEERING STANDARD")
print("="*70)

feature_parts = [df[all_base_cols]]
feature_parts.append(g[all_base_cols].cumsum().add_suffix('_cumsum'))
feature_parts.append(g[all_base_cols].cumsum().div(g.cumcount() + 1, axis=0).add_suffix('_cummean'))
feature_parts.append(g[all_base_cols].cummax().add_suffix('_cummax'))

roll3  = g[all_base_cols].rolling(3,  min_periods=1).mean().reset_index(level=0, drop=True)
roll12 = g[all_base_cols].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
feature_parts.append(roll3.add_suffix('_roll3'))
feature_parts.append(roll12.add_suffix('_roll12'))
feature_parts.append(g[all_base_cols].shift(1).add_suffix('_lag1'))

age = g.cumcount().rename('age_months')

X_full = pd.concat(feature_parts + [age], axis=1)
y = df['corrosion_risk']
groups = df['aircraft_id']

print(f"\n📊 Nombre TOTAL de features: {X_full.shape[1]}")

# ============================================================================
# Split
# ============================================================================

gss = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
trainval_idx, test_idx = next(gss.split(X_full, y, groups))
train_idx, val_idx = next(gss.split(
    X_full.iloc[trainval_idx], y.iloc[trainval_idx], groups.iloc[trainval_idx]))
train_idx, val_idx = trainval_idx[train_idx], trainval_idx[val_idx]

X_train_full = X_full.iloc[train_idx]
X_val_full = X_full.iloc[val_idx]
X_test_full = X_full.iloc[test_idx]
y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

# ============================================================================
# 🧹 SUPPRESSION DES FEATURES REDONDANTES
# ============================================================================

print("\n" + "="*70)
print("🧹 SUPPRESSION DES FEATURES REDONDANTES")
print("="*70)

print(f"\n📊 Nombre de features avant nettoyage: {X_train_full.shape[1]}")

# Calculer la matrice de corrélation
print("⏳ Calcul de la matrice de corrélation...")
corr_matrix = X_train_full.corr().abs()

# Trouver les paires de features avec corrélation > 0.95
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

print(f"🗑️  Features à supprimer (corrélation > 0.95): {len(to_drop)}")

# Supprimer les features redondantes
X_train_clean = X_train_full.drop(columns=to_drop)
X_val_clean = X_val_full.drop(columns=to_drop)
X_test_clean = X_test_full.drop(columns=to_drop)

print(f"✅ Nombre de features après nettoyage: {X_train_clean.shape[1]}")
print(f"📉 Réduction: {len(to_drop)} features supprimées ({len(to_drop)/X_train_full.shape[1]*100:.1f}%)")

# ============================================================================
# 🎯 SÉLECTION AUTOMATIQUE DES TOP K FEATURES
# ============================================================================

# Définir le nombre de features à garder
TOP_K = 50

print("\n" + "="*70)
print(f"🎯 SÉLECTION AUTOMATIQUE DES TOP {TOP_K} FEATURES")
print("="*70)

print("\n⏳ Entraînement d'un modèle rapide pour obtenir les importances...")

# Modèle rapide pour feature importance (sur features nettoyées)
temp_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='auc', early_stopping_rounds=20,
)
temp_model.fit(X_train_clean, y_train, eval_set=[(X_val_clean, y_val)], verbose=False)

# Récupérer les importances
importances = pd.Series(temp_model.feature_importances_, index=X_train_clean.columns)
importances = importances.sort_values(ascending=False)

# Sélectionner top K
top_features = importances.nlargest(TOP_K).index.tolist()

print(f"\n✅ Top {TOP_K} features sélectionnées")
print(f"\n🏆 Top 30 features les plus importantes:")
for i, (feat, imp) in enumerate(importances.head(30).items(), 1):
    # Marquer si c'est une nouvelle feature
    is_new = any(nf in feat for nf in new_features)
    marker = "🆕" if is_new else "  "
    print(f"   {marker} {i:2d}. {feat:55s} {imp:.4f}")

# Compter combien de nouvelles features dans le top TOP_K
new_in_topK = sum(1 for feat in top_features if any(nf in feat for nf in new_features))
print(f"\n📊 Nouvelles features dans le top {TOP_K}: {new_in_topK}/{TOP_K} ({new_in_topK/TOP_K*100:.1f}%)")

# Appliquer la sélection
X_train = X_train_clean[top_features]
X_val = X_val_clean[top_features]
X_test = X_test_clean[top_features]

print("="*70)

# ============================================================================
# 🚀 MODÈLE FINAL AVEC TOP K FEATURES
# ============================================================================

print("\n" + "="*70)
print("🚀 ENTRAÎNEMENT DU MODÈLE FINAL (TOP K FEATURES)")
print(f"Nombre de features sélectionnées: {len(top_features)}")
print("="*70)

model = xgb.XGBClassifier(
    n_estimators=2000, max_depth=6, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, gamma=1.0, reg_lambda=5.0,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric='auc', early_stopping_rounds=50,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Évaluation
proba_train = model.predict_proba(X_train)[:, 1]
proba_test = model.predict_proba(X_test)[:, 1]

auc_train = roc_auc_score(y_train, proba_train)
auc_test = roc_auc_score(y_test, proba_test)

brier_train = brier_score_loss(y_train, proba_train)
brier_test = brier_score_loss(y_test, proba_test)

print("\n" + "="*70)
print(f"📊 RÉSULTATS FINAUX (TOP {TOP_K} FEATURES)")
print("="*70)
print(f"Nb features    : {len(top_features)}")
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
# 📈 COMPARAISON AVEC MODÈLE DE BASE
# ============================================================================

print("\n" + "="*70)
print("📈 COMPARAISON AVEC MODÈLE DE BASE")
print("="*70)

print("\n⏳ Entraînement du modèle de base (sans nouvelles features)...")

# Features de base uniquement
base_features_only = [col for col in X_full.columns 
                      if not any(nf in col for nf in new_features)]

X_train_base = X_train_full[base_features_only]
X_val_base = X_val_full[base_features_only]
X_test_base = X_test_full[base_features_only]

# Sélectionner top K des features de base
temp_model_base = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    eval_metric='auc', early_stopping_rounds=20,
)
temp_model_base.fit(X_train_base, y_train, eval_set=[(X_val_base, y_val)], verbose=False)

imp_base = pd.Series(temp_model_base.feature_importances_, index=X_train_base.columns)
top_base = imp_base.nlargest(TOP_K).index.tolist()

X_train_base_top = X_train_base[top_base]
X_val_base_top = X_val_base[top_base]
X_test_base_top = X_test_base[top_base]

model_base = xgb.XGBClassifier(
    n_estimators=2000, max_depth=6, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, gamma=1.0, reg_lambda=5.0,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    eval_metric='auc', early_stopping_rounds=50,
)
model_base.fit(X_train_base_top, y_train, 
               eval_set=[(X_val_base_top, y_val)], verbose=False)

proba_test_base = model_base.predict_proba(X_test_base_top)[:, 1]
brier_test_base = brier_score_loss(y_test, proba_test_base)

print("\n📊 RÉSULTATS COMPARATIFS:")
print(f"{'Modèle':<50} {'Brier Test':<15} {'Amélioration'}")
print("-" * 80)
print(f"{'Base (top ' + str(TOP_K) + ' features originales)':<50} {brier_test_base:.4f}")
print(f"{'Optimisé (top ' + str(TOP_K) + ' avec nouvelles features)':<50} {brier_test:.4f}         {(brier_test_base - brier_test):.4f} {'✅' if brier_test < brier_test_base else '❌'}")

improvement_pct = ((brier_test_base - brier_test) / brier_test_base) * 100
print(f"\n{'Amélioration relative:':<45} {improvement_pct:+.2f}%")

print("\n" + "="*70)
print("💾 SAUVEGARDE DES TOP FEATURES")
print("="*70)

# Sauvegarder la liste des top features
top_features_df = pd.DataFrame({
    'feature': top_features,
    'importance': [importances[f] for f in top_features],
    'is_new': [any(nf in f for nf in new_features) for f in top_features]
})
filename = f'top_{TOP_K}_features.csv'
top_features_df.to_csv(filename, index=False)
print(f"\n✅ Liste sauvegardée dans '{filename}'")

print("\n" + "="*70)
print("🎉 ANALYSE TERMINÉE!")
print("="*70)

# Made with Bob
