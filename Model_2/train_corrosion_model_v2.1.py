"""
Aircraft Corrosion Prediction Model V2 - Enhanced Feature Engineering
Creates ~2000 features through polynomial interactions, advanced aggregations,
and domain-specific feature engineering.
Optimized to minimize Brier score.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("AIRCRAFT CORROSION PREDICTION MODEL V2.1 - OVERFITTING TEST")
print("Different Random Seed for Train/Test Split")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Step 1: Loading data...")
env = pd.read_csv('data/environment_training.csv')
corrosions = pd.read_csv('data/corrosions_training.csv')
labels = pd.read_csv('generer_predictions/prediction_training.csv')

print(f"  - Environment data: {env.shape[0]:,} rows, {env.shape[1]} columns")
print(f"  - Corrosions data: {corrosions.shape[0]:,} rows, {corrosions.shape[1]} columns")
print(f"  - Labels data: {labels.shape[0]:,} rows, {labels.shape[1]} columns")

# ============================================================================
# 2. PREPARE LABELS
# ============================================================================
print("\nStep 2: Preparing labels...")
labels[['aircraft_id', 'date']] = labels['id'].str.split('_', expand=True)
env['date'] = pd.to_datetime(env['month_start_date']).dt.to_period('M').astype(str)
labels['date'] = pd.to_datetime(labels['date']).dt.to_period('M').astype(str)

print(f"  - Unique aircraft in labels: {labels['aircraft_id'].nunique()}")
print(f"  - Corrosion rate: {labels['corrosion_risk'].mean():.2%}")

# ============================================================================
# 3. MERGE DATA
# ============================================================================
print("\nStep 3: Merging environment data with labels...")
df = env.merge(labels[['aircraft_id', 'date', 'corrosion_risk']], 
               on=['aircraft_id', 'date'], 
               how='inner')

print(f"  - Merged dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"  - Unique aircraft: {df['aircraft_id'].nunique()}")

df = df.sort_values(['aircraft_id', 'date']).reset_index(drop=True)

# ============================================================================
# 4. ENHANCED FEATURE ENGINEERING (~2000 features)
# ============================================================================
print("\nStep 4: Engineering ~2000 features (this may take a few minutes)...")

meta_cols = ['aircraft_id', 'year_month', 'month_start_date', 'date', 'corrosion_risk']
base_cols = [c for c in df.select_dtypes('number').columns if c not in meta_cols]
print(f"  - Base features: {len(base_cols)}")

g = df.groupby('aircraft_id', sort=False)
feature_parts = []
feature_count = 0

# ============================================================================
# 4.1 ORIGINAL FEATURES
# ============================================================================
print("  - Adding original features...")
feature_parts.append(df[base_cols])
feature_count += len(base_cols)

# ============================================================================
# 4.2 CUMULATIVE FEATURES (sum, mean, max, min, std)
# ============================================================================
print("  - Adding cumulative features (sum, mean, max, min, std)...")
feature_parts.append(g[base_cols].cumsum().add_suffix('_cumsum'))
feature_parts.append(g[base_cols].cumsum().div(g.cumcount() + 1, axis=0).add_suffix('_cummean'))
feature_parts.append(g[base_cols].cummax().add_suffix('_cummax'))
feature_parts.append(g[base_cols].cummin().add_suffix('_cummin'))
# Cumulative std (expanding window)
cumstd = g[base_cols].expanding().std().reset_index(level=0, drop=True).fillna(0)
feature_parts.append(cumstd.add_suffix('_cumstd'))
feature_count += len(base_cols) * 5

# ============================================================================
# 4.3 ROLLING WINDOW FEATURES (multiple windows: 2, 3, 6, 12, 24 months)
# ============================================================================
print("  - Adding rolling window features (2, 3, 6, 12, 24 months)...")
for window in [2, 3, 6, 12, 24]:
    roll_mean = g[base_cols].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    roll_std = g[base_cols].rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    roll_min = g[base_cols].rolling(window, min_periods=1).min().reset_index(level=0, drop=True)
    roll_max = g[base_cols].rolling(window, min_periods=1).max().reset_index(level=0, drop=True)
    
    feature_parts.append(roll_mean.add_suffix(f'_roll{window}_mean'))
    feature_parts.append(roll_std.add_suffix(f'_roll{window}_std'))
    feature_parts.append(roll_min.add_suffix(f'_roll{window}_min'))
    feature_parts.append(roll_max.add_suffix(f'_roll{window}_max'))
    feature_count += len(base_cols) * 4

# ============================================================================
# 4.4 LAG FEATURES (1, 2, 3, 6, 12 months)
# ============================================================================
print("  - Adding lag features (1, 2, 3, 6, 12 months)...")
for lag in [1, 2, 3, 6, 12]:
    feature_parts.append(g[base_cols].shift(lag).add_suffix(f'_lag{lag}'))
    feature_count += len(base_cols)

# ============================================================================
# 4.5 DIFFERENCE FEATURES (change from previous periods)
# ============================================================================
print("  - Adding difference features...")
for lag in [1, 3, 6, 12]:
    diff = g[base_cols].diff(lag).add_suffix(f'_diff{lag}')
    feature_parts.append(diff)
    feature_count += len(base_cols)

# ============================================================================
# 4.6 RATE OF CHANGE FEATURES
# ============================================================================
print("  - Adding rate of change features...")
for lag in [1, 3, 6]:
    pct_change = g[base_cols].pct_change(lag).replace([np.inf, -np.inf], 0).fillna(0).add_suffix(f'_pct_change{lag}')
    feature_parts.append(pct_change)
    feature_count += len(base_cols)

# ============================================================================
# 4.7 EXPONENTIAL WEIGHTED MOVING AVERAGES
# ============================================================================
print("  - Adding exponential weighted moving averages...")
for span in [3, 6, 12]:
    ewm = g[base_cols].ewm(span=span, min_periods=1).mean().reset_index(level=0, drop=True)
    feature_parts.append(ewm.add_suffix(f'_ewm{span}'))
    feature_count += len(base_cols)

# ============================================================================
# 4.8 AIRCRAFT AGE AND TIME-BASED FEATURES
# ============================================================================
print("  - Adding time-based features...")
age = g.cumcount().rename('age_months')
age_squared = (age ** 2).rename('age_months_squared')
age_cubed = (age ** 3).rename('age_months_cubed')
age_sqrt = np.sqrt(age).rename('age_months_sqrt')
feature_parts.extend([age, age_squared, age_cubed, age_sqrt])
feature_count += 4

# ============================================================================
# 4.9 DOMAIN-SPECIFIC INTERACTION FEATURES
# ============================================================================
print("  - Adding domain-specific interaction features...")

# Corrosion risk factors: temperature × humidity × sea salt
if 'temperature' in base_cols and 'metar_relative_humidity' in base_cols:
    temp_humidity = (df['temperature'] * df['metar_relative_humidity']).rename('temp_humidity_interaction')
    feature_parts.append(temp_humidity)
    feature_count += 1

# Sea salt exposure index (sum of all sea salt aerosols)
sea_salt_cols = [c for c in base_cols if 'sea_salt' in c]
if sea_salt_cols:
    sea_salt_total = df[sea_salt_cols].sum(axis=1).rename('sea_salt_total')
    feature_parts.append(sea_salt_total)
    feature_count += 1

# Dust exposure index
dust_cols = [c for c in base_cols if 'dust' in c]
if dust_cols:
    dust_total = df[dust_cols].sum(axis=1).rename('dust_total')
    feature_parts.append(dust_total)
    feature_count += 1

# Pollution index (carbon monoxide + nitrogen oxides + sulfur dioxide)
pollution_cols = ['carbon_monoxide_mass_mixing_ratio', 'nitrogen_dioxide_mass_mixing_ratio', 
                  'nitrogen_monoxide_mass_mixing_ratio', 'sulphur_dioxide_mass_mixing_ratio']
pollution_cols = [c for c in pollution_cols if c in base_cols]
if pollution_cols:
    pollution_index = df[pollution_cols].sum(axis=1).rename('pollution_index')
    feature_parts.append(pollution_index)
    feature_count += 1

# Aerosol exposure index
aerosol_cols = [c for c in base_cols if 'aerosol' in c]
if aerosol_cols:
    aerosol_total = df[aerosol_cols].sum(axis=1).rename('aerosol_total')
    feature_parts.append(aerosol_total)
    feature_count += 1

# Temperature × parking time (longer exposure at high temps)
if 'temperature' in base_cols and 'total_parking_minutes' in base_cols:
    temp_parking = (df['temperature'] * df['total_parking_minutes']).rename('temp_parking_interaction')
    feature_parts.append(temp_parking)
    feature_count += 1

# Humidity × sea salt (enhanced corrosion)
if 'metar_relative_humidity' in base_cols and sea_salt_cols:
    humidity_seasalt = (df['metar_relative_humidity'] * df[sea_salt_cols].sum(axis=1)).rename('humidity_seasalt_interaction')
    feature_parts.append(humidity_seasalt)
    feature_count += 1

# ============================================================================
# 4.10 POLYNOMIAL FEATURES (degree 2) on selected high-importance features
# ============================================================================
print("  - Adding polynomial interaction features (degree 2)...")

# Select top features for polynomial expansion to avoid explosion
top_features = ['temperature', 'total_parking_minutes', 'metar_relative_humidity',
                'metar_wind_speed_kn', 'specific_humidity']
top_features = [f for f in top_features if f in base_cols]

if len(top_features) >= 2:
    # Fill NaN values before polynomial transformation
    top_features_data = df[top_features].fillna(0)
    
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(top_features_data)
    poly_feature_names = poly.get_feature_names_out(top_features)
    
    # Only add the interaction terms (not the original features which we already have)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    # Remove original features from poly_df
    poly_df = poly_df[[c for c in poly_df.columns if c not in top_features]]
    feature_parts.append(poly_df)
    feature_count += len(poly_df.columns)

# ============================================================================
# 4.11 RATIO FEATURES (important ratios)
# ============================================================================
print("  - Adding ratio features...")

# Temperature to humidity ratio
if 'temperature' in base_cols and 'metar_relative_humidity' in base_cols:
    temp_humidity_ratio = (df['temperature'] / (df['metar_relative_humidity'] + 1)).rename('temp_humidity_ratio')
    feature_parts.append(temp_humidity_ratio)
    feature_count += 1

# Sea salt to dust ratio
if sea_salt_cols and dust_cols:
    seasalt_dust_ratio = (df[sea_salt_cols].sum(axis=1) / (df[dust_cols].sum(axis=1) + 1e-6)).rename('seasalt_dust_ratio')
    feature_parts.append(seasalt_dust_ratio)
    feature_count += 1

# Parking time to age ratio
if 'total_parking_minutes' in base_cols:
    parking_age_ratio = (df['total_parking_minutes'] / (age + 1)).rename('parking_age_ratio')
    feature_parts.append(parking_age_ratio)
    feature_count += 1

# ============================================================================
# 4.12 STATISTICAL AGGREGATIONS (skewness, kurtosis)
# ============================================================================
print("  - Adding statistical aggregation features...")

for window in [6, 12]:
    # Skewness
    roll_skew = g[base_cols[:10]].rolling(window, min_periods=3).skew().reset_index(level=0, drop=True).fillna(0)
    feature_parts.append(roll_skew.add_suffix(f'_roll{window}_skew'))
    feature_count += len(base_cols[:10])
    
    # Kurtosis
    roll_kurt = g[base_cols[:10]].rolling(window, min_periods=3).kurt().reset_index(level=0, drop=True).fillna(0)
    feature_parts.append(roll_kurt.add_suffix(f'_roll{window}_kurt'))
    feature_count += len(base_cols[:10])

# ============================================================================
# 4.13 COMBINE ALL FEATURES
# ============================================================================
print(f"\n  - Combining all features...")
X = pd.concat(feature_parts, axis=1)
y = df['corrosion_risk']
groups = df['aircraft_id']

# Handle any remaining NaN or inf values
X = X.replace([np.inf, -np.inf], 0).fillna(0)

print(f"\n  ✓ Total engineered features: {X.shape[1]:,}")
print(f"  ✓ Target: {feature_count:,} features (actual: {X.shape[1]:,})")

# ============================================================================
# 5. TRAIN/TEST SPLIT (80/20 by aircraft)
# ============================================================================
print("\nStep 5: Splitting data (80% train, 20% test by aircraft)...")
print("  ⚠ Using DIFFERENT random seed (99 instead of 42) to test for overfitting")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=99)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"  - Train set: {len(X_train):,} samples ({len(X_train)/len(X):.1%})")
print(f"  - Test set: {len(X_test):,} samples ({len(X_test)/len(X):.1%})")
print(f"  - Train aircraft: {groups.iloc[train_idx].nunique()}")
print(f"  - Test aircraft: {groups.iloc[test_idx].nunique()}")
print(f"  - Train corrosion rate: {y_train.mean():.2%}")
print(f"  - Test corrosion rate: {y_test.mean():.2%}")

# ============================================================================
# 6. TRAIN MODEL
# ============================================================================
print("\nStep 6: Training XGBoost model with enhanced features...")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,      # Additional column sampling per level
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

print("  - Training in progress...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print("  ✓ Model training complete")

# ============================================================================
# 7. EVALUATE MODEL
# ============================================================================
print("\nStep 7: Evaluating model performance...")

y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

train_brier = brier_score_loss(y_train, y_train_proba)
test_brier = brier_score_loss(y_test, y_test_proba)
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
train_logloss = log_loss(y_train, y_train_proba)
test_logloss = log_loss(y_test, y_test_proba)

print("\n" + "=" * 80)
print("MODEL V2 PERFORMANCE METRICS")
print("=" * 80)
print(f"\nBRIER SCORE (lower is better, optimized metric):")
print(f"  - Train: {train_brier:.6f}")
print(f"  - Test:  {test_brier:.6f}")
print(f"\nAUC-ROC (higher is better):")
print(f"  - Train: {train_auc:.4f}")
print(f"  - Test:  {test_auc:.4f}")
print(f"\nLog Loss (lower is better):")
print(f"  - Train: {train_logloss:.6f}")
print(f"  - Test:  {test_logloss:.6f}")

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("TOP 30 MOST IMPORTANT FEATURES")
print("=" * 80)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in importance_df.head(30).iterrows():
    print(f"  {row['feature']:70s} {row['importance']:.4f}")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS TO Model_2/")
print("=" * 80)

test_results = pd.DataFrame({
    'aircraft_id': groups.iloc[test_idx].values,
    'date': df.iloc[test_idx]['date'].values,
    'actual_corrosion_risk': y_test.values,
    'predicted_corrosion_probability': y_test_proba
})
test_results.to_csv('Model_2/test_predictions_v2.1.csv', index=False)
print(f"  ✓ Test predictions saved to: Model_2/test_predictions_v2.1.csv")

summary = {
    'metric': ['Brier Score', 'AUC-ROC', 'Log Loss', 'Train Size', 'Test Size',
               'Train Aircraft', 'Test Aircraft', 'Total Features', 'Random Seed'],
    'train': [train_brier, train_auc, train_logloss, len(X_train), '',
              groups.iloc[train_idx].nunique(), '', X.shape[1], ''],
    'test': [test_brier, test_auc, test_logloss, '', len(X_test),
             '', groups.iloc[test_idx].nunique(), '', 99]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv('Model_2/model_performance_summary_v2.1.csv', index=False)
print(f"  ✓ Performance summary saved to: Model_2/model_performance_summary_v2.1.csv")

importance_df.to_csv('Model_2/feature_importance_v2.1.csv', index=False)
print(f"  ✓ Feature importance saved to: Model_2/feature_importance_v2.1.csv")

# ============================================================================
# 10. OVERFITTING TEST COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("OVERFITTING TEST: COMPARISON WITH ORIGINAL MODEL V2")
print("=" * 80)
print(f"\nModel V2 Original (random_state=42):")
print(f"  - Test Brier Score: 0.098912")
print(f"  - Test AUC-ROC: 0.9242")
print(f"\nModel V2.1 Different Split (random_state=99):")
print(f"  - Test Brier Score: {test_brier:.6f}")
print(f"  - Test AUC-ROC: {test_auc:.4f}")
print(f"\nDifference (V2.1 vs V2):")
brier_diff = test_brier - 0.098912
auc_diff = test_auc - 0.9242
brier_pct = (brier_diff / 0.098912) * 100
auc_pct = (auc_diff / 0.9242) * 100
print(f"  - Brier Score: {brier_diff:+.6f} ({brier_pct:+.2f}%)")
print(f"  - AUC-ROC: {auc_diff:+.4f} ({auc_pct:+.2f}%)")

print("\n" + "=" * 80)
print("OVERFITTING ASSESSMENT")
print("=" * 80)
if abs(brier_pct) < 5 and abs(auc_pct) < 2:
    print("✓ GOOD: Performance is stable across different train/test splits")
    print("  The model generalizes well and is NOT overfitted")
elif abs(brier_pct) < 10 and abs(auc_pct) < 5:
    print("⚠ MODERATE: Some variation in performance across splits")
    print("  The model shows acceptable generalization")
else:
    print("✗ WARNING: Significant performance degradation on different split")
    print("  The model may be overfitted to the specific train/test split")

print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)
print(f"\n✓ Model V2.1 (overfitting test) successfully trained and evaluated")
print(f"✓ Test set Brier Score: {test_brier:.6f}")
print(f"✓ Test set AUC-ROC: {test_auc:.4f}")
print(f"✓ Total features: {X.shape[1]:,}")
print(f"✓ Random seed: 99 (different from original 42)")
print(f"✓ Results saved to Model_2/ directory with _v2.1 suffix")
print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Made with Bob
