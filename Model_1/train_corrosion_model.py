"""
Aircraft Corrosion Prediction Model
Predicts the probability of aircraft oxidation using environmental conditions.
Optimized to minimize Brier score.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
import xgboost as xgb
from datetime import datetime

print("=" * 80)
print("AIRCRAFT CORROSION PREDICTION MODEL")
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
# Split id into aircraft_id and date
labels[['aircraft_id', 'date']] = labels['id'].str.split('_', expand=True)

# Convert dates to period format for merging
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

# Sort by aircraft and date for time-series features
df = df.sort_values(['aircraft_id', 'date']).reset_index(drop=True)

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\nStep 4: Engineering features...")

# Identify base numerical columns (exclude metadata)
meta_cols = ['aircraft_id', 'year_month', 'month_start_date', 'date', 'corrosion_risk']
base_cols = [c for c in df.select_dtypes('number').columns if c not in meta_cols]
print(f"  - Base features: {len(base_cols)}")

# Group by aircraft for time-series features
g = df.groupby('aircraft_id', sort=False)

# Initialize feature list
feature_parts = []

# 1. Original features
feature_parts.append(df[base_cols])

# 2. Cumulative features (total exposure over time)
feature_parts.append(g[base_cols].cumsum().add_suffix('_cumsum'))

# 3. Cumulative mean (average exposure over aircraft lifetime)
feature_parts.append(g[base_cols].cumsum().div(g.cumcount() + 1, axis=0).add_suffix('_cummean'))

# 4. Cumulative max (peak exposure)
feature_parts.append(g[base_cols].cummax().add_suffix('_cummax'))

# 5. Rolling averages (recent exposure patterns)
roll3 = g[base_cols].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
roll6 = g[base_cols].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
roll12 = g[base_cols].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
feature_parts.append(roll3.add_suffix('_roll3'))
feature_parts.append(roll6.add_suffix('_roll6'))
feature_parts.append(roll12.add_suffix('_roll12'))

# 6. Lag features (previous month's conditions)
feature_parts.append(g[base_cols].shift(1).add_suffix('_lag1'))
feature_parts.append(g[base_cols].shift(3).add_suffix('_lag3'))

# 7. Aircraft age in months
age = g.cumcount().rename('age_months')

# 8. Standard deviation features (variability in exposure)
roll12_std = g[base_cols].rolling(12, min_periods=1).std().reset_index(level=0, drop=True)
feature_parts.append(roll12_std.add_suffix('_roll12_std'))

# Combine all features
X = pd.concat(feature_parts + [age], axis=1)
y = df['corrosion_risk']
groups = df['aircraft_id']

print(f"  - Total engineered features: {X.shape[1]:,}")
print(f"  - Feature types:")
print(f"    * Original: {len(base_cols)}")
print(f"    * Cumulative (sum, mean, max): {len(base_cols) * 3}")
print(f"    * Rolling averages (3, 6, 12): {len(base_cols) * 3}")
print(f"    * Lags (1, 3): {len(base_cols) * 2}")
print(f"    * Rolling std (12): {len(base_cols)}")
print(f"    * Age: 1")

# ============================================================================
# 5. TRAIN/TEST SPLIT (80/20 by aircraft)
# ============================================================================
print("\nStep 5: Splitting data (80% train, 20% test by aircraft)...")

# Use GroupShuffleSplit to ensure aircraft are not split between train/test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
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
print("\nStep 6: Training XGBoost model...")

# XGBoost optimized for probability prediction (Brier score)
model = xgb.XGBClassifier(
    n_estimators=500,           # More trees for better learning
    max_depth=6,                # Moderate depth to prevent overfitting
    learning_rate=0.05,         # Lower learning rate for better generalization
    subsample=0.8,              # Row sampling to prevent overfitting
    colsample_bytree=0.8,       # Column sampling for diversity
    min_child_weight=3,         # Minimum samples in leaf
    gamma=0.1,                  # Minimum loss reduction for split
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    eval_metric='logloss',      # Optimize for probability calibration
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

# Get probability predictions
y_train_proba = model.predict_proba(X_train)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
train_brier = brier_score_loss(y_train, y_train_proba)
test_brier = brier_score_loss(y_test, y_test_proba)
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
train_logloss = log_loss(y_train, y_train_proba)
test_logloss = log_loss(y_test, y_test_proba)

print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
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
print("TOP 20 MOST IMPORTANT FEATURES")
print("=" * 80)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in importance_df.head(20).iterrows():
    print(f"  {row['feature']:60s} {row['importance']:.4f}")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save test predictions
test_results = pd.DataFrame({
    'aircraft_id': groups.iloc[test_idx].values,
    'date': df.iloc[test_idx]['date'].values,
    'actual_corrosion_risk': y_test.values,
    'predicted_corrosion_probability': y_test_proba
})
test_results.to_csv('test_predictions.csv', index=False)
print(f"  ✓ Test predictions saved to: test_predictions.csv")

# Save model performance summary
summary = {
    'metric': ['Brier Score', 'AUC-ROC', 'Log Loss', 'Train Size', 'Test Size', 
               'Train Aircraft', 'Test Aircraft', 'Total Features'],
    'train': [train_brier, train_auc, train_logloss, len(X_train), '', 
              groups.iloc[train_idx].nunique(), '', X.shape[1]],
    'test': [test_brier, test_auc, test_logloss, '', len(X_test), 
             '', groups.iloc[test_idx].nunique(), '']
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv('model_performance_summary.csv', index=False)
print(f"  ✓ Performance summary saved to: model_performance_summary.csv")

# Save feature importance
importance_df.to_csv('feature_importance.csv', index=False)
print(f"  ✓ Feature importance saved to: feature_importance.csv")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)
print(f"\n✓ Model successfully trained and evaluated")
print(f"✓ Test set Brier Score: {test_brier:.6f}")
print(f"✓ Test set AUC-ROC: {test_auc:.4f}")
print(f"✓ Results saved to CSV files")
print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Made with Bob
