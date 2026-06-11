"""
XGBoost Inference with Corrosion Exposure Indices - FIXED VERSION
Generates predictions for ALL rows in environment_test.csv (~14k rows)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb

warnings.filterwarnings('ignore')

from data_utils import load_corrosion_data, load_environment_data, create_training_pairs, ENV_FEATURES

RANDOM_SEED = 42
DATA_DIR = Path(__file__).parent.parent / "data"
ENV_TEST = DATA_DIR / "environment_test.csv"

def aggregate_environmental_features_per_month(
    env_df: pd.DataFrame,
    features: list[str] = ENV_FEATURES,
) -> pd.DataFrame:
    """
    Aggregate environmental features per aircraft per month.
    Returns DataFrame with aircraft_id, month_start_date, and aggregated features.
    """
    # Calculate corrosion exposure indices
    env_df = env_df.copy()
    
    env_df['corrosion_exposure_index'] = (
        (env_df['sulphur_dioxide_mass_mixing_ratio'] + 
         env_df['nitrogen_dioxide_mass_mixing_ratio']) * 
        env_df['metar_relative_humidity'] * 
        env_df['total_parking_minutes'] / 1000
    )
    
    env_df['moisture_exposure_index'] = (
        env_df['metar_relative_humidity'] * 
        env_df['total_parking_minutes'] / 100
    )
    
    # Add indices to features
    features_with_indices = list(features) + ['corrosion_exposure_index', 'moisture_exposure_index']
    available_features = [f for f in features_with_indices if f in env_df.columns]
    
    if not available_features:
        raise ValueError(f"None of the requested features found in environment data")
    
    # For each aircraft-month, compute mean of available features
    result = env_df.groupby(["aircraft_id", "month_start_date"])[available_features].mean().reset_index()
    result = result.fillna(0)
    
    return result


def prepare_full_test_data(
    env_test_df: pd.DataFrame,
    features: list[str] = ENV_FEATURES,
) -> pd.DataFrame:
    """
    Prepare FULL test data - one prediction per row in environment_test.csv.
    """
    # Aggregate features per aircraft-month
    test_agg = aggregate_environmental_features_per_month(env_test_df, features=features)
    
    # Calculate months since first observation for each aircraft
    first_obs = env_test_df.groupby("aircraft_id")["month_start_date"].min().reset_index()
    first_obs.columns = ["aircraft_id", "first_observation_date"]
    
    result = test_agg.merge(first_obs, on="aircraft_id", how="left")
    
    # Calculate reference_month (months since first observation)
    result["reference_month"] = (
        (result["month_start_date"].dt.year - result["first_observation_date"].dt.year) * 12
        + (result["month_start_date"].dt.month - result["first_observation_date"].dt.month)
    )
    
    # Create id column in format: aircraft_id_YYYY-MM
    result["id"] = (
        result["aircraft_id"] + "_" + 
        result["month_start_date"].dt.strftime("%Y-%m")
    )
    
    return result


def main():
    print("="*80)
    print("XGBoost Corrosion Risk Prediction - FULL 14k predictions")
    print("="*80)
    
    # Load data
    print("\n[1/6] Loading data...")
    corr_df = load_corrosion_data()
    env_df = load_environment_data()
    
    # Create training pairs
    print("\n[2/6] Creating training pairs...")
    pairs_df = create_training_pairs(corr_df, env_df)
    
    # Prepare features
    feature_cols = ['reference_month'] + [c for c in pairs_df.columns if '__' in c and c != 'aircraft_id']
    X = pairs_df[feature_cols].values
    y = pairs_df['corrosion_risk'].values
    groups = pairs_df['aircraft_id'].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Cross-validation
    print("\n[3/6] Cross-validation with XGBoost...")
    n_splits = 5
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    aucs = []
    briers = []
    aps = []
    
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), 1):
        print(f"  Fold {fold}/{n_splits}...", end=" ")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            random_state=RANDOM_SEED,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(X_train_scaled, y_train, verbose=False)
        
        # Predict
        y_pred = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred)
        
        aucs.append(auc)
        briers.append(brier)
        aps.append(ap)
        
        print(f"AUC={auc:.4f}, Brier={brier:.4f}, AP={ap:.4f}")
    
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION RESULTS:")
    print(f"{'='*80}")
    print(f"AUC:            {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Brier Score:    {np.mean(briers):.4f} ± {np.std(briers):.4f}")
    print(f"Avg Precision:  {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"{'='*80}")
    
    # Train final model
    print("\n[4/6] Training final model on all data...")
    scaler_final = RobustScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    model_final = xgb.XGBClassifier(
        random_state=RANDOM_SEED,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model_final.fit(X_scaled, y, verbose=False)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Load test data
    print("\n[5/6] Loading FULL test data...")
    env_test_df = load_environment_data(ENV_TEST)
    
    print(f"\nEnvironment test data:")
    print(f"  - Total rows: {len(env_test_df)}")
    print(f"  - Aircraft: {env_test_df['aircraft_id'].nunique()}")
    
    # Prepare FULL test data
    test_df = prepare_full_test_data(env_test_df, features=ENV_FEATURES)
    
    print(f"\nTest predictions to generate:")
    print(f"  - Total predictions: {len(test_df)}")
    print(f"  - Aircraft: {test_df['aircraft_id'].nunique()}")
    print(f"  - Date range: {test_df['month_start_date'].min()} to {test_df['month_start_date'].max()}")
    
    print(f"\n[6/6] Making predictions for {len(test_df)} rows...")
    
    # Build feature matrix for test
    X_test_values = []
    
    # reference_month
    X_test_values.append(test_df["reference_month"].values)
    
    # Environmental features - match training structure
    for feat in feature_cols[1:]:  # Skip reference_month
        # Extract base feature name (remove __mean or __std suffix)
        base_feat = feat.replace("__mean", "").replace("__std", "")
        
        if base_feat in test_df.columns:
            X_test_values.append(test_df[base_feat].values)
        else:
            print(f"  WARNING: Missing feature {feat}, using zeros")
            X_test_values.append(np.zeros(len(test_df)))
    
    X_test = np.column_stack(X_test_values)
    
    # Handle any NaN values
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Scale using training statistics
    X_test_scaled = scaler_final.transform(X_test)
    
    # Predict
    y_pred_proba = model_final.predict_proba(X_test_scaled)[:, 1]
    
    # Clip to valid range
    y_pred_proba = np.clip(y_pred_proba, 1e-6, 1 - 1e-6)
    
    # Sanity checks
    print("\n  Sanity checks:")
    print(f"    - Predictions in [0,1]: {(y_pred_proba >= 0).all() and (y_pred_proba <= 1).all()}")
    print(f"    - Mean prediction: {y_pred_proba.mean():.4f}")
    print(f"    - Median prediction: {np.median(y_pred_proba):.4f}")
    print(f"    - Prediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    print(f"    - Std deviation: {y_pred_proba.std():.4f}")
    
    # Check monotonicity per aircraft
    test_df["corrosion_risk"] = y_pred_proba
    monotonic_checks = []
    for aircraft_id, group in test_df.groupby("aircraft_id"):
        if len(group) > 1:
            sorted_group = group.sort_values("reference_month")
            is_monotonic = sorted_group["corrosion_risk"].is_monotonic_increasing
            monotonic_checks.append(is_monotonic)
    
    if monotonic_checks:
        monotonic_pct = np.mean(monotonic_checks) * 100
        print(f"    - Aircraft with monotonic risk: {monotonic_pct:.1f}%")
    
    # Create submission
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'corrosion_risk': y_pred_proba
    })
    
    submission_df.to_csv('submission_xgboost.csv', index=False)
    
    print(f"\n✓ Predictions complete!")
    print(f"  Total predictions: {len(submission_df)}")
    print(f"  Unique aircraft: {test_df['aircraft_id'].nunique()}")
    print(f"  Prediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    print(f"  Mean prediction: {y_pred_proba.mean():.4f}")
    print(f"  Median prediction: {np.median(y_pred_proba):.4f}")
    print(f"\n✓ Submission saved to: submission_xgboost.csv")
    
    # Show sample predictions
    print("\n  Sample predictions:")
    print(submission_df.head(10).to_string(index=False))
    print("\n  Last predictions:")
    print(submission_df.tail(5).to_string(index=False))
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
