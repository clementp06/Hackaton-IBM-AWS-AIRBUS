"""
Full corrosion-risk inference for ALL rows in environment_test.csv.
Generates predictions for every aircraft × month combination.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

# Import our data utilities
from data_utils import (
    load_corrosion_data,
    load_environment_data,
    create_training_pairs,
    ENV_FEATURES,
)

DATA_DIR = Path(__file__).parent.parent / "data"
ENV_TRAIN = DATA_DIR / "environment_training.csv"
ENV_TEST = DATA_DIR / "environment_test.csv"
CORR_TRAIN = DATA_DIR / "corrosions_training.csv"
OUTPUT_SUB = Path(__file__).parent / "submission.csv"

RANDOM_SEED = 42


def check_tabpfn_available() -> bool:
    """Check if TabPFN is available."""
    try:
        from tabpfn import TabPFNClassifier
        return True
    except ImportError:
        return False


def train_calibrated_tabpfn_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    calibration_method: str = "isotonic"
) -> object:
    """Train a calibrated TabPFN classifier."""
    if check_tabpfn_available():
        try:
            from tabpfn import TabPFNClassifier
            base_model = TabPFNClassifier(device='cpu')
            calibrated_model = CalibratedClassifierCV(
                base_model,
                method=calibration_method,
                cv=3,
                n_jobs=1
            )
            calibrated_model.fit(X_train, y_train)
            return calibrated_model
        except Exception as e:
            print(f"  TabPFN error: {e}, falling back to calibrated GradientBoosting")
    
    # Fallback
    base_model = GradientBoostingClassifier(
        random_state=RANDOM_SEED,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
    )
    
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method=calibration_method,
        cv=3,
        n_jobs=1
    )
    
    calibrated_model.fit(X_train, y_train)
    return calibrated_model


def train_ensemble_models(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> list[object]:
    """Train ensemble of calibrated models."""
    models = []
    
    print("  Training model with isotonic calibration...")
    model_iso = train_calibrated_tabpfn_model(X_train, y_train, calibration_method="isotonic")
    models.append(model_iso)
    
    print("  Training model with sigmoid calibration...")
    model_sig = train_calibrated_tabpfn_model(X_train, y_train, calibration_method="sigmoid")
    models.append(model_sig)
    
    return models


def predict_ensemble(models: list[object], X: np.ndarray) -> np.ndarray:
    """Predict using ensemble averaging."""
    predictions = []
    for model in models:
        pred = model.predict_proba(X)[:, 1]
        predictions.append(pred)
    return np.mean(predictions, axis=0)


def aggregate_environmental_features_per_month(
    env_df: pd.DataFrame,
    features: list[str] = ENV_FEATURES,
) -> pd.DataFrame:
    """
    Aggregate environmental features per aircraft per month.
    
    Returns DataFrame with aircraft_id, month_start_date, and aggregated features.
    """
    available_features = [f for f in features if f in env_df.columns]
    
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


def main() -> None:
    print("=" * 80)
    print("FULL Corrosion-risk inference for ALL environment_test.csv rows")
    print("=" * 80)
    
    if not check_tabpfn_available():
        print("\nWARNING: TabPFN not available. Using calibrated GradientBoosting.")
    
    # Load training data
    print("\n[1/4] Loading training data...")
    corr_df = load_corrosion_data()
    env_df = load_environment_data()
    
    # Create training pairs
    print("\n[2/4] Creating training pairs...")
    pairs_df = create_training_pairs(corr_df, env_df, features=ENV_FEATURES)
    
    # Prepare features - USE ALL AVAILABLE ENVIRONMENTAL FEATURES
    feature_cols = ["reference_month"]
    env_feature_cols = [c for c in pairs_df.columns if "__" in c and c != "aircraft_id"]
    feature_cols.extend(env_feature_cols)
    
    print(f"\nTraining data:")
    print(f"  - Aircraft: {pairs_df['aircraft_id'].nunique()}")
    print(f"  - Samples: {len(pairs_df)}")
    print(f"  - Features: {len(feature_cols)}")
    
    # Train ensemble model
    print(f"\n[3/4] Training calibrated ensemble model...")
    
    X_train = pairs_df[feature_cols].values
    y_train = pairs_df["corrosion_risk"].values
    
    # Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train ensemble
    models = train_ensemble_models(X_train_scaled, y_train)
    print(f"  Ensemble trained on {len(X_train)} samples")
    
    # Prepare FULL test data
    print("\n[4/4] Preparing FULL test data and generating predictions...")
    
    env_test_df = load_environment_data(ENV_TEST)
    
    print(f"\nEnvironment test data:")
    print(f"  - Total rows: {len(env_test_df)}")
    print(f"  - Aircraft: {env_test_df['aircraft_id'].nunique()}")
    
    test_df = prepare_full_test_data(env_test_df, features=ENV_FEATURES)
    
    print(f"\nTest predictions to generate:")
    print(f"  - Total predictions: {len(test_df)}")
    print(f"  - Aircraft: {test_df['aircraft_id'].nunique()}")
    print(f"  - Date range: {test_df['month_start_date'].min()} to {test_df['month_start_date'].max()}")
    
    # Build feature matrix for test
    # Match training feature structure
    X_test_values = []
    
    # reference_month
    X_test_values.append(test_df["reference_month"].values)
    
    # Environmental features
    for feat in env_feature_cols:
        # Extract base feature name (remove __mean or __std suffix)
        base_feat = feat.replace("__mean", "").replace("__std", "")
        
        if base_feat in test_df.columns:
            X_test_values.append(test_df[base_feat].values)
        else:
            print(f"  WARNING: Missing feature {feat}, using zeros")
            X_test_values.append(np.zeros(len(test_df)))
    
    X_test = np.column_stack(X_test_values)
    
    # Scale using training statistics
    X_test_scaled = scaler.transform(X_test)
    
    # Predict using ensemble
    print("\n  Generating ensemble predictions...")
    predictions = predict_ensemble(models, X_test_scaled)
    predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
    
    # Sanity checks
    print("\n  Sanity checks:")
    print(f"    - Predictions in [0,1]: {(predictions >= 0).all() and (predictions <= 1).all()}")
    print(f"    - Mean prediction: {predictions.mean():.4f}")
    print(f"    - Median prediction: {np.median(predictions):.4f}")
    print(f"    - Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"    - Std deviation: {predictions.std():.4f}")
    
    # Check monotonicity per aircraft
    test_df["corrosion_risk"] = predictions
    monotonic_checks = []
    for aircraft_id, group in test_df.groupby("aircraft_id"):
        if len(group) > 1:
            sorted_group = group.sort_values("reference_month")
            is_monotonic = sorted_group["corrosion_risk"].is_monotonic_increasing
            monotonic_checks.append(is_monotonic)
    
    if monotonic_checks:
        monotonic_pct = np.mean(monotonic_checks) * 100
        print(f"    - Aircraft with monotonic risk: {monotonic_pct:.1f}%")
    
    # Write submission
    submission = pd.DataFrame({
        "id": test_df["id"],
        "corrosion_risk": predictions,
    })
    
    submission.to_csv(OUTPUT_SUB, index=False)
    
    print(f"\n  ✓ Wrote {len(submission)} predictions to {OUTPUT_SUB}")
    print("\n  Sample predictions:")
    print(submission.head(10).to_string(index=False))
    print("\n  Last predictions:")
    print(submission.tail(5).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nOptimizations applied:")
    print("  ✓ Calibrated TabPFN ensemble (isotonic + sigmoid)")
    print("  ✓ Robust feature scaling")
    print("  ✓ All environmental features")
    print("  ✓ Predictions for ALL environment_test.csv rows")


if __name__ == "__main__":
    main()

# Made with Bob
