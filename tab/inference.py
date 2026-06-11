"""
Corrosion-risk inference on the test set using OPTIMIZED TabPFN with calibration.

Uses the calibrated ensemble approach from model_tabpfn_aircraft.py
to minimize Brier score on test predictions.
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
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"
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
    """
    Train a calibrated TabPFN classifier for optimal Brier score.
    """
    if check_tabpfn_available():
        try:
            from tabpfn import TabPFNClassifier
            
            # Base TabPFN model
            base_model = TabPFNClassifier(device='cpu')
            
            # Calibrate using cross-validation
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
    
    # Fallback to calibrated sklearn
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
    """
    Train an ensemble of calibrated models with different calibration methods.
    """
    models = []
    
    # Model 1: Isotonic calibration
    print("  Training model with isotonic calibration...")
    model_iso = train_calibrated_tabpfn_model(X_train, y_train, calibration_method="isotonic")
    models.append(model_iso)
    
    # Model 2: Sigmoid calibration (Platt scaling)
    print("  Training model with sigmoid calibration...")
    model_sig = train_calibrated_tabpfn_model(X_train, y_train, calibration_method="sigmoid")
    models.append(model_sig)
    
    return models


def predict_ensemble(models: list[object], X: np.ndarray) -> np.ndarray:
    """
    Predict using ensemble averaging.
    """
    predictions = []
    for model in models:
        pred = model.predict_proba(X)[:, 1]
        predictions.append(pred)
    
    # Average predictions
    return np.mean(predictions, axis=0)


def aggregate_environmental_features(
    env_df: pd.DataFrame,
    features: list[str] = ENV_FEATURES,
) -> pd.DataFrame:
    """
    Aggregate environmental features per aircraft.
    """
    available_features = [f for f in features if f in env_df.columns]
    
    if not available_features:
        raise ValueError(f"None of the requested features found in environment data")
    
    # Build aggregation dictionary
    agg_dict = {}
    for feat in available_features:
        for func in ["mean", "std"]:
            agg_dict[f"{feat}__{func}"] = (feat, func)
    
    # Aggregate
    result = env_df.groupby("aircraft_id").agg(**agg_dict).reset_index()
    
    # Fill NaN with 0
    result = result.fillna(0)
    
    return result


def prepare_test_data(
    env_test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    features: list[str] = ENV_FEATURES,
) -> pd.DataFrame:
    """
    Prepare test data for prediction.
    """
    # Parse sample submission IDs
    sample_submission_df = sample_submission_df.copy()
    sample_submission_df["aircraft_id"] = sample_submission_df["id"].str.split("_").str[0]
    sample_submission_df["query_date"] = pd.to_datetime(
        sample_submission_df["id"].str.split("_").str[1]
    )
    
    # Aggregate environmental features
    env_agg = aggregate_environmental_features(env_test_df, features=features)
    
    # Merge
    result = sample_submission_df.merge(env_agg, on="aircraft_id", how="left")
    
    # Calculate query_month relative to first observation
    first_obs = env_test_df.groupby("aircraft_id")["month_start_date"].min().reset_index()
    first_obs.columns = ["aircraft_id", "first_observation_date"]
    
    result = result.merge(first_obs, on="aircraft_id", how="left")
    
    result["query_month"] = (
        (result["query_date"].dt.year - result["first_observation_date"].dt.year) * 12
        + (result["query_date"].dt.month - result["first_observation_date"].dt.month)
    )
    
    return result


def main() -> None:
    print("=" * 80)
    print("Corrosion-risk inference — OPTIMIZED TabPFN with Calibration")
    print("=" * 80)
    
    # Check TabPFN availability
    if not check_tabpfn_available():
        print("\nWARNING: TabPFN not available. Using calibrated GradientBoosting as fallback.")
        print("To install TabPFN: pip install tabpfn")
    
    # --- Load training data ---
    print("\n[1/4] Loading training data...")
    corr_df = load_corrosion_data()
    env_df = load_environment_data()
    
    # --- Create training pairs ---
    print("\n[2/4] Creating training pairs (Airbus convention)...")
    pairs_df = create_training_pairs(corr_df, env_df, features=ENV_FEATURES)
    
    print(f"\nTraining data:")
    print(f"  - Aircraft: {pairs_df['aircraft_id'].nunique()}")
    print(f"  - Samples: {len(pairs_df)}")
    print(f"  - Reference month range: {pairs_df['reference_month'].min():.0f} - {pairs_df['reference_month'].max():.0f}")
    
    # Prepare features - USE ALL AVAILABLE ENVIRONMENTAL FEATURES
    feature_cols = ["reference_month"]
    env_feature_cols = [c for c in pairs_df.columns if "__" in c and c != "aircraft_id"]
    feature_cols.extend(env_feature_cols)
    
    print(f"  - Features: {len(feature_cols)} (including all environmental features)")
    
    # --- Train final ensemble model on all data ---
    print(f"\n[3/4] Training calibrated ensemble model on all data...")
    
    X_train = pairs_df[feature_cols].values
    y_train = pairs_df["corrosion_risk"].values
    
    # Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train ensemble
    models = train_ensemble_models(X_train_scaled, y_train)
    
    print(f"  Ensemble trained on {len(X_train)} samples")
    
    # --- Prepare test data and predict ---
    print("\n[4/4] Preparing test data and generating predictions...")
    
    # Load test data
    env_test_df = load_environment_data(ENV_TEST)
    sample_sub_df = pd.read_csv(SAMPLE_SUB)
    
    # Prepare test features
    test_df = prepare_test_data(env_test_df, sample_sub_df, features=ENV_FEATURES)
    
    print(f"\nTest data:")
    print(f"  - Queries: {len(test_df)}")
    print(f"  - Aircraft: {test_df['aircraft_id'].nunique()}")
    
    # Build feature matrix for test
    X_test_values = []
    for feat in feature_cols:
        if feat == "reference_month":
            X_test_values.append(test_df["query_month"].values)
        elif feat in test_df.columns:
            X_test_values.append(test_df[feat].values)
        else:
            # Missing feature, use zeros
            print(f"  WARNING: Missing feature {feat}, using zeros")
            X_test_values.append(np.zeros(len(test_df)))
    
    X_test = np.column_stack(X_test_values)
    
    # Scale using training statistics
    X_test_scaled = scaler.transform(X_test)
    
    # Predict using ensemble
    print("\n  Generating ensemble predictions...")
    predictions = predict_ensemble(models, X_test_scaled)
    
    # Clip to valid range
    predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
    
    # --- Sanity checks ---
    print("\n  Sanity checks:")
    print(f"    - Predictions in [0,1]: {(predictions >= 0).all() and (predictions <= 1).all()}")
    print(f"    - Mean prediction: {predictions.mean():.4f}")
    print(f"    - Median prediction: {np.median(predictions):.4f}")
    print(f"    - Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"    - Std deviation: {predictions.std():.4f}")
    
    # Check monotonicity per aircraft (older age => higher risk)
    test_df["corrosion_risk"] = predictions
    monotonic_checks = []
    for aircraft_id, group in test_df.groupby("aircraft_id"):
        if len(group) > 1:
            sorted_group = group.sort_values("query_month")
            is_monotonic = sorted_group["corrosion_risk"].is_monotonic_increasing
            monotonic_checks.append(is_monotonic)
    
    if monotonic_checks:
        monotonic_pct = np.mean(monotonic_checks) * 100
        print(f"    - Aircraft with monotonic risk (older => higher): {monotonic_pct:.1f}%")
    
    # --- Write submission ---
    submission = pd.DataFrame({
        "id": test_df["id"],
        "corrosion_risk": predictions,
    })
    
    submission.to_csv(OUTPUT_SUB, index=False)
    
    print(f"\n  ✓ Wrote {len(submission)} predictions to {OUTPUT_SUB}")
    print("\n  Sample predictions:")
    print(submission.head(10).to_string(index=False))
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nOptimizations applied:")
    print("  ✓ Calibrated TabPFN ensemble (isotonic + sigmoid)")
    print("  ✓ Robust feature scaling")
    print("  ✓ All environmental features")
    print("  ✓ Ensemble averaging for reduced variance")


if __name__ == "__main__":
    main()

# Made with Bob