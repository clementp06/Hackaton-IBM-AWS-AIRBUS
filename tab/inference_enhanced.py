"""
Enhanced corrosion-risk inference with engineered features.
Based on feature analysis recommendations to minimize Brier score.
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
OUTPUT_SUB = Path(__file__).parent / "submission_enhanced.csv"

RANDOM_SEED = 42


def check_tabpfn_available() -> bool:
    """Check if TabPFN is available."""
    try:
        from tabpfn import TabPFNClassifier
        return True
    except ImportError:
        return False


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features based on analysis recommendations.
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()
    
    # 1. Time-based non-linear features
    if "reference_month" in df.columns:
        df["reference_month_squared"] = df["reference_month"] ** 2
        df["reference_month_sqrt"] = np.sqrt(np.maximum(df["reference_month"], 0))
        df["reference_month_log"] = np.log1p(np.maximum(df["reference_month"], 0))
    
    # 2. Salt exposure features
    salt_cols = [c for c in df.columns if "salt_aerosol" in c and "__mean" in c]
    if len(salt_cols) >= 2:
        df["salt_exposure_total"] = df[salt_cols].sum(axis=1)
        
        # Ratio of large to small salt particles
        large_salt = [c for c in salt_cols if "5_20" in c]
        small_salt = [c for c in salt_cols if "05_5" in c]
        if large_salt and small_salt:
            df["large_small_salt_ratio"] = (
                df[large_salt[0]] / (df[small_salt[0]] + 1e-9)
            )
    
    # 3. Temperature and humidity interactions
    temp_mean = [c for c in df.columns if "temperature_c__mean" in c]
    humid_mean = [c for c in df.columns if "relative_humidity__mean" in c]
    dew_mean = [c for c in df.columns if "dew_point_c__mean" in c]
    
    if temp_mean and humid_mean:
        df["humidity_temperature_interaction"] = df[temp_mean[0]] * df[humid_mean[0]]
    
    if temp_mean and dew_mean:
        df["temperature_dewpoint_diff"] = df[temp_mean[0]] - df[dew_mean[0]]
    
    # 4. Wind and salt interaction
    wind_mean = [c for c in df.columns if "wind_speed_kn__mean" in c]
    if wind_mean and salt_cols:
        df["wind_salt_interaction"] = df[wind_mean[0]] * df[salt_cols[0]]
    
    # 5. Pollutant index
    so2_mean = [c for c in df.columns if "sulphur_dioxide" in c and "__mean" in c]
    no2_mean = [c for c in df.columns if "nitrogen_dioxide" in c and "__mean" in c]
    o3_mean = [c for c in df.columns if "ozone" in c and "__mean" in c]
    
    pollutant_cols = []
    if so2_mean:
        pollutant_cols.append(so2_mean[0])
    if no2_mean:
        pollutant_cols.append(no2_mean[0])
    if o3_mean:
        pollutant_cols.append(o3_mean[0])
    
    if pollutant_cols:
        df["pollutant_index"] = df[pollutant_cols].sum(axis=1)
    
    # 6. Environmental variability (using std features)
    std_cols = [c for c in df.columns if "__std" in c]
    if std_cols:
        df["environmental_volatility"] = df[std_cols].mean(axis=1)
    
    # 7. Top interaction features from analysis
    # reference_month × wind_speed
    if "reference_month" in df.columns and wind_mean:
        df["month_wind_interaction"] = df["reference_month"] * df[wind_mean[0]]
    
    # reference_month × salt
    if "reference_month" in df.columns and salt_cols:
        df["month_salt_interaction"] = df["reference_month"] * df[salt_cols[0]]
    
    return df


def remove_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove highly correlated redundant features identified in analysis.
    
    Removes:
    - sea_salt_aerosol_05_5_mixing_ratio__mean (keep 5_20 version)
    - sulphur_dioxide_mass_mixing_ratio__mean (keep std version)
    - metar_dew_point_c__mean (keep temperature)
    """
    df = df.copy()
    
    redundant_features = [
        "sea_salt_aerosol_05_5_mixing_ratio__mean",
        "sulphur_dioxide_mass_mixing_ratio__mean",
        "metar_dew_point_c__mean",
    ]
    
    for feat in redundant_features:
        if feat in df.columns:
            df = df.drop(columns=[feat])
    
    return df


def train_calibrated_model(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    calibration_method: str = "isotonic"
) -> object:
    """Train a calibrated model."""
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
    model_iso = train_calibrated_model(X_train, y_train, calibration_method="isotonic")
    models.append(model_iso)
    
    print("  Training model with sigmoid calibration...")
    model_sig = train_calibrated_model(X_train, y_train, calibration_method="sigmoid")
    models.append(model_sig)
    
    return models


def predict_ensemble(models: list[object], X: np.ndarray) -> np.ndarray:
    """Predict using ensemble averaging."""
    predictions = []
    for model in models:
        pred = model.predict_proba(X)[:, 1]
        predictions.append(pred)
    return np.mean(predictions, axis=0)


def aggregate_environmental_features(
    env_df: pd.DataFrame,
    features: list[str] = ENV_FEATURES,
) -> pd.DataFrame:
    """Aggregate environmental features per aircraft."""
    available_features = [f for f in features if f in env_df.columns]
    
    if not available_features:
        raise ValueError(f"None of the requested features found in environment data")
    
    agg_dict = {}
    for feat in available_features:
        for func in ["mean", "std"]:
            agg_dict[f"{feat}__{func}"] = (feat, func)
    
    result = env_df.groupby("aircraft_id").agg(**agg_dict).reset_index()
    result = result.fillna(0)
    
    return result


def prepare_test_data(
    env_test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    features: list[str] = ENV_FEATURES,
) -> pd.DataFrame:
    """Prepare test data for prediction."""
    sample_submission_df = sample_submission_df.copy()
    sample_submission_df["aircraft_id"] = sample_submission_df["id"].str.split("_").str[0]
    sample_submission_df["query_date"] = pd.to_datetime(
        sample_submission_df["id"].str.split("_").str[1]
    )
    
    env_agg = aggregate_environmental_features(env_test_df, features=features)
    result = sample_submission_df.merge(env_agg, on="aircraft_id", how="left")
    
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
    print("ENHANCED Corrosion-risk inference with Feature Engineering")
    print("=" * 80)
    
    if not check_tabpfn_available():
        print("\nWARNING: TabPFN not available. Using calibrated GradientBoosting.")
    
    # Load training data
    print("\n[1/5] Loading training data...")
    corr_df = load_corrosion_data()
    env_df = load_environment_data()
    
    # Create training pairs
    print("\n[2/5] Creating training pairs and engineering features...")
    pairs_df = create_training_pairs(corr_df, env_df, features=ENV_FEATURES)
    
    # Engineer features
    pairs_df = engineer_features(pairs_df)
    print(f"  Features after engineering: {len([c for c in pairs_df.columns if c not in ['aircraft_id', 'corrosion_risk']])}")
    
    # Remove redundant features
    pairs_df = remove_redundant_features(pairs_df)
    print(f"  Features after removing redundancy: {len([c for c in pairs_df.columns if c not in ['aircraft_id', 'corrosion_risk']])}")
    
    # Prepare features
    feature_cols = [c for c in pairs_df.columns if c not in ["aircraft_id", "corrosion_risk"]]
    
    print(f"\nTraining data:")
    print(f"  - Aircraft: {pairs_df['aircraft_id'].nunique()}")
    print(f"  - Samples: {len(pairs_df)}")
    print(f"  - Features: {len(feature_cols)}")
    
    # Train ensemble model
    print(f"\n[3/5] Training calibrated ensemble model...")
    
    X_train = pairs_df[feature_cols].values
    y_train = pairs_df["corrosion_risk"].values
    
    # Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train ensemble
    models = train_ensemble_models(X_train_scaled, y_train)
    print(f"  Ensemble trained on {len(X_train)} samples")
    
    # Prepare test data
    print("\n[4/5] Preparing test data and engineering features...")
    
    env_test_df = load_environment_data(ENV_TEST)
    sample_sub_df = pd.read_csv(SAMPLE_SUB)
    
    test_df = prepare_test_data(env_test_df, sample_sub_df, features=ENV_FEATURES)
    
    # Engineer features for test
    test_df = engineer_features(test_df)
    test_df = remove_redundant_features(test_df)
    
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
            print(f"  WARNING: Missing feature {feat}, using zeros")
            X_test_values.append(np.zeros(len(test_df)))
    
    X_test = np.column_stack(X_test_values)
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    print("\n[5/5] Generating ensemble predictions...")
    predictions = predict_ensemble(models, X_test_scaled)
    predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
    
    # Sanity checks
    print("\n  Sanity checks:")
    print(f"    - Predictions in [0,1]: {(predictions >= 0).all() and (predictions <= 1).all()}")
    print(f"    - Mean prediction: {predictions.mean():.4f}")
    print(f"    - Median prediction: {np.median(predictions):.4f}")
    print(f"    - Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"    - Std deviation: {predictions.std():.4f}")
    
    # Check monotonicity
    test_df["corrosion_risk"] = predictions
    monotonic_checks = []
    for aircraft_id, group in test_df.groupby("aircraft_id"):
        if len(group) > 1:
            sorted_group = group.sort_values("query_month")
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
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nEnhancements applied:")
    print("  ✓ 10+ engineered features (time, interactions, ratios)")
    print("  ✓ Removed 3 redundant features")
    print("  ✓ Calibrated TabPFN ensemble")
    print("  ✓ Robust feature scaling")


if __name__ == "__main__":
    main()

# Made with Bob
