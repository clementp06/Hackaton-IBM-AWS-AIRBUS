"""
Shared data utilities for Airbus corrosion prediction models.

Implements the official Airbus convention:
- corrosion_risk = 1 at the month of corrosion observation
- corrosion_risk = 0 exactly 24 months before observation

This reflects the non-linear nature of corrosion progression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# Default data paths
DATA_DIR = Path(__file__).parent.parent / "data"
ENV_TRAIN_PATH = DATA_DIR / "environment_training.csv"
ENV_TEST_PATH = DATA_DIR / "environment_test.csv"
CORR_TRAIN_PATH = DATA_DIR / "corrosions_training.csv"

# Key environmental features (from feature importance analysis)
ENV_FEATURES = [
    "sea_salt_aerosol_5_20_mixing_ratio",
    "sea_salt_aerosol_05_5_mixing_ratio",
    "metar_relative_humidity",
    "metar_wind_speed_kn",
    "ozone_mass_mixing_ratio",
    "metar_temperature_c",
    "metar_dew_point_c",
    "sulphur_dioxide_mass_mixing_ratio",
    "nitrogen_dioxide_mass_mixing_ratio",
]


def load_corrosion_data(path: Path = CORR_TRAIN_PATH) -> pd.DataFrame:
    """
    Load corrosion training data and compute months since delivery.
    
    Returns:
        DataFrame with columns: aircraft_id, observation_date, 
        aircraft_delivery_year, aircraft_delivery_month, months_since_delivery
    """
    df = pd.read_csv(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    
    # Calculate months since delivery
    df["months_since_delivery"] = (
        (df["observation_date"].dt.year - df["aircraft_delivery_year"]) * 12
        + (df["observation_date"].dt.month - df["aircraft_delivery_month"])
    )
    
    return df


def load_environment_data(path: Path = ENV_TRAIN_PATH, test: bool = False) -> pd.DataFrame:
    """
    Load environmental data with timezone-naive timestamps.
    
    Args:
        path: Path to environment data file (default: training data)
        test: If True, load test data instead (default: False)
    
    Returns:
        DataFrame with environmental features per aircraft per month
    """
    if test:
        path = ENV_TEST_PATH
    df = pd.read_csv(path)
    df["month_start_date"] = pd.to_datetime(df["month_start_date"], utc=True).dt.tz_convert(None)
    return df


def aggregate_environmental_features(
    env_df: pd.DataFrame,
    features: list[str] = ENV_FEATURES,
    agg_funcs: list[str] = ["mean", "std"],
) -> pd.DataFrame:
    """
    Aggregate environmental features per aircraft.
    
    Args:
        env_df: Environmental data DataFrame
        features: List of feature columns to aggregate
        agg_funcs: Aggregation functions to apply
        
    Returns:
        DataFrame with aircraft_id and aggregated features
    """
    # Calculate corrosion exposure indices
    env_df = env_df.copy()
    
    # Alternative corrosion index using available data
    # Combines corrosive gases (SO2, NO2) with humidity and exposure time
    env_df["corrosion_exposure_index"] = (
        (env_df["sulphur_dioxide_mass_mixing_ratio"] + 
         env_df["nitrogen_dioxide_mass_mixing_ratio"]) * 
        env_df["metar_relative_humidity"] * 
        env_df["total_parking_minutes"] / 1000
    )
    
    # Moisture exposure index (humidity × parking time)
    env_df["moisture_exposure_index"] = (
        env_df["metar_relative_humidity"] * 
        env_df["total_parking_minutes"] / 100
    )
    
    # Add new indices to features list
    features_with_indices = list(features) + ["corrosion_exposure_index", "moisture_exposure_index"]
    
    # Filter to available features
    available_features = [f for f in features_with_indices if f in env_df.columns]
    
    if not available_features:
        raise ValueError(f"None of the requested features found in environment data")
    
    # Build aggregation dictionary
    agg_dict = {}
    for feat in available_features:
        for func in agg_funcs:
            agg_dict[f"{feat}__{func}"] = (feat, func)
    
    # Aggregate
    result = env_df.groupby("aircraft_id").agg(**agg_dict).reset_index()
    
    # Fill NaN with 0 (for std when only one observation)
    result = result.fillna(0)
    
    return result


def create_training_pairs(
    corr_df: pd.DataFrame,
    env_df: pd.DataFrame,
    reference_offset: int = 24,
    min_observation_month: int = 24,
    features: list[str] = ENV_FEATURES,
) -> pd.DataFrame:
    """
    Create training pairs following the Airbus convention.
    
    For each aircraft:
    - One sample at observation_month with corrosion_risk=1
    - One sample at (observation_month - reference_offset) with corrosion_risk=0
    
    Args:
        corr_df: Corrosion data (from load_corrosion_data)
        env_df: Environmental data (from load_environment_data)
        reference_offset: Months before observation for healthy reference (default: 24)
        min_observation_month: Minimum observation month to include (default: 24)
        features: Environmental features to include
        
    Returns:
        DataFrame with columns:
        - aircraft_id
        - reference_month (months since delivery)
        - corrosion_risk (0 or 1)
        - environmental features (aggregated)
    """
    # Filter aircraft with sufficient observation time
    valid_aircraft = corr_df[
        corr_df["months_since_delivery"] >= min_observation_month
    ].copy()
    
    print(f"Aircraft with observation >= {min_observation_month} months: {len(valid_aircraft)}")
    
    # Aggregate environmental features per aircraft
    env_agg = aggregate_environmental_features(env_df, features=features)
    
    # Create pairs
    pairs = []
    
    for _, row in valid_aircraft.iterrows():
        aircraft_id = row["aircraft_id"]
        obs_month = row["months_since_delivery"]
        
        # Get environmental features for this aircraft
        env_features = env_agg[env_agg["aircraft_id"] == aircraft_id]
        
        if env_features.empty:
            continue
        
        env_features = env_features.iloc[0].to_dict()
        
        # Pair 1: Corroded state (observation month)
        pair_1 = {
            "aircraft_id": aircraft_id,
            "reference_month": obs_month,
            "corrosion_risk": 1,
            **{k: v for k, v in env_features.items() if k != "aircraft_id"}
        }
        pairs.append(pair_1)
        
        # Pair 2: Healthy state (24 months before)
        pair_0 = {
            "aircraft_id": aircraft_id,
            "reference_month": obs_month - reference_offset,
            "corrosion_risk": 0,
            **{k: v for k, v in env_features.items() if k != "aircraft_id"}
        }
        pairs.append(pair_0)
    
    result = pd.DataFrame(pairs)
    
    print(f"Created {len(result)} training samples ({len(result)//2} aircraft pairs)")
    print(f"  - corrosion_risk=1: {(result['corrosion_risk']==1).sum()}")
    print(f"  - corrosion_risk=0: {(result['corrosion_risk']==0).sum()}")
    
    return result


def create_cv_splits(
    pairs_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create cross-validation splits at the aircraft level.
    
    Ensures that both samples from the same aircraft stay together in train or test.
    
    Args:
        pairs_df: Training pairs DataFrame
        n_splits: Number of CV folds
        random_state: Random seed
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Get unique aircraft
    unique_aircraft = pairs_df["aircraft_id"].unique()
    
    # Create KFold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_aircraft_idx, test_aircraft_idx in kf.split(unique_aircraft):
        train_aircraft = unique_aircraft[train_aircraft_idx]
        test_aircraft = unique_aircraft[test_aircraft_idx]
        
        # Get row indices for train and test aircraft
        train_idx = pairs_df[pairs_df["aircraft_id"].isin(train_aircraft)].index.values
        test_idx = pairs_df[pairs_df["aircraft_id"].isin(test_aircraft)].index.values
        
        splits.append((train_idx, test_idx))
    
    return splits


def prepare_test_data(
    env_test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    features: list[str] = ENV_FEATURES,
) -> pd.DataFrame:
    """
    Prepare test data for prediction.
    
    Args:
        env_test_df: Test environmental data
        sample_submission_df: Sample submission with query IDs
        features: Environmental features to include
        
    Returns:
        DataFrame with columns:
        - id (from sample submission)
        - aircraft_id
        - query_month (extracted from id)
        - environmental features (aggregated)
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
    
    # For test data, we need to calculate query_month relative to first observation
    # (since we don't have delivery dates in test set)
    first_obs = env_test_df.groupby("aircraft_id")["month_start_date"].min().reset_index()
    first_obs.columns = ["aircraft_id", "first_observation_date"]
    
    result = result.merge(first_obs, on="aircraft_id", how="left")
    
    result["query_month"] = (
        (result["query_date"].dt.year - result["first_observation_date"].dt.year) * 12
        + (result["query_date"].dt.month - result["first_observation_date"].dt.month)
    )
    
    return result


def validate_training_pairs(pairs_df: pd.DataFrame) -> dict:
    """
    Validate training pairs and return statistics.
    
    Args:
        pairs_df: Training pairs DataFrame
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        "n_aircraft": pairs_df["aircraft_id"].nunique(),
        "n_samples": len(pairs_df),
        "n_risk_1": (pairs_df["corrosion_risk"] == 1).sum(),
        "n_risk_0": (pairs_df["corrosion_risk"] == 0).sum(),
        "reference_month_min": pairs_df["reference_month"].min(),
        "reference_month_max": pairs_df["reference_month"].max(),
        "reference_month_mean": pairs_df["reference_month"].mean(),
        "reference_month_median": pairs_df["reference_month"].median(),
    }
    
    # Check for missing values
    feature_cols = [c for c in pairs_df.columns if c not in ["aircraft_id", "reference_month", "corrosion_risk"]]
    stats["n_missing_features"] = pairs_df[feature_cols].isna().sum().sum()
    
    # Check balance
    stats["class_balance"] = stats["n_risk_1"] / stats["n_samples"]
    
    return stats


if __name__ == "__main__":
    # Test the data utilities
    print("="*60)
    print("Testing data utilities")
    print("="*60)
    
    # Load data
    print("\n1. Loading corrosion data...")
    corr_df = load_corrosion_data()
    print(f"   Loaded {len(corr_df)} aircraft")
    print(f"   Months range: {corr_df['months_since_delivery'].min():.0f} - {corr_df['months_since_delivery'].max():.0f}")
    
    print("\n2. Loading environment data...")
    env_df = load_environment_data()
    print(f"   Loaded {len(env_df)} observations")
    print(f"   Unique aircraft: {env_df['aircraft_id'].nunique()}")
    
    print("\n3. Creating training pairs...")
    pairs_df = create_training_pairs(corr_df, env_df)
    
    print("\n4. Validating training pairs...")
    stats = validate_training_pairs(pairs_df)
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n5. Creating CV splits...")
    splits = create_cv_splits(pairs_df, n_splits=5)
    print(f"   Created {len(splits)} folds")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"   Fold {i+1}: train={len(train_idx)}, test={len(test_idx)}")
    
    print("\n" + "="*60)
    print("Data utilities test complete!")
    print("="*60)

# Made with Bob
