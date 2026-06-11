"""
TabPFN per-aircraft corrosion risk prediction using Airbus convention.
OPTIMIZED FOR BRIER SCORE MINIMIZATION.

Key optimizations:
1. Probability calibration (Platt scaling + Isotonic regression)
2. Extended environmental features (all available features)
3. Robust feature scaling
4. Ensemble of calibrated models
5. Optimized cross-validation strategy
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Import our data utilities
from data_utils import (
    load_corrosion_data,
    load_environment_data,
    create_training_pairs,
    create_cv_splits,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_PNG = "model_tabpfn_aircraft.png"
RANDOM_SEED = 42
MAX_AIRCRAFT_DISPLAY = 6  # Number of aircraft to show in plots

# ---------------------------------------------------------------------------
# TabPFN Integration with Calibration
# ---------------------------------------------------------------------------

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
    
    Args:
        X_train: Training features
        y_train: Training labels
        calibration_method: 'isotonic' or 'sigmoid' (Platt scaling)
        
    Returns:
        Calibrated TabPFN model or sklearn fallback
    """
    if check_tabpfn_available():
        try:
            from tabpfn import TabPFNClassifier
            
            # Base TabPFN model
            base_model = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
            
            # Calibrate using cross-validation
            # This is crucial for Brier score optimization
            calibrated_model = CalibratedClassifierCV(
                base_model,
                method=calibration_method,
                cv=3,  # Internal CV for calibration
                n_jobs=1
            )
            
            calibrated_model.fit(X_train, y_train)
            return calibrated_model
            
        except Exception as e:
            print(f"  TabPFN error: {e}, falling back to calibrated GradientBoosting")
    
    # Fallback to calibrated sklearn
    from sklearn.ensemble import GradientBoostingClassifier
    
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
    
    Returns:
        List of trained models
    """
    models = []
    
    # Model 1: Isotonic calibration
    print("    Training model with isotonic calibration...")
    model_iso = train_calibrated_tabpfn_model(X_train, y_train, calibration_method="isotonic")
    models.append(model_iso)
    
    # Model 2: Sigmoid calibration (Platt scaling)
    print("    Training model with sigmoid calibration...")
    model_sig = train_calibrated_tabpfn_model(X_train, y_train, calibration_method="sigmoid")
    models.append(model_sig)
    
    return models


def predict_ensemble(models: list[object], X: np.ndarray) -> np.ndarray:
    """
    Predict using ensemble averaging.
    
    Args:
        models: List of trained models
        X: Features to predict
        
    Returns:
        Averaged probabilities
    """
    predictions = []
    for model in models:
        pred = model.predict_proba(X)[:, 1]
        predictions.append(pred)
    
    # Average predictions
    return np.mean(predictions, axis=0)


# ---------------------------------------------------------------------------
# Per-Aircraft Analysis
# ---------------------------------------------------------------------------

@dataclass
class AircraftResult:
    """Results for one aircraft."""
    aircraft_id: str
    n_samples: int
    reference_months: np.ndarray
    actual_risk: np.ndarray
    predicted_risk: np.ndarray
    auc: float
    brier: float


def analyze_aircraft(
    aircraft_id: str,
    pairs_df: pd.DataFrame,
    feature_cols: list[str],
    use_ensemble: bool = True,
) -> AircraftResult | None:
    """
    Analyze one aircraft using its (1/0) pairs with calibrated predictions.
    """
    # Get data for this aircraft
    aircraft_data = pairs_df[pairs_df["aircraft_id"] == aircraft_id].copy()
    
    if len(aircraft_data) < 2:
        return None
    
    # Get data for other aircraft (for training)
    other_data = pairs_df[pairs_df["aircraft_id"] != aircraft_id].copy()
    
    if len(other_data) < 10:
        return None
    
    # Prepare features
    X_train = other_data[feature_cols].values
    y_train = other_data["corrosion_risk"].values
    
    X_test = aircraft_data[feature_cols].values
    y_test = aircraft_data["corrosion_risk"].values
    
    # Robust scaling (better for outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model(s)
    try:
        if use_ensemble:
            models = train_ensemble_models(X_train_scaled, y_train)
            y_pred = predict_ensemble(models, X_test_scaled)
        else:
            model = train_calibrated_tabpfn_model(X_train_scaled, y_train)
            y_pred = model.predict_proba(X_test_scaled)[:, 1]
    except Exception as e:
        print(f"  Error training model for {aircraft_id}: {e}")
        return None
    
    # Compute metrics
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred)
    else:
        auc = np.nan
    
    brier = brier_score_loss(y_test, y_pred)
    
    return AircraftResult(
        aircraft_id=aircraft_id,
        n_samples=len(aircraft_data),
        reference_months=aircraft_data["reference_month"].values,
        actual_risk=y_test,
        predicted_risk=y_pred,
        auc=auc,
        brier=brier,
    )


# ---------------------------------------------------------------------------
# Cross-Validation with Calibration
# ---------------------------------------------------------------------------

def cross_validate_calibrated_tabpfn(
    pairs_df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    use_ensemble: bool = True,
) -> dict:
    """
    Cross-validate calibrated TabPFN on the (1/0) pairs.
    
    Returns metrics aggregated across folds.
    """
    print("\n  Running cross-validation with calibrated models...")
    
    splits = create_cv_splits(pairs_df, n_splits=n_splits)
    
    aucs = []
    briers = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"    Fold {fold_idx + 1}/{n_splits}...")
        
        train_df = pairs_df.iloc[train_idx]
        test_df = pairs_df.iloc[test_idx]
        
        X_train = train_df[feature_cols].values
        y_train = train_df["corrosion_risk"].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df["corrosion_risk"].values
        
        # Robust scaling
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and predict
        try:
            if use_ensemble:
                models = train_ensemble_models(X_train_scaled, y_train)
                y_pred = predict_ensemble(models, X_test_scaled)
            else:
                model = train_calibrated_tabpfn_model(X_train_scaled, y_train)
                y_pred = model.predict_proba(X_test_scaled)[:, 1]
            
            auc = roc_auc_score(y_test, y_pred)
            brier = brier_score_loss(y_test, y_pred)
            
            aucs.append(auc)
            briers.append(brier)
            
            print(f"      AUC={auc:.4f}, Brier={brier:.4f}")
        except Exception as e:
            print(f"      Error: {e}")
    
    return {
        "auc_mean": np.mean(aucs) if aucs else np.nan,
        "auc_std": np.std(aucs) if aucs else np.nan,
        "brier_mean": np.mean(briers) if briers else np.nan,
        "brier_std": np.std(briers) if briers else np.nan,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    aircraft_results: list[AircraftResult],
    cv_results: dict,
    output_path: str = OUTPUT_PNG,
) -> None:
    """Create visualization of per-aircraft results."""
    # Select aircraft to display
    display_aircraft = aircraft_results[:MAX_AIRCRAFT_DISPLAY]
    
    n_aircraft = len(display_aircraft)
    n_cols = 3
    n_rows = (n_aircraft + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(18, 4 * n_rows + 2))
    fig.suptitle(
        f"Calibrated TabPFN Per-Aircraft Corrosion Risk Prediction\n"
        f"Cross-Validation: AUC = {cv_results['auc_mean']:.4f} ± {cv_results['auc_std']:.4f}, "
        f"Brier = {cv_results['brier_mean']:.4f} ± {cv_results['brier_std']:.4f}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    for idx, result in enumerate(display_aircraft):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Sort by reference month
        sort_idx = np.argsort(result.reference_months)
        months = result.reference_months[sort_idx]
        actual = result.actual_risk[sort_idx]
        predicted = result.predicted_risk[sort_idx]
        
        # Plot
        ax.scatter(months, actual, s=100, alpha=0.6, color="steelblue", 
                  label="Actual", zorder=3, marker='o')
        ax.scatter(months, predicted, s=100, alpha=0.6, color="darkorange",
                  label="Calibrated Prediction", zorder=3, marker='s')
        
        # Connect points
        if len(months) == 2:
            ax.plot(months, actual, 'b--', alpha=0.3, linewidth=1)
            ax.plot(months, predicted, 'r--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel("Reference Month (since delivery)", fontsize=9)
        ax.set_ylabel("Corrosion Risk", fontsize=9)
        ax.set_title(
            f"Aircraft {result.aircraft_id[:8]}\n"
            f"Brier = {result.brier:.4f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Calibrated TabPFN Per-Aircraft Corrosion Risk Prediction")
    print("OPTIMIZED FOR BRIER SCORE MINIMIZATION")
    print("=" * 80)
    
    # Check TabPFN availability
    if not check_tabpfn_available():
        print("\nWARNING: TabPFN not available. Using calibrated GradientBoosting as fallback.")
        print("To install TabPFN: pip install tabpfn")
    
    # Load data
    print("\n[1/4] Loading data...")
    corr_df = load_corrosion_data()
    env_df = load_environment_data()
    
    # Create training pairs
    print("\n[2/4] Creating training pairs (Airbus convention)...")
    pairs_df = create_training_pairs(corr_df, env_df)
    
    print(f"\nTraining data:")
    print(f"  - Aircraft: {pairs_df['aircraft_id'].nunique()}")
    print(f"  - Samples: {len(pairs_df)}")
    print(f"  - Reference month range: {pairs_df['reference_month'].min():.0f} - {pairs_df['reference_month'].max():.0f}")
    
    # Prepare features - USE ALL AVAILABLE ENVIRONMENTAL FEATURES
    feature_cols = ["reference_month"]
    env_feature_cols = [c for c in pairs_df.columns if "__" in c and c != "aircraft_id"]
    feature_cols.extend(env_feature_cols)  # Use ALL environmental features
    
    print(f"  - Features: {len(feature_cols)} (including all environmental features)")
    
    # Cross-validation with ensemble
    print("\n[3/4] Cross-validation with calibrated ensemble...")
    cv_results = cross_validate_calibrated_tabpfn(
        pairs_df, 
        feature_cols, 
        n_splits=5,
        use_ensemble=True
    )
    
    print(f"\n  Results:")
    print(f"    AUC:   {cv_results['auc_mean']:.4f} ± {cv_results['auc_std']:.4f}")
    print(f"    Brier: {cv_results['brier_mean']:.4f} ± {cv_results['brier_std']:.4f}")
    
    # Per-aircraft analysis
    print(f"\n[4/4] Analyzing individual aircraft (showing {MAX_AIRCRAFT_DISPLAY})...")
    
    aircraft_ids = pairs_df["aircraft_id"].unique()[:MAX_AIRCRAFT_DISPLAY]
    aircraft_results = []
    
    for aircraft_id in aircraft_ids:
        result = analyze_aircraft(aircraft_id, pairs_df, feature_cols, use_ensemble=False)
        if result:
            aircraft_results.append(result)
            print(f"  {aircraft_id[:8]}: Brier={result.brier:.4f}")
    
    # Plot results
    if aircraft_results:
        print("\nGenerating plots...")
        plot_results(aircraft_results, cv_results)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Cross-validation AUC: {cv_results['auc_mean']:.4f}")
    print(f"  - Cross-validation Brier: {cv_results['brier_mean']:.4f}")
    print(f"  - Aircraft analyzed: {len(aircraft_results)}")
    print(f"\nOptimizations applied:")
    print(f"  ✓ Probability calibration (isotonic + sigmoid)")
    print(f"  ✓ Ensemble averaging")
    print(f"  ✓ Robust feature scaling")
    print(f"  ✓ All environmental features used")


if __name__ == "__main__":
    main()

# Made with Bob