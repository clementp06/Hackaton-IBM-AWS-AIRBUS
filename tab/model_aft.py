"""
Bayesian Binary Classification with Environmental Covariates
============================================================
Uses the Airbus convention (1/0 pairs) with environmental features.
Three experiments with different feature sets; plots saved to model_aft.png.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pymc as pm
import arviz as az
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

# Import our data utilities
from data_utils import (
    load_corrosion_data,
    load_environment_data,
    create_training_pairs,
    create_cv_splits,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Environmental features (from feature importance analysis)
FEATURES = [
    "sea_salt_aerosol_5_20_mixing_ratio",
    "metar_relative_humidity",
    "metar_wind_speed_kn",
    "ozone_mass_mixing_ratio",
]

FEATURE_LABELS = [
    "sea_salt",
    "rel_humidity",
    "wind_speed",
    "ozone",
]

# Experiments with different feature combinations
EXPERIMENTS = {
    "A — sea salt only": ["sea_salt"],
    "B — salt + humidity + wind": ["sea_salt", "rel_humidity", "wind_speed"],
    "C — all 4 features": FEATURE_LABELS,
}

RANDOM_SEED = 42
DRAWS = 2000
TUNE = 1000
TARGET_ACCEPT = 0.9

# ---------------------------------------------------------------------------
# Model Building
# ---------------------------------------------------------------------------


def build_logistic_model_with_covariates(
    X_month: np.ndarray,
    X_features: np.ndarray,
    y: np.ndarray,
) -> pm.Model:
    """
    Bayesian logistic regression with month + environmental covariates.
    
    Args:
        X_month: Reference month (standardized)
        X_features: Environmental features (standardized)
        y: Binary outcome (0 or 1)
    """
    n_features = X_features.shape[1]
    
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
        beta_month = pm.Normal("beta_month", mu=0.0, sigma=0.5)
        beta_features = pm.Normal("beta_features", mu=0.0, sigma=0.3, shape=n_features)
        
        # Linear predictor
        eta = intercept + beta_month * X_month + pm.math.dot(X_features, beta_features)
        
        # Logistic link
        p = pm.math.invlogit(eta)
        
        # Likelihood
        _obs = pm.Bernoulli("obs", p=p, observed=y)
    
    return model


def sample_model(model: pm.Model, model_name: str) -> az.InferenceData:
    """Sample from a PyMC model using NUTS."""
    with model:
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=False,
            return_inferencedata=True,
        )
        pm.compute_log_likelihood(idata)
    return idata


def predict_probability(
    idata: az.InferenceData,
    X_month: np.ndarray,
    X_features: np.ndarray,
) -> np.ndarray:
    """
    Predict probabilities using posterior samples.
    
    Returns mean predicted probability across posterior samples.
    """
    posterior = idata.posterior
    
    # Extract parameters
    intercept = posterior["intercept"].values.flatten()
    beta_month = posterior["beta_month"].values.flatten()
    beta_features = posterior["beta_features"].values.reshape(-1, X_features.shape[1])
    
    # Compute eta for each posterior sample
    eta = (
        intercept[:, None]
        + beta_month[:, None] * X_month[None, :]
        + beta_features @ X_features.T
    )
    
    # Apply logistic link
    p = 1.0 / (1.0 + np.exp(-eta))
    
    # Return mean across posterior samples
    return p.mean(axis=0)


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------


def cross_validate_experiment(
    pairs_df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
) -> dict:
    """
    Perform cross-validation for one experiment.
    
    Returns dictionary with metrics and feature importance.
    """
    # Prepare data
    X_month = pairs_df["reference_month"].values
    y = pairs_df["corrosion_risk"].values
    
    # Get feature columns (with aggregation suffix)
    feature_col_names = []
    for feat in feature_cols:
        # Find columns that start with this feature name
        matching_cols = [c for c in pairs_df.columns if c.startswith(f"{FEATURES[FEATURE_LABELS.index(feat)]}__")]
        feature_col_names.extend(matching_cols)
    
    X_features = pairs_df[feature_col_names].values
    
    # Standardize
    X_month_mean = X_month.mean()
    X_month_std = X_month.std()
    X_month_scaled = (X_month - X_month_mean) / X_month_std
    
    X_features_mean = X_features.mean(axis=0)
    X_features_std = X_features.std(axis=0)
    X_features_std[X_features_std == 0] = 1.0  # Avoid division by zero
    X_features_scaled = (X_features - X_features_mean) / X_features_std
    
    splits = create_cv_splits(pairs_df, n_splits=n_splits)
    
    aucs = []
    briers = []
    logloss_scores = []
    beta_features_samples = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"      Fold {fold_idx + 1}/{n_splits}...", end=" ")
        
        X_month_train = X_month_scaled[train_idx]
        X_features_train = X_features_scaled[train_idx]
        y_train = y[train_idx]
        
        X_month_test = X_month_scaled[test_idx]
        X_features_test = X_features_scaled[test_idx]
        y_test = y[test_idx]
        
        # Build and sample model
        model = build_logistic_model_with_covariates(
            X_month_train, X_features_train, y_train
        )
        idata = sample_model(model, f"Fold {fold_idx + 1}")
        
        # Predict on test set
        y_pred = predict_probability(idata, X_month_test, X_features_test)
        
        # Compute metrics
        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        ll = log_loss(y_test, y_pred)
        
        aucs.append(auc)
        briers.append(brier)
        logloss_scores.append(ll)
        
        # Store beta_features for feature importance
        beta_features_samples.append(idata.posterior["beta_features"].values.reshape(-1, len(feature_col_names)))
        
        print(f"AUC: {auc:.4f}, Brier: {brier:.4f}")
    
    # Aggregate beta_features across folds
    beta_features_all = np.concatenate(beta_features_samples, axis=0)
    
    return {
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "brier_mean": np.mean(briers),
        "brier_std": np.std(briers),
        "logloss_mean": np.mean(logloss_scores),
        "logloss_std": np.std(logloss_scores),
        "beta_features_mean": beta_features_all.mean(axis=0),
        "beta_features_std": beta_features_all.std(axis=0),
        "feature_names": feature_col_names,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    pairs_df: pd.DataFrame,
    results: dict,
    output_path: str = "model_aft.png",
) -> None:
    """Create visualization of model results."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Bayesian Binary Classification with Environmental Covariates\n"
        "Corrosion Risk Prediction (Airbus Convention)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
    
    # Panel 1: Cross-validation metrics comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    exp_names = list(EXPERIMENTS.keys())
    x_pos = np.arange(len(exp_names))
    
    aucs = [results[exp]["auc_mean"] for exp in exp_names]
    auc_stds = [results[exp]["auc_std"] for exp in exp_names]
    
    ax1.bar(x_pos, aucs, yerr=auc_stds, capsize=5, color=["#2196F3", "#4CAF50", "#FF5722"], alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([exp.split("—")[0].strip() for exp in exp_names], fontsize=9)
    ax1.set_ylabel("AUC-ROC")
    ax1.set_title("Cross-Validation Performance (5-fold)")
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(axis="y", alpha=0.3)
    
    # Panel 2: Brier score comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    briers = [results[exp]["brier_mean"] for exp in exp_names]
    brier_stds = [results[exp]["brier_std"] for exp in exp_names]
    
    ax2.bar(x_pos, briers, yerr=brier_stds, capsize=5, color=["#2196F3", "#4CAF50", "#FF5722"], alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([exp.split("—")[0].strip() for exp in exp_names], fontsize=9)
    ax2.set_ylabel("Brier Score (lower is better)")
    ax2.set_title("Calibration Performance")
    ax2.grid(axis="y", alpha=0.3)
    
    # Panel 3: Feature importance for Experiment C
    ax3 = fig.add_subplot(gs[1, 0])
    
    exp_c_key = "C — all 4 features"
    beta_mean = results[exp_c_key]["beta_features_mean"]
    beta_std = results[exp_c_key]["beta_features_std"]
    feature_names = results[exp_c_key]["feature_names"]
    
    # Simplify feature names for display
    display_names = [name.replace("__mean", " (mean)").replace("__std", " (std)") for name in feature_names]
    
    y_pos = np.arange(len(display_names))
    colors = ["#E53935" if b < 0 else "#43A047" for b in beta_mean]
    
    ax3.barh(y_pos, beta_mean, xerr=beta_std, capsize=3, color=colors, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(display_names, fontsize=8)
    ax3.set_xlabel("Beta coefficient (standardized)")
    ax3.set_title("Feature Importance — Experiment C")
    ax3.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.grid(axis="x", alpha=0.3)
    
    # Panel 4: Predicted risk vs reference month (Experiment C)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Train final model on all data for visualization
    X_month = pairs_df["reference_month"].values
    y = pairs_df["corrosion_risk"].values
    
    feature_col_names = results[exp_c_key]["feature_names"]
    X_features = pairs_df[feature_col_names].values
    
    # Standardize
    X_month_mean = X_month.mean()
    X_month_std = X_month.std()
    X_month_scaled = (X_month - X_month_mean) / X_month_std
    
    X_features_mean = X_features.mean(axis=0)
    X_features_std = X_features.std(axis=0)
    X_features_std[X_features_std == 0] = 1.0
    X_features_scaled = (X_features - X_features_mean) / X_features_std
    
    # Train model
    model = build_logistic_model_with_covariates(X_month_scaled, X_features_scaled, y)
    idata = sample_model(model, "Final model")
    
    # Predict on grid
    X_month_grid = np.linspace(X_month.min(), X_month.max(), 200)
    X_month_grid_scaled = (X_month_grid - X_month_mean) / X_month_std
    
    # Use median feature values for prediction
    X_features_median = np.median(X_features_scaled, axis=0)
    X_features_grid = np.tile(X_features_median, (len(X_month_grid), 1))
    
    y_pred_grid = predict_probability(idata, X_month_grid_scaled, X_features_grid)
    
    # Plot
    ax4.scatter(X_month[y == 0], y[y == 0], alpha=0.3, s=20, color="steelblue", label="Risk=0")
    ax4.scatter(X_month[y == 1], y[y == 1], alpha=0.3, s=20, color="darkorange", label="Risk=1")
    ax4.plot(X_month_grid, y_pred_grid, color="red", linewidth=2, label="Predicted P(Risk=1)")
    ax4.set_xlabel("Reference Month (since delivery)")
    ax4.set_ylabel("Corrosion Risk")
    ax4.set_title("Predicted Risk vs Month — Experiment C")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("Bayesian Binary Classification with Environmental Covariates")
    print("=" * 80)
    
    # Load data
    print("\n[1/3] Loading data...")
    corr_df = load_corrosion_data()
    env_df = load_environment_data()
    
    # Create training pairs
    print("\n[2/3] Creating training pairs (Airbus convention)...")
    pairs_df = create_training_pairs(corr_df, env_df, features=FEATURES)
    
    print(f"\nTraining data summary:")
    print(f"  - Total samples: {len(pairs_df)}")
    print(f"  - Aircraft: {pairs_df['aircraft_id'].nunique()}")
    print(f"  - Features: {len([c for c in pairs_df.columns if '__' in c])}")
    
    # Cross-validation for each experiment
    print("\n[3/3] Running cross-validation (5-fold)...")
    
    results = {}
    
    for exp_name, feature_cols in EXPERIMENTS.items():
        print(f"\n  {exp_name}:")
        print(f"    Features: {', '.join(feature_cols)}")
        
        cv_results = cross_validate_experiment(pairs_df, feature_cols, n_splits=5)
        results[exp_name] = cv_results
        
        print(f"\n    Results:")
        print(f"      AUC:     {cv_results['auc_mean']:.4f} ± {cv_results['auc_std']:.4f}")
        print(f"      Brier:   {cv_results['brier_mean']:.4f} ± {cv_results['brier_std']:.4f}")
        print(f"      LogLoss: {cv_results['logloss_mean']:.4f} ± {cv_results['logloss_std']:.4f}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<30} {'AUC':<20} {'Brier Score':<20}")
    print("-" * 80)
    
    for exp_name, res in results.items():
        print(
            f"{exp_name:<30} "
            f"{res['auc_mean']:.4f} ± {res['auc_std']:.4f}    "
            f"{res['brier_mean']:.4f} ± {res['brier_std']:.4f}"
        )
    
    # Find best experiment
    best_exp = max(results.items(), key=lambda x: x[1]["auc_mean"])
    print(f"\nBEST EXPERIMENT: {best_exp[0]} (AUC = {best_exp[1]['auc_mean']:.4f})")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(pairs_df, results)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
