"""
Bayesian Binary Classification for Corrosion Risk Prediction.

Uses the Airbus convention:
- corrosion_risk = 1 at observation month
- corrosion_risk = 0 at observation month - 24

Three link functions: Logistic, Probit, Complementary log-log
Two prior experiments: A (weakly informative) and B (diffuse)
Outputs model_basic.png and cross-validation metrics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)

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

RANDOM_SEED = 42
DRAWS = 2000
TUNE = 1000
TARGET_ACCEPT = 0.9
OUTPUT_PATH = "model_basic.png"

# ---------------------------------------------------------------------------
# Prior Definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryPrior:
    """Priors for binary classification model."""
    intercept_mu: float
    intercept_sigma: float
    slope_sigma: float


@dataclass(frozen=True)
class ExperimentPriors:
    name: str
    prior: BinaryPrior


EXPERIMENT_A = ExperimentPriors(
    name="Experiment A — Weakly Informative",
    prior=BinaryPrior(intercept_mu=0.0, intercept_sigma=2.0, slope_sigma=0.5),
)

EXPERIMENT_B = ExperimentPriors(
    name="Experiment B — Diffuse",
    prior=BinaryPrior(intercept_mu=0.0, intercept_sigma=5.0, slope_sigma=1.0),
)


# ---------------------------------------------------------------------------
# Model Building
# ---------------------------------------------------------------------------


def build_logistic_model(
    X: np.ndarray, y: np.ndarray, prior: BinaryPrior
) -> pm.Model:
    """Bayesian logistic regression."""
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal("intercept", mu=prior.intercept_mu, sigma=prior.intercept_sigma)
        slope = pm.Normal("slope", mu=0.0, sigma=prior.slope_sigma)
        
        # Linear predictor
        eta = intercept + slope * X
        
        # Logistic link
        p = pm.math.invlogit(eta)
        
        # Likelihood
        _obs = pm.Bernoulli("obs", p=p, observed=y)
    
    return model


def build_probit_model(
    X: np.ndarray, y: np.ndarray, prior: BinaryPrior
) -> pm.Model:
    """Bayesian probit regression."""
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal("intercept", mu=prior.intercept_mu, sigma=prior.intercept_sigma)
        slope = pm.Normal("slope", mu=0.0, sigma=prior.slope_sigma)
        
        # Linear predictor
        eta = intercept + slope * X
        
        # Probit link (Phi = standard normal CDF)
        p = pm.math.invprobit(eta)
        
        # Likelihood
        _obs = pm.Bernoulli("obs", p=p, observed=y)
    
    return model


def build_cloglog_model(
    X: np.ndarray, y: np.ndarray, prior: BinaryPrior
) -> pm.Model:
    """Bayesian complementary log-log regression."""
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal("intercept", mu=prior.intercept_mu, sigma=prior.intercept_sigma)
        slope = pm.Normal("slope", mu=0.0, sigma=prior.slope_sigma)
        
        # Linear predictor
        eta = intercept + slope * X
        
        # Complementary log-log link: p = 1 - exp(-exp(eta))
        p = 1.0 - pm.math.exp(-pm.math.exp(eta))
        
        # Likelihood
        _obs = pm.Bernoulli("obs", p=p, observed=y)
    
    return model


# ---------------------------------------------------------------------------
# Sampling & Prediction
# ---------------------------------------------------------------------------


def sample_model(model: pm.Model, model_name: str) -> az.InferenceData:
    """Sample from a PyMC model using NUTS."""
    print(f"  Sampling {model_name}...")
    with model:
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=True,
            return_inferencedata=True,
        )
        pm.compute_log_likelihood(idata)
    return idata


def predict_probability(
    idata: az.InferenceData,
    X_new: np.ndarray,
    link_function: str,
) -> np.ndarray:
    """
    Predict probabilities for new data using posterior samples.
    
    Returns mean predicted probability across posterior samples.
    """
    posterior = idata.posterior
    
    # Extract parameters
    intercept = posterior["intercept"].values.flatten()
    slope = posterior["slope"].values.flatten()
    
    # Compute eta for each posterior sample
    eta = intercept[:, None] + slope[:, None] * X_new[None, :]
    
    # Apply link function
    if link_function == "logistic":
        p = 1.0 / (1.0 + np.exp(-eta))
    elif link_function == "probit":
        from scipy.stats import norm
        p = norm.cdf(eta)
    elif link_function == "cloglog":
        p = 1.0 - np.exp(-np.exp(eta))
    else:
        raise ValueError(f"Unknown link function: {link_function}")
    
    # Return mean across posterior samples
    return p.mean(axis=0)


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------


def cross_validate_model(
    pairs_df: pd.DataFrame,
    model_builder: callable,
    link_function: str,
    prior: BinaryPrior,
    n_splits: int = 5,
) -> dict:
    """
    Perform cross-validation for a model.
    
    Returns dictionary with metrics: AUC, Brier score, log-loss.
    """
    X = pairs_df["reference_month"].values
    y = pairs_df["corrosion_risk"].values
    
    # Standardize X
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std
    
    splits = create_cv_splits(pairs_df, n_splits=n_splits)
    
    aucs = []
    briers = []
    logloss_scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"    Fold {fold_idx + 1}/{n_splits}...")
        
        X_train = X_scaled[train_idx]
        y_train = y[train_idx]
        X_test = X_scaled[test_idx]
        y_test = y[test_idx]
        
        # Build and sample model
        model = model_builder(X_train, y_train, prior)
        idata = sample_model(model, f"Fold {fold_idx + 1}")
        
        # Predict on test set
        y_pred = predict_probability(idata, X_test, link_function)
        
        # Compute metrics
        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        ll = log_loss(y_test, y_pred)
        
        aucs.append(auc)
        briers.append(brier)
        logloss_scores.append(ll)
        
        print(f"      AUC: {auc:.4f}, Brier: {brier:.4f}, LogLoss: {ll:.4f}")
    
    return {
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "brier_mean": np.mean(briers),
        "brier_std": np.std(briers),
        "logloss_mean": np.mean(logloss_scores),
        "logloss_std": np.std(logloss_scores),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    pairs_df: pd.DataFrame,
    results: dict,
    output_path: str,
) -> None:
    """Create visualization of model results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Bayesian Binary Classification — Corrosion Risk Prediction (Airbus Convention)",
        fontsize=14,
        fontweight="bold",
    )
    
    X = pairs_df["reference_month"].values
    y = pairs_df["corrosion_risk"].values
    
    # Standardize X for modeling
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std
    
    # Grid for predictions
    X_grid = np.linspace(X.min(), X.max(), 200)
    X_grid_scaled = (X_grid - X_mean) / X_std
    
    # Plot each model
    models_info = [
        ("Logistic", "logistic", build_logistic_model, 0),
        ("Probit", "probit", build_probit_model, 1),
        ("Cloglog", "cloglog", build_cloglog_model, 2),
    ]
    
    for model_name, link_func, model_builder, col_idx in models_info:
        # Experiment A
        ax_a = axes[0, col_idx]
        prior_a = EXPERIMENT_A.prior
        
        # Train model
        model_a = model_builder(X_scaled, y, prior_a)
        idata_a = sample_model(model_a, f"{model_name} (Exp A)")
        
        # Predict
        y_pred_a = predict_probability(idata_a, X_grid_scaled, link_func)
        
        # Plot
        ax_a.scatter(X[y == 0], y[y == 0], alpha=0.3, s=20, color="steelblue", label="Risk=0")
        ax_a.scatter(X[y == 1], y[y == 1], alpha=0.3, s=20, color="darkorange", label="Risk=1")
        ax_a.plot(X_grid, y_pred_a, color="red", linewidth=2, label="Predicted P(Risk=1)")
        ax_a.set_xlabel("Reference Month (since delivery)")
        ax_a.set_ylabel("Corrosion Risk")
        ax_a.set_title(f"{model_name} — {EXPERIMENT_A.name}")
        ax_a.legend(fontsize=8)
        ax_a.grid(alpha=0.3)
        ax_a.set_ylim(-0.1, 1.1)
        
        # Experiment B
        ax_b = axes[1, col_idx]
        prior_b = EXPERIMENT_B.prior
        
        # Train model
        model_b = model_builder(X_scaled, y, prior_b)
        idata_b = sample_model(model_b, f"{model_name} (Exp B)")
        
        # Predict
        y_pred_b = predict_probability(idata_b, X_grid_scaled, link_func)
        
        # Plot
        ax_b.scatter(X[y == 0], y[y == 0], alpha=0.3, s=20, color="steelblue", label="Risk=0")
        ax_b.scatter(X[y == 1], y[y == 1], alpha=0.3, s=20, color="darkorange", label="Risk=1")
        ax_b.plot(X_grid, y_pred_b, color="red", linewidth=2, label="Predicted P(Risk=1)")
        ax_b.set_xlabel("Reference Month (since delivery)")
        ax_b.set_ylabel("Corrosion Risk")
        ax_b.set_title(f"{model_name} — {EXPERIMENT_B.name}")
        ax_b.legend(fontsize=8)
        ax_b.grid(alpha=0.3)
        ax_b.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 80)
    print("Bayesian Binary Classification — Corrosion Risk Prediction")
    print("=" * 80)
    
    # Load data
    print("\n[1/4] Loading data...")
    corr_df = load_corrosion_data()
    env_df = load_environment_data()
    
    # Create training pairs
    print("\n[2/4] Creating training pairs (Airbus convention)...")
    pairs_df = create_training_pairs(corr_df, env_df)
    
    print(f"\nTraining data summary:")
    print(f"  - Total samples: {len(pairs_df)}")
    print(f"  - Aircraft: {pairs_df['aircraft_id'].nunique()}")
    print(f"  - Reference month range: {pairs_df['reference_month'].min():.0f} - {pairs_df['reference_month'].max():.0f}")
    print(f"  - Class balance: {(pairs_df['corrosion_risk']==1).mean():.2%}")
    
    # Cross-validation
    print("\n[3/4] Running cross-validation (5-fold)...")
    
    models = [
        ("Logistic", "logistic", build_logistic_model),
        ("Probit", "probit", build_probit_model),
        ("Cloglog", "cloglog", build_cloglog_model),
    ]
    
    results = {}
    
    for exp_name, exp_priors in [("Exp A", EXPERIMENT_A), ("Exp B", EXPERIMENT_B)]:
        print(f"\n  {exp_priors.name}:")
        for model_name, link_func, model_builder in models:
            print(f"\n    {model_name}:")
            cv_results = cross_validate_model(
                pairs_df,
                model_builder,
                link_func,
                exp_priors.prior,
                n_splits=5,
            )
            results[f"{exp_name}_{model_name}"] = cv_results
            
            print(f"\n    Results:")
            print(f"      AUC:     {cv_results['auc_mean']:.4f} ± {cv_results['auc_std']:.4f}")
            print(f"      Brier:   {cv_results['brier_mean']:.4f} ± {cv_results['brier_std']:.4f}")
            print(f"      LogLoss: {cv_results['logloss_mean']:.4f} ± {cv_results['logloss_std']:.4f}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'AUC':<20} {'Brier Score':<20} {'Log Loss':<20}")
    print("-" * 80)
    
    for key, res in results.items():
        print(
            f"{key:<20} "
            f"{res['auc_mean']:.4f} ± {res['auc_std']:.4f}    "
            f"{res['brier_mean']:.4f} ± {res['brier_std']:.4f}    "
            f"{res['logloss_mean']:.4f} ± {res['logloss_std']:.4f}"
        )
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]["auc_mean"])
    print(f"\nBEST MODEL: {best_model[0]} (AUC = {best_model[1]['auc_mean']:.4f})")
    
    # Generate plots
    print("\n[4/4] Generating plots...")
    plot_results(pairs_df, results, OUTPUT_PATH)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
