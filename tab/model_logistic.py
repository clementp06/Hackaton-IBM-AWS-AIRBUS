"""
Bayesian Beta-Binomial Model for Aircraft Corrosion Risk
=========================================================
Uses the Airbus convention (1/0 pairs) to model P(corrosion_risk=1 | month).

Three Beta prior experiments (A: Uniform, B: Weakly informative, C: Informative)
plus sklearn logistic regression for comparison.

Outputs:
  - Console table: P(corroded by t) at selected months
  - model_logistic.png: 4-panel figure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# Import our data utilities
from data_utils import (
    load_corrosion_data,
    load_environment_data,
    create_training_pairs,
    create_cv_splits,
)


class PriorConfig(NamedTuple):
    name: str
    label: str
    alpha: float
    beta: float
    color: str
    linestyle: str


@dataclass(frozen=True)
class BetaBinomialResult:
    t_values: np.ndarray
    posterior_mean: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ci_width: np.ndarray
    prior: PriorConfig


@dataclass(frozen=True)
class SklearnResult:
    name: str
    predictions: np.ndarray
    color: str
    linestyle: str


# ---------------------------------------------------------------------------
# Beta-Binomial conjugate update
# ---------------------------------------------------------------------------

def run_beta_binomial(
    pairs_df: pd.DataFrame,
    prior: PriorConfig,
    t_grid: np.ndarray,
) -> BetaBinomialResult:
    """
    At each integer month t:
      Count how many (1/0) pairs have reference_month <= t
      Update Beta posterior based on observed risk labels
      
    For the Airbus convention:
      - At month t, we have observations with reference_month <= t
      - Some have corrosion_risk=1, some have corrosion_risk=0
      - Posterior: Beta(alpha + n_risk_1, beta + n_risk_0)
    """
    posterior_mean = np.empty(len(t_grid))
    ci_lower = np.empty(len(t_grid))
    ci_upper = np.empty(len(t_grid))
    
    for i, t in enumerate(t_grid):
        # Get all pairs with reference_month <= t
        mask = pairs_df["reference_month"] <= t
        subset = pairs_df[mask]
        
        if len(subset) == 0:
            # No data yet, use prior
            rv = beta_dist(prior.alpha, prior.beta)
        else:
            # Count successes (risk=1) and failures (risk=0)
            n_risk_1 = (subset["corrosion_risk"] == 1).sum()
            n_risk_0 = (subset["corrosion_risk"] == 0).sum()
            
            # Posterior
            a_post = prior.alpha + n_risk_1
            b_post = prior.beta + n_risk_0
            rv = beta_dist(a_post, b_post)
        
        posterior_mean[i] = rv.mean()
        ci_lower[i] = rv.ppf(0.025)
        ci_upper[i] = rv.ppf(0.975)
    
    ci_width = ci_upper - ci_lower
    
    return BetaBinomialResult(
        t_values=t_grid,
        posterior_mean=posterior_mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_width,
        prior=prior,
    )


# ---------------------------------------------------------------------------
# Sklearn logistic regression
# ---------------------------------------------------------------------------

def fit_sklearn_logistic(
    pairs_df: pd.DataFrame,
    t_grid: np.ndarray,
) -> SklearnResult:
    """Fit sklearn logistic regression for comparison."""
    X = pairs_df["reference_month"].values.reshape(-1, 1)
    y = pairs_df["corrosion_risk"].values
    
    # Fit logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    # Predict on grid
    X_grid = t_grid.reshape(-1, 1)
    predictions = model.predict_proba(X_grid)[:, 1]
    
    return SklearnResult(
        name="Sklearn Logistic",
        predictions=predictions,
        color="#e67e22",
        linestyle="--",
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_beta_binomial(
    pairs_df: pd.DataFrame,
    prior: PriorConfig,
    n_splits: int = 5,
) -> dict:
    """
    Cross-validate Beta-Binomial model.
    
    Returns AUC and Brier score.
    """
    splits = create_cv_splits(pairs_df, n_splits=n_splits)
    
    aucs = []
    briers = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_df = pairs_df.iloc[train_idx]
        test_df = pairs_df.iloc[test_idx]
        
        # For each test sample, predict using Beta-Binomial trained on train set
        y_test = test_df["corrosion_risk"].values
        y_pred = np.empty(len(test_df))
        
        for i, (_, row) in enumerate(test_df.iterrows()):
            t = row["reference_month"]
            
            # Get training samples with reference_month <= t
            mask = train_df["reference_month"] <= t
            subset = train_df[mask]
            
            if len(subset) == 0:
                # No data, use prior mean
                y_pred[i] = prior.alpha / (prior.alpha + prior.beta)
            else:
                n_risk_1 = (subset["corrosion_risk"] == 1).sum()
                n_risk_0 = (subset["corrosion_risk"] == 0).sum()
                
                a_post = prior.alpha + n_risk_1
                b_post = prior.beta + n_risk_0
                
                # Posterior mean
                y_pred[i] = a_post / (a_post + b_post)
        
        # Compute metrics
        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        
        aucs.append(auc)
        briers.append(brier)
    
    return {
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "brier_mean": np.mean(briers),
        "brier_std": np.std(briers),
    }


def cross_validate_sklearn(
    pairs_df: pd.DataFrame,
    n_splits: int = 5,
) -> dict:
    """Cross-validate sklearn logistic regression."""
    splits = create_cv_splits(pairs_df, n_splits=n_splits)
    
    aucs = []
    briers = []
    
    for train_idx, test_idx in splits:
        X_train = pairs_df.iloc[train_idx]["reference_month"].values.reshape(-1, 1)
        y_train = pairs_df.iloc[train_idx]["corrosion_risk"].values
        
        X_test = pairs_df.iloc[test_idx]["reference_month"].values.reshape(-1, 1)
        y_test = pairs_df.iloc[test_idx]["corrosion_risk"].values
        
        # Fit and predict
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        
        aucs.append(auc)
        briers.append(brier)
    
    return {
        "auc_mean": np.mean(aucs),
        "auc_std": np.std(aucs),
        "brier_mean": np.mean(briers),
        "brier_std": np.std(briers),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PRIORS: list[PriorConfig] = [
    PriorConfig("A", "Beta(1,1) Uniform", 1.0, 1.0, "#2980b9", "-"),
    PriorConfig("B", "Beta(2,2) Weak", 2.0, 2.0, "#27ae60", "--"),
    PriorConfig("C", "Beta(5,5) Informative", 5.0, 5.0, "#c0392b", ":"),
]

REPORT_MONTHS = [24, 36, 48, 60, 72, 84, 96, 108, 120]


def build_figure(
    pairs_df: pd.DataFrame,
    t_grid: np.ndarray,
    bb_results: list[BetaBinomialResult],
    sklearn_result: SklearnResult,
) -> plt.Figure:
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Bayesian Beta-Binomial Corrosion Risk Model\n"
        f"{pairs_df['aircraft_id'].nunique()} Aircraft — Airbus Convention (1/0 pairs)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # ------------------------------------------------------------------
    # Panel 1 — Risk curves
    # ------------------------------------------------------------------
    # Plot raw data points
    risk_0 = pairs_df[pairs_df["corrosion_risk"] == 0]
    risk_1 = pairs_df[pairs_df["corrosion_risk"] == 1]
    
    ax1.scatter(
        risk_0["reference_month"],
        risk_0["corrosion_risk"],
        alpha=0.3,
        s=15,
        color="steelblue",
        label="Risk=0 (healthy)",
        zorder=2,
    )
    ax1.scatter(
        risk_1["reference_month"],
        risk_1["corrosion_risk"],
        alpha=0.3,
        s=15,
        color="darkorange",
        label="Risk=1 (corroded)",
        zorder=2,
    )
    
    # Plot Beta-Binomial posteriors
    for r in bb_results:
        ax1.plot(
            r.t_values,
            r.posterior_mean,
            color=r.prior.color,
            linestyle=r.prior.linestyle,
            linewidth=1.8,
            label=f"Posterior — {r.prior.label}",
        )
    
    # Plot sklearn
    ax1.plot(
        t_grid,
        sklearn_result.predictions,
        color=sklearn_result.color,
        linestyle=sklearn_result.linestyle,
        linewidth=1.8,
        label=sklearn_result.name,
    )
    
    ax1.set_xlabel("Reference Month (since delivery)")
    ax1.set_ylabel("P(corrosion risk = 1)")
    ax1.set_title("Panel 1 — Risk Curves: Beta posteriors & Sklearn")
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(t_grid[0], t_grid[-1])
    ax1.set_ylim(-0.05, 1.05)
    
    # ------------------------------------------------------------------
    # Panel 2 — Credible interval width
    # ------------------------------------------------------------------
    for r in bb_results:
        ax2.plot(
            r.t_values,
            r.ci_width,
            color=r.prior.color,
            linestyle=r.prior.linestyle,
            linewidth=1.8,
            label=f"{r.prior.label}",
        )
    
    ax2.set_xlabel("Month t")
    ax2.set_ylabel("95% CI width (posterior)")
    ax2.set_title("Panel 2 — Posterior uncertainty vs month")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(t_grid[0], t_grid[-1])
    
    # ------------------------------------------------------------------
    # Panel 3 — Histogram of reference months
    # ------------------------------------------------------------------
    ax3.hist(
        risk_0["reference_month"],
        bins=30,
        density=False,
        color="steelblue",
        alpha=0.6,
        label="Risk=0",
        edgecolor="white",
        linewidth=0.5,
    )
    ax3.hist(
        risk_1["reference_month"],
        bins=30,
        density=False,
        color="darkorange",
        alpha=0.6,
        label="Risk=1",
        edgecolor="white",
        linewidth=0.5,
    )
    
    ax3.set_xlabel("Reference Month (since delivery)")
    ax3.set_ylabel("Count")
    ax3.set_title("Panel 3 — Distribution of training pairs")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------
    # Panel 4 — Bar chart: P(risk=1) at milestone months
    # ------------------------------------------------------------------
    n_groups = len(REPORT_MONTHS)
    n_bars = len(bb_results) + 1  # +1 for sklearn
    bar_width = 0.15
    x = np.arange(n_groups)
    
    bar_series: list[tuple[str, np.ndarray, str]] = []
    for r in bb_results:
        idxs = [np.argmin(np.abs(t_grid - t)) for t in REPORT_MONTHS]
        bar_series.append((r.prior.label, r.posterior_mean[idxs], r.prior.color))
    
    # Add sklearn
    idxs = [np.argmin(np.abs(t_grid - t)) for t in REPORT_MONTHS]
    bar_series.append((sklearn_result.name, sklearn_result.predictions[idxs], sklearn_result.color))
    
    total_width = bar_width * n_bars
    offsets = np.linspace(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, n_bars)
    
    for j, (label, vals, color) in enumerate(bar_series):
        ax4.bar(
            x + offsets[j],
            vals,
            width=bar_width,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor="white",
        )
    
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"t={t}" for t in REPORT_MONTHS], fontsize=8)
    ax4.set_ylabel("P(risk=1)")
    ax4.set_title("Panel 4 — P(risk=1) at milestone months")
    ax4.legend(fontsize=7.5, ncol=2)
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.set_ylim(0, 1.05)
    
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Bayesian Beta-Binomial Corrosion Risk Model")
    print("=" * 80)
    
    # Load data
    print("\n[1/4] Loading data...")
    corr_df = load_corrosion_data()
    env_df = load_environment_data()
    
    # Create training pairs
    print("\n[2/4] Creating training pairs (Airbus convention)...")
    pairs_df = create_training_pairs(corr_df, env_df)
    
    n_aircraft = pairs_df["aircraft_id"].nunique()
    print(f"\nDataset: {n_aircraft} aircraft, {len(pairs_df)} samples")
    print(f"Reference month range: {pairs_df['reference_month'].min():.0f} – {pairs_df['reference_month'].max():.0f}")
    print(f"Class balance: {(pairs_df['corrosion_risk']==1).mean():.2%}")
    
    t_grid = np.arange(0, 137, dtype=float)
    
    # Beta-Binomial posterior for each prior
    print("\n[3/4] Computing Beta-Binomial posteriors...")
    bb_results = [run_beta_binomial(pairs_df, prior, t_grid) for prior in PRIORS]
    
    # Sklearn logistic regression
    print("Computing sklearn logistic regression...")
    sklearn_result = fit_sklearn_logistic(pairs_df, t_grid)
    
    # Cross-validation
    print("\n[4/4] Running cross-validation (5-fold)...")
    
    cv_results = {}
    
    for prior in PRIORS:
        print(f"\n  {prior.label}:")
        cv_res = cross_validate_beta_binomial(pairs_df, prior, n_splits=5)
        cv_results[prior.label] = cv_res
        print(f"    AUC:   {cv_res['auc_mean']:.4f} ± {cv_res['auc_std']:.4f}")
        print(f"    Brier: {cv_res['brier_mean']:.4f} ± {cv_res['brier_std']:.4f}")
    
    print(f"\n  Sklearn Logistic:")
    sklearn_cv = cross_validate_sklearn(pairs_df, n_splits=5)
    cv_results["Sklearn"] = sklearn_cv
    print(f"    AUC:   {sklearn_cv['auc_mean']:.4f} ± {sklearn_cv['auc_std']:.4f}")
    print(f"    Brier: {sklearn_cv['brier_mean']:.4f} ± {sklearn_cv['brier_std']:.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'AUC':<20} {'Brier Score':<20}")
    print("-" * 80)
    
    for model_name, res in cv_results.items():
        print(
            f"{model_name:<25} "
            f"{res['auc_mean']:.4f} ± {res['auc_std']:.4f}    "
            f"{res['brier_mean']:.4f} ± {res['brier_std']:.4f}"
        )
    
    best_model = max(cv_results.items(), key=lambda x: x[1]["auc_mean"])
    print(f"\nBEST MODEL: {best_model[0]} (AUC = {best_model[1]['auc_mean']:.4f})")
    
    # Figure
    print("\nGenerating figure...")
    fig = build_figure(pairs_df, t_grid, bb_results, sklearn_result)
    output_path = "model_logistic.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    plt.close(fig)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
