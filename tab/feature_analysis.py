"""
Comprehensive feature analysis for corrosion prediction.

Performs:
1. PCA analysis to identify redundant features
2. Correlation analysis
3. Feature importance from trained models
4. Feature engineering suggestions
5. Interaction feature discovery
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
# import seaborn as sns  # Not required, using matplotlib directly
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

warnings.filterwarnings("ignore")

# Import our data utilities
from data_utils import (
    load_corrosion_data,
    load_environment_data,
    create_training_pairs,
    ENV_FEATURES,
)

OUTPUT_DIR = Path(__file__).parent
RANDOM_SEED = 42


def analyze_pca(X: np.ndarray, feature_names: list[str], n_components: int = 10) -> dict:
    """
    Perform PCA analysis to identify redundant features.
    
    Returns:
        Dictionary with PCA results
    """
    print("\n" + "=" * 80)
    print("PCA ANALYSIS")
    print("=" * 80)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=min(n_components, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    print(f"\nExplained variance by component:")
    for i, (var, cum_var) in enumerate(zip(explained_var, cumulative_var)):
        print(f"  PC{i+1}: {var:.4f} (cumulative: {cum_var:.4f})")
    
    # Find number of components for 95% variance
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    print(f"\nComponents needed for 95% variance: {n_95} (out of {len(feature_names)})")
    
    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=feature_names
    )
    
    print(f"\nTop features for first 3 components:")
    for i in range(min(3, pca.n_components_)):
        pc_name = f"PC{i+1}"
        top_features = loadings[pc_name].abs().nlargest(5)
        print(f"\n  {pc_name} (explains {explained_var[i]:.2%}):")
        for feat, loading in top_features.items():
            print(f"    {feat:<50} {loadings.loc[feat, pc_name]:+.3f}")
    
    return {
        "pca": pca,
        "explained_variance": explained_var,
        "cumulative_variance": cumulative_var,
        "loadings": loadings,
        "n_components_95": n_95,
        "X_pca": X_pca,
    }


def analyze_correlations(X: np.ndarray, feature_names: list[str], threshold: float = 0.9) -> dict:
    """
    Analyze feature correlations to identify redundant features.
    
    Returns:
        Dictionary with correlation results
    """
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)
    corr_df = pd.DataFrame(corr_matrix, index=feature_names, columns=feature_names)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) >= threshold:
                high_corr_pairs.append({
                    "feature1": feature_names[i],
                    "feature2": feature_names[j],
                    "correlation": corr_matrix[i, j],
                })
    
    print(f"\nHighly correlated feature pairs (|r| >= {threshold}):")
    if high_corr_pairs:
        for pair in sorted(high_corr_pairs, key=lambda x: -abs(x["correlation"])):
            print(f"  {pair['feature1']:<40} <-> {pair['feature2']:<40} r={pair['correlation']:+.3f}")
    else:
        print("  None found")
    
    # Hierarchical clustering of features
    print(f"\nPerforming hierarchical clustering of features...")
    linkage = hierarchy.linkage(corr_matrix, method='average')
    
    return {
        "correlation_matrix": corr_df,
        "high_corr_pairs": high_corr_pairs,
        "linkage": linkage,
    }


def analyze_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str]
) -> dict:
    """
    Analyze feature importance using multiple methods.
    
    Returns:
        Dictionary with importance results
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Method 1: Random Forest
    print("\n1. Random Forest Feature Importance:")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, max_depth=10)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    print("  Top 10 features:")
    for feat, imp in rf_importance.head(10).items():
        print(f"    {feat:<50} {imp:.4f}")
    
    # Method 2: Gradient Boosting
    print("\n2. Gradient Boosting Feature Importance:")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED, max_depth=5)
    gb.fit(X, y)
    gb_importance = pd.Series(gb.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    print("  Top 10 features:")
    for feat, imp in gb_importance.head(10).items():
        print(f"    {feat:<50} {imp:.4f}")
    
    # Method 3: Mutual Information
    print("\n3. Mutual Information:")
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    mi_importance = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
    
    print("  Top 10 features:")
    for feat, imp in mi_importance.head(10).items():
        print(f"    {feat:<50} {imp:.4f}")
    
    # Aggregate importance
    importance_df = pd.DataFrame({
        "random_forest": rf_importance,
        "gradient_boosting": gb_importance,
        "mutual_info": mi_importance,
    })
    importance_df["mean_rank"] = importance_df.rank(ascending=False).mean(axis=1)
    importance_df = importance_df.sort_values("mean_rank")
    
    print("\n4. Aggregated Importance (by mean rank):")
    print("  Top 10 features:")
    for feat in importance_df.head(10).index:
        print(f"    {feat:<50} rank={importance_df.loc[feat, 'mean_rank']:.1f}")
    
    return {
        "rf_importance": rf_importance,
        "gb_importance": gb_importance,
        "mi_importance": mi_importance,
        "importance_df": importance_df,
    }


def suggest_interaction_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    top_n: int = 5
) -> list[tuple[str, str]]:
    """
    Suggest potential interaction features based on top features.
    
    Returns:
        List of feature pairs for interactions
    """
    print("\n" + "=" * 80)
    print("INTERACTION FEATURE SUGGESTIONS")
    print("=" * 80)
    
    # Get top features from mutual information
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    top_features_idx = np.argsort(mi_scores)[-top_n:]
    top_features = [feature_names[i] for i in top_features_idx]
    
    print(f"\nTop {top_n} features for interaction:")
    for feat in top_features:
        print(f"  - {feat}")
    
    # Suggest pairwise interactions
    interactions = []
    print(f"\nSuggested interaction features:")
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            interactions.append((feat1, feat2))
            print(f"  - {feat1} × {feat2}")
    
    return interactions


def suggest_new_features(pairs_df: pd.DataFrame) -> list[str]:
    """
    Suggest new engineered features based on domain knowledge.
    
    Returns:
        List of new feature suggestions
    """
    print("\n" + "=" * 80)
    print("NEW FEATURE SUGGESTIONS")
    print("=" * 80)
    
    suggestions = []
    
    # 1. Time-based features
    print("\n1. Time-based features:")
    time_features = [
        "reference_month_squared (non-linear time effect)",
        "reference_month_log (logarithmic time effect)",
        "reference_month_sqrt (square root time effect)",
        "months_since_delivery_binned (categorical time periods)",
    ]
    for feat in time_features:
        print(f"  - {feat}")
        suggestions.append(feat)
    
    # 2. Environmental aggregations
    print("\n2. Enhanced environmental aggregations:")
    env_agg_features = [
        "environmental_stress_index (weighted combination of harsh conditions)",
        "salt_exposure_total (sum of all salt aerosol features)",
        "humidity_temperature_interaction (humidity × temperature)",
        "wind_salt_interaction (wind speed × salt aerosol)",
        "seasonal_variation (std of environmental features)",
    ]
    for feat in env_agg_features:
        print(f"  - {feat}")
        suggestions.append(feat)
    
    # 3. Ratios and differences
    print("\n3. Ratio and difference features:")
    ratio_features = [
        "temperature_dewpoint_diff (temperature - dew point)",
        "large_small_salt_ratio (large salt / small salt aerosol)",
        "pollutant_index (SO2 + NO2 + O3)",
    ]
    for feat in ratio_features:
        print(f"  - {feat}")
        suggestions.append(feat)
    
    # 4. Statistical features
    print("\n4. Statistical features from time series:")
    stat_features = [
        "environmental_trend (slope of environmental features over time)",
        "environmental_volatility (coefficient of variation)",
        "extreme_event_count (number of extreme environmental conditions)",
    ]
    for feat in stat_features:
        print(f"  - {feat}")
        suggestions.append(feat)
    
    return suggestions


def plot_analysis_results(
    pca_results: dict,
    corr_results: dict,
    importance_results: dict,
    feature_names: list[str],
) -> None:
    """
    Create comprehensive visualization of analysis results.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. PCA Explained Variance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(1, len(pca_results["explained_variance"]) + 1),
            pca_results["explained_variance"])
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("PCA Explained Variance")
    ax1.grid(alpha=0.3)
    
    # 2. Cumulative Variance
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(1, len(pca_results["cumulative_variance"]) + 1),
             pca_results["cumulative_variance"], marker='o')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Feature Loadings Heatmap (first 3 PCs)
    ax3 = fig.add_subplot(gs[0, 2])
    loadings_subset = pca_results["loadings"].iloc[:, :3]
    im3 = ax3.imshow(loadings_subset.values, cmap="RdBu_r", aspect='auto', vmin=-1, vmax=1)
    ax3.set_yticks(range(len(loadings_subset.index)))
    ax3.set_yticklabels([name[:30] for name in loadings_subset.index], fontsize=6)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['PC1', 'PC2', 'PC3'])
    plt.colorbar(im3, ax=ax3, label="Loading")
    ax3.set_title("Feature Loadings (First 3 PCs)")
    ax3.set_xlabel("Principal Component")
    
    # 4. Correlation Matrix
    ax4 = fig.add_subplot(gs[1, :])
    corr_matrix = corr_results["correlation_matrix"].values
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_masked = np.ma.array(corr_matrix, mask=mask)
    im4 = ax4.imshow(corr_masked, cmap="RdBu_r", aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(corr_results["correlation_matrix"].columns)))
    ax4.set_yticks(range(len(corr_results["correlation_matrix"].index)))
    ax4.set_xticklabels([name[:20] for name in corr_results["correlation_matrix"].columns], rotation=90, fontsize=6)
    ax4.set_yticklabels([name[:20] for name in corr_results["correlation_matrix"].index], fontsize=6)
    plt.colorbar(im4, ax=ax4, label="Correlation")
    ax4.set_title("Feature Correlation Matrix")
    
    # 5. Feature Importance Comparison
    ax5 = fig.add_subplot(gs[2, 0])
    top_n = 10
    importance_df = importance_results["importance_df"].head(top_n)
    x = np.arange(top_n)
    width = 0.25
    
    ax5.barh(x - width, importance_df["random_forest"], width, label="Random Forest")
    ax5.barh(x, importance_df["gradient_boosting"], width, label="Gradient Boosting")
    ax5.barh(x + width, importance_df["mutual_info"], width, label="Mutual Info")
    
    ax5.set_yticks(x)
    ax5.set_yticklabels([name[:30] for name in importance_df.index])
    ax5.set_xlabel("Importance Score")
    ax5.set_title(f"Top {top_n} Features by Importance")
    ax5.legend()
    ax5.invert_yaxis()
    
    # 6. Dendrogram
    ax6 = fig.add_subplot(gs[2, 1:])
    hierarchy.dendrogram(
        corr_results["linkage"],
        labels=[name[:20] for name in feature_names],
        ax=ax6,
        leaf_rotation=90,
    )
    ax6.set_title("Hierarchical Clustering of Features")
    ax6.set_xlabel("Feature")
    ax6.set_ylabel("Distance")
    
    plt.savefig(OUTPUT_DIR / "feature_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {OUTPUT_DIR / 'feature_analysis.png'}")
    plt.close()


def main() -> None:
    print("=" * 80)
    print("COMPREHENSIVE FEATURE ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n[1/6] Loading data...")
    corr_df = load_corrosion_data()
    env_df = load_environment_data()
    
    # Create training pairs
    print("\n[2/6] Creating training pairs...")
    pairs_df = create_training_pairs(corr_df, env_df, features=ENV_FEATURES)
    
    # Prepare features
    feature_cols = ["reference_month"]
    env_feature_cols = [c for c in pairs_df.columns if "__" in c and c != "aircraft_id"]
    feature_cols.extend(env_feature_cols)
    
    X = pairs_df[feature_cols].values
    y = pairs_df["corrosion_risk"].values
    
    print(f"\nDataset:")
    print(f"  - Samples: {len(X)}")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Class balance: {y.mean():.2%}")
    
    # Perform analyses
    print("\n[3/6] Performing PCA analysis...")
    pca_results = analyze_pca(X, feature_cols, n_components=10)
    
    print("\n[4/6] Analyzing correlations...")
    corr_results = analyze_correlations(X, feature_cols, threshold=0.9)
    
    print("\n[5/6] Analyzing feature importance...")
    importance_results = analyze_feature_importance(X, y, feature_cols)
    
    print("\n[6/6] Generating suggestions...")
    interaction_suggestions = suggest_interaction_features(X, y, feature_cols, top_n=5)
    new_feature_suggestions = suggest_new_features(pairs_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_analysis_results(pca_results, corr_results, importance_results, feature_cols)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\n1. DIMENSIONALITY:")
    print(f"   - Current features: {len(feature_cols)}")
    print(f"   - Components for 95% variance: {pca_results['n_components_95']}")
    print(f"   - Potential reduction: {len(feature_cols) - pca_results['n_components_95']} features")
    
    print(f"\n2. REDUNDANCY:")
    n_high_corr = len(corr_results["high_corr_pairs"])
    print(f"   - Highly correlated pairs: {n_high_corr}")
    if n_high_corr > 0:
        print(f"   - Consider removing one feature from each pair")
    
    print(f"\n3. MOST IMPORTANT FEATURES:")
    top_features = importance_results["importance_df"].head(5).index.tolist()
    for i, feat in enumerate(top_features, 1):
        print(f"   {i}. {feat}")
    
    print(f"\n4. RECOMMENDED ACTIONS:")
    print(f"   a) Remove redundant features (high correlation)")
    print(f"   b) Focus on top {pca_results['n_components_95']} most important features")
    print(f"   c) Add {len(interaction_suggestions)} interaction features")
    print(f"   d) Engineer {len(new_feature_suggestions)} new domain-specific features")
    print(f"   e) Consider PCA transformation for dimensionality reduction")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
