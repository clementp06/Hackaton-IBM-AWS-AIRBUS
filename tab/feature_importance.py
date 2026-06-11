import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

env_training = pd.read_csv("../data/environment_training.csv")
corr_training = pd.read_csv("../data/corrosions_training.csv")
corr_training["observation_date"] = pd.to_datetime(corr_training["observation_date"])
corr_training = corr_training.assign(
    months=(
        (corr_training["observation_date"].dt.year - corr_training["aircraft_delivery_year"]) * 12
        + (corr_training["observation_date"].dt.month - corr_training["aircraft_delivery_month"])
    )
)
res = pd.merge(env_training, corr_training, on="aircraft_id", how="inner")

env_cols = [
    "total_parking_minutes", "metar_temperature_c", "metar_relative_humidity",
    "metar_dew_point_c", "metar_wind_speed_kn", "metar_visibility_mi",
    "metar_hour_precipitation",
    "sea_salt_aerosol_003_05_mixing_ratio", "sea_salt_aerosol_05_5_mixing_ratio",
    "sea_salt_aerosol_5_20_mixing_ratio",
    "dust_aerosol_003_055_mixing_ratio", "dust_aerosol_055_09_mixing_ratio",
    "dust_aerosol_09_20_mixing_ratio",
    "hydrophilic_organic_matter_aerosol_mixing_ratio",
    "hydrophobic_organic_matter_aerosol_mixing_ratio",
    "hydrophilic_black_carbon_aerosol_mixing_ratio",
    "hydrophobic_black_carbon_aerosol_mixing_ratio",
    "sulphate_aerosol_mixing_ratio", "ethane", "c3h8", "isoprene",
    "carbon_monoxide_mass_mixing_ratio", "ozone_mass_mixing_ratio",
    "h2o2", "formaldehyde", "hno3",
    "nitrogen_monoxide_mass_mixing_ratio", "nitrogen_dioxide_mass_mixing_ratio",
    "oh", "organic_nitrates", "specific_humidity",
    "sulphur_dioxide_mass_mixing_ratio", "temperature",
]

agg_dict = {}
for col in env_cols:
    agg_dict[f"{col}__mean"] = (col, "mean")
    agg_dict[f"{col}__std"]  = (col, "std")

per_ac = res.groupby("aircraft_id").agg(
    months=("months", "first"),
    n_obs=("year_month", "count"),
    **agg_dict
).reset_index().fillna(0)

feature_cols = [c for c in per_ac.columns if c not in ("aircraft_id", "months")]
# Exclude n_obs — it's an observation-length proxy, not a causal env factor
env_feature_cols = [c for c in feature_cols if c != "n_obs"]

X_full = per_ac[feature_cols].values
X_env  = per_ac[env_feature_cols].values
y = per_ac["months"].values

# ── Train RF on pure environmental features ──
rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_env, y)
cv_scores = cross_val_score(rf, X_env, y, cv=5, scoring="r2")

# Permutation importance (more reliable than impurity-based)
perm = permutation_importance(rf, X_env, y, n_repeats=30, random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=env_feature_cols)

# Mutual information
mi = mutual_info_regression(X_env, y, random_state=42)
mi_series = pd.Series(mi, index=env_feature_cols)

# Pearson correlation
pearson = per_ac[env_feature_cols + ["months"]].corr()["months"].drop("months").abs()

# ── Combine into a readable name ──
def clean_name(raw):
    raw = raw.replace("_mixing_ratio", "").replace("_mass", "")
    raw = raw.replace("aerosol_", "aerosol ").replace("_", " ")
    raw = raw.replace("  ", " ").strip()
    suffix = ""
    if raw.endswith(" mean"):
        suffix = " (mean)"
        raw = raw[:-5]
    elif raw.endswith(" std"):
        suffix = " (variability)"
        raw = raw[:-4]
    return raw.strip() + suffix

TOP_N = 15

perm_top  = perm_imp.nlargest(TOP_N)
mi_top    = mi_series.nlargest(TOP_N)
pear_top  = pearson.nlargest(TOP_N)

# ── Consensus: rank across all three methods ──
all_feats = set(perm_top.index) | set(mi_top.index) | set(pear_top.index)
rank_df = pd.DataFrame({
    "perm_rank":   perm_imp.rank(ascending=False),
    "mi_rank":     mi_series.rank(ascending=False),
    "pearson_rank": pearson.rank(ascending=False),
}, index=env_feature_cols)
rank_df["avg_rank"] = rank_df.mean(axis=1)
consensus = rank_df.loc[list(all_feats)].sort_values("avg_rank").head(TOP_N)
consensus["label"] = [clean_name(f) for f in consensus.index]

print(f"\nRF R² (5-fold CV, env features only): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print("\nTop features by consensus rank:")
for feat, row in consensus.iterrows():
    print(f"  avg rank {row['avg_rank']:5.1f}  {clean_name(feat)}")

# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Feature Importance for Predicting Corrosion Onset (months)",
             fontsize=14, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.45)

palette = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, TOP_N))

# ── 1. Permutation importance ──
ax1 = fig.add_subplot(gs[0, 0])
vals = perm_top.values
labels = [clean_name(f) for f in perm_top.index]
bars = ax1.barh(range(TOP_N), vals[::-1], color=palette)
ax1.set_yticks(range(TOP_N))
ax1.set_yticklabels(labels[::-1], fontsize=8)
ax1.set_xlabel("Mean accuracy drop when shuffled")
ax1.set_title("Permutation Importance\n(most reliable)", fontweight="bold")
ax1.axvline(0, color="black", linewidth=0.6)

# ── 2. Mutual information ──
ax2 = fig.add_subplot(gs[0, 1])
vals2 = mi_top.values
labels2 = [clean_name(f) for f in mi_top.index]
ax2.barh(range(TOP_N), vals2[::-1], color=palette)
ax2.set_yticks(range(TOP_N))
ax2.set_yticklabels(labels2[::-1], fontsize=8)
ax2.set_xlabel("Mutual information score")
ax2.set_title("Mutual Information\n(captures non-linear relationships)", fontweight="bold")

# ── 3. Pearson correlation ──
ax3 = fig.add_subplot(gs[1, 0])
vals3 = pear_top.values
labels3 = [clean_name(f) for f in pear_top.index]
ax3.barh(range(TOP_N), vals3[::-1], color=palette)
ax3.set_yticks(range(TOP_N))
ax3.set_yticklabels(labels3[::-1], fontsize=8)
ax3.set_xlabel("|Pearson r| with months")
ax3.set_title("Linear Correlation\n(absolute value)", fontweight="bold")

# ── 4. Consensus ranking ──
ax4 = fig.add_subplot(gs[1, 1])
score = 1 / consensus["avg_rank"]
score = score / score.max()
colors4 = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(consensus)))
ax4.barh(range(len(consensus)), score.values[::-1], color=colors4)
ax4.set_yticks(range(len(consensus)))
ax4.set_yticklabels(consensus["label"].tolist()[::-1], fontsize=8)
ax4.set_xlabel("Normalised consensus score  (higher = more important)")
ax4.set_title(f"Consensus Ranking  (all 3 methods)\nRF R² = {cv_scores.mean():.2f} ± {cv_scores.std():.2f}",
              fontweight="bold")

plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print("\nSaved feature_importance.png")
