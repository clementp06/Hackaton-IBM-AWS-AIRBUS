"""
Complete end-to-end pipeline for Airbus corrosion risk prediction.
Brier Score metric -> probability calibration is critical.

Pipeline:
  1. Load & preprocess data
  2. Feature engineering (cumulative + rolling windows, no look-ahead)
  3. Target construction (corrosion month = 1, 24 months prior = 0)
  4. GroupKFold CV + CalibratedClassifierCV + Brier Score evaluation
  5. Inference on test set & submission generation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
DATA_DIR = "data"
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Data Loading & Preprocessing
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

corrosions = pd.read_csv(f"{DATA_DIR}/corrosions_training.csv")
env_train = pd.read_csv(f"{DATA_DIR}/environment_training.csv")
env_test = pd.read_csv(f"{DATA_DIR}/environment_test.csv")

print(f"  corrosions:              {corrosions.shape}")
print(f"  environment_training:    {env_train.shape}")
print(f"  environment_test:        {env_test.shape}")

# Parse dates
corrosions["observation_date"] = pd.to_datetime(corrosions["observation_date"])
corrosions["obs_year_month"] = corrosions["observation_date"].dt.strftime("%Y-%m")

env_train["month_start_date"] = pd.to_datetime(env_train["month_start_date"])
env_test["month_start_date"] = pd.to_datetime(env_test["month_start_date"])

# Ensure year_month is sorted properly (YYYY-MM already lexicographically sortable)
env_train = env_train.sort_values(["aircraft_id", "year_month"]).reset_index(drop=True)
env_test = env_test.sort_values(["aircraft_id", "year_month"]).reset_index(drop=True)

# Only keep aircraft present in both corrosions and environment_training
valid_ids = set(corrosions["aircraft_id"]) & set(env_train["aircraft_id"])
print(f"\n  Aircraft in both files:  {len(valid_ids)}")
corrosions = corrosions[corrosions["aircraft_id"].isin(valid_ids)].copy()

# ---------------------------------------------------------------------------
# Helper: feature engineering on a raw environment dataframe
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    col for col in env_train.columns
    if col not in ("aircraft_id", "year_month", "month_start_date")
]

# Composite salt aerosol feature
SALT_COLS = [
    "sea_salt_aerosol_003_05_mixing_ratio",
    "sea_salt_aerosol_05_5_mixing_ratio",
    "sea_salt_aerosol_5_20_mixing_ratio",
]

# Key feature groups for cumulative & rolling transformations
PARKING_COL = "total_parking_minutes"
HUMIDITY_COLS = [
    "metar_relative_humidity",
    "specific_humidity",
]
SULFUR_COLS = [
    "sulphate_aerosol_mixing_ratio",
    "sulphur_dioxide_mass_mixing_ratio",
]


def engineer_features(df_raw, corrosions_ref=None):
    """
    Feature engineering on a chronologically sorted environment dataframe.
    - Composite salt creation
    - Cumulative sums and rolling means (3, 6, 12 months) for key features
    - months_since_delivery
    - No look-ahead bias: rolling uses .shift(1) to exclude current month

    If corrosions_ref is provided, also merges delivery info (train)
    otherwise infers delivery from the first available month (test).
    """
    df = df_raw.copy()

    # --- Merge delivery info ---
    if corrosions_ref is not None:
        df = df.merge(
            corrosions_ref[["aircraft_id", "aircraft_delivery_year", "aircraft_delivery_month"]],
            on="aircraft_id",
            how="left",
        )
    else:
        # Test set: infer delivery from first month of data
        delivery_info = (
            df.groupby("aircraft_id")["year_month"]
            .min()
            .reset_index()
        )
        delivery_info["delivery_year"] = delivery_info["year_month"].str[:4].astype(int)
        delivery_info["delivery_month"] = delivery_info["year_month"].str[5:7].astype(int)
        delivery_info = delivery_info.rename(
            columns={"delivery_year": "aircraft_delivery_year", "delivery_month": "aircraft_delivery_month"}
        )
        df = df.merge(delivery_info[["aircraft_id", "aircraft_delivery_year", "aircraft_delivery_month"]],
                      on="aircraft_id", how="left")

    yr = df["year_month"].str[:4].astype(int)
    mo = df["year_month"].str[5:7].astype(int)
    df["months_since_delivery"] = (
        (yr - df["aircraft_delivery_year"]) * 12
        + (mo - df["aircraft_delivery_month"])
    )
    df["month_number"] = mo  # seasonality signal

    # --- Composite salt ---
    for c in SALT_COLS:
        if c not in df.columns:
            df[c] = 0  # safety guard
    df["total_sea_salt"] = df[SALT_COLS].sum(axis=1)

    # --- Cumulative sums (with shift to avoid including current month) ---
    # We apply .shift(1) so cumulative only includes past months.
    # For the very first row of each aircraft, shift produces NaN -> fill with 0.
    def cumsum_shifted(series, name):
        shifted = series.groupby(df["aircraft_id"]).transform(lambda x: x.shift(1))
        shifted = shifted.fillna(0)
        return shifted.groupby(df["aircraft_id"]).transform("cumsum").rename(name)

    for label, col in [
        ("parking_cum", PARKING_COL),
        ("salt_cum", "total_sea_salt"),
    ]:
        df[label] = cumsum_shifted(df[col], label)

    for hum_col in HUMIDITY_COLS:
        df[f"{hum_col}_cum"] = cumsum_shifted(df[hum_col], f"{hum_col}_cum")

    for sul_col in SULFUR_COLS:
        df[f"{sul_col}_cum"] = cumsum_shifted(df[sul_col], f"{sul_col}_cum")

    # --- Rolling means (3, 6, 12 months, also shifted to exclude current month) ---
    for window in [3, 6, 12]:
        for label, col in [
            ("parking", PARKING_COL),
            ("salt", "total_sea_salt"),
        ]:
            df[f"{label}_roll{window}m"] = (
                df.groupby("aircraft_id")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
        for hum_col in HUMIDITY_COLS:
            df[f"{hum_col}_roll{window}m"] = (
                df.groupby("aircraft_id")[hum_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
        for sul_col in SULFUR_COLS:
            df[f"{sul_col}_roll{window}m"] = (
                df.groupby("aircraft_id")[sul_col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

    # --- Ratio features (current vs rolling average) ---
    for window in [3, 6]:
        df[f"parking_vs_roll{window}m"] = df[PARKING_COL] / (df[f"parking_roll{window}m"] + 1e-8)
        df[f"salt_vs_roll{window}m"] = df["total_sea_salt"] / (df[f"salt_roll{window}m"] + 1e-8)

    # Drop ID cols
    for c in ["aircraft_delivery_year", "aircraft_delivery_month"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df


# ---------------------------------------------------------------------------
# 2. Feature Engineering (on full timelines before filtering!)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Feature engineering on full timelines")
print("=" * 60)

env_train_fe = engineer_features(env_train, corrosions_ref=corrosions)
env_test_fe = engineer_features(env_test, corrosions_ref=None)  # no delivery info in test

print(f"  env_train_fe: {env_train_fe.shape}")
print(f"  env_test_fe:  {env_test_fe.shape}")

# ---------------------------------------------------------------------------
# 3. Target Construction
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Target construction")
print("=" * 60)

# Merge observation year_month for each aircraft
corrosions_targets = corrosions[["aircraft_id", "obs_year_month"]].drop_duplicates()

# Label rows: positive (corrosion month) and negative (24 months prior)
corrosions_targets["target_1"] = corrosions_targets["obs_year_month"]

# Compute the year_month 24 months prior
obs_dt = pd.to_datetime(corrosions_targets["obs_year_month"] + "-01")
obs_dt_24m_ago = obs_dt - pd.DateOffset(months=24)
corrosions_targets["target_0"] = obs_dt_24m_ago.dt.strftime("%Y-%m")

# Build the training set by merging positive and negative rows
env_train_fe_with_target = env_train_fe.merge(
    corrosions_targets, on="aircraft_id", how="left"
)

# Positive rows: match year_month == target_1
mask_pos = env_train_fe_with_target["year_month"] == env_train_fe_with_target["target_1"]
# Negative rows: match year_month == target_0
mask_neg = env_train_fe_with_target["year_month"] == env_train_fe_with_target["target_0"]

# Combine
train_data_raw = pd.concat([
    env_train_fe_with_target[mask_pos].assign(corrosion_risk=1),
    env_train_fe_with_target[mask_neg].assign(corrosion_risk=0),
], axis=0).reset_index(drop=True)

print(f"  Positive matches found:  {mask_pos.sum()}")
print(f"  Negative matches found:  {mask_neg.sum()}")

# Keep only aircraft that have BOTH reference months available
aircraft_pos = set(train_data_raw[train_data_raw["corrosion_risk"] == 1]["aircraft_id"])
aircraft_neg = set(train_data_raw[train_data_raw["corrosion_risk"] == 0]["aircraft_id"])
complete_aircraft = aircraft_pos & aircraft_neg

train_data = train_data_raw[train_data_raw["aircraft_id"].isin(complete_aircraft)].reset_index(drop=True)

print(f"  Complete aircraft (both ref months): {len(complete_aircraft)}")
print(f"  Total training rows: {train_data.shape[0]}")

# Verify the 2-rows-per-aircraft constraint
counts = train_data.groupby("aircraft_id").size()
assert (counts == 2).all(), \
    f"ERROR: Not all aircraft have exactly 2 rows! Min={counts.min()}, Max={counts.max()}"
print("  Verified: each aircraft has exactly 2 rows.")

# ---------------------------------------------------------------------------
# 4. Prepare feature matrices
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Preparing feature matrices")
print("=" * 60)

exclude_cols = [
    "aircraft_id", "year_month", "month_start_date",
    "obs_year_month", "target_1", "target_0", "corrosion_risk"
]
feature_cols = [c for c in train_data.columns if c not in exclude_cols]

# Encode aircraft_id to an integer group ID for GroupKFold
le = LabelEncoder()
train_data["aircraft_group"] = le.fit_transform(train_data["aircraft_id"].astype(str))

X_train = train_data[feature_cols].copy()
y_train = train_data["corrosion_risk"].values
groups_train = train_data["aircraft_group"].values

# Impute missing values in both train and test
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index
)

X_test_raw = env_test_fe[feature_cols].copy()
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test_raw), columns=feature_cols, index=X_test_raw.index
)

print(f"  X_train: {X_train_imputed.shape}  |  y_train: {len(y_train)}")
print(f"  X_test:  {X_test_imputed.shape}")
print(f"  Class balance: 1={y_train.sum()} ({(y_train.sum() / len(y_train)):.1%}), 0={(1 - y_train).sum()}")

# ---------------------------------------------------------------------------
# 5. Cross-Validation with Calibration
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: GroupKFold CV + CalibratedClassifierCV")
print("=" * 60)

gkf = GroupKFold(n_splits=N_SPLITS)
oof_preds = np.zeros(len(y_train))

lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 5,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
    "force_col_wise": True,
}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_imputed, y_train, groups=groups_train)):
    X_tr = X_train_imputed.iloc[train_idx]
    X_val = X_train_imputed.iloc[val_idx]
    y_tr = y_train[train_idx]
    y_val = y_train[val_idx]
    groups_tr = groups_train[train_idx]

    # Inner GroupKFold for calibration (keeps aircraft together)
    inner_gkf = GroupKFold(n_splits=3)
    inner_splits = list(inner_gkf.split(X_tr, y_tr, groups=groups_tr))

    base_model = lgb.LGBMClassifier(**lgb_params)
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=inner_splits,
        n_jobs=-1,
    )
    calibrated.fit(X_tr, y_tr)

    oof_preds[val_idx] = calibrated.predict_proba(X_val)[:, 1]

brier_cv = brier_score_loss(y_train, oof_preds)
print(f"\n  >>> Out-of-fold Brier Score: {brier_cv:.6f}")
print(f"  >>> Baseline (predict 0.5):  {0.25:.6f}")
print(f"  >>> Improvement over baseline: {0.25 - brier_cv:.6f}")

# ---------------------------------------------------------------------------
# 6. Train final model on all training data
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Training final calibrated model on all training data")
print("=" * 60)

final_gkf = GroupKFold(n_splits=3)
final_cv_splits = list(final_gkf.split(X_train_imputed, y_train, groups=groups_train))

final_model = CalibratedClassifierCV(
    estimator=lgb.LGBMClassifier(**lgb_params),
    method="isotonic",
    cv=final_cv_splits,
    n_jobs=-1,
)
final_model.fit(X_train_imputed, y_train)

# Check training Brier (on OOF from CalibratedClassifierCV internal CV)
train_preds = final_model.predict_proba(X_train_imputed)[:, 1]
brier_train = brier_score_loss(y_train, train_preds)
print(f"  Training Brier Score: {brier_train:.6f}")

# ---------------------------------------------------------------------------
# 7. Inference on test set
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: Inference on test set")
print("=" * 60)

test_proba = final_model.predict_proba(X_test_imputed)[:, 1]

# Build submission
submission = pd.DataFrame({
    "id": env_test_fe["aircraft_id"].astype(str) + "_" + env_test_fe["year_month"].astype(str),
    "corrosion_risk": test_proba,
})

print(f"  Predictions generated for {len(submission)} rows")
print(f"  Test aircraft: {env_test_fe.aircraft_id.nunique()}")
print(f"  Prediction stats: min={test_proba.min():.4f}, max={test_proba.max():.4f}, mean={test_proba.mean():.4f}")

# ---------------------------------------------------------------------------
# 8. Save submission
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: Saving submission.csv")
print("=" * 60)

submission.to_csv("submission.csv", index=False)
print("  Saved submission.csv")
print(f"\n  First 5 rows:\n{submission.head()}")
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
