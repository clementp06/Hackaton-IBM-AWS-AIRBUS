"""
SOLUTION SIMPLE ET RAPIDE - LightGBM seul
Top 30 features + LightGBM optimisé + Calibration Sigmoid
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

from features import build_history_feature_table
from data import split_prediction_id

# Chemins
DATA_DIR = Path("../../data")
CORR_PATH = DATA_DIR / "prediction_training.csv"
ENV_TRAIN_PATH = DATA_DIR / "environment_training.csv"
ENV_TEST_PATH = DATA_DIR / "environment_test.csv"

# Top 30 features
TOP_FEATURES = [
    'gold__aircraft_age_months', 'calendar_year', 'gold__dose_acceleration',
    'gold__temperature_range', 'gold__avg_temperature', 'platinum__chemical_dominance',
    'gold__humidity_stability', 'gold__critical_time_cumulative',
    'platinum__accelerated_aging_index', 'gold__atm_corrosion_avg_24m',
    'gold__avg_sea_salt_exposure', 'gold__chem_aggr_max_24m',
    'gold__avg_parking_hours', 'gold__long_parking_frequency',
    'gold__salt_exposure_avg_24m', 'gold__critical_exposure_last_24m',
    'gold__chem_aggr_max_12m', 'metar_relative_humidity__last_12_mean',
    'gold__critical_exposure_last_12m', 'platinum__aggressivity_streak',
    'total_parking_minutes__last_12_mean', 'gold__humidity_volatility_12m',
    'sea_salt_aerosol_003_05_mixing_ratio__last_12_mean', 'sci__biogenic_voc',
    'bronze__parking_hours', 'sci__sulfate_chloride_interaction',
    'sci__dust_moisture_trap', 'gold__salt_exposure_avg_6m',
    'sci__hno3_wet_film', 'sci__marine_industrial_sulfate',
]

print("\n" + "=" * 80)
print("SOLUTION SIMPLE - LightGBM Optimisé")
print("=" * 80)

# Charger
print("\n[1/4] Chargement...")
corr_df = pd.read_csv(CORR_PATH)
env_train = pd.read_csv(ENV_TRAIN_PATH)
env_test = pd.read_csv(ENV_TEST_PATH)

# Features
print("[2/4] Feature engineering...")
feature_table_train = build_history_feature_table(env_train)
feature_table_test = build_history_feature_table(env_test)

corr_df = split_prediction_id(corr_df)
training_data = feature_table_train.merge(
    corr_df[['aircraft_id', 'year_month', 'corrosion_risk']],
    on=['aircraft_id', 'year_month'], how='inner'
)

available_features = [f for f in TOP_FEATURES if f in training_data.columns]
print(f"  Features: {len(available_features)}")

X = training_data[available_features].values
y = training_data['corrosion_risk'].values

# Scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train
print("[3/4] Training LightGBM calibré...")
base_model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
    force_col_wise=True
)

calibrated_model = CalibratedClassifierCV(
    base_model, method="sigmoid", cv=5, n_jobs=-1
)
calibrated_model.fit(X_scaled, y)

# Predict
print("[4/4] Prédictions...")
X_test = feature_table_test[available_features].values
X_test_scaled = scaler.transform(X_test)
predictions = calibrated_model.predict_proba(X_test_scaled)[:, 1]

# Save
submission = pd.DataFrame({
    'id': feature_table_test['aircraft_id'] + '_' + feature_table_test['year_month'],
    'corrosion_risk': predictions
})

output_file = "submission_simple_lgb.csv"
submission.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("TERMINÉ !")
print("=" * 80)
print(f"\nFichier: {output_file}")
print(f"Moyenne: {predictions.mean():.4f}")
print(f"Std:     {predictions.std():.4f}")
print("=" * 80)

# Made with Bob
