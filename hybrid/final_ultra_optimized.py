"""
SOLUTION FINALE ULTRA-OPTIMISÉE
- Top 30 features seulement (basé sur gain)
- Ensemble de 3 modèles différents (LightGBM, XGBoost, HistGB)
- Calibration Sigmoid optimale
- Poids optimisés par CV
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier

from features import build_history_feature_table
from data import split_prediction_id

# Chemins
DATA_DIR = Path("../../data")
CORR_PATH = DATA_DIR / "prediction_training.csv"
ENV_TRAIN_PATH = DATA_DIR / "environment_training.csv"
ENV_TEST_PATH = DATA_DIR / "environment_test.csv"

# Top 30 features basées sur gain
TOP_FEATURES = [
    'gold__aircraft_age_months',
    'calendar_year',
    'gold__dose_acceleration',
    'gold__temperature_range',
    'gold__avg_temperature',
    'platinum__chemical_dominance',
    'gold__humidity_stability',
    'gold__critical_time_cumulative',
    'platinum__accelerated_aging_index',
    'gold__atm_corrosion_avg_24m',
    'gold__avg_sea_salt_exposure',
    'gold__chem_aggr_max_24m',
    'gold__avg_parking_hours',
    'gold__long_parking_frequency',
    'gold__salt_exposure_avg_24m',
    'gold__critical_exposure_last_24m',
    'gold__chem_aggr_max_12m',
    'metar_relative_humidity__last_12_mean',
    'gold__critical_exposure_last_12m',
    'platinum__aggressivity_streak',
    'total_parking_minutes__last_12_mean',
    'gold__humidity_volatility_12m',
    'sea_salt_aerosol_003_05_mixing_ratio__last_12_mean',
    'sci__biogenic_voc',
    'bronze__parking_hours',
    'sci__sulfate_chloride_interaction',
    'sci__dust_moisture_trap',
    'gold__salt_exposure_avg_6m',
    'sci__hno3_wet_film',
    'sci__marine_industrial_sulfate',
]


def train_lgb_calibrated(X, y):
    """LightGBM avec calibration Sigmoid."""
    base = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        force_col_wise=True
    )
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=-1)
    calibrated.fit(X, y)
    return calibrated


def train_xgb_calibrated(X, y):
    """XGBoost avec calibration Sigmoid."""
    base = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        tree_method='hist'
    )
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=-1)
    calibrated.fit(X, y)
    return calibrated


def train_histgb_calibrated(X, y):
    """HistGradientBoosting avec calibration Sigmoid."""
    base = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.03,
        max_depth=5,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30
    )
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=-1)
    calibrated.fit(X, y)
    return calibrated


def optimize_ensemble_weights(X, y, n_splits=5):
    """Optimise les poids de l'ensemble par CV."""
    print("\n[OPTIMISATION DES POIDS PAR CV]")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_preds_lgb = np.zeros(len(y))
    cv_preds_xgb = np.zeros(len(y))
    cv_preds_hist = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {fold}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train 3 models
        print("    Training LightGBM...")
        model_lgb = train_lgb_calibrated(X_train, y_train)
        cv_preds_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
        
        print("    Training XGBoost...")
        model_xgb = train_xgb_calibrated(X_train, y_train)
        cv_preds_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
        
        print("    Training HistGB...")
        model_hist = train_histgb_calibrated(X_train, y_train)
        cv_preds_hist[val_idx] = model_hist.predict_proba(X_val)[:, 1]
    
    # Compute individual Brier scores
    brier_lgb = brier_score_loss(y, cv_preds_lgb)
    brier_xgb = brier_score_loss(y, cv_preds_xgb)
    brier_hist = brier_score_loss(y, cv_preds_hist)
    
    print(f"\n  Brier scores individuels:")
    print(f"    LightGBM: {brier_lgb:.4f}")
    print(f"    XGBoost:  {brier_xgb:.4f}")
    print(f"    HistGB:   {brier_hist:.4f}")
    
    # Grid search for optimal weights
    print("\n  Recherche des poids optimaux...")
    best_brier = float('inf')
    best_weights = (1/3, 1/3, 1/3)
    
    for w1 in np.arange(0.0, 1.01, 0.1):
        for w2 in np.arange(0.0, 1.01 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            pred = w1 * cv_preds_lgb + w2 * cv_preds_xgb + w3 * cv_preds_hist
            brier = brier_score_loss(y, pred)
            
            if brier < best_brier:
                best_brier = brier
                best_weights = (w1, w2, w3)
    
    print(f"    Meilleurs poids: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, HistGB={best_weights[2]:.2f}")
    print(f"    Meilleur Brier: {best_brier:.4f}")
    
    return best_weights


def main():
    print("\n" + "=" * 80)
    print("SOLUTION FINALE ULTRA-OPTIMISÉE")
    print("=" * 80)
    
    # Charger données
    print("\n[1/5] Chargement des données...")
    corr_df = pd.read_csv(CORR_PATH)
    env_train = pd.read_csv(ENV_TRAIN_PATH)
    env_test = pd.read_csv(ENV_TEST_PATH)
    
    print(f"  Samples: {len(corr_df):,}")
    
    # Feature engineering
    print("\n[2/5] Feature engineering (top 30 features)...")
    feature_table_train = build_history_feature_table(env_train)
    feature_table_test = build_history_feature_table(env_test)
    
    # Préparer données
    print("\n[3/5] Préparation des données...")
    corr_df = split_prediction_id(corr_df)
    
    training_data = feature_table_train.merge(
        corr_df[['aircraft_id', 'year_month', 'corrosion_risk']],
        on=['aircraft_id', 'year_month'],
        how='inner'
    )
    
    # Sélectionner top features disponibles
    available_features = [f for f in TOP_FEATURES if f in training_data.columns]
    print(f"  Features disponibles: {len(available_features)}/{len(TOP_FEATURES)}")
    
    X = training_data[available_features].values
    y = training_data['corrosion_risk'].values
    
    # Scaling
    print("\n[4/5] Scaling et optimisation...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Optimiser poids
    weights = optimize_ensemble_weights(X_scaled, y, n_splits=5)
    
    # Train final models
    print("\n  Training des modèles finaux...")
    print("    [1/3] LightGBM...")
    model_lgb = train_lgb_calibrated(X_scaled, y)
    
    print("    [2/3] XGBoost...")
    model_xgb = train_xgb_calibrated(X_scaled, y)
    
    print("    [3/3] HistGradientBoosting...")
    model_hist = train_histgb_calibrated(X_scaled, y)
    
    # Prédictions
    print("\n[5/5] Génération des prédictions...")
    X_test = feature_table_test[available_features].values
    X_test_scaled = scaler.transform(X_test)
    
    pred_lgb = model_lgb.predict_proba(X_test_scaled)[:, 1]
    pred_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]
    pred_hist = model_hist.predict_proba(X_test_scaled)[:, 1]
    
    predictions = weights[0] * pred_lgb + weights[1] * pred_xgb + weights[2] * pred_hist
    
    # Sauvegarder
    submission = pd.DataFrame({
        'id': feature_table_test['aircraft_id'] + '_' + feature_table_test['year_month'],
        'corrosion_risk': predictions
    })
    
    output_file = "submission_final_ultra_optimized.csv"
    submission.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("TERMINÉ !")
    print("=" * 80)
    print(f"\nFichier: {output_file}")
    print(f"Prédictions: {len(submission):,}")
    print(f"\nStatistiques:")
    print(f"  Moyenne: {predictions.mean():.4f}")
    print(f"  Std:     {predictions.std():.4f}")
    print(f"  Min:     {predictions.min():.4f}")
    print(f"  Max:     {predictions.max():.4f}")
    print("\nPoids optimaux:")
    print(f"  LightGBM: {weights[0]:.2f}")
    print(f"  XGBoost:  {weights[1]:.2f}")
    print(f"  HistGB:   {weights[2]:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
