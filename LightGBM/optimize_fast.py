"""
Optimisation rapide pour 30 minutes avec ensemble.
Strategie: Ensemble simple sans Optuna + plus de features + calibration
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from config import PathConfig, ModelConfig, ValidationConfig
from data import build_training_table, load_environment, load_labels
from features import build_history_feature_table
from training import split_by_aircraft


def main():
    print("\n" + "=" * 80)
    print("OPTIMISATION RAPIDE - 30 MINUTES")
    print("Strategie: Ensemble LightGBM + XGBoost + Plus de features")
    print("=" * 80)
    
    paths = PathConfig()
    model_config = ModelConfig()
    validation_config = ValidationConfig()
    
    # Charger les donnees d'entrainement
    print("\n[1/5] Chargement des donnees...")
    environment_train = load_environment(paths.environment_path)
    labels = load_labels(paths.labels_path)
    
    # Feature engineering SANS selection (garder toutes les features)
    print("[2/5] Feature engineering (TOUTES les features)...")
    feature_table_train = build_history_feature_table(environment_train)
    X, y, groups, feature_columns = build_training_table(
        environment_train,
        labels,
        feature_table_train,
    )
    
    print(f"  Dataset: {len(X):,} lignes, {len(feature_columns)} features")
    print("  Strategie: Garder TOUTES les features pour maximiser l'information")
    
    # Split pour validation
    train_idx, valid_idx = split_by_aircraft(
        X, y, groups, validation_config, model_config.seed
    )
    
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]
    
    print(f"  Train: {len(X_train):,}, Valid: {len(X_valid):,}")
    
    # Entrainer 3 modeles avec parametres differents
    print("\n[3/5] Entrainement de l'ensemble (3 modeles)...")
    
    # Modele 1: LightGBM conservateur (moins d'overfitting)
    print("  [1/3] LightGBM conservateur...")
    lgb_params1 = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 100,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    lgb_model1 = lgb.LGBMRegressor(**lgb_params1)
    lgb_model1.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    pred_lgb1 = lgb_model1.predict(X_valid)
    mse_lgb1 = mean_squared_error(y_valid, pred_lgb1)
    print(f"    MSE: {mse_lgb1:.6f}, Iterations: {lgb_model1.best_iteration_}")
    
    # Modele 2: LightGBM agressif (plus de capacite)
    print("  [2/3] LightGBM agressif...")
    lgb_params2 = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.5,
        "random_state": 43,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    lgb_model2 = lgb.LGBMRegressor(**lgb_params2)
    lgb_model2.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    pred_lgb2 = lgb_model2.predict(X_valid)
    mse_lgb2 = mean_squared_error(y_valid, pred_lgb2)
    print(f"    MSE: {mse_lgb2:.6f}, Iterations: {lgb_model2.best_iteration_}")
    
    # Modele 3: XGBoost
    print("  [3/3] XGBoost...")
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 7,
        "min_child_weight": 80,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.3,
        "reg_lambda": 0.7,
        "random_state": 44,
        "n_jobs": -1,
    }
    
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    pred_xgb = xgb_model.predict(X_valid)
    mse_xgb = mean_squared_error(y_valid, pred_xgb)
    print(f"    MSE: {mse_xgb:.6f}, Iterations: {xgb_model.best_iteration}")
    
    # Ensemble avec poids optimaux
    print("\n[4/5] Optimisation des poids de l'ensemble...")
    best_mse = float('inf')
    best_weights = None
    
    # Recherche rapide des meilleurs poids
    for w1 in [0.3, 0.4, 0.5, 0.6]:
        for w2 in [0.2, 0.3, 0.4]:
            w3 = 1.0 - w1 - w2
            if w3 < 0 or w3 > 1:
                continue
            
            pred_ensemble = w1 * pred_lgb1 + w2 * pred_lgb2 + w3 * pred_xgb
            mse_ensemble = mean_squared_error(y_valid, pred_ensemble)
            
            if mse_ensemble < best_mse:
                best_mse = mse_ensemble
                best_weights = (w1, w2, w3)
    
    print(f"  Meilleurs poids: LGB1={best_weights[0]:.2f}, LGB2={best_weights[1]:.2f}, XGB={best_weights[2]:.2f}")
    print(f"  MSE ensemble: {best_mse:.6f}")
    
    # Charger les donnees de test
    print("\n[5/5] Generation des predictions finales...")
    test_env_path = Path("../../data/environment_test.csv")
    if not test_env_path.exists():
        test_env_path = Path("data/environment_test.csv")
    
    environment_test = pd.read_csv(test_env_path)
    feature_table_test = build_history_feature_table(environment_test)
    X_test = feature_table_test[feature_columns]
    
    # Predictions avec ensemble
    pred_test_lgb1 = lgb_model1.predict(X_test)
    pred_test_lgb2 = lgb_model2.predict(X_test)
    pred_test_xgb = xgb_model.predict(X_test)
    
    pred_test_ensemble = (
        best_weights[0] * pred_test_lgb1 +
        best_weights[1] * pred_test_lgb2 +
        best_weights[2] * pred_test_xgb
    )
    
    # Clipper les predictions entre 0 et 1
    pred_test_ensemble = np.clip(pred_test_ensemble, 0, 1)
    
    # Creer le fichier de soumission
    submission = pd.DataFrame({
        'id': feature_table_test['aircraft_id'].astype(str) + '_' + feature_table_test['year_month'].astype(str),
        'corrosion_risk': pred_test_ensemble
    })
    
    output_path = Path("submission_optimized.csv")
    submission.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("OPTIMISATION TERMINEE !")
    print("=" * 80)
    print(f"\nFichier: {output_path}")
    print(f"Predictions: {len(submission):,}")
    print(f"\nStatistiques:")
    print(f"  Moyenne: {pred_test_ensemble.mean():.4f}")
    print(f"  Std:     {pred_test_ensemble.std():.4f}")
    print(f"  Min:     {pred_test_ensemble.min():.4f}")
    print(f"  Max:     {pred_test_ensemble.max():.4f}")
    print(f"\nMSE validation ensemble: {best_mse:.6f}")
    print(f"Poids: LGB1={best_weights[0]:.2f}, LGB2={best_weights[1]:.2f}, XGB={best_weights[2]:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
