"""
Module d'optimisation des hyperparametres avec Optuna.
"""
import optuna
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import roc_auc_score
import numpy as np


def optimize_lightgbm(X_train, y_train, X_valid, y_valid, n_trials=100, seed=42):
    """Optimise les hyperparametres de LightGBM."""
    
    print("\n" + "=" * 80)
    print("OPTIMIZING LIGHTGBM HYPERPARAMETERS")
    print("=" * 80)
    
    def objective(trial):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.01),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 0.1, log=True),
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        
        pred = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, pred)
    
    study = optuna.create_study(direction="maximize", study_name="lightgbm_optimization")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest AUC: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    return study.best_params


def optimize_xgboost(X_train, y_train, X_valid, y_valid, n_trials=100, seed=42):
    """Optimise les hyperparametres de XGBoost."""
    
    print("\n" + "=" * 80)
    print("OPTIMIZING XGBOOST HYPERPARAMETERS")
    print("=" * 80)
    
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.01),
            "scale_pos_weight": 1.0,
            "random_state": seed,
            "n_jobs": -1,
            "early_stopping_rounds": 100,
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        
        pred = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, pred)
    
    study = optuna.create_study(direction="maximize", study_name="xgboost_optimization")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest AUC: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    return study.best_params


def optimize_catboost(X_train, y_train, X_valid, y_valid, n_trials=100, seed=42):
    """Optimise les hyperparametres de CatBoost."""
    
    print("\n" + "=" * 80)
    print("OPTIMIZING CATBOOST HYPERPARAMETERS")
    print("=" * 80)
    
    def objective(trial):
        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "depth": trial.suggest_int("depth", 5, 12),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.0, 2.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "random_seed": seed,
            "thread_count": -1,
            "early_stopping_rounds": 100,
            "verbose": False,
        }
        
        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            verbose=False,
        )
        
        pred = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, pred)
    
    study = optuna.create_study(direction="maximize", study_name="catboost_optimization")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest AUC: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    return study.best_params


def optimize_all_models(X_train, y_train, X_valid, y_valid, n_trials=100, seed=42):
    """Optimise les hyperparametres de tous les modeles."""
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZATION - ALL MODELS")
    print(f"Trials per model: {n_trials}")
    print("=" * 80)
    
    # Optimiser chaque modele
    lgb_params = optimize_lightgbm(X_train, y_train, X_valid, y_valid, n_trials, seed)
    xgb_params = optimize_xgboost(X_train, y_train, X_valid, y_valid, n_trials, seed)
    cb_params = optimize_catboost(X_train, y_train, X_valid, y_valid, n_trials, seed)
    
    # Ajouter les parametres fixes
    lgb_params.update({
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    })
    
    xgb_params.update({
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 2000,
        "scale_pos_weight": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": 100,
    })
    
    cb_params.update({
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 2000,
        "random_seed": seed,
        "thread_count": -1,
        "early_stopping_rounds": 100,
        "verbose": False,
    })
    
    return lgb_params, xgb_params, cb_params


def save_best_params(lgb_params, xgb_params, cb_params, output_path):
    """Sauvegarde les meilleurs parametres dans un fichier."""
    import json
    
    params = {
        "lightgbm": lgb_params,
        "xgboost": xgb_params,
        "catboost": cb_params,
    }
    
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)
    
    print(f"\nBest parameters saved to: {output_path}")


def load_best_params(input_path):
    """Charge les meilleurs parametres depuis un fichier."""
    import json
    
    with open(input_path, "r") as f:
        params = json.load(f)
    
    return params["lightgbm"], params["xgboost"], params["catboost"]

# Made with Bob
