"""
Module d'ensembling pour combiner LightGBM, XGBoost et CatBoost.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, mean_squared_error


class EnsembleModel:
    """Ensemble de modeles de gradient boosting."""
    
    def __init__(self, lgb_params=None, xgb_params=None, cb_params=None, weights=None):
        """
        Args:
            lgb_params: Parametres pour LightGBM
            xgb_params: Parametres pour XGBoost
            cb_params: Parametres pour CatBoost
            weights: Poids pour chaque modele (default: [0.4, 0.3, 0.3])
        """
        self.lgb_params = lgb_params or {}
        self.xgb_params = xgb_params or {}
        self.cb_params = cb_params or {}
        self.weights = weights or [0.4, 0.3, 0.3]
        
        self.lgb_model = None
        self.xgb_model = None
        self.cb_model = None
        
    def fit(self, X_train, y_train, X_valid, y_valid, verbose=True):
        """Entraine les 3 modeles."""
        if verbose:
            print("\n" + "=" * 80)
            print("TRAINING ENSEMBLE MODELS")
            print("=" * 80)
        
        # LightGBM
        if verbose:
            print("\n[1/3] Training LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        lgb_pred = self.lgb_model.predict_proba(X_valid)[:, 1]
        lgb_auc = roc_auc_score(y_valid, lgb_pred)
        if verbose:
            print(f"  LightGBM AUC: {lgb_auc:.4f} (best_iter: {self.lgb_model.best_iteration_})")
        
        # XGBoost
        if verbose:
            print("\n[2/3] Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        xgb_pred = self.xgb_model.predict_proba(X_valid)[:, 1]
        xgb_auc = roc_auc_score(y_valid, xgb_pred)
        if verbose:
            print(f"  XGBoost AUC: {xgb_auc:.4f} (best_iter: {self.xgb_model.best_iteration})")
        
        # CatBoost
        if verbose:
            print("\n[3/3] Training CatBoost...")
        self.cb_model = cb.CatBoostClassifier(**self.cb_params)
        self.cb_model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            verbose=False,
        )
        cb_pred = self.cb_model.predict_proba(X_valid)[:, 1]
        cb_auc = roc_auc_score(y_valid, cb_pred)
        if verbose:
            print(f"  CatBoost AUC: {cb_auc:.4f} (best_iter: {self.cb_model.best_iteration_})")
        
        # Ensemble
        ensemble_pred = self._weighted_average([lgb_pred, xgb_pred, cb_pred])
        ensemble_auc = roc_auc_score(y_valid, ensemble_pred)
        
        if verbose:
            print(f"\n  Ensemble AUC: {ensemble_auc:.4f}")
            print(f"  Weights: LGB={self.weights[0]:.2f}, XGB={self.weights[1]:.2f}, CB={self.weights[2]:.2f}")
            print("=" * 80)
        
        return self
    
    def predict_proba(self, X):
        """Predictions de l'ensemble."""
        lgb_pred = self.lgb_model.predict_proba(X)[:, 1]
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        cb_pred = self.cb_model.predict_proba(X)[:, 1]
        return self._weighted_average([lgb_pred, xgb_pred, cb_pred])
    
    def _weighted_average(self, predictions):
        """Moyenne ponderee des predictions."""
        return sum(w * p for w, p in zip(self.weights, predictions))
    
    def get_individual_predictions(self, X):
        """Retourne les predictions individuelles de chaque modele."""
        return {
            "lightgbm": self.lgb_model.predict_proba(X)[:, 1],
            "xgboost": self.xgb_model.predict_proba(X)[:, 1],
            "catboost": self.cb_model.predict_proba(X)[:, 1],
        }


def evaluate_ensemble(ensemble, X, y, set_name="Validation"):
    """Evalue l'ensemble et les modeles individuels."""
    print(f"\n{set_name} Set Evaluation:")
    print("-" * 60)
    
    # Predictions individuelles
    individual_preds = ensemble.get_individual_predictions(X)
    
    results = {}
    for name, pred in individual_preds.items():
        metrics = {
            "auc": roc_auc_score(y, pred),
            "ap": average_precision_score(y, pred),
            "logloss": log_loss(y, pred),
            "mse": mean_squared_error(y, pred),
        }
        results[name] = metrics
        print(f"{name:12s}: AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}, "
              f"Loss={metrics['logloss']:.4f}, MSE={metrics['mse']:.6f}")
    
    # Ensemble
    ensemble_pred = ensemble.predict_proba(X)
    ensemble_metrics = {
        "auc": roc_auc_score(y, ensemble_pred),
        "ap": average_precision_score(y, ensemble_pred),
        "logloss": log_loss(y, ensemble_pred),
        "mse": mean_squared_error(y, ensemble_pred),
    }
    results["ensemble"] = ensemble_metrics
    
    print("-" * 60)
    print(f"{'ENSEMBLE':12s}: AUC={ensemble_metrics['auc']:.4f}, AP={ensemble_metrics['ap']:.4f}, "
          f"Loss={ensemble_metrics['logloss']:.4f}, MSE={ensemble_metrics['mse']:.6f}")
    
    return results


def build_default_params(seed=42):
    """Construit les parametres par defaut pour chaque modele."""
    
    lgb_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 1200,
        "learning_rate": 0.015,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 60,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 0.8,
        "min_split_gain": 0.001,
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 1200,
        "learning_rate": 0.015,
        "max_depth": 7,
        "min_child_weight": 60,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 0.8,
        "gamma": 0.001,
        "scale_pos_weight": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }
    
    cb_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1200,
        "learning_rate": 0.015,
        "depth": 7,
        "l2_leaf_reg": 0.8,
        "subsample": 0.8,
        "colsample_bylevel": 0.8,
        "min_data_in_leaf": 60,
        "random_seed": seed,
        "thread_count": -1,
        "early_stopping_rounds": 50,
        "verbose": False,
    }
    
    return lgb_params, xgb_params, cb_params


def optimize_ensemble_weights(ensemble, X_valid, y_valid, n_trials=100):
    """Optimise les poids de l'ensemble sur le validation set."""
    import optuna
    
    print("\n" + "=" * 80)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("=" * 80)
    
    # Obtenir les predictions individuelles
    individual_preds = ensemble.get_individual_predictions(X_valid)
    lgb_pred = individual_preds["lightgbm"]
    xgb_pred = individual_preds["xgboost"]
    cb_pred = individual_preds["catboost"]
    
    def objective(trial):
        # Suggerer des poids qui somment a 1
        w1 = trial.suggest_float("w_lgb", 0.0, 1.0)
        w2 = trial.suggest_float("w_xgb", 0.0, 1.0 - w1)
        w3 = 1.0 - w1 - w2
        
        # Calculer la prediction ensemble
        ensemble_pred = w1 * lgb_pred + w2 * xgb_pred + w3 * cb_pred
        
        # Retourner l'AUC (a maximiser)
        return roc_auc_score(y_valid, ensemble_pred)
    
    study = optuna.create_study(direction="maximize", study_name="ensemble_weights")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_weights = [
        study.best_params["w_lgb"],
        study.best_params["w_xgb"],
        1.0 - study.best_params["w_lgb"] - study.best_params["w_xgb"],
    ]
    
    print(f"\nBest weights: LGB={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, CB={best_weights[2]:.3f}")
    print(f"Best AUC: {study.best_value:.4f}")
    print("=" * 80)
    
    return best_weights

# Made with Bob
