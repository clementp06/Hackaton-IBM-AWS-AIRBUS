import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def build_model(model_config):
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": model_config.n_estimators,
        "learning_rate": model_config.learning_rate,
        "num_leaves": model_config.num_leaves,
        "min_child_samples": model_config.min_child_samples,
        "subsample": model_config.subsample,
        "subsample_freq": 1,
        "colsample_bytree": model_config.colsample_bytree,
        "reg_alpha": model_config.reg_alpha,
        "reg_lambda": model_config.reg_lambda,
        "class_weight": "balanced",
        "random_state": model_config.seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    # Ajouter les paramètres optionnels s'ils existent
    if hasattr(model_config, 'max_depth'):
        params["max_depth"] = model_config.max_depth
    if hasattr(model_config, 'min_split_gain'):
        params["min_split_gain"] = model_config.min_split_gain
    if hasattr(model_config, 'min_child_weight'):
        params["min_child_weight"] = model_config.min_child_weight
    
    return lgb.LGBMClassifier(**params)


def fit_model(model, X_train, y_train, X_valid, y_valid, early_stopping_rounds):
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def evaluate_binary_classifier(y_true, proba):
    return {
        "auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, proba),
        "mse": mean_squared_error(y_true, proba),
    }


def split_by_aircraft(X, y, groups, validation_config, seed):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_config.valid_size,
        random_state=seed,
    )
    return next(splitter.split(X, y, groups))


def run_grouped_cross_validation(
    X,
    y,
    groups,
    validation_config,
    model_config,
):
    splitter = StratifiedGroupKFold(
        n_splits=validation_config.cv_splits,
        shuffle=True,
        random_state=model_config.seed,
    )
    rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
        fold_config = model_config.__class__(
            **{
                **model_config.__dict__,
                "seed": model_config.seed + fold,
            }
        )
        model = build_model(fold_config)
        fit_model(
            model,
            X.iloc[train_idx],
            y.iloc[train_idx],
            X.iloc[valid_idx],
            y.iloc[valid_idx],
            model_config.early_stopping_rounds,
        )

        proba = model.predict_proba(X.iloc[valid_idx])[:, 1]
        row = evaluate_binary_classifier(y.iloc[valid_idx], proba)
        row["fold"] = fold
        row["best_iteration"] = model.best_iteration_
        row["valid_aircraft"] = groups.iloc[valid_idx].nunique()
        rows.append(row)

    return pd.DataFrame(rows)


def train_holdout_model(X, y, train_idx, valid_idx, model_config):
    model = build_model(model_config)
    fit_model(
        model,
        X.iloc[train_idx],
        y.iloc[train_idx],
        X.iloc[valid_idx],
        y.iloc[valid_idx],
        model_config.early_stopping_rounds,
    )
    # Evaluate on validation set
    proba_valid = model.predict_proba(X.iloc[valid_idx])[:, 1]
    metrics_valid = evaluate_binary_classifier(y.iloc[valid_idx], proba_valid)
    metrics_valid["best_iteration"] = model.best_iteration_
    
    # Evaluate on train set
    proba_train = model.predict_proba(X.iloc[train_idx])[:, 1]
    metrics_train = evaluate_binary_classifier(y.iloc[train_idx], proba_train)
    
    return model, metrics_train, metrics_valid


def get_feature_importance(model, feature_columns):
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
