import argparse
from dataclasses import replace
from pathlib import Path

from config import ModelConfig, PathConfig, ValidationConfig
from data import build_training_table, load_environment, load_labels
from features import build_history_feature_table
from training import (
    get_feature_importance,
    run_grouped_cross_validation,
    split_by_aircraft,
    train_holdout_model,
)


def parse_args():
    path_defaults = PathConfig()
    model_defaults = ModelConfig()
    validation_defaults = ValidationConfig()

    parser = argparse.ArgumentParser(
        description="Train LightGBM avec features historiques par avion."
    )
    parser.add_argument("--environment", type=Path, default=path_defaults.environment_path)
    parser.add_argument("--labels", type=Path, default=path_defaults.labels_path)
    parser.add_argument("--model-output", type=Path, default=path_defaults.model_output_path)
    parser.add_argument(
        "--importance-output",
        type=Path,
        default=path_defaults.importance_output_path,
    )
    parser.add_argument("--valid-size", type=float, default=validation_defaults.valid_size)
    parser.add_argument("--cv-splits", type=int, default=validation_defaults.cv_splits)
    parser.add_argument("--n-estimators", type=int, default=model_defaults.n_estimators)
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=model_defaults.early_stopping_rounds,
    )
    parser.add_argument("--learning-rate", type=float, default=model_defaults.learning_rate)
    parser.add_argument("--num-leaves", type=int, default=model_defaults.num_leaves)
    parser.add_argument("--seed", type=int, default=model_defaults.seed)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def build_configs(args):
    paths = PathConfig(
        environment_path=args.environment,
        labels_path=args.labels,
        model_output_path=args.model_output,
        importance_output_path=args.importance_output,
    )
    model = replace(
        ModelConfig(),
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        seed=args.seed,
    )
    validation = ValidationConfig(
        valid_size=args.valid_size,
        cv_splits=args.cv_splits,
    )
    return paths, model, validation


def print_cv_report(cv_results):
    print("\nCross-validation par avion")
    for row in cv_results.itertuples(index=False):
        print(
            f"Fold {row.fold}: "
            f"AUC={row.auc:.4f}, "
            f"AP={row.average_precision:.4f}, "
            f"logloss={row.log_loss:.4f}, "
            f"MSE={row.mse:.6f}, "
            f"best_iter={row.best_iteration}, "
            f"valid_aircraft={row.valid_aircraft}"
        )

    print("\nCV mean/std:")
    print(cv_results[["auc", "average_precision", "log_loss", "mse"]].agg(["mean", "std"]))


def save_outputs(model, feature_importance, paths):
    paths.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    paths.importance_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(paths.model_output_path)
    feature_importance.to_csv(paths.importance_output_path, index=False)
    print(f"\nModele sauvegarde: {paths.model_output_path}")
    print(f"Importances sauvegardees: {paths.importance_output_path}")


def main():
    args = parse_args()
    paths, model_config, validation_config = build_configs(args)

    print("Chargement des donnees...")
    environment = load_environment(paths.environment_path)
    labels = load_labels(paths.labels_path)

    print("Feature engineering historique...")
    feature_table = build_history_feature_table(environment)
    X, y, groups, feature_columns = build_training_table(
        environment,
        labels,
        feature_table,
    )
    print(
        f"Dataset: {len(X)} lignes, {len(feature_columns)} features, "
        f"{groups.nunique()} avions, positif={y.mean():.3f}"
    )

    train_idx, valid_idx = split_by_aircraft(
        X,
        y,
        groups,
        validation_config,
        model_config.seed,
    )
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    groups_train = groups.iloc[train_idx]

    cv_results = run_grouped_cross_validation(
        X_train,
        y_train,
        groups_train,
        validation_config,
        model_config,
    )
    print_cv_report(cv_results)

    print("\nValidation holdout 20% avions")
    model, train_metrics, valid_metrics = train_holdout_model(
        X,
        y,
        train_idx,
        valid_idx,
        model_config,
    )
    print(
        f"\nTrain Set: "
        f"AUC={train_metrics['auc']:.4f}, "
        f"AP={train_metrics['average_precision']:.4f}, "
        f"logloss={train_metrics['log_loss']:.4f}, "
        f"MSE={train_metrics['mse']:.6f}"
    )
    print(
        f"Validation Set: "
        f"AUC={valid_metrics['auc']:.4f}, "
        f"AP={valid_metrics['average_precision']:.4f}, "
        f"logloss={valid_metrics['log_loss']:.4f}, "
        f"MSE={valid_metrics['mse']:.6f}, "
        f"best_iter={valid_metrics['best_iteration']}"
    )
    print(
        f"Avions train={groups.iloc[train_idx].nunique()}, "
        f"validation={groups.iloc[valid_idx].nunique()}"
    )

    feature_importance = get_feature_importance(model, feature_columns)
    print("\nTop 20 features:")
    print(feature_importance.head(20).to_string(index=False))

    if not args.no_save:
        save_outputs(model, feature_importance, paths)


if __name__ == "__main__":
    main()
