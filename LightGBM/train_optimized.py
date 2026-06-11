import argparse
from dataclasses import replace
from pathlib import Path

from config import ModelConfig, PathConfig, ValidationConfig
from data import build_training_table, load_environment, load_labels
from features import build_history_feature_table
from feature_selection import select_features_smart, analyze_feature_categories
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
        description="Train LightGBM avec feature selection intelligente."
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
    
    # Feature selection parameters
    parser.add_argument("--top-n-features", type=int, default=50, 
                       help="Nombre maximum de features à garder")
    parser.add_argument("--min-gain", type=float, default=800,
                       help="Gain minimum requis pour une feature")
    parser.add_argument("--corr-threshold", type=float, default=0.95,
                       help="Seuil de corrélation pour élimination")
    parser.add_argument("--use-feature-selection", action="store_true",
                       help="Activer la sélection de features")
    
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
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION PAR AVION")
    print("=" * 80)
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

    print("\nCV Statistics:")
    stats = cv_results[["auc", "average_precision", "log_loss", "mse"]].agg(["mean", "std"])
    print(stats)
    print("=" * 80)


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

    print("\n" + "=" * 80)
    print("LIGHTGBM TRAINING - VERSION OPTIMISEE")
    print("=" * 80)
    
    print("\nChargement des donnees...")
    environment = load_environment(paths.environment_path)
    labels = load_labels(paths.labels_path)

    print("Feature engineering historique...")
    feature_table = build_history_feature_table(environment)
    X, y, groups, feature_columns = build_training_table(
        environment,
        labels,
        feature_table,
    )
    
    print(f"\nDataset initial:")
    print(f"  - Lignes: {len(X):,}")
    print(f"  - Features: {len(feature_columns)}")
    print(f"  - Avions: {groups.nunique()}")
    print(f"  - Taux positif: {y.mean():.3f}")
    
    # Feature selection si activee
    if args.use_feature_selection:
        print("\n" + "=" * 80)
        print("FEATURE SELECTION ACTIVEE")
        print("=" * 80)
        
        selected_features = select_features_smart(
            importance_path=paths.importance_output_path,
            X=X,
            top_n=args.top_n_features,
            min_gain=args.min_gain,
            corr_threshold=args.corr_threshold
        )
        
        # Analyser les catégories
        analyze_feature_categories(selected_features)
        
        # Filtrer X
        X = X[selected_features]
        feature_columns = selected_features
        
        print(f"\nDataset apres selection:")
        print(f"  - Features: {len(feature_columns)} (reduction de {100 * (1 - len(feature_columns) / len(X.columns)):.1f}%)")
    
    # Split train/validation
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

    # Cross-validation
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION")
    print("=" * 80)
    cv_results = run_grouped_cross_validation(
        X_train,
        y_train,
        groups_train,
        validation_config,
        model_config,
    )
    print_cv_report(cv_results)

    # Validation holdout
    print("\n" + "=" * 80)
    print("VALIDATION HOLDOUT (20% avions)")
    print("=" * 80)
    model, train_metrics, valid_metrics = train_holdout_model(
        X,
        y,
        train_idx,
        valid_idx,
        model_config,
    )
    
    print(f"\nTrain Set:")
    print(f"  AUC:  {train_metrics['auc']:.4f}")
    print(f"  AP:   {train_metrics['average_precision']:.4f}")
    print(f"  Loss: {train_metrics['log_loss']:.4f}")
    print(f"  MSE:  {train_metrics['mse']:.6f}")
    
    print(f"\nValidation Set:")
    print(f"  AUC:  {valid_metrics['auc']:.4f}")
    print(f"  AP:   {valid_metrics['average_precision']:.4f}")
    print(f"  Loss: {valid_metrics['log_loss']:.4f}")
    print(f"  MSE:  {valid_metrics['mse']:.6f}")
    print(f"  Best iter: {valid_metrics['best_iteration']}")
    
    overfitting = train_metrics['auc'] - valid_metrics['auc']
    print(f"\nOverfitting (Train-Val AUC): {overfitting:.4f}")
    
    print(f"\nAvions:")
    print(f"  Train:      {groups.iloc[train_idx].nunique()}")
    print(f"  Validation: {groups.iloc[valid_idx].nunique()}")

    # Feature importance
    feature_importance = get_feature_importance(model, feature_columns)
    print("\n" + "=" * 80)
    print("TOP 20 FEATURES")
    print("=" * 80)
    print(feature_importance.head(20).to_string(index=False))

    if not args.no_save:
        save_outputs(model, feature_importance, paths)
    
    print("\n" + "=" * 80)
    print("TRAINING TERMINE")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
