"""
Script d'entrainement avec ensemble et optimisation des hyperparametres.
Pipeline complete: 60% train / 20% valid / 20% test
"""
import argparse
from pathlib import Path
import json

from config import PathConfig
from data import build_training_table, load_environment, load_labels
from features import build_history_feature_table
from feature_selection import select_features_smart, analyze_feature_categories
from data_split import split_train_valid_test, print_split_info, get_split_datasets
from ensemble import EnsembleModel, evaluate_ensemble, build_default_params, optimize_ensemble_weights
from hyperopt import optimize_all_models, save_best_params, load_best_params


def parse_args():
    path_defaults = PathConfig()
    
    parser = argparse.ArgumentParser(
        description="Train ensemble avec optimisation des hyperparametres."
    )
    parser.add_argument("--environment", type=Path, default=path_defaults.environment_path)
    parser.add_argument("--labels", type=Path, default=path_defaults.labels_path)
    parser.add_argument("--seed", type=int, default=42)
    
    # Feature selection
    parser.add_argument("--use-feature-selection", action="store_true",
                       help="Activer la selection de features")
    parser.add_argument("--top-n-features", type=int, default=50)
    parser.add_argument("--min-gain", type=float, default=800)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    
    # Hyperparameter optimization
    parser.add_argument("--optimize-hyperparams", action="store_true",
                       help="Optimiser les hyperparametres avec Optuna")
    parser.add_argument("--n-trials", type=int, default=100,
                       help="Nombre de trials pour Optuna")
    parser.add_argument("--load-params", type=Path, default=None,
                       help="Charger les parametres depuis un fichier JSON")
    parser.add_argument("--save-params", type=Path, default=None,
                       help="Sauvegarder les parametres optimises")
    
    # Ensemble weights
    parser.add_argument("--optimize-weights", action="store_true",
                       help="Optimiser les poids de l'ensemble")
    parser.add_argument("--weights", type=float, nargs=3, default=[0.4, 0.3, 0.3],
                       help="Poids pour LightGBM, XGBoost, CatBoost")
    
    # Split configuration
    parser.add_argument("--train-size", type=float, default=0.6)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 80)
    print("ENSEMBLE TRAINING PIPELINE")
    print("=" * 80)
    print(f"Split: {args.train_size*100:.0f}% train / {args.valid_size*100:.0f}% valid / {args.test_size*100:.0f}% test")
    print(f"Feature selection: {args.use_feature_selection}")
    print(f"Hyperparameter optimization: {args.optimize_hyperparams}")
    print(f"Ensemble weight optimization: {args.optimize_weights}")
    print("=" * 80)
    
    # Charger les donnees
    print("\nChargement des donnees...")
    paths = PathConfig()
    environment = load_environment(paths.environment_path)
    labels = load_labels(paths.labels_path)
    
    # Feature engineering
    print("Feature engineering historique...")
    feature_table = build_history_feature_table(environment)
    X, y, groups, feature_columns = build_training_table(
        environment,
        labels,
        feature_table,
    )
    
    print(f"\nDataset initial:")
    print(f"  Lignes: {len(X):,}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Avions: {groups.nunique()}")
    print(f"  Taux positif: {y.mean():.3f}")
    
    # Feature selection
    if args.use_feature_selection:
        print("\n" + "=" * 80)
        print("FEATURE SELECTION")
        print("=" * 80)
        
        selected_features = select_features_smart(
            importance_path=paths.importance_output_path,
            X=X,
            top_n=args.top_n_features,
            min_gain=args.min_gain,
            corr_threshold=args.corr_threshold
        )
        
        analyze_feature_categories(selected_features)
        
        X = X[selected_features]
        feature_columns = selected_features
        
        print(f"\nDataset apres selection:")
        print(f"  Features: {len(feature_columns)}")
    
    # Split train/valid/test
    print("\n" + "=" * 80)
    print("SPLITTING DATA")
    print("=" * 80)
    train_idx, valid_idx, test_idx = split_train_valid_test(
        X, y, groups,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    
    print_split_info(X, y, groups, train_idx, valid_idx, test_idx)
    
    datasets = get_split_datasets(X, y, groups, train_idx, valid_idx, test_idx)
    X_train = datasets["train"]["X"]
    y_train = datasets["train"]["y"]
    X_valid = datasets["valid"]["X"]
    y_valid = datasets["valid"]["y"]
    X_test = datasets["test"]["X"]
    y_test = datasets["test"]["y"]
    
    # Optimisation des hyperparametres
    if args.optimize_hyperparams:
        print("\n" + "=" * 80)
        print("HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        
        lgb_params, xgb_params, cb_params = optimize_all_models(
            X_train, y_train, X_valid, y_valid,
            n_trials=args.n_trials,
            seed=args.seed,
        )
        
        if args.save_params:
            save_best_params(lgb_params, xgb_params, cb_params, args.save_params)
    
    elif args.load_params:
        print(f"\nChargement des parametres depuis: {args.load_params}")
        lgb_params, xgb_params, cb_params = load_best_params(args.load_params)
    
    else:
        print("\nUtilisation des parametres par defaut...")
        lgb_params, xgb_params, cb_params = build_default_params(args.seed)
    
    # Entrainement de l'ensemble
    print("\n" + "=" * 80)
    print("TRAINING ENSEMBLE")
    print("=" * 80)
    
    ensemble = EnsembleModel(
        lgb_params=lgb_params,
        xgb_params=xgb_params,
        cb_params=cb_params,
        weights=args.weights,
    )
    
    ensemble.fit(X_train, y_train, X_valid, y_valid, verbose=True)
    
    # Optimisation des poids
    if args.optimize_weights:
        print("\n" + "=" * 80)
        print("OPTIMIZING ENSEMBLE WEIGHTS")
        print("=" * 80)
        best_weights = optimize_ensemble_weights(ensemble, X_valid, y_valid, n_trials=100)
        ensemble.weights = best_weights
    
    # Evaluation sur tous les sets
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    train_results = evaluate_ensemble(ensemble, X_train, y_train, "Train")
    valid_results = evaluate_ensemble(ensemble, X_valid, y_valid, "Validation")
    test_results = evaluate_ensemble(ensemble, X_test, y_test, "Test")
    
    # Resume final
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nEnsemble AUC:")
    print(f"  Train:      {train_results['ensemble']['auc']:.4f}")
    print(f"  Validation: {valid_results['ensemble']['auc']:.4f}")
    print(f"  Test:       {test_results['ensemble']['auc']:.4f}")
    
    print("\nOverfitting:")
    print(f"  Train-Valid: {train_results['ensemble']['auc'] - valid_results['ensemble']['auc']:.4f}")
    print(f"  Train-Test:  {train_results['ensemble']['auc'] - test_results['ensemble']['auc']:.4f}")
    print(f"  Valid-Test:  {valid_results['ensemble']['auc'] - test_results['ensemble']['auc']:.4f}")
    
    print("\nEnsemble weights:")
    print(f"  LightGBM: {ensemble.weights[0]:.3f}")
    print(f"  XGBoost:  {ensemble.weights[1]:.3f}")
    print(f"  CatBoost: {ensemble.weights[2]:.3f}")
    
    print("\nIndividual model performance (Test set):")
    for model_name in ["lightgbm", "xgboost", "catboost"]:
        auc = test_results[model_name]["auc"]
        print(f"  {model_name:12s}: {auc:.4f}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    # Sauvegarder les resultats
    results_summary = {
        "train": {k: {m: float(v) for m, v in v.items()} for k, v in train_results.items()},
        "valid": {k: {m: float(v) for m, v in v.items()} for k, v in valid_results.items()},
        "test": {k: {m: float(v) for m, v in v.items()} for k, v in test_results.items()},
        "ensemble_weights": ensemble.weights,
        "n_features": len(feature_columns),
        "n_samples": {
            "train": len(X_train),
            "valid": len(X_valid),
            "test": len(X_test),
        },
    }
    
    results_path = Path("Hackaton-IBM-AWS-AIRBUS/LightGBM/ensemble_results.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResultats sauvegardes: {results_path}")


if __name__ == "__main__":
    main()

# Made with Bob
