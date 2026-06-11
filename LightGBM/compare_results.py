"""
Script pour comparer les resultats de differents modeles.
"""
import json
import pandas as pd
from pathlib import Path


def load_results(results_path):
    """Charge les resultats depuis un fichier JSON."""
    with open(results_path, "r") as f:
        return json.load(f)


def format_metrics(metrics):
    """Formate les metriques pour l'affichage."""
    return {
        "AUC": f"{metrics['auc']:.4f}",
        "AP": f"{metrics['ap']:.4f}",
        "LogLoss": f"{metrics['logloss']:.4f}",
        "MSE": f"{metrics['mse']:.6f}",
    }


def print_comparison(results):
    """Affiche une comparaison des resultats."""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    
    # Extraire les metriques
    models = ["lightgbm", "xgboost", "catboost", "ensemble"]
    sets = ["train", "valid", "test"]
    
    for set_name in sets:
        print(f"\n{set_name.upper()} SET:")
        print("-" * 100)
        
        data = []
        for model in models:
            metrics = results[set_name][model]
            row = {
                "Model": model.upper(),
                "AUC": f"{metrics['auc']:.4f}",
                "AP": f"{metrics['ap']:.4f}",
                "LogLoss": f"{metrics['logloss']:.4f}",
                "MSE": f"{metrics['mse']:.6f}",
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
    
    # Overfitting analysis
    print("\n" + "=" * 100)
    print("OVERFITTING ANALYSIS")
    print("=" * 100)
    
    ensemble_train = results["train"]["ensemble"]["auc"]
    ensemble_valid = results["valid"]["ensemble"]["auc"]
    ensemble_test = results["test"]["ensemble"]["auc"]
    
    print(f"\nEnsemble AUC:")
    print(f"  Train:      {ensemble_train:.4f}")
    print(f"  Validation: {ensemble_valid:.4f}")
    print(f"  Test:       {ensemble_test:.4f}")
    
    print(f"\nOverfitting:")
    print(f"  Train-Valid: {ensemble_train - ensemble_valid:+.4f}")
    print(f"  Train-Test:  {ensemble_train - ensemble_test:+.4f}")
    print(f"  Valid-Test:  {ensemble_valid - ensemble_test:+.4f}")
    
    # Ensemble weights
    print("\n" + "=" * 100)
    print("ENSEMBLE CONFIGURATION")
    print("=" * 100)
    
    weights = results["ensemble_weights"]
    print(f"\nWeights:")
    print(f"  LightGBM: {weights[0]:.3f}")
    print(f"  XGBoost:  {weights[1]:.3f}")
    print(f"  CatBoost: {weights[2]:.3f}")
    
    print(f"\nDataset:")
    print(f"  Features: {results['n_features']}")
    print(f"  Train samples:      {results['n_samples']['train']:,}")
    print(f"  Validation samples: {results['n_samples']['valid']:,}")
    print(f"  Test samples:       {results['n_samples']['test']:,}")
    
    # Best model per metric
    print("\n" + "=" * 100)
    print("BEST MODEL PER METRIC (Test Set)")
    print("=" * 100)
    
    metrics_names = ["auc", "ap", "logloss", "mse"]
    for metric in metrics_names:
        if metric in ["logloss", "mse"]:
            # Lower is better
            best_model = min(models, key=lambda m: results["test"][m][metric])
            best_value = results["test"][best_model][metric]
            print(f"\n{metric.upper()}: {best_model.upper()} ({best_value:.6f})")
        else:
            # Higher is better
            best_model = max(models, key=lambda m: results["test"][m][metric])
            best_value = results["test"][best_model][metric]
            print(f"\n{metric.upper()}: {best_model.upper()} ({best_value:.4f})")
    
    print("\n" + "=" * 100)


def compare_multiple_runs(results_paths):
    """Compare plusieurs runs."""
    print("\n" + "=" * 100)
    print("MULTIPLE RUNS COMPARISON")
    print("=" * 100)
    
    runs = []
    for path in results_paths:
        if Path(path).exists():
            results = load_results(path)
            run_name = Path(path).stem
            runs.append({
                "name": run_name,
                "test_auc": results["test"]["ensemble"]["auc"],
                "valid_auc": results["valid"]["ensemble"]["auc"],
                "overfitting": results["train"]["ensemble"]["auc"] - results["test"]["ensemble"]["auc"],
                "n_features": results["n_features"],
            })
    
    if runs:
        df = pd.DataFrame(runs)
        df = df.sort_values("test_auc", ascending=False)
        print("\n" + df.to_string(index=False))
    else:
        print("\nNo results found.")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument("--results", type=Path, default=Path("ensemble_results.json"),
                       help="Path to results JSON file")
    parser.add_argument("--compare", type=Path, nargs="+",
                       help="Compare multiple results files")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_multiple_runs(args.compare)
    elif args.results.exists():
        results = load_results(args.results)
        print_comparison(results)
    else:
        print(f"Results file not found: {args.results}")


if __name__ == "__main__":
    main()

# Made with Bob
