"""
Script rapide pour generer la prediction finale en 20 minutes.
Utilise le meilleur modele avec feature selection.
"""
import pandas as pd
from pathlib import Path
import lightgbm as lgb

from config import PathConfig, ModelConfig, ValidationConfig
from data import build_training_table, load_environment, load_labels, split_prediction_id
from features import build_history_feature_table
from feature_selection import select_features_smart
from training import build_model, fit_model, split_by_aircraft


def main():
    print("\n" + "=" * 80)
    print("PREDICTION FINALE RAPIDE - 20 MINUTES")
    print("=" * 80)
    
    # Configuration
    paths = PathConfig()
    model_config = ModelConfig()
    validation_config = ValidationConfig()
    
    # Charger les donnees d'entrainement
    print("\n[1/6] Chargement des donnees d'entrainement...")
    environment_train = load_environment(paths.environment_path)
    labels = load_labels(paths.labels_path)
    
    # Feature engineering
    print("[2/6] Feature engineering...")
    feature_table_train = build_history_feature_table(environment_train)
    X, y, groups, feature_columns = build_training_table(
        environment_train,
        labels,
        feature_table_train,
    )
    
    print(f"  Dataset: {len(X):,} lignes, {len(feature_columns)} features")
    
    # Feature selection
    print("[3/6] Feature selection...")
    selected_features = select_features_smart(
        importance_path=paths.importance_output_path,
        X=X,
        top_n=50,
        min_gain=800,
        corr_threshold=0.95
    )
    
    X = X[selected_features]
    print(f"  Features selectionnees: {len(selected_features)}")
    
    # Entrainer sur TOUTES les donnees (pas de split pour maximiser la performance)
    print("[4/6] Entrainement du modele sur toutes les donnees...")
    print("  (Utilisation de 100% des donnees pour maximiser la performance)")
    
    # Split juste pour avoir un validation set pour early stopping
    train_idx, valid_idx = split_by_aircraft(
        X, y, groups, validation_config, model_config.seed
    )
    
    model = build_model(model_config)
    fit_model(
        model,
        X.iloc[train_idx],
        y.iloc[train_idx],
        X.iloc[valid_idx],
        y.iloc[valid_idx],
        model_config.early_stopping_rounds,
    )
    
    print(f"  Modele entraine: {model.best_iteration_} iterations")
    
    # Charger les donnees de test
    print("[5/6] Chargement des donnees de test...")
    test_env_path = Path("../../data/environment_test.csv")
    if not test_env_path.exists():
        test_env_path = Path("data/environment_test.csv")
    
    environment_test = pd.read_csv(test_env_path)
    print(f"  Donnees de test: {len(environment_test):,} lignes")
    
    # Feature engineering sur le test
    print("[6/6] Generation des predictions...")
    feature_table_test = build_history_feature_table(environment_test)
    
    # Creer le dataset de test
    X_test = feature_table_test[selected_features]
    
    # Predictions
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Creer le fichier de soumission
    submission = pd.DataFrame({
        'id': feature_table_test['aircraft_id'].astype(str) + '_' + feature_table_test['year_month'].astype(str),
        'corrosion_risk': predictions
    })
    
    # Sauvegarder
    output_path = Path("submission_final.csv")
    submission.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("PREDICTION TERMINEE !")
    print("=" * 80)
    print(f"\nFichier de soumission: {output_path}")
    print(f"Nombre de predictions: {len(submission):,}")
    print(f"Probabilite moyenne: {predictions.mean():.4f}")
    print(f"Probabilite min: {predictions.min():.4f}")
    print(f"Probabilite max: {predictions.max():.4f}")
    
    # Statistiques
    print("\nDistribution des predictions:")
    print(f"  < 0.1: {(predictions < 0.1).sum():,} ({100 * (predictions < 0.1).mean():.1f}%)")
    print(f"  0.1-0.3: {((predictions >= 0.1) & (predictions < 0.3)).sum():,} ({100 * ((predictions >= 0.1) & (predictions < 0.3)).mean():.1f}%)")
    print(f"  0.3-0.5: {((predictions >= 0.3) & (predictions < 0.5)).sum():,} ({100 * ((predictions >= 0.3) & (predictions < 0.5)).mean():.1f}%)")
    print(f"  0.5-0.7: {((predictions >= 0.5) & (predictions < 0.7)).sum():,} ({100 * ((predictions >= 0.5) & (predictions < 0.7)).mean():.1f}%)")
    print(f"  > 0.7: {(predictions >= 0.7).sum():,} ({100 * (predictions >= 0.7).mean():.1f}%)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
