"""
Script pour evaluer le modele avec un test set et obtenir le MSE.
"""
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, log_loss

from config import PathConfig, ModelConfig, ValidationConfig
from data import build_training_table, load_environment, load_labels
from features import build_history_feature_table
from feature_selection import select_features_smart
from training import build_model, fit_model
from data_split import split_train_valid_test


def main():
    print("\n" + "=" * 80)
    print("EVALUATION DU MODELE AVEC TEST SET")
    print("=" * 80)
    
    # Configuration
    paths = PathConfig()
    model_config = ModelConfig()
    validation_config = ValidationConfig()
    
    # Charger les donnees
    print("\n[1/5] Chargement des donnees...")
    environment = load_environment(paths.environment_path)
    labels = load_labels(paths.labels_path)
    
    # Feature engineering
    print("[2/5] Feature engineering...")
    feature_table = build_history_feature_table(environment)
    X, y, groups, feature_columns = build_training_table(
        environment,
        labels,
        feature_table,
    )
    
    print(f"  Dataset: {len(X):,} lignes, {len(feature_columns)} features")
    
    # Feature selection
    print("[3/5] Feature selection...")
    selected_features = select_features_smart(
        importance_path=paths.importance_output_path,
        X=X,
        top_n=50,
        min_gain=800,
        corr_threshold=0.95
    )
    
    X = X[selected_features]
    print(f"  Features selectionnees: {len(selected_features)}")
    
    # Split 60/20/20
    print("[4/5] Split des donnees (60% train / 20% valid / 20% test)...")
    train_idx, valid_idx, test_idx = split_train_valid_test(
        X, y, groups,
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
        seed=model_config.seed,
    )
    
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Valid: {len(X_valid):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # Entrainement
    print("[5/5] Entrainement et evaluation...")
    model = build_model(model_config)
    fit_model(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        model_config.early_stopping_rounds,
    )
    
    print(f"  Modele entraine: {model.best_iteration_} iterations")
    
    # Predictions
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_valid = model.predict_proba(X_valid)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]
    
    # Metriques
    print("\n" + "=" * 80)
    print("RESULTATS")
    print("=" * 80)
    
    # Train set
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_auc = roc_auc_score(y_train, y_pred_train)
    train_ap = average_precision_score(y_train, y_pred_train)
    train_logloss = log_loss(y_train, y_pred_train)
    
    print("\nTrain Set:")
    print(f"  MSE:     {train_mse:.6f}")
    print(f"  AUC:     {train_auc:.4f}")
    print(f"  AP:      {train_ap:.4f}")
    print(f"  LogLoss: {train_logloss:.4f}")
    
    # Validation set
    valid_mse = mean_squared_error(y_valid, y_pred_valid)
    valid_auc = roc_auc_score(y_valid, y_pred_valid)
    valid_ap = average_precision_score(y_valid, y_pred_valid)
    valid_logloss = log_loss(y_valid, y_pred_valid)
    
    print("\nValidation Set:")
    print(f"  MSE:     {valid_mse:.6f}")
    print(f"  AUC:     {valid_auc:.4f}")
    print(f"  AP:      {valid_ap:.4f}")
    print(f"  LogLoss: {valid_logloss:.4f}")
    
    # Test set
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_pred_test)
    test_ap = average_precision_score(y_test, y_pred_test)
    test_logloss = log_loss(y_test, y_pred_test)
    
    print("\nTest Set:")
    print(f"  MSE:     {test_mse:.6f}")
    print(f"  AUC:     {test_auc:.4f}")
    print(f"  AP:      {test_ap:.4f}")
    print(f"  LogLoss: {test_logloss:.4f}")
    
    # Overfitting analysis
    print("\n" + "=" * 80)
    print("OVERFITTING ANALYSIS")
    print("=" * 80)
    
    print(f"\nMSE:")
    print(f"  Train-Valid: {train_mse - valid_mse:+.6f}")
    print(f"  Train-Test:  {train_mse - test_mse:+.6f}")
    print(f"  Valid-Test:  {valid_mse - test_mse:+.6f}")
    
    print(f"\nAUC:")
    print(f"  Train-Valid: {train_auc - valid_auc:+.4f}")
    print(f"  Train-Test:  {train_auc - test_auc:+.4f}")
    print(f"  Valid-Test:  {valid_auc - test_auc:+.4f}")
    
    print("\n" + "=" * 80)
    print(f"\n🎯 SCORE MSE EN TEST: {test_mse:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
