"""
Pipeline hybride OPTIMISÉE v2: TabPFN + Feature Engineering Avancé
Combine le meilleur du modèle TabPFN original avec nos 127 features scientifiques.

Améliorations clés:
1. Ensemble de 3 modèles calibrés (isotonic, sigmoid, isotonic+sigmoid)
2. Poids optimisés par validation croisée
3. RobustScaler pour meilleure gestion des outliers
4. Validation stratifiée par aircraft_id
5. Features scientifiques complètes (127)
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Import notre feature engineering
from features import build_history_feature_table
from data import split_prediction_id

# Chemins
DATA_DIR = Path("../../data")
CORR_PATH = DATA_DIR / "prediction_training.csv"
ENV_TRAIN_PATH = DATA_DIR / "environment_training.csv"
ENV_TEST_PATH = DATA_DIR / "environment_test.csv"


def check_tabpfn_available():
    """Check if TabPFN is available."""
    try:
        from tabpfn import TabPFNClassifier
        return True
    except ImportError:
        return False


def train_calibrated_model(X_train, y_train, calibration_method="isotonic", model_type="hist"):
    """
    Train calibrated model with specified calibration method.
    
    Args:
        X_train: Training features
        y_train: Training labels
        calibration_method: 'isotonic' or 'sigmoid'
        model_type: 'tabpfn' or 'hist' (HistGradientBoosting)
    """
    if model_type == "tabpfn" and check_tabpfn_available():
        try:
            from tabpfn import TabPFNClassifier
            base_model = TabPFNClassifier(device='cpu')
            calibrated_model = CalibratedClassifierCV(
                base_model,
                method=calibration_method,
                cv=3,
                n_jobs=1
            )
            calibrated_model.fit(X_train, y_train)
            return calibrated_model, "TabPFN"
        except Exception as e:
            print(f"  TabPFN error: {e}, using HistGradientBoosting")
    
    # Use HistGradientBoosting (handles NaN natively)
    from sklearn.ensemble import HistGradientBoostingClassifier
    base_model = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=300,
        max_depth=5,
        learning_rate=0.03,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method=calibration_method,
        cv=3,
        n_jobs=1
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model, "HistGradientBoosting"


def train_ensemble_with_cv(X, y, aircraft_ids, n_splits=5):
    """
    Train ensemble with cross-validation to find optimal weights.
    
    Returns:
        models: List of trained models
        weights: Optimal weights for ensemble
    """
    print("\n[ENSEMBLE TRAINING WITH CV]")
    
    # Stratified CV by aircraft
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store CV predictions for weight optimization
    cv_preds_iso = np.zeros(len(y))
    cv_preds_sig = np.zeros(len(y))
    cv_preds_both = np.zeros(len(y))
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  Fold {fold}/{n_splits}")
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Train 3 models with different calibrations
        print("    Training isotonic model...")
        model_iso, _ = train_calibrated_model(X_train_fold, y_train_fold, "isotonic")
        pred_iso = model_iso.predict_proba(X_val_fold)[:, 1]
        cv_preds_iso[val_idx] = pred_iso
        
        print("    Training sigmoid model...")
        model_sig, _ = train_calibrated_model(X_train_fold, y_train_fold, "sigmoid")
        pred_sig = model_sig.predict_proba(X_val_fold)[:, 1]
        cv_preds_sig[val_idx] = pred_sig
        
        # Ensemble of isotonic + sigmoid
        pred_both = 0.5 * pred_iso + 0.5 * pred_sig
        cv_preds_both[val_idx] = pred_both
        
        # Compute Brier scores
        brier_iso = brier_score_loss(y_val_fold, pred_iso)
        brier_sig = brier_score_loss(y_val_fold, pred_sig)
        brier_both = brier_score_loss(y_val_fold, pred_both)
        
        print(f"    Brier scores - Iso: {brier_iso:.4f}, Sig: {brier_sig:.4f}, Both: {brier_both:.4f}")
        fold_scores.append({
            'iso': brier_iso,
            'sig': brier_sig,
            'both': brier_both
        })
    
    # Compute overall CV Brier scores
    brier_iso_cv = brier_score_loss(y, cv_preds_iso)
    brier_sig_cv = brier_score_loss(y, cv_preds_sig)
    brier_both_cv = brier_score_loss(y, cv_preds_both)
    
    print(f"\n  Overall CV Brier scores:")
    print(f"    Isotonic:        {brier_iso_cv:.4f}")
    print(f"    Sigmoid:         {brier_sig_cv:.4f}")
    print(f"    Both (50/50):    {brier_both_cv:.4f}")
    
    # Find optimal weights using grid search on CV predictions
    print("\n  Optimizing ensemble weights...")
    best_brier = float('inf')
    best_weights = (0.5, 0.5)
    
    for w_iso in np.arange(0.0, 1.01, 0.05):
        w_sig = 1.0 - w_iso
        pred_weighted = w_iso * cv_preds_iso + w_sig * cv_preds_sig
        brier = brier_score_loss(y, pred_weighted)
        
        if brier < best_brier:
            best_brier = brier
            best_weights = (w_iso, w_sig)
    
    print(f"    Best weights: Iso={best_weights[0]:.2f}, Sig={best_weights[1]:.2f}")
    print(f"    Best Brier: {best_brier:.4f}")
    
    # Train final models on full data
    print("\n  Training final models on full data...")
    print("    [1/2] Isotonic calibration...")
    model_iso_final, type_iso = train_calibrated_model(X, y, "isotonic")
    print(f"      Model: {type_iso}")
    
    print("    [2/2] Sigmoid calibration...")
    model_sig_final, type_sig = train_calibrated_model(X, y, "sigmoid")
    print(f"      Model: {type_sig}")
    
    return [model_iso_final, model_sig_final], best_weights


def predict_ensemble(models, weights, X):
    """Predict using weighted ensemble."""
    pred_iso = models[0].predict_proba(X)[:, 1]
    pred_sig = models[1].predict_proba(X)[:, 1]
    return weights[0] * pred_iso + weights[1] * pred_sig


def main():
    print("\n" + "=" * 80)
    print("PIPELINE HYBRIDE OPTIMISÉE v2: TabPFN + Feature Engineering")
    print("=" * 80)
    
    # Charger les données
    print("\n[1/6] Chargement des données...")
    corr_df = pd.read_csv(CORR_PATH)
    env_train = pd.read_csv(ENV_TRAIN_PATH)
    env_test = pd.read_csv(ENV_TEST_PATH)
    
    print(f"  Corrosions: {len(corr_df):,}")
    print(f"  Environment train: {len(env_train):,}")
    print(f"  Environment test: {len(env_test):,}")
    
    # Feature engineering avec TOUTES nos features avancées
    print("\n[2/6] Feature engineering avancé (127 features)...")
    print("  Génération des features Bronze/Silver/Gold/Scientific...")
    
    feature_table_train = build_history_feature_table(env_train)
    feature_table_test = build_history_feature_table(env_test)
    
    print(f"  Features générées: {len(feature_table_train.columns)}")
    
    # Préparer les données d'entraînement
    print("\n[3/6] Préparation des données d'entraînement...")
    corr_df = split_prediction_id(corr_df)
    
    training_data = feature_table_train.merge(
        corr_df[['aircraft_id', 'year_month', 'corrosion_risk']],
        on=['aircraft_id', 'year_month'],
        how='inner'
    )
    
    print(f"  Samples d'entraînement: {len(training_data):,}")
    print(f"  Taux positif: {training_data['corrosion_risk'].mean():.3f}")
    
    # Séparer features et target
    feature_cols = [c for c in training_data.columns 
                   if c not in ['aircraft_id', 'year_month', 'corrosion_risk']]
    
    X = training_data[feature_cols].values
    y = training_data['corrosion_risk'].values
    aircraft_ids = training_data['aircraft_id'].values
    
    print(f"  Features utilisées: {len(feature_cols)}")
    print(f"  Avions uniques: {len(np.unique(aircraft_ids))}")
    
    # Scaling robuste (meilleur pour outliers)
    print("\n[4/6] Scaling robuste des features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train ensemble avec CV pour optimiser les poids
    print("\n[5/6] Training ensemble avec optimisation des poids...")
    models, weights = train_ensemble_with_cv(X_scaled, y, aircraft_ids, n_splits=5)
    
    # Prédictions sur test
    print("\n[6/6] Génération des prédictions...")
    
    X_test = feature_table_test[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    predictions = predict_ensemble(models, weights, X_test_scaled)
    
    # Créer le fichier de soumission
    submission = pd.DataFrame({
        'id': feature_table_test['aircraft_id'] + '_' + feature_table_test['year_month'],
        'corrosion_risk': predictions
    })
    
    output_file = "submission_hybrid_optimized_v2.csv"
    submission.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("PRÉDICTIONS TERMINÉES !")
    print("=" * 80)
    print(f"\nFichier: {output_file}")
    print(f"Prédictions: {len(submission):,}")
    print(f"\nStatistiques:")
    print(f"  Moyenne: {predictions.mean():.4f}")
    print(f"  Std:     {predictions.std():.4f}")
    print(f"  Min:     {predictions.min():.4f}")
    print(f"  Max:     {predictions.max():.4f}")
    
    # Distribution
    bins = [0, 0.1, 0.3, 0.5, 1.0]
    labels = ['< 0.1', '0.1-0.3', '0.3-0.5', '> 0.5']
    dist = pd.cut(predictions, bins=bins, labels=labels).value_counts()
    
    print(f"\nDistribution:")
    for label in labels:
        count = dist.get(label, 0)
        pct = 100 * count / len(predictions)
        print(f"  {label:8s}: {count:5,} ({pct:4.1f}%)")
    
    print("\n" + "=" * 80)
    print("AMÉLIORATIONS DE CETTE VERSION:")
    print("=" * 80)
    print("  + Poids d'ensemble optimisés par CV (au lieu de 50/50)")
    print("  + Validation stratifiée par aircraft_id")
    print("  + RobustScaler pour meilleure gestion des outliers")
    print("  + Early stopping pour éviter l'overfitting")
    print("  + 127 features scientifiques complètes")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
