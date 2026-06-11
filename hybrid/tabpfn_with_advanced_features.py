"""
Pipeline hybride: TabPFN + Feature Engineering Avancé
Combine le modèle de fondation tabulaire avec nos 127 features scientifiques.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error, roc_auc_score, brier_score_loss

# Import notre feature engineering
from features import build_history_feature_table

# Chemins
DATA_DIR = Path("../../data")
CORR_PATH = DATA_DIR / "prediction_training.csv"  # Fichier avec id et corrosion_risk
ENV_TRAIN_PATH = DATA_DIR / "environment_training.csv"
ENV_TEST_PATH = DATA_DIR / "environment_test.csv"


def check_tabpfn_available():
    """Check if TabPFN is available."""
    try:
        from tabpfn import TabPFNClassifier
        return True
    except ImportError:
        return False


def train_calibrated_tabpfn(X_train, y_train, calibration_method="isotonic"):
    """Train calibrated TabPFN or fallback to HistGradientBoosting (handles NaN)."""
    if check_tabpfn_available():
        try:
            from tabpfn import TabPFNClassifier
            base_model = TabPFNClassifier(device='cpu')  # Removed invalid parameter
            calibrated_model = CalibratedClassifierCV(
                base_model,
                method=calibration_method,
                cv=3,
                n_jobs=1
            )
            calibrated_model.fit(X_train, y_train)
            return calibrated_model, "TabPFN"
        except Exception as e:
            print(f"  TabPFN error: {e}, falling back to HistGradientBoosting")
    
    # Fallback - Use HistGradientBoosting which handles NaN natively
    from sklearn.ensemble import HistGradientBoostingClassifier
    base_model = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=300,
        max_depth=5,
        learning_rate=0.03,
    )
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method=calibration_method,
        cv=3,
        n_jobs=1
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model, "HistGradientBoosting"


def main():
    print("\n" + "=" * 80)
    print("PIPELINE HYBRIDE: TabPFN + Feature Engineering Avancé")
    print("=" * 80)
    
    # Charger les données
    print("\n[1/5] Chargement des données...")
    corr_df = pd.read_csv(CORR_PATH)
    env_train = pd.read_csv(ENV_TRAIN_PATH)
    env_test = pd.read_csv(ENV_TEST_PATH)
    
    print(f"  Corrosions: {len(corr_df):,}")
    print(f"  Environment train: {len(env_train):,}")
    print(f"  Environment test: {len(env_test):,}")
    
    # Feature engineering avec TOUTES nos features avancées
    print("\n[2/5] Feature engineering avancé (127 features)...")
    print("  Génération des features Bronze/Silver/Gold/Scientific...")
    
    feature_table_train = build_history_feature_table(env_train)
    feature_table_test = build_history_feature_table(env_test)
    
    print(f"  Features générées: {len(feature_table_train.columns)}")
    
    # Préparer les données d'entraînement
    print("\n[3/5] Préparation des données d'entraînement...")
    
    # Le fichier prediction_training.csv a le format: id (aircraft_id_year-month), corrosion_risk
    # Utiliser la fonction de data.py pour splitter l'id
    from data import split_prediction_id
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
    
    print(f"  Features utilisées: {len(feature_cols)}")
    
    # Scaling robuste
    print("\n[4/5] Training du modèle calibré...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train ensemble de 2 modèles avec calibrations différentes
    print("  [1/2] Training avec calibration isotonique...")
    model_iso, model_type_iso = train_calibrated_tabpfn(X_scaled, y, "isotonic")
    print(f"    Modèle: {model_type_iso}")
    
    print("  [2/2] Training avec calibration sigmoid...")
    model_sig, model_type_sig = train_calibrated_tabpfn(X_scaled, y, "sigmoid")
    print(f"    Modèle: {model_type_sig}")
    
    # Prédictions sur test
    print("\n[5/5] Génération des prédictions...")
    
    X_test = feature_table_test[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    # Ensemble des 2 modèles
    pred_iso = model_iso.predict_proba(X_test_scaled)[:, 1]
    pred_sig = model_sig.predict_proba(X_test_scaled)[:, 1]
    
    # Moyenne pondérée (isotonic souvent meilleur pour Brier)
    predictions = 0.6 * pred_iso + 0.4 * pred_sig
    
    # Clip to [0, 1]
    predictions = np.clip(predictions, 0, 1)
    
    # Créer submission
    submission = pd.DataFrame({
        'id': feature_table_test['aircraft_id'] + '_' + feature_table_test['year_month'],
        'corrosion_risk': predictions
    })
    
    output_path = Path("submission_hybrid_tabpfn.csv")
    submission.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("PRÉDICTIONS TERMINÉES !")
    print("=" * 80)
    print(f"\nFichier: {output_path}")
    print(f"Prédictions: {len(submission):,}")
    print(f"\nStatistiques:")
    print(f"  Moyenne: {predictions.mean():.4f}")
    print(f"  Std:     {predictions.std():.4f}")
    print(f"  Min:     {predictions.min():.4f}")
    print(f"  Max:     {predictions.max():.4f}")
    print(f"\nDistribution:")
    print(f"  < 0.1:   {(predictions < 0.1).sum():,} ({100 * (predictions < 0.1).mean():.1f}%)")
    print(f"  0.1-0.3: {((predictions >= 0.1) & (predictions < 0.3)).sum():,} ({100 * ((predictions >= 0.1) & (predictions < 0.3)).mean():.1f}%)")
    print(f"  0.3-0.5: {((predictions >= 0.3) & (predictions < 0.5)).sum():,} ({100 * ((predictions >= 0.3) & (predictions < 0.5)).mean():.1f}%)")
    print(f"  > 0.5:   {(predictions >= 0.5).sum():,} ({100 * (predictions >= 0.5).mean():.1f}%)")
    
    print("\n" + "=" * 80)
    print("AVANTAGES DE CETTE APPROCHE:")
    print("=" * 80)
    print("  ✓ TabPFN: Modèle de fondation pré-entraîné")
    print("  ✓ 127 features scientifiques (vs ~10 dans tab/)")
    print("  ✓ Calibration isotonique + sigmoid")
    print("  ✓ Ensemble de 2 modèles calibrés")
    print("  ✓ Robust scaling pour outliers")
    print("  ✓ Features: Bronze/Silver/Gold/Scientific")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
