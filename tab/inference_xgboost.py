"""
XGBoost Inference with Corrosion Exposure Indices
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb

warnings.filterwarnings('ignore')

from data_utils import load_corrosion_data, load_environment_data, create_training_pairs

RANDOM_SEED = 42

def main():
    print("="*80)
    print("XGBoost Corrosion Risk Prediction with Corrosion Exposure Indices")
    print("="*80)
    
    # Load data
    print("\n[1/5] Loading data...")
    corr_df = load_corrosion_data()
    env_df = load_environment_data()
    
    # Create training pairs
    print("\n[2/5] Creating training pairs...")
    pairs_df = create_training_pairs(corr_df, env_df)
    
    # Prepare features
    feature_cols = ['reference_month'] + [c for c in pairs_df.columns if '__' in c and c != 'aircraft_id']
    X = pairs_df[feature_cols].values
    y = pairs_df['corrosion_risk'].values
    groups = pairs_df['aircraft_id'].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    
    # Check for new features
    new_features = [f for f in feature_cols if 'corrosion_exposure' in f or 'moisture_exposure' in f]
    print(f"\nNew corrosion indices: {len(new_features)}")
    for feat in new_features:
        print(f"  - {feat}")
    
    # Cross-validation
    print("\n[3/5] Cross-validation with XGBoost...")
    n_splits = 5
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    aucs = []
    briers = []
    aps = []
    
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), 1):
        print(f"  Fold {fold}/{n_splits}...", end=" ")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            random_state=RANDOM_SEED,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(X_train_scaled, y_train, verbose=False)
        
        # Predict
        y_pred = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred)
        
        aucs.append(auc)
        briers.append(brier)
        aps.append(ap)
        
        print(f"AUC={auc:.4f}, Brier={brier:.4f}, AP={ap:.4f}")
    
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION RESULTS:")
    print(f"{'='*80}")
    print(f"AUC:            {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Brier Score:    {np.mean(briers):.4f} ± {np.std(briers):.4f}")
    print(f"Avg Precision:  {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"{'='*80}")
    
    # Train final model
    print("\n[4/5] Training final model on all data...")
    scaler_final = RobustScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    model_final = xgb.XGBClassifier(
        random_state=RANDOM_SEED,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model_final.fit(X_scaled, y, verbose=False)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Load test data
    print("\n[5/5] Making predictions on test data...")
    env_test_df = load_environment_data(test=True)
    
    # Calculate test indices
    env_test_df['corrosion_exposure_index'] = (
        (env_test_df['sulphur_dioxide_mass_mixing_ratio'] + 
         env_test_df['nitrogen_dioxide_mass_mixing_ratio']) * 
        env_test_df['metar_relative_humidity'] * 
        env_test_df['total_parking_minutes'] / 1000
    )
    
    env_test_df['moisture_exposure_index'] = (
        env_test_df['metar_relative_humidity'] * 
        env_test_df['total_parking_minutes'] / 100
    )
    
    # Aggregate test features using the same features as training
    from data_utils import aggregate_environmental_features, ENV_FEATURES
    env_test_agg = aggregate_environmental_features(env_test_df, features=ENV_FEATURES)
    
    # Prepare test features
    reference_month_test = pairs_df['reference_month'].median()
    
    test_predictions = []
    for _, row in env_test_agg.iterrows():
        aircraft_id = row['aircraft_id']
        features = {'reference_month': reference_month_test}
        for col in env_test_agg.columns:
            if col != 'aircraft_id':
                features[col] = row[col]
        test_predictions.append({
            'aircraft_id': aircraft_id,
            **features
        })
    
    test_df = pd.DataFrame(test_predictions)
    X_test = test_df[feature_cols].values
    X_test_scaled = scaler_final.transform(X_test)
    
    # Predict
    y_pred_proba = model_final.predict_proba(X_test_scaled)[:, 1]
    
    # Create submission
    submission_df = pd.DataFrame({
        'aircraft_id': test_df['aircraft_id'],
        'corrosion_risk': y_pred_proba
    })
    
    submission_df.to_csv('submission_xgboost.csv', index=False)
    
    print(f"\n✓ Predictions complete!")
    print(f"  Test aircraft: {len(submission_df)}")
    print(f"  Prediction range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")
    print(f"  Mean prediction: {y_pred_proba.mean():.4f}")
    print(f"\n✓ Submission saved to: submission_xgboost.csv")
    
    print(f"\n{'='*80}")
    print("COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
