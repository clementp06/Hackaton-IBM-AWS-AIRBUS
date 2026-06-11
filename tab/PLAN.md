# Model Update Plan - Airbus Corrosion Convention

## Overview
Update all models in the `tab/` directory to align with the Airbus convention for corrosion risk prediction.

## Airbus Convention
The official convention relies on **two reference points per aircraft**:
1. **Month of corrosion observation** → `corrosion_risk = 1` (corroded state)
2. **Exactly 24 months earlier** → `corrosion_risk = 0` (healthy state, assumption: no corrosion 2 years before observation)

This reflects the **non-linear nature** of corrosion progression.

## Current State Analysis

### Data Structure
- **corrosions_training.csv**: Contains `aircraft_id`, `observation_date`, `aircraft_delivery_year`, `aircraft_delivery_month`
- **environment_training.csv**: Monthly environmental data per aircraft
- Current models calculate: `months = (observation_year - delivery_year) * 12 + (observation_month - delivery_month)`

### Current Models
1. **model_basic.py**: Bayesian parametric survival (Weibull, LogNormal, Gamma) - uses months from delivery
2. **model_aft.py**: Bayesian AFT with environmental covariates - uses log(months) from delivery
3. **model_logistic.py**: Beta-Binomial empirical CDF - uses months from delivery
4. **model_tabpfn_aircraft.py**: Per-aircraft time series forecasting - uses binary step function
5. **model_tabpfn_fleet.py**: Fleet-level CDF forecasting - uses empirical CDF
6. **inference.py**: Weibull AFT for test predictions - uses months from first observation

## Implementation Strategy

### Phase 1: Shared Data Preparation Module
**File**: `tab/data_utils.py`

Create utility functions:
```python
def create_training_pairs(corr_df: pd.DataFrame, env_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each aircraft, create two training samples:
    - One at observation_month with corrosion_risk=1
    - One at observation_month-24 with corrosion_risk=0
    
    Returns DataFrame with columns:
    - aircraft_id
    - reference_month (months since delivery)
    - corrosion_risk (0 or 1)
    - environmental features (aggregated)
    """
```

### Phase 2: Model Updates

#### 2.1 model_basic.py
**Current**: Survival analysis on months-to-corrosion distribution
**Update**: Binary classification using logistic regression or probit models
- Input: (reference_month, environmental_features) pairs
- Output: P(corrosion_risk=1 | reference_month)
- Use Bayesian logistic regression with PyMC
- Compare: Logistic, Probit, Complementary log-log link functions

#### 2.2 model_aft.py
**Current**: AFT model with log(months) as outcome
**Update**: Binary classification with environmental covariates
- Input: (reference_month, environmental_features, corrosion_risk) pairs
- Model: Bayesian logistic regression with environmental covariates
- Experiments: Same feature sets (sea_salt only, salt+humidity+wind, all 4)
- Output: P(corrosion_risk=1 | month, features)

#### 2.3 model_logistic.py
**Current**: Beta-Binomial on empirical CDF
**Update**: Adapt to use (1/0) pairs
- For each month t, count:
  - k_1 = number of aircraft with observation_month <= t
  - k_0 = number of aircraft with (observation_month-24) <= t
- Posterior: Beta distribution on P(corroded | month=t)
- Keep Beta prior experiments (Uniform, Weakly informative, Informative)

#### 2.4 inference.py
**Current**: Weibull AFT from first observation
**Update**: Binary classification model for test predictions
- Train on (1/0) pairs with environmental features
- For test queries (aircraft_id, year_month):
  - Extract environmental features for that aircraft
  - Calculate months since delivery (or first observation)
  - Predict: P(corrosion_risk=1 | months, features)
- Use best-performing model from cross-validation

#### 2.5 model_tabpfn_aircraft.py
**Current**: Binary step function (0 before corrosion, 1 after)
**Update**: Incorporate (1/0) pair convention
- For each aircraft, create explicit training points:
  - target=0 at observation_month-24
  - target=1 at observation_month
- Context: months before observation_month-24 (all zeros)
- Forecast: months from observation_month-24 to observation_month+12
- Evaluate: Does model correctly predict 0→1 transition?

#### 2.6 model_tabpfn_fleet.py
**Current**: Fleet-level empirical CDF
**Update**: Use (1/0) pairs for fleet aggregation
- Build fleet-level time series:
  - At each month t: fraction of aircraft with observation_month <= t (risk=1)
  - At each month t: fraction of aircraft with (observation_month-24) <= t (risk=0)
- Forecast: Predict future corrosion risk across fleet
- Compare: With/without seasonal covariates

### Phase 3: Validation

#### 3.1 Data Validation
- Verify all aircraft have observation_month >= 24 (can create valid pairs)
- Check for aircraft with observation_month < 24 (exclude or handle specially)
- Ensure environmental data exists for both reference points

#### 3.2 Model Validation
- Cross-validation: 5-fold CV on aircraft level
- Metrics:
  - AUC-ROC for binary classification
  - Calibration plots (predicted vs actual risk)
  - Brier score
  - Log-loss
- Compare: Old approach vs new (1/0) pair approach

#### 3.3 Prediction Validation
- Ensure all predictions in [0, 1] range
- Check monotonicity: risk should increase with time
- Sanity checks:
  - High sea-salt → higher risk
  - High humidity → higher risk
  - Longer time → higher risk

## Detailed Implementation Steps

### Step 1: Create data_utils.py
```python
# Core functions:
# - load_corrosion_data()
# - load_environment_data()
# - create_training_pairs()
# - aggregate_environmental_features()
# - validate_training_pairs()
```

### Step 2: Update each model file
For each model:
1. Import from data_utils
2. Modify data loading to use create_training_pairs()
3. Update model specification for binary outcome
4. Adjust prediction logic for corrosion_risk
5. Update visualization to show (1/0) pair structure
6. Add validation metrics

### Step 3: Update inference.py
1. Train final model on all (1/0) pairs
2. Load test environment data
3. For each test query:
   - Calculate reference month
   - Extract environmental features
   - Predict corrosion_risk
4. Write submission.csv

### Step 4: Testing & Validation
1. Run all updated models
2. Compare predictions across models
3. Validate against known patterns (sea-salt correlation, etc.)
4. Generate comparison report

## Key Considerations

### Time Reference
- **Training**: Use months since delivery for consistency
- **Test**: May need to use months since first observation (delivery date not available)
- **Solution**: Train models that work with both reference systems

### Environmental Features
- Aggregate per aircraft: mean, std, min, max
- Most important: sea_salt_aerosol (coarse and fine), humidity, wind_speed
- Consider temporal aggregation: features at observation_month vs observation_month-24

### Model Selection
- Compare multiple approaches:
  - Bayesian logistic regression (interpretable, uncertainty quantification)
  - Gradient boosting (high performance, handles non-linearity)
  - TabPFN (state-of-art tabular, no hyperparameter tuning)
- Use cross-validation to select best model for inference.py

### Edge Cases
- Aircraft with observation_month < 24: Cannot create valid (0) reference point
  - Option 1: Exclude from training
  - Option 2: Use observation_month/2 as reference point
  - Option 3: Use delivery month as reference point (risk=0)
- Missing environmental data: Impute or exclude

## Success Criteria
1. All models successfully train on (1/0) pairs
2. Predictions are well-calibrated (predicted risk matches actual risk)
3. Models show expected relationships (sea-salt → higher risk, etc.)
4. Cross-validation metrics improve or remain comparable
5. Test predictions are reasonable (0 < risk < 1, monotonic with time)
6. Code is clean, documented, and reproducible

## Timeline Estimate
- Phase 1 (data_utils.py): 1-2 hours
- Phase 2 (model updates): 4-6 hours (6 models × ~1 hour each)
- Phase 3 (validation): 2-3 hours
- **Total**: 7-11 hours of focused work

## Next Steps
1. Review and approve this plan
2. Create data_utils.py with core functions
3. Update models one by one, testing each
4. Run comprehensive validation
5. Generate final submission.csv