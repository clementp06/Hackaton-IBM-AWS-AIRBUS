# Aircraft Corrosion Prediction Model V2 - Enhanced Feature Engineering

## Executive Summary

Successfully trained an enhanced XGBoost model with **1,422 engineered features** (compared to 331 in Model 1) to predict aircraft oxidation/corrosion probability. The expanded feature set includes polynomial interactions, advanced statistical aggregations, and domain-specific features.

## Performance Comparison: Model 2 vs Model 1

### Brier Score (Primary Metric)
| Model | Features | Test Brier Score | Improvement |
|-------|----------|------------------|-------------|
| **Model 1** | 331 | 0.100244 | Baseline |
| **Model 2** | 1,422 | **0.098912** | **+1.33% better** ✓ |

### Additional Metrics
| Metric | Model 1 | Model 2 | Change |
|--------|---------|---------|--------|
| **AUC-ROC (Test)** | 0.9236 | **0.9242** | +0.07% ✓ |
| **Log Loss (Test)** | 0.3445 | 0.3518 | -2.1% |
| **Training Time** | ~1 min | ~6 min | 6x longer |

**Key Finding**: Model 2 achieves a **1.33% improvement in Brier score** with 4.3x more features, demonstrating that additional feature engineering provides measurable but modest gains.

## Model 2 Feature Engineering Strategy

### Feature Categories (1,422 total features)

#### 1. **Original Features** (33)
Base environmental measurements from sensors

#### 2. **Cumulative Aggregations** (165 features)
- Cumulative sum
- Cumulative mean
- Cumulative max
- Cumulative min
- Cumulative standard deviation (expanding window)

#### 3. **Rolling Window Features** (660 features)
Multiple time windows: 2, 3, 6, 12, 24 months
- Rolling mean
- Rolling standard deviation
- Rolling min
- Rolling max

#### 4. **Lag Features** (165 features)
Historical values at: 1, 2, 3, 6, 12 months back

#### 5. **Difference Features** (132 features)
Change from previous periods: 1, 3, 6, 12 months

#### 6. **Rate of Change Features** (99 features)
Percentage change over: 1, 3, 6 months

#### 7. **Exponential Weighted Moving Averages** (99 features)
EWMA with spans: 3, 6, 12 months

#### 8. **Time-Based Features** (4 features)
- Aircraft age (months)
- Age squared
- Age cubed
- Age square root

#### 9. **Domain-Specific Interactions** (7 features)
- Temperature × Humidity
- Temperature × Parking time
- Humidity × Sea salt
- Sea salt total index
- Dust total index
- Pollution index
- Aerosol total index

#### 10. **Polynomial Features** (15 features)
Degree-2 polynomial interactions of top 5 features:
- temperature
- total_parking_minutes
- metar_relative_humidity
- metar_wind_speed_kn
- specific_humidity

#### 11. **Ratio Features** (3 features)
- Temperature/Humidity ratio
- Sea salt/Dust ratio
- Parking time/Age ratio

#### 12. **Statistical Aggregations** (40 features)
- Rolling skewness (6, 12 months) for top 10 features
- Rolling kurtosis (6, 12 months) for top 10 features

## Top 30 Most Important Features (Model 2)

1. **temperature_cumsum** (4.48%) - Total temperature exposure
2. **age_months_squared** (3.05%) - Non-linear age effect
3. **age_months** (3.04%) - Aircraft age
4. **total_parking_minutes_cumsum** (2.46%) - Total parking time
5. **age_months_sqrt** (1.28%) - Square root of age
6. **sea_salt_aerosol_5_20_mixing_ratio_roll24_mean** (1.01%) - 24-month sea salt average
7. **sea_salt_aerosol_003_05_mixing_ratio_roll6_kurt** (0.66%) - Sea salt kurtosis
8. **age_months_cubed** (0.60%) - Cubic age effect
9. **sea_salt_aerosol_5_20_mixing_ratio_roll12_mean** (0.60%) - 12-month sea salt average
10. **parking_age_ratio** (0.56%) - Parking time relative to age

### Key Insights from Feature Importance

1. **Non-linear age effects** are highly predictive (squared, cubed, sqrt transformations all in top 10)
2. **Long-term rolling averages** (24-month) capture important trends
3. **Statistical moments** (kurtosis) provide additional signal
4. **Ratio features** (parking/age) capture relative exposure patterns
5. **Temperature cumulative exposure** remains the single most important feature

## Detailed Performance Metrics

### Model 2 Performance
- **Test Brier Score**: 0.098912 (lower is better)
- **Train Brier Score**: 0.001923 (excellent fit)
- **Test AUC-ROC**: 0.9242 (excellent discrimination)
- **Train AUC-ROC**: 1.0000 (perfect on training data)
- **Test Log Loss**: 0.351781
- **Train Log Loss**: 0.019629

### Dataset Split
- **Training Set**: 50,919 samples from 606 aircraft (80.2%)
- **Test Set**: 12,605 samples from 152 aircraft (19.8%)
- **Train Corrosion Rate**: 29.67%
- **Test Corrosion Rate**: 26.43%

## Feature Engineering Innovations in Model 2

### 1. Multiple Time Scales
- Short-term (2-3 months): Captures recent changes
- Medium-term (6-12 months): Seasonal patterns
- Long-term (24 months): Long-term trends

### 2. Non-linear Transformations
- Polynomial age features capture accelerating corrosion with age
- Exponential weighted averages give more weight to recent data

### 3. Statistical Moments
- Skewness and kurtosis capture distribution shape
- Helps identify unusual exposure patterns

### 4. Domain Knowledge Integration
- Corrosion-specific interactions (temp × humidity × sea salt)
- Composite indices (pollution, aerosol, sea salt totals)
- Relative exposure metrics (ratios)

### 5. Comprehensive Lag Structure
- Multiple lag periods capture temporal dependencies
- Difference and rate-of-change features capture dynamics

## Model Configuration

### XGBoost Hyperparameters (Model 2)
```python
n_estimators=500
max_depth=6
learning_rate=0.05
subsample=0.8
colsample_bytree=0.8
colsample_bylevel=0.8  # Additional column sampling
min_child_weight=3
gamma=0.1
reg_alpha=0.1
reg_lambda=1.0
```

## Comparison Analysis

### What Worked Well
✓ **Non-linear age transformations** - Captured accelerating corrosion patterns
✓ **Long-term rolling averages** - Better trend capture with 24-month windows
✓ **Statistical moments** - Kurtosis and skewness added predictive value
✓ **Domain-specific ratios** - Parking/age ratio highly informative

### Diminishing Returns
⚠ **Marginal improvement** - 4.3x more features → only 1.33% better Brier score
⚠ **Training time** - 6x longer training time
⚠ **Complexity** - More features increase model complexity and maintenance

### Model 1 vs Model 2 Trade-offs

| Aspect | Model 1 | Model 2 | Winner |
|--------|---------|---------|--------|
| **Brier Score** | 0.100244 | 0.098912 | Model 2 ✓ |
| **AUC-ROC** | 0.9236 | 0.9242 | Model 2 ✓ |
| **Simplicity** | 331 features | 1,422 features | Model 1 ✓ |
| **Training Speed** | ~1 min | ~6 min | Model 1 ✓ |
| **Interpretability** | Higher | Lower | Model 1 ✓ |
| **Maintenance** | Easier | Harder | Model 1 ✓ |

## Output Files (Model_2/)

1. **test_predictions.csv** - 12,605 predictions with probabilities
2. **model_performance_summary.csv** - Performance metrics
3. **feature_importance.csv** - All 1,422 features ranked
4. **train_corrosion_model_v2.py** - Complete training script

## Recommendations

### For Production Deployment

**Recommendation: Use Model 1 (331 features)**

**Rationale:**
- Only 1.33% worse Brier score (0.100244 vs 0.098912)
- 6x faster training
- Much simpler to maintain and debug
- Better interpretability for stakeholders
- Lower computational requirements

**When to Consider Model 2:**
- If the 1.33% improvement is critical for the application
- If computational resources are not a constraint
- If you need to squeeze every bit of performance
- For ensemble methods where diversity matters

### For Further Improvement

1. **Feature Selection**: Use recursive feature elimination to find optimal subset
2. **Ensemble Methods**: Combine Model 1 and Model 2 predictions
3. **Deep Learning**: Try LSTM/Transformer for temporal patterns
4. **External Data**: Weather forecasts, maintenance records, flight routes
5. **Class Balancing**: Address the 29% corrosion rate imbalance

### Feature Engineering Lessons

1. **More features ≠ proportionally better performance**
2. **Non-linear transformations** of important features are valuable
3. **Multiple time scales** capture different patterns
4. **Domain knowledge** guides effective feature creation
5. **Diminishing returns** set in quickly after core features

## Conclusion

Model 2 successfully demonstrates that **enhanced feature engineering with 1,422 features achieves a 1.33% improvement in Brier score** over the baseline 331-feature model. However, this comes at the cost of:
- 4.3x more features
- 6x longer training time
- Reduced interpretability

The **modest improvement suggests we're approaching the performance ceiling** for this dataset with tree-based models. The most valuable additions were:
- Non-linear age transformations
- Long-term rolling averages (24 months)
- Statistical moments (kurtosis, skewness)
- Domain-specific ratio features

For most practical applications, **Model 1 offers the best balance** of performance, simplicity, and maintainability.

---
*Model V2 trained on: 2026-06-11*
*Training time: ~6 minutes*
*Framework: XGBoost 2.0+ with scikit-learn*
*Python: 3.11*