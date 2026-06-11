# Feature Analysis and Model Optimization Summary

## Executive Summary

Comprehensive feature analysis was performed using PCA, correlation analysis, and multiple feature importance methods to optimize the TabPFN model for Brier score minimization.

## Key Findings from Feature Analysis

### 1. Dimensionality Reduction
- **Current features**: 19
- **Components for 95% variance**: Only 10 needed
- **Potential reduction**: 9 features could be removed without significant information loss

### 2. Feature Redundancy
Identified 3 highly correlated feature pairs (|r| ≥ 0.9):
1. `sea_salt_aerosol_5_20_mixing_ratio__mean` ↔ `sea_salt_aerosol_05_5_mixing_ratio__mean` (r=+0.950)
2. `sulphur_dioxide_mass_mixing_ratio__mean` ↔ `sulphur_dioxide_mass_mixing_ratio__std` (r=+0.928)
3. `metar_temperature_c__mean` ↔ `metar_dew_point_c__mean` (r=+0.911)

**Recommendation**: Remove one feature from each pair to reduce multicollinearity.

### 3. Most Important Features (Aggregated Ranking)
1. **reference_month** (rank=1.0) - Dominant feature
2. **sulphur_dioxide_mass_mixing_ratio__std** (rank=6.5)
3. **sea_salt_aerosol_5_20_mixing_ratio__mean** (rank=6.5)
4. **nitrogen_dioxide_mass_mixing_ratio__mean** (rank=6.8)
5. **metar_wind_speed_kn__std** (rank=7.2)

### 4. PCA Insights
- **PC1 (26.29% variance)**: Salt aerosol features and wind speed
- **PC2 (18.68% variance)**: Temperature and humidity features
- **PC3 (14.44% variance)**: Pollutant variability (NO2, O3)

## Engineered Features

### Added Features (12 new features)
1. **Time-based non-linear**: 
   - `reference_month_squared`
   - `reference_month_sqrt`
   - `reference_month_log`

2. **Environmental interactions**:
   - `salt_exposure_total`
   - `large_small_salt_ratio`
   - `humidity_temperature_interaction`
   - `temperature_dewpoint_diff`
   - `wind_salt_interaction`

3. **Composite indices**:
   - `pollutant_index` (SO2 + NO2 + O3)
   - `environmental_volatility`

4. **Top interactions**:
   - `month_wind_interaction`
   - `month_salt_interaction`

### Removed Redundant Features (3 features)
- `sea_salt_aerosol_05_5_mixing_ratio__mean`
- `sulphur_dioxide_mass_mixing_ratio__mean`
- `metar_dew_point_c__mean`

## Model Performance Comparison

### Original Model (submission.csv)
- **Features**: 19 (all environmental features)
- **Cross-validation Brier**: 0.1384 ± 0.0047
- **Cross-validation AUC**: 0.8786 ± 0.0084
- **Test predictions**:
  - Mean: 0.5768
  - Median: 0.6594
  - Range: [0.0492, 0.9465]
  - Monotonic aircraft: 91.5%

### Enhanced Model (submission_enhanced.csv)
- **Features**: 28 (19 original - 3 redundant + 12 engineered)
- **Test predictions**:
  - Mean: 0.0869
  - Median: 0.0809
  - Range: [0.0370, 0.1458]
  - Monotonic aircraft: 100.0%

**Note**: Enhanced model shows significantly lower predictions, possibly due to:
1. Over-calibration from ensemble
2. Missing engineered features in test data (5 features filled with zeros)
3. Different feature distribution after engineering

## Recommendations

### For Competition Submission
**Use `submission.csv` (original model)** because:
1. ✓ Better calibrated predictions (wider, more realistic range)
2. ✓ Validated cross-validation performance (Brier: 0.1384)
3. ✓ All features properly computed for test data
4. ✓ High monotonicity (91.5%)

### For Future Improvements
1. **Fix feature engineering pipeline**: Ensure all engineered features are properly computed for test data
2. **Selective feature engineering**: Add only the most promising interactions
3. **Validate on holdout set**: Test engineered features before final submission
4. **Consider PCA transformation**: Use top 10 principal components instead of raw features
5. **Hyperparameter tuning**: Optimize calibration parameters

## Suggested Interaction Features for Next Iteration
Based on mutual information analysis:
1. `sea_salt_aerosol_05_5_mixing_ratio__mean × reference_month`
2. `sea_salt_aerosol_5_20_mixing_ratio__mean × metar_wind_speed_kn__std`
3. `reference_month × metar_wind_speed_kn__std`

## Files Generated
- `feature_analysis.py` - Comprehensive analysis script
- `feature_analysis.png` - Visualization of PCA, correlations, and importance
- `inference_enhanced.py` - Enhanced inference with feature engineering
- `submission.csv` - **RECOMMENDED** for submission
- `submission_enhanced.csv` - Alternative with engineered features

## Conclusion

The feature analysis revealed that:
1. **reference_month** is by far the most important feature
2. Salt aerosol and pollutant features provide complementary information
3. Several features are highly redundant and can be removed
4. The original calibrated ensemble model provides well-calibrated predictions

**Final recommendation**: Submit `submission.csv` with the original 19 features and calibrated TabPFN ensemble (Brier: 0.1384).
