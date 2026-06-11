import numpy as np
import pandas as pd

from data import ID_COLUMNS


ROLLING_WINDOWS = [3, 6, 12, 24]


def add_dates(environment):
    environment = environment.copy()
    environment["date"] = pd.to_datetime(environment["year_month"] + "-01")
    return environment.sort_values(["aircraft_id", "date"]).reset_index(drop=True)


def get_numeric_environment_columns(environment):
    return environment.select_dtypes(include="number").columns.tolist()


# ============================================================================
# BRONZE FEATURES: Features brutes et de base
# ============================================================================

def make_bronze_corrosion_factors(environment):
    """Features Bronze: Facteurs directs de corrosion basés sur la physique"""
    
    # Humidité relative critique pour la corrosion (>60%)
    humidity_risk = (environment["metar_relative_humidity"] > 60).astype(float)
    
    # Température favorable à la corrosion (15-35°C)
    temp_risk = ((environment["metar_temperature_c"] >= 15) &
                 (environment["metar_temperature_c"] <= 35)).astype(float)
    
    # Point de rosée proche de la température (condensation)
    dew_point_diff = environment["metar_temperature_c"] - environment["metar_dew_point_c"]
    condensation_risk = (dew_point_diff < 3).astype(float)
    
    # Sel marin total (facteur majeur de corrosion)
    sea_salt_total = (
        environment["sea_salt_aerosol_003_05_mixing_ratio"] +
        environment["sea_salt_aerosol_05_5_mixing_ratio"] +
        environment["sea_salt_aerosol_5_20_mixing_ratio"]
    )
    
    # Sulfates (corrosion acide)
    sulfur_compounds = (
        environment["sulphate_aerosol_mixing_ratio"] +
        environment["sulphur_dioxide_mass_mixing_ratio"]
    )
    
    # Composés azotés (corrosion acide)
    nitrogen_compounds = (
        environment["nitrogen_dioxide_mass_mixing_ratio"] +
        environment["nitrogen_monoxide_mass_mixing_ratio"] +
        environment["hno3"]
    )
    
    # Temps d'exposition critique (parking + conditions favorables)
    parking_hours = environment["total_parking_minutes"] / 60
    critical_exposure_time = (
        parking_hours *
        (humidity_risk + condensation_risk) / 2
    )
    
    # Time of Wetness (TOW) - Durée d'humidification critique
    # RH > 80% et T > 0°C = conditions de film d'eau
    tow_conditions = (
        (environment["metar_relative_humidity"] > 80) &
        (environment["metar_temperature_c"] > 0)
    ).astype(float)
    
    # Indice de corrosion ISO 9223 simplifié
    # Basé sur SO2 et Cl- (sel marin)
    iso_corrosivity = (
        np.log1p(sulfur_compounds * 1e8) * 0.5 +
        np.log1p(sea_salt_total * 1e10) * 0.5
    )
    
    return pd.DataFrame({
        "bronze__humidity_risk": humidity_risk,
        "bronze__temp_risk": temp_risk,
        "bronze__condensation_risk": condensation_risk,
        "bronze__sea_salt_total": sea_salt_total,
        "bronze__sulfur_compounds": sulfur_compounds,
        "bronze__nitrogen_compounds": nitrogen_compounds,
        "bronze__dew_point_diff": dew_point_diff,
        "bronze__parking_hours": parking_hours,
        "bronze__critical_exposure_time": critical_exposure_time,
        "bronze__time_of_wetness": tow_conditions,
        "bronze__iso_corrosivity": iso_corrosivity,
    }, index=environment.index)


def make_bronze_aerosol_features(environment):
    """Features Bronze: Aérosols et polluants"""
    
    # Poussières totales
    dust_total = (
        environment["dust_aerosol_003_055_mixing_ratio"] +
        environment["dust_aerosol_055_09_mixing_ratio"] +
        environment["dust_aerosol_09_20_mixing_ratio"]
    )
    
    # Matière organique totale
    organic_matter_total = (
        environment["hydrophilic_organic_matter_aerosol_mixing_ratio"] +
        environment["hydrophobic_organic_matter_aerosol_mixing_ratio"]
    )
    
    # Carbone noir total
    black_carbon_total = (
        environment["hydrophilic_black_carbon_aerosol_mixing_ratio"] +
        environment["hydrophobic_black_carbon_aerosol_mixing_ratio"]
    )
    
    return pd.DataFrame({
        "bronze__dust_total": dust_total,
        "bronze__organic_matter_total": organic_matter_total,
        "bronze__black_carbon_total": black_carbon_total,
    }, index=environment.index)


# ============================================================================
# SILVER FEATURES: Interactions et indices composites
# ============================================================================

def make_silver_corrosion_indices(environment, bronze_features):
    """Features Silver: Indices de corrosion composites"""
    
    # Indice de corrosion atmosphérique (combinaison humidité + température + sel)
    atmospheric_corrosion_index = (
        bronze_features["bronze__humidity_risk"] * 0.3 +
        bronze_features["bronze__temp_risk"] * 0.2 +
        bronze_features["bronze__condensation_risk"] * 0.3 +
        np.log1p(bronze_features["bronze__sea_salt_total"] * 1e10) * 0.2
    )
    
    # Indice d'agressivité chimique
    chemical_aggressivity = (
        np.log1p(bronze_features["bronze__sulfur_compounds"] * 1e8) * 0.4 +
        np.log1p(bronze_features["bronze__nitrogen_compounds"] * 1e8) * 0.3 +
        np.log1p(bronze_features["bronze__sea_salt_total"] * 1e10) * 0.3
    )
    
    # Temps d'exposition critique (parking + conditions favorables)
    critical_exposure_time = (
        bronze_features["bronze__parking_hours"] *
        (bronze_features["bronze__humidity_risk"] +
         bronze_features["bronze__condensation_risk"]) / 2
    )
    
    # Ratio humidité/température (indicateur de condensation)
    humidity_temp_ratio = (
        environment["metar_relative_humidity"] /
        (environment["metar_temperature_c"] + 273.15)  # Kelvin
    )
    
    # Interaction sel marin × humidité
    salt_humidity_interaction = (
        np.log1p(bronze_features["bronze__sea_salt_total"] * 1e10) *
        environment["metar_relative_humidity"] / 100
    )
    
    # Indice de pollution industrielle
    industrial_pollution_index = (
        np.log1p(environment["carbon_monoxide_mass_mixing_ratio"] * 1e6) * 0.3 +
        np.log1p(bronze_features["bronze__sulfur_compounds"] * 1e8) * 0.4 +
        np.log1p(bronze_features["bronze__black_carbon_total"] * 1e9) * 0.3
    )
    
    # Indice de corrosion galvanique (sel + humidité + température)
    galvanic_corrosion_index = (
        np.log1p(bronze_features["bronze__sea_salt_total"] * 1e10) *
        (environment["metar_relative_humidity"] / 100) *
        np.clip((environment["metar_temperature_c"] - 10) / 30, 0, 1)  # Normalisation température
    )
    
    # Synergie chimique (sulfates + nitrates + sel)
    chemical_synergy = (
        np.log1p(bronze_features["bronze__sulfur_compounds"] * 1e8) *
        np.log1p(bronze_features["bronze__nitrogen_compounds"] * 1e8) *
        np.log1p(bronze_features["bronze__sea_salt_total"] * 1e10)
    ) ** (1/3)  # Moyenne géométrique
    
    # Indice de stress environnemental total
    environmental_stress_index = (
        atmospheric_corrosion_index * 0.4 +
        chemical_aggressivity * 0.3 +
        industrial_pollution_index * 0.3
    )
    
    return pd.DataFrame({
        "silver__atmospheric_corrosion_index": atmospheric_corrosion_index,
        "silver__chemical_aggressivity": chemical_aggressivity,
        "silver__critical_exposure_time": critical_exposure_time,
        "silver__humidity_temp_ratio": humidity_temp_ratio,
        "silver__salt_humidity_interaction": salt_humidity_interaction,
        "silver__industrial_pollution_index": industrial_pollution_index,
        "silver__galvanic_corrosion_index": galvanic_corrosion_index,
        "silver__chemical_synergy": chemical_synergy,
        "silver__environmental_stress_index": environmental_stress_index,
    }, index=environment.index)


def make_silver_weather_interactions(environment):
    """Features Silver: Interactions météorologiques"""
    
    # Vent faible + humidité élevée = stagnation humide
    stagnant_humid_conditions = (
        (environment["metar_wind_speed_kn"] < 5) *
        (environment["metar_relative_humidity"] > 70)
    ).astype(float)
    
    # Précipitations + température modérée
    wet_moderate_temp = (
        environment["metar_hour_precipitation"] *
        ((environment["metar_temperature_c"] >= 10) &
         (environment["metar_temperature_c"] <= 30)).astype(float)
    )
    
    # Visibilité réduite (indicateur de pollution/aérosols)
    low_visibility = (environment["metar_visibility_mi"] < 3).astype(float)
    
    return pd.DataFrame({
        "silver__stagnant_humid_conditions": stagnant_humid_conditions,
        "silver__wet_moderate_temp": wet_moderate_temp,
        "silver__low_visibility": low_visibility,
    }, index=environment.index)


# ============================================================================
# GOLD FEATURES: Features temporelles avancées et patterns
# ============================================================================

def make_gold_temporal_patterns(environment, groups, bronze_features, silver_features):
    """Features Gold: Patterns temporels de corrosion"""
    
    # Créer un DataFrame temporaire avec les features nécessaires
    temp_df = pd.concat([
        environment[["aircraft_id"]],
        bronze_features[["bronze__critical_exposure_time", "bronze__humidity_risk", "bronze__sea_salt_total"]],
        silver_features[["silver__atmospheric_corrosion_index", "silver__chemical_aggressivity"]]
    ], axis=1)
    temp_groups = temp_df.groupby("aircraft_id", sort=False)
    
    # Accumulation de risque sur les derniers mois
    corrosion_risk_windows = [6, 12, 24]  # Focus sur moyen/long terme
    gold_features = {}
    
    for window in corrosion_risk_windows:
        # Accumulation d'exposition critique
        critical_exposure_cumsum = (
            temp_groups["bronze__critical_exposure_time"]
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        gold_features[f"gold__critical_exposure_last_{window}m"] = critical_exposure_cumsum
        
        # Moyenne de l'indice de corrosion atmosphérique
        atm_corr_mean = (
            temp_groups["silver__atmospheric_corrosion_index"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        gold_features[f"gold__atm_corrosion_avg_{window}m"] = atm_corr_mean
        
        # Max de l'agressivité chimique (pic d'exposition)
        chem_aggr_max = (
            temp_groups["silver__chemical_aggressivity"]
            .rolling(window, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )
        gold_features[f"gold__chem_aggr_max_{window}m"] = chem_aggr_max
        
        # Exposition au sel marin (moyenne)
        salt_exposure_mean = (
            temp_groups["bronze__sea_salt_total"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        gold_features[f"gold__salt_exposure_avg_{window}m"] = np.log1p(salt_exposure_mean * 1e10)
    
    # Tendances de dégradation
    atm_corr_trend_6m = (
        silver_features["silver__atmospheric_corrosion_index"] -
        temp_groups["silver__atmospheric_corrosion_index"].shift(6).reset_index(level=0, drop=True)
    )
    atm_corr_trend_12m = (
        silver_features["silver__atmospheric_corrosion_index"] -
        temp_groups["silver__atmospheric_corrosion_index"].shift(12).reset_index(level=0, drop=True)
    )
    gold_features["gold__corrosion_trend_6m"] = atm_corr_trend_6m
    gold_features["gold__corrosion_trend_12m"] = atm_corr_trend_12m
    
    # Variabilité des conditions (stress cyclique)
    humidity_volatility = (
        groups["metar_relative_humidity"]
        .rolling(12, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )
    gold_features["gold__humidity_volatility_12m"] = humidity_volatility
    
    # Ratio exposition récente vs historique (accélération locale)
    recent_exposure = (
        temp_groups["bronze__critical_exposure_time"]
        .rolling(6, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    historical_exposure = (
        temp_groups["bronze__critical_exposure_time"]
        .rolling(24, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    gold_features["gold__exposure_acceleration_ratio"] = recent_exposure / (historical_exposure + 1e-6)
    
    return pd.DataFrame(gold_features, index=environment.index)


def make_gold_cumulative_damage(environment, groups, silver_features):
    """Features Gold: Dommages cumulatifs et mémoire"""
    
    # Créer un DataFrame temporaire avec les features nécessaires
    temp_df = pd.concat([
        environment[["aircraft_id"]],
        silver_features[["silver__atmospheric_corrosion_index", "silver__chemical_aggressivity", "silver__critical_exposure_time"]]
    ], axis=1)
    temp_groups = temp_df.groupby("aircraft_id", sort=False)
    
    # Dose cumulative de corrosion (intégrale du risque)
    corrosion_dose = (
        temp_groups["silver__atmospheric_corrosion_index"]
        .cumsum()
        .reset_index(level=0, drop=True)
    )
    
    # Dose chimique cumulative
    chemical_dose = (
        temp_groups["silver__chemical_aggressivity"]
        .cumsum()
        .reset_index(level=0, drop=True)
    )
    
    # Temps total en conditions critiques
    critical_time_total = (
        temp_groups["silver__critical_exposure_time"]
        .cumsum()
        .reset_index(level=0, drop=True)
    )
    
    # Ratio dose récente / dose totale (accélération)
    recent_dose = (
        temp_groups["silver__atmospheric_corrosion_index"]
        .rolling(6, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    dose_acceleration = recent_dose / (corrosion_dose + 1e-6)
    
    return pd.DataFrame({
        "gold__corrosion_dose_cumulative": corrosion_dose,
        "gold__chemical_dose_cumulative": chemical_dose,
        "gold__critical_time_cumulative": critical_time_total,
        "gold__dose_acceleration": dose_acceleration,
    }, index=environment.index)


def make_gold_aircraft_profile(environment, groups, history_count):
    """Features Gold: Profil de l'avion et son historique"""
    
    # Âge de l'avion (nombre de mois d'historique)
    aircraft_age = history_count
    
    # Utilisation moyenne (parking moyen)
    avg_parking = (
        groups[["total_parking_minutes"]]
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # Exposition moyenne au sel marin (indicateur de zone côtière)
    avg_sea_salt = (
        groups[["sea_salt_aerosol_003_05_mixing_ratio"]]
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # Stabilité des conditions (écart-type de l'humidité sur tout l'historique)
    humidity_stability = (
        groups[["metar_relative_humidity"]]
        .expanding(min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )
    
    # Température moyenne historique
    avg_temp = (
        groups[["metar_temperature_c"]]
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # Variabilité de la température (stress thermique)
    temp_range = (
        groups[["metar_temperature_c"]]
        .expanding(min_periods=2)
        .apply(lambda x: x.max() - x.min())
        .reset_index(level=0, drop=True)
    )
    
    # Fréquence de parking prolongé (>500h/mois)
    long_parking_freq = (
        (environment["total_parking_minutes"] > 30000).astype(float)
    )
    avg_long_parking = (
        groups[["aircraft_id"]]
        .apply(lambda x: pd.Series(long_parking_freq[x.index]).expanding(min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    
    return pd.DataFrame({
        "gold__aircraft_age_months": aircraft_age,
        "gold__avg_parking_hours": avg_parking.iloc[:, 0] / 60,
        "gold__avg_sea_salt_exposure": np.log1p(avg_sea_salt.iloc[:, 0] * 1e10),
        "gold__humidity_stability": humidity_stability.iloc[:, 0],
        "gold__avg_temperature": avg_temp.iloc[:, 0],
        "gold__temperature_range": temp_range.iloc[:, 0],
        "gold__long_parking_frequency": avg_long_parking.iloc[:, 0] if len(avg_long_parking.shape) > 1 else avg_long_parking,
    }, index=environment.index)


# ============================================================================
# FEATURES CLASSIQUES (simplifiées pour réduire l'overfitting)
# ============================================================================

def make_calendar_features(environment):
    month = environment["date"].dt.month
    return pd.DataFrame(
        {
            "calendar_year": environment["date"].dt.year,
            "calendar_month": month,
            "calendar_month_sin": np.sin(2 * np.pi * month / 12),
            "calendar_month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=environment.index,
    )


def make_selective_rolling_features(environment, groups, key_columns):
    """Rolling features seulement sur les colonnes clés pour éviter l'overfitting"""
    feature_parts = []
    
    for window in [6, 12]:  # Réduction des fenêtres
        rolling = groups[key_columns].rolling(window, min_periods=1)
        rolling_mean = rolling.mean().reset_index(level=0, drop=True)
        
        feature_parts.append(
            rolling_mean.add_suffix(f"__last_{window}_mean")
        )
    
    return feature_parts


def make_selective_lag_features(environment, groups, key_columns):
    """Lag features seulement sur les colonnes clés"""
    lag1 = groups[key_columns].shift(1)
    
    return [
        lag1.add_suffix("__lag1"),
        (environment[key_columns] - lag1).add_suffix("__delta_lag1"),
    ]


# ============================================================================
# PIPELINE PRINCIPALE
# ============================================================================

def make_platinum_advanced_features(environment, groups, bronze_features, silver_features, gold_features):
    """Features Platinum: Features avancées de science de la corrosion"""
    
    # Créer DataFrame temporaire
    temp_df = pd.concat([
        environment[["aircraft_id"]],
        bronze_features[["bronze__sea_salt_total", "bronze__sulfur_compounds", "bronze__iso_corrosivity"]],
        silver_features[["silver__atmospheric_corrosion_index", "silver__chemical_aggressivity"]],
        gold_features[["gold__corrosion_dose_cumulative", "gold__chemical_dose_cumulative"]]
    ], axis=1)
    temp_groups = temp_df.groupby("aircraft_id", sort=False)
    
    platinum_features = {}
    
    # Taux de corrosion instantané (dérivée de la dose)
    corrosion_rate = (
        temp_groups["gold__corrosion_dose_cumulative"]
        .diff()
        .fillna(0)
        .reset_index(level=0, drop=True)
    )
    platinum_features["platinum__corrosion_rate"] = corrosion_rate
    
    # Accélération de la corrosion (dérivée seconde)
    corrosion_acceleration = (
        temp_groups["gold__corrosion_dose_cumulative"]
        .diff()
        .diff()
        .fillna(0)
        .reset_index(level=0, drop=True)
    )
    platinum_features["platinum__corrosion_acceleration"] = corrosion_acceleration
    
    # Ratio dose chimique / dose totale (dominance chimique)
    chemical_dominance = (
        temp_df["gold__chemical_dose_cumulative"] / 
        (temp_df["gold__corrosion_dose_cumulative"] + 1e-6)
    )
    platinum_features["platinum__chemical_dominance"] = chemical_dominance
    
    # Persistance de l'agressivité (combien de mois consécutifs avec forte agressivité)
    high_aggr = (silver_features["silver__chemical_aggressivity"] > 
                 silver_features["silver__chemical_aggressivity"].quantile(0.75)).astype(int)
    aggr_streak = temp_groups.apply(
        lambda x: pd.Series(high_aggr[x.index]).groupby(
            (pd.Series(high_aggr[x.index]) != pd.Series(high_aggr[x.index]).shift()).cumsum()
        ).cumsum()
    ).reset_index(level=0, drop=True)
    platinum_features["platinum__aggressivity_streak"] = aggr_streak.iloc[:, 0] if len(aggr_streak.shape) > 1 else aggr_streak
    
    # Ratio exposition récente vs dose totale (sur-exposition récente)
    recent_exposure_6m = (
        temp_groups["silver__atmospheric_corrosion_index"]
        .rolling(6, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    overexposure_ratio = recent_exposure_6m / (temp_df["gold__corrosion_dose_cumulative"] + 1)
    platinum_features["platinum__overexposure_ratio"] = overexposure_ratio
    
    # Indice de vieillissement accéléré (combinaison âge × dose × taux)
    aging_index = (
        np.log1p(temp_df["gold__corrosion_dose_cumulative"]) *
        np.log1p(np.abs(corrosion_rate) + 1e-6)
    )
    platinum_features["platinum__accelerated_aging_index"] = aging_index
    
    return pd.DataFrame(platinum_features, index=environment.index)


def build_history_feature_table(environment):
    """Pipeline complète Bronze → Silver → Gold → Platinum"""
    environment = add_dates(environment)
    groups = environment.groupby("aircraft_id", sort=False)
    history_count = groups.cumcount() + 1
    
    # BRONZE: Features de base
    bronze_corrosion = make_bronze_corrosion_factors(environment)
    bronze_aerosol = make_bronze_aerosol_features(environment)
    bronze_features = pd.concat([bronze_corrosion, bronze_aerosol], axis=1)
    
    # SILVER: Interactions et indices
    silver_corrosion = make_silver_corrosion_indices(environment, bronze_features)
    silver_weather = make_silver_weather_interactions(environment)
    silver_features = pd.concat([silver_corrosion, silver_weather], axis=1)
    
    # GOLD: Patterns temporels et cumulatifs
    gold_temporal = make_gold_temporal_patterns(environment, groups, bronze_features, silver_features)
    gold_cumulative = make_gold_cumulative_damage(environment, groups, silver_features)
    gold_profile = make_gold_aircraft_profile(environment, groups, history_count)
    gold_features = pd.concat([gold_temporal, gold_cumulative, gold_profile], axis=1)
    
    # PLATINUM: Features avancées de science de la corrosion
    platinum_features = make_platinum_advanced_features(environment, groups, bronze_features, silver_features, gold_features)
    
    # Features classiques sélectives (colonnes clés uniquement)
    key_columns = [
        "metar_relative_humidity",
        "metar_temperature_c",
        "total_parking_minutes",
        "sea_salt_aerosol_003_05_mixing_ratio",
    ]
    
    feature_parts = [
        bronze_features,
        silver_features,
        gold_features,
        platinum_features,
        pd.DataFrame({"history_count": history_count}, index=environment.index),
        make_calendar_features(environment),
    ]
    
    # Ajout sélectif de rolling et lag features
    feature_parts.extend(make_selective_rolling_features(environment, groups, key_columns))
    feature_parts.extend(make_selective_lag_features(environment, groups, key_columns))
    
    features = pd.concat(feature_parts, axis=1).astype("float32")
    keys = environment[ID_COLUMNS].copy()
    return pd.concat([keys, features], axis=1)
