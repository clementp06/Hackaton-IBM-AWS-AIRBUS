import numpy as np
import pandas as pd


def make_scientific_corrosion_features(environment):
    """
    Features basées sur le rapport scientifique de corrosion des avions.
    Implémente les découvertes clés du rapport de recherche.
    """
    
    features = {}
    
    # ============================================================================
    # 1. TIME OF WETNESS (TOW) - ISO 9223 Standard
    # RH > 80% ET T > 0°C = conditions de film d'eau
    # ============================================================================
    
    tow_iso = (
        (environment["metar_relative_humidity"] > 80) &
        (environment["metar_temperature_c"] > 0)
    ).astype(float)
    features["sci__time_of_wetness_iso"] = tow_iso
    
    # TOW avec différents seuils RH (70%, 78%, 90%, 93%)
    for threshold in [70, 78, 90, 93]:
        tow = (
            (environment["metar_relative_humidity"] > threshold) &
            (environment["metar_temperature_c"] > 0)
        ).astype(float)
        features[f"sci__tow_rh{threshold}"] = tow
    
    # ============================================================================
    # 2. CONDENSATION RISK - Dew Point Analysis
    # Surface condensation when T approaches dew point
    # ============================================================================
    
    dew_point_diff = environment["metar_temperature_c"] - environment["metar_dew_point_c"]
    
    # Condensation très probable (< 1°C)
    features["sci__condensation_imminent"] = (dew_point_diff < 1).astype(float)
    
    # Condensation probable (< 2°C)
    features["sci__condensation_likely"] = (dew_point_diff < 2).astype(float)
    
    # Condensation possible (< 3°C)
    features["sci__condensation_possible"] = (dew_point_diff < 3).astype(float)
    
    # Gradient de condensation (plus proche = plus risqué)
    features["sci__condensation_gradient"] = np.exp(-dew_point_diff / 2)
    
    # ============================================================================
    # 3. FILIFORM CORROSION CONDITIONS
    # RH 78-90% + surface légèrement acide
    # ============================================================================
    
    filiform_rh = (
        (environment["metar_relative_humidity"] >= 78) &
        (environment["metar_relative_humidity"] <= 90)
    ).astype(float)
    features["sci__filiform_rh_range"] = filiform_rh
    
    # Filiform avec acidité (HNO3 + SO2)
    acid_load = (
        np.log1p(environment["hno3"] * 1e9) +
        np.log1p(environment["sulphur_dioxide_mass_mixing_ratio"] * 1e8)
    )
    features["sci__filiform_risk"] = filiform_rh * acid_load
    
    # ============================================================================
    # 4. SEA SALT DELIQUESCENCE
    # MgCl2 deliquescence à 33% RH, NaCl-MgCl2 mixtures à 15.9% RH
    # ============================================================================
    
    sea_salt_total = (
        environment["sea_salt_aerosol_003_05_mixing_ratio"] +
        environment["sea_salt_aerosol_05_5_mixing_ratio"] +
        environment["sea_salt_aerosol_5_20_mixing_ratio"]
    )
    
    # Deliquescence à bas RH (15-35%)
    low_rh_deliquescence = (
        (environment["metar_relative_humidity"] > 15) &
        (environment["metar_relative_humidity"] < 35) &
        (sea_salt_total > 0)
    ).astype(float)
    features["sci__salt_deliquescence_low_rh"] = low_rh_deliquescence * np.log1p(sea_salt_total * 1e10)
    
    # Deliquescence NaCl standard (>75% RH)
    nacl_deliquescence = (
        (environment["metar_relative_humidity"] > 75) &
        (sea_salt_total > 0)
    ).astype(float)
    features["sci__salt_deliquescence_nacl"] = nacl_deliquescence * np.log1p(sea_salt_total * 1e10)
    
    # ============================================================================
    # 5. WIND-DRIVEN SALT DEPOSITION
    # Corrosion increases with wind > 5 m/s (≈ 9.7 knots)
    # ============================================================================
    
    # Conversion knots to m/s: 1 knot = 0.514444 m/s
    wind_ms = environment["metar_wind_speed_kn"] * 0.514444
    
    # Vent fort (> 5 m/s) avec sel marin
    strong_wind_salt = (wind_ms > 5).astype(float) * np.log1p(sea_salt_total * 1e10)
    features["sci__wind_driven_salt_deposition"] = strong_wind_salt
    
    # Gradient de vent (effet non-linéaire)
    features["sci__wind_deposition_factor"] = np.where(
        wind_ms > 5,
        (wind_ms - 5) * np.log1p(sea_salt_total * 1e10),
        0
    )
    
    # ============================================================================
    # 6. PARTICLE SIZE DISTRIBUTION EFFECTS
    # Larger particles deposit locally, finer travel farther
    # ============================================================================
    
    # Ratio particules fines/grosses (indicateur de transport longue distance)
    fine_coarse_ratio = (
        environment["sea_salt_aerosol_003_05_mixing_ratio"] /
        (environment["sea_salt_aerosol_5_20_mixing_ratio"] + 1e-12)
    )
    features["sci__salt_fine_coarse_ratio"] = np.log1p(fine_coarse_ratio)
    
    # Déposition locale (particules grosses)
    features["sci__local_salt_deposition"] = np.log1p(
        environment["sea_salt_aerosol_5_20_mixing_ratio"] * 1e10
    )
    
    # Transport longue distance (particules fines)
    features["sci__regional_salt_transport"] = np.log1p(
        environment["sea_salt_aerosol_003_05_mixing_ratio"] * 1e10
    )
    
    # ============================================================================
    # 7. MULTI-POLLUTANT CORROSIVITY (ISO/ICP Materials)
    # NOx, O3, HNO3, SO2, PM10 + Temperature + RH
    # ============================================================================
    
    # Dose-response multi-polluants pour aluminium
    multi_pollutant_dose = (
        np.log1p(environment["nitrogen_dioxide_mass_mixing_ratio"] * 1e8) * 0.3 +
        np.log1p(environment["hno3"] * 1e9) * 0.3 +
        np.log1p(environment["sulphur_dioxide_mass_mixing_ratio"] * 1e8) * 0.2 +
        np.log1p(environment["ozone_mass_mixing_ratio"] * 1e7) * 0.1 +
        np.log1p(environment["dust_aerosol_003_055_mixing_ratio"] * 1e9) * 0.1
    )
    features["sci__multi_pollutant_dose"] = multi_pollutant_dose
    
    # Multi-pollutant avec humidité (gatekeeper)
    features["sci__multi_pollutant_wet"] = (
        multi_pollutant_dose * environment["metar_relative_humidity"] / 100
    )
    
    # ============================================================================
    # 8. NITRIC ACID DOMINANCE
    # HNO3 is strong acid + hygroscopic salts
    # ============================================================================
    
    # HNO3 comme driver principal
    features["sci__hno3_corrosivity"] = np.log1p(environment["hno3"] * 1e9)
    
    # HNO3 + humidité (films persistants)
    features["sci__hno3_wet_film"] = (
        np.log1p(environment["hno3"] * 1e9) *
        (environment["metar_relative_humidity"] / 100) ** 2
    )
    
    # NO2 + particules (urban corrosion pattern)
    features["sci__no2_particle_corrosion"] = (
        np.log1p(environment["nitrogen_dioxide_mass_mixing_ratio"] * 1e8) *
        np.log1p(environment["dust_aerosol_003_055_mixing_ratio"] * 1e9)
    )
    
    # ============================================================================
    # 9. SULFATE INTERACTION EFFECTS
    # Sulfate can inhibit OR promote depending on context
    # ============================================================================
    
    sulfate_total = (
        environment["sulphate_aerosol_mixing_ratio"] +
        environment["sulphur_dioxide_mass_mixing_ratio"]
    )
    
    # Sulfate avec chloride (interaction complexe)
    features["sci__sulfate_chloride_interaction"] = (
        np.log1p(sulfate_total * 1e8) *
        np.log1p(sea_salt_total * 1e10)
    )
    
    # Sulfate en environnement marin-industriel
    features["sci__marine_industrial_sulfate"] = (
        np.log1p(sulfate_total * 1e8) *
        np.log1p(sea_salt_total * 1e10) *
        (environment["metar_relative_humidity"] / 100)
    )
    
    # ============================================================================
    # 10. ATMOSPHERIC OXIDATION CHEMISTRY
    # OH, H2O2, O3 drive acid formation
    # ============================================================================
    
    # Potentiel d'oxydation atmosphérique
    oxidation_potential = (
        np.log1p(environment["oh"] * 1e14) * 0.4 +
        np.log1p(environment["h2o2"] * 1e9) * 0.3 +
        np.log1p(environment["ozone_mass_mixing_ratio"] * 1e7) * 0.3
    )
    features["sci__atmospheric_oxidation"] = oxidation_potential
    
    # Formation d'acide sulfurique (SO2 + oxidants)
    features["sci__sulfuric_acid_formation"] = (
        np.log1p(environment["sulphur_dioxide_mass_mixing_ratio"] * 1e8) *
        oxidation_potential
    )
    
    # Formation d'acide nitrique (NOx + oxidants)
    features["sci__nitric_acid_formation"] = (
        np.log1p(environment["nitrogen_dioxide_mass_mixing_ratio"] * 1e8) *
        oxidation_potential
    )
    
    # ============================================================================
    # 11. DUST AS MOISTURE TRAP
    # Dust creates under-deposit conditions
    # ============================================================================
    
    dust_total = (
        environment["dust_aerosol_003_055_mixing_ratio"] +
        environment["dust_aerosol_055_09_mixing_ratio"] +
        environment["dust_aerosol_09_20_mixing_ratio"]
    )
    
    # Poussière + humidité (piégeage d'eau)
    features["sci__dust_moisture_trap"] = (
        np.log1p(dust_total * 1e9) *
        (environment["metar_relative_humidity"] / 100) ** 2
    )
    
    # Poussière avec contaminants (sulfate, nitrate, chloride)
    features["sci__contaminated_dust"] = (
        np.log1p(dust_total * 1e9) *
        (np.log1p(sulfate_total * 1e8) + 
         np.log1p(environment["hno3"] * 1e9) +
         np.log1p(sea_salt_total * 1e10)) / 3
    )
    
    # ============================================================================
    # 12. VISIBILITY AS WETNESS PROXY
    # Low visibility = fog, mist, drizzle, haze
    # ============================================================================
    
    # Visibilité réduite avec humidité élevée (brouillard)
    fog_conditions = (
        (environment["metar_visibility_mi"] < 3) &
        (environment["metar_relative_humidity"] > 90)
    ).astype(float)
    features["sci__fog_corrosion_risk"] = fog_conditions
    
    # Brume/haze avec pollution
    haze_pollution = (
        (environment["metar_visibility_mi"] < 6) &
        (environment["metar_relative_humidity"] < 90)
    ).astype(float) * multi_pollutant_dose
    features["sci__haze_pollution_risk"] = haze_pollution
    
    # ============================================================================
    # 13. PRECIPITATION WASHING EFFECTS
    # Moderate rain increases wetness, heavy rain washes contaminants
    # ============================================================================
    
    precip = environment["metar_hour_precipitation"]
    
    # Précipitation modérée (augmente humidité)
    features["sci__moderate_rain_wetting"] = np.where(
        (precip > 0) & (precip < 5),
        precip,
        0
    )
    
    # Précipitation forte (lavage possible)
    features["sci__heavy_rain_washing"] = np.where(
        precip >= 5,
        -np.log1p(precip),  # Effet négatif (lavage)
        0
    )
    
    # Précipitation avec sel (transport)
    features["sci__rain_salt_transport"] = (
        precip * np.log1p(sea_salt_total * 1e10)
    )
    
    # ============================================================================
    # 14. THERMAL CYCLING AND CONDENSATION
    # Temperature changes drive moisture redistribution
    # ============================================================================
    
    # Différence température-dew point normalisée
    features["sci__thermal_condensation_risk"] = np.exp(-np.abs(dew_point_diff) / 3)
    
    # Température dans zone critique corrosion (15-35°C)
    temp_in_critical_range = (
        (environment["metar_temperature_c"] >= 15) &
        (environment["metar_temperature_c"] <= 35)
    ).astype(float)
    features["sci__critical_temp_range"] = temp_in_critical_range
    
    # Température optimale pour corrosion (20-30°C)
    temp_optimal = (
        (environment["metar_temperature_c"] >= 20) &
        (environment["metar_temperature_c"] <= 30)
    ).astype(float)
    features["sci__optimal_corrosion_temp"] = temp_optimal
    
    # ============================================================================
    # 15. HYDROPHILIC AEROSOL EFFECTS
    # Hydrophilic particles retain water better
    # ============================================================================
    
    # Matière organique hydrophile (rétention d'eau)
    features["sci__hydrophilic_water_retention"] = np.log1p(
        environment["hydrophilic_organic_matter_aerosol_mixing_ratio"] * 1e8
    )
    
    # Carbone noir hydrophile (pollution combustion)
    features["sci__hydrophilic_bc_pollution"] = np.log1p(
        environment["hydrophilic_black_carbon_aerosol_mixing_ratio"] * 1e9
    )
    
    # Ratio hydrophile/hydrophobe (capacité rétention eau)
    hydrophilic_ratio = (
        (environment["hydrophilic_organic_matter_aerosol_mixing_ratio"] +
         environment["hydrophilic_black_carbon_aerosol_mixing_ratio"]) /
        (environment["hydrophobic_organic_matter_aerosol_mixing_ratio"] +
         environment["hydrophobic_black_carbon_aerosol_mixing_ratio"] + 1e-12)
    )
    features["sci__hydrophilic_ratio"] = np.log1p(hydrophilic_ratio)
    
    # ============================================================================
    # 16. COMBUSTION PLUME INDICATORS
    # CO, ethane, propane indicate traffic/industrial influence
    # ============================================================================
    
    # Indice de plume de combustion
    combustion_index = (
        np.log1p(environment["carbon_monoxide_mass_mixing_ratio"] * 1e6) * 0.4 +
        np.log1p(environment["ethane"] * 1e9) * 0.3 +
        np.log1p(environment["c3h8"] * 1e9) * 0.3
    )
    features["sci__combustion_plume_index"] = combustion_index
    
    # Combustion + pollution acide
    features["sci__combustion_acid_synergy"] = (
        combustion_index *
        (np.log1p(environment["nitrogen_dioxide_mass_mixing_ratio"] * 1e8) +
         np.log1p(environment["sulphur_dioxide_mass_mixing_ratio"] * 1e8)) / 2
    )
    
    # ============================================================================
    # 17. BIOGENIC VOC CONTEXT
    # Isoprene indicates biogenic activity and photochemistry
    # ============================================================================
    
    # Isoprène (activité biogénique)
    features["sci__biogenic_voc"] = np.log1p(environment["isoprene"] * 1e9)
    
    # Formaldéhyde (produit d'oxydation)
    features["sci__formaldehyde_oxidation"] = np.log1p(environment["formaldehyde"] * 1e9)
    
    # ============================================================================
    # 18. SPECIFIC HUMIDITY AS ABSOLUTE MOISTURE
    # Absolute moisture load independent of temperature
    # ============================================================================
    
    # Humidité spécifique élevée
    features["sci__high_specific_humidity"] = (
        environment["specific_humidity"] > 0.015
    ).astype(float)
    
    # Humidité spécifique × température
    features["sci__moisture_temp_product"] = (
        environment["specific_humidity"] * environment["temperature"]
    )
    
    return pd.DataFrame(features, index=environment.index)

# Made with Bob
