import pandas as pd
import numpy as np


def select_top_features_by_gain(importance_df, top_n=50, min_gain=1000):
    """
    Sélectionne les features les plus importantes basées sur le gain.
    
    Args:
        importance_df: DataFrame avec colonnes 'feature', 'gain', 'split'
        top_n: Nombre maximum de features à garder
        min_gain: Gain minimum requis
    
    Returns:
        Liste des noms de features sélectionnées
    """
    # Filtrer par gain minimum
    filtered = importance_df[importance_df['gain'] >= min_gain].copy()
    
    # Trier par gain décroissant
    filtered = filtered.sort_values('gain', ascending=False)
    
    # Prendre les top_n
    selected = filtered.head(top_n)
    
    print(f"\nFeature Selection:")
    print(f"  - Features totales: {len(importance_df)}")
    print(f"  - Features avec gain >= {min_gain}: {len(filtered)}")
    print(f"  - Features sélectionnées: {len(selected)}")
    print(f"  - Gain total sélectionné: {selected['gain'].sum():.0f}")
    print(f"  - % du gain total: {100 * selected['gain'].sum() / importance_df['gain'].sum():.1f}%")
    
    return selected['feature'].tolist()


def remove_correlated_features(X, feature_list, threshold=0.95):
    """
    Élimine les features fortement corrélées.
    
    Args:
        X: DataFrame avec les features
        feature_list: Liste des features à analyser
        threshold: Seuil de corrélation (défaut 0.95)
    
    Returns:
        Liste des features après élimination des corrélations
    """
    # Calculer la matrice de corrélation
    X_subset = X[feature_list]
    corr_matrix = X_subset.corr().abs()
    
    # Trouver les paires de features fortement corrélées
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Features à supprimer
    to_drop = [
        column for column in upper_triangle.columns 
        if any(upper_triangle[column] > threshold)
    ]
    
    # Features à garder
    features_to_keep = [f for f in feature_list if f not in to_drop]
    
    print(f"\nCorrelation Analysis:")
    print(f"  - Features analysées: {len(feature_list)}")
    print(f"  - Features corrélées (>{threshold}): {len(to_drop)}")
    print(f"  - Features gardées: {len(features_to_keep)}")
    
    if to_drop:
        print(f"\n  Features supprimées pour corrélation:")
        for feat in to_drop[:10]:  # Afficher les 10 premières
            print(f"    - {feat}")
        if len(to_drop) > 10:
            print(f"    ... et {len(to_drop) - 10} autres")
    
    return features_to_keep


def select_features_smart(importance_path, X, top_n=50, min_gain=800, corr_threshold=0.95):
    """
    Pipeline complète de sélection de features.
    
    Args:
        importance_path: Chemin vers le CSV d'importance
        X: DataFrame avec toutes les features
        top_n: Nombre maximum de features
        min_gain: Gain minimum
        corr_threshold: Seuil de corrélation
    
    Returns:
        Liste finale des features sélectionnées
    """
    print("=" * 70)
    print("FEATURE SELECTION INTELLIGENTE")
    print("=" * 70)
    
    # Charger les importances
    importance_df = pd.read_csv(importance_path)
    
    # Étape 1: Sélection par gain
    selected_by_gain = select_top_features_by_gain(importance_df, top_n, min_gain)
    
    # Étape 2: Élimination des corrélations
    # Filtrer les features qui existent dans X
    available_features = [f for f in selected_by_gain if f in X.columns]
    
    if len(available_features) < len(selected_by_gain):
        print(f"\n{len(selected_by_gain) - len(available_features)} features non disponibles dans X")
    
    final_features = remove_correlated_features(X, available_features, corr_threshold)
    
    print("\n" + "=" * 70)
    print(f"SELECTION FINALE: {len(final_features)} features")
    print("=" * 70)
    
    return final_features


def analyze_feature_categories(selected_features):
    """
    Analyse la distribution des catégories de features sélectionnées.
    """
    categories = {
        'GOLD': 0,
        'PLATINUM': 0,
        'SCIENTIFIC': 0,
        'BRONZE': 0,
        'SILVER': 0,
        'ROLLING': 0,
        'LAG': 0,
        'CALENDAR': 0,
        'OTHER': 0
    }
    
    for feat in selected_features:
        if 'gold__' in feat:
            categories['GOLD'] += 1
        elif 'platinum__' in feat:
            categories['PLATINUM'] += 1
        elif 'sci__' in feat:
            categories['SCIENTIFIC'] += 1
        elif 'bronze__' in feat:
            categories['BRONZE'] += 1
        elif 'silver__' in feat:
            categories['SILVER'] += 1
        elif '__last_' in feat or '__mean' in feat:
            categories['ROLLING'] += 1
        elif '__lag' in feat or '__delta' in feat:
            categories['LAG'] += 1
        elif 'calendar' in feat:
            categories['CALENDAR'] += 1
        else:
            categories['OTHER'] += 1
    
    print("\nDistribution par categorie:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = 100 * count / len(selected_features)
            print(f"  {cat:15s}: {count:3d} features ({pct:5.1f}%)")
    
    return categories

# Made with Bob
