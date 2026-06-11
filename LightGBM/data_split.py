"""
Module pour le split des donnees en train/valid/test avec stratification par avion.
Split: 60% train, 20% validation, 20% test
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_train_valid_test(X, y, groups, train_size=0.6, valid_size=0.2, test_size=0.2, seed=42):
    """
    Split les donnees en train/valid/test en respectant les groupes (avions).
    
    Args:
        X: Features
        y: Target
        groups: Groupes (aircraft_id)
        train_size: Proportion du train set (default 0.6)
        valid_size: Proportion du validation set (default 0.2)
        test_size: Proportion du test set (default 0.2)
        seed: Random seed
        
    Returns:
        train_idx, valid_idx, test_idx: Indices pour chaque split
    """
    assert abs(train_size + valid_size + test_size - 1.0) < 1e-6, "Les proportions doivent sommer a 1"
    
    # Premier split: train vs (valid+test)
    splitter1 = GroupShuffleSplit(
        n_splits=1,
        test_size=(valid_size + test_size),
        random_state=seed,
    )
    train_idx, temp_idx = next(splitter1.split(X, y, groups))
    
    # Deuxieme split: valid vs test
    # On recalcule la proportion relative
    relative_test_size = test_size / (valid_size + test_size)
    
    X_temp = X.iloc[temp_idx]
    y_temp = y.iloc[temp_idx]
    groups_temp = groups.iloc[temp_idx]
    
    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_size,
        random_state=seed + 1,
    )
    valid_idx_temp, test_idx_temp = next(splitter2.split(X_temp, y_temp, groups_temp))
    
    # Convertir les indices temporaires en indices globaux
    valid_idx = temp_idx[valid_idx_temp]
    test_idx = temp_idx[test_idx_temp]
    
    return train_idx, valid_idx, test_idx


def print_split_info(X, y, groups, train_idx, valid_idx, test_idx):
    """Affiche les informations sur les splits."""
    print("\n" + "=" * 80)
    print("SPLIT INFORMATION")
    print("=" * 80)
    
    total_samples = len(X)
    total_aircraft = groups.nunique()
    
    splits = {
        "Train": train_idx,
        "Valid": valid_idx,
        "Test": test_idx,
    }
    
    for name, idx in splits.items():
        n_samples = len(idx)
        n_aircraft = groups.iloc[idx].nunique()
        pos_rate = y.iloc[idx].mean()
        
        print(f"\n{name} Set:")
        print(f"  Samples:  {n_samples:,} ({100 * n_samples / total_samples:.1f}%)")
        print(f"  Aircraft: {n_aircraft} ({100 * n_aircraft / total_aircraft:.1f}%)")
        print(f"  Positive: {pos_rate:.3f}")
    
    print("=" * 80)


def get_split_datasets(X, y, groups, train_idx, valid_idx, test_idx):
    """Retourne les datasets splites."""
    return {
        "train": {
            "X": X.iloc[train_idx],
            "y": y.iloc[train_idx],
            "groups": groups.iloc[train_idx],
        },
        "valid": {
            "X": X.iloc[valid_idx],
            "y": y.iloc[valid_idx],
            "groups": groups.iloc[valid_idx],
        },
        "test": {
            "X": X.iloc[test_idx],
            "y": y.iloc[test_idx],
            "groups": groups.iloc[test_idx],
        },
    }

# Made with Bob
