from dataclasses import dataclass
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = REPO_DIR.parent
DATA_DIR = ROOT_DIR / "data"


@dataclass(frozen=True)
class PathConfig:
    environment_path: Path = DATA_DIR / "environment_training.csv"
    labels_path: Path = DATA_DIR / "prediction_training.csv"
    model_output_path: Path = REPO_DIR / "LightGBM" / "lightgbm_history_model.txt"
    importance_output_path: Path = DATA_DIR / "lightgbm_history_feature_importance.csv"


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 800
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 40
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.05
    reg_lambda: float = 0.2
    early_stopping_rounds: int = 60
    seed: int = 42


@dataclass(frozen=True)
class ValidationConfig:
    valid_size: float = 0.2
    cv_splits: int = 5
