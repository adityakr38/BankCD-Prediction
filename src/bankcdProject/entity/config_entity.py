from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    preprocessing: Dict[str, Any] 

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict 
    preprocessing: Dict[str, Any]

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    target_column: str

    #XGBoost
    subsample: float
    reg_lambda: int
    reg_alpha: int
    gamma: int
    colsample_bytree: float
    
    #SVM
    kernel : str
    C: float
    probability: bool

    #For GB and XGB
    max_depth: int
    learning_rate: float
    eval_metric: str
    random_state: int
    n_estimators: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    model_name: str 
    all_params: dict
    metric_file_name: Path
    target_column: str