from src.bankcdProject.constant import *
from src.bankcdProject.utils.common import read_yaml, create_directories
from src.bankcdProject.entity.config_entity import (DataIngestionConfig,
                                                    DataValidationConfig, DataTransformationConfig, ModelTrainerConfig,ModelEvaluationConfig)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        preprocessing_config = self.config.preprocessing
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            preprocessing=preprocessing_config
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        preprocessing_config = self.config.preprocessing 

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=Path(config.STATUS_FILE),
            unzip_data_dir=Path(config.unzip_data_dir),
            all_schema=schema,
            preprocessing=preprocessing_config  
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config

    def get_model_trainer_config(self, model_type: str) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params[model_type]  
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=model_type,
            target_column=schema.name,
            
            # Common parameters
            n_estimators=params.n_estimators if hasattr(params, "n_estimators") else None,
            max_depth=params.max_depth if hasattr(params, "max_depth") else None,
            learning_rate=params.learning_rate if hasattr(params, "learning_rate") else None,
            random_state=params.random_state,

            # XGBoost-specific parameters
            subsample=params.subsample if model_type == "XGBoost" else None,
            reg_lambda=params.reg_lambda if model_type == "XGBoost" else None,
            reg_alpha=params.reg_alpha if model_type == "XGBoost" else None,
            gamma=params.gamma if model_type == "XGBoost" else None,
            colsample_bytree=params.colsample_bytree if model_type == "XGBoost" else None,
            eval_metric=params.eval_metric if model_type == "XGBoost" else None,

            # SVM-specific parameters
            kernel=params.kernel if model_type == "SVM" else None,
            C=params.C if model_type == "SVM" else None,
            probability=params.probability if model_type == "SVM" else None
        )

        return model_trainer_config


    def get_model_evaluation_config(self, model_type: str) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        params = getattr(self.params, model_type)
        schema = self.schema.TARGET_COLUMN

        model_path = f"{config.model_path}/{model_type}.joblib"

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=Path(model_path),
            model_name=model_type,  
            all_params=params,
            metric_file_name=config.metric_file_name,
            target_column=schema.name
        )

        return model_evaluation_config
