import pandas as pd
import os
from src.bankcdProject import logger
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import joblib
from src.bankcdProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the appropriate model based on model_name."""
        if self.config.model_name == "XGBoost":
            return XGBClassifier(
                subsample=self.config.subsample,
                reg_lambda=self.config.reg_lambda,
                reg_alpha=self.config.reg_alpha,
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                colsample_bytree=self.config.colsample_bytree,
                eval_metric=self.config.eval_metric,
                random_state=self.config.random_state
            )
        elif self.config.model_name == "GradientBoosting":
            return GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state
            )
        elif self.config.model_name == "SVM":
            return SVC(
                kernel=self.config.kernel,
                C=self.config.C,
                probability=self.config.probability,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_name}")

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        self.model.fit(train_x, train_y)
        model_path = os.path.join(self.config.root_dir, f"{self.config.model_name}.joblib")
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved at: {model_path}")
