import pandas as pd
import os
from src.bankcdProject import logger
from xgboost import XGBClassifier
import joblib
from src.bankcdProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]

        xgb_model = XGBClassifier(
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
        xgb_model.fit(train_x, train_y)

        joblib.dump(xgb_model, os.path.join(self.config.root_dir, self.config.model_name))