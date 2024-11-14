import os
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from src.bankcdProject.utils.common import save_json
from urllib.parse import urlparse
import numpy as np
import joblib
from src.bankcdProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self, y_test, y_test_pred, y_test_proba):
        accuracy = accuracy_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        
        return accuracy, roc_auc

    def save_results(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        y_test_pred = model.predict(test_x)
        y_test_proba = model.predict_proba(test_x)[:, 1]

        accuracy, roc_auc = self.eval_metrics(test_y, y_test_pred, y_test_proba)
        
        # Saving metrics as local
        scores = {"accuracy": accuracy, "roc_auc": roc_auc}
        save_json(path=Path(self.config.metric_file_name), data=scores)