import os
import pandas as pd
from src.bankcdProject.utils.common import save_json
from src.bankcdProject.entity.config_entity import ModelEvaluationConfig
from sklearn.model_selection import cross_val_score
from src.bankcdProject import logger
import joblib
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, y_test, y_test_pred, y_test_proba=None):
        """Evaluate multiple metrics."""
        metrics = {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred, zero_division=0),
            "recall": recall_score(y_test, y_test_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
        }

        if y_test_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_test_proba)
        return metrics

    def save_results(self):

        # Load test data
        test_data = pd.read_csv(self.config.test_data_path)
        logger.info(f"Loaded test data from {self.config.test_data_path}")
        model = joblib.load(self.config.model_path)
        logger.info(f"Model loaded from {self.config.model_path}")

        # Prepare test data
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        # Make predictions
        y_test_pred = model.predict(test_x)
        y_test_proba = model.predict_proba(test_x)[:, 1] if hasattr(model, "predict_proba") else None

        # Evaluate metrics
        metrics = self.eval_metrics(test_y, y_test_pred, y_test_proba)

        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(
            model, test_x, test_y, cv=5, scoring='roc_auc', n_jobs=-1
        )
        metrics["ROC-AUC_cross_validation_scores"] = cv_scores.tolist()
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()

        # Save metrics to a JSON file with the model name
        save_path = Path(self.config.root_dir) / f"{self.config.model_name}_metrics.json"
        save_json(path=save_path, data=metrics)
        logger.info(f"Metrics for {self.config.model_name} saved to {save_path}")

