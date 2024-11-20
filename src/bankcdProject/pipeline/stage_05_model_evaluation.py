from src.bankcdProject.config.configuration import ConfigurationManager
from src.bankcdProject.components.model_evaluation import ModelEvaluation
from src.bankcdProject import logger

STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        for model_type in ["XGBoost", "GradientBoosting", "SVM"]:
            logger.info(f"Starting evaluation for model: {model_type}")
            model_evaluation_config = config.get_model_evaluation_config(model_type=model_type)
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.save_results()
            logger.info(f"Completed evaluation for model: {model_type}")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e