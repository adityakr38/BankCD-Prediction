from src.bankcdProject.config.configuration import ConfigurationManager
from src.bankcdProject.components.model_trainer import ModelTrainer
from src.bankcdProject import logger

STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        for model_type in ["XGBoost", "GradientBoosting", "SVM"]:
            try:
                logger.info(f"Starting training for model: {model_type}")
                model_trainer_config = config.get_model_trainer_config(model_type=model_type)
                model_trainer = ModelTrainer(config=model_trainer_config)
                model_trainer.train()
                logger.info(f"Completed training for model: {model_type}")
            except Exception as e:
                logger.exception(f"Error occurred during training of model: {model_type}. Details: {e}")
                raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e