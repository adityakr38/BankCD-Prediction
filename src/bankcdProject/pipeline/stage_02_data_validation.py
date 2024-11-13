from src.bankcdProject.config.configuration import ConfigurationManager
from src.bankcdProject.components.data_validation import DataValidation
from src.bankcdProject import logger
import pandas as pd
from pathlib import Path

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)

        processed_file_path = Path("artifacts/data_ingestion/processed_train.csv")
        df = pd.read_csv(processed_file_path)
        
        if data_validation.validate_all(df):
            logger.info("Data Validation Stage: All validations passed.")
        else:
            logger.error("Data Validation Stage: Validations failed. Check logs for details.")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
