from src.bankcdProject.config.configuration import ConfigurationManager
from src.bankcdProject.components.data_validation import DataValidation
from src.bankcdProject import logger
import pandas as pd

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)

        # Load the preprocessed data
        try:
            df = pd.read_csv(data_validation_config.unzip_data_dir)
            logger.info("Preprocessed data loaded successfully for validation.")
            validation_status = data_validation.validate_all_columns()
            if validation_status:
                logger.info("Data Validation Stage: All preprocessing validations passed successfully.")
            else:
                logger.error("Data Validation Stage: Preprocessing validations failed. Refer to STATUS_FILE for details.")
        except FileNotFoundError:
            logger.error(f"Preprocessed data file not found at {data_validation_config.unzip_data_dir}")
        except Exception as e:
            logger.exception("An error occurred during data validation.")
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e
