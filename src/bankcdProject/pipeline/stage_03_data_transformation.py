from src.bankcdProject.config.configuration import ConfigurationManager
from src.bankcdProject.components.data_transformation import DataTransformation
from src.bankcdProject import logger
from pathlib import Path

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status_content = f.read()

            if "Passed" in status_content:
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_spliting()
            else:
                raise Exception("Your data schema is not valid")

        except Exception as e:
            print(e)
