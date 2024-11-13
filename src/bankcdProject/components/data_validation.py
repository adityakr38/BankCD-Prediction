import os
import pandas as pd
from src.bankcdProject import logger
from src.bankcdProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_columns(self, df: pd.DataFrame) -> bool:
        expected_columns = [
            'default', 'balance', 'housing', 'loan', 'contact', 'day', 'duration',
            'campaign', 'pdays', 'previous', 'poutcome', 'cd', 'age_bin_Medium',
            'age_bin_High', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
            'job_management', 'job_retired', 'job_self-employed', 'job_services',
            'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
            'marital_married', 'marital_single', 'education_Secondary',
            'education_Tertiary', 'education_Unknown', 'generation_millenials',
            'generation_older boomers', 'generation_silent generation',
            'generation_younger boomers', 'month_aug', 'month_dec', 'month_feb',
            'month_jan', 'month_jul', 'month_jun', 'month_mar', 'month_may',
            'month_nov', 'month_oct', 'month_sep'
        ]

        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return False
        logger.info("Column validation passed.")
        return True

    def validate_all(self, df: pd.DataFrame) -> bool:
        validation_status = True
        status_messages = []

        # Column validation
        if self.validate_columns(df):
            status_messages.append("Column validation passed.")
        else:
            validation_status = False
            status_messages.append("Column validation failed: Missing or extra columns detected.")

        # Binary columns validation
        if self.validate_binary_columns(df):
            status_messages.append("Binary columns validation passed.")
        else:
            validation_status = False
            status_messages.append("Binary columns validation failed.")

        # Dropped columns validation
        if self.validate_dropped_columns(df):
            status_messages.append("Dropped columns validation passed.")
        else:
            validation_status = False
            status_messages.append("Dropped columns validation failed: Unexpected columns found.")

        # Writing validation results to STATUS_FILE
        with open(self.config.STATUS_FILE, 'w') as f:
            f.write(f"Validation status: {'Passed' if validation_status else 'Failed'}\n")
            for message in status_messages:
                f.write(f"{message}\n")

        # Log final status
        if validation_status:
            logger.info("All validations passed.")
        else:
            logger.error("Some validations failed. Check STATUS_FILE for details.")

        return validation_status

    def validate_binary_columns(self, df: pd.DataFrame) -> bool:
        binary_cols = self.config.preprocessing.binary_conversion.get("columns")
        if not binary_cols:
            logger.error("Binary columns not specified in preprocessing configuration.")
            with open(self.config.STATUS_FILE, 'a') as f:
                f.write("Binary columns validation failed: 'binary_conversion.columns' not specified in configuration.\n")
            return False
n
        for col in binary_cols:
            if col not in df.columns:
                logger.error(f"Binary column {col} is missing from the DataFrame.")
                return False
            if df[col].nunique() != 2:
                logger.error(f"Binary column {col} does not contain exactly two unique values.")
                return False

        logger.info("Binary columns validation passed.")
        return True



    def validate_dropped_columns(self, df: pd.DataFrame) -> bool:
        dropped_columns = self.config.preprocessing.drop_columns
        unexpected_columns = set(dropped_columns).intersection(df.columns)
        if unexpected_columns:
            logger.error(f"Dropped columns still found in DataFrame: {unexpected_columns}")
            return False
        return True
