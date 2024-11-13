import os
import pandas as pd
from src.bankcdProject import logger
from src.bankcdProject.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            # Load data and get column lists
            df = pd.read_csv(self.config.unzip_data_dir)
            all_cols = set(df.columns)
            all_schema = set(self.config.all_schema.keys())

            # Check if any required column is missing
            missing_cols = all_schema - all_cols
            if missing_cols:
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}\n")
                    f.write(f"Missing columns: {', '.join(missing_cols)}\n")
                logger.info(f"Missing columns: {missing_cols}")
                return validation_status
            else:
                logger.info("All required columns are present.")
            
            # Proceed with further validations if all columns are present
            return self.validate_preprocessing_steps(df)
        
        except Exception as e:
            logger.exception("Error during validation")
            raise e

    def validate_text_standardization(self, df: pd.DataFrame) -> bool:
        # Check if 'job' and 'generation' are lowercase
        if not df['job'].str.islower().all():
            logger.error("Text standardization failed: 'job' is not fully lowercase.")
            return False
        if not df['generation'].str.islower().all():
            logger.error("Text standardization failed: 'generation' is not fully lowercase.")
            return False
        
        # Check 'education' has expected standardized values
        valid_education_values = {"Tertiary", "Primary", "Secondary", "Unknown"}
        if not df['education'].isin(valid_education_values).all():
            logger.error("Text standardization failed: 'education' contains unexpected values.")
            return False
        
        return True

    def validate_binary_conversion(self, df: pd.DataFrame) -> bool:
        # Check binary columns for correct values
        binary_cols = ['default', 'housing', 'loan']
        for col in binary_cols:
            if not df[col].isin([0, 1]).all():
                logger.error(f"Binary conversion failed: Column {col} contains non-binary values.")
                return False
        return True

    def validate_column_dropping(self, df: pd.DataFrame) -> bool:
        # Verify that specified columns were dropped
        dropped_columns = ["id", "Unnamed: 21", "Unnamed: 22", "state", "zipcode"]
        if any(col in df.columns for col in dropped_columns):
            logger.error(f"Column dropping failed: Found columns that should have been dropped: {dropped_columns}")
            return False
        return True

    def validate_missing_value_imputation(self, df: pd.DataFrame) -> bool:
        # Verify missing values have been imputed
        imputed_columns = ['job', 'marital', 'generation', 'previous', 'campaign']
        missing_after_imputation = df[imputed_columns].isnull().sum()
        if missing_after_imputation.any():
            logger.error("Imputation failed: Some columns still contain missing values after imputation.")
            return False
        return True

    def validate_outlier_handling(self, df: pd.DataFrame) -> bool:
        # Check that 'previous' column doesn't contain extreme values beyond IQR bounds
        Q1 = df['previous'].quantile(0.25)
        Q3 = df['previous'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df['previous'] < (Q1 - 1.5 * IQR)) | (df['previous'] > (Q3 + 1.5 * IQR))]
        if not outliers.empty:
            logger.error("Outlier handling failed: 'previous' contains extreme values beyond IQR bounds.")
            return False
        return True

    def validate_age_binning(self, df: pd.DataFrame) -> bool:
        # Check that 'age' column is dropped and 'age_bin' columns are present
        if 'age' in df.columns:
            logger.error("Age binning failed: 'age' column was not dropped after binning.")
            return False
        age_bin_columns = ['age_bin_Medium', 'age_bin_High']
        if not all(col in df.columns for col in age_bin_columns):
            logger.error("Age binning failed: One-hot encoded columns for 'age_bin' are missing.")
            return False
        return True

    def validate_preprocessing_steps(self, df: pd.DataFrame) -> bool:
        # Run each validation check
        checks = [
            self.validate_text_standardization(df),
            self.validate_binary_conversion(df),
            self.validate_column_dropping(df),
            self.validate_missing_value_imputation(df),
            self.validate_outlier_handling(df),
            self.validate_age_binning(df)
        ]

        # Check if all validations passed
        validation_status = all(checks)
        with open(self.config.STATUS_FILE, 'w') as f:
            f.write(f"Validation status: {validation_status}\n")
            if not validation_status:
                f.write("Some preprocessing validations failed. See logs for details.\n")
            else:
                f.write("All preprocessing validations passed successfully.\n")
        
        if validation_status:
            logger.info("All preprocessing validations passed successfully.")
        else:
            logger.error("Some preprocessing validations failed.")
        
        return validation_status
