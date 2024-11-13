import urllib.request as request
import zipfile
from src.bankcdProject import logger
from src.bankcdProject.utils.common import get_size
from src.bankcdProject.entity.config_entity import DataIngestionConfig
from pathlib import Path
import pandas as pd
import os

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        preprocessing = self.config.preprocessing

        # Step 1: Standardize text values and correct entries

        df['job'] = df['job'].str.lower()
        df['generation'] = df['generation'].str.lower()

        # Replace specific values in 'generation'
        df['generation'] = df['generation'].replace({
            'oler boomers': 'older boomers',
            'millennials': 'millenials'
        })

        # Correct specific values in 'education'
        df['education'] = df['education'].replace({
            'tertiary': 'Tertiary',
            'primary': 'Primary',
            'primery': 'Primary',    
            'secondary': 'Secondary',
            'secendary': 'Secondary', 
            'unknown': 'Unknown'
        })

        # Convert binary columns to numeric values
        binary_cols = ['default', 'housing', 'loan']
        df[binary_cols] = df[binary_cols].replace({'yes': 1, 'no': 0})

        # Drop specified columns
        df.drop(columns=preprocessing['drop_columns'], inplace=True)

        # Impute missing values
        for col in preprocessing['imputation']['mode_cols']:
            df[col] = df[col].fillna(df[col].mode()[0])
        for col in preprocessing['imputation']['median_cols']:
            df[col] = df[col].fillna(df[col].median())
        for col in preprocessing['imputation']['mean_cols']:
            df[col] = df[col].fillna(df[col].mean())

        # Handling outliers in 'previous' using IQR
        Q1 = df['previous'].quantile(0.25)
        Q3 = df['previous'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['previous'] < (Q1 - 1.5 * IQR)) | (df['previous'] > (Q3 + 1.5 * IQR)))]

        # Binning 'age' and encoding bins
        df['age_bin'] = pd.cut(df['age'], bins=preprocessing['binning']['age']['bins'], labels=preprocessing['binning']['age']['labels'])
        df = pd.get_dummies(df, columns=['age_bin'], drop_first=True)

        # Frequency Mean Encoding for specified columns
        for col in preprocessing['encoding']['frequency_mean']:
            freq_mean = df.groupby(col)['cd'].mean()  # Assumes 'cd' is the target
            df[col] = df[col].map(freq_mean)

        # One-Hot Encoding for other categorical columns
        df = pd.get_dummies(df, columns=preprocessing['encoding']['one_hot'], drop_first=True)

        #Dropping the 'age' column after binning
        df.drop(columns=['age'], inplace=True)

        return df

    def run(self):
        self.download_file()
        self.extract_zip_file()
        df = pd.read_csv(Path(self.config.unzip_dir) / "train.csv", low_memory=False)  # Load the extracted data
        processed_df = self.preprocess_data(df)  # Preprocess the data
        processed_df.to_csv(Path(self.config.root_dir) / "processed_train.csv", index=False)
        return processed_df