"""
Data preprocessing module for feature engineering and data cleaning.
"""
import sys
import os 
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path


class DataPreprocessor:
    """Class for data preprocessing and feature engineering."""

    def __init__(self, config_path="config.yaml"):
        """Initialize DataPreprocessor with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.numerical_features = self.config['model']['features']['numerical']
        self.categorical_features = self.config['model']['features']['categorical']
        self.target_column = self.config['model']['target_column']
        self.test_size = self.config['model']['test_size']
        self.random_state = self.config['model']['random_state']

    def create_age_groups(self, df, bins=[0, 12, 25, 55, 80]):
        """Create age groups from age column."""
        df_with_groups = df.copy()
        if 'Age' in df_with_groups.columns:
            df_with_groups['AgeGroup'] = pd.cut(df_with_groups['Age'], bins=bins, right=True)
            df_with_groups['AgeGroup'] = df_with_groups['AgeGroup'].astype(str)
        return df_with_groups

    def handle_missing_values(self, df, strategy='median'):
        """Handle missing values in the dataset."""
        df_cleaned = df.copy()

        if 'Age' in df_cleaned.columns:
            df_cleaned['Age'] = df_cleaned['Age'].fillna(df_cleaned['Age'].median())

        if 'Embarked' in df_cleaned.columns:
            try:
                mode_value = df_cleaned['Embarked'].mode()[0]
                df_cleaned['Embarked'] = df_cleaned['Embarked'].fillna(mode_value)
            except Exception:
                pass

        if 'Cabin' in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns='Cabin', axis=1)

        return df_cleaned

    def encode_categorical_variables(self, df):
        """Encode categorical variables to numerical."""
        df_encoded = df.copy()

        if 'Sex' in df_encoded.columns:
            df_encoded['Sex'] = df_encoded['Sex'].replace({'male': 1, 'female': 0})

        if 'Embarked' in df_encoded.columns:
            df_encoded['Embarked'] = df_encoded['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})

        if 'Pclass' in df_encoded.columns:
            # Ensure Pclass is treated as numeric category
            df_encoded['Pclass'] = pd.to_numeric(df_encoded['Pclass'], errors='coerce')

        return df_encoded

    def prepare_features(self, df):
        """Prepare features for machine learning."""
        df_processed = self.handle_missing_values(df)
        df_processed = self.encode_categorical_variables(df_processed)
        df_processed = self.create_age_groups(df_processed)
        return df_processed

    def _make_parquet_friendly(self, df):
        """Convert unsupported dtypes (e.g., Interval, categorical) to strings for Parquet."""
        df_out = df.copy()
        for col in df_out.columns:
            if pd.api.types.is_categorical_dtype(df_out[col]) or str(df_out[col].dtype).startswith('interval'):
                df_out[col] = df_out[col].astype(str)
        return df_out

    def split_data(self, df):
        """Split data into training and testing sets."""
        df_processed = self.prepare_features(df)

        feature_columns = [c for c in (self.numerical_features + self.categorical_features) if c in df_processed.columns]

        train_data, test_data = train_test_split(
            df_processed,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df_processed[self.target_column] if self.target_column in df_processed.columns else None
        )

        logging.info(f"Train data shape: {train_data.shape}")
        logging.info(f"Test data shape: {test_data.shape}")

        return train_data, test_data, feature_columns

    def create_reference_dataset(self, df, save_path="data/reference.parquet"):
        """Create reference dataset for drift detection."""
        df_processed = self.prepare_features(df)
        df_parquet = self._make_parquet_friendly(df_processed)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df_parquet.to_parquet(save_path, index=False)
        logging.info(f"Reference dataset saved to {save_path}")

        return df_processed

    def create_current_dataset(self, df, save_path="data/current.parquet"):
        """Create current dataset for drift detection using drop-rows strategy."""
        df_current = df.copy()
        if 'Cabin' in df_current.columns:
            df_current = df_current.drop(columns='Cabin', axis=1)
        # Drop any rows with missing values to simulate "current" clean slice
        df_current = df_current.dropna(axis=0)

        # Apply encoding and feature engineering consistent with training
        df_processed = self.encode_categorical_variables(df_current)
        df_processed = self.create_age_groups(df_processed)

        df_parquet = self._make_parquet_friendly(df_processed)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df_parquet.to_parquet(save_path, index=False)
        logging.info(f"Current dataset saved to {save_path}")
        logging.info("Data preprocessing completed")
        return df_processed
    