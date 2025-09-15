"""
Data loading and preprocessing module for the Titanic dataset.
"""
import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Class for loading and basic preprocessing of the Titanic dataset."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize DataLoader with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.raw_data_path = self.config['data']['raw_data_path']
        self.processed_data_path = Path(self.config['data']['processed_data_path'])
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    # Load data set
    def load_raw_data(self):
        """Load raw Titanic dataset."""
        try:
            df = pd.read_csv(self.raw_data_path)
            logging.info(f"Dataset loaded successfully: {df.shape}")
            return df
        except Exception as e:
            logging.info(f"Error: Could not find dataset at {self.raw_data_path}")
            CustomException(e,sys)
            return None
    
    # get data set overview
    def get_data_overview(self, df):
        """Get comprehensive dataset overview."""
        logging.info("\n1. DATASET OVERVIEW")
        logging.info(f"Dataset shape: {df.shape}")
        logging.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logging.info(f"Data types:\n{df.dtypes}")
        
        return {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes
        }
    
    
    # analyz data quality
    def analyze_data_quality(self, df):
        """Analyze data quality and missing values."""
        logging.info("\n=== DATA QUALITY ASSESSMENT ===")
        
        # Total missing values
        missing_count = df.isnull().sum().sum()
        total_values = df.shape[0] * df.shape[1]
        missing_percentage = (missing_count / total_values) * 100
        
        logging.info(f"Total missing values in the dataset: {missing_count}")
        logging.info(f"Total values in the dataset: {total_values}")
        logging.info(f"Percentage of missing values in the dataset: {missing_percentage:.2f}%")
        
        # Missing values per column
        logging.info("\nMissing values per column:")
        missing_data = df.isnull().sum()
        missing_percentage_col = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percentage_col
        })
        logging.info(missing_df[missing_df['Missing Count'] > 0])
        
        # Data quality score
        quality_score = ((total_values - missing_count) / total_values) * 100
        logging.info(f"\nOverall data quality score: {quality_score:.2f}%")
        
        return {
            'missing_count': missing_count,
            'total_values': total_values,
            'missing_percentage': missing_percentage,
            'quality_score': quality_score,
            'missing_by_column': missing_df
        }
    
    
    # Feature columns data tyes
    def get_feature_types(self, df):
        """Identify numerical and categorical features."""
        numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
        categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
        
        logging.info('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
        logging.info('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))
        
        return {
            'numerical': numeric_features,
            'categorical': categorical_features
        }
    
    
    # Analyze descriptive statistics
    def get_descriptive_statistics(self, df):
        """Get descriptive statistics for numerical features."""
        logging.info("\n NUMERICAL FEATURES DESCRIPTIVE STATISTICS")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('PassengerId', errors='ignore')
        desc_stats = df[numeric_cols].describe().T
        return desc_stats
    
    
    # Analyze categorical and target features
    def analyze_categorical_features(self, df):
        """Analyze categorical features."""
        logging.info("\n CATEGORICAL FEATURES ANALYSIS")
        categorical_cols = df.select_dtypes(include=['object']).columns
        analysis = {}
        
        for col in categorical_cols:
            logging.info(f"\n{col}:")
            unique_count = df[col].nunique()
            logging.info(f"  Unique values: {unique_count}")
            value_counts = df[col].value_counts().head(3)
            logging.info(value_counts)
            
            analysis[col] = {
                'unique_count': unique_count,
                'top_values': value_counts.to_dict()
            }
        
        return analysis
    
    def analyze_target_variable(self, df, target_col='Survived'):
        """Analyze target variable distribution."""
        logging.info(f"\n TARGET VARIABLE ANALYSIS ({target_col})")
        survival_stats = {
            'Total_Passengers': len(df),
            'Survived': df[target_col].sum(),
            'Not_Survived': (df[target_col] == 0).sum(),
            'Survival_Rate': (df[target_col].sum() / len(df)) * 100
        }
        
        for key, value in survival_stats.items():
            logging.info(f"{key}: {value}")
        
        return survival_stats
