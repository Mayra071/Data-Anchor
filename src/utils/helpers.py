"""
Utility helper functions for the Data Anchor project.
"""
import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import json
from datetime import datetime


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def save_json(data, filepath):
    """Save data to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_directory_structure(base_path="."):
    """Create the complete directory structure for the project."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/reference",
        "data/current",
        "src/data_processing",
        "src/validation",
        "src/drift_detection",
        "src/ml",
        "src/visualization",
        "src/utils",
        "reports",
        "notebooks",
        "tests",
        "docs",
        "models"
    ]
    
    for directory in directories:
        Path(base_path, directory).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created successfully")


def get_data_summary(df):
    """Get comprehensive data summary."""
    summary = {
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    return summary


def print_section_header(title, char="=", length=60):
    """Print a formatted section header."""
    print(f"\n{char * length}")
    print(title)
    print(f"{char * length}")


def format_percentage(value, total):
    """Format value as percentage."""
    return f"{(value / total) * 100:.2f}%"


def get_timestamp():
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_dataframe(df, required_columns=None):
    """Validate DataFrame structure and content."""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("DataFrame is empty")
        return validation_results
    
    # Check required columns
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for completely empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        validation_results['warnings'].append(f"Completely empty columns: {empty_columns}")
    
    return validation_results


def clean_column_names(df):
    """Clean column names by removing special characters and converting to lowercase."""
    df_cleaned = df.copy()
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '')
    return df_cleaned


def detect_outliers_iqr(df, column, multiplier=1.5):
    """Detect outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers


def calculate_data_quality_score(df):
    """Calculate overall data quality score."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    quality_score = ((total_cells - missing_cells) / total_cells) * 100
    return quality_score


def create_summary_report(data_summary, validation_results, model_metrics=None):
    """Create a comprehensive summary report."""
    report = {
        'timestamp': get_timestamp(),
        'data_summary': data_summary,
        'validation_results': validation_results,
        'model_metrics': model_metrics or {},
        'overall_status': 'SUCCESS' if validation_results.get('is_valid', False) else 'FAILED'
    }
    return report
