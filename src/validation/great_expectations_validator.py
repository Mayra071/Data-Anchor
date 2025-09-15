"""
Great Expectations validation module for data quality assurance.
"""
import os
import sys
from src.exception import CustomException
from src.logger import logging

import great_expectations as gx
from great_expectations.core import ExpectationSuite
import pandas as pd
import json
from pathlib import Path


class GreatExpectationsValidator:
    """Class for data validation using Great Expectations."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize Great Expectations validator."""
        self.context = gx.get_context()
        self.suite_name = "titanic_validation_suite"
        self.suite = None
        self.validator = None
    
    def create_expectation_suite(self):
        """Create expectation suite for Titanic dataset."""
        logging.info("\nCreating Great Expectations validation suite...")
        
        # Create expectation suite
        self.suite = self.context.add_or_update_expectation_suite(
            expectation_suite_name=self.suite_name
        )
        
        logging.info(f"Expectation suite '{self.suite_name}' created successfully")
        return self.suite
    
    def setup_validator(self, df):
        """Setup validator with DataFrame."""
        # Connect DataFrame to GX
        validator = self.context.sources.pandas_default.read_dataframe(df)
        
        self.validator = self.context.get_validator(
            batch_request=validator.active_batch.batch_request,
            expectation_suite=self.suite
        )
        
        return self.validator
    
    def add_table_level_expectations(self):
        """Add table-level expectations."""
        logging.info("\nAdding table-level expectations...")
        
        # Table columns should match expected set
        self.validator.expect_table_columns_to_match_set(
            column_set=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 
                       'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
            result_format="BOOLEAN_ONLY"
        )
        
        # Table should have expected number of rows
        self.validator.expect_table_row_count_to_be_between(
            min_value=800, 
            max_value=1000
        )
    
    def add_column_existence_expectations(self):
        """Add column existence and type expectations."""
        logging.info("\nAdding column existence expectations...")
        
        required_columns = [
            "PassengerId", "Survived", "Name", "Sex", 
            "Age", "Ticket", "Pclass", "Fare"
        ]
        
        for column in required_columns:
            self.validator.expect_column_to_exist(column)
    
    def add_data_quality_expectations(self):
        """Add data quality expectations."""
        logging.info("\nAdding data quality expectations...")
        
        # PassengerId should be unique
        self.validator.expect_column_values_to_be_unique("PassengerId")
        
        # Survived should be 0 or 1
        self.validator.expect_column_values_to_be_in_set("Survived", [0, 1])
        
        # Pclass should be 1, 2, or 3
        self.validator.expect_column_values_to_be_in_set("Pclass", [1, 2, 3])
        
        # Sex should be male or female
        self.validator.expect_column_values_to_be_in_set("Sex", ["male", "female"])
        
        # Age should be between 0.3 and 80
        self.validator.expect_column_values_to_be_between("Age", min_value=0.3, max_value=80)
        
        # Fare should be positive
        self.validator.expect_column_values_to_be_between("Fare", min_value=0, max_value=600)
    
    def add_missing_value_expectations(self):
        """Add missing value expectations."""
        logging.info("\nAdding missing value expectations...")
        
        # Critical columns should not be null
        critical_columns = ["PassengerId", "Survived", "Pclass", "Name", "Sex"]
        for column in critical_columns:
            self.validator.expect_column_values_to_not_be_null(column)
        
        # Age can have some missing values (mostly=0.8 means 80% should not be null)
        self.validator.expect_column_values_to_not_be_null("Age", mostly=0.8)
    
    def run_validation(self):
        """Run validation and return results."""
        logging.info("\nRunning validation...")
        
        # Save the expectation suite
        self.validator.save_expectation_suite(discard_failed_expectations=False)
        
        # Run validation
        validation_result = self.validator.validate()
        
        logging.info(f"Validation successful: {validation_result['success']}")
        logging.info(f"Number of expectations: {len(validation_result['results'])}")
        
        # Count successful/failed expectations
        successful_expectations = sum(
            1 for result in validation_result['results'] if result['success']
        )
        logging.info(f"Successful expectations: {successful_expectations}/{len(validation_result['results'])}")
        
        return validation_result
    
    def save_validation_result(self, validation_result, filepath="reports/ge_validation_result.json"):
        """Save validation result to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(validation_result, f, indent=2, default=str)
        
        logging.info(f"Validation result saved to {filepath}")
    
    def validate_dataset(self, df):
        """Complete validation workflow for a dataset."""
      
        logging.info("GREAT EXPECTATIONS VALIDATION WORKFLOW")
        
        # Create expectation suite
        self.create_expectation_suite()
        
        # Setup validator
        self.setup_validator(df)
        
        # Add all expectations
        self.add_table_level_expectations()
        self.add_column_existence_expectations()
        self.add_data_quality_expectations()
        self.add_missing_value_expectations()
        
        # Run validation
        validation_result = self.run_validation()
        
        # Save results
        self.save_validation_result(validation_result)
        
        return validation_result
    
    def generate_validation_report(self, validation_result, filepath="reports/ge_validation_report.html"):
        """Generate HTML validation report."""
        try:
            # Create a more comprehensive validation suite for reporting
            gdf_demo = gx.from_pandas(df)
            
            # Add expectations for demo
            gdf_demo.expect_table_columns_to_match_set(column_set=list(df.columns))
            gdf_demo.expect_table_row_count_to_be_between(min_value=800, max_value=1000)
            gdf_demo.expect_column_values_to_be_unique("PassengerId")
            gdf_demo.expect_column_values_to_be_in_set("Survived", [0, 1])
            gdf_demo.expect_column_values_to_be_in_set("Pclass", [1, 2, 3])
            gdf_demo.expect_column_values_to_be_in_set("Sex", ["male", "female"])
            gdf_demo.expect_column_values_to_be_between("Age", min_value=0, max_value=100)
            gdf_demo.expect_column_values_to_be_between("Fare", min_value=0, max_value=600)
            gdf_demo.expect_column_values_to_not_be_null("PassengerId")
            gdf_demo.expect_column_values_to_not_be_null("Survived")
            gdf_demo.expect_column_values_to_not_be_null("Pclass")
            gdf_demo.expect_column_values_to_not_be_null("Name")
            gdf_demo.expect_column_values_to_not_be_null("Sex")
            
            # Run validation
            validation_result = gdf_demo.validate()
            
            logging.info(f"Validation completed: {validation_result['success']}")
            logging.info(f"Total expectations: {len(validation_result['results'])}")
            
            # Count successful/failed expectations
            successful = sum(1 for result in validation_result['results'] if result['success'])
            failed = len(validation_result['results']) - successful
            
            logging.info(f"Successful: {successful}")
            logging.info(f"Failed: {failed}")
            
            # Save validation result
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath.replace('.html', '.json'), "w") as f:
                json.dump(validation_result, f, indent=2, default=str)
            
            logging.info(f"Great Expectations validation result saved to '{filepath.replace('.html', '.json')}'")
            
        except Exception as e:
            logging.info(f"Error in Great Expectations validation: {e}")
            CustomException(e,sys)
