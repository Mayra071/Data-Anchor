"""
Evidently AI drift detection module for data and model drift monitoring.
"""
import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from evidently import DataDefinition, Dataset, Report, compare
from evidently.metrics import *
from evidently.presets import *
from evidently import BinaryClassification
import yaml
from pathlib import Path


class EvidentlyDriftDetector:
    """Class for drift detection using Evidently AI."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize Evidently drift detector."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.numerical_columns = self.config['model']['features']['numerical']
        self.categorical_columns = self.config['model']['features']['categorical']
        self.target_column = self.config['model']['target_column']
        self.reports_path = Path(self.config['data']['reports_path'])
        self.reports_path.mkdir(parents=True, exist_ok=True)
    
    def create_data_definition(self, reference_data, prediction_column='prediction'):
        """Create data definition for Evidently."""
        categorical_columns = self.categorical_columns + [self.target_column]
        classification = None

        if prediction_column in reference_data.columns:
            categorical_columns.append(prediction_column)
            classification = [BinaryClassification(
                target=self.target_column,
                prediction_labels=prediction_column
            )]

        data_definition = DataDefinition(
            numerical_columns=self.numerical_columns,
            categorical_columns=categorical_columns,
            classification=classification
        )
        return data_definition
    
    def create_datasets(self, reference_data, current_data, data_definition):
        """Create reference and current datasets for drift detection."""
        # Reference dataset
        ref_dataset = Dataset.from_pandas(
            reference_data,
            data_definition
        )
        
        # Current dataset
        cur_dataset = Dataset.from_pandas(
            current_data,
            data_definition
        )
        
        return ref_dataset, cur_dataset
    
    def create_drift_report(self, data_definition, include_tests=True):
        """Create comprehensive drift detection report."""
        logging.info("\nCreating drift detection report...")

        metrics = [
            DataDriftPreset(),
            DataSummaryPreset(),
            DriftedColumnsCount(),
        ]

        # Add prediction-related metrics only if prediction column is present
        if 'prediction' in data_definition.categorical_columns:
            metrics.extend([
                ValueDrift(column='prediction'),
                MissingValueCount(column='prediction'),
                ClassificationPreset(),
            ])

        report = Report(metrics, include_tests=include_tests)

        return report
    
    def run_drift_detection(self, reference_data, current_data, report_name="drift_report.html"):
        """Run complete drift detection workflow."""
        
        logging.info("EVIDENTLY DRIFT DETECTION WORKFLOW")
        
        # Create data definition
        data_definition = self.create_data_definition(reference_data)
        
        # Create datasets
        ref_dataset, cur_dataset = self.create_datasets(
            reference_data, current_data, data_definition
        )
        
        # Create drift report
        report = self.create_drift_report(data_definition)
        
        # Run drift detection
        logging.info("\nRunning drift detection...")
        snapshot = report.run(current_data=cur_dataset, reference_data=ref_dataset)
        
        # Save report
        report_path = self.reports_path / report_name
        snapshot.save_html(str(report_path))
        logging.info(f"Drift report saved as {report_path}")
        
        # Save report dictionary
        report_dict = snapshot.dict()
        dict_path = self.reports_path / report_name.replace('.html', '.json')
        with open(dict_path, 'w') as f:
            import json
            json.dump(report_dict, f, indent=2, default=str)
        
        return snapshot, report_dict
    
    def compare_reports(self, snapshot1, snapshot2, comparison_name="compare_report.html"):
        """Compare two drift detection reports."""
        logging.info("\nComparing drift detection reports...")
        
        # Compare reports
        compare_dataframe = compare(snapshot1, snapshot2)
        
        # Create generic version names
        new_names = {
            old: f"Version {i+1}" 
            for i, old in enumerate(compare_dataframe.columns[0:])
        }
        
        # Rename columns
        compare_dataframe.rename(columns=new_names, inplace=True)
        
        # Save comparison report
        comparison_path = self.reports_path / comparison_name
        compare_dataframe.to_html(str(comparison_path))
        logging.info(f"Comparison report saved as {comparison_path}")
        
        return compare_dataframe
    
    # def detect_data_drift(self, reference_data, current_data):
    #     """Detect data drift between reference and current datasets."""
    #     logging.info("\nDetecting data drift...")

    #     # Create data definition
    #     data_definition = self.create_data_definition(reference_data)

    #     # Create datasets
    #     ref_dataset, cur_dataset = self.create_datasets(
    #         reference_data, current_data, data_definition
    #     )

    #     # Create data drift report
    #     drift_report = Report([
    #         DataDriftPreset(),
    #         DriftedColumnsCount(),
    #     ])

    #     # Run drift detection
    #     drift_snapshot = drift_report.run(
    #         current_data=cur_dataset,
    #         reference_data=ref_dataset
    #     )

    #     # Extract drift information
    #     drift_results = drift_snapshot.dict()

    #     # Check if drift was detected
    #     drift_detected = False
    #     drift_share = 0.0

    #     for metric in drift_results['metrics']:
    #         if metric['metric'] == 'DatasetDriftMetric':
    #             drift_detected = metric['result']['dataset_drift']
    #             drift_share = metric['result']['drift_share']
    #             break

    #     logging.info(f"Data drift detected: {drift_detected}")
    #     logging.info(f"Drift share: {drift_share:.4f}")

    #     return {
    #         'drift_detected': drift_detected,
    #         'drift_share': drift_share,
    #         'snapshot': drift_snapshot
    #     }
    
    # def detect_target_drift(self, reference_data, current_data):
    #     """Detect target drift between reference and current datasets."""
    #     print("\nDetecting target drift...")

    #     # Create data definition
    #     data_definition = self.create_data_definition(reference_data)

    #     # Create datasets
    #     ref_dataset, cur_dataset = self.create_datasets(
    #         reference_data, current_data, data_definition
    #     )

    #     # Create target drift report
    #     target_report = Report([
    #         TargetDriftPreset(),
    #     ])

    #     # Run target drift detection
    #     target_snapshot = target_report.run(
    #         current_data=cur_dataset,
    #         reference_data=ref_dataset
    #     )

    #     # Extract target drift information
    #     target_results = target_snapshot.dict()

    #     # Check if target drift was detected
    #     target_drift_detected = False
    #     target_drift_score = 0.0

    #     for metric in target_results['metrics']:
    #         if 'TargetDriftMetric' in metric['metric']:
    #             target_drift_detected = metric['result']['drift_detected']
    #             target_drift_score = metric['result']['drift_score']
    #             break

    #     print(f"Target drift detected: {target_drift_detected}")
    #     print(f"Target drift score: {target_drift_score:.4f}")

    #     return {
    #         'target_drift_detected': target_drift_detected,
    #         'target_drift_score': target_drift_score,
    #         'snapshot': target_snapshot
    #     }
    
    # def generate_comprehensive_report(self, reference_data, current_data, report_name="comprehensive_drift_report.html"):
    #     """Generate comprehensive drift detection report."""
    #     print("\nGenerating comprehensive drift detection report...")

    #     # Create data definition
    #     data_definition = self.create_data_definition(reference_data)

    #     # Create datasets
    #     ref_dataset, cur_dataset = self.create_datasets(
    #         reference_data, current_data, data_definition
    #     )

    #     # Create comprehensive report
    #     metrics = [
    #         DataDriftPreset(),
    #         TargetDriftPreset(),
    #         DataSummaryPreset(),
    #     ]

    #     # Add ClassificationPreset only if prediction column is present
    #     if 'prediction' in data_definition.categorical_columns:
    #         metrics.append(ClassificationPreset())

    #     comprehensive_report = Report(metrics)

    #     # Run comprehensive analysis
    #     comprehensive_snapshot = comprehensive_report.run(
    #         current_data=cur_dataset,
    #         reference_data=ref_dataset
    #     )

    #     # Save comprehensive report
    #     comprehensive_path = self.reports_path / report_name
    #     comprehensive_snapshot.save_html(str(comprehensive_path))
    #     print(f"Comprehensive drift report saved as {comprehensive_path}")

    #     return comprehensive_snapshot
