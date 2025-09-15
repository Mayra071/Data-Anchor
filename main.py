"""
Main execution script for the Data Anchor project.
This script demonstrates the complete data validation and monitoring workflow.
"""
import os
import sys
from src.exception import CustomException
from src.logger import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_preprocessor import DataPreprocessor
from src.validation.great_expectations_validator import GreatExpectationsValidator
from src.drift_detection.evidently_drift_detector import EvidentlyDriftDetector
from src.ml.model_trainer import ModelTrainer
from src.visualization.data_visualizer import DataVisualizer
from src.utils.helpers import print_section_header, save_json, get_timestamp


def main():
    """Main execution function."""
    logging.info("DATA ANCHOR - COMPREHENSIVE DATA VALIDATION & MONITORING")
    logging.info(f"Execution started at: {get_timestamp()}")
    
    try:
        #  Load and analyze data
        logging.info(" DATA LOADING AND ANALYSIS")
        data_loader = DataLoader()
        df = data_loader.load_raw_data()
        
        if df is None:
            logging.info("Failed to load data. Exiting...")
            return
        
        # Get data overview
        data_overview = data_loader.get_data_overview(df)
        data_quality = data_loader.analyze_data_quality(df)
        feature_types = data_loader.get_feature_types(df)
        desc_stats = data_loader.get_descriptive_statistics(df)
        categorical_analysis = data_loader.analyze_categorical_features(df)
        target_analysis = data_loader.analyze_target_variable(df)
        #  Data visualization
        logging.info(" DATA VISUALIZATION")
        visualizer = DataVisualizer()
        visualizer.create_comprehensive_visualization(df, save_plots=True)
        
        # Create correlation heatmap
        visualizer.create_correlation_heatmap(df, "reports/correlation_heatmap.png")
        
        
            
        #  Data preprocessing
        logging.info(" DATA PREPROCESSING")
        preprocessor = DataPreprocessor()
        train_data, test_data, feature_columns = preprocessor.split_data(df)
        
        # Create reference and current datasets for drift detection
        reference_data = preprocessor.create_reference_dataset(df)
        current_data = preprocessor.create_current_dataset(df)
        
        #  Data validation with Great Expectations
        logging.info(" DATA VALIDATION WITH GREAT EXPECTATIONS")
        ge_validator = GreatExpectationsValidator()
        validation_result = ge_validator.validate_dataset(df)
        
        #  Machine learning model training
        logging.info(" MACHINE LEARNING MODEL TRAINING")
        model = ModelTrainer()
        ml_results1 = model.train_and_evaluate(train_data, test_data)
        ml_results2 = model.train_and_evaluate(reference_data, current_data)
        
        # Create feature importance plot if available
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            visualizer.create_feature_importance_plot(
                feature_importance, 
                "reports/feature_importance.png"
            )
        
        # Add predictions to datasets for drift detection
        train_data_with_pred = ml_results1['train_data']
        test_data_with_pred = ml_results1['test_data']
        
        ref_data_with_pred = ml_results2['train_data']
        cur_data_with_pred = ml_results2['test_data']
        
        # Drift detection with Evidently
        logging.info(" DRIFT DETECTION WITH EVIDENTLY")
        drift_detector = EvidentlyDriftDetector()
        
        # Run drift detection between train and test data
        drift_snapshot, drift_results = drift_detector.run_drift_detection(
            train_data_with_pred,
            test_data_with_pred,
            "drift_reportv1.html"
        )
        
        # Run drift detection between reference and current data
        drift_snapshot2, drift_results2 = drift_detector.run_drift_detection(
            ref_data_with_pred, 
            cur_data_with_pred, 
            "drift_reportv2.html"
        )
        
        # Compare the two drift reports
        comparison_df = drift_detector.compare_reports(
            drift_snapshot, 
            drift_snapshot2, 
            "drift_comparison_report.html"
        )
        
        
        
        #  Generate comprehensive summary
        logging.info(" GENERATING COMPREHENSIVE SUMMARY")
        
        summary_data = {
            "execution_timestamp": get_timestamp(),
            "data_summary": {
                "dataset_shape": df.shape,
                "data_quality_score": data_quality['quality_score'],
                "missing_values_count": data_quality['missing_count'],
                "numerical_features": len(feature_types['numerical']),
                "categorical_features": len(feature_types['categorical'])
            },
            "validation_results": {
                "great_expectations_success": validation_result['success'],
                "total_expectations": len(validation_result['results']),
                "successful_expectations": sum(1 for r in validation_result['results'] if r['success'])
            },
            "model_performance1": {
                "train_accuracy": ml_results1['train_metrics']['accuracy'],
                "test_accuracy": ml_results1['test_metrics']['accuracy'],
                "train_f1_score": ml_results1['train_metrics']['f1_score'],
                "test_f1_score": ml_results1['test_metrics']['f1_score']
            },
            "model_performance2": {
                "train_accuracy": ml_results2['train_metrics']['accuracy'],
                "test_accuracy": ml_results2['test_metrics']['accuracy'],
                "train_f1_score": ml_results2['train_metrics']['f1_score'],
                "test_f1_score": ml_results2['test_metrics']['f1_score']
            },
            "drift_detection": {
                "train_test_drift_detected": drift_results.get('drift_detected', False),
                "reference_current_drift_detected": drift_results2.get('drift_detected', False)
            },
            "generated_files": [
                "reports/drift_reportv1.html",
                "reports/drift_reportv2.html", 
                "reports/drift_comparison_report.html",
                "reports/ge_validation_result.json",
                "data/reference.parquet",
                "data/current.parquet"
            ]
        }
        
        # Save comprehensive summary
        save_json(summary_data, "reports/comprehensive_summary.json")
        
        # Print final summary
        logging.info("EXECUTION COMPLETED SUCCESSFULLY")
        logging.info(f"Data Quality Score: {data_quality['quality_score']:.2f}%")
        logging.info(f"Model Test Accuracy: {ml_results1['test_metrics']['accuracy']:.4f}")
        logging.info(f"Great Expectations Validation: {'PASSED' if validation_result['success'] else 'FAILED'}")
        logging.info(f"Drift Detection Reports: Generated")
        logging.info(f"Visualizations: Created and saved to reports/")
        logging.info(f"Summary Report: reports/comprehensive_summary.json")
        
        logging.info("FILES GENERATED")
        for file in summary_data["generated_files"]:
            logging.info(f"✓ {file}")
        
        logging.info("NEXT STEPS")
        logging.info("1. Open HTML reports in browser for detailed analysis")
        logging.info("2. Review validation results in JSON files")
        logging.info("3. Use generated datasets for further analysis")
        # logging.info("4. Integrate monitoring into your MLOps pipeline")
        
    except Exception as e:
        logging.info(f"\nError during execution: {str(e)}")
        logging.info("Please check the error and try again.")
        CustomException(e,sys)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
