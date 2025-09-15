"""
Machine learning model training and evaluation module.
"""
import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    mean_absolute_error, 
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
import joblib
import yaml
from pathlib import Path


class ModelTrainer:
    """Class for training and evaluating machine learning models."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize ModelTrainer with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.numerical_features = self.config['model']['features']['numerical']
        self.categorical_features = self.config['model']['features']['categorical']
        self.target_column = self.config['model']['target_column']
        self.test_size = self.config['model']['test_size']
        self.random_state = self.config['model']['random_state']
        
        self.model = None
        self.feature_columns = self.numerical_features + self.categorical_features
    
    def train_model(self, train_data, model_type='logistic_regression'):
        """Train machine learning model."""
        logging.info(f"\nTraining {model_type} model...")
        
        # Initialize model
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Prepare features and target
        X_train = train_data[self.feature_columns]
        y_train = train_data[self.target_column]
        
        # Train model
        self.model.fit(X_train, y_train)
        
        logging.info("Model training completed successfully")
        return self.model
    
    def make_predictions(self, data):
        """Make predictions on given data."""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        X = data[self.feature_columns]
        predictions = self.model.predict(X)
        
        return predictions
    
    def evaluate_model(self, test_data, predictions):
        """Evaluate model performance."""
        logging.info("\nEvaluating model performance...")
        
        y_true = test_data[self.target_column]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        mae = mean_absolute_error(y_true, predictions)
        
        # Print metrics
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1-Score: {f1:.4f}")
        logging.info(f"Mean Absolute Error: {mae:.4f}")
        
        # Generate classification report
        logging.info("\nClassification Report:")
        cr = classification_report(y_true, predictions)
        logging.info(cr)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mae': mae,
            'classification_report': cr
        }
    
    def train_and_evaluate(self, train_data, test_data):
        """Complete training and evaluation workflow."""
        logging.info("MACHINE LEARNING TRAINING AND EVALUATION")
       
        
        # Train model
        self.train_model(train_data)
        
        # Make predictions on training data
        train_predictions = self.make_predictions(train_data)
        train_data_with_pred = train_data.copy()
        train_data_with_pred['prediction'] = train_predictions
        
        # Make predictions on test data
        test_predictions = self.make_predictions(test_data)
        test_data_with_pred = test_data.copy()
        test_data_with_pred['prediction'] = test_predictions
        
        # Evaluate on training data
        logging.info("\nTraining Data Performance:")
        train_metrics = self.evaluate_model(train_data, train_predictions)
        
        # Evaluate on test data
        logging.info("\nTest Data Performance:")
        test_metrics = self.evaluate_model(test_data, test_predictions)
        
        return {
            'model': self.model,
            'train_data': train_data_with_pred,
            'test_data': test_data_with_pred,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
    
    def save_model(self, filepath="models/titanic_model.pkl"):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="models/titanic_model.pkl"):
        """Load trained model from file."""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def get_feature_importance(self):
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("No model available. Please train the model first.")
        
        if hasattr(self.model, 'coef_'):
            # For logistic regression
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            return feature_importance
        else:
            logging.info("Feature importance not available for this model type")
            return None
    
    def cross_validate_model(self, data, cv_folds=5):
        """Perform cross-validation on the model."""
        from sklearn.model_selection import cross_val_score
        
        if self.model is None:
            raise ValueError("No model available. Please train the model first.")
        
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='accuracy')
        
        logging.info(f"\nCross-Validation Results ({cv_folds} folds):")
        logging.info(f"Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        logging.info(f"Individual fold scores: {cv_scores}")
        
        return cv_scores
