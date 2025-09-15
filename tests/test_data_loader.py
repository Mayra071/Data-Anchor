"""
Unit tests for data_loader module.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_processing.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Survived': [0, 1, 1, 0, 1],
            'Pclass': [3, 1, 3, 1, 3],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 
                    'Heikkinen, Miss. Laina', 'Futrelle, Mrs. Jacques Heath', 
                    'Allen, Mr. William Henry'],
            'Sex': ['male', 'female', 'female', 'female', 'male'],
            'Age': [22.0, 38.0, 26.0, 35.0, 35.0],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282', '113803', '373450'],
            'Fare': [7.2500, 71.2833, 7.9250, 53.1000, 8.0500],
            'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan],
            'Embarked': ['S', 'C', 'S', 'S', 'S']
        })
    
    def test_get_data_overview(self):
        """Test data overview functionality."""
        overview = self.loader.get_data_overview(self.sample_data)
        
        self.assertIn('shape', overview)
        self.assertIn('memory_usage', overview)
        self.assertIn('dtypes', overview)
        self.assertEqual(overview['shape'], (5, 12))
    
    def test_analyze_data_quality(self):
        """Test data quality analysis."""
        quality = self.loader.analyze_data_quality(self.sample_data)
        
        self.assertIn('missing_count', quality)
        self.assertIn('total_values', quality)
        self.assertIn('missing_percentage', quality)
        self.assertIn('quality_score', quality)
        self.assertIn('missing_by_column', quality)
    
    def test_get_feature_types(self):
        """Test feature type identification."""
        feature_types = self.loader.get_feature_types(self.sample_data)
        
        self.assertIn('numerical', feature_types)
        self.assertIn('categorical', feature_types)
        self.assertIsInstance(feature_types['numerical'], list)
        self.assertIsInstance(feature_types['categorical'], list)
    
    def test_get_descriptive_statistics(self):
        """Test descriptive statistics generation."""
        desc_stats = self.loader.get_descriptive_statistics(self.sample_data)
        
        self.assertIsInstance(desc_stats, pd.DataFrame)
        self.assertGreater(len(desc_stats), 0)
    
    def test_analyze_categorical_features(self):
        """Test categorical feature analysis."""
        analysis = self.loader.analyze_categorical_features(self.sample_data)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('Name', analysis)
        self.assertIn('Sex', analysis)
    
    def test_analyze_target_variable(self):
        """Test target variable analysis."""
        target_analysis = self.loader.analyze_target_variable(self.sample_data)
        
        self.assertIn('Total_Passengers', target_analysis)
        self.assertIn('Survived', target_analysis)
        self.assertIn('Not_Survived', target_analysis)
        self.assertIn('Survival_Rate', target_analysis)
        self.assertEqual(target_analysis['Total_Passengers'], 5)


if __name__ == '__main__':
    unittest.main()
