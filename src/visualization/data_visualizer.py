"""
Data visualization module for creating charts and plots.
"""
import os 
import sys
from src.logger import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import yaml


class DataVisualizer:
    """Class for creating data visualizations."""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize DataVisualizer with configuration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set visualization parameters
        self.figure_size = self.config['visualization']['figure_size']
        self.style = self.config['visualization']['style']
        self.color_palette = self.config['visualization']['color_palette']
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_style(self.style)
        sns.set_palette(self.color_palette)
        
        # Create reports directory
        self.reports_path = Path(self.config['data']['reports_path'])
        self.reports_path.mkdir(parents=True, exist_ok=True)
    
    def plot_survival_distribution(self, df, save_path=None):
        """Plot survival distribution."""
        plt.figure(figsize=self.figure_size)
        sns.countplot(x='Survived', data=df)
        plt.title('Survived Distribution')
        plt.xlabel('Survived (0=No, 1=Yes)')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_gender_distribution(self, df, save_path=None):
        """Plot gender distribution."""
        plt.figure(figsize=self.figure_size)
        sns.countplot(x='Sex', data=df)
        plt.title('Gender Distribution')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_siblings_distribution(self, df, save_path=None):
        """Plot siblings distribution."""
        plt.figure(figsize=self.figure_size)
        sns.countplot(x='SibSp', data=df)
        plt.title('Siblings Distribution')
        plt.xlabel('Number of Siblings/Spouses')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_age_distribution(self, df, save_path=None):
        """Plot age distribution with age groups."""
        # Create age groups
        bins = [0, 12, 25, 55, 80]
        df['AgeGroup'] = pd.cut(df['Age'], bins=bins, right=True)
        
        # Count people in each group
        age_counts = df['AgeGroup'].value_counts().sort_index()
        
        # Explode settings (highlight the largest group)
        explode = [0.1 if count == age_counts.max() else 0 for count in age_counts]
        
        # Plot pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            age_counts,
            labels=age_counts.index.astype(str),
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2", len(age_counts)),
            explode=explode,
            shadow=True
        )
        plt.title('Age Distribution in 4 Categories')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_parch_distribution(self, df, save_path=None):
        """Plot parents/children distribution."""
        plt.figure(figsize=self.figure_size)
        sns.countplot(x='Parch', data=df)
        plt.title('Parents/Children Distribution')
        plt.xlabel('Number of Parents/Children')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_fare_by_class(self, df, save_path=None):
        """Plot fare distribution by passenger class."""
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Pclass', y='Fare', data=df, palette='Set2')
        plt.title('Fare Distribution by Passenger Class')
        plt.xlabel('Passenger Class')
        plt.ylabel('Fare')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_embarked_distribution(self, df, save_path=None):
        """Plot embarked distribution."""
        plt.figure(figsize=self.figure_size)
        sns.countplot(x='Embarked', data=df)
        plt.title('Embarked Distribution')
        plt.xlabel('Port of Embarkation')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_passenger_class_distribution(self, df, save_path=None):
        """Plot passenger class distribution."""
        plt.figure(figsize=self.figure_size)
        sns.countplot(x='Pclass', data=df)
        plt.title('Passenger Class Distribution')
        plt.xlabel('Passenger Class')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_age_group_distribution(self, df, save_path=None):
        """Plot age group distribution."""
        # Create age groups if not exists
        if 'AgeGroup' not in df.columns:
            bins = [0, 12, 25, 55, 80]
            df['AgeGroup'] = pd.cut(df['Age'], bins=bins, right=True)
        
        plt.figure(figsize=self.figure_size)
        sns.countplot(x='AgeGroup', data=df)
        plt.title('Age Group Distribution')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_survival_by_gender(self, df, save_path=None):
        """Plot survival by gender."""
        # Count survivors grouped by gender
        survival_counts = df[df['Survived'] == 1]['Sex'].value_counts()
        
        # Explode effect (highlight the group with more survivors)
        explode = [0.1 if count == survival_counts.max() else 0 for count in survival_counts]
        
        # Total number of males and females
        logging.info(df['Sex'].value_counts())
        
        # Plot pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            survival_counts,
            labels=survival_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("pastel", len(survival_counts)),
            explode=explode,
            shadow=True
        )
        plt.title('Survived Passengers by Gender')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_survival_by_class(self, df, save_path=None):
        """Plot survival by passenger class."""
        # Count survivors grouped by passenger class
        survival_counts = df[df['Survived'] == 1]['Pclass'].value_counts()
        
        # Explode effect (highlight the group with more survivors)
        explode = [0.1 if count == survival_counts.max() else 0 for count in survival_counts]
        
        # Total number of passengers in each class
        logging.info(df['Pclass'].value_counts())
        
        # Plot pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            survival_counts,
            labels=survival_counts.index.astype(str) + ' Class',
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set3", len(survival_counts)),
            explode=explode,
            shadow=True
        )
        plt.title('Survived Passengers by Class')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def create_comprehensive_visualization(self, df, save_plots=True):
        """Create comprehensive visualization suite."""
        logging.info("CREATING COMPREHENSIVE DATA VISUALIZATIONS")
        
        
        plots = [
            ('survival_distribution', self.plot_survival_distribution),
            ('gender_distribution', self.plot_gender_distribution),
            ('siblings_distribution', self.plot_siblings_distribution),
            ('age_distribution', self.plot_age_distribution),
            ('parch_distribution', self.plot_parch_distribution),
            ('fare_by_class', self.plot_fare_by_class),
            ('embarked_distribution', self.plot_embarked_distribution),
            ('passenger_class_distribution', self.plot_passenger_class_distribution),
            ('age_group_distribution', self.plot_age_group_distribution),
            ('survival_by_gender', self.plot_survival_by_gender),
            ('survival_by_class', self.plot_survival_by_class),
        ]
        
        for plot_name, plot_func in plots:
            logging.info(f"\nCreating {plot_name}...")
            save_path = None
            if save_plots:
                save_path = self.reports_path / f"{plot_name}.png"
            
            plot_func(df, save_path)
        
        logging.info(f"\nAll visualizations completed. Plots saved to {self.reports_path}")
    
    def create_correlation_heatmap(self, df, save_path=None):
        """Create correlation heatmap for numerical features."""
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap of Numerical Features')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
    
    def create_feature_importance_plot(self, feature_importance_df, save_path=None):
        """Create feature importance plot."""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
