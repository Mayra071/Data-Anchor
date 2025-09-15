# Data Anchor - Comprehensive Data Validation & Monitoring

A comprehensive data science project for data validation, drift detection, and machine learning model monitoring using Great Expectations and Evidently AI.

## 🚀 Project Overview

Data Anchor is a complete data validation and monitoring solution that provides:

- **Data Quality Assessment**: Comprehensive data quality analysis and scoring
- **Data Validation**: Automated validation using Great Expectations
- **Drift Detection**: Data and model drift monitoring with Evidently AI
- **Machine Learning**: Model training and evaluation with performance metrics
- **Visualization**: Rich data visualizations and reporting
- **Monitoring**: Continuous data and model monitoring capabilities

## 📁 Project Structure

```
Data Anchor/
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data files
│   ├── reference/              # Reference datasets for drift detection
│   └── current/                # Current datasets for drift detection
├── src/
│   ├── data_processing/        # Data loading and preprocessing modules
│   │   ├── data_loader.py
│   │   └── data_preprocessor.py
│   ├── validation/             # Data validation modules
│   │   └── great_expectations_validator.py
│   ├── drift_detection/        # Drift detection modules
│   │   └── evidently_drift_detector.py
│   ├── ml/                     # Machine learning modules
│   │   └── model_trainer.py
│   ├── visualization/          # Data visualization modules
│   │   └── data_visualizer.py
│   └── utils/                  # Utility functions
│       └── helpers.py
├── reports/                    # Generated reports and visualizations
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── models/                     # Trained models
├── main.py                     # Main execution script
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites
- **Python 3.9 - 3.12** (required)
- Windows, macOS, or Linux

### Quick Setup (Recommended)

1. **Clone or download the repository**:
   ```bash
   git clone <repository-url>
   cd "Data Anchor"
   ```

2. **Run the automated setup**:
   
   **For Windows:**
   ```cmd
   setup.bat
   ```
   
   **For macOS/Linux:**
   ```bash
   python setup.py
   ```

### Manual Setup

1. **Create virtual environment**:
   ```bash
   python -m venv .venv
   ```

2. **Activate virtual environment**:
   
   **Windows:**
   ```cmd
   .venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   # Try the main requirements first
   pip install -r requirements.txt
   
   # If that fails, use the simplified version
   pip install -r requirements-simple.txt
   ```

4. **Install optional packages** (if needed):
   ```bash
   pip install great-expectations==0.17.11
   pip install evidently==0.2.8
   ```

5. **Set up data path**:
   - Update the `raw_data_path` in `config.yaml` to point to your Titanic dataset
   - Default path: `"../../Data_Set/Titanic-Dataset.csv"`

### Troubleshooting

If you encounter compilation errors (especially on Windows):

1. **Install Microsoft Visual C++ Build Tools**:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "C++ build tools" workload

2. **Use pre-compiled packages**:
   ```bash
   pip install --only-binary=all -r requirements-simple.txt
   ```

3. **Alternative: Use conda**:
   ```bash
   conda create -n data-anchor python=3.11
   conda activate data-anchor
   conda install pandas numpy scikit-learn matplotlib seaborn
   pip install great-expectations evidently
   ```

## 🚀 Quick Start

### Run Complete Workflow

Execute the main script to run the complete data validation and monitoring workflow:

```bash
python main.py
```

This will:
1. Load and analyze the Titanic dataset
2. Perform data preprocessing and feature engineering
3. Validate data quality using Great Expectations
4. Train a machine learning model
5. Detect data and model drift using Evidently AI
6. Generate comprehensive visualizations
7. Create detailed reports

### Individual Module Usage

#### Data Loading and Analysis
```python
from src.data_processing.data_loader import DataLoader

loader = DataLoader()
df = loader.load_raw_data()
overview = loader.get_data_overview(df)
quality = loader.analyze_data_quality(df)
```

#### Data Validation
```python
from src.validation.great_expectations_validator import GreatExpectationsValidator

validator = GreatExpectationsValidator()
validation_result = validator.validate_dataset(df)
```

#### Drift Detection
```python
from src.drift_detection.evidently_drift_detector import EvidentlyDriftDetector

drift_detector = EvidentlyDriftDetector()
snapshot, results = drift_detector.run_drift_detection(reference_data, current_data)
```

#### Machine Learning
```python
from src.ml.model_trainer import ModelTrainer

trainer = ModelTrainer()
results = trainer.train_and_evaluate(train_data, test_data)
```

#### Visualization
```python
from src.visualization.data_visualizer import DataVisualizer

visualizer = DataVisualizer()
visualizer.create_comprehensive_visualization(df)
```

## 📊 Features

### Data Quality Assessment
- Missing value analysis
- Data type validation
- Statistical summary
- Data quality scoring

### Data Validation (Great Expectations)
- Table-level expectations
- Column existence validation
- Data quality constraints
- Missing value expectations
- Statistical validations

### Drift Detection (Evidently AI)
- Data drift detection
- Target drift detection
- Model performance drift
- Statistical drift analysis
- HTML report generation

### Machine Learning
- Logistic regression model
- Cross-validation
- Performance metrics
- Feature importance analysis
- Model persistence

### Visualization
- Distribution plots
- Correlation heatmaps
- Survival analysis charts
- Feature importance plots
- Comprehensive report generation

## 📈 Generated Reports

The project generates several types of reports:

1. **Great Expectations Validation Report**: `reports/ge_validation_result.json`
2. **Drift Detection Reports**: 
   - `reports/train_test_drift_report.html`
   - `reports/reference_current_drift_report.html`
   - `reports/drift_comparison_report.html`
3. **Data Visualizations**: Various PNG files in `reports/`
4. **Comprehensive Summary**: `reports/comprehensive_summary.json`

## ⚙️ Configuration

Edit `config.yaml` to customize:

- Data paths
- Model parameters
- Validation settings
- Drift detection configuration
- Visualization preferences

## 🧪 Testing

Run tests to verify functionality:

```bash
python -m pytest tests/
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- API documentation
- Configuration guide
- Troubleshooting guide
- Best practices

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Great Expectations](https://greatexpectations.io/) for data validation
- [Evidently AI](https://www.evidentlyai.com/) for drift detection
- [Scikit-learn](https://scikit-learn.org/) for machine learning
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualization

## 📞 Support

For questions and support, please open an issue in the repository.

---

**Data Anchor** - Your comprehensive solution for data validation and monitoring! 🎯
