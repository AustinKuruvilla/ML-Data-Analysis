# ML Data Analysis 📊

A comprehensive machine learning workflow demonstrating data exploration, feature analysis, classical ML modeling, and evaluation with scikit-learn. Perfect for learning the complete data science pipeline from EDA to model deployment.

## 📋 Overview

This project showcases a production-grade ML workflow including exploratory data analysis (EDA), feature engineering, Random Forest classification with cross-validation, and comprehensive model evaluation. Includes visualization of feature importance and confusion matrices.

## ✨ Features

- **Exploratory Data Analysis**: Dataset shape, types, statistics, missing values
- **Feature Analysis**: Correlation matrices with target and inter-feature relationships
- **Data Quality Checks**: Missing data patterns and data type validation
- **Model Training**: Random Forest Classifier with 5-fold cross-validation
- **Feature Importance**: Identification of most predictive features
- **Evaluation Metrics**: Precision, recall, F1-score, confusion matrix, and ROC-AUC
- **Comprehensive Logging**: Detailed outputs for reproducibility

## 📁 Project Structure

```
ml-data-analysis/
├── src/
│   ├── __init__.py
│   └── analysis.py          # Main ML workflow
├── tests/
│   └── test_analysis.py     # Unit tests for analysis
├── notebooks/
│   ├── 01-ml-data-analysis.ipynb      # Interactive Jupyter notebook
│   └── 01-ml-data-analysis.md         # Notebook markdown export
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Analysis

```bash
# Execute the complete ML workflow
python -m src.analysis
```

### 3. Run Tests

```bash
# From portfolio root
python run_basic_tests.py
```

### 4. Explore in Jupyter

```bash
jupyter notebook notebooks/01-ml-data-analysis.ipynb
```

## 📊 Example Workflow & Output

```
=== ML Data Analysis Workflow ===

Step 1: Data Exploration
  Dataset Shape: (100, 5)
  Features: 4 | Target: 1
  Missing Values: 0
  
Step 2: Feature Analysis
  Correlation with Target:
    feature_1: +0.85 (strong positive)
    feature_2: -0.42 (moderate negative)
    feature_3: +0.12 (weak positive)

Step 3: Model Training
  Algorithm: Random Forest (100 trees)
  Cross-validation: 5-fold
  Mean Accuracy: 0.92 ± 0.03

Step 4: Evaluation
  Precision: 0.91
  Recall: 0.93
  F1-Score: 0.92
  ROC-AUC: 0.96
  
Step 5: Feature Importance
  1. feature_1: 45.3%
  2. feature_2: 32.1%
  3. feature_3: 15.2%
  4. feature_4: 7.4%
```

## 🔧 Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.7+ |
| **ML Framework** | scikit-learn |
| **Data Analysis** | pandas, numpy |
| **Visualization** | (output-based) |
| **Notebooks** | Jupyter |

## 📦 Dependencies

- scikit-learn (machine learning and validation)
- pandas (data manipulation and analysis)
- numpy (numerical computing)

See `requirements.txt` for exact versions.

## 🎓 What You'll Learn

- **EDA Techniques**: Statistical summaries, data quality checks
- **Feature Analysis**: Correlation matrices and target relationships
- **Model Selection**: Random Forest advantages and configuration
- **Cross-Validation**: K-fold validation for robust evaluation
- **Feature Importance**: Interpreting model feature selection
- **Metrics**: Precision, recall, F1-score, ROC-AUC for classification
- **Best Practices**: Reproducibility, logging, and documentation

## 📝 Key Functions

### `build_demo_dataset() -> pd.DataFrame`
Generates a realistic 100-sample binary classification dataset with 4 features.

### `explore_data(df: pd.DataFrame) -> None`
Comprehensive EDA: shape, types, missing values, basic statistics.

### `analyze_features(df: pd.DataFrame) -> None`
Feature analysis: correlation with target, inter-feature relationships.

### `train_and_eval(df: pd.DataFrame) -> None`
Complete ML pipeline: train Random Forest with cross-validation, evaluate with metrics.

## 🔍 Example Usage

```python
from src.analysis import build_demo_dataset, explore_data, analyze_features, train_and_eval

# Load or create data
df = build_demo_dataset()

# Explore the data
explore_data(df)

# Analyze features
analyze_features(df)

# Train and evaluate model
train_and_eval(df)
```

## 🧪 Testing

Unit tests verify:
- ✅ Dataset creation with correct dimensions
- ✅ Feature columns present and correct types
- ✅ Target variable validity
- ✅ Complete pipeline execution

Run tests with: `python run_basic_tests.py`

## 📈 ML Pipeline Stages

1. **Data Loading**: Generate or load dataset
2. **EDA**: Understand data distributions and quality
3. **Feature Engineering**: Analyze relationships and correlations
4. **Train/Test Split**: 80/20 stratified split
5. **Model Training**: Random Forest with 5-fold cross-validation
6. **Evaluation**: Comprehensive metrics and confusion matrix
7. **Feature Importance**: Interpret model decisions

## 🎯 Performance Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / All positive predictions
- **Recall**: True positives / All actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve

## 📚 Further Reading

- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forests)
- [Cross-Validation Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Feature Importance Interpretation](https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-d2ff84607b7b)
- [EDA Guide](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15)

## 📄 License

This project is part of a portfolio and open to use for educational purposes.

---

**Last Updated**: August 2026  
**Status**: Production Ready ✅
