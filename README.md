# ML Data Analysis 📊

End-to-end machine learning pipeline: data exploration, feature analysis, Random Forest classification, and model evaluation with comprehensive logging and testing.

## ✨ Highlights

- **Complete ML Pipeline**: EDA → Feature Analysis → Model Training → Evaluation
- **Random Forest Classifier**: Train-test split with multi-metric evaluation (accuracy, precision, recall, F1)
- **Modular Design**: Clean separation of concerns with unit tests
- **Production-Ready**: Structured logging and reproducible workflows

## 📁 Project Structure

```
ml-data-analysis/
├── src/analysis.py           # Core ML pipeline
├── tests/test_analysis.py    # Unit tests
├── notebooks/                # Jupyter notebook & export
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## 🚀 Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run pipeline
python -m src.analysis

# Run tests
python -m pytest tests/test_analysis.py

# Run interactive Streamlit demo
streamlit run streamlit_app.py
```

## 📊 Pipeline

1. **Data Generation**: Synthetic 100-sample binary classification dataset
2. **EDA**: Shape, types, missing values, class distribution
3. **Feature Analysis**: Correlation matrices with target variable
4. **Model Training**: Random Forest (50 trees, max_depth=5) with 75/25 train-test split
5. **Evaluation**: Accuracy, precision, recall, F1-score

## � Tech Stack

**Python 3.7+** with scikit-learn, pandas, numpy, matplotlib, seaborn, jupyter


## Output
[ML Data Analysis](https://austinkuruvilla-ml-data-analysis-streamlit-app-limzig.streamlit.app/)
## 🧪 Testing

```bash
python -m pytest tests/test_analysis.py -v
```

Validates: dataset creation, feature columns, target variable, EDA execution, feature analysis, and model pipeline.

---

**Python 3.7+** | **scikit-learn** | Production Ready ✅



