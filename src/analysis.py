"""Machine Learning Data Analysis and Modeling.

This module provides tools for data exploration, feature analysis,
model training, and evaluation.

Run as: python -m src.analysis
"""
import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_demo_dataset() -> pd.DataFrame:
    """Create a synthetic classification dataset for demonstration.
    
    Returns:
        DataFrame with features and binary target
    """
    np.random.seed(42)
    n_samples = 100
    
    # Generate synthetic features with correlation to target
    feature1 = np.random.normal(loc=4, scale=2, size=n_samples)
    feature2 = np.random.normal(loc=5, scale=3, size=n_samples)
    
    # Create target based on feature combinations
    label = (feature1 > 3) & (feature2 < 6)
    label = label.astype(int)
    
    df = pd.DataFrame({
        "feature1": feature1,
        "feature2": feature2,
        "label": label,
    })
    
    logger.info(f"Created dataset with {len(df)} samples and {len(df.columns)-1} features")
    return df


def explore_data(df: pd.DataFrame) -> Dict[str, any]:
    """Perform exploratory data analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with exploration statistics
    """
    logger.info("=" * 60)
    logger.info("DATA EXPLORATION")
    logger.info("=" * 60)
    
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Column types:\n{df.dtypes}")
    logger.info(f"\nMissing values:\n{df.isnull().sum()}")
    logger.info(f"\nBasic statistics:\n{df.describe()}")
    
    # Class distribution
    if 'label' in df.columns:
        logger.info(f"\nClass distribution:\n{df['label'].value_counts()}")
    
    stats = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
    }
    
    return stats


def analyze_features(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze feature importance and correlation.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with correlation statistics
    """
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE ANALYSIS")
    logger.info("=" * 60)
    
    feature_cols = [col for col in df.columns if col != 'label']
    
    # Feature correlations with target
    if 'label' in df.columns:
        logger.info("\nFeature Correlations with Target:")
        correlations = {}
        for feat in feature_cols:
            corr = df[feat].corr(df['label'])
            correlations[feat] = corr
            logger.info(f"  {feat}: {corr:.4f}")
        
        # Inter-feature correlation
        logger.info("\nFeature Correlation Matrix:")
        corr_matrix = df[feature_cols].corr()
        logger.info(f"\n{corr_matrix}")
        
        return correlations
    
    return {}


def train_and_eval(df: pd.DataFrame) -> Dict[str, any]:
    """Train a Random Forest classifier and evaluate performance.
    
    Args:
        df: DataFrame with features and target label
        
    Returns:
        Dictionary with model metrics and trained model
    """
    logger.info("\n" + "=" * 60)
    logger.info("MODEL TRAINING")
    logger.info("=" * 60)
    
    X = df[["feature1", "feature2"]]
    y = df["label"]
    
    # Lazy-import heavy ML libs to keep imports lightweight
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    logger.info(f"Training set size: {len(X)} samples")
    
    # Split data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    logger.info(f"Train: {len(Xtr)}, Test: {len(Xte)}")
    
    # Train model
    logger.info("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    model.fit(Xtr, ytr)
    
    # Evaluate
    logger.info("\nModel Evaluation:")
    train_accuracy = model.score(Xtr, ytr)
    test_accuracy = model.score(Xte, yte)
    logger.info(f"Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Predictions
    preds = model.predict(Xte)
    logger.info("\nClassification Report:")
    print(classification_report(yte, preds))
    
    # Confusion matrix
    cm = confusion_matrix(yte, preds)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Feature importance
    logger.info("\nFeature Importance:")
    for feat, imp in zip(["feature1", "feature2"], model.feature_importances_):
        logger.info(f"  {feat}: {imp:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    logger.info(f"\nCross-Validation Scores: {cv_scores}")
    logger.info(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return {
        "model": model,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "cv_scores": cv_scores,
    }


def main():
    """Main entry point for the analysis pipeline."""
    # Load data
    df = build_demo_dataset()
    
    # Explore
    explore_data(df)
    
    # Analyze features
    analyze_features(df)
    
    # Train and evaluate
    results = train_and_eval(df)
    
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
