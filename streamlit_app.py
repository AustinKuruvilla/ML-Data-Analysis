"""
Streamlit Demo: ML Data Analysis Pipeline
Showcases the complete machine learning workflow with interactive visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(page_title="ML Data Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 ML Data Analysis Pipeline")
st.markdown("**Interactive demonstration of data exploration, feature analysis, and model training**")

# ==================== HELPER FUNCTIONS ====================
@st.cache_data
def generate_dataset(n_samples=100, random_seed=42):
    """Generate synthetic binary classification dataset"""
    np.random.seed(random_seed)
    feature1 = np.random.normal(loc=4, scale=2, size=n_samples)
    feature2 = np.random.normal(loc=5, scale=3, size=n_samples)
    label = ((feature1 > 3) & (feature2 < 6)).astype(int)
    
    return pd.DataFrame({
        "feature1": feature1,
        "feature2": feature2,
        "label": label
    })

# ==================== MAIN APP ====================
st.sidebar.header("⚙️ Configuration")
n_samples = st.sidebar.slider("Dataset Size", 50, 500, 100, 10)
test_size = st.sidebar.slider("Test Split Ratio", 0.1, 0.5, 0.25, 0.05)
max_depth = st.sidebar.slider("Model Max Depth", 1, 10, 5)
n_trees = st.sidebar.slider("Random Forest Trees", 10, 200, 50, 10)

# Generate data
df = generate_dataset(n_samples=n_samples)

# ==================== TAB 1: DATA EXPLORATION ====================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploration", "🔗 Features", "🎯 Training", "📈 Results"])

with tab1:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Classes", df["label"].nunique())
    
    st.divider()
    
    # Data table
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("Raw Data (First 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col_right:
        st.subheader("Data Types")
        st.write(df.dtypes)
        st.subheader("Class Distribution")
        class_dist = df["label"].value_counts()
        st.bar_chart(class_dist)
    
    # Statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# ==================== TAB 2: FEATURE ANALYSIS ====================
with tab2:
    st.subheader("Feature Correlations")
    
    col_left, col_right = st.columns(2)
    
    # Correlation with target
    with col_left:
        st.markdown("**Correlation with Target:**")
        correlations = {}
        for feat in ["feature1", "feature2"]:
            corr = df[feat].corr(df["label"])
            correlations[feat] = corr
            st.write(f"• {feat}: **{corr:.4f}**")
    
    # Correlation matrix heatmap
    with col_right:
        st.markdown("**Correlation Matrix Heatmap:**")
        corr_matrix = df[["feature1", "feature2", "label"]].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Correlation"})
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)
    
    # Scatter plot
    st.divider()
    st.markdown("**Feature Distributions:**")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Scatter plot colored by class
    for label in df["label"].unique():
        mask = df["label"] == label
        axes[0].scatter(df[mask]["feature1"], df[mask]["feature2"], 
                       label=f"Class {label}", alpha=0.6, s=50)
    axes[0].set_xlabel("feature1")
    axes[0].set_ylabel("feature2")
    axes[0].set_title("Feature Space (colored by class)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histograms
    axes[1].hist(df["feature1"], alpha=0.5, label="feature1", bins=20)
    axes[1].hist(df["feature2"], alpha=0.5, label="feature2", bins=20)
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Feature Distributions")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    st.pyplot(fig)

# ==================== TAB 3: MODEL TRAINING ====================
with tab3:
    st.subheader("Random Forest Classifier Training")
    
    # Prepare data
    X = df[["feature1", "feature2"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Display training info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Train Set Size:** {len(X_train)} samples")
    with col2:
        st.info(f"**Test Set Size:** {len(X_test)} samples")
    with col3:
        st.info(f"**Model:** Random Forest ({n_trees} trees)")
    
    # Training progress
    st.divider()
    st.success("✅ Model training completed successfully!")

# ==================== TAB 4: RESULTS ====================
with tab4:
    st.subheader("Model Evaluation Results")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Accuracy", f"{train_acc:.4f}", f"{train_acc*100:.2f}%")
    with col2:
        st.metric("Test Accuracy", f"{test_acc:.4f}", f"{test_acc*100:.2f}%")
    
    st.divider()
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)
    
    st.divider()
    
    # Confusion Matrix
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_test_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    with col_right:
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feature_names = ["feature1", "feature2"]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title('Random Forest Feature Importance')
        st.pyplot(fig)
    
    st.divider()
    
    # Decision Boundary
    st.subheader("Decision Boundary Visualization")
    
    # Create mesh
    x_min, x_max = X["feature1"].min() - 1, X["feature1"].max() + 1
    y_min, y_max = X["feature2"].min() - 1, X["feature2"].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    for label in sorted(df["label"].unique()):
        mask = df["label"] == label
        ax.scatter(df[mask]["feature1"], df[mask]["feature2"],
                  label=f"Class {label}", alpha=0.7, s=50)
    
    ax.set_xlabel("feature1")
    ax.set_ylabel("feature2")
    ax.set_title("Decision Boundary - Random Forest Classifier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ==================== FOOTER ====================
st.divider()
st.markdown("""
---
**📚 Project:** ML Data Analysis Pipeline  
**🛠️ Technologies:** Python • scikit-learn • pandas • Streamlit  
**✅ Status:** Production Ready
""")
