# ⚡ AutoPilot ML — Enterprise Automated Machine Learning Platform

**AutoPilot ML** is an enterprise-grade Automated Machine Learning and Predictive Analytics platform built with Python, Scikit-Learn, and Streamlit. Designed for data scientists, ML engineers, and business analysts, AutoPilot ML automates the end-to-end data science lifecycle from data ingestion and cleaning to model exploration, training, diagnostic comparison, and real-time/batch inference deployment.

---

## 🌟 Key Features

### 1. Robust Dataset Ingestion & Explorer
- **Multi-Format Ingestion**: Upload `.csv`, `.xlsx`, and `.xls` datasets seamlessly with encoding autodetection.
- **Pre-packaged Demos**: Built-in Customer Churn (Classification) and Real Estate Housing Valuation (Regression) datasets.
- **Dataset Dimensions & Schema**: Inspect rows, feature counts, memory usage, and column data types.
- **Duplicate Value Engine**: Detect duplicate rows with 1-click removal.
- **Missing Value Analysis**: Detailed missing record counts and percentages with visual bar charts.

### 2. Exploratory Data Analytics (EDA)
- **Feature Distributions**: Interactive multi-feature histograms and box plots.
- **Correlation Heatmap**: Annotated Pearson correlation matrix with interactive hover tooltips.
- **Categorical Breakdown**: Frequency analysis and visual distribution bar charts.
- **Outlier & Anomaly Detection**: Interquartile Range (IQR) bounds identification ($Q_1 - 1.5 \times \text{IQR}$, $Q_3 + 1.5 \times \text{IQR}$) with outlier counts and percentage flags.

### 3. Automated Machine Learning Engine
- **Intelligent Task Detection**: Automatic classification vs. regression task detection with **manual override**.
- **Automated Preprocessing**:
  - Missing value imputation (auto, median, mean, mode, or drop rows).
  - Feature scaling (StandardScaler, MinMaxScaler, RobustScaler, or none).
  - Outlier handling (optional 1st/99th percentile clipping).
  - Automatic categorical one-hot encoding with unknown category handling.
  - Stratified train-test splitting.
- **Parallel Multi-Model Training**: Train and compare up to 10 algorithms simultaneously.

### 4. Comprehensive Model Comparison & Diagnostic Leaderboard
- **Supported Algorithms**:
  - *Classification*: Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, Decision Tree, K-Nearest Neighbors, Naive Bayes, Support Vector Machine (SVC), XGBoost, LightGBM.
  - *Regression*: Ridge, Lasso, Random Forest, Gradient Boosting, AdaBoost, Decision Tree, K-Nearest Neighbors, Support Vector Regressor (SVR), XGBoost, LightGBM.
- **Exhaustive Evaluation Metrics**:
  - *Classification*: Accuracy, Precision, Recall, F1 Score, ROC AUC, and **Confusion Matrix Heatmap**.
  - *Regression*: $R^2$ Score, RMSE, MAE, MSE, and **Residuals / Actual vs. Predicted Plots**.
- **Training Latency & Radar Profiles**: Speed benchmark and multi-dimensional performance radar charts.
- **Feature Importance**: Top feature ranking and impact visualization.

### 5. Interactive & Batch Inference Engine
- **Single Record Prediction**: Dynamic form with smart input widgets (dropdowns for categories with dataset choices, numeric inputs initialized to distribution medians).
- **Class Probabilities**: Complete confidence breakdown for classification targets.
- **Batch Inference**: Upload new unlabeled CSV/Excel test sets, generate predictions, and export results with 1-click CSV download.

### 6. Production Model Export
- **Serialized Pipeline Package (.pkl)**: Exports standalone `AutoPilotModelPipeline` bundling both preprocessing and model weights.
- **Leaderboard CSV Export**: Download complete benchmark metrics.
- **Integration Code Snippet**: Ready-to-use Python snippet for REST APIs and batch jobs.

---

## 🏗️ Project Architecture

```
automl-dashboard/
├── app.py                      # Main AutoPilot ML Streamlit application
├── requirements.txt            # Project dependencies
├── .streamlit/
│   └── config.toml             # SaaS theme & server settings
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data cleaning, type detection, ColumnTransformer pipeline
│   ├── model_trainer.py        # Model registries, parallel training, evaluation metrics
│   ├── visualizations.py       # Plotly visualizations (heatmaps, distributions, residuals)
│   └── utils.py                # Dataset loaders, demo generators, export utilities
└── sample_data/                # Sample datasets for quick demonstration
    ├── customer_churn_classification.csv
    ├── customer_churn_classification.xlsx
    ├── real_estate_regression.csv
    └── real_estate_regression.xlsx
```

---

## 🚀 Running Locally

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13, 3.14)
- (macOS only) Homebrew OpenMP runtime for XGBoost / LightGBM:
  ```bash
  brew install libomp
  ```

### 2. Environment Setup
```bash
# Clone the repository and navigate into it
git clone https://github.com/Param1616/AutoML-Dashboard.git
cd AutoML-Dashboard

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Application
```bash
streamlit run app.py
```

The application will start immediately at:
**[http://localhost:8501](http://localhost:8501)**

---

## 📖 Step-by-Step Workflow Guide

1. **Ingest Data**: In **Dataset Explorer**, drag-and-drop any `.csv` or `.xlsx` file, or select one of the built-in demo datasets from the sidebar. Inspect data types, detect and remove duplicate rows with 1 click, and review missing value breakdowns.
2. **Explore Analytics**: Go to **Data Analytics** to inspect distribution histograms, feature correlation heatmaps, categorical value counts, and IQR outlier boundaries.
3. **Train Models**: Switch to **AutoML Training**, pick your prediction target, select preprocessing options (imputation, scaling, outlier clipping), choose which algorithms to evaluate, and click **Start AutoML Training**.
4. **Evaluate & Diagnose**: In **Model Comparison**, review the leaderboard ranked by performance, inspect confusion matrices (for classification) or actual-vs-predicted residual charts (for regression), and analyze top feature importances.
5. **Predict**: In **Predictions**, generate real-time inferences with interactive widgets or upload a batch test file for instant predictions.
6. **Deploy**: In **Model Export**, download the self-contained `.pkl` pipeline package and use it in any Python application.

---

## 📦 Using Exported Models in Production

Once you download your trained pipeline (`autopilot_<model>_pipeline.pkl`), you can make predictions in any Python application:

```python
import joblib
import pandas as pd

# 1. Load the serialized pipeline
pipeline = joblib.load('autopilot_random_forest_pipeline.pkl')

# 2. Pass raw data (no manual scaling or categorical encoding needed!)
sample_data = pd.DataFrame([{
    'tenure_months': 12,
    'monthly_charges': 65.50,
    'total_charges': 786.00,
    'contract': 'Month-to-month',
    'payment_method': 'Electronic check',
    'internet_service': 'Fiber optic',
    'senior_citizen': 0,
    'support_tickets': 1
}])

# 3. Generate predictions directly
predictions = pipeline.predict(sample_data)
print("Predicted Class:", predictions)

# 4. Optional: Retrieve prediction confidence probabilities
probabilities = pipeline.predict_proba(sample_data)
print("Class Probabilities:", probabilities)
```

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Core ML & Preprocessing**: [Scikit-Learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Spreadsheet Support**: [OpenPyXL](https://openpyxl.readthedocs.io/)
- **Interactive Visualizations**: [Plotly](https://plotly.com/python/)
- **Hyperparameter Optimization**: [Optuna](https://optuna.org/)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)

---

## 👨‍💻 Author & Developer

**Param Jaiswal**  
Lead Developer & ML Engineer  
Project: **AutoPilot ML**  


