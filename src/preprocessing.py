import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


HIGH_CARDINALITY_THRESHOLD = 50
CLASSIFICATION_UNIQUE_LIMIT = 20


def detect_task_type(df, target_col, override=None):
    """
    Detect whether the problem is classification or regression.
    Supports user manual override ('classification' or 'regression').
    """
    if override in ["classification", "regression"]:
        return override

    col = df[target_col]
    
    # If explicitly string, object, categorical, or boolean -> classification
    if (
        pd.api.types.is_string_dtype(col)
        or pd.api.types.is_object_dtype(col)
        or pd.api.types.is_categorical_dtype(col)
        or pd.api.types.is_bool_dtype(col)
    ):
        return "classification"
    
    # If float with few unique values relative to total count -> classification, else regression
    if pd.api.types.is_float_dtype(col):
        if col.nunique() <= 10 and (col.nunique() / len(col)) < 0.05:
            return "classification"
        return "regression"
    
    # If integer with few unique values
    unique_count = col.nunique()
    if unique_count <= CLASSIFICATION_UNIQUE_LIMIT or (unique_count / len(col)) < 0.05:
        return "classification"
    
    return "regression"


def detect_feature_types(df, target_col=None):
    """
    Classifies columns into numerical, categorical, datetime, and high-cardinality/id columns.
    """
    features = [c for c in df.columns if c != target_col]
    num_cols = []
    cat_cols = []
    date_cols = []
    dropped_cols = []

    for col in features:
        series = df[col]
        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(col)
            continue
        
        # Check string / categorical / object
        if (
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            nunique = series.nunique()
            if nunique > HIGH_CARDINALITY_THRESHOLD and (nunique / len(df)) > 0.8:
                dropped_cols.append(col)
            else:
                cat_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # If numeric has only 2 unique values and is boolean-like (0/1)
            if series.nunique() <= 2 and series.dropna().isin([0, 1]).all():
                cat_cols.append(col)
            else:
                num_cols.append(col)
        elif pd.api.types.is_bool_dtype(series):
            cat_cols.append(col)
        else:
            cat_cols.append(col)

    return {
        "numerical": num_cols,
        "categorical": cat_cols,
        "datetime": date_cols,
        "high_cardinality": dropped_cols
    }


def analyze_missing_values(df):
    """
    Returns summary DataFrame of missing values.
    """
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    res = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing Count": null_counts.values,
        "Missing (%)": np.round(null_pct.values, 2)
    })
    return res.sort_values("Missing Count", ascending=False).reset_index(drop=True)


def analyze_duplicates(df):
    """
    Analyzes duplicate rows in the dataset.
    """
    dup_count = int(df.duplicated().sum())
    dup_pct = round((dup_count / len(df)) * 100, 2) if len(df) > 0 else 0.0
    return {
        "duplicate_count": dup_count,
        "duplicate_percentage": dup_pct
    }


def drop_duplicates(df):
    """
    Removes duplicate rows.
    """
    return df.drop_duplicates().reset_index(drop=True)


def analyze_outliers(df, num_cols=None):
    """
    Identifies outliers using IQR (Interquartile Range) method.
    """
    if num_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_info = []
    for col in num_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 4:
            continue
        q25, q75 = np.percentile(series, 25), np.percentile(series, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_info.append({
            "Column": col,
            "Outlier Count": len(outliers),
            "Outlier (%)": round(len(outliers) / len(series) * 100, 2),
            "Lower Bound": round(lower_bound, 3),
            "Upper Bound": round(upper_bound, 3),
            "Min": round(series.min(), 3),
            "Max": round(series.max(), 3)
        })
    return pd.DataFrame(outlier_info)


class AutoPilotModelPipeline:
    """
    Standalone serializable inference pipeline bundling preprocessor,
    feature metadata, and trained model.
    Enables direct .predict() and .predict_proba() on raw input DataFrames.
    """
    def __init__(self, model, preprocessor, task_type, target_col, feature_cols,
                 num_cols, cat_cols, target_encoder=None, original_categories=None):
        self.model = model
        self.preprocessor = preprocessor
        self.task_type = task_type
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.target_encoder = target_encoder
        self.original_categories = original_categories or {}

    def transform_features(self, df):
        input_data = df.copy()
        for col in self.feature_cols:
            if col not in input_data.columns:
                input_data[col] = np.nan
        df_sub = input_data[self.feature_cols]
        return self.preprocessor.transform(df_sub)

    def predict(self, df):
        X_trans = self.transform_features(df)
        preds = self.model.predict(X_trans)
        if self.task_type == "classification" and self.target_encoder is not None:
            return self.target_encoder.inverse_transform(np.asarray(preds).astype(int))
        return preds

    def predict_proba(self, df):
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification models.")
        X_trans = self.transform_features(df)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_trans)
        elif hasattr(self.model, "decision_function"):
            dfunc = self.model.decision_function(X_trans)
            if dfunc.ndim == 1:
                p1 = 1 / (1 + np.exp(-dfunc))
                return np.vstack([1 - p1, p1]).T
            exp_d = np.exp(dfunc - np.max(dfunc, axis=1, keepdims=True))
            return exp_d / np.sum(exp_d, axis=1, keepdims=True)
        return None


def full_preprocess(df, target_col, task_type=None, impute_strategy="auto",
                    scaler_type="standard", test_size=0.2, handle_outliers="none", random_state=42):
    """
    Comprehensive AutoML preprocessing engine:
    1. Splits target and features.
    2. Encodes target (if classification).
    3. Detects numerical vs categorical columns and drops high-cardinality ID features.
    4. Handles missing values and outliers.
    5. Trains ColumnTransformer (imputation + scaling + one-hot encoding).
    6. Returns train/test sets, transformers, and metadata.
    """
    df = df.copy()

    # Determine task type
    task_type = detect_task_type(df, target_col, override=task_type)

    # Separate target and features
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Target encoding
    target_encoder = None
    target_classes = None
    if task_type == "classification":
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y.astype(str))
        y = pd.Series(y_encoded, name=target_col)
        target_classes = list(target_encoder.classes_)
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid_idx = ~y.isna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

    # Detect feature types
    type_info = detect_feature_types(X)
    num_cols = type_info["numerical"]
    cat_cols = type_info["categorical"]
    dropped_cols = type_info["high_cardinality"]

    # Drop high cardinality columns from features
    if dropped_cols:
        X = X.drop(columns=dropped_cols, errors="ignore")

    active_features = [c for c in X.columns if c in num_cols or c in cat_cols]
    num_cols = [c for c in num_cols if c in active_features]
    cat_cols = [c for c in cat_cols if c in active_features]

    # Outlier handling (optional clipping)
    if handle_outliers == "clip" and num_cols:
        for col in num_cols:
            s_num = pd.to_numeric(X[col], errors="coerce")
            lower = s_num.quantile(0.01)
            upper = s_num.quantile(0.99)
            X[col] = s_num.clip(lower, upper)

    # Store original category unique values for smart UI widgets
    original_categories = {}
    for col in cat_cols:
        unique_vals = X[col].dropna().unique().tolist()
        original_categories[col] = [str(v) for v in unique_vals][:100]

    # Handle missing row drop if strategy == "drop"
    if impute_strategy == "drop":
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

    # Train / Test split
    stratify = y if (task_type == "classification" and len(np.unique(y)) > 1 and min(pd.Series(y).value_counts()) >= 2) else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Build Numerical Pipeline
    num_strat = "median" if impute_strategy in ("auto", "median") else ("mean" if impute_strategy == "mean" else "most_frequent")
    num_steps = [("imputer", SimpleImputer(strategy=num_strat))]
    if scaler_type == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaler_type == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    elif scaler_type == "robust":
        num_steps.append(("scaler", RobustScaler()))
    
    num_pipeline = Pipeline(num_steps)

    # Build Categorical Pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Assemble ColumnTransformer
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    
    # Fit & Transform
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    # Get transformed feature names
    try:
        transformed_feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        transformed_feature_names = [f"feat_{i}" for i in range(X_train_trans.shape[1])]

    return {
        "X_train": X_train_trans,
        "X_test": X_test_trans,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "task_type": task_type,
        "target_col": target_col,
        "feature_cols": active_features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "transformed_feature_names": transformed_feature_names,
        "target_encoder": target_encoder,
        "target_classes": target_classes,
        "preprocessor": preprocessor,
        "dropped_cols": dropped_cols,
        "original_categories": original_categories,
        "impute_strategy": impute_strategy,
        "scaler_type": scaler_type
    }
