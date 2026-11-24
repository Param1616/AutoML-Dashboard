import os
import sys
import time
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# macOS OpenMP helper for XGBoost / LightGBM
if sys.platform == "darwin":
    for p in ["/opt/homebrew/opt/libomp/lib/libomp.dylib", "/usr/local/opt/libomp/lib/libomp.dylib"]:
        if os.path.exists(p):
            try:
                import ctypes
                ctypes.CDLL(p)
                break
            except Exception:
                pass

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, r2_score, mean_squared_error, mean_absolute_error,
    confusion_matrix
)

XGBOOST_AVAILABLE = False
LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False


def get_classification_models():
    """Returns dictionary of classification model instances."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=75, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0, n_jobs=-1
        )
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1, n_jobs=-1
        )
    return models


def get_regression_models():
    """Returns dictionary of regression model instances."""
    models = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=5000, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=75, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "Support Vector Regressor": SVR(kernel="rbf", C=1.0),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0, n_jobs=-1
        )
    if LGBM_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbose=-1, n_jobs=-1
        )
    return models


def evaluate_classification(model, X_test, y_test):
    """
    Computes all standard classification evaluation metrics.
    Accuracy, Precision, Recall, F1 score, Confusion Matrix, and ROC AUC.
    """
    y_pred = model.predict(X_test)
    classes = np.unique(y_test)
    n_classes = len(classes)
    avg = "weighted" if n_classes > 2 else "binary"

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
    rec = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if n_classes == 2:
                roc_auc = float(roc_auc_score(y_test, proba[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
        elif hasattr(model, "decision_function"):
            dfunc = model.decision_function(X_test)
            if n_classes == 2:
                roc_auc = float(roc_auc_score(y_test, dfunc))
            else:
                roc_auc = float(roc_auc_score(y_test, dfunc, multi_class="ovr", average="weighted"))
    except Exception:
        roc_auc = None

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
    }
    if roc_auc is not None:
        metrics["ROC AUC"] = round(roc_auc, 4)

    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classes": [str(c) for c in classes]
    }


def evaluate_regression(model, X_test, y_test):
    """
    Computes all standard regression evaluation metrics:
    R² Score, RMSE, MAE, MSE, and residuals.
    """
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    residuals = (y_test - y_pred).tolist()

    metrics = {
        "R2 Score": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
    }
    return {
        "metrics": metrics,
        "y_pred": y_pred.tolist(),
        "residuals": residuals
    }


def get_feature_importance(model, feature_names):
    """
    Extracts feature importances or coefficient magnitudes.
    """
    if not feature_names:
        return None

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)

    if importances is not None and len(importances) == len(feature_names):
        # Normalize sum to 1.0 if possible
        total = np.sum(importances)
        if total > 0:
            importances = importances / total
        return dict(zip(feature_names, [round(float(v), 5) for v in importances]))

    return None


def train_all_models(X_train, X_test, y_train, y_test, task_type,
                     selected_models=None, progress_callback=None):
    """
    Trains and evaluates all candidate models for the task.
    """
    all_models = get_classification_models() if task_type == "classification" else get_regression_models()
    
    if selected_models:
        models = {k: v for k, v in all_models.items() if k in selected_models}
    else:
        models = all_models

    results = {}
    total = len(models)

    for i, (name, model) in enumerate(models.items()):
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - start_time

            if task_type == "classification":
                eval_data = evaluate_classification(model, X_test, y_test)
            else:
                eval_data = evaluate_regression(model, X_test, y_test)

            results[name] = {
                "model": model,
                "metrics": eval_data["metrics"],
                "y_pred": eval_data["y_pred"],
                "training_time": round(elapsed, 4),
                "error": None,
                "extra": eval_data
            }
        except Exception as e:
            results[name] = {
                "model": None,
                "metrics": {},
                "y_pred": None,
                "training_time": 0.0,
                "error": str(e),
                "extra": {}
            }

        if progress_callback:
            progress_callback((i + 1) / total, name)

    return results


def get_best_model(results, task_type, primary_metric=None):
    """
    Selects the best performing model based on the primary metric.
    """
    if not results:
        return None, None

    if primary_metric is None:
        primary_metric = "F1 Score" if task_type == "classification" else "R2 Score"

    best_name, best_score = None, -np.inf
    # For metrics where lower is better (RMSE, MAE, MSE)
    lower_is_better = primary_metric in ["RMSE", "MAE", "MSE"]
    if lower_is_better:
        best_score = np.inf

    for name, info in results.items():
        if info.get("error") or not info.get("metrics"):
            continue
        score = info["metrics"].get(primary_metric)
        if score is None:
            continue
        
        if lower_is_better:
            if score < best_score:
                best_score = score
                best_name = name
        else:
            if score > best_score:
                best_score = score
                best_name = name

    # Fallback to first available if none matched primary
    if best_name is None:
        for name, info in results.items():
            if not info.get("error") and info.get("model") is not None:
                best_name = name
                break

    return best_name, results.get(best_name)
