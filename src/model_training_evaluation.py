import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score
)
import numpy as np
from typing import Dict, Any

def _evaluate_single_model(model: Any, X_data: pd.DataFrame, y_data: pd.Series, task_type: str) -> Dict[str, Any]:
    """Helper to calculate metrics for a single trained model on a given dataset."""
    results = {}
    
    try:
        if task_type == "classification":
            y_pred = model.predict(X_data)
            results["accuracy"] = accuracy_score(y_data, y_pred)
            results["precision"] = precision_score(y_data, y_pred, average="weighted", zero_division=0)
            results["recall"] = recall_score(y_data, y_pred, average="weighted", zero_division=0)
            results["f1_score"] = f1_score(y_data, y_pred, average="weighted", zero_division=0)

        elif task_type == "regression":
            y_pred = model.predict(X_data)
            results["r2_score"] = r2_score(y_data, y_pred)
            results["mean_squared_error"] = mean_squared_error(y_data, y_pred)
            results["mean_absolute_error"] = mean_absolute_error(y_data, y_pred)
            
        elif task_type == "unsupervised":
            y_pred = model.labels_
            results["silhouette_score"] = silhouette_score(X_data, y_pred)
            
    except Exception as e:
        results["error"] = str(e)
        
    return results

def train_and_evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_variable: str,
    problem_type: str,
    models: Dict[str, Any]
) -> Dict[str, Any]:
    """Trains a set of models on pre-split data and returns a full report."""
    print("\n--- Starting Model Training & Evaluation ---")
    
    is_unsupervised = problem_type == "unsupervised"
    
    if not is_unsupervised:
        X_train = train_df.drop(columns=[target_variable])
        y_train = train_df[target_variable]
        X_test = test_df.drop(columns=[target_variable])
        y_test = test_df[target_variable]
    else: 
        X_train, y_train = train_df, None
        X_test, y_test = test_df, None

    all_model_reports = {}
    best_primary_score = -np.inf 
    best_model_name = ""

    for name, model in models.items():
        try:
            print(f"Training {name}...")
            model.fit(X_train, y_train)

            eval_X = X_train if is_unsupervised else X_test
            eval_y = y_train if is_unsupervised else y_test
            
            report = _evaluate_single_model(model, eval_X, eval_y, problem_type)
            all_model_reports[name] = report
            
            primary_metric = -np.inf
            if problem_type == "regression":
                primary_metric = report.get("r2_score", -np.inf)
            elif problem_type == "classification":
                primary_metric = report.get("accuracy", -np.inf)
            elif problem_type == "unsupervised":
                primary_metric = report.get("silhouette_score", -np.inf)

            if primary_metric > best_primary_score:
                best_primary_score = primary_metric
                best_model_name = name

        except Exception as e:
            all_model_reports[name] = {"error": str(e)}

    print(f"Best Model Found: {best_model_name} (Primary Score: {best_primary_score:.4f})")
    
    return {
        'best_model_name': best_model_name,
        'best_model_score': best_primary_score,
        'all_model_reports': all_model_reports
    }

