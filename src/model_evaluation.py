import sys
import logging
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from dataclasses import dataclass
import numpy as np
import pickle
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, f1_score, precision_score, recall_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from typing import Union

    
@dataclass
class RegressionMetricArtifact:
    r2_score: float
    mean_squared_error: float
    root_mean_squared_error: float

@dataclass
class ClassificationMetricArtifact:
    accuracy: float
    f1_score: float
    precision: float
    recall: float

@dataclass
class ClusteringMetricArtifact:
    silhouette_score: float
    davies_bouldin_score: float

@dataclass
class ModelTrainerArtifacts:
    train_metric_artifact: Union[RegressionMetricArtifact, ClassificationMetricArtifact, ClusteringMetricArtifact]
    test_metric_artifacts: Union[RegressionMetricArtifact, ClassificationMetricArtifact, ClusteringMetricArtifact]
    model_name: str


def model_train(df,target,models,task_type):
        try:
            X,y = df.drop(columns=[target]),df[target]
            x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
            
            param_grids = {
                # --- Classification ---
                "LogisticRegression": [
                    {  # l2
                        "penalty": ["l2"],
                        "C": [0.01, 0.1, 1, 10, 100],
                        "solver": ["lbfgs", "liblinear", "saga", "newton-cg"],
                        "max_iter": [100, 200, 500],
                        "class_weight": [None, "balanced"]
                    },
                    {  # l1
                        "penalty": ["l1"],
                        "C": [0.01, 0.1, 1, 10, 100],
                        "solver": ["liblinear", "saga"],
                        "max_iter": [100, 200, 500],
                        "class_weight": [None, "balanced"]
                    },
                    {  # elasticnet
                        "penalty": ["elasticnet"],
                        "C": [0.01, 0.1, 1, 10, 100],
                        "solver": ["saga"],
                        "l1_ratio": [0.1, 0.5, 0.9],
                        "max_iter": [100, 200, 500],
                        "class_weight": [None, "balanced"]
                    },
                    {  # no penalty
                        "penalty": [None],
                        "solver": ["lbfgs", "newton-cg", "sag", "saga"],
                        "max_iter": [100, 200, 500],
                        "class_weight": [None, "balanced"]
                    }
                ],

                "RandomForestClassifier": {
                    "n_estimators": [100, 200, 300],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None],
                    "bootstrap": [True, False],
                    "class_weight": [None, "balanced"]
                },

                "SVC": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "degree": [2, 3, 4],
                    "gamma": ["scale", "auto"],
                    "class_weight": [None, "balanced"]
                },

                # --- Regression ---
                "LinearRegression": {
                    "fit_intercept": [True, False],
                    "positive": [False, True]
                },

                "RandomForestRegressor": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None],
                    "bootstrap": [True, False]
                },

                "SVR": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "degree": [2, 3, 4],
                    "gamma": ["scale", "auto"],
                    "epsilon": [0.01, 0.1, 0.5]
                },

                # --- Clustering ---
                "KMeans": {
                    "n_clusters": [2, 3, 5, 8],
                    "init": ["k-means++", "random"],
                    "n_init": [10, 20],
                    "max_iter": [300, 500],
                    "algorithm": ["lloyd", "elkan"]
                },

                "AgglomerativeClustering": {
                    "n_clusters": [2, 3, 5, 8],
                    "linkage": ["ward", "complete", "average", "single"],
                    "affinity": ["euclidean", "l1", "l2", "manhattan", "cosine"]
                },

                "DBSCAN": {
                    "eps": [0.3, 0.5, 0.7],
                    "min_samples": [3, 5, 10],
                    "metric": ["euclidean", "manhattan", "cosine"]
                },

                # --- Anomaly Detection ---
                "IsolationForest": {
                    "n_estimators": [100, 200, 300],
                    "max_samples": ["auto", 0.5, 0.8],
                    "contamination": [0.01, 0.05, 0.1],
                    "max_features": [0.5, 1.0]
                },

                "LocalOutlierFactor": {
                    "n_neighbors": [5, 10, 20],
                    "contamination": [0.01, 0.05, 0.1],
                    "metric": ["euclidean", "manhattan", "cosine"]
                },

                # --- Dimensionality Reduction ---
                "PCA": {
                    "n_components": [2, 3, 5],
                    "svd_solver": ["auto", "full", "randomized"]
                }
            }
            
            model_report,tunned_models=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=param_grids,task_type=task_type)

            best_model_score=max(list(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            
            best_model=tunned_models[best_model_name]
            
            y_train_pred=best_model.predict(x_train)
            
            y_test_pred=best_model.predict(x_test)
            
            if task_type == "clustering":
                train_metrics = get_score(None, y_train_pred, task_type, X_data=x_train)
                test_metrics = get_score(None, y_test_pred, task_type, X_data=x_test)
            else:
                train_metrics = get_score(y_train, y_train_pred, task_type)
                test_metrics = get_score(y_test, y_test_pred, task_type)

            
            model_trainer_artifact=ModelTrainerArtifacts(
                train_metric_artifact=train_metrics,
                test_metric_artifacts=test_metrics,
                model_name=best_model_name
            )
            
            logging.info(f'Model Trainer Artifacts : {model_trainer_artifact}')

            return model_trainer_artifact

        except Exception as e:
            raise Model_trainer_error(e,sys)        

        
def evaluate_models(x_train, y_train, x_test, y_test, models, params, task_type):
        """
    Evaluate models with hyperparameter tuning for classification, regression, or clustering.

    Parameters:
    - x_train, y_train, x_test, y_test: Data splits (for clustering y_train/y_test can be None)
    - models: dict of model_name -> model_instance
    - params: dict of model_name -> param_grid
    - task_type: str, one of ['classification','regression','clustering']

    Returns:
    - report: dict of model_name -> score on test data
    - tuned_models: dict of model_name -> best_model
    """
        report = {}
        tuned_models = {}

        for model_name, model in models.items():
            print(f"Training {model_name}...")

            # Handle parameter grid
            param_grid = params.get(model_name, {})

            # For clustering we don't have y
            if task_type == "clustering":
                # no CV for unsupervised usually, just fit directly
                best_model = model.set_params(**param_grid) if param_grid else model
                best_model.fit(x_train)  # only X
                labels = best_model.labels_ if hasattr(best_model, 'labels_') else best_model.predict(x_train)
                score = silhouette_score(x_train, labels)
            else:
                # supervised → use RandomizedSearchCV
                rcv = RandomizedSearchCV(
                    model, param_distributions=param_grid, cv=5, n_jobs=-1, n_iter=20, random_state=42
                )
                rcv.fit(x_train, y_train)
                best_model = rcv.best_estimator_
                y_pred = best_model.predict(x_test)

                if task_type == "regression":
                    score = r2_score(y_test, y_pred)

                elif task_type == "classification":
                    score = accuracy_score(y_test, y_pred)  # or f1_score for imbalance
                    # e.g. score = f1_score(y_test, y_pred, average='weighted')

                else:
                    raise ValueError("Unsupported task_type. Use 'classification','regression','clustering'")

            report[model_name] = score
            tuned_models[model_name] = best_model

        return report, tuned_models
            
def get_score(y_true, y_pred, task_type, X_data=None):
    """
    Compute metrics based on the ML task type.
    y_true can be None for unsupervised tasks.
    For clustering, pass X_data with the original features.
    """
    try:
        if task_type == "regression":
            model_mse = mean_squared_error(y_true, y_pred)
            model_r2_score = r2_score(y_true, y_pred)
            model_rmse = np.sqrt(model_mse)
            return RegressionMetricArtifact(
                r2_score=model_r2_score,
                mean_squared_error=model_mse,
                root_mean_squared_error=model_rmse
            )

        elif task_type == "classification":
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            precision = precision_score(y_true, y_pred, average="weighted")
            recall = recall_score(y_true, y_pred, average="weighted")
            return ClassificationMetricArtifact(
                accuracy=acc,
                f1_score=f1,
                precision=precision,
                recall=recall
            )

        elif task_type == "clustering":
            if X_data is None:
                raise ValueError("For clustering metrics, X_data (features) must be provided.")
            sil_score = silhouette_score(X_data, y_pred)
            db_score = davies_bouldin_score(X_data, y_pred)
            return ClusteringMetricArtifact(
                silhouette_score=sil_score,
                davies_bouldin_score=db_score
            )

        elif task_type == "anomaly_detection":
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="binary")
            precision = precision_score(y_true, y_pred, average="binary")
            recall = recall_score(y_true, y_pred, average="binary")
            return ClassificationMetricArtifact(
                accuracy=acc,
                f1_score=f1,
                precision=precision,
                recall=recall
            )

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    except Exception as e:
        raise Model_trainer_error(e, sys)




class Model_trainer_error(Exception):
    def __init__(self,error_message,error_details:sys):
        
        self.error_message=error_message
        _,_,_exc_db=error_details.exc_info()
        
        self.file_name=_exc_db.tb_frame.f_code.co_filename
        self.line_no=_exc_db.tb_lineno
        
    def __str__(self):
        return "An error occurred on line {0} of '{1}': {2}".format(self.line_no, self.file_name, self.error_message)
    
    
