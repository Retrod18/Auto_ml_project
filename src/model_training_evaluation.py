# In model_training_evaluation.py

import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, silhouette_score

def train_and_evaluate_models(train_df, test_df, target_variable, problem_type, models):
    """
    Trains and evaluates a dictionary of models based on the problem type.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The testing dataset.
        target_variable (str or None): The name of the target variable.
        problem_type (str): 'classification', 'regression', or 'clustering'.
        models (dict): A dictionary of model instances to train.

    Returns:
        dict: A dictionary containing evaluation results.
    """
    all_model_reports = {}
    best_model_name = None
    best_model_score = -1 # Use -1 for regression/classification, needs adjustment for clustering

    # --- THIS IS THE CRITICAL FIX ---
    # Separate features (X) and target (y) based on whether a target_variable is provided.
    if target_variable and target_variable in train_df.columns:
        # Supervised path
        X_train = train_df.drop(columns=[target_variable])
        y_train = train_df[target_variable]
        X_test = test_df.drop(columns=[target_variable])
        y_test = test_df[target_variable]
    else:
        # Unsupervised path (target_variable is None)
        X_train = train_df
        y_train = None  # No target
        X_test = test_df
        y_test = None   # No target

    for name, model in models.items():
        try:
            # Train the model
            # Note: scikit-learn clustering models ignore the `y` argument if provided.
            model.fit(X_train, y_train)
            
            # --- Evaluation Logic ---
            report = {}
            if problem_type == 'classification':
                preds = model.predict(X_test)
                score = accuracy_score(y_test, preds)
                report['accuracy'] = score
            
            elif problem_type == 'regression':
                preds = model.predict(X_test)
                score = r2_score(y_test, preds)
                report['r2_score'] = score
            
            elif problem_type == 'clustering':
                # Predict cluster labels for each point
                preds = model.fit_predict(X_train)
                # Silhouette score requires at least 2 clusters
                if len(set(preds)) > 1:
                    score = silhouette_score(X_train, preds)
                    report['silhouette_score'] = score
                else:
                    score = -1 # Cannot compute score for a single cluster
                    report['silhouette_score'] = 'N/A (1 cluster found)'

            all_model_reports[name] = report

            # Update best model
            if score > best_model_score:
                best_model_score = score
                best_model_name = name

        except Exception as e:
            all_model_reports[name] = {'error': str(e)}

    return {
        'best_model_name': best_model_name,
        'best_model_score': best_model_score,
        'all_model_reports': all_model_reports
    }