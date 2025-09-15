# src/main.py

import pandas as pd
import os
import json
from data_ingestion import DataIngestion
from data_processing import process_data, to_camel_case 
from eda import perform_eda
from feature_engineering import preprocess_features
from model_selection import get_models

def run_pipeline(file_path: str, target_variable: str):
    """
    Executes the full AutoML pipeline from ingestion to model selection.
    """
    all_results = {}
    try:
        # --- Step 1: Data Ingestion ---
        ingestion = DataIngestion(file_path=file_path, target_variable=target_variable)
        if ingestion.df is None:
            raise ValueError("Data could not be loaded.")
        all_results['ingestion'] = {
            'original_shape': ingestion.df.shape,
            'head': ingestion.df.head().to_dict()
        }
        df = ingestion.df

        # --- Step 2: Data Processing ---
        cleaned_df = process_data(df)
        all_results['processing'] = {
            'cleaned_shape': cleaned_df.shape,
            'head': cleaned_df.head().to_dict()
        }
        
        # --- THIS IS THE CRITICAL FIX ---
        # Convert the target variable string to match the new column names, only if it exists
        if target_variable:
            target_variable = to_camel_case(target_variable)
            if target_variable not in cleaned_df.columns:
                 raise ValueError(f"Target variable '{target_variable}' not found after processing.")
        # ----------------------------------------

        # --- Step 3: Exploratory Data Analysis (EDA) ---
        eda_results = perform_eda(cleaned_df, target_variable=target_variable)
        all_results['eda'] = eda_results

        # --- Step 4: Feature Engineering ---
        # Only run feature engineering if there is a target
        if target_variable:
            processed_df, scaler, encoder = preprocess_features(cleaned_df, target_variable=target_variable)
            all_results['feature_engineering'] = {
                'processed_shape': processed_df.shape,
                'head': processed_df.head().to_dict(),
                'scaler': scaler,
                'encoder': encoder
            }
        else: # For unsupervised tasks, the data for clustering is just the cleaned data
            processed_df = cleaned_df
            all_results['feature_engineering'] = {
                 'processed_shape': processed_df.shape,
                 'head': processed_df.head().to_dict(),
                 'message': 'No feature engineering applied for unsupervised task.'
            }


        # --- Step 5: Model Selection ---
        problem_type = eda_results.get('insights', {}).get('problem_type', 'unsupervised').lower()
        models = get_models(problem_type)
        all_results['model_selection'] = {
            'problem_type': problem_type,
            'models_selected': list(models.keys())
        }
        
    except Exception as e:
        all_results['error'] = str(e)

    return all_results

if __name__ == "__main__":
    # This block allows you to test the pipeline directly
    if not os.path.exists('data'):
        os.makedirs('data')
    pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['X', 'Y', 'X', 'Z', 'Y'],
        'Target': [0, 1, 0, 1, 0]
    }).to_csv('data/dummy_data.csv', index=False)

    results = run_pipeline(file_path='data/dummy_data.csv', target_variable='Target')
    print("--- Pipeline Finished Successfully! ---")
    # Use default=str to handle non-serializable objects for printing
    print(json.dumps(results, indent=2, default=str))

