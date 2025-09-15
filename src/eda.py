import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import List, Dict, Any, Union

def get_problem_type(df: pd.DataFrame, target_variable: str) -> str:
    if target_variable not in df.columns:
        return "Unsupervised"
    # Check if the target is categorical or has few unique values
    if pd.api.types.is_categorical_dtype(df[target_variable]) or df[target_variable].nunique() < 20:
        return "Classification"
    else:
        return "Regression"

def generate_plot_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    return img_base64

def perform_eda(df: pd.DataFrame, target_variable: Union[str, None] = None) -> Dict[str, Any]:
    insights = {}
    plots = {}
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    print("\n=== EDA: Data Overview ===")
    insights['shape'] = df.shape
    # --- FIX 1: Convert dtypes to strings for JSON compatibility ---
    insights['datatypes'] = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    insights['missing_values'] = df.isnull().sum().to_dict()
    insights['statistical_summary'] = df.describe(include='all').to_dict()
    
    print("DataFrame shape:", insights['shape'])
    
    if target_variable:
        problem_type = get_problem_type(df, target_variable)
        insights['problem_type'] = problem_type
        print(f"\nDetected problem type: {problem_type}")
    else:
        insights['problem_type'] = "Unsupervised"
        print("\nNo target variable provided. Assuming unsupervised learning.")

    # Histograms for Numerical Features
    if numerical_cols:
        print("\n=== EDA: Histograms for Numerical Features ===")
        num_plots = len(numerical_cols)
        fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 5 * num_plots))
        if not isinstance(axes, (np.ndarray, list)):
            axes = [axes]
        for i, col in enumerate(numerical_cols):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
        plt.tight_layout()
        plots['histograms'] = generate_plot_base64(fig)
        
    # Count Plots for Categorical Features
    if categorical_cols:
        print("\n=== EDA: Count Plots for Categorical Features ===")
        cat_plots = len(categorical_cols)
        fig, axes = plt.subplots(nrows=cat_plots, ncols=1, figsize=(10, 5 * cat_plots))
        if not isinstance(axes, (np.ndarray, list)):
            axes = [axes]
        for i, col in enumerate(categorical_cols):
            sns.countplot(y=col, data=df, ax=axes[i], order=df[col].value_counts().index)
            axes[i].set_title(f'Count Plot of {col}', fontsize=14)
        plt.tight_layout()
        plots['count_plots'] = generate_plot_base64(fig)
        
    # Correlation Heatmap for numerical features
    if len(numerical_cols) > 1:
        print("\n=== EDA: Correlation Heatmap ===")
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=16)
        plots['correlation_heatmap'] = generate_plot_base64(fig)
        
        # --- FIX 2: Convert tuple keys to strings for JSON compatibility ---
        corr_abs = corr_matrix.abs().unstack()
        corr_pairs = corr_abs[corr_abs < 1].sort_values(ascending=False)
        strongest_corr_dict = corr_pairs.head(5).to_dict()
        insights['strongest_correlations'] = {f"{k[0]} & {k[1]}": v for k, v in strongest_corr_dict.items()}

    # Target Distribution Plot
    if target_variable and target_variable in df.columns:
        print(f"\n=== EDA: Distribution of Target Variable '{target_variable}' ===")
        fig, ax = plt.subplots(figsize=(8, 6))
        problem_type = get_problem_type(df, target_variable)
        if problem_type == "Classification":
            sns.countplot(x=target_variable, data=df, ax=ax)
        else: # Regression
            sns.histplot(data=df, x=target_variable, kde=True, ax=ax)
        ax.set_title(f'Target Variable Distribution ({problem_type})', fontsize=14)
        plt.tight_layout()
        plots['target_distribution'] = generate_plot_base64(fig)

    eda_results = {
        'insights': insights,
        'plots': plots
    }

    return eda_results
