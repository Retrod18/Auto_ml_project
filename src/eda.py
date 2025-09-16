import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, Any, Union

# This function defines the logic, it does not need to import itself.

def get_problem_type(df: pd.DataFrame, target_variable: str) -> str:
    """Determines if the problem is regression or classification based on the target variable."""
    if not target_variable or target_variable not in df.columns:
        return "Unsupervised"
    
    target_series = df[target_variable]
    
    # Heuristic: If the column is numeric and has many unique values, it's likely regression.
    if pd.api.types.is_numeric_dtype(target_series):
        if target_series.nunique() > 25 or pd.api.types.is_float_dtype(target_series):
            return "Regression"
    
    return "Classification"

def generate_plot_base64(fig: plt.Figure) -> str:
    """Converts a matplotlib figure to a Base64 encoded string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    return img_base64

def _format_for_json(data: dict) -> dict:
    """Helper to convert complex data types in a dictionary to JSON-serializable formats."""
    formatted_data = {}
    for key, value in data.items():
        if isinstance(key, tuple):
            key = f"{key[0]} & {key[1]}"
        
        if isinstance(value, np.dtype):
            value = str(value)
        elif isinstance(value, pd.DataFrame):
            value = value.to_dict()
        elif isinstance(value, pd.Series):
             value = value.to_dict()
        
        formatted_data[key] = value
    return formatted_data

def perform_eda(df: pd.DataFrame, target_variable: Union[str, None] = None) -> Dict[str, Any]:
    """Performs EDA and returns a dictionary of insights and plot strings."""
    insights = {}
    plots = {}
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    insights['shape'] = df.shape
    insights['datatypes'] = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    insights['missing_values'] = df.isnull().sum().to_dict()
    insights['statistical_summary'] = df.describe(include='all').to_dict()
    
    problem_type = get_problem_type(df, target_variable)
    insights['problem_type'] = problem_type

    if numerical_cols:
        try:
            fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(8, 4 * len(numerical_cols)))
            if len(numerical_cols) == 1: axes = [axes]
            for ax, col in zip(axes, numerical_cols):
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
            plt.tight_layout()
            plots['histograms'] = generate_plot_base64(fig)
        except Exception as e:
            plots['histograms_error'] = str(e)

    if categorical_cols:
        try:
            fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(8, 4 * len(categorical_cols)))
            if len(categorical_cols) == 1: axes = [axes]
            for ax, col in zip(axes, categorical_cols):
                sns.countplot(y=col, data=df, ax=ax, order=df[col].value_counts().index[:15])
                ax.set_title(f'Count of {col}')
            plt.tight_layout()
            plots['count_plots'] = generate_plot_base64(fig)
        except Exception as e:
            plots['count_plots_error'] = str(e)
            
    if len(numerical_cols) > 1:
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numerical_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')
            plots['correlation_heatmap'] = generate_plot_base64(fig)
            
            corr_pairs = corr.unstack().sort_values(kind="quicksort", ascending=False)
            unique_pairs = corr_pairs[corr_pairs < 1.0]
            insights['strongest_correlations'] = _format_for_json(unique_pairs.head(5).to_dict())
        except Exception as e:
            plots['correlation_heatmap_error'] = str(e)

    if target_variable and target_variable in df.columns:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            if problem_type == "Classification":
                sns.countplot(y=target_variable, data=df, ax=ax, order=df[target_variable].value_counts().index[:15])
            else:
                sns.histplot(df[target_variable], kde=True, ax=ax)
            ax.set_title(f'Distribution of Target: {target_variable}')
            plt.tight_layout()
            plots['target_distribution'] = generate_plot_base64(fig)
        except Exception as e:
            plots['target_distribution_error'] = str(e)

    return {'insights': insights, 'plots': plots}

