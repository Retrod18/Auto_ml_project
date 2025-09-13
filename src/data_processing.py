import pandas as pd
import numpy as np
import re

def to_camel_case(column_name: str) -> str:
    """Convert a column name to CamelCase."""
    cleaned = re.sub(r'[^a-zA-Z0-9]', ' ', column_name)
    return ''.join(word.capitalize() for word in cleaned.split())


def convert_columns_to_camel_case(df: pd.DataFrame) -> pd.DataFrame:
    """Converts DataFrame column names to CamelCase."""
    df = df.copy()
    df.columns = [to_camel_case(col) for col in df.columns]
    return df


def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """Dynamically convert column types: numeric where possible, datetime where likely."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            converted_numeric = pd.to_numeric(df[col], errors="coerce")
            if converted_numeric.notna().sum() / len(df[col]) > 0.8:
                df[col] = converted_numeric
                continue

            # Datetime conversion (if majority of values look like dates)
            non_null_values = df[col].dropna()
            date_like_count = non_null_values.astype(str).str.match(
                r"^\d{4}-\d{2}-\d{2}|^\d{1,2}/\d{1,2}/\d{2,4}$"
            ).sum()

            if len(non_null_values) > 0 and date_like_count / len(non_null_values) > 0.8:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by:
    1. Handling missing values dynamically
    2. Removing duplicates on Percentage
    3. Handling outliers (IQR method)
    4. Fixing data types
    """
    df = df.copy()


# Drop a Column: > 50% missing
# Mode Imputation: < 15% missing
# Assign "Unknown": 15% - 50% missing
    
    # Calculate percentage of missing values for each column
    missing_percentage = df.isnull().sum() / len(df)

    # Drop columns with more than 50% missing data
    cols_to_drop = missing_percentage[missing_percentage > 0.50].index
    if not cols_to_drop.empty:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped columns with >50% missing values: {list(cols_to_drop)}")

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            missing_pct = df[col].isnull().sum() / len(df)
            
            if pd.api.types.is_numeric_dtype(df[col]):
                
                df[col] = df[col].fillna(df[col].median())
            else:
                if missing_pct < 0.15:
                    # For categorical data with low missing values,  mode imputation.
                    try:
                        df[col] = df[col].fillna(df[col].mode()[0])
                    except IndexError:
                        # If mode is empty  (NaN) then 'Unknown'
                        df[col] = df[col].fillna("Unknown")
                else: # 15% to 50% missing
                    # For categorical data with moderate missing values, use 'Unknown'.
                    df[col] = df[col].fillna("Unknown")

    # Remove Duplicates
    df = df.drop_duplicates()

    # Outliers (IQR) Handling 
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    # Data type Conversion
    df = convert_column_types(df)

    return df

def process_data(df: pd.DataFrame):
    """
    Accepts a DataFrame (from data ingestion), applies:
    1. Column name conversion
    2. Data cleaning
    Returns:
        original_shape: tuple (rows, columns)
        cleaned_df: cleaned DataFrame
        cleaned_shape: tuple (rows, columns)
    """
    print(f"Original shape: {df.shape}")
    
    df = convert_columns_to_camel_case(df)
    df = clean_data(df)
    
    print(f"Shape after cleaning: {df.shape}")
    return df





