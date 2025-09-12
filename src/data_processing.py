import pandas as pd
import re

def to_camel_case(column_name: str) -> str:
    """
    Convert a column name to CamelCase.
    Example: "first name" -> "FirstName"
             "order_date" -> "OrderDate"
             "Customer id" -> "CustomerId"
    """
    # Replace non-alphanumeric characters with space
    cleaned = re.sub(r'[^a-zA-Z0-9]', ' ', column_name)
    # Split by space, capitalize each word, and join
    camel_case = ''.join(word.capitalize() for word in cleaned.split())
    return camel_case

def convert_columns_to_camel_case(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV/Excel file, converts column names to CamelCase,
    and returns the transformed DataFrame.
    """
    # Auto-detect file type
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please use CSV or Excel.")

    # Convert column names
    df.columns = [to_camel_case(col) for col in df.columns]
    return df


if __name__ == "__main__":
    file_path = "../data/sample.csv"  
    transformed_df = convert_columns_to_camel_case(file_path)
    print("✅ Columns converted to CamelCase:")
    print(transformed_df.head())
    print("New Columns:", transformed_df.columns.tolist())