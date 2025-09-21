import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, Optional

def preprocess_features(
    df: pd.DataFrame,
    target_variable: Optional[str] = None,
    scaler_obj: Optional[StandardScaler] = None,
    encoder_obj: Optional[OneHotEncoder] = None
) -> Tuple[pd.DataFrame, Optional[StandardScaler], Optional[OneHotEncoder]]:
    """
    Preprocesses data by encoding categorical and scaling numerical features.
    Handles supervised (with target) and unsupervised (no target) cases robustly.
    """
    df_copy = df.copy()

    # 1. Separate target variable if it exists
    if target_variable and target_variable in df_copy.columns:
        y = df_copy[target_variable]
        X = df_copy.drop(columns=[target_variable])
    else:
        X = df_copy
        y = None

    # 2. Identify column types
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 3. Process Numerical Columns
    if numeric_cols:
        if scaler_obj is None:
            scaler_obj = StandardScaler()
            X[numeric_cols] = scaler_obj.fit_transform(X[numeric_cols])
        else:
            X[numeric_cols] = scaler_obj.transform(X[numeric_cols])

    # 4. Process Categorical Columns
    if categorical_cols:
        if encoder_obj is None:
            encoder_obj = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = encoder_obj.fit_transform(X[categorical_cols])
        else:
            encoded_data = encoder_obj.transform(X[categorical_cols])
        
        try: # For newer scikit-learn versions
            encoded_feature_names = encoder_obj.get_feature_names_out(categorical_cols)
        except AttributeError: # For older scikit-learn versions
            encoded_feature_names = encoder_obj.get_feature_names(categorical_cols)
        
        encoded_df = pd.DataFrame(encoded_data, index=X.index, columns=encoded_feature_names)
        
        X = X.drop(columns=categorical_cols)
        X = pd.concat([X, encoded_df], axis=1)

    # 5. Re-attach target variable for supervised learning evaluation
    if y is not None:
        X[target_variable] = y.values

    return X, scaler_obj, encoder_obj