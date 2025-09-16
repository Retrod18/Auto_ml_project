import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, Optional

def preprocess_features(
    df: pd.DataFrame,
    target_variable: Optional[str] = None,
    scaler_obj: StandardScaler = None,
    encoder_obj: OneHotEncoder = None
) -> Tuple[pd.DataFrame, Optional[StandardScaler], Optional[OneHotEncoder]]:
    """
    Preprocesses data by encoding categorical and scaling numerical features.
    Handles supervised (with target) and unsupervised (no target) cases robustly.
    """
    df_processed = df.copy()

    if target_variable and target_variable in df_processed.columns:
        y = df_processed[target_variable]
        X = df_processed.drop(columns=[target_variable])
    else:
        X = df_processed
        y = None

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    X_scaled_df = pd.DataFrame(index=X.index)
    X_encoded_df = pd.DataFrame(index=X.index)

    if categorical_cols:
        if encoder_obj is None:
            encoder_obj = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            X_encoded_array = encoder_obj.fit_transform(X[categorical_cols])
        else:
            X_encoded_array = encoder_obj.transform(X[categorical_cols])
        
        encoded_cols = encoder_obj.get_feature_names_out(categorical_cols)
        X_encoded_df = pd.DataFrame(X_encoded_array, index=X.index, columns=encoded_cols)

    if numeric_cols:
        if scaler_obj is None:
            scaler_obj = StandardScaler()
            X_scaled_array = scaler_obj.fit_transform(X[numeric_cols])
        else:
            X_scaled_array = scaler_obj.transform(X[numeric_cols])
        
        X_scaled_df = pd.DataFrame(X_scaled_array, index=X.index, columns=numeric_cols)

    X_final = pd.concat([X_scaled_df, X_encoded_df], axis=1)
    
    if y is not None:
        X_final[target_variable] = y

    return X_final, scaler_obj, encoder_obj

