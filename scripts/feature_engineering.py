import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import sys

# Add the current directory to the path if it's not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from scripts.minio_utils import save_df_to_minio

def identify_column_types(df):
    """Identify numeric and categorical columns in the dataframe"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return numeric_cols, categorical_cols

def create_preprocessing_pipeline(numeric_cols, categorical_cols):
    """Create a scikit-learn preprocessing pipeline"""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

def engineer_features(df, target_col=None, dataset_name=None, save_to_minio=False):
    """Engineer features for a dataset and optionally save to MinIO"""
    # Handle missing values in the target column if present
    if target_col and target_col in df.columns:
        df = df.dropna(subset=[target_col])
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()
        y = None
    
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(X)
    
    # Create and fit the preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols)
    X_processed = preprocessor.fit_transform(X)
    
    # Convert processed features to DataFrame (if not sparse)
    try:
        # Get feature names from pipeline
        numeric_features = numeric_cols
        
        # Get categorical feature names after one-hot encoding
        try:
            cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
        except:
            cat_features = []
        
        # Combine all feature names
        feature_names = list(numeric_features) + list(cat_features)
        
        # Convert to DataFrame
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    except:
        # Fall back to numpy array if conversion fails
        X_processed_df = pd.DataFrame(X_processed)
    
    # Add target column back if it exists
    if y is not None:
        X_processed_df[target_col] = y.values
    
    # Save to MinIO if requested
    if save_to_minio and dataset_name:
        save_df_to_minio(X_processed_df, dataset_name, f"{dataset_name}_processed.csv")
    
    return X_processed_df, preprocessor

def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    from sklearn.model_selection import train_test_split
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state) 