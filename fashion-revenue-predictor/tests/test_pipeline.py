import pytest
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from train_model import train_model
from predictor import predict_and_explain
from utils.feature_engineering import derive_features

def test_full_train_pipeline():
    """Test the complete training pipeline including model and feature list saving."""
    # Load sample data
    df_sales = pd.read_csv("data/sales_sample.csv")
    df_stores = pd.read_csv("data/stores_sample.csv")
    
    # Ensure models directory exists
    Path("models").mkdir(exist_ok=True)
    
    # Run training
    iteration_log = train_model(df_sales, df_stores)
    
    # Assertions
    assert len(iteration_log) >= 1  # At least one iteration
    assert 'r2' in iteration_log[0]  # R² score present
    assert 'importances' in iteration_log[0]  # Feature importances present
    
    # Check model file
    assert os.path.exists("models/brandA_model.pkl")
    
    # Check feature list
    assert os.path.exists("models/brandA_features.json")
    with open("models/brandA_features.json", 'r') as f:
        features = json.load(f)
    assert len(features) > 0  # Non-empty feature list

def test_prediction():
    """Test prediction pipeline with a single-row input."""
    # Load sample data for reference
    df_sales = pd.read_csv("data/sales_sample.csv")
    df_stores = pd.read_csv("data/stores_sample.csv")
    
    # Create single-row input
    input_dict = {
        'store': df_stores['id'].iloc[0],
        'sku_id': 'SKU_PLACEHOLDER',
        'date': pd.Timestamp('2024-01-01'),
        'qty_sold': 0,
        'revenue': 0,
        'disc_value': 10,
        'id': df_stores['id'].iloc[0],
        'channel': df_stores['channel'].iloc[0],
        'city': df_stores['city'].iloc[0],
        'region': df_stores['region'].iloc[0],
        'is_online': df_stores['is_online'].iloc[0],
        'store_area': df_stores['store_area'].iloc[0]
    }
    
    input_df_sales = pd.DataFrame([input_dict], columns=df_sales.columns)
    input_df_stores = pd.DataFrame([df_stores.iloc[0]], columns=df_stores.columns)
    
    # Derive features
    X_pred, _ = derive_features(input_df_sales, input_df_stores)
    
    # Make prediction
    results = predict_and_explain(X_pred)
    
    # Assertions
    assert isinstance(results['p50'][0], float)  # p50 is a float
    assert len(results['shap_values']) == 1  # One row of SHAP values
    assert len(results['top_5_features']) == 5  # Top 5 features
    assert all(isinstance(x, tuple) for x in results['top_5_features'])  # Each feature is a tuple
    assert all(len(x) == 2 for x in results['top_5_features'])  # Each tuple has 2 elements
    
    # Check prediction ranges
    assert results['p10'][0] <= results['p50'][0] <= results['p90'][0]  # Percentiles in order 