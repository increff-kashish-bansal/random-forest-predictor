import pytest
import pandas as pd
import numpy as np
from utils.mapping_helpers import suggest_mappings
from utils.feature_engineering import derive_features

def test_suggest_mappings():
    """Test that suggest_mappings correctly matches columns using fuzzy matching."""
    # Test data
    uploaded_cols = ['Store_ID', 'Product_Category', 'Sale_Date', 'Units_Sold', 'Total_Revenue', 'Discount_Amount']
    required_cols = ['store', 'category', 'date', 'qty_sold', 'revenue', 'disc_value']
    
    # Get mappings
    mappings = suggest_mappings(uploaded_cols, required_cols)
    
    # Assertions
    assert len(mappings) == len(required_cols)
    assert mappings['store'] == 'Store_ID'
    assert mappings['category'] == 'Product_Category'
    assert mappings['date'] == 'Sale_Date'
    assert mappings['qty_sold'] == 'Units_Sold'
    assert mappings['revenue'] == 'Total_Revenue'
    assert mappings['disc_value'] == 'Discount_Amount'

def test_apply_mapping():
    """Test that column mapping correctly renames columns to match schema."""
    # Test data
    df = pd.DataFrame({
        'Store_ID': [1, 2, 3],
        'Product_Category': ['A', 'B', 'C'],
        'Sale_Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Units_Sold': [10, 20, 30],
        'Total_Revenue': [100, 200, 300],
        'Discount_Amount': [10, 20, 30]
    })
    
    # Mapping dictionary
    mapping = {
        'Store_ID': 'store',
        'Product_Category': 'category',
        'Sale_Date': 'date',
        'Units_Sold': 'qty_sold',
        'Total_Revenue': 'revenue',
        'Discount_Amount': 'disc_value'
    }
    
    # Apply mapping
    df_mapped = df.rename(columns=mapping)
    
    # Assertions
    expected_cols = ['store', 'category', 'date', 'qty_sold', 'revenue', 'disc_value']
    assert list(df_mapped.columns) == expected_cols

def test_derive_features():
    """Test that derive_features creates the expected number of features with no NaNs."""
    # Create dummy sales data
    df_sales = pd.DataFrame({
        'store': [1, 1, 2, 2],
        'date': pd.date_range('2024-01-01', periods=4),
        'qty_sold': [10, 20, 30, 40],
        'revenue': [100, 200, 300, 400],
        'disc_value': [10, 20, 30, 40]
    })
    
    # Create dummy stores data
    df_stores = pd.DataFrame({
        'id': [1, 2],
        'channel': ['online', 'offline'],
        'city': ['NY', 'LA'],
        'region': ['East', 'West'],
        'is_online': [1, 0],
        'store_area': [1000, 2000]
    })
    
    # Derive features
    X, y = derive_features(df_sales, df_stores)
    
    # Assertions
    assert X.shape[1] == 53  # Expected number of features
    assert not X.isna().any().any()  # No NaN values
    assert len(y) == len(df_sales)  # Target variable length matches input
    assert not y.isna().any()  # No NaN values in target
    
    # Check specific feature groups
    assert 'day_of_week' in X.columns
    assert 'is_weekend' in X.columns
    assert 'channel_online' in X.columns
    assert 'channel_offline' in X.columns
    assert 'area_very_small' in X.columns
    assert 'rolling_mean_revenue_7d' in X.columns
    assert 'mom_growth' in X.columns 