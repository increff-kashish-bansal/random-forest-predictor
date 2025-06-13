import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import json
import logging
from pathlib import Path
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log'),
        logging.StreamHandler()
    ]
)

def derive_features(df_sales: pd.DataFrame, df_stores: pd.DataFrame, is_prediction: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Derive features from sales and stores data.
    
    Args:
        df_sales: DataFrame with sales data
        df_stores: DataFrame with store information
        is_prediction: Whether this is being called for prediction (single row) or training
        
    Returns:
        Tuple of (X: feature matrix, y: target variable)
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    logging.info("Starting feature engineering...")
    logging.info(f"Input shapes - Sales: {df_sales.shape}, Stores: {df_stores.shape}")
    
    # Ensure date is datetime
    df_sales['date'] = pd.to_datetime(df_sales['date'])
    
    # Ensure store IDs are strings and validate
    df_sales['store'] = df_sales['store'].astype(str)
    df_stores['id'] = df_stores['id'].astype(str)
    
    # Log store IDs for debugging
    logging.info(f"Sales store IDs: {df_sales['store'].unique()}")
    logging.info(f"Stores IDs: {df_stores['id'].unique()}")
    
    # Validate required columns
    required_sales_cols = ['store', 'date', 'qty_sold', 'revenue', 'disc_perc']
    required_stores_cols = ['id', 'channel', 'city', 'region', 'store_area', 'is_online']
    
    missing_sales_cols = [col for col in required_sales_cols if col not in df_sales.columns]
    missing_stores_cols = [col for col in required_stores_cols if col not in df_stores.columns]
    
    if missing_sales_cols:
        raise ValueError(f"Missing required columns in sales data: {missing_sales_cols}")
    if missing_stores_cols:
        raise ValueError(f"Missing required columns in stores data: {missing_stores_cols}")
    
    # Merge store data
    df = pd.merge(df_sales, df_stores, left_on='store', right_on='id', how='left')
    logging.info(f"Merged data shape: {df.shape}")
    
    # Validate merge
    if len(df) != len(df_sales):
        logging.error(f"Merge resulted in different number of rows: {len(df)} vs {len(df_sales)}")
        raise ValueError(f"Merge resulted in different number of rows: {len(df)} vs {len(df_sales)}")
    
    # Check for missing store data
    missing_stores = df[df['channel'].isna()]['store'].unique()
    if len(missing_stores) > 0:
        logging.error(f"Missing store data for stores: {missing_stores}")
        raise ValueError(f"Missing store data for stores: {missing_stores}")
    
    # Log column types after merge
    logging.info("Column types after merge:")
    logging.info(df.dtypes)
    
    # Initialize feature list
    features = []
    
    # 1. Temporal Features
    logging.info("Generating temporal features...")
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['lead_time_days'] = (df['date'] - pd.Timestamp.today()).dt.days
    
    # Add cyclical encoding for temporal features
    logging.info("Adding cyclical encoding for temporal features...")
    
    # Month cyclical encoding (12 months)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Day of week cyclical encoding (7 days)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Day of month cyclical encoding (assuming max 31 days)
    df['day_of_month'] = df['date'].dt.day
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    
    # Add interaction features for seasonality
    df['month_weekend'] = df['month_sin'] * df['is_weekend']  # Weekend effect varies by month
    df['month_day'] = df['month_sin'] * df['day_of_week_sin']  # Day effect varies by month
    
    # Update features list with new cyclical features
    features.extend([
        'day_of_week', 'month', 'quarter', 'year', 'is_weekend', 
        'is_month_start', 'is_month_end', 'lead_time_days',
        'month_sin', 'month_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'day_of_month_sin', 'day_of_month_cos',
        'month_weekend', 'month_day'
    ])
    
    # 2. Store/Location Features
    logging.info("Generating store/location features...")
    
    # Define all possible categorical values
    if not is_prediction:
        # During training, get unique values from data
        all_categories = {
            'channel': ['ONLINE', 'LFR_LIFESTYLE'],
            'region': df['region'].unique().tolist(),
            'city': df['city'].unique().tolist()
        }
        # Save categories for prediction
        with open('models/categories.json', 'w') as f:
            json.dump(all_categories, f)
    else:
        # During prediction, load categories from training
        try:
            with open('models/categories.json', 'r') as f:
                all_categories = json.load(f)
        except FileNotFoundError:
            raise ValueError("Categories file not found. Please train the model first.")
    
    # One-hot encode categorical features
    for col, possible_values in all_categories.items():
        # Create dummy variables for all possible values
        for value in possible_values:
            col_name = f"{col}_{value}"
            df[col_name] = (df[col] == value).astype(int)
            features.append(col_name)
    
    # Store area features
    df['store_area'] = df['store_area'].fillna(df['store_area'].median())
    df['store_area_bucket'] = pd.qcut(df['store_area'], q=5, labels=False, duplicates='drop')
    features.extend(['store_area', 'store_area_bucket', 'is_online'])
    
    # 3. Discount Features
    logging.info("Generating discount features...")
    # Use disc_perc directly
    df['discount_pct'] = df['disc_perc'].fillna(0)
    # Ensure discount percentage is between 0 and 1
    df['discount_pct'] = df['discount_pct'].clip(0, 1)
    df['is_discounted'] = (df['discount_pct'] > 0).astype(int)
    features.extend(['discount_pct', 'is_discounted'])
    
    # 4. Historical Features
    logging.info("Generating historical features...")
    if not is_prediction:
        # During training, calculate store-level statistics
        store_stats = df.groupby('store').agg({
            'revenue': ['mean', 'std', 'median'],
            'qty_sold': ['mean', 'std', 'median']
        }).fillna(0)
        store_stats.columns = ['_'.join(col).strip() for col in store_stats.columns.values]
        
        # Calculate store-month seasonal indices
        store_month_stats = df.groupby(['store', 'month']).agg({
            'revenue': ['mean', 'std']
        }).fillna(0)
        store_month_stats.columns = ['store_month_revenue_mean', 'store_month_revenue_std']
        
        # Save store stats for prediction
        store_stats.to_json('models/store_stats.json')
        store_month_stats.to_json('models/store_month_stats.json')
        
        # Add store stats to features
        store_stats_features = store_stats.columns.tolist()
        store_month_features = store_month_stats.columns.tolist()
        features.extend(store_stats_features)
        features.extend(store_month_features)
        
        # Merge store stats
        df = df.merge(store_stats, on='store', how='left')
        df = df.merge(store_month_stats, on=['store', 'month'], how='left')
    else:
        # During prediction, load store stats
        try:
            store_stats = pd.read_json('models/store_stats.json')
            store_month_stats = pd.read_json('models/store_month_stats.json')
            
            # Ensure store IDs are strings
            store_stats.index = store_stats.index.astype(str)
            store_month_stats.index = store_month_stats.index.get_level_values(0).astype(str)
            
            # Reset index to make store ID a column
            store_stats = store_stats.reset_index()
            store_stats = store_stats.rename(columns={'index': 'store'})
            
            store_month_stats = store_month_stats.reset_index()
            store_month_stats = store_month_stats.rename(columns={'level_0': 'store', 'level_1': 'month'})
            
            # Add store stats to features
            store_stats_features = store_stats.columns.drop('store').tolist()
            store_month_features = store_month_stats.columns.drop(['store', 'month']).tolist()
            features.extend(store_stats_features)
            features.extend(store_month_features)
            
            # Merge store stats
            df = df.merge(store_stats, on='store', how='left')
            df = df.merge(store_month_stats, on=['store', 'month'], how='left')
        except:
            # If no stats available, create empty stats
            store_stats_features = [
                'revenue_mean', 'revenue_std', 'revenue_median',
                'qty_sold_mean', 'qty_sold_std', 'qty_sold_median'
            ]
            store_month_features = [
                'store_month_revenue_mean', 'store_month_revenue_std'
            ]
            features.extend(store_stats_features)
            features.extend(store_month_features)
            
            # Add empty stats
            for feat in store_stats_features:
                df[feat] = 0
            for feat in store_month_features:
                df[feat] = 0
    
    # 5. Lightweight Lag Proxies
    logging.info("Generating lightweight lag proxies...")
    if not is_prediction:
        # During training, calculate lag features
        # Sort by store and date to ensure correct lag calculation
        df = df.sort_values(['store', 'date'])
        
        # Calculate lag features
        df['lag_1'] = df.groupby('store')['revenue'].shift(1)
        df['lag_7'] = df.groupby('store')['revenue'].shift(7)
        
        # Calculate rolling means
        df['rolling_mean_7d'] = df.groupby('store')['revenue'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df['rolling_mean_30d'] = df.groupby('store')['revenue'].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        )
        
        # Add lag features to feature list
        features.extend(['lag_1', 'lag_7', 'rolling_mean_7d', 'rolling_mean_30d'])
        
        # Save lag parameters for prediction
        lag_params = {
            'lag_periods': [1, 7],
            'rolling_windows': [7, 30]
        }
        with open('models/lag_params.json', 'w') as f:
            json.dump(lag_params, f)
    else:
        # During prediction, load lag parameters
        try:
            with open('models/lag_params.json', 'r') as f:
                lag_params = json.load(f)
            
            # Sort by store and date
            df = df.sort_values(['store', 'date'])
            
            # Calculate lag features
            df['lag_1'] = df.groupby('store')['revenue'].shift(1)
            df['lag_7'] = df.groupby('store')['revenue'].shift(7)
            
            # Calculate rolling means
            df['rolling_mean_7d'] = df.groupby('store')['revenue'].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            df['rolling_mean_30d'] = df.groupby('store')['revenue'].transform(
                lambda x: x.rolling(30, min_periods=1).mean()
            )
            
            # Add lag features to feature list
            features.extend(['lag_1', 'lag_7', 'rolling_mean_7d', 'rolling_mean_30d'])
        except FileNotFoundError:
            # If no lag parameters available, use zeros
            lag_features = ['lag_1', 'lag_7', 'rolling_mean_7d', 'rolling_mean_30d']
            for feature in lag_features:
                df[feature] = 0
            features.extend(lag_features)
    
    # Fill NaN values in lag features with 0
    lag_features = ['lag_1', 'lag_7', 'rolling_mean_7d', 'rolling_mean_30d']
    df[lag_features] = df[lag_features].fillna(0)
    
    # Select features and handle missing values
    logging.info("Selecting final features...")
    X = df[features].copy()
    X = X.fillna(0)  # Fill any remaining missing values with 0
    
    # Validate final dimensions
    if len(X) != len(df_sales):
        logging.error(f"Final feature matrix has different number of rows: {len(X)} vs {len(df_sales)}")
        raise ValueError(f"Final feature matrix has different number of rows: {len(X)} vs {len(df_sales)}")
    
    # Log final feature matrix info
    logging.info(f"Final feature count: {len(features)}")
    logging.info(f"Final feature matrix shape: {X.shape}")
    logging.info("Feature engineering complete.")
    
    return X, features
