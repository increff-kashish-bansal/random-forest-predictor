import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log'),
        logging.StreamHandler()
    ]
)

def derive_features(df_sales: pd.DataFrame, df_stores: pd.DataFrame, is_prediction: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Derive features from sales and stores data.
    
    Args:
        df_sales: DataFrame with sales data
        df_stores: DataFrame with store information
        is_prediction: Whether this is being called for prediction (single row) or training
        
    Returns:
        Tuple of (X: feature matrix, y: target variable)
    """
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
    
    features.extend(['day_of_week', 'month', 'quarter', 'year', 'is_weekend', 
                    'is_month_start', 'is_month_end', 'lead_time_days'])
    
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
        
        # Save store stats for prediction
        store_stats.to_json('models/store_stats.json')
        
        # Add store stats to features
        store_stats_features = store_stats.columns.tolist()
        features.extend(store_stats_features)
        
        # Merge store stats
        df = df.merge(store_stats, on='store', how='left')
    else:
        # During prediction, load store stats
        try:
            store_stats = pd.read_json('models/store_stats.json')
            # Ensure store IDs are strings
            store_stats.index = store_stats.index.astype(str)
            # Reset index to make store ID a column
            store_stats = store_stats.reset_index()
            store_stats = store_stats.rename(columns={'index': 'store'})
            
            # Add store stats to features
            store_stats_features = store_stats.columns.drop('store').tolist()
            features.extend(store_stats_features)
            
            # Merge store stats
            df = df.merge(store_stats, on='store', how='left')
        except:
            # If no stats available, create empty stats
            store_stats_features = [
                'revenue_mean', 'revenue_std', 'revenue_median',
                'qty_sold_mean', 'qty_sold_std', 'qty_sold_median'
            ]
            features.extend(store_stats_features)
            
            # Add empty stats
            for feat in store_stats_features:
                df[feat] = 0
    
    # 5. Lagged Features
    logging.info("Generating lagged features...")
    if not is_prediction:
        # During training, calculate all lags
        for lag in [1, 7, 14, 30]:
            df[f'revenue_lag_{lag}'] = df.groupby('store')['revenue'].shift(lag)
            df[f'qty_sold_lag_{lag}'] = df.groupby('store')['qty_sold'].shift(lag)
            features.extend([f'revenue_lag_{lag}', f'qty_sold_lag_{lag}'])
    else:
        # During prediction, use last known values or zeros
        for lag in [1, 7, 14, 30]:
            df[f'revenue_lag_{lag}'] = 0
            df[f'qty_sold_lag_{lag}'] = 0
            features.extend([f'revenue_lag_{lag}', f'qty_sold_lag_{lag}'])
    
    # 6. Rolling Features
    logging.info("Generating rolling features...")
    if not is_prediction:
        # During training, calculate all rolling features
        for window in [7, 14, 30]:
            df[f'revenue_rolling_mean_{window}'] = df.groupby('store')['revenue'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'qty_sold_rolling_mean_{window}'] = df.groupby('store')['qty_sold'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            features.extend([f'revenue_rolling_mean_{window}', f'qty_sold_rolling_mean_{window}'])
    else:
        # During prediction, use store-level means
        for window in [7, 14, 30]:
            df[f'revenue_rolling_mean_{window}'] = df['revenue_mean']
            df[f'qty_sold_rolling_mean_{window}'] = df['qty_sold_mean']
            features.extend([f'revenue_rolling_mean_{window}', f'qty_sold_rolling_mean_{window}'])
    
    # 7. Growth & Seasonality
    logging.info("Generating growth and seasonality features...")
    if not is_prediction:
        # During training, calculate growth rates
        df['revenue_growth'] = df.groupby('store')['revenue'].pct_change().fillna(0)
        df['qty_sold_growth'] = df.groupby('store')['qty_sold'].pct_change().fillna(0)
    else:
        # During prediction, use zeros
        df['revenue_growth'] = 0
        df['qty_sold_growth'] = 0
    
    features.extend(['revenue_growth', 'qty_sold_growth'])
    
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
