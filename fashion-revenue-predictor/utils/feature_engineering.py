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
import pickle
from scipy.stats import boxcox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log'),
        logging.StreamHandler()
    ]
)

def apply_cluster_specific_transforms(df: pd.DataFrame, cluster: str) -> pd.DataFrame:
    """
    Apply cluster-specific feature transformations.
    
    Args:
        df: DataFrame containing features
        cluster: Store cluster identifier
        
    Returns:
        DataFrame with transformed features
    """
    df = df.copy()
    
    # Define cluster-specific transformations
    if cluster == 'PREMIUM_OFFLINE':
        # Box-Cox transform for revenue and quantity
        if 'revenue' in df.columns:
            df['revenue'], lambda_revenue = boxcox(df['revenue'] + 1)
        if 'qty_sold' in df.columns:
            df['qty_sold'], lambda_qty = boxcox(df['qty_sold'] + 1)
            
        # Target encoding for discounts
        if 'discount_pct' in df.columns:
            discount_means = df.groupby('discount_pct')['revenue'].transform('mean')
            df['discount_target_encoded'] = discount_means
            
    elif cluster == 'ONLINE':
        # Log transform with offset for online sales
        if 'revenue' in df.columns:
            df['revenue'] = np.log1p(df['revenue'] + 100)  # Add offset for online sales
        if 'qty_sold' in df.columns:
            df['qty_sold'] = np.log1p(df['qty_sold'] + 10)
            
        # Discount interaction features
        if 'discount_pct' in df.columns:
            df['discount_squared'] = df['discount_pct'] ** 2
            df['discount_cubed'] = df['discount_pct'] ** 3
            
    elif cluster == 'SEASONAL_SPECIALIST':
        # Seasonal-specific transformations
        if 'revenue' in df.columns:
            # Use Box-Cox for revenue
            df['revenue'], lambda_revenue = boxcox(df['revenue'] + 1)
            
        # Create seasonal discount features
        if 'discount_pct' in df.columns and 'month' in df.columns:
            df['seasonal_discount'] = df['discount_pct'] * np.sin(2 * np.pi * df['month'] / 12)
            
    elif cluster == 'HOLIDAY_SPECIALIST':
        # Holiday-specific transformations
        if 'revenue' in df.columns:
            # Use log transform for holiday stores
            df['revenue'] = np.log1p(df['revenue'])
            
        # Create holiday-specific discount features
        if 'discount_pct' in df.columns:
            holiday_months = [10, 11, 12, 1, 2]  # October to February
            df['is_holiday_month'] = df['month'].isin(holiday_months).astype(int)
            df['holiday_discount'] = df['discount_pct'] * df['is_holiday_month']
            
    else:  # Default for other clusters
        # Standard log transform
        if 'revenue' in df.columns:
            df['revenue'] = np.log1p(df['revenue'])
        if 'qty_sold' in df.columns:
            df['qty_sold'] = np.log1p(df['qty_sold'])
    
    return df

def derive_features(df_sales: pd.DataFrame, df_stores: pd.DataFrame, historical_sales: pd.DataFrame = None, is_prediction: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Derive features for model training or prediction.
    
    Args:
        df_sales: Sales DataFrame
        df_stores: Stores DataFrame
        historical_sales: Optional historical sales data for prediction
        is_prediction: Whether this is for prediction (True) or training (False)
        
    Returns:
        Tuple of (feature DataFrame, list of feature names)
    """
    logging.info("Starting feature engineering...")
    logging.info(f"Input shapes - Sales: {df_sales.shape}, Stores: {df_stores.shape}")
    if historical_sales is not None:
        logging.info(f"Historical sales shape: {historical_sales.shape}")
    
    # Ensure consistent data types for categorical columns
    categorical_cols = ['channel', 'region', 'city']
    for col in categorical_cols:
        if col in df_stores.columns:
            df_stores[col] = df_stores[col].astype(str)
        if col in df_sales.columns:
            df_sales[col] = df_sales[col].astype(str)
        if historical_sales is not None and col in historical_sales.columns:
            historical_sales[col] = historical_sales[col].astype(str)
    
    # Merge sales and store data
    df = df_sales.merge(df_stores, left_on='store', right_on='id', how='left')
    logging.info(f"Merged data shape: {df.shape}")
    logging.info("Column types after merge:")
    logging.info(df.dtypes)
    
    features = []
    
    # Apply cluster-specific transformations
    logging.info("Applying cluster-specific transformations...")
    if 'store_cluster' in df.columns:
        # Group by cluster and apply transformations
        transformed_dfs = []
        for cluster in df['store_cluster'].unique():
            cluster_df = df[df['store_cluster'] == cluster].copy()
            transformed_df = apply_cluster_specific_transforms(cluster_df, cluster)
            transformed_dfs.append(transformed_df)
        
        # Combine transformed DataFrames
        df = pd.concat(transformed_dfs, axis=0)
        df = df.sort_index()  # Restore original order
    
    # 1. Temporal Features
    logging.info("Generating temporal features...")
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Add cyclical encoding for temporal features
    logging.info("Adding cyclical encoding for temporal features...")
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    features.extend([
        'year', 'month', 'day', 'day_of_week', 'quarter', 'is_weekend',
        'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'
    ])
    
    # 2. Store/Location Features
    logging.info("Generating store/location features...")
    
    # Define all possible categorical values
    if not is_prediction:
        # During training, get unique values from data
        all_categories = {
            'channel': sorted(df['channel'].unique().tolist()),
            'region': sorted(df['region'].unique().tolist()),
            'city': sorted(df['city'].unique().tolist())
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
    df['discount_pct'] = df['disc_perc'].fillna(0)
    df['discount_pct'] = df['discount_pct'].clip(0, 1)
    df['is_discounted'] = (df['discount_pct'] > 0).astype(int)
    
    # Add interaction features
    logging.info("Generating interaction features...")
    
    # 1. Discount and weekend interaction
    df['discount_weekend_interaction'] = df['discount_pct'] * df['is_weekend']
    
    # 2. Discount and store area bucket interaction
    df['discount_store_area_interaction'] = df['discount_pct'] * df['store_area_bucket']
    
    # 3. Region and quarter interaction using label encoding
    if not is_prediction:
        region_encoder = LabelEncoder()
        df['region_encoded'] = region_encoder.fit_transform(df['region'])
        # Save encoder for prediction
        with open('models/region_encoder.pkl', 'wb') as f:
            pickle.dump(region_encoder, f)
    else:
        try:
            with open('models/region_encoder.pkl', 'rb') as f:
                region_encoder = pickle.load(f)
            df['region_encoded'] = region_encoder.transform(df['region'])
        except:
            df['region_encoded'] = 0  # Default if encoder not found
    
    # Create region-quarter interaction
    df['region_quarter_interaction'] = df['region_encoded'] * df['quarter']
    
    # 4. Polynomial interactions
    df['discount_pct_squared'] = df['discount_pct'] ** 2
    df['store_area_squared'] = df['store_area'] ** 2
    
    features.extend([
        'discount_pct', 'is_discounted',
        'discount_weekend_interaction',
        'discount_store_area_interaction',
        'region_quarter_interaction',
        'discount_pct_squared',
        'store_area_squared'
    ])
    
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
    
    # 5. Rolling Statistics
    logging.info("Generating rolling statistics...")
    if historical_sales is not None:
        # Calculate rolling statistics using historical data
        historical_sales['date'] = pd.to_datetime(historical_sales['date'])
        historical_sales = historical_sales.sort_values('date')
        
        # Calculate rolling means and stds
        for window in [7, 14, 30]:
            rolling_mean = historical_sales['revenue'].rolling(window=window, min_periods=1).mean()
            rolling_std = historical_sales['revenue'].rolling(window=window, min_periods=1).std()
            
            df[f'rolling_mean_{window}d'] = rolling_mean.iloc[-1]
            df[f'rolling_std_{window}d'] = rolling_std.iloc[-1]
            
            features.extend([f'rolling_mean_{window}d', f'rolling_std_{window}d'])
    else:
        # During training, calculate rolling statistics
        df = df.sort_values('date')
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}d'] = df.groupby('store')['revenue'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}d'] = df.groupby('store')['revenue'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            features.extend([f'rolling_mean_{window}d', f'rolling_std_{window}d'])
    
    # 6. Lead Time Features
    logging.info("Generating lead time features...")
    if historical_sales is not None:
        # Calculate lead time using historical data
        last_date = historical_sales['date'].max()
        df['lead_time_days'] = (df['date'] - last_date).dt.days
    else:
        # During training, calculate lead time from previous day
        df['lead_time_days'] = df.groupby('store')['date'].diff().dt.days
        df['lead_time_days'] = df['lead_time_days'].fillna(0)
    
    features.append('lead_time_days')
    
    # Ensure all features are present and in the correct order
    if is_prediction:
        try:
            with open('models/brandA_feature_names.json', 'r') as f:
                saved_features = json.load(f)
            
            # Ensure all required features are present
            for feature_set in saved_features.values():
                missing_features = set(feature_set) - set(df.columns)
                if missing_features:
                    for feat in missing_features:
                        df[feat] = 0  # Add missing features with zeros
        except FileNotFoundError:
            raise ValueError("Feature names file not found. Please train the model first.")
    
    # Select final features
    logging.info("Selecting final features...")
    if is_prediction:
        # During prediction, use all saved feature sets
        X = df[saved_features['all_features']]
    else:
        # During training, use all generated features
        X = df[features]
    
    logging.info(f"Final feature count: {len(features)}")
    logging.info(f"Final feature matrix shape: {X.shape}")
    logging.info("Feature engineering complete.")
    
    return X, features
