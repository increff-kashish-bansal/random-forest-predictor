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
        
        # Calculate target encoding for cities, regions, and channels
        city_target = df.groupby('city')['revenue'].transform('mean')
        region_target = df.groupby('region')['revenue'].transform('mean')
        channel_target = df.groupby('channel')['revenue'].transform('mean')
        
        # Save categories and target encodings for prediction
        with open('models/categories.json', 'w') as f:
            json.dump(all_categories, f)
        with open('models/city_target_encoding.json', 'w') as f:
            json.dump(city_target.to_dict(), f)
        with open('models/region_target_encoding.json', 'w') as f:
            json.dump(region_target.to_dict(), f)
        with open('models/channel_target_encoding.json', 'w') as f:
            json.dump(channel_target.to_dict(), f)
    else:
        # During prediction, load categories and target encodings
        try:
            with open('models/categories.json', 'r') as f:
                all_categories = json.load(f)
            with open('models/city_target_encoding.json', 'r') as f:
                city_target = pd.Series(json.load(f))
            with open('models/region_target_encoding.json', 'r') as f:
                region_target = pd.Series(json.load(f))
            with open('models/channel_target_encoding.json', 'r') as f:
                channel_target = pd.Series(json.load(f))
        except FileNotFoundError:
            raise ValueError("Categories or target encoding files not found. Please train the model first.")
    
    # Apply target encoding for cities, regions, and channels
    df['city_encoded'] = df['city'].map(city_target).fillna(0)
    df['region_encoded'] = df['region'].map(region_target).fillna(0)
    df['channel_encoded'] = df['channel'].map(channel_target).fillna(0)
    features.extend(['city_encoded', 'region_encoded', 'channel_encoded'])
    
    # Store area features
    df['store_area'] = df['store_area'].fillna(df['store_area'].median())
    df['store_area_bucket'] = pd.qcut(df['store_area'], q=5, labels=False, duplicates='drop')
    features.extend(['store_area', 'store_area_bucket', 'is_online'])
    
    # 3. Discount Features
    logging.info("Generating discount features...")
    df['discount_pct'] = df['disc_perc'].fillna(0)
    df['discount_pct'] = df['discount_pct'].clip(0, 1)
    df['is_discounted'] = (df['discount_pct'] > 0).astype(int)
    
    # 4. Historical Features
    logging.info("Generating historical features...")
    if not is_prediction:
        # During training, calculate store-level statistics using only past data
        df = df.sort_values(['store', 'date'])
        
        # Calculate expanding statistics (using only past data)
        store_stats = df.groupby('store').apply(
            lambda x: pd.DataFrame({
                'date': x['date'],  # Include date in the output
                'revenue_median': x['revenue'].expanding().median(),  # Keep median instead of mean
                'revenue_std': x['revenue'].expanding().std(),
                'qty_sold_median': x['qty_sold'].expanding().median(),  # Keep median instead of mean
                'qty_sold_std': x['qty_sold'].expanding().std()
            })
        ).reset_index()
        
        # Calculate store-month seasonal indices using only past data
        store_month_stats = df.groupby(['store', 'month']).apply(
            lambda x: pd.DataFrame({
                'date': x['date'],  # Include date in the output
                'store_month_revenue_median': x['revenue'].expanding().median(),  # Use median instead of mean
                'store_month_revenue_std': x['revenue'].expanding().std()
            })
        ).reset_index()
        
        # Calculate seasonal index for each store
        store_seasonal_index = df.groupby(['store', 'month']).apply(
            lambda x: pd.DataFrame({
                'date': x['date'],
                'seasonal_index': x['revenue'].expanding().median() / x.groupby('store')['revenue'].transform('median')
            })
        ).reset_index()
        
        # Save store stats for prediction
        store_stats.to_json('models/store_stats.json')
        store_month_stats.to_json('models/store_month_stats.json')
        store_seasonal_index.to_json('models/store_seasonal_index.json')
        
        # Add store stats to features
        store_stats_features = ['revenue_median', 'revenue_std',
                              'qty_sold_median', 'qty_sold_std']
        store_month_features = ['store_month_revenue_median', 'store_month_revenue_std']
        features.extend(store_stats_features)
        features.extend(store_month_features)
        
        # Merge store stats
        df = df.merge(store_stats, on=['store', 'date'], how='left')
        df = df.merge(store_month_stats, on=['store', 'month', 'date'], how='left')
        df = df.merge(store_seasonal_index, on=['store', 'month', 'date'], how='left')
        
        # Add store-seasonality interaction features
        df['store_seasonal_index'] = df['seasonal_index'] * df['store_area_bucket']
        df['region_seasonal_index'] = df['seasonal_index'] * df['region_encoded']
        features.extend(['store_seasonal_index', 'region_seasonal_index'])
        
        # Add lag features (using only past data)
        for lag in [1, 3, 7, 14, 30]:
            df[f'revenue_lag_{lag}'] = df.groupby('store')['revenue'].shift(lag)
            df[f'qty_sold_lag_{lag}'] = df.groupby('store')['qty_sold'].shift(lag)
            features.extend([f'revenue_lag_{lag}', f'qty_sold_lag_{lag}'])
        
        # Add rolling statistics (using only past data)
        for window in [7, 14, 30]:
            # Use consistent naming for rolling statistics
            df[f'revenue_rolling_mean_{window}d'] = df.groupby('store')['revenue'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            df[f'revenue_rolling_std_{window}d'] = df.groupby('store')['revenue'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
            )
            features.extend([f'revenue_rolling_mean_{window}d', f'revenue_rolling_std_{window}d'])
        
    else:
        # During prediction, load store stats
        try:
            store_stats = pd.read_json('models/store_stats.json')
            store_month_stats = pd.read_json('models/store_month_stats.json')
            store_seasonal_index = pd.read_json('models/store_seasonal_index.json')
            
            # Ensure store IDs are strings and convert dates
            store_stats['store'] = store_stats['store'].astype(str)
            store_stats['date'] = pd.to_datetime(store_stats['date'])
            store_month_stats['store'] = store_month_stats['store'].astype(str)
            store_month_stats['date'] = pd.to_datetime(store_month_stats['date'])
            store_seasonal_index['store'] = store_seasonal_index['store'].astype(str)
            store_seasonal_index['date'] = pd.to_datetime(store_seasonal_index['date'])
            
            # Add store stats to features
            store_stats_features = ['revenue_median', 'revenue_std',
                                  'qty_sold_median', 'qty_sold_std']
            store_month_features = ['store_month_revenue_median', 'store_month_revenue_std']
            features.extend(store_stats_features)
            features.extend(store_month_features)
            
            # Merge store stats
            df = df.merge(store_stats, on=['store', 'date'], how='left')
            df = df.merge(store_month_stats, on=['store', 'month', 'date'], how='left')
            df = df.merge(store_seasonal_index, on=['store', 'month', 'date'], how='left')
            
            # Add store-seasonality interaction features
            df['store_seasonal_index'] = df['seasonal_index'] * df['store_area_bucket']
            df['region_seasonal_index'] = df['seasonal_index'] * df['region_encoded']
            features.extend(['store_seasonal_index', 'region_seasonal_index'])
            
            # Add lag features from historical data
            if historical_sales is not None:
                historical_sales = historical_sales.sort_values(['store', 'date'])
                for lag in [1, 3, 7, 14, 30]:
                    df[f'revenue_lag_{lag}'] = historical_sales.groupby('store')['revenue'].last()
                    df[f'qty_sold_lag_{lag}'] = historical_sales.groupby('store')['qty_sold'].last()
                    features.extend([f'revenue_lag_{lag}', f'qty_sold_lag_{lag}'])
                
                # Add rolling statistics from historical data
                for window in [7, 14, 30]:
                    # Use consistent naming for rolling statistics
                    df[f'revenue_rolling_mean_{window}d'] = historical_sales.groupby('store')['revenue'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean().iloc[-1]
                    )
                    df[f'revenue_rolling_std_{window}d'] = historical_sales.groupby('store')['revenue'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std().iloc[-1]
                    )
                    features.extend([f'revenue_rolling_mean_{window}d', f'revenue_rolling_std_{window}d'])
            
        except FileNotFoundError:
            # If no stats available, create empty stats
            store_stats_features = ['revenue_median', 'revenue_std',
                                  'qty_sold_median', 'qty_sold_std']
            store_month_features = ['store_month_revenue_median', 'store_month_revenue_std']
            features.extend(store_stats_features)
            features.extend(store_month_features)
            
            # Add empty stats
            for feat in store_stats_features + store_month_features:
                df[feat] = 0
            
            # Add empty lag features
            for lag in [1, 3, 7, 14, 30]:
                df[f'revenue_lag_{lag}'] = 0
                df[f'qty_sold_lag_{lag}'] = 0
                features.extend([f'revenue_lag_{lag}', f'qty_sold_lag_{lag}'])
            
            # Add empty rolling statistics
            for window in [7, 14, 30]:
                # Use consistent naming for rolling statistics
                df[f'revenue_rolling_mean_{window}d'] = 0
                df[f'revenue_rolling_std_{window}d'] = 0
                features.extend([f'revenue_rolling_mean_{window}d', f'revenue_rolling_std_{window}d'])
            
            # Add empty seasonal features
            df['seasonal_index'] = 1.0  # Default to neutral seasonal effect
            df['store_seasonal_index'] = df['store_area_bucket']
            df['region_seasonal_index'] = df['region_encoded']
            features.extend(['store_seasonal_index', 'region_seasonal_index'])
    
    # Fill NaN values in historical features with 0
    historical_features = [f for f in features if any(x in f for x in ['mean', 'std', 'median', 'lag', 'rolling'])]
    df[historical_features] = df[historical_features].fillna(0)
    
    # Add interaction features
    logging.info("Generating interaction features...")
    
    # 1. Store Characteristics Interactions
    # Store area and city interaction (captures premium locations)
    df['store_area_city_interaction'] = df['store_area'] * df['city_encoded']
    
    # Store area and region interaction (captures regional preferences)
    df['store_area_region_interaction'] = df['store_area'] * df['region_encoded']
    
    # Online presence and city interaction (captures digital adoption by city)
    df['online_city_interaction'] = df['is_online'] * df['city_encoded']
    
    # 2. Discount Strategy Interactions
    # Discount and store tier interaction (captures premium vs regular pricing)
    df['discount_store_tier_interaction'] = df['discount_pct'] * df['store_area_bucket']
    
    # Discount and region interaction (captures regional price sensitivity)
    df['discount_region_interaction'] = df['discount_pct'] * df['region_encoded']
    
    # Discount and city interaction (captures city-specific price sensitivity)
    df['discount_city_interaction'] = df['discount_pct'] * df['city_encoded']
    
    # 3. Temporal Interactions
    # Month and region interaction (captures seasonal patterns by region)
    df['month_region_interaction'] = df['month'] * df['region_encoded']
    
    # Month and store tier interaction (captures seasonal patterns by store type)
    df['month_store_tier_interaction'] = df['month'] * df['store_area_bucket']
    
    # 4. Historical Performance Interactions
    # Recent performance and discount interaction
    df['revenue_lag_7_discount_interaction'] = df['revenue_lag_7'] * df['discount_pct']
    
    # Store volatility and discount interaction
    df['revenue_std_discount_interaction'] = df['revenue_std'] * df['discount_pct']
    
    # 5. Complex Interactions
    # Premium location discount sensitivity
    df['premium_location_discount'] = df['store_area_city_interaction'] * df['discount_pct']
    
    # Seasonal premium location effect
    df['seasonal_premium_location'] = df['month'] * df['store_area_city_interaction']
    
    # Add new interaction features to the feature list
    interaction_features = [
        'store_area_city_interaction',
        'store_area_region_interaction',
        'online_city_interaction',
        'discount_store_tier_interaction',
        'discount_region_interaction',
        'discount_city_interaction',
        'month_region_interaction',
        'month_store_tier_interaction',
        'revenue_lag_7_discount_interaction',
        'revenue_std_discount_interaction',
        'premium_location_discount',
        'seasonal_premium_location'
    ]
    features.extend(interaction_features)
    
    # Remove old low-variance interactions
    old_interactions = [
        'discount_weekend_interaction',
        'discount_store_area_interaction',
        'region_quarter_interaction'
    ]
    features = [f for f in features if f not in old_interactions]
    
    # 4. Polynomial interactions
    df['discount_pct_squared'] = df['discount_pct'] ** 2
    df['store_area_squared'] = df['store_area'] ** 2
    
    features.extend([
        'discount_pct', 'is_discounted',
        'discount_pct_squared',
        'store_area_squared'
    ])
    
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
    
    # Remove any duplicate features that might have been created
    features = list(dict.fromkeys(features))
    
    return X, features
