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

def calculate_sample_weights(df: pd.DataFrame, revenue_col: str = 'revenue', time_col: str = 'date') -> np.ndarray:
    """
    Calculate fine-tuned sample weights that combine time decay and capped log-based weights.
    
    Args:
        df: DataFrame containing the data
        revenue_col: Name of the revenue column
        time_col: Name of the date column
        
    Returns:
        Array of sample weights
    """
    # Calculate time decay weights (more recent samples get higher weights)
    max_date = df[time_col].max()
    days_diff = (max_date - df[time_col]).dt.days
    time_decay_weights = 1 / (1 + np.log1p(days_diff))
    
    # Calculate capped log-based weights for revenue
    revenue_values = df[revenue_col].values
    revenue_cap = np.percentile(revenue_values, 95)  # Cap at 95th percentile
    log_weights = np.log1p(np.minimum(revenue_values, revenue_cap))
    
    # Normalize both weight components
    time_decay_weights = time_decay_weights / np.mean(time_decay_weights)
    log_weights = log_weights / np.mean(log_weights)
    
    # Combine weights with 60% time decay and 40% revenue importance
    combined_weights = 0.6 * time_decay_weights + 0.4 * log_weights
    
    # Normalize final weights to have mean of 1
    combined_weights = combined_weights / np.mean(combined_weights)
    
    return combined_weights

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
    
    # Calculate sample weights if not in prediction mode
    if not is_prediction and 'revenue' in df_sales.columns:
        sample_weights = calculate_sample_weights(df_sales)
        # Save weights for later use in model training
        np.save('models/sample_weights.npy', sample_weights)
        logging.info("Sample weights calculated and saved")
    
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
    
    # Add 2nd and 3rd harmonic cyclical encodings for day-of-year and week-of-year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    # 2nd harmonic
    df['day_of_year_sin2'] = np.sin(4 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos2'] = np.cos(4 * np.pi * df['day_of_year'] / 365)
    df['week_of_year_sin2'] = np.sin(4 * np.pi * df['week_of_year'] / 52)
    df['week_of_year_cos2'] = np.cos(4 * np.pi * df['week_of_year'] / 52)
    
    # 3rd harmonic
    df['day_of_year_sin3'] = np.sin(6 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos3'] = np.cos(6 * np.pi * df['day_of_year'] / 365)
    df['week_of_year_sin3'] = np.sin(6 * np.pi * df['week_of_year'] / 52)
    df['week_of_year_cos3'] = np.cos(6 * np.pi * df['week_of_year'] / 52)
    
    features.extend([
        'year', 'month', 'day', 'day_of_week', 'quarter', 'is_weekend',
        'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
        'day_of_year_sin2', 'day_of_year_cos2', 'week_of_year_sin2', 'week_of_year_cos2',
        'day_of_year_sin3', 'day_of_year_cos3', 'week_of_year_sin3', 'week_of_year_cos3'
    ])
    
    # --- Weekday × Store Interactions ---
    # Ordinal encoding: store_id may be string, so encode as categorical codes
    if 'store' in df.columns and 'day_of_week' in df.columns:
        df['store_id_ord'] = pd.Categorical(df['store']).codes
        df['weekday_store_ord'] = df['day_of_week'] * 10000 + df['store_id_ord']
        features.append('weekday_store_ord')
        # One-hot encoding for (store, weekday) pairs
        weekday_store_ohe = pd.get_dummies(df['day_of_week'].astype(str) + '_' + df['store'].astype(str), prefix='wdst')
        df = pd.concat([df, weekday_store_ohe], axis=1)
        features.extend(list(weekday_store_ohe.columns))
    
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
    
    # Create city and region tiers based on revenue metrics
    if not is_prediction:
        # During training, calculate tiers based on historical data
        # City tiers based on average revenue
        city_revenue = df.groupby('city')['revenue'].agg(['mean', 'std', 'count'])
        city_revenue['score'] = (
            city_revenue['mean'] * 0.6 +  # Revenue importance
            city_revenue['std'] * 0.2 +   # Revenue stability
            city_revenue['count'] * 0.2   # Data volume
        )
        city_tiers = pd.qcut(city_revenue['score'], q=4, labels=["D", "C", "B", "A"])
        city_tier_map = city_tiers.to_dict()
        
        # Region tiers based on average revenue
        region_revenue = df.groupby('region')['revenue'].agg(['mean', 'std', 'count'])
        region_revenue['score'] = (
            region_revenue['mean'] * 0.6 +  # Revenue importance
            region_revenue['std'] * 0.2 +   # Revenue stability
            region_revenue['count'] * 0.2   # Data volume
        )
        region_tiers = pd.qcut(region_revenue['score'], q=4, labels=["D", "C", "B", "A"])
        region_tier_map = region_tiers.to_dict()
        
        # Save tier mappings for prediction
        with open('models/city_tiers.json', 'w') as f:
            json.dump(city_tier_map, f)
        with open('models/region_tiers.json', 'w') as f:
            json.dump(region_tier_map, f)
    else:
        # During prediction, load tier mappings
        try:
            with open('models/city_tiers.json', 'r') as f:
                city_tier_map = json.load(f)
            with open('models/region_tiers.json', 'r') as f:
                region_tier_map = json.load(f)
        except FileNotFoundError:
            raise ValueError("Tier mapping files not found. Please train the model first.")
    
    # Apply tier mappings
    df['city_tier'] = df['city'].map(city_tier_map).fillna('D')  # Default to lowest tier
    df['region_tier'] = df['region'].map(region_tier_map).fillna('D')  # Default to lowest tier
    
    # Convert tiers to numeric values for modeling
    tier_map = {"D": 0, "C": 1, "B": 2, "A": 3}
    df['city_tier_encoded'] = df['city_tier'].map(tier_map)
    df['region_tier_encoded'] = df['region_tier'].map(tier_map)
    
    # Add tier features to feature list
    features.extend(['city_tier_encoded', 'region_tier_encoded'])
    
    # Store area features
    df['store_area'] = df['store_area'].fillna(df['store_area'].median())
    df['store_area_bucket'] = pd.qcut(df['store_area'], q=5, labels=False, duplicates='drop')
    features.extend(['store_area', 'store_area_bucket', 'is_online'])
    
    # 3. Discount Features
    logging.info("Generating discount features...")
    df['discount_pct'] = df['disc_perc'].fillna(0)
    df['discount_pct'] = df['discount_pct'].clip(0, 1)
    df['is_discounted'] = (df['discount_pct'] > 0).astype(int)
    
    # Add discount per square foot feature
    df['discount_per_sqft'] = (df['discount_pct'] * df['revenue']) / df['store_area']
    features.append('discount_per_sqft')
    
    # Add region × channel × month one-hot encoded features
    logging.info("Adding region × channel × month interaction features...")
    region_channel_month = pd.get_dummies(
        df['region'].astype(str) + '_' + 
        df['channel'].astype(str) + '_' + 
        df['month'].astype(str),
        prefix='rcm'
    )
    df = pd.concat([df, region_channel_month], axis=1)
    features.extend(list(region_channel_month.columns))
    
    # Add region_weekday_std: std dev of revenue for each region × weekday pair (past 4 weeks rolling)
    logging.info("Adding region weekday volatility features...")
    df['region_weekday_std'] = df.groupby(['region', 'day_of_week'])['revenue'].transform(
        lambda x: x.rolling(window=28, min_periods=1).std().shift(1)
    )
    features.append('region_weekday_std')
    
    # Add time_since_last_high_revenue: days since last revenue > 90th percentile per store
    logging.info("Adding time since last high revenue feature...")
    def time_since_last_high_revenue_transform(rev):
        pct90 = rev.expanding().quantile(0.9)
        last_high_idx = -1 * np.ones(len(rev), dtype=int)
        days_since = np.zeros(len(rev), dtype=int)
        for i in range(len(rev)):
            if i == 0:
                days_since[i] = 0
            else:
                if rev.iloc[i-1] > pct90.iloc[i-1]:
                    last_high_idx[i] = i-1
                else:
                    last_high_idx[i] = last_high_idx[i-1]
                if last_high_idx[i] == -1:
                    days_since[i] = i
                else:
                    days_since[i] = i - last_high_idx[i]
        return pd.Series(days_since, index=rev.index)
    
    df['time_since_last_high_revenue'] = df.groupby('store')['revenue'].transform(time_since_last_high_revenue_transform)
    features.append('time_since_last_high_revenue')
    
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
                'revenue_std': x['revenue'].expanding().std()
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
        store_stats_features = ['revenue_median', 'revenue_std']
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
            features.extend([f'revenue_lag_{lag}'])
        
        # Add lag-based difference features to capture trends
        # Revenue differences
        df['revenue_lag_diff_1_3'] = df['revenue_lag_1'] - df['revenue_lag_3']
        df['revenue_lag_diff_3_7'] = df['revenue_lag_3'] - df['revenue_lag_7']
        df['revenue_lag_diff_7_14'] = df['revenue_lag_7'] - df['revenue_lag_14']
        df['revenue_lag_diff_14_30'] = df['revenue_lag_14'] - df['revenue_lag_30']
        
        # Add difference features to feature list
        diff_features = [
            'revenue_lag_diff_1_3', 'revenue_lag_diff_3_7',
            'revenue_lag_diff_7_14', 'revenue_lag_diff_14_30'
        ]
        features.extend(diff_features)
        
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
        
        # Add short-term stability features
        logging.info("Generating short-term stability features...")
        
        # 1. Short-term rolling aggregates (last 3 days)
        df['revenue_last_3_days'] = df.groupby('store')['revenue'].transform(
            lambda x: x.rolling(3, min_periods=1).mean().shift(1)
        )
        df['revenue_last_3_days_std'] = df.groupby('store')['revenue'].transform(
            lambda x: x.rolling(3, min_periods=1).std().shift(1)
        )
        
        # 2. Weekday-specific features
        df['weekday_avg'] = df.groupby('day_of_week')['revenue'].transform('mean')
        df['weekday_std'] = df.groupby('day_of_week')['revenue'].transform('std')
        
        # 3. Store-specific weekday patterns
        df['store_weekday_avg'] = df.groupby(['store', 'day_of_week'])['revenue'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['store_weekday_std'] = df.groupby(['store', 'day_of_week'])['revenue'].transform(
            lambda x: x.expanding().std().shift(1)
        )
        
        # 4. Week-over-week changes
        df['revenue_week_over_week'] = df.groupby(['store', 'day_of_week'])['revenue'].transform(
            lambda x: x.pct_change(7)
        )
        
        # 5. Day-over-day changes
        df['revenue_day_over_day'] = df.groupby('store')['revenue'].transform(
            lambda x: x.pct_change(1)
        )
        
        # 6. Volatility features
        df['revenue_volatility_3d'] = df.groupby('store')['revenue'].transform(
            lambda x: x.rolling(3, min_periods=1).std().shift(1) / 
                     x.rolling(3, min_periods=1).mean().shift(1)
        )
        
        # Add stability features to feature list
        stability_features = [
            'revenue_last_3_days',
            'revenue_last_3_days_std',
            'weekday_avg',
            'weekday_std',
            'store_weekday_avg',
            'store_weekday_std',
            'revenue_week_over_week',
            'revenue_day_over_day',
            'revenue_volatility_3d'
        ]
        features.extend(stability_features)
        
        # Add revenue_lag_7_percentile: percentile rank of revenue 7 days ago in the last 30 days per store
        def lag_7_percentile(x):
            # For each row, compute the percentile rank of revenue_lag_7 among the last 30 days
            rev = x['revenue'].values
            lag_7 = pd.Series(rev).shift(7)
            percentiles = []
            for i in range(len(x)):
                if i < 30 or pd.isna(lag_7.iloc[i]):
                    percentiles.append(np.nan)
                else:
                    window = rev[max(0, i-30):i]
                    if len(window) == 0 or pd.isna(lag_7.iloc[i]):
                        percentiles.append(np.nan)
                    else:
                        percentiles.append((window < lag_7.iloc[i]).sum() / len(window))
            return pd.Series(percentiles, index=x.index)
        df['revenue_lag_7_percentile'] = df.groupby('store', group_keys=False).apply(lag_7_percentile)
        features.append('revenue_lag_7_percentile')
        
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
            store_stats_features = ['revenue_median', 'revenue_std']
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
                    features.extend([f'revenue_lag_{lag}'])
                
                # Add lag-based difference features to capture trends
                # Revenue differences
                df['revenue_lag_diff_1_3'] = df['revenue_lag_1'] - df['revenue_lag_3']
                df['revenue_lag_diff_3_7'] = df['revenue_lag_3'] - df['revenue_lag_7']
                df['revenue_lag_diff_7_14'] = df['revenue_lag_7'] - df['revenue_lag_14']
                df['revenue_lag_diff_14_30'] = df['revenue_lag_14'] - df['revenue_lag_30']
                
                # Add difference features to feature list
                diff_features = [
                    'revenue_lag_diff_1_3', 'revenue_lag_diff_3_7',
                    'revenue_lag_diff_7_14', 'revenue_lag_diff_14_30'
                ]
                features.extend(diff_features)
                
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
            store_stats_features = ['revenue_median', 'revenue_std']
            store_month_features = ['store_month_revenue_median', 'store_month_revenue_std']
            features.extend(store_stats_features)
            features.extend(store_month_features)
            
            # Add empty stats
            for feat in store_stats_features + store_month_features:
                df[feat] = 0
            
            # Add empty lag features
            for lag in [1, 3, 7, 14, 30]:
                df[f'revenue_lag_{lag}'] = 0
                features.extend([f'revenue_lag_{lag}'])
            
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
    historical_features = [f for f in features if any(x in f for x in ['mean', 'std'])]
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
    
    # 6. SHAP-based feature re-addition with orthogonalization
    logging.info("Re-adding top SHAP features with orthogonalization...")
    
    # Define high-importance features to re-add
    high_importance_features = [
        'store_area',
        'revenue_rolling_mean_7d',
        'revenue_rolling_std_7d'
    ]
    
    # Orthogonalize features
    for feat in high_importance_features:
        if feat in df.columns:
            # Mean-center the feature
            df[f'{feat}_centered'] = df[feat] - df[feat].mean()
            
            # Create residualized version by removing linear correlation with other features
            # Use a subset of key features for residualization
            residualization_features = [
                'store_area_bucket',
                'city_tier_encoded',
                'region_tier_encoded',
                'discount_pct',
                'month'
            ]
            
            # Only use features that exist in the DataFrame
            residualization_features = [f for f in residualization_features if f in df.columns]
            
            if residualization_features:
                # Calculate residuals using linear regression
                X_resid = df[residualization_features].copy()
                y_resid = df[f'{feat}_centered'].copy()
                
                # Add constant for intercept
                X_resid = pd.concat([pd.Series(1, index=X_resid.index, name='const'), X_resid], axis=1)
                
                # Calculate coefficients using normal equation
                try:
                    # Ensure all data is numeric and handle any missing values
                    X_resid = X_resid.fillna(0)
                    y_resid = y_resid.fillna(0)
                    
                    # Convert to numpy arrays for matrix operations
                    X_np = X_resid.values
                    y_np = y_resid.values
                    
                    # Calculate coefficients
                    beta = np.linalg.inv(X_np.T @ X_np) @ X_np.T @ y_np
                    
                    # Calculate residuals
                    df[f'{feat}_orthogonal'] = y_np - (X_np @ beta)
                except np.linalg.LinAlgError:
                    # If matrix is singular, use simple mean-centering
                    df[f'{feat}_orthogonal'] = df[f'{feat}_centered']
            else:
                # If no residualization features available, use mean-centered version
                df[f'{feat}_orthogonal'] = df[f'{feat}_centered']
            
            # Add orthogonalized feature to feature list
            features.append(f'{feat}_orthogonal')
    
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
    
    # --- FIXED: Use transform for index alignment ---
    # region_weekday_volatility: std of past 4-week revenue by region and weekday
    def region_weekday_volatility_transform(x):
        return x.rolling(window=28, min_periods=1).std().shift(1)
    df['region_weekday_volatility'] = df.groupby(['region', 'day_of_week'])['revenue'].transform(region_weekday_volatility_transform)
    features.append('region_weekday_volatility')

    # time_since_last_peak_revenue: days since last revenue > 90th percentile per store
    def time_since_last_peak_transform(rev):
        pct90 = rev.expanding().quantile(0.9)
        last_peak_idx = -1 * np.ones(len(rev), dtype=int)
        days_since = np.zeros(len(rev), dtype=int)
        for i in range(len(rev)):
            if i == 0:
                days_since[i] = 0
            else:
                if rev.iloc[i-1] > pct90.iloc[i-1]:
                    last_peak_idx[i] = i-1
                else:
                    last_peak_idx[i] = last_peak_idx[i-1]
                if last_peak_idx[i] == -1:
                    days_since[i] = i
                else:
                    days_since[i] = i - last_peak_idx[i]
        return pd.Series(days_since, index=rev.index)
    df['time_since_last_peak_revenue'] = df.groupby('store')['revenue'].transform(time_since_last_peak_transform)
    features.append('time_since_last_peak_revenue')

    # revenue_lag_7_percentile: percentile rank of revenue 7 days ago in the last 30 days per store
    def lag_7_percentile_transform(rev):
        lag_7 = rev.shift(7)
        percentiles = []
        for i in range(len(rev)):
            if i < 30 or pd.isna(lag_7.iloc[i]):
                percentiles.append(np.nan)
            else:
                window = rev.iloc[max(0, i-30):i]
                if len(window) == 0 or pd.isna(lag_7.iloc[i]):
                    percentiles.append(np.nan)
                else:
                    percentiles.append((window < lag_7.iloc[i]).sum() / len(window))
        return pd.Series(percentiles, index=rev.index)
    df['revenue_lag_7_percentile'] = df.groupby('store')['revenue'].transform(lag_7_percentile_transform)
    features.append('revenue_lag_7_percentile')
    
    return X, features
