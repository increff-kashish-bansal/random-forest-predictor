import pandas as pd
import numpy as np
import json
import pickle
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from utils.custom_cv import GroupTimeSeriesSplit
from utils.feature_engineering import derive_features
from scipy.stats import boxcox
import joblib
from sklearn.preprocessing import OneHotEncoder
from predictor import calculate_prediction_metrics
from utils.feature_selection import iterative_feature_pruning, select_features_by_global_shap, drop_low_importance_features
import lightgbm as lgb
import optuna  # Added for hyperparameter tuning
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', mode='w'),  # Use 'w' mode to overwrite previous logs
        logging.StreamHandler(sys.stdout)
    ]
)

# Add a filter to prevent duplicate logs
class DuplicateFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last_log = None

    def filter(self, record):
        current_log = (record.levelno, record.getMessage())
        if current_log == self.last_log:
            return False
        self.last_log = current_log
        return True

# Add the filter to all handlers
for handler in logging.root.handlers:
    handler.addFilter(DuplicateFilter())

def calculate_sample_weights(df: pd.DataFrame, decay_factor: float = 0.1) -> np.ndarray:
    """
    Calculate sample weights using log-based decay and day-of-week weighting.
    
    Args:
        df: DataFrame with sales data
        decay_factor: Controls the rate of time decay (higher = faster decay)
        
    Returns:
        Array of sample weights
    """
    # Calculate days difference from most recent date
    max_date = df['date'].max()
    days_diff = (max_date - df['date']).dt.days
    
    # Apply log-based decay
    time_weights = np.exp(-decay_factor * np.log1p(days_diff))
    
    # Calculate day-of-week weights (weekends get higher weight)
    day_of_week = df['date'].dt.dayofweek
    dow_weights = np.where(day_of_week >= 5, 1.5, 1.0)  # Weekend days get 1.5x weight
    
    # Combine weights
    weights = time_weights * dow_weights
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    return weights

def calculate_dynamic_test_size(df: pd.DataFrame, store: str) -> float:
    """
    Calculate dynamic test size based on store's historical data length.
    
    Args:
        df: DataFrame with sales data
        store: Store ID
        
    Returns:
        Float between 0.1 and 0.3 representing test size
    """
    store_data = df[df['store'] == store]
    data_length = len(store_data)
    
    # Base test size on data length
    if data_length < 30:  # Less than a month
        return 0.1
    elif data_length < 90:  # Less than 3 months
        return 0.15
    elif data_length < 180:  # Less than 6 months
        return 0.2
    else:  # More than 6 months
        return 0.3

def select_features_by_importance(X: pd.DataFrame, feature_importances: Dict[str, float], 
                                quantile: str, top_n: int = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features based on importance scores, with quantile-specific adjustments.
    
    Args:
        X: Feature DataFrame
        feature_importances: Dictionary of feature importance scores
        quantile: Which quantile model ('median', 'lower', or 'upper')
        top_n: Number of top features to select (if None, uses adaptive threshold)
        
    Returns:
        Tuple of (selected features DataFrame, list of selected feature names)
    """
    # Sort features by importance
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    
    if top_n is None:
        # Adaptive threshold: keep features that explain 95% of total importance
        cumulative_importance = 0
        selected_features = []
        total_importance = sum(feature_importances.values())
        
        for feature, importance in sorted_features:
            cumulative_importance += importance
            selected_features.append(feature)
            if cumulative_importance / total_importance >= 0.95:
                break
    else:
        # Use fixed number of top features
        selected_features = [f[0] for f in sorted_features[:top_n]]
    
    # Quantile-specific adjustments
    if quantile == 'lower':
        # For lower quantile, emphasize features that might predict lower values
        # Add more weight to features related to negative trends or risk factors
        selected_features = [f for f in selected_features if not f.startswith('trend_positive')]
    elif quantile == 'upper':
        # For upper quantile, emphasize features that might predict higher values
        # Add more weight to features related to positive trends or growth factors
        selected_features = [f for f in selected_features if not f.startswith('trend_negative')]
    
    return X[selected_features], selected_features

def train_model(df_sales, df_stores):
    """Train a random forest model for revenue prediction."""
    # Clear any existing logs
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    
    logging.info("Starting model training...")
    logging.info(f"Input shapes - Sales: {df_sales.shape}, Stores: {df_stores.shape}")
    
    # Validate input data
    if df_sales.empty or df_stores.empty:
        raise ValueError("Empty input data provided")
    
    if 'revenue' not in df_sales.columns:
        raise ValueError("Missing 'revenue' column in sales data")
    
    # Calculate store_dayofweek_avg for each row (move this up)
    df_sales['day_of_week'] = pd.to_datetime(df_sales['date']).dt.dayofweek
    store_dayofweek_avg = df_sales.groupby(['store', 'day_of_week'])['revenue'].transform('mean')
    df_sales['store_dayofweek_avg'] = store_dayofweek_avg

    # Outlier handling: clip revenue to 95th percentile per store-month
    df_sales['month'] = pd.to_datetime(df_sales['date']).dt.month
    revenue_95 = df_sales.groupby(['store', 'month'])['revenue'].transform(lambda x: x.quantile(0.95))
    df_sales['revenue'] = np.minimum(df_sales['revenue'], revenue_95)
    # Remove low-activity days
    df_sales = df_sales[df_sales['store_dayofweek_avg'] >= 100]
    
    # Analyze revenue distribution
    logging.info("\nRevenue Distribution Analysis:")
    revenue_stats = df_sales['revenue'].describe()
    logging.info("\nRevenue Statistics:")
    logging.info(f"Count: {revenue_stats['count']:.0f}")
    logging.info(f"Mean: {revenue_stats['mean']:.2f}")
    logging.info(f"Std: {revenue_stats['std']:.2f}")
    logging.info(f"Min: {revenue_stats['min']:.2f}")
    logging.info(f"25%: {revenue_stats['25%']:.2f}")
    logging.info(f"50%: {revenue_stats['50%']:.2f}")
    logging.info(f"75%: {revenue_stats['75%']:.2f}")
    logging.info(f"Max: {revenue_stats['max']:.2f}")
    
    # Check for missing values
    missing_revenue = df_sales['revenue'].isnull().sum()
    if missing_revenue > 0:
        logging.warning(f"Found {missing_revenue} missing values in revenue column")
        # Fill missing values with median revenue for that store
        df_sales['revenue'] = df_sales.groupby('store')['revenue'].transform(
            lambda x: x.fillna(x.median())
        )
        logging.info("Filled missing revenue values with store-specific medians")
    
    # Check for zeros and extreme outliers
    zero_revenue = (df_sales['revenue'] == 0).sum()
    if zero_revenue > 0:
        logging.warning(f"Found {zero_revenue} zero revenue entries")
    
    # Calculate outlier thresholds using IQR method
    Q1 = revenue_stats['25%']
    Q3 = revenue_stats['75%']
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_sales[(df_sales['revenue'] < lower_bound) | (df_sales['revenue'] > upper_bound)]
    if len(outliers) > 0:
        logging.warning(f"Found {len(outliers)} revenue outliers")
        logging.info(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        # Cap outliers instead of removing them
        df_sales['revenue'] = df_sales['revenue'].clip(lower=lower_bound, upper=upper_bound)
        logging.info("Capped revenue outliers using IQR method")
    
    # Log revenue distribution after cleaning
    logging.info("\nRevenue Distribution After Cleaning:")
    clean_stats = df_sales['revenue'].describe()
    logging.info(f"Min: {clean_stats['min']:.2f}")
    logging.info(f"Max: {clean_stats['max']:.2f}")
    logging.info(f"Mean: {clean_stats['mean']:.2f}")
    
    # Train Random Forest to predict revenue_ratio = revenue / store_dayofweek_avg
    revenue_ratio = df_sales['revenue'] / df_sales['store_dayofweek_avg']
    y = revenue_ratio

    # Calculate revenue percentile for classification-style prediction
    df_sales['revenue_percentile'] = df_sales.groupby('store')['revenue'].rank(pct=True)
    logging.info(f"Sample revenue_percentile values: {df_sales['revenue_percentile'].head()}")

    # Derive features
    logging.info("Deriving features...")
    X, features = derive_features(df_sales, df_stores, is_prediction=False)
    
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Number of features: {len(features)}")
    
    # Sort data by date
    df_sales['date'] = pd.to_datetime(df_sales['date'])
    date_order = df_sales['date'].values
    sort_idx = np.argsort(date_order)
    X = X.iloc[sort_idx]
    y = y.iloc[sort_idx]
    store_ids = df_sales['store'].values[sort_idx]
    
    # Calculate sample weights
    sample_weights = calculate_sample_weights(df_sales.iloc[sort_idx])
    logging.info("Sample weights calculated with log-based decay and day-of-week weighting")

    # Instantiate Random Forest models for median, lower, and upper quantiles
    rf_median = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_lower = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_upper = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)

    # Select features (skip SHAP and pruning for simplicity)
    selected_features = X.columns.tolist()
    X_selected = X[selected_features]

    # Fit rf_median before feature selection
    rf_median.fit(X_selected, y)

    # Drop low-importance features, but always keep key business features
    key_features = ['revenue_lag_7', 'revenue_rolling_mean_7d', 'discount_pct', 'weekday_store_avg']
    X_selected, selected_features = drop_low_importance_features(rf_median, X_selected, threshold=0.001, keep_features=key_features)

    # Re-split after feature selection
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )

    # Retrain models on reduced feature set
    rf_median.fit(X_train, y_train)
    median_pred_train = rf_median.predict(X_train)
    median_pred_test = rf_median.predict(X_test)
    residuals_train = y_train - median_pred_train
    residuals_test = y_test - median_pred_test

    lower_targets_train = np.clip(residuals_train, a_max=0, a_min=None) * -1
    rf_lower.fit(X_train, lower_targets_train)
    lower_pred_train = rf_lower.predict(X_train)
    lower_pred_test = rf_lower.predict(X_test)

    upper_targets_train = np.clip(residuals_train, a_min=0, a_max=None)
    rf_upper.fit(X_train, upper_targets_train)
    upper_pred_train = rf_upper.predict(X_train)
    upper_pred_test = rf_upper.predict(X_test)

    # Compute R² scores
    train_scores = {
        'median': r2_score(y_train, median_pred_train),
        'lower': r2_score(lower_targets_train, lower_pred_train),
        'upper': r2_score(upper_targets_train, upper_pred_train)
    }
    test_scores = {
        'median': r2_score(y_test, median_pred_test),
        'lower': r2_score(np.clip(y_test - median_pred_test, a_max=0, a_min=None) * -1, lower_pred_test),
        'upper': r2_score(np.clip(y_test - median_pred_test, a_min=0, a_max=None), upper_pred_test)
    }

    # Compute test set prediction intervals for metrics
    p10_test = median_pred_test - lower_pred_test
    p50_test = median_pred_test
    p90_test = median_pred_test + upper_pred_test

    # Calculate prediction interval metrics
    prediction_metrics = calculate_prediction_metrics(y_test, p10_test, p50_test, p90_test)
    
    # Save models
    joblib.dump({'median': rf_median, 'lower': rf_lower, 'upper': rf_upper}, 'models/brandA_model.pkl')
    logging.info("Saved Random Forest models to models/brandA_model.pkl")

    # Save feature names used for training
    with open('models/brandA_feature_names.json', 'w') as f:
        json.dump({'all_features': selected_features}, f)
    logging.info("Saved feature names to models/brandA_feature_names.json")

    # Get feature importances from the median model
    importances = rf_median.feature_importances_
    feature_importances = dict(zip(selected_features, importances))

    # Permutation importance on test set
    perm_result = permutation_importance(rf_median, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    perm_sorted_idx = perm_result.importances_mean.argsort()[::-1]
    top_20_perm = [(X_test.columns[i], perm_result.importances_mean[i]) for i in perm_sorted_idx[:20]]
    logging.info('Top 20 permutation importance features:')
    for feat, score in top_20_perm:
        logging.info(f'{feat}: {score:.4f}')
    
    return {
        'selected_features': selected_features,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'prediction_metrics': prediction_metrics,
        'feature_importances': feature_importances
    }
