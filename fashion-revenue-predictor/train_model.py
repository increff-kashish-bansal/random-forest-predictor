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
import shap
import os

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
    
    logging.info("="*50)
    logging.info("Starting model training process")
    logging.info("="*50)
    logging.info(f"Input shapes - Sales: {df_sales.shape}, Stores: {df_stores.shape}")
    logging.info(f"Date range: {df_sales['date'].min()} to {df_sales['date'].max()}")
    logging.info(f"Number of unique stores: {df_sales['store'].nunique()}")
    
    # Validate input data
    if df_sales.empty or df_stores.empty:
        raise ValueError("Empty input data provided")
    
    if 'revenue' not in df_sales.columns:
        raise ValueError("Missing 'revenue' column in sales data")
    
    logging.info("\nCalculating store-day-of-week averages...")
    # Calculate store_dayofweek_avg for each row (move this up)
    df_sales['day_of_week'] = pd.to_datetime(df_sales['date']).dt.dayofweek
    store_dayofweek_avg = df_sales.groupby(['store', 'day_of_week'])['revenue'].transform('mean')
    df_sales['store_dayofweek_avg'] = store_dayofweek_avg
    
    # Log day-of-week statistics
    dow_stats = df_sales.groupby('day_of_week')['revenue'].agg(['mean', 'std', 'count'])
    logging.info("\nDay-of-week revenue statistics:")
    for dow, stats in dow_stats.iterrows():
        logging.info(f"Day {dow}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Count={stats['count']}")

    logging.info("\nHandling outliers and low-activity days...")
    # Outlier handling: clip revenue to 95th percentile per store-month
    df_sales['month'] = pd.to_datetime(df_sales['date']).dt.month
    revenue_95 = df_sales.groupby(['store', 'month'])['revenue'].transform(lambda x: x.quantile(0.95))
    df_sales['revenue'] = np.minimum(df_sales['revenue'], revenue_95)
    
    # Log outlier statistics
    outlier_stats = df_sales.groupby('month')['revenue'].agg(['mean', 'std', 'min', 'max'])
    logging.info("\nMonthly revenue statistics after outlier handling:")
    for month, stats in outlier_stats.iterrows():
        logging.info(f"Month {month}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, "
                    f"Min={stats['min']:.2f}, Max={stats['max']:.2f}")
    
    # Remove low-activity days
    initial_rows = len(df_sales)
    df_sales = df_sales[df_sales['store_dayofweek_avg'] >= 100]
    removed_rows = initial_rows - len(df_sales)
    logging.info(f"Removed {removed_rows} low-activity days (revenue < 100)")
    logging.info(f"Remaining data points: {len(df_sales)}")
    
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
        df_sales['revenue'] = df_sales.groupby('store')['revenue'].transform(lambda x: x.fillna(x.median()))
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
    logging.info(f"Count: {clean_stats['count']:.0f}")
    logging.info(f"Mean: {clean_stats['mean']:.2f}")
    logging.info(f"Std: {clean_stats['std']:.2f}")
    logging.info(f"Min: {clean_stats['min']:.2f}")
    logging.info(f"25%: {clean_stats['25%']:.2f}")
    logging.info(f"50%: {clean_stats['50%']:.2f}")
    logging.info(f"75%: {clean_stats['75%']:.2f}")
    logging.info(f"Max: {clean_stats['max']:.2f}")
    
    logging.info("\nStarting feature engineering...")
    # Feature engineering
    df_features = derive_features(df_sales, df_stores)
    logging.info(f"Generated {df_features.shape[1]} features")
    
    # Log feature types and missing values
    feature_types = df_features.dtypes.value_counts()
    logging.info("\nFeature Types:")
    for dtype, count in feature_types.items():
        logging.info(f"{dtype}: {count} features")
    
    missing_values = df_features.isnull().sum()
    if missing_values.any():
        logging.warning("\nFeatures with missing values:")
        for col in missing_values[missing_values > 0].index:
            logging.warning(f"{col}: {missing_values[col]} missing values")
    
    logging.info("\nPreparing data for training...")
    # Prepare data for training
    X = df_features.drop(['revenue', 'date', 'store'], axis=1)
    y = df_features['revenue']
    
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Target variable shape: {y.shape}")
    
    # Calculate sample weights
    logging.info("\nCalculating sample weights...")
    sample_weights = calculate_sample_weights(df_features)
    logging.info(f"Sample weights range: {sample_weights.min():.4f} to {sample_weights.max():.4f}")
    logging.info(f"Sample weights mean: {sample_weights.mean():.4f}")
    logging.info(f"Sample weights std: {sample_weights.std():.4f}")
    
    # Split data
    logging.info("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )
    logging.info(f"Train set size: {X_train.shape[0]} samples")
    logging.info(f"Test set size: {X_test.shape[0]} samples")
    
    # Log train/test set statistics
    logging.info("\nTrain set statistics:")
    logging.info(f"Mean revenue: {y_train.mean():.2f}")
    logging.info(f"Std revenue: {y_train.std():.2f}")
    logging.info(f"Min revenue: {y_train.min():.2f}")
    logging.info(f"Max revenue: {y_train.max():.2f}")
    
    logging.info("\nTest set statistics:")
    logging.info(f"Mean revenue: {y_test.mean():.2f}")
    logging.info(f"Std revenue: {y_test.std():.2f}")
    logging.info(f"Min revenue: {y_test.min():.2f}")
    logging.info(f"Max revenue: {y_test.max():.2f}")
    
    logging.info("\nStarting model training...")
    # Train model
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    logging.info("\nModel parameters:")
    for param, value in model_params.items():
        logging.info(f"{param}: {value}")
    
    model = RandomForestRegressor(**model_params)
    
    logging.info("\nFitting Random Forest model...")
    logging.info("Training progress:")
    for i in range(0, model_params['n_estimators'], 10):
        model.n_estimators = i + 10
        model.fit(X_train, y_train, sample_weight=weights_train)
        train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        logging.info(f"Trees {i+1}-{i+10}: Train R² = {train_r2:.4f}")
    
    logging.info("Model training completed")
    
    # Feature importance
    logging.info("\nCalculating feature importance...")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Log cumulative feature importance
    cumulative_importance = 0
    logging.info("\nCumulative feature importance:")
    for idx, row in feature_importance.iterrows():
        cumulative_importance += row['importance']
        if cumulative_importance <= 0.95:  # Log until we reach 95% of total importance
            logging.info(f"{row['feature']}: {row['importance']:.4f} (Cumulative: {cumulative_importance:.4f})")
    
    logging.info("\nTop 20 most important features:")
    for idx, row in feature_importance.head(20).iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")
    
    # Model evaluation
    logging.info("\nEvaluating model performance...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    logging.info(f"Train R² score: {train_r2:.4f}")
    logging.info(f"Test R² score: {test_r2:.4f}")
    
    # Calculate and log prediction metrics
    logging.info("\nCalculating prediction metrics...")
    metrics = calculate_prediction_metrics(y_test, test_pred)
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.4f}")
    
    # Log prediction error distribution
    test_errors = y_test - test_pred
    error_stats = pd.Series(test_errors).describe()
    logging.info("\nTest set prediction error distribution:")
    logging.info(f"Mean error: {error_stats['mean']:.2f}")
    logging.info(f"Std error: {error_stats['std']:.2f}")
    logging.info(f"Min error: {error_stats['min']:.2f}")
    logging.info(f"25% error: {error_stats['25%']:.2f}")
    logging.info(f"50% error: {error_stats['50%']:.2f}")
    logging.info(f"75% error: {error_stats['75%']:.2f}")
    logging.info(f"Max error: {error_stats['max']:.2f}")
    
    # Log error distribution by store
    logging.info("\nError distribution by store:")
    store_errors = pd.DataFrame({
        'store': df_features.loc[X_test.index, 'store'],
        'error': test_errors
    })
    store_error_stats = store_errors.groupby('store')['error'].agg(['mean', 'std', 'count'])
    for store, stats in store_error_stats.iterrows():
        logging.info(f"Store {store}: Mean error = {stats['mean']:.2f}, Std = {stats['std']:.2f}, "
                    f"Count = {stats['count']}")
    
    logging.info("\nSaving model and feature names...")
    # Save model and feature names
    model_path = Path('models')
    model_path.mkdir(exist_ok=True)
    
    joblib.dump(model, model_path / 'model.joblib')
    with open(model_path / 'feature_names.json', 'w') as f:
        json.dump(list(X.columns), f)
    
    logging.info("="*50)
    logging.info("Model training and saving completed successfully")
    logging.info("="*50)
    return model, X.columns
