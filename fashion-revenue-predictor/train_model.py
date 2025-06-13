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
from utils.feature_selection import iterative_feature_pruning

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
    
    try:
        # Derive features
        logging.info("Deriving features...")
        X, features = derive_features(df_sales, df_stores, is_prediction=False)
        y = df_sales['revenue']
        
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
        
        # Common model parameters
        model_params = {
            'n_estimators': 100,  # Increased from 50 for better feature importance estimation
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
        
        # Apply iterative feature pruning
        logging.info("\nStarting iterative feature pruning...")
        best_model, selected_features, r2_scores = iterative_feature_pruning(
            X=X,
            y=y,
            model_params=model_params,
            n_iter=10
        )
        
        logging.info(f"Selected {len(selected_features)} features after pruning")
        
        # Train final models with selected features
        models = {}
        cv_scores = {model_type: [] for model_type in ['median', 'lower', 'upper']}
        cv_metrics = []
        
        # Initialize GroupTimeSeriesSplit
        n_splits = 5
        gtscv = GroupTimeSeriesSplit(n_splits=n_splits, test_size=0.2)
        
        # 1. Train median model
        logging.info("\nTraining median model with selected features...")
        y_log = np.log1p(y)
        median_model = RandomForestRegressor(**model_params)
        
        for train_idx, test_idx in gtscv.split(X[selected_features], groups=store_ids):
            X_train, X_test = X[selected_features].iloc[train_idx], X[selected_features].iloc[test_idx]
            y_train, y_test = y_log[train_idx], y_log[test_idx]
            weights_train = sample_weights[train_idx]
            
            median_model.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = median_model.predict(X_test)
            
            # Transform predictions and true values back to original scale
            y_pred_orig = np.expm1(y_pred)
            y_test_orig = np.expm1(y_test)
            
            # Calculate R² on original scale
            score = np.corrcoef(y_test_orig, y_pred_orig)[0,1]**2
            cv_scores['median'].append(score)
        
        # Train final median model on all data
        median_model.fit(X[selected_features], y_log, sample_weight=sample_weights)
        models['median'] = median_model
        
        # Get median predictions for residual calculation
        median_preds = np.expm1(median_model.predict(X[selected_features]))
        
        # Calculate residuals for lower and upper quantile models
        lower_targets = np.clip(median_preds - y, 0, None)  # Positive residuals for lower bound
        upper_targets = np.clip(y - median_preds, 0, None)  # Positive residuals for upper bound
        
        # Get feature importances from median model
        importances = dict(zip(selected_features, median_model.feature_importances_))
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        # Select features for lower and upper quantile models
        X_lower, lower_features = select_features_by_importance(X[selected_features], sorted_importances, 'lower')
        X_upper, upper_features = select_features_by_importance(X[selected_features], sorted_importances, 'upper')
        
        # 2. Train lower tail model on residuals
        logging.info("\nTraining lower tail model on residuals...")
        lower_model = RandomForestRegressor(**model_params)
        
        for train_idx, test_idx in gtscv.split(X_lower, groups=store_ids):
            X_train, X_test = X_lower.iloc[train_idx], X_lower.iloc[test_idx]
            y_train, y_test = lower_targets[train_idx], lower_targets[test_idx]
            weights_train = sample_weights[train_idx]
            
            lower_model.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = lower_model.predict(X_test)
            score = np.corrcoef(y_test, y_pred)[0,1]**2
            cv_scores['lower'].append(score)
        
        lower_model.fit(X_lower, lower_targets, sample_weight=sample_weights)
        models['lower'] = lower_model
        
        # 3. Train upper tail model on residuals
        logging.info("\nTraining upper tail model on residuals...")
        upper_model = RandomForestRegressor(**model_params)
        
        for train_idx, test_idx in gtscv.split(X_upper, groups=store_ids):
            X_train, X_test = X_upper.iloc[train_idx], X_upper.iloc[test_idx]
            y_train, y_test = upper_targets[train_idx], upper_targets[test_idx]
            weights_train = sample_weights[train_idx]
            
            upper_model.fit(X_train, y_train, sample_weight=weights_train)
            y_pred = upper_model.predict(X_test)
            score = np.corrcoef(y_test, y_pred)[0,1]**2
            cv_scores['upper'].append(score)
        
        upper_model.fit(X_upper, upper_targets, sample_weight=sample_weights)
        models['upper'] = upper_model
        
        # Calculate prediction interval metrics
        for train_idx, test_idx in gtscv.split(X[selected_features], groups=store_ids):
            X_train, X_test = X[selected_features].iloc[train_idx], X[selected_features].iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Get predictions for this fold
            p50 = np.expm1(median_model.predict(X_test[selected_features]))
            lower_residuals = lower_model.predict(X_test[lower_features])
            upper_residuals = upper_model.predict(X_test[upper_features])
            
            # Calculate final predictions
            p10 = p50 - lower_residuals
            p90 = p50 + upper_residuals
            
            # Calculate metrics
            fold_metrics = calculate_prediction_metrics(y_test, p10, p50, p90)
            cv_metrics.append(fold_metrics)
        
        # Calculate average metrics
        avg_metrics = {
            'coverage': np.mean([m['coverage'] for m in cv_metrics]),
            'sharpness': np.mean([m['sharpness'] for m in cv_metrics]),
            'rmse': np.mean([m['rmse'] for m in cv_metrics]),
            'mae': np.mean([m['mae'] for m in cv_metrics])
        }
        
        # Log cross-validation scores and metrics
        logging.info("\nCross-validation scores and metrics:")
        for model_type in ['median', 'lower', 'upper']:
            mean_score = np.mean(cv_scores[model_type])
            std_score = np.std(cv_scores[model_type])
            feature_count = len(selected_features) if model_type == 'median' else len(lower_features if model_type == 'lower' else upper_features)
            logging.info(f"{model_type} - Mean R²: {mean_score:.3f} (±{std_score:.3f}), Features: {feature_count}")
        
        logging.info("\nPrediction Interval Metrics:")
        logging.info(f"Coverage: {avg_metrics['coverage']:.1f}%")
        logging.info(f"Sharpness: {avg_metrics['sharpness']:.2f}")
        logging.info(f"RMSE: {avg_metrics['rmse']:.2f}")
        logging.info(f"MAE: {avg_metrics['mae']:.2f}")
        
        # Save models and features
        Path('models').mkdir(exist_ok=True)
        model_path = 'models/brandA_models.pkl'
        features_path = 'models/brandA_features.json'
        selected_features_path = 'models/brandA_selected_features.json'
        feature_names_path = 'models/brandA_feature_names.json'
        
        # Save models
        joblib.dump(models, model_path, compress=3)
        
        # Save feature information
        feature_info = {
            'all_features': selected_features,
            'lower_features': lower_features,
            'upper_features': upper_features
        }
        
        with open(features_path, 'w') as f:
            json.dump(features, f)
        with open(selected_features_path, 'w') as f:
            json.dump(feature_info, f)
        with open(feature_names_path, 'w') as f:
            json.dump(feature_info, f)
        
        logging.info("Models and features saved successfully")
        
        return {
            'cv_scores': cv_scores,
            'feature_importances': sorted_importances,
            'selected_features': feature_info,
            'feature_names': feature_info,
            'train_scores': {
                'median': np.mean(cv_scores['median']),
                'lower': np.mean(cv_scores['lower']),
                'upper': np.mean(cv_scores['upper'])
            },
            'test_scores': {
                'median': np.mean(cv_scores['median']),
                'lower': np.mean(cv_scores['lower']),
                'upper': np.mean(cv_scores['upper'])
            },
            'prediction_metrics': avg_metrics,
            'r2_progression': r2_scores
        }
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        logging.error("Full traceback:")
        import traceback
        logging.error(traceback.format_exc())
        raise
