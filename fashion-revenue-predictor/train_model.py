import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from utils.feature_engineering import derive_features
from scipy.stats import boxcox
import joblib
from sklearn.preprocessing import OneHotEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

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
    logging.info("Starting model training...")
    logging.info(f"Input shapes - Sales: {df_sales.shape}, Stores: {df_stores.shape}")
    
    # Derive features
    X, features = derive_features(df_sales, df_stores, is_prediction=False)
    y = df_sales['revenue'].values
    
    # Sort data by date
    df_sales['date'] = pd.to_datetime(df_sales['date'])
    date_order = df_sales['date'].values
    sort_idx = np.argsort(date_order)
    X = X.iloc[sort_idx]
    y = y[sort_idx]
    
    # Calculate sample weights
    sample_weights = calculate_sample_weights(df_sales.iloc[sort_idx])
    logging.info("Sample weights calculated with log-based decay and day-of-week weighting")
    
    # Initialize TimeSeriesSplit with fewer splits for faster training
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Common model parameters
    model_params = {
        'n_estimators': 50,
        'max_depth': 8,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }
    
    # Train three separate models for different quantiles
    models = {}
    cv_scores = {model_type: [] for model_type in ['median', 'lower', 'upper']}
    selected_features = {}
    
    # 1. Train median model first to get feature importances
    logging.info("Training median model with log transformation...")
    y_log = np.log1p(y)
    median_model = RandomForestRegressor(**model_params)
    
    # Perform time series cross-validation for median model
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_log[train_idx], y_log[test_idx]
        weights_train = sample_weights[train_idx]
        
        median_model.fit(X_train, y_train, sample_weight=weights_train)
        y_pred = median_model.predict(X_test)
        score = np.corrcoef(y_test, y_pred)[0,1]**2
        cv_scores['median'].append(score)
    
    # Train final median model on all data
    median_model.fit(X, y_log, sample_weight=sample_weights)
    models['median'] = median_model
    
    # Get feature importances from median model
    importances = dict(zip(X.columns, median_model.feature_importances_))
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    # Select features for lower and upper quantile models
    X_lower, selected_features['lower'] = select_features_by_importance(X, sorted_importances, 'lower')
    X_upper, selected_features['upper'] = select_features_by_importance(X, sorted_importances, 'upper')
    
    # 2. Train lower tail model with selected features
    logging.info("Training lower tail model with selected features...")
    lower_model = RandomForestRegressor(**model_params)
    
    for train_idx, test_idx in tscv.split(X_lower):
        X_train, X_test = X_lower.iloc[train_idx], X_lower.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        weights_train = sample_weights[train_idx]
        
        lower_model.fit(X_train, y_train, sample_weight=weights_train)
        y_pred = lower_model.predict(X_test)
        score = np.corrcoef(y_test, y_pred)[0,1]**2
        cv_scores['lower'].append(score)
    
    lower_model.fit(X_lower, y, sample_weight=sample_weights)
    models['lower'] = lower_model
    
    # 3. Train upper tail model with selected features
    logging.info("Training upper tail model with selected features...")
    upper_model = RandomForestRegressor(**model_params)
    
    for train_idx, test_idx in tscv.split(X_upper):
        X_train, X_test = X_upper.iloc[train_idx], X_upper.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        weights_train = sample_weights[train_idx]
        
        upper_model.fit(X_train, y_train, sample_weight=weights_train)
        y_pred = upper_model.predict(X_test)
        score = np.corrcoef(y_test, y_pred)[0,1]**2
        cv_scores['upper'].append(score)
    
    upper_model.fit(X_upper, y, sample_weight=sample_weights)
    models['upper'] = upper_model
    
    # Log cross-validation scores and feature counts
    logging.info("Cross-validation scores and feature counts:")
    for model_type in ['median', 'lower', 'upper']:
        mean_score = np.mean(cv_scores[model_type])
        std_score = np.std(cv_scores[model_type])
        feature_count = len(X.columns) if model_type == 'median' else len(selected_features[model_type])
        logging.info(f"{model_type} - Mean R²: {mean_score:.3f} (±{std_score:.3f}), Features: {feature_count}")
    
    # Print top 5 features for each model
    print("\nTop 5 Most Important Features by Model:")
    for model_type in ['median', 'lower', 'upper']:
        if model_type == 'median':
            model_features = sorted_importances
        else:
            model_features = dict(zip(selected_features[model_type], 
                                    models[model_type].feature_importances_))
            model_features = dict(sorted(model_features.items(), key=lambda x: x[1], reverse=True))
        
        print(f"\n{model_type.upper()} Model:")
        for feat, imp in list(model_features.items())[:5]:
            print(f"{feat}: {imp:.3f}")
    
    # Save models, features, and selected features
    Path('models').mkdir(exist_ok=True)
    model_path = 'models/brandA_models.pkl'
    features_path = 'models/brandA_features.json'
    selected_features_path = 'models/brandA_selected_features.json'
    feature_names_path = 'models/brandA_feature_names.json'
    
    # Save feature names for consistent encoding
    feature_names = {
        'all_features': list(X.columns),
        'lower_features': selected_features['lower'],
        'upper_features': selected_features['upper']
    }
    
    joblib.dump(models, model_path, compress=3)
    with open(features_path, 'w') as f:
        json.dump(features, f)
    with open(selected_features_path, 'w') as f:
        json.dump(selected_features, f)
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)
    
    print(f"\nModels saved to: {model_path}")
    print(f"Features saved to: {features_path}")
    print(f"Selected features saved to: {selected_features_path}")
    print(f"Feature names saved to: {feature_names_path}")
    
    logging.info("Models and features saved successfully")
    
    return {
        'cv_scores': cv_scores,
        'feature_importances': sorted_importances,
        'selected_features': selected_features,
        'feature_names': feature_names,
        'train_scores': {
            'median': np.mean(cv_scores['median']),
            'lower': np.mean(cv_scores['lower']),
            'upper': np.mean(cv_scores['upper'])
        },
        'test_scores': {
            'median': np.mean(cv_scores['median']),
            'lower': np.mean(cv_scores['upper']),
            'upper': np.mean(cv_scores['upper'])
        }
    }
