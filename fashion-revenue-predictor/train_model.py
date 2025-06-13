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
from utils.feature_selection import iterative_feature_pruning, select_features_by_global_shap
import lightgbm as lgb
import optuna  # Added for hyperparameter tuning

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
    
    # Calculate avg_day_month_revenue for each row
    df_sales['month'] = pd.to_datetime(df_sales['date']).dt.month
    df_sales['day_of_week'] = pd.to_datetime(df_sales['date']).dt.dayofweek
    avg_day_month = df_sales.groupby(['store', 'month', 'day_of_week'])['revenue'].transform('mean')
    df_sales['avg_day_month_revenue'] = avg_day_month
    # Relative log target
    y = np.log1p(df_sales['revenue'] / df_sales['avg_day_month_revenue'])

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
    
    # Apply global SHAP-based feature selection
    logging.info("\nApplying global SHAP-based feature selection...")
    selected_features = select_features_by_global_shap(
        X=X[selected_features],
        y=y,
        store_ids=store_ids,
        n_features=25,  # Keep top 25 stable features
        n_trees=100,
        n_splits=5
    )
    logging.info(f"Selected {len(selected_features)} features after global SHAP analysis")
    
    # Train final models with selected features
    models = {}
    cv_scores = {model_type: [] for model_type in ['median', 'lower', 'upper']}
    cv_metrics = []
    
    # Initialize GroupTimeSeriesSplit
    n_splits = 5
    gtscv = GroupTimeSeriesSplit(n_splits=n_splits, test_size=0.2)
    
    # LightGBM default parameters with regularization and early stopping
    lgb_base_params = {
        'objective': 'quantile',
        'metric': 'quantile',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 50,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    # If Optuna is available, tune LightGBM hyperparameters for the median model
    def tune_lgbm(X, y_log, groups, sample_weights):
        def objective(trial):
            params = lgb_base_params.copy()
            params.update({
                'num_leaves': trial.suggest_int('num_leaves', 16, 64),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
            })
            params['alpha'] = 0.5
            n_splits = 3
            gtscv = GroupTimeSeriesSplit(n_splits=n_splits, test_size=0.2)
            scores = []
            for train_idx, test_idx in gtscv.split(X, groups=groups):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_log[train_idx], y_log[test_idx]
                weights_train = sample_weights[train_idx]
                model = lgb.LGBMRegressor(**params, n_estimators=100)
                model.fit(
                    X_train, y_train,
                    sample_weight=weights_train,
                    eval_set=[(X_test, y_test)],
                    eval_sample_weight=[sample_weights[test_idx]],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                y_pred = model.predict(X_test)
                y_pred_orig = np.expm1(y_pred)
                y_test_orig = np.expm1(y_test)
                score = np.corrcoef(y_test_orig, y_pred_orig)[0,1]**2
                scores.append(score)
            return -np.mean(scores)  # Minimize negative R^2
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        best_params = lgb_base_params.copy()
        best_params.update(study.best_params)
        return best_params
    # Tune or use defaults
    try:
        best_lgb_params = tune_lgbm(X[selected_features], y, store_ids, sample_weights)
        logging.info(f"Optuna best LightGBM params: {best_lgb_params}")
    except Exception as e:
        logging.warning(f"Optuna tuning failed or not available, using defaults. Error: {e}")
        best_lgb_params = lgb_base_params.copy()
    # 1. Train median model
    logging.info("\nTraining median model with selected features and early stopping...")
    lgb_median_params = best_lgb_params.copy()
    lgb_median_params['alpha'] = 0.5
    median_model = lgb.LGBMRegressor(**lgb_median_params, n_estimators=100)
    for train_idx, test_idx in gtscv.split(X[selected_features], groups=store_ids):
        X_train, X_test = X[selected_features].iloc[train_idx], X[selected_features].iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        weights_train = sample_weights[train_idx]
        median_model.fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_test, y_test)],
            eval_sample_weight=[sample_weights[test_idx]],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        y_pred = median_model.predict(X_test)
        y_pred_orig = np.expm1(y_pred)
        y_test_orig = np.expm1(y_test)
        score = np.corrcoef(y_test_orig, y_pred_orig)[0,1]**2
        cv_scores['median'].append(score)
    median_model.fit(
        X[selected_features], y,
        sample_weight=sample_weights,
        eval_set=[(X[selected_features], y)],
        eval_sample_weight=[sample_weights],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    models['median'] = median_model
    median_preds = np.expm1(median_model.predict(X[selected_features]))
    lower_targets = np.clip(median_preds - y, 0, None)
    upper_targets = np.clip(y - median_preds, 0, None)
    importances = dict(zip(selected_features, median_model.feature_importances_))
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    X_lower, lower_features = select_features_by_importance(X[selected_features], sorted_importances, 'lower')
    X_upper, upper_features = select_features_by_importance(X[selected_features], sorted_importances, 'upper')
    # Clean lower_targets and upper_targets for NaN/inf (after X_lower/X_upper are defined)
    lower_mask = np.isfinite(lower_targets)
    upper_mask = np.isfinite(upper_targets)
    n_lower_removed = np.sum(~lower_mask)
    n_upper_removed = np.sum(~upper_mask)
    if n_lower_removed > 0:
        logging.warning(f"Removed {n_lower_removed} rows with NaN/inf in lower_targets for quantile model training.")
    if n_upper_removed > 0:
        logging.warning(f"Removed {n_upper_removed} rows with NaN/inf in upper_targets for quantile model training.")
    X_lower_clean = X_lower[lower_mask]
    lower_targets_clean = lower_targets[lower_mask]
    X_upper_clean = X_upper[upper_mask]
    upper_targets_clean = upper_targets[upper_mask]
    sample_weights_lower = sample_weights[lower_mask]
    sample_weights_upper = sample_weights[upper_mask]
    # 2. Train lower tail model on residuals
    logging.info("\nTraining lower tail model with LightGBM quantile regression and early stopping...")
    lgb_lower_params = best_lgb_params.copy()
    lgb_lower_params['alpha'] = 0.05
    lower_model = lgb.LGBMRegressor(**lgb_lower_params, n_estimators=100)
    for train_idx, test_idx in gtscv.split(X_lower_clean, groups=store_ids[lower_mask]):
        X_train, X_test = X_lower_clean.iloc[train_idx], X_lower_clean.iloc[test_idx]
        y_train, y_test = lower_targets_clean[train_idx], lower_targets_clean[test_idx]
        weights_train = sample_weights_lower[train_idx]
        lower_model.fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_test, y_test)],
            eval_sample_weight=[sample_weights_lower[test_idx]],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        y_pred = lower_model.predict(X_test)
        score = np.corrcoef(y_test, y_pred)[0,1]**2
        cv_scores['lower'].append(score)
    lower_model.fit(
        X_lower_clean, lower_targets_clean,
        sample_weight=sample_weights_lower,
        eval_set=[(X_lower_clean, lower_targets_clean)],
        eval_sample_weight=[sample_weights_lower],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    models['lower'] = lower_model
    # 3. Train upper tail model on residuals
    logging.info("\nTraining upper tail model with LightGBM quantile regression and early stopping...")
    lgb_upper_params = best_lgb_params.copy()
    lgb_upper_params['alpha'] = 0.95
    upper_model = lgb.LGBMRegressor(**lgb_upper_params, n_estimators=100)
    for train_idx, test_idx in gtscv.split(X_upper_clean, groups=store_ids[upper_mask]):
        X_train, X_test = X_upper_clean.iloc[train_idx], X_upper_clean.iloc[test_idx]
        y_train, y_test = upper_targets_clean[train_idx], upper_targets_clean[test_idx]
        weights_train = sample_weights_upper[train_idx]
        upper_model.fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_test, y_test)],
            eval_sample_weight=[sample_weights_upper[test_idx]],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        y_pred = upper_model.predict(X_test)
        score = np.corrcoef(y_test, y_pred)[0,1]**2
        cv_scores['upper'].append(score)
    upper_model.fit(
        X_upper_clean, upper_targets_clean,
        sample_weight=sample_weights_upper,
        eval_set=[(X_upper_clean, upper_targets_clean)],
        eval_sample_weight=[sample_weights_upper],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    models['upper'] = upper_model
    
    # Calculate prediction interval metrics
    historical_predictions = []
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
        
        # Store predictions for calibration
        fold_predictions = pd.DataFrame({
            'revenue': y_test,
            'p10': p10,
            'p50': p50,
            'p90': p90
        })
        historical_predictions.append(fold_predictions)
        
        # Calculate metrics
        fold_metrics = calculate_prediction_metrics(y_test, p10, p50, p90)
        cv_metrics.append(fold_metrics)
    
    # Combine and save historical predictions
    historical_predictions_df = pd.concat(historical_predictions, axis=0)
    historical_predictions_df.to_json('models/historical_predictions.json')
    
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
    
    # 2. After model training, split last 10% of data as test set by date
    n_test = int(0.1 * len(X))
    test_idx = np.arange(len(X) - n_test, len(X))
    train_idx = np.arange(0, len(X) - n_test)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Predict on test set
    p50_test = np.expm1(median_model.predict(X_test[selected_features]))
    lower_resid_test = lower_model.predict(X_test[lower_features])
    upper_resid_test = upper_model.predict(X_test[upper_features])
    p10_test = p50_test - lower_resid_test
    p90_test = p50_test + upper_resid_test
    # Ensure p10 < p50 < p90
    p10_test, p50_test, p90_test = np.sort(np.vstack([p10_test, p50_test, p90_test]), axis=0)
    # Calculate metrics
    future_metrics = calculate_prediction_metrics(y_test, p10_test, p50_test, p90_test)
    logging.info(f"Future slice metrics (last 10%): {future_metrics}")
    
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
        'r2_progression': r2_scores,
        'future_metrics': future_metrics
    }
