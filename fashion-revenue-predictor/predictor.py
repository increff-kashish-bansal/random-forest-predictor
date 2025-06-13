import joblib
import json
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple
from scipy.special import inv_boxcox
from utils.feature_engineering import apply_cluster_specific_transforms
from utils.conformal_calibration import ConformalCalibrator
import logging

def predict_and_explain(df_X: pd.DataFrame) -> Dict:
    """
    Make predictions and generate SHAP explanations for input data.
    
    Args:
        df_X: DataFrame with features matching the model's expected input
        
    Returns:
        Dictionary containing:
        - p10, p50, p90 predictions with dynamic confidence intervals
        - SHAP values
        - Top 5 most important features
    """
    # Load models and features
    models = joblib.load('models/brandA_models.pkl')
    with open('models/brandA_feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    # Apply cluster-specific transformations
    if 'store_cluster' in df_X.columns:
        transformed_dfs = []
        for cluster in df_X['store_cluster'].unique():
            cluster_df = df_X[df_X['store_cluster'] == cluster].copy()
            transformed_df = apply_cluster_specific_transforms(cluster_df, cluster)
            transformed_dfs.append(transformed_df)
        
        # Combine transformed DataFrames
        df_X = pd.concat(transformed_dfs, axis=0)
        df_X = df_X.sort_index()  # Restore original order
    
    # Get predictions from each model using appropriate feature sets
    # 1. Median predictions (log scale)
    median_pred = np.expm1(models['median'].predict(df_X[feature_names['all_features']]))
    
    # 2. Lower tail predictions (residuals)
    lower_residuals = models['lower'].predict(df_X[feature_names['lower_features']])
    
    # 3. Upper tail predictions (residuals)
    upper_residuals = models['upper'].predict(df_X[feature_names['upper_features']])
    
    # Calculate initial prediction intervals
    p10 = median_pred - lower_residuals
    p50 = median_pred
    p90 = median_pred + upper_residuals
    
    # Ensure p10 < p50 < p90 for all predictions
    preds = np.vstack([p10, p50, p90])
    preds_sorted = np.sort(preds, axis=0)
    p10, p50, p90 = preds_sorted[0], preds_sorted[1], preds_sorted[2]
    
    # Apply conformal calibration
    calibrator = ConformalCalibrator(alpha=0.1)  # 90% coverage
    
    # Load historical data for calibration
    try:
        historical_data = pd.read_json('models/historical_predictions.json')
        calibrator.calibrate(
            y_true=historical_data['revenue'].values,
            p10=historical_data['p10'].values,
            p50=historical_data['p50'].values,
            p90=historical_data['p90'].values
        )
        
        # Apply calibration to new predictions
        p10, p50, p90 = calibrator.calibrate_predictions(p10, p50, p90)
    except FileNotFoundError:
        logging.warning("Historical predictions not found, skipping calibration")
    
    # Calculate SHAP values using median model
    explainer = shap.TreeExplainer(models['median'])
    shap_values = explainer.shap_values(df_X[feature_names['all_features']])
    
    # Get feature importance scores
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(zip(feature_names['all_features'], feature_importance))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    # Get top 5 features
    top_5_features = list(sorted_importance.items())[:5]
    
    # After all calibration and sorting, invert log1p transformation for all predictions
    p10 = np.expm1(p10)
    p50 = np.expm1(p50)
    p90 = np.expm1(p90)
    
    return {
        'p10': p10,
        'p50': p50,
        'p90': p90,
        'shap_values': shap_values,
        'top_5_features': top_5_features
    }

def calculate_prediction_metrics(y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> Dict:
    """
    Calculate prediction interval coverage and sharpness metrics.
    
    Args:
        y_true: Array of true values
        p10: Array of 10th percentile predictions
        p50: Array of 50th percentile predictions
        p90: Array of 90th percentile predictions
        
    Returns:
        Dictionary containing:
        - coverage: Percentage of true values falling within prediction intervals
        - sharpness: Average width of prediction intervals (p90 - p10)
        - rmse: Root mean squared error of median predictions
        - mae: Mean absolute error of median predictions
    """
    # Calculate coverage
    in_interval = np.logical_and(y_true >= p10, y_true <= p90)
    coverage = np.mean(in_interval) * 100
    
    # Calculate sharpness
    interval_width = p90 - p10
    sharpness = np.mean(interval_width)
    
    # Calculate RMSE and MAE for median predictions
    rmse = np.sqrt(np.mean((y_true - p50) ** 2))
    mae = np.mean(np.abs(y_true - p50))
    
    return {
        'coverage': coverage,
        'sharpness': sharpness,
        'rmse': rmse,
        'mae': mae
    }
