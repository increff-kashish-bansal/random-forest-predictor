import pickle
import json
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple
from scipy.special import inv_boxcox

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
    # Load model and features
    with open('models/brandA_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/brandA_features.json', 'r') as f:
        required_features = json.load(f)
    
    # Load Box-Cox lambda parameter
    with open('models/boxcox_lambda.json', 'r') as f:
        lambda_ = json.load(f)['lambda']
    
    # Ensure all required features are present
    missing_features = set(required_features) - set(df_X.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Get predictions from each tree
    predictions = []
    for estimator in model.estimators_:
        pred = estimator.predict(df_X[required_features])
        predictions.append(pred)
    
    # Stack predictions
    predictions = np.stack(predictions, axis=1)
    
    # Calculate base predictions
    base_predictions = np.median(predictions, axis=1)
    
    # Calculate prediction uncertainty based on:
    # 1. Standard deviation of tree predictions
    # 2. Store's historical volatility (revenue_std)
    # 3. Whether it's a weekend/holiday
    uncertainty = np.std(predictions, axis=1)
    
    # Add store volatility component if available
    if 'revenue_std' in df_X.columns and 'revenue_mean' in df_X.columns:
        # Avoid division by zero by using a small epsilon
        epsilon = 1e-6
        revenue_mean = np.maximum(df_X['revenue_mean'].values, epsilon)
        uncertainty *= (1 + df_X['revenue_std'].values / revenue_mean)
    
    # Add weekend effect if available
    if 'is_weekend' in df_X.columns:
        weekend_multiplier = 1.2  # Increase uncertainty for weekends
        uncertainty *= (1 + (df_X['is_weekend'].values * (weekend_multiplier - 1)))
    
    # Calculate prediction intervals
    p10 = base_predictions - 1.28 * uncertainty
    p50 = base_predictions
    p90 = base_predictions + 1.28 * uncertainty
    
    # Inverse transform predictions
    p10 = inv_boxcox(p10, lambda_) - 1e-3
    p50 = inv_boxcox(p50, lambda_) - 1e-3
    p90 = inv_boxcox(p90, lambda_) - 1e-3
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_X[required_features])
    
    # Get feature importance scores
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(zip(required_features, feature_importance))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    # Get top 5 features
    top_5_features = list(sorted_importance.items())[:5]
    
    return {
        'p10': p10,
        'p50': p50,
        'p90': p90,
        'shap_values': shap_values,
        'top_5_features': top_5_features
    }
