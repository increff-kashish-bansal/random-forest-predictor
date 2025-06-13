import pickle
import json
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple

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
    if 'revenue_std' in df_X.columns:
        uncertainty *= (1 + df_X['revenue_std'].values / df_X['revenue_mean'].values)
    if 'is_weekend' in df_X.columns:
        uncertainty *= (1 + 0.2 * df_X['is_weekend'].values)  # 20% more uncertainty on weekends
    
    # Calculate dynamic percentiles
    pred_dict = {
        'p50': base_predictions,
        'p10': base_predictions - 1.28 * uncertainty,  # 1.28 is the z-score for 10th percentile
        'p90': base_predictions + 1.28 * uncertainty   # 1.28 is the z-score for 90th percentile
    }
    
    # Ensure predictions are non-negative
    for key in pred_dict:
        pred_dict[key] = np.maximum(pred_dict[key], 0)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_X[required_features])
    
    # Get top 5 features by absolute SHAP value
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_5_idx = np.argsort(mean_shap)[-5:][::-1]
    top_5_features = [(required_features[i], mean_shap[i]) for i in top_5_idx]
    
    # Convert SHAP values to list for JSON serialization
    shap_list = shap_values.tolist()
    
    return {
        'p10': pred_dict['p10'].tolist(),
        'p50': pred_dict['p50'].tolist(),
        'p90': pred_dict['p90'].tolist(),
        'shap_values': shap_list,
        'top_5_features': top_5_features
    }
