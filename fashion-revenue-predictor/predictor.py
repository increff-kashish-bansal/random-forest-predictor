import joblib
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
    # Load models and features
    models = joblib.load('models/brandA_models.pkl')
    with open('models/brandA_features.json', 'r') as f:
        required_features = json.load(f)
    
    # Ensure all required features are present
    missing_features = set(required_features) - set(df_X.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Get predictions from each model
    # 1. Median predictions (log scale)
    median_pred = np.expm1(models['median'].predict(df_X[required_features]))
    
    # 2. Lower tail predictions (untransformed)
    lower_pred = models['lower'].predict(df_X[required_features])
    
    # 3. Upper tail predictions (untransformed)
    upper_pred = models['upper'].predict(df_X[required_features])
    
    # Calculate prediction intervals
    p10 = lower_pred
    p50 = median_pred
    p90 = upper_pred
    
    # Calculate SHAP values using median model
    explainer = shap.TreeExplainer(models['median'])
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
