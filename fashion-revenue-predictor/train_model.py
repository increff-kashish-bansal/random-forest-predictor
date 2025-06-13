import pandas as pd
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils.feature_engineering import derive_features
from scipy.stats import boxcox

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train_model(df_sales, df_stores):
    """Train a random forest model for revenue prediction."""
    logging.info("Starting model training...")
    logging.info(f"Input shapes - Sales: {df_sales.shape}, Stores: {df_stores.shape}")
    
    # Derive features
    X, features = derive_features(df_sales, df_stores, is_prediction=False)
    y = df_sales['revenue'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training set shape: {X_train.shape}")
    logging.info(f"Test set shape: {X_test.shape}")
    
    # Train three separate models for different quantiles
    models = {}
    
    # 1. Median (0.5 quantile) model - use log transformation
    logging.info("Training median model with log transformation...")
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    median_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    median_model.fit(X_train, y_train_log)
    models['median'] = median_model
    
    # 2. Lower tail (0.1 quantile) model - use untransformed data
    logging.info("Training lower tail model on untransformed data...")
    lower_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    lower_model.fit(X_train, y_train)
    models['lower'] = lower_model
    
    # 3. Upper tail (0.9 quantile) model - use untransformed data
    logging.info("Training upper tail model on untransformed data...")
    upper_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    upper_model.fit(X_train, y_train)
    models['upper'] = upper_model
    
    # Evaluate models
    train_scores = {}
    test_scores = {}
    
    # Median model evaluation
    train_pred_median = np.expm1(median_model.predict(X_train))
    test_pred_median = np.expm1(median_model.predict(X_test))
    train_scores['median'] = np.corrcoef(y_train, train_pred_median)[0,1]**2
    test_scores['median'] = np.corrcoef(y_test, test_pred_median)[0,1]**2
    
    # Lower tail model evaluation
    train_pred_lower = lower_model.predict(X_train)
    test_pred_lower = lower_model.predict(X_test)
    train_scores['lower'] = np.corrcoef(y_train, train_pred_lower)[0,1]**2
    test_scores['lower'] = np.corrcoef(y_test, test_pred_lower)[0,1]**2
    
    # Upper tail model evaluation
    train_pred_upper = upper_model.predict(X_train)
    test_pred_upper = upper_model.predict(X_test)
    train_scores['upper'] = np.corrcoef(y_train, train_pred_upper)[0,1]**2
    test_scores['upper'] = np.corrcoef(y_test, test_pred_upper)[0,1]**2
    
    # Log scores
    logging.info("Model evaluation scores:")
    for model_type in ['median', 'lower', 'upper']:
        logging.info(f"{model_type} - Train R²: {train_scores[model_type]:.3f}, Test R²: {test_scores[model_type]:.3f}")
    
    # Get feature importances from median model
    importances = dict(zip(X_train.columns, median_model.feature_importances_))
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    # Print top 5 features
    print("\nTop 5 Most Important Features:")
    for feat, imp in list(sorted_importances.items())[:5]:
        print(f"{feat}: {imp:.3f}")
    
    # Save models and features
    Path('models').mkdir(exist_ok=True)
    model_path = 'models/brandA_models.pkl'
    features_path = 'models/brandA_features.json'
    
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    with open(features_path, 'w') as f:
        json.dump(features, f)
    
    print(f"\nModels saved to: {model_path}")
    print(f"Features saved to: {features_path}")
    print(f"Final feature count: {len(features)}")
    
    logging.info("Models and features saved successfully")
    
    # Return training metrics
    return {
        'train_scores': train_scores,
        'test_scores': test_scores,
        'feature_importances': sorted_importances
    }
