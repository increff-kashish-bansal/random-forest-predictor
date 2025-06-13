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
    
    # Log feature matrix info after first derive_features call
    print(f"X.shape: {X.shape}")
    print(f"X.columns: {X.columns.tolist()}")
    print("\nMissing values in X:")
    print(X.isnull().sum().sort_values(ascending=False).head(10))
    
    # Remove lag-dependent features
    lag_features = [col for col in X.columns if any(term in col for term in ['lag', 'rolling', 'growth'])]
    print(f"\nRemoving {len(lag_features)} lag-dependent features:")
    for feat in lag_features:
        print(f"- {feat}")
    
    X = X.drop(columns=lag_features)
    features = [f for f in features if f not in lag_features]
    
    print(f"\nRemaining features ({len(features)}):")
    for feat in features:
        print(f"- {feat}")
    
    # Ensure X and y have same number of samples
    if len(X) != len(y):
        logging.error(f"Feature matrix X ({len(X)} samples) and target y ({len(y)} samples) have different lengths")
        raise ValueError(f"Feature matrix X ({len(X)} samples) and target y ({len(y)} samples) have different lengths")
    
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Target vector shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training set shape: {X_train.shape}")
    logging.info(f"Test set shape: {X_test.shape}")
    
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Train model
    logging.info("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logging.info(f"Training R² score: {train_score:.3f}")
    logging.info(f"Test R² score: {test_score:.3f}")
    
    # Get feature importances
    importances = dict(zip(X_train.columns, model.feature_importances_))
    sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    # Print top 5 features
    print("\nTop 5 Most Important Features:")
    for feat, imp in list(sorted_importances.items())[:5]:
        print(f"{feat}: {imp:.3f}")
    
    # Save model and features
    Path('models').mkdir(exist_ok=True)
    model_path = 'models/brandA_model.pkl'
    features_path = 'models/brandA_features.json'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(features_path, 'w') as f:
        json.dump(features, f)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Features saved to: {features_path}")
    print(f"Final feature count: {len(features)}")
    
    logging.info("Model and features saved successfully")
    
    # Return training metrics
    return {
        'train_score': train_score,
        'test_score': test_score,
        'feature_importances': sorted_importances
    }
