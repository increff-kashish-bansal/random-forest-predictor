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
    
    # Apply Box-Cox transformation to revenue
    logging.info("Applying Box-Cox transformation to revenue...")
    y_transformed, lambda_ = boxcox(y + 1e-3)  # Add small constant to handle zeros
    
    # Save lambda parameter for prediction
    Path('models').mkdir(exist_ok=True)
    with open('models/boxcox_lambda.json', 'w') as f:
        json.dump({'lambda': float(lambda_)}, f)
    
    # Log feature matrix info after first derive_features call
    print(f"X.shape: {X.shape}")
    print(f"X.columns: {X.columns.tolist()}")
    print("\nMissing values in X:")
    print(X.isnull().sum().sort_values(ascending=False).head(10))
    
    # Keep all features including lag features
    print(f"\nUsing all features ({len(features)}):")
    for feat in features:
        print(f"- {feat}")
    
    # Ensure X and y have same number of samples
    if len(X) != len(y_transformed):
        logging.error(f"Feature matrix X ({len(X)} samples) and target y ({len(y_transformed)} samples) have different lengths")
        raise ValueError(f"Feature matrix X ({len(X)} samples) and target y ({len(y_transformed)} samples) have different lengths")
    
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Target vector shape: {y_transformed.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)
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
