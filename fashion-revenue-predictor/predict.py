import logging
import pandas as pd
import numpy as np
from predictor import predict_and_explain
from utils.feature_engineering import derive_features

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example: generate a synthetic prediction for a single store and date
    store_id = "STORE_1"
    prediction_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=7)
    # Minimal synthetic sales and store data
    synthetic_sales = pd.DataFrame({
        'store': [store_id],
        'date': [prediction_date],
        'qty_sold': [0],
        'revenue': [0],
        'disc_value': [0],
        'disc_perc': [0]
    })
    synthetic_stores = pd.DataFrame({
        'id': [store_id],
        'channel': ['ONLINE'],
        'city': ['MUMBAI'],
        'region': ['NORTH1'],
        'store_area': [2000],
        'is_online': [1]
    })
    # For demo, no historical sales
    historical_sales = None
    # Derive features
    X_pred, _ = derive_features(synthetic_sales, synthetic_stores, historical_sales=historical_sales, is_prediction=True)
    # Run prediction
    results = predict_and_explain(X_pred, historical_sales=historical_sales, original_input=synthetic_sales)
    print("Prediction result:", results) 