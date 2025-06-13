import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from train_model import train_model
from predictor import predict_and_explain
from utils.feature_engineering import derive_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_pipeline.log'),
        logging.StreamHandler()
    ]
)

def generate_synthetic_data(n_stores=10, n_days=90):
    """Generate synthetic sales and store data for testing."""
    logging.info(f"Generating synthetic data for {n_stores} stores over {n_days} days")
    
    # Generate store data
    stores = []
    for i in range(n_stores):
        store = {
            'id': f'STORE_{i+1}',
            'channel': np.random.choice(['ONLINE', 'LFR_LIFESTYLE']),
            'city': np.random.choice(['MUMBAI', 'DELHI', 'BANGALORE', 'HYDERABAD', 'CHENNAI']),
            'region': np.random.choice(['NORTH1', 'NORTH2', 'SOUTH', 'EAST', 'WEST1', 'WEST2']),
            'store_area': np.random.randint(1000, 5000),
            'is_online': np.random.choice([0, 1])
        }
        stores.append(store)
    
    df_stores = pd.DataFrame(stores)
    logging.info(f"Generated store data shape: {df_stores.shape}")
    
    # Generate sales data
    sales = []
    start_date = datetime.now()
    
    for store in stores:
        for day in range(n_days):
            date = start_date + timedelta(days=day)
            
            # Base revenue with some randomness
            base_revenue = np.random.normal(10000, 2000)
            
            # Add weekend effect
            if date.weekday() >= 5:  # Weekend
                base_revenue *= 1.5
            
            # Add monthly seasonality
            month_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)
            base_revenue *= month_factor
            
            # Add some noise
            revenue = max(0, base_revenue + np.random.normal(0, 1000))
            
            # Calculate quantity sold (assuming average price of 1000)
            qty_sold = int(revenue / 1000)
            
            # Random discount (0-30%)
            disc_value = revenue * np.random.uniform(0, 0.3)
            
            # Calculate discount percentage
            disc_perc = disc_value / (revenue + disc_value)
            
            sale = {
                'store': store['id'],
                'date': date,
                'qty_sold': qty_sold,
                'revenue': revenue,
                'disc_perc': disc_perc
            }
            sales.append(sale)
    
    df_sales = pd.DataFrame(sales)
    logging.info(f"Generated sales data shape: {df_sales.shape}")
    
    return df_sales, df_stores

def test_training(df_sales, df_stores):
    """Test the training pipeline."""
    logging.info("Testing training pipeline...")
    try:
        results = train_model(df_sales, df_stores)
        logging.info("Training completed successfully")
        
        # Log scores for each model
        for model_type in ['median', 'lower', 'upper']:
            logging.info(f"{model_type} - Train R²: {results['train_scores'][model_type]:.3f}, Test R²: {results['test_scores'][model_type]:.3f}")
        
        logging.info("Top 5 features:")
        for feat, imp in list(results['feature_importances'].items())[:5]:
            logging.info(f"{feat}: {imp:.3f}")
        return True
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        return False

def test_prediction(df_sales, df_stores):
    """Test the prediction pipeline."""
    logging.info("Testing prediction pipeline...")
    try:
        # Select a random store and date
        store_id = np.random.choice(df_stores['id'].unique())
        future_date = df_sales['date'].max() + timedelta(days=7)
        
        logging.info(f"Testing prediction for store {store_id} on {future_date}")
        
        # Get last 7 days of historical sales for this store
        historical_sales = df_sales[
            (df_sales['store'] == store_id) & 
            (df_sales['date'] >= future_date - timedelta(days=7)) &
            (df_sales['date'] < future_date)
        ].copy()
        
        logging.info(f"Found {len(historical_sales)} days of historical sales data")
        
        # Create synthetic sales row
        synthetic_sales = pd.DataFrame({
            'store': [store_id],
            'date': [future_date],
            'qty_sold': [0],
            'revenue': [0],
            'disc_perc': [0]
        })
        
        # Get store metadata
        store_meta = df_stores[df_stores['id'] == store_id].iloc[0]
        
        # Create synthetic stores row
        synthetic_stores = pd.DataFrame({
            'id': [store_id],
            'channel': [store_meta['channel']],
            'city': [store_meta['city']],
            'region': [store_meta['region']],
            'store_area': [store_meta['store_area']],
            'is_online': [store_meta['is_online']]
        })
        
        # Derive features with historical data
        X_pred, _ = derive_features(synthetic_sales, synthetic_stores, historical_sales=historical_sales, is_prediction=True)
        
        # Make prediction
        results = predict_and_explain(X_pred)
        
        logging.info("Prediction completed successfully")
        logging.info(f"Predicted revenue: ${results['p50'][0]:,.2f}")
        logging.info(f"Prediction range: ${results['p10'][0]:,.2f} - ${results['p90'][0]:,.2f}")
        logging.info("Top 5 features:")
        for feat, val in results['top_5_features']:
            logging.info(f"{feat}: {val:.3f}")
        return True
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return False

if __name__ == "__main__":
    logging.info("Starting test pipeline...")
    
    # Generate synthetic data
    df_sales, df_stores = generate_synthetic_data()
    
    # Run tests
    training_success = test_training(df_sales, df_stores)
    prediction_success = test_prediction(df_sales, df_stores)
    
    # Log results
    logging.info("\nTest Results:")
    logging.info(f"Training test: {'PASSED' if training_success else 'FAILED'}")
    logging.info(f"Prediction test: {'PASSED' if prediction_success else 'FAILED'}")
    
    if not (training_success and prediction_success):
        logging.error("Some tests failed. Please check the logs for details.") 