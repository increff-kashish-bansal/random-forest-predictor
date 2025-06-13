import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import json
import joblib
from project.utils.feature_engineering import derive_features

def train_model(brand):
    # Load data
    sales_df = pd.read_csv(f'project/data/sales_{brand}.csv')
    stores_df = pd.read_csv(f'project/data/stores_{brand}.csv')
    
    # Merge and preprocess
    df = pd.merge(sales_df, stores_df, left_on='store', right_on='id', how='left')
    df = derive_features(df)
    
    # Define features and target
    features = ['day_of_week', 'week_of_year', 'month', 'is_weekend', 'store_area', 'lagged_sales_7d'] # Example features
    target = 'revenue'
    
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and selected features
    joblib.dump(model, f'project/models/{brand}_final_model.pkl')
    with open(f'project/models/{brand}_selected_features.json', 'w') as f:
        json.dump(features, f)

if __name__ == '__main__':
    train_model('brandA')
