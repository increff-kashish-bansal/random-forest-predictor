import joblib
import json
import pandas as pd

def predict(input_data):
    # Load model and features
    model = joblib.load('project/models/brandA_final_model.pkl')
    with open('project/models/brandA_selected_features.json', 'r') as f:
        features = json.load(f)
    
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required features are present
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0 # Or some other default value
            
    input_df = input_df[features]
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # For quantile forecasts, you would typically use a different model or method
    # This is a placeholder for P50
    return {'P50': prediction[0]}
