# Fashion Revenue Predictor

A Streamlit-based web application for predicting fashion retail revenue using machine learning. The application allows users to upload sales and store data, train a model, and make revenue predictions for future dates.

## Features

- Data upload and validation
- Automatic column mapping
- Model training with feature importance visualization
- Revenue prediction with confidence intervals
- Historical data comparison
- Interactive UI with Streamlit

## Data Requirements

### Sales Data Template
The sales data should include the following columns:
- `store`: Store identifier
- `date`: Transaction date
- `qty_sold`: Quantity of items sold
- `revenue`: Total revenue
- `disc_value`: Discount value in currency units

The application automatically calculates the discount percentage (`disc_perc`) as:
```
disc_perc = disc_value / (revenue + disc_value)
```

### Stores Data Template
The stores data should include:
- `id`: Store identifier (matching sales data)
- `channel`: Sales channel
- `city`: Store location city
- `region`: Store region
- `is_online`: Boolean indicating if store is online
- `store_area`: Store area in square feet

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Upload Data**
   - Download the templates for sales and stores data
   - Fill in your data following the template format
   - Upload the CSV files through the interface

2. **Train Model**
   - After successful data upload, navigate to the Train page
   - Click "Train Model" to start the training process
   - View training results and feature importances

3. **Make Predictions**
   - Select a store and future date
   - Adjust store area and discount percentage if needed
   - Get revenue predictions with confidence intervals
   - Compare with historical data

## Notes

- The model uses Random Forest for prediction
- Predictions include P10, P50, and P90 confidence intervals
- Historical comparisons show same-day/month data from previous years
- High discount values (>50%) may affect prediction accuracy
