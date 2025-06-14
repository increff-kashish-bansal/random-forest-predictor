# Fashion Revenue Predictor

A Streamlit-based web application for predicting fashion retail revenue using machine learning. The application allows users to upload sales and store data, train a model, and make revenue predictions for future dates.

## Features

- Data upload and validation
- Automatic column mapping
- Advanced model training with:
  - LightGBM and Random Forest ensemble
  - Hyperparameter optimization using Optuna
  - Feature importance visualization with SHAP values
  - Time-based sample weighting
  - Dynamic test size based on data length
- Sophisticated revenue prediction with:
  - P10, P50, and P90 confidence intervals
  - Conformal calibration for reliable intervals
  - Store-specific seasonal adjustments
  - Historical data comparison
  - Dynamic confidence bands based on store history
- Interactive UI with Streamlit
- Comprehensive logging and debugging

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
disc_perc = disc_value * 100 / (revenue + disc_value)
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
   - View training results, including:
     - Model performance metrics
     - Feature importance rankings
     - Training logs and debugging information

3. **Make Predictions**
   - Select a store and future date
   - Adjust store area and discount percentage if needed
   - Get revenue predictions with:
     - P10 (lower bound)
     - P50 (median prediction)
     - P90 (upper bound)
   - View SHAP explanations for predictions
   - Compare with historical data

## Advanced Features

- **Time-based Weighting**: Recent data points are weighted more heavily in training
- **Dynamic Test Size**: Test set size adapts based on available historical data
- **Conformal Calibration**: Ensures prediction intervals maintain desired coverage
- **Store-specific Adjustments**: 
  - Seasonal indices by store, month, and weekday
  - Historical revenue floors for P10 predictions
  - Store-specific volatility adjustments
- **Feature Engineering**:
  - Rolling statistics (3-day, 7-day, 14-day, 30-day)
  - Store clustering
  - Channel and region-specific discount normalization
  - Premium location adjustments

## Notes

- The model uses an ensemble of LightGBM and Random Forest for prediction
- Predictions include P10, P50, and P90 confidence intervals with conformal calibration
- Historical comparisons show same-day/month data from previous years
- High discount values (>50%) may affect prediction accuracy
- Training logs are saved to `training.log` for debugging and analysis

## Performance Considerations

- The model automatically handles outliers and missing values
- Low-activity days (revenue < 100) are filtered out during training
- Store-specific seasonal patterns are incorporated into predictions
- Confidence intervals are dynamically adjusted based on historical volatility
- Fallback mechanisms ensure reasonable predictions even with limited data
