# Fashion Retail Revenue Prediction Platform (India)
🖊️ **Author**: MS Developers – Increff
📅 **Last Updated**: June 13, 2025
📌 **Objective**: Build a brand-specific, modular forecasting system that ingests historical sales and store metadata, automatically engineers and prunes features, trains the best-performing model with quantile outputs, and serves predictions via an interactive Streamlit UI supporting dynamic parameter variation and scenario analysis, tailored for the Indian fashion market.

## 🎯 Scope
- **Data Sources**: Historical sales data, store metadata, promotions, and holiday calendar
- **Features**: Temporal, store-specific, and product-specific features
- **Models**: Random Forest, XGBoost, LightGBM, Prophet, and ARIMA
- **Output**: Revenue predictions with confidence intervals
- **UI**: Interactive Streamlit interface for scenario analysis

## 📁 File Structure
```
project/
├── app.py                 # Streamlit UI application
├── train_model.py         # Model training pipeline
├── predictor.py           # Prediction functionality
├── utils/
│   └── feature_engineering.py  # Feature engineering utilities
├── data/
│   ├── sales_brandA.csv   # Historical sales data
│   ├── stores_brandA.csv  # Store metadata
│   ├── promotions_brandA.csv  # Promotion calendar
│   └── holiday_calendar.csv   # Holiday calendar
└── models/                # Trained model artifacts
```

## 📊 Input Specifications
### Sales Data (sales_brandA.csv)
- date: Date of sale
- store: Store identifier
- sku_id: Product identifier
- revenue: Daily revenue
- units_sold: Number of units sold

### Store Data (stores_brandA.csv)
- id: Store identifier
- store_area: Store size in sq ft
- location_type: Type of location (Mall/High Street)
- city: City name
- state: State name

## 🎯 Key Performance Indicators (KPIs)
1. **Model Performance**
   - Mean Absolute Percentage Error (MAPE) < 15%
   - Root Mean Square Error (RMSE) < 20%
   - R-squared > 0.85

2. **System Performance**
   - Prediction latency < 2 seconds
   - Training time < 30 minutes
   - Memory usage < 4GB

## 🚀 Getting Started
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the Streamlit app: `streamlit run project/app.py`

## 🔧 Development
- Python 3.8+
- Dependencies listed in requirements.txt
- Git for version control

## 📝 License
Proprietary - All rights reserved by Increff
