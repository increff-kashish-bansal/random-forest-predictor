import streamlit as st
import pandas as pd
from project.predictor import predict

st.title('Fashion Retail Revenue Prediction Platform (India)')

# Brand selector
brand = st.selectbox('Select Brand', ['brandA'])

st.sidebar.header('Input Parameters for Prediction')

# Interactive input fields
store_code = st.sidebar.text_input('Store Code', 'S001')
store_area = st.sidebar.number_input('Store Area (sq ft)', min_value=500, max_value=10000, value=1500)
discount_value = st.sidebar.slider('Discount (%)', 0, 80, 10)
future_date = st.sidebar.date_input('Prediction Date')

# Prediction button
if st.sidebar.button('Predict Revenue'):
    input_data = {
        'store_area': store_area,
        'discount_value': float(discount_value),
        'day_of_week': future_date.weekday(),
        'week_of_year': future_date.isocalendar()[1],
        'month': future_date.month,
        'is_weekend': 1 if future_date.weekday() >= 5 else 0
        # Add other necessary features with default or user-provided values
    }
    
    prediction = predict(input_data)
    
    st.subheader('Forecasted Revenue')
    st.write(f'Predicted Revenue (P50): {prediction["P50"]:.2f}')
