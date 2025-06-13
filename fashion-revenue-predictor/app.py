import streamlit as st
import pandas as pd
import json
import plotly.express as px
from pathlib import Path
from utils.mapping_helpers import suggest_mappings
from utils.feature_engineering import derive_features
from train_model import train_model
from predictor import predict_and_explain
import logging

# Page config
st.set_page_config(
    page_title="Fashion Revenue Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Upload", "Train", "Predict"])

if page == "Upload":
    st.title("Data Upload")
    
    # Template download section
    st.header("Download Templates")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Sales Template"):
            sales_template = pd.read_csv("data/sales_template.csv")
            st.download_button(
                "Download CSV",
                sales_template.to_csv(index=False),
                "sales_template.csv",
                "text/csv"
            )
    
    with col2:
        if st.button("Download Stores Template"):
            stores_template = pd.read_csv("data/stores_template.csv")
            st.download_button(
                "Download CSV",
                stores_template.to_csv(index=False),
                "stores_template.csv",
                "text/csv"
            )
    
    # File upload section
    st.header("Upload Data")
    uploaded_sales = st.file_uploader("Upload Sales Data (CSV)", type="csv")
    uploaded_stores = st.file_uploader("Upload Stores Data (CSV)", type="csv")
    
    if uploaded_sales and uploaded_stores:
        try:
            with st.spinner("Processing data..."):
                # Load data
                df_sales = pd.read_csv(uploaded_sales)
                df_stores = pd.read_csv(uploaded_stores)
                
                # Show preview of uploaded data
                st.subheader("Preview of Uploaded Data")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Sales Data Preview:")
                    st.dataframe(df_sales.head())
                with col2:
                    st.write("Stores Data Preview:")
                    st.dataframe(df_stores.head())
                
                # Column mapping section
                st.header("Column Mapping")
                
                # Sales columns mapping
                st.subheader("Sales Data Mapping")
                sales_required = ["store", "date", "qty_sold", "revenue", "disc_value"]
                sales_suggestions = suggest_mappings(df_sales.columns.tolist(), sales_required)
                
                sales_mapping = {}
                for req_col in sales_required:
                    suggested = sales_suggestions[req_col]
                    options = [""] + df_sales.columns.tolist()
                    selected = st.selectbox(
                        f"Map to {req_col}",
                        options=options,
                        index=options.index(suggested) if suggested else 0
                    )
                    if selected:
                        sales_mapping[selected] = req_col
                
                # Stores columns mapping
                st.subheader("Stores Data Mapping")
                stores_required = ["id", "channel", "city", "region", "is_online", "store_area"]
                stores_suggestions = suggest_mappings(df_stores.columns.tolist(), stores_required)
                
                stores_mapping = {}
                for req_col in stores_required:
                    suggested = stores_suggestions[req_col]
                    options = [""] + df_stores.columns.tolist()
                    selected = st.selectbox(
                        f"Map to {req_col}",
                        options=options,
                        index=options.index(suggested) if suggested else 0
                    )
                    if selected:
                        stores_mapping[selected] = req_col
                
                # Rename columns according to mapping
                st.info("Mapping columns...")
                df_sales = df_sales.rename(columns=sales_mapping)
                df_stores = df_stores.rename(columns=stores_mapping)
                
                # Ensure date column is datetime
                st.info("Processing dates...")
                df_sales['date'] = pd.to_datetime(df_sales['date'])
                
                # Automatically roll up sales by store and date
                st.info("Rolling up sales data by store and date...")
                df_sales = df_sales.groupby(['store', 'date']).agg({
                    'qty_sold': 'sum',
                    'revenue': 'sum',
                    'disc_value': 'sum'
                }).reset_index()
                
                # Calculate discount percentage
                df_sales['disc_perc'] = df_sales['disc_value'] / (df_sales['revenue'] + df_sales['disc_value'])
                
                # Store processed data in session state
                st.session_state['processed_sales'] = df_sales
                st.session_state['processed_stores'] = df_stores
                
                # Show summary of processed data
                st.success("Data processed successfully!")
                st.write("Processed Data Summary:")
                st.write(f"- Number of unique stores: {df_sales['store'].nunique()}")
                st.write(f"- Date range: {df_sales['date'].min().date()} to {df_sales['date'].max().date()}")
                st.write(f"- Total number of store-days: {len(df_sales)}")
                
                # Add a preview of the processed data
                st.subheader("Preview of Processed Data")
                st.dataframe(df_sales.head())
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

elif page == "Train":
    st.title("Model Training")
    
    if 'processed_sales' not in st.session_state or 'processed_stores' not in st.session_state:
        st.warning("Please upload and process data in the Upload page first.")
    else:
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Train model
                results = train_model(
                    st.session_state['processed_sales'],
                    st.session_state['processed_stores']
                )
                
                # Display training results
                
                st.header("Training Results")
                                # Display R² scores
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training R² Score", f"{results['train_score']:.3f}")
                with col2:
                    st.metric("Test R² Score", f"{results['test_score']:.3f}")
                
                # Display feature importances as bar chart
                importances_df = pd.DataFrame(
                    list(results['feature_importances'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importances_df,
                    x='Feature',
                    y='Importance',
                    title='Feature Importances'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display top 10 features
                st.subheader("Top 10 Most Important Features")
                for feat, imp in list(results['feature_importances'].items())[:10]:
                    st.write(f"{feat}: {imp:.3f}")

elif page == "Predict":
    st.title("Revenue Prediction")
    
    if 'processed_sales' not in st.session_state or 'processed_stores' not in st.session_state:
        st.warning("Please upload and process data in the Upload page first.")
    else:
        df_sales = st.session_state['processed_sales']
        df_stores = st.session_state['processed_stores']
        
        # Input section
        col1, col2 = st.columns(2)
        
        with col1:
            # Store selection
            store_ids = df_stores['id'].unique().tolist()
            
            # Get list of stores with sales data
            stores_with_sales = df_sales['store'].unique().tolist()
            valid_stores = [s for s in store_ids if s in stores_with_sales]
            
            if not valid_stores:
                st.error("No stores found with sales data. Please upload sales data first.")
                st.stop()
            
            # Show store selection with validation
            store_id = st.selectbox(
                "Select Store",
                valid_stores,
                format_func=lambda x: f"Store {x} ({len(df_sales[df_sales['store'] == x])} sales records)"
            )
            
            # Date selection
            future_date = st.date_input("Select Future Date")
        
        with col2:
            # Get store metadata for pre-filling
            store_meta = df_stores[df_stores['id'] == store_id].iloc[0]
            
            # Store area input
            store_area = st.number_input(
                "Store Area (sq ft)",
                min_value=0,
                value=int(store_meta['store_area'])
            )
            
            # Discount input with validation
            disc_value = st.slider(
                "Discount Value (%)",
                min_value=0,
                max_value=100,
                value=0,
                help="Enter a discount percentage between 0 and 100"
            )
            
            # Convert percentage to decimal for calculation
            disc_decimal = disc_value / 100.0
            
            # Add a warning for high discounts
            if disc_value > 50:
                st.warning("High discount values may affect prediction accuracy")
        
        # Predict button
        if st.button("Predict Revenue"):
            with st.spinner("Generating prediction..."):
                try:
                    # Get store metadata and ensure store_id is string
                    store_id = str(store_id)
                    st.info(f"Selected store ID: {store_id}")
                    
                    # Get store metadata
                    store_meta = df_stores[df_stores['id'] == store_id]
                    if len(store_meta) == 0:
                        raise ValueError(f"Store {store_id} not found in store data")
                    store_meta = store_meta.iloc[0]
                    
                    # Validate store exists in sales data
                    logging.info(f"Available store IDs in sales data: {df_sales['store'].unique()}")
                    logging.info(f"Store ID type in sales data: {df_sales['store'].dtype}")
                    
                    # Ensure store IDs are strings in both dataframes
                    df_sales['store'] = df_sales['store'].astype(str)
                    df_stores['id'] = df_stores['id'].astype(str)
                    
                    # Check if store has sales data
                    store_sales = df_sales[df_sales['store'] == store_id]
                    if len(store_sales) == 0:
                        st.error(f"No sales data found for store {store_id}. Please ensure the store has historical sales data.")
                        st.stop()
                    
                    # Create synthetic sales row with user inputs
                    synthetic_sales = pd.DataFrame({
                        'store': [store_id],
                        'date': [pd.to_datetime(future_date)],
                        'qty_sold': [0],
                        'revenue': [1000],  # Set a base revenue to calculate discount
                        'disc_perc': [disc_decimal]  # Use the discount percentage directly
                    })
                    
                    # Create synthetic stores row with store metadata
                    synthetic_stores = pd.DataFrame({
                        'id': [store_id],
                        'channel': [store_meta['channel']],
                        'city': [store_meta['city']],
                        'region': [store_meta['region']],
                        'store_area': [store_area],
                        'is_online': [store_meta['is_online']]
                    })
                    
                    # Generate features
                    X, _ = derive_features(synthetic_sales, synthetic_stores, is_prediction=True)
                    
                    # Make prediction
                    prediction = predict_and_explain(X)
                    
                    # Display results
                    st.header("Prediction Results")
                    
                    # Display prediction intervals
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("P10 Revenue", f"₹{prediction['p10'][0]:,.0f}")
                    with col2:
                        st.metric("P50 Revenue", f"₹{prediction['p50'][0]:,.0f}")
                    with col3:
                        st.metric("P90 Revenue", f"₹{prediction['p90'][0]:,.0f}")
                    
                    # Show historical revenue for same store
                    st.subheader("Historical Revenue")
                    
                    # Convert dates to datetime
                    df_sales['date'] = pd.to_datetime(df_sales['date'])
                    future_date = pd.to_datetime(future_date)
                    
                    logging.info(f"Looking up historical data for store {store_id}")
                    logging.info(f"Future date: {future_date}")
                    logging.info(f"Total sales records: {len(df_sales)}")
                    
                    # Get historical data for same store
                    store_history = df_sales[df_sales['store'] == store_id].copy()
                    logging.info(f"Store history records: {len(store_history)}")
                    logging.info(f"Store history date range: {store_history['date'].min()} to {store_history['date'].max()}")
                    
                    # Add day and month columns
                    store_history['day'] = store_history['date'].dt.day
                    store_history['month'] = store_history['date'].dt.month
                    
                    logging.info(f"Looking for day {future_date.day} and month {future_date.month}")
                    
                    # Get same day/month from previous years
                    same_day_month = store_history[
                        (store_history['day'] == future_date.day) & 
                        (store_history['month'] == future_date.month) &
                        (store_history['date'] < future_date)  # Only look at past dates
                    ].sort_values('date', ascending=False)  # Most recent first
                    
                    logging.info(f"Found {len(same_day_month)} records for same day/month")
                    if not same_day_month.empty:
                        logging.info("Same day/month records:")
                        for _, row in same_day_month.iterrows():
                            logging.info(f"Date: {row['date']}, Revenue: {row['revenue']}")
                    
                    if not same_day_month.empty:
                        # Display historical revenue for same day/month
                        st.write("Revenue on same day/month in previous years:")
                        for _, row in same_day_month.iterrows():
                            st.write(f"{row['date'].strftime('%Y-%m-%d')}: ₹{row['revenue']:,.0f}")
                        
                        # Calculate average historical revenue
                        avg_historical = same_day_month['revenue'].mean()
                        st.metric("Average Historical Revenue", f"₹{avg_historical:,.0f}")
                        logging.info(f"Average historical revenue: {avg_historical:,.0f}")
                        
                        # Show year-over-year comparison
                        if len(same_day_month) > 1:
                            latest = same_day_month.iloc[0]['revenue']
                            previous = same_day_month.iloc[1]['revenue']
                            yoy_change = ((latest - previous) / previous) * 100
                            st.metric("Year-over-Year Change", f"{yoy_change:+.1f}%")
                            logging.info(f"YoY change: {yoy_change:+.1f}% (Latest: {latest:,.0f}, Previous: {previous:,.0f})")
                    else:
                        st.write("No historical data available for this day/month")
                        logging.info("No historical data found for same day/month")
                        
                        # Show closest available dates
                        st.write("Closest available dates:")
                        closest_dates = store_history.nsmallest(3, 'date')
                        logging.info("Closest available dates:")
                        for _, row in closest_dates.iterrows():
                            st.write(f"{row['date'].strftime('%Y-%m-%d')}: ₹{row['revenue']:,.0f}")
                            logging.info(f"Date: {row['date']}, Revenue: {row['revenue']:,.0f}")

                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
