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
from datetime import datetime

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
    st.title("Predict Revenue")
    
    if 'processed_sales' not in st.session_state or 'processed_stores' not in st.session_state:
        st.warning("Please upload and process data first.")
    else:
        # Get inputs
        store_id = st.selectbox("Select Store", options=st.session_state.processed_stores['id'].unique())
        future_date = st.date_input("Select Future Date", min_value=datetime.now().date())
        store_area = st.number_input("Store Area (sq ft)", min_value=0.0, value=float(st.session_state.processed_stores[st.session_state.processed_stores['id'] == store_id]['store_area'].iloc[0]))
        discount_pct = st.slider("Discount Percentage", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        
        if st.button("Predict"):
            try:
                # Create synthetic sales row
                synthetic_sales = pd.DataFrame({
                    'store': [store_id],
                    'date': [pd.Timestamp(future_date)],
                    'qty_sold': [0],
                    'revenue': [0],
                    'disc_perc': [discount_pct]
                })
                
                # Get matching store row
                store_row = st.session_state.processed_stores[st.session_state.processed_stores['id'] == store_id].copy()
                store_row['store_area'] = store_area  # Update with user input
                
                # Generate features
                X_pred, features = derive_features(synthetic_sales, store_row, is_prediction=True)
                
                # Fill any remaining NaN values with 0
                X_pred = X_pred.fillna(0)
                
                # Load required features
                with open('models/brandA_features.json', 'r') as f:
                    required_features = json.load(f)
                
                # Check for missing features
                missing_features = set(required_features) - set(X_pred.columns)
                if missing_features:
                    st.error(f"Missing features: {missing_features}")
                    st.stop()
                
                # Select features in correct order
                X_pred = X_pred[required_features]
                
                # Make prediction
                results = predict_and_explain(X_pred)
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Predicted Revenue",
                        f"₹{float(results['p50'][0]):,.2f}",
                        delta=None
                    )
                with col2:
                    st.metric(
                        "Prediction Range",
                        f"₹{float(results['p10'][0]):,.2f} - ₹{float(results['p90'][0]):,.2f}",
                        delta=None
                    )
                
                # Historical Comparison Section
                st.markdown("### Historical Comparison")
                
                # 1. Predicted Sales for Future Date
                st.markdown(f"#### Predicted Sales for {future_date.strftime('%d %B %Y')}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("P50 Revenue", f"₹{float(results['p50'][0]):,.2f}")
                with col2:
                    st.metric("Prediction Range", f"₹{float(results['p10'][0]):,.2f} - ₹{float(results['p90'][0]):,.2f}")
                
                st.markdown("---")
                
                # 2. Monthly Averages by Year
                st.markdown("#### Monthly Averages by Year")
                historical_data = st.session_state.processed_sales[
                    st.session_state.processed_sales['store'] == store_id
                ].copy()
                
                # Convert future_date to datetime64[ns]
                future_date = pd.to_datetime(future_date)
                
                # Add year and month columns for comparison
                historical_data['year'] = historical_data['date'].dt.year
                historical_data['month'] = historical_data['date'].dt.month
                historical_data['day'] = historical_data['date'].dt.day
                
                # Filter out future data
                historical_data = historical_data[historical_data['date'] < future_date]
                
                same_month_data = historical_data[
                    (historical_data['month'] == future_date.month)
                ].sort_values('year', ascending=False)
                
                if not same_month_data.empty:
                    # Calculate monthly totals and averages
                    same_month_metrics = same_month_data.groupby('year').agg({
                        'revenue': ['sum', 'mean'],
                        'qty_sold': ['sum', 'mean'],
                        'disc_perc': 'mean'
                    }).round(2)
                    
                    # Display metrics in columns
                    cols = st.columns(len(same_month_metrics))
                    for idx, (year, metrics) in enumerate(same_month_metrics.iterrows()):
                        with cols[idx]:
                            st.markdown(f"**{future_date.strftime('%B')} {year}**")
                            st.metric(
                                "Total Revenue",
                                f"₹{metrics[('revenue', 'sum')]:,.2f}"
                            )
                            st.metric(
                                "Avg Daily Revenue",
                                f"₹{metrics[('revenue', 'mean')]:,.2f}"
                            )
                            st.metric(
                                "Total Qty Sold",
                                f"{metrics[('qty_sold', 'sum')]:,.0f}"
                            )
                            st.metric(
                                "Avg Daily Qty",
                                f"{metrics[('qty_sold', 'mean')]:,.0f}"
                            )
                            st.metric(
                                "Avg Discount",
                                f"{metrics[('disc_perc', 'mean')]*100:.1f}%"
                            )
                
                st.markdown("---")
                
                # 3. Same Day Actuals by Year
                st.markdown(f"#### Actual Sales for {future_date.strftime('%d %B')} by Year")
                same_day_data = historical_data[
                    (historical_data['month'] == future_date.month) & 
                    (historical_data['day'] == future_date.day)
                ].sort_values('year', ascending=False)
                
                if not same_day_data.empty:
                    # Group by year and get the exact day's data
                    same_day_metrics = same_day_data.groupby('year').agg({
                        'revenue': 'first',
                        'qty_sold': 'first',
                        'disc_perc': 'first'
                    }).round(2)
                    
                    # Display metrics in columns
                    cols = st.columns(len(same_day_metrics))
                    for idx, (year, metrics) in enumerate(same_day_metrics.iterrows()):
                        with cols[idx]:
                            st.markdown(f"**{year}**")
                            st.metric(
                                "Revenue",
                                f"₹{metrics['revenue']:,.2f}"
                            )
                            st.metric(
                                "Qty Sold",
                                f"{metrics['qty_sold']:,.0f}"
                            )
                            st.metric(
                                "Discount",
                                f"{metrics['disc_perc']*100:.1f}%"
                            )
                
                st.markdown("---")
                
                # 4. Yearly Averages
                st.markdown("#### Yearly Averages")
                yearly_avg = historical_data.groupby('year').agg({
                    'revenue': ['sum', 'mean'],
                    'qty_sold': ['sum', 'mean'],
                    'disc_perc': 'mean'
                }).round(2)
                
                if not yearly_avg.empty:
                    # Display metrics in columns
                    cols = st.columns(len(yearly_avg))
                    for idx, (year, metrics) in enumerate(yearly_avg.iterrows()):
                        with cols[idx]:
                            st.markdown(f"**{year}**")
                            st.metric(
                                "Total Revenue",
                                f"₹{metrics[('revenue', 'sum')]:,.2f}"
                            )
                            st.metric(
                                "Avg Daily Revenue",
                                f"₹{metrics[('revenue', 'mean')]:,.2f}"
                            )
                            st.metric(
                                "Total Qty Sold",
                                f"{metrics[('qty_sold', 'sum')]:,.0f}"
                            )
                            st.metric(
                                "Avg Daily Qty",
                                f"{metrics[('qty_sold', 'mean')]:,.0f}"
                            )
                            st.metric(
                                "Avg Discount",
                                f"{metrics[('disc_perc', 'mean')]*100:.1f}%"
                            )
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.error("Full error details:")
                st.exception(e)
