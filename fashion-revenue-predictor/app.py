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
import numpy as np

# Page config
st.set_page_config(
    page_title="Fashion Revenue Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'processed_sales' not in st.session_state:
    st.session_state['processed_sales'] = None
if 'processed_stores' not in st.session_state:
    st.session_state['processed_stores'] = None
if 'training_results' not in st.session_state:
    st.session_state['training_results'] = None

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
    
    if uploaded_sales is not None and uploaded_stores is not None:
        try:
            # Read the uploaded files
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
                    index=options.index(suggested) if suggested else 0,
                    key=f"sales_{req_col}"  # Add unique key for each selectbox
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
                    index=options.index(suggested) if suggested else 0,
                    key=f"stores_{req_col}"  # Add unique key for each selectbox
                )
                if selected:
                    stores_mapping[selected] = req_col
            
            # Process button
            if st.button("Process Data"):
                with st.spinner("Processing data..."):
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
                    # Ensure we don't divide by zero and handle negative values
                    df_sales['disc_perc'] = 0.00
                    
                    # Debug prints
                    st.write("Debug - Before calculation:")
                    st.write(f"Sample revenue: {df_sales['revenue'].head()}")
                    st.write(f"Sample disc_value: {df_sales['disc_value'].head()}")
                    
                    # Calculate discount percentage
                    mask = (df_sales['revenue'] + df_sales['disc_value']) > 0
                    df_sales.loc[mask, 'disc_perc'] = (
                        df_sales.loc[mask, 'disc_value'] * 100 / 
                        (df_sales.loc[mask, 'revenue'] + df_sales.loc[mask, 'disc_value'])
                    )
                    
                    # Debug prints
                    st.write("Debug - After calculation:")
                    st.write(f"Sample disc_perc: {df_sales['disc_perc'].head()}")
                    st.write(f"Max disc_perc: {df_sales['disc_perc'].max()}")
                    st.write(f"Mean disc_perc: {df_sales['disc_perc'].mean()}")
                    
                    # Log detailed discount statistics for debugging
                    st.write("Discount Statistics:")
                    st.write(f"- Min discount: {df_sales['disc_perc'].min():.2f}%")
                    st.write(f"- Max discount: {df_sales['disc_perc'].max():.2f}%")
                    st.write(f"- Mean discount: {df_sales['disc_perc'].mean():.2f}%")
                    st.write(f"- Number of discounted sales: {len(df_sales[df_sales['disc_perc'] > 0])}")
                    
                    # Add detailed debug information
                    st.write("Debug Information:")
                    st.write(f"- Total number of sales: {len(df_sales)}")
                    st.write(f"- Average revenue: ₹{df_sales['revenue'].mean():,.2f}")
                    st.write(f"- Average discount value: ₹{df_sales['disc_value'].mean():,.2f}")
                    st.write(f"- Median discount value: ₹{df_sales['disc_value'].median():,.2f}")
                    
                    # Add detailed discount analysis
                    st.write("### Detailed Discount Analysis")
                    
                    # 1. Discount Range Distribution
                    st.write("#### Discount Range Distribution")
                    discount_ranges = [
                        (0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
                        (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)
                    ]
                    
                    range_counts = []
                    for start, end in discount_ranges:
                        count = len(df_sales[(df_sales['disc_perc'] > start) & (df_sales['disc_perc'] <= end)])
                        range_counts.append({
                            'Range': f'{start}-{end}%',
                            'Count': count,
                            'Percentage': (count / len(df_sales)) * 100
                        })
                    
                    range_df = pd.DataFrame(range_counts)
                    st.dataframe(range_df)
                    
                    # 2. High Discount Analysis
                    st.write("#### High Discount Analysis (>50%)")
                    high_discounts = df_sales[df_sales['disc_perc'] > 50].copy()
                    
                    if not high_discounts.empty:
                        # Add month and day of week for pattern analysis
                        high_discounts['month'] = high_discounts['date'].dt.month
                        high_discounts['day_of_week'] = high_discounts['date'].dt.dayofweek
                        high_discounts['is_weekend'] = high_discounts['day_of_week'].isin([5, 6]).astype(int)
                        
                        # Monthly pattern
                        st.write("##### Monthly Distribution of High Discounts")
                        monthly_counts = high_discounts['month'].value_counts().sort_index()
                        monthly_df = pd.DataFrame({
                            'Month': monthly_counts.index,
                            'Count': monthly_counts.values,
                            'Percentage': (monthly_counts.values / len(high_discounts)) * 100
                        })
                        st.dataframe(monthly_df)
                        
                        # Weekend vs Weekday pattern
                        st.write("##### Weekend vs Weekday Distribution")
                        weekend_counts = high_discounts['is_weekend'].value_counts()
                        weekend_df = pd.DataFrame({
                            'Type': ['Weekday', 'Weekend'],
                            'Count': [weekend_counts[0], weekend_counts[1]],
                            'Percentage': [(weekend_counts[0] / len(high_discounts)) * 100,
                                         (weekend_counts[1] / len(high_discounts)) * 100]
                        })
                        st.dataframe(weekend_df)
                        
                        # Revenue and Discount Value Analysis
                        st.write("##### Revenue and Discount Value Analysis for High Discounts")
                        high_discount_metrics = pd.DataFrame({
                            'Metric': ['Average Revenue', 'Median Revenue', 
                                     'Average Discount Value', 'Median Discount Value',
                                     'Average Discount %', 'Median Discount %'],
                            'Value': [
                                f"₹{high_discounts['revenue'].mean():,.2f}",
                                f"₹{high_discounts['revenue'].median():,.2f}",
                                f"₹{high_discounts['disc_value'].mean():,.2f}",
                                f"₹{high_discounts['disc_value'].median():,.2f}",
                                f"{high_discounts['disc_perc'].mean():.2f}%",
                                f"{high_discounts['disc_perc'].median():.2f}%"
                            ]
                        })
                        st.dataframe(high_discount_metrics)
                        
                        # Sample of highest discount sales
                        st.write("##### Sample of Highest Discount Sales")
                        highest_discounts = high_discounts.nlargest(5, 'disc_perc')
                        st.dataframe(highest_discounts[['store', 'date', 'revenue', 'disc_value', 'disc_perc']])
                    
                    # Show distribution of discount values
                    st.write("Discount Value Distribution:")
                    st.write(df_sales['disc_value'].describe())
                    
                    # Show distribution of discount percentages
                    st.write("Discount Percentage Distribution:")
                    disc_perc_desc = df_sales['disc_perc'].describe()
                    disc_perc_desc = disc_perc_desc.apply(lambda x: f"{x:.2f}%")
                    st.write(disc_perc_desc)
                    
                    # Show sample of rows with high discounts
                    high_discounts = df_sales[df_sales['disc_perc'] > 50].head()
                    if not high_discounts.empty:
                        st.write("Sample of High Discount Sales (>50%):")
                        st.write(high_discounts[['store', 'date', 'revenue', 'disc_value', 'disc_perc']])
                    
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
            st.error(f"Error processing uploaded files: {str(e)}")
            st.error("Full error details:")
            st.exception(e)

elif page == "Train":
    st.title("Model Training")
    
    if st.session_state['processed_sales'] is None or st.session_state['processed_stores'] is None:
        st.warning("Please upload and process data in the Upload page first.")
    else:
        st.write("Data loaded successfully:")
        st.write(f"Sales data shape: {st.session_state['processed_sales'].shape}")
        st.write(f"Stores data shape: {st.session_state['processed_stores'].shape}")
        
        # Add a flag to track if training is in progress
        if 'training_in_progress' not in st.session_state:
            st.session_state['training_in_progress'] = False
        
        # Only show the train button if not already training
        if not st.session_state['training_in_progress']:
            if st.button("Train Model"):
                st.session_state['training_in_progress'] = True
                st.rerun()
        
        # If training is in progress, show the training process
        if st.session_state['training_in_progress']:
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Starting model training...")
                progress_bar.progress(10)
                
                # Train model
                status_text.text("Training model...")
                results = train_model(
                    st.session_state['processed_sales'],
                    st.session_state['processed_stores']
                )
                progress_bar.progress(50)
                
                status_text.text("Processing results...")
                # Store results in session state
                st.session_state['training_results'] = results
                st.session_state['training_in_progress'] = False
                progress_bar.progress(60)
                
                # Display training results
                st.header("Training Results")
                progress_bar.progress(70)
                
                # Display R² scores for each model
                st.subheader("Model Performance")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Median Model",
                        f"Train R²: {results['train_scores']['median']:.3f}",
                        f"Test R²: {results['test_scores']['median']:.3f}"
                    )
                
                with col2:
                    st.metric(
                        "Lower Tail Model",
                        f"Train R²: {results['train_scores']['lower']:.3f}",
                        f"Test R²: {results['test_scores']['lower']:.3f}"
                    )
                
                with col3:
                    st.metric(
                        "Upper Tail Model",
                        f"Train R²: {results['train_scores']['upper']:.3f}",
                        f"Test R²: {results['test_scores']['upper']:.3f}"
                    )
                progress_bar.progress(80)
                
                # Display prediction interval metrics
                st.subheader("Prediction Interval Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Coverage",
                        f"{results['prediction_metrics']['coverage']:.1f}%",
                        help="Percentage of actual values falling within prediction intervals"
                    )
                
                with col2:
                    st.metric(
                        "Sharpness",
                        f"₹{results['prediction_metrics']['sharpness']:,.2f}",
                        help="Average width of prediction intervals (P90 - P10)"
                    )
                
                with col3:
                    st.metric(
                        "RMSE",
                        f"₹{results['prediction_metrics']['rmse']:,.2f}",
                        help="Root Mean Squared Error of median predictions"
                    )
                
                with col4:
                    st.metric(
                        "MAE",
                        f"₹{results['prediction_metrics']['mae']:,.2f}",
                        help="Mean Absolute Error of median predictions"
                    )
                progress_bar.progress(90)
                
                # Display feature importances as bar chart
                importances_df = pd.DataFrame(
                    list(results['feature_importances'].items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importances_df,
                    x='Feature',
                    y='Importance',
                    title='Feature Importances (Median Model)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display top 10 features
                st.subheader("Top 10 Most Important Features")
                for feat, imp in list(results['feature_importances'].items())[:10]:
                    st.write(f"{feat}: {imp:.3f}")
                
                # Display R² progression
                if 'r2_progression' in results:
                    st.subheader("R² Score Progression")
                    r2_df = pd.DataFrame({
                        'Iteration': range(len(results['r2_progression'])),
                        'R² Score': results['r2_progression']
                    })
                    fig = px.line(r2_df, x='Iteration', y='R² Score', title='R² Score Progression')
                    st.plotly_chart(fig, use_container_width=True)
                
                progress_bar.progress(100)
                status_text.text("Model training completed successfully!")
                st.success("Model training completed successfully!")
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.error("Full error details:")
                st.exception(e)
                # Log the full error traceback
                import traceback
                st.error("Full traceback:")
                st.code(traceback.format_exc())
                # Reset training state on error
                st.session_state['training_in_progress'] = False

elif page == "Predict":
    st.title("Predict Revenue")
    
    if 'processed_sales' not in st.session_state or 'processed_stores' not in st.session_state:
        st.warning("Please upload and process data first.")
    else:
        # Get inputs
        store_id = st.selectbox("Select Store", options=st.session_state.processed_stores['id'].unique())
        future_date = st.date_input("Select Future Date", min_value=datetime.now().date())
        store_area = st.number_input("Store Area (sq ft)", min_value=0.0, value=float(st.session_state.processed_stores[st.session_state.processed_stores['id'] == store_id]['store_area'].iloc[0]))
        
        # Update discount slider to handle percentages
        discount_pct = st.slider(
            "Discount Percentage",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            format="%.1f%%"
        ) / 100.0  # Convert percentage to decimal
        
        if st.button("Predict"):
            try:
                # Create synthetic sales row
                synthetic_sales = pd.DataFrame({
                    'store': [store_id],
                    'date': [pd.Timestamp(future_date)],
                    'qty_sold': [0],
                    'revenue': [0],
                    'disc_value': [0],  # Placeholder, user can update if needed
                    'disc_perc': [discount_pct]  # Already in decimal form
                })
                
                # Get matching store row
                store_row = st.session_state.processed_stores[st.session_state.processed_stores['id'] == store_id].copy()
                store_row['store_area'] = store_area  # Update with user input
                
                # Get recent historical sales for the store (e.g., last 60 days before the prediction date)
                historical_sales = st.session_state.processed_sales[
                    (st.session_state.processed_sales['store'] == store_id) &
                    (st.session_state.processed_sales['date'] < pd.Timestamp(future_date))
                ].sort_values('date').tail(60)

                # Generate features using historical sales
                X_pred, features = derive_features(synthetic_sales, store_row, historical_sales=historical_sales, is_prediction=True)
                # Fill any remaining NaN values with 0
                X_pred = X_pred.fillna(0)
                # Load model feature names
                with open('models/brandA_feature_names.json', 'r') as f:
                    feature_names = json.load(f)
                required_features = feature_names['all_features']
                # Add any missing features with zeros and reorder columns
                missing_features = set(required_features) - set(X_pred.columns)
                if missing_features:
                    for feat in missing_features:
                        X_pred[feat] = 0
                X_pred = X_pred.reindex(columns=required_features, fill_value=0)
                # Make prediction
                results = predict_and_explain(X_pred, historical_sales=historical_sales, original_input=synthetic_sales)
                
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
                        'disc_perc': ['mean', 'count']  # Add count to verify data points
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
                            # Only show discount if we have data points
                            if metrics[('disc_perc', 'count')] > 0:
                                st.metric(
                                    "Avg Discount",
                                    f"{metrics[('disc_perc', 'mean')]:.2f}%"
                                )
                            else:
                                st.metric(
                                    "Avg Discount",
                                    "N/A"
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
                        'disc_perc': ['first', 'count']  # Add count to verify data points
                    }).round(2)
                    
                    # Display metrics in columns
                    cols = st.columns(len(same_day_metrics))
                    for idx, (year, metrics) in enumerate(same_day_metrics.iterrows()):
                        with cols[idx]:
                            st.markdown(f"**{year}**")
                            st.metric(
                                "Revenue",
                                f"₹{metrics[('revenue', 'first')]:,.2f}"
                            )
                            st.metric(
                                "Qty Sold",
                                f"{metrics[('qty_sold', 'first')]:,.0f}"
                            )
                            # Only show discount if we have data points
                            if metrics[('disc_perc', 'count')] > 0:
                                st.metric(
                                    "Discount",
                                    f"{metrics[('disc_perc', 'first')]:.2f}%"
                                )
                            else:
                                st.metric(
                                    "Discount",
                                    "N/A"
                                )
                
                st.markdown("---")
                
                # 4. Yearly Averages
                st.markdown("#### Yearly Averages")
                yearly_avg = historical_data.groupby('year').agg({
                    'revenue': ['sum', 'mean'],
                    'qty_sold': ['sum', 'mean'],
                    'disc_perc': ['mean', 'count']  # Add count to verify data points
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
                            # Only show discount if we have data points
                            if metrics[('disc_perc', 'count')] > 0:
                                st.metric(
                                    "Avg Discount",
                                    f"{metrics[('disc_perc', 'mean')]:.2f}%"
                                )
                            else:
                                st.metric(
                                    "Avg Discount",
                                    "N/A"
                                )
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.error("Full error details:")
                st.exception(e)

def main():
    st.title("Fashion Revenue Predictor")
    
    # Sidebar
    st.sidebar.header("Model Training")
    if st.sidebar.button("Train New Model"):
        with st.spinner("Training model..."):
            # Generate synthetic data
            df_sales, df_stores = generate_synthetic_data()
            
            # Train model
            results = train_model(df_sales, df_stores)
            
            # Display training results
            st.sidebar.success("Model training completed!")
            
            # Display scores for each model
            st.sidebar.subheader("Model Performance")
            for model_type in ['median', 'lower', 'upper']:
                st.sidebar.metric(
                    f"{model_type.title()} Model",
                    f"Train R²: {results['train_scores'][model_type]:.3f}",
                    f"Test R²: {results['test_scores'][model_type]:.3f}"
                )
            
            # Display top features
            st.sidebar.subheader("Top 5 Features")
            for feat, imp in list(results['feature_importances'].items())[:5]:
                st.sidebar.text(f"{feat}: {imp:.3f}")
