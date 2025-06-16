import joblib
import json
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple
from scipy.special import inv_boxcox
from utils.feature_engineering import apply_cluster_specific_transforms, align_features
from utils.conformal_calibration import ConformalCalibrator
import logging
import warnings

def predict_and_explain(df_X: pd.DataFrame, historical_sales: pd.DataFrame = None, original_input: pd.DataFrame = None) -> Dict:
    """
    Make predictions and generate SHAP explanations for input data.
    
    Args:
        df_X: DataFrame with features matching the model's expected input
        historical_sales: DataFrame with recent historical sales for the store (optional, for p10 floor)
        original_input: (optional) DataFrame with original input data, used to restore 'date' if dropped
        
    Returns:
        Dictionary containing:
        - p10, p50, p90 predictions with dynamic confidence intervals
        - SHAP values
        - Top 5 most important features
    """
    # Load models and features
    models = joblib.load('models/brandA_model.pkl')
    with open('models/brandA_feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
    # Remove 'revenue_percentile' from features if present
    if 'revenue_percentile' in feature_names['all_features']:
        feature_names['all_features'].remove('revenue_percentile')
    if 'revenue_percentile' in df_X.columns:
        df_X = df_X.drop(columns=['revenue_percentile'])
    
    # Lock feature set for inference
    df_X = align_features(df_X)
    
    # Debug: Print input features
    numeric_X = df_X[feature_names['all_features']].apply(pd.to_numeric, errors='coerce')
    print("DEBUG: Input features to model (first row):")
    print(df_X[feature_names['all_features']].iloc[0])
    print("Any NaN in X_pred:", numeric_X.isna().any().any())
    print("Any inf in X_pred:", numeric_X.replace([np.inf, -np.inf], np.nan).isna().any().any())
    print("Max value in X_pred:", numeric_X.max().max())
    print("Min value in X_pred:", numeric_X.min().min())
    
    # Apply cluster-specific transformations
    if 'store_cluster' in df_X.columns:
        transformed_dfs = []
        for cluster in df_X['store_cluster'].unique():
            cluster_df = df_X[df_X['store_cluster'] == cluster].copy()
            transformed_df = apply_cluster_specific_transforms(cluster_df, cluster)
            transformed_dfs.append(transformed_df)
        
        # Combine transformed DataFrames
        df_X = pd.concat(transformed_dfs, axis=0)
        df_X = df_X.sort_index()  # Restore original order
    
    # Remove qty_sold based features from inference
    qty_cols = [col for col in df_X.columns if 'qty_sold' in col or col == 'qty_sold']
    if qty_cols:
        df_X = df_X.drop(columns=qty_cols)

    # Add normalized discount_pct (z-score by channel and region)
    if 'discount_pct' in df_X.columns and 'channel_encoded' in df_X.columns and 'region_encoded' in df_X.columns:
        for group_col in ['channel_encoded', 'region_encoded']:
            group_means = df_X.groupby(group_col)['discount_pct'].transform('mean')
            group_stds = df_X.groupby(group_col)['discount_pct'].transform('std').replace(0, 1)
            df_X[f'discount_pct_z_{group_col}'] = (df_X['discount_pct'] - group_means) / group_stds

    # Add store×month×weekday seasonal index
    if historical_sales is not None:
        # Ensure 'date' is present in df_X
        if 'date' not in df_X.columns and original_input is not None and 'date' in original_input.columns:
            df_X['date'] = original_input['date'].values
        if 'date' not in df_X.columns:
            raise KeyError("'date' column is required in df_X for seasonal index calculation.")
        # Ensure 'store' is present in df_X
        if 'store' not in df_X.columns and original_input is not None and 'store' in original_input.columns:
            df_X['store'] = original_input['store'].values
        if 'store' not in df_X.columns:
            raise KeyError("'store' column is required in df_X for seasonal index calculation.")
        seasonal_idx = historical_sales.groupby([
            'store',
            historical_sales['date'].dt.month.rename('month'),
            historical_sales['date'].dt.dayofweek.rename('weekday')
        ])['revenue'].mean().reset_index()
        seasonal_idx.columns = ['store', 'month', 'weekday', 'store_month_weekday_avg']
        df_X = df_X.copy()
        df_X['month'] = df_X['date'].dt.month
        # Robust assignment for weekday
        df_X['weekday'] = df_X['date'].dt.dayofweek
        df_X = df_X.merge(seasonal_idx, how='left', left_on=['store', 'month', 'weekday'], right_on=['store', 'month', 'weekday'])
        if 'store_month_weekday_avg' not in df_X.columns:
            df_X['store_month_weekday_avg'] = 0

    # --- Ensure all required features are present in X_pred before prediction ---
    required_features = feature_names['all_features']
    # Drop any extra columns not in required_features
    extra_cols = [col for col in df_X.columns if col not in required_features]
    if extra_cols:
        df_X = df_X.drop(columns=extra_cols)
    # Add any missing columns as zeros
    missing_cols = [col for col in required_features if col not in df_X.columns]
    if missing_cols:
        for col in missing_cols:
            df_X[col] = 0
    df_X = df_X[required_features]

    # --- Force-inject key business features if missing ---
    for key_feat in ['store_dayofweek_avg', 'store_month_avg', 'weekday_store_avg']:
        if key_feat not in df_X.columns:
            df_X[key_feat] = 0

    # --- Backfill lag/rolling features with store_dayofweek_avg if available ---
    lag_rolling_cols = [col for col in df_X.columns if ('lag' in col or 'rolling' in col)]
    if 'store_dayofweek_avg' in df_X.columns:
        for col in lag_rolling_cols:
            if df_X[col].isna().any() or (df_X[col] == 0).all():
                df_X[col] = df_X[col].fillna(df_X['store_dayofweek_avg'])
                if (df_X[col] == 0).all():
                    df_X[col] = df_X['store_dayofweek_avg']

    # --- DEBUG: Log X_pred sum, NaNs, columns, and row data ---
    row_sum = df_X.sum(axis=1)
    row_nans = df_X.isna().sum(axis=1)
    print('DEBUG: X_pred sum(axis=1):', row_sum.values)
    print('DEBUG: X_pred isna().sum(axis=1):', row_nans.values)
    print('DEBUG: X_pred columns:', list(df_X.columns))
    print('DEBUG: X_pred row data:', df_X.iloc[0].to_dict())

    # --- Patch revenue_lag_7 if zero or NaN ---
    if 'revenue_lag_7' in df_X.columns:
        if np.isnan(df_X['revenue_lag_7'].values[0]) or df_X['revenue_lag_7'].values[0] == 0:
            if 'store_dayofweek_avg' in df_X.columns:
                df_X['revenue_lag_7'].values[0] = df_X['store_dayofweek_avg'].values[0]

    # --- Patch and warn if all lag/rolling features are zero ---
    logger = logging.getLogger("predictor")
    lag_feats = [col for col in df_X.columns if 'lag' in col or 'rolling' in col]
    if all(df_X[lag_feats].iloc[0] == 0):
        logger.warning("All lag/rolling features are zero. Prediction likely to be unreliable.")
    # Patch zero lag/rolling features (use .at for safe assignment)
    critical_lag_features = [col for col in lag_feats if col in df_X.columns]
    global_avg = df_X['store_month_avg'].values[0] if 'store_month_avg' in df_X.columns else 0
    for col in critical_lag_features:
        if df_X.at[0, col] == 0.0:
            fallback_val = df_X.at[0, 'store_dayofweek_avg'] if 'store_dayofweek_avg' in df_X.columns else global_avg
            df_X.at[0, col] = fallback_val

    # --- Ensure 'store', 'month', 'day_of_week' columns are present for fallback logic ---
    if 'store' not in df_X.columns and original_input is not None and 'store' in original_input.columns:
        df_X['store'] = original_input['store'].values
    if 'month' not in df_X.columns and original_input is not None and 'month' in original_input.columns:
        df_X['month'] = original_input['month'].values
    if 'day_of_week' not in df_X.columns and original_input is not None and 'day_of_week' in original_input.columns:
        df_X['day_of_week'] = original_input['day_of_week'].values
    # If still missing, raise a clear error
    for col in ['store', 'month', 'day_of_week']:
        if col not in df_X.columns:
            raise KeyError(f"Required column '{col}' missing from features for fallback logic. Check feature engineering and input data.")

    # --- Backfill store_month_avg if zero ---
    if 'store_month_avg' in df_X.columns and df_X['store_month_avg'].values[0] == 0.0 and historical_sales is not None:
        store = df_X['store'].values[0]
        month = df_X['month'].values[0]
        # Level 1: Try exact store and month match
        mask = (historical_sales['store'] == store) & (historical_sales['date'].dt.month == month)
        if not historical_sales[mask].empty:
            fallback = historical_sales[mask]['revenue'].mean()
            logger.info(f"[Fallback][store_month_avg] Level 1: store={store}, month={month}, value={fallback}")
        else:
            # Level 2: Try same store, any month
            mask = (historical_sales['store'] == store)
            if not historical_sales[mask].empty:
                fallback = historical_sales[mask]['revenue'].mean()
                logger.info(f"[Fallback][store_month_avg] Level 2: store={store}, any month, value={fallback}")
            else:
                # Level 3: Try same month, any store
                mask = (historical_sales['date'].dt.month == month)
                if not historical_sales[mask].empty:
                    fallback = historical_sales[mask]['revenue'].mean()
                    logger.info(f"[Fallback][store_month_avg] Level 3: any store, month={month}, value={fallback}")
                else:
                    # Level 4: Use global average
                    fallback = historical_sales['revenue'].mean()
                    logger.info(f"[Fallback][store_month_avg] Level 4: global average, value={fallback}")
        df_X['store_month_avg'] = fallback
        logger.info(f"Backfilled store_month_avg with {fallback:.2f} using fallback logic")
        logger.info(f"[Fallback][store_month_avg] Final value used: {df_X['store_month_avg'].values[0]}")

    # --- Backfill store_dayofweek_avg if zero ---
    if 'store_dayofweek_avg' in df_X.columns and df_X['store_dayofweek_avg'].values[0] == 0.0 and historical_sales is not None:
        store = df_X['store'].values[0]
        dow = df_X['day_of_week'].values[0]
        # Level 1: Try exact store and day of week match
        mask = (historical_sales['store'] == store) & (historical_sales['date'].dt.dayofweek == dow)
        if not historical_sales[mask].empty:
            fallback = historical_sales[mask]['revenue'].mean()
            logger.info(f"[Fallback][store_dayofweek_avg] Level 1: store={store}, dow={dow}, value={fallback}")
        else:
            # Level 2: Try same store, any day of week
            mask = (historical_sales['store'] == store)
            if not historical_sales[mask].empty:
                fallback = historical_sales[mask]['revenue'].mean()
                logger.info(f"[Fallback][store_dayofweek_avg] Level 2: store={store}, any dow, value={fallback}")
            else:
                # Level 3: Try same day of week, any store
                mask = (historical_sales['date'].dt.dayofweek == dow)
                if not historical_sales[mask].empty:
                    fallback = historical_sales[mask]['revenue'].mean()
                    logger.info(f"[Fallback][store_dayofweek_avg] Level 3: any store, dow={dow}, value={fallback}")
                else:
                    # Level 4: Use store_month_avg or global average
                    fallback = df_X['store_month_avg'].values[0] if df_X['store_month_avg'].values[0] > 0 else historical_sales['revenue'].mean()
                    logger.info(f"[Fallback][store_dayofweek_avg] Level 4: fallback to store_month_avg/global, value={fallback}")
        df_X['store_dayofweek_avg'] = fallback
        logger.info(f"Backfilled store_dayofweek_avg with {fallback:.2f} using fallback logic")
        logger.info(f"[Fallback][store_dayofweek_avg] Final value used: {df_X['store_dayofweek_avg'].values[0]}")

    # --- Enhanced backfill for lag/rolling features ---
    lag_rolling_cols = [col for col in df_X.columns if ('lag' in col or 'rolling' in col)]
    if lag_rolling_cols:
        # Calculate global statistics for fallback
        global_mean = historical_sales['revenue'].mean() if historical_sales is not None else 10000
        global_std = historical_sales['revenue'].std() if historical_sales is not None else 5000
        for col in lag_rolling_cols:
            if df_X[col].isna().any() or (df_X[col] == 0).all():
                # Try multiple fallback levels
                if 'store_dayofweek_avg' in df_X.columns and df_X['store_dayofweek_avg'].values[0] > 0:
                    fallback = df_X['store_dayofweek_avg'].values[0]
                    logger.info(f"[Fallback][{col}] Level 1: store_dayofweek_avg, value={fallback}")
                elif 'store_month_avg' in df_X.columns and df_X['store_month_avg'].values[0] > 0:
                    fallback = df_X['store_month_avg'].values[0]
                    logger.info(f"[Fallback][{col}] Level 2: store_month_avg, value={fallback}")
                else:
                    # Use global statistics with some randomness to avoid identical predictions
                    fallback = global_mean + np.random.normal(0, global_std * 0.1)
                    logger.info(f"[Fallback][{col}] Level 3: global mean + noise, value={fallback}")
                df_X[col] = df_X[col].fillna(fallback)
                if (df_X[col] == 0).all():
                    df_X[col] = fallback
                logger.info(f"Backfilled {col} with {fallback:.2f} using fallback logic")
                logger.info(f"[Fallback][{col}] Final value used: {df_X[col].values[0]}")

    # --- Custom fallback value calculation as per user formula ---
    def calculate_custom_fallback(df_X, historical_sales, original_input):
        # Get store, month, and day_of_week
        store = df_X['store'].values[0] if 'store' in df_X.columns else None
        month = df_X['month'].values[0] if 'month' in df_X.columns else None
        day_of_week = df_X['day_of_week'].values[0] if 'day_of_week' in df_X.columns else None
        # Filter historical sales for this store, month, and day_of_week
        if store is not None and month is not None and day_of_week is not None and historical_sales is not None:
            mask = (
                (historical_sales['store'] == store) &
                (historical_sales['date'].dt.month == month) &
                (historical_sales['date'].dt.dayofweek == day_of_week)
            )
            store_hist = historical_sales[mask]
            if not store_hist.empty:
                # Use the most recent year's average as base
                latest_year = store_hist['date'].dt.year.max()
                latest_year_avg = store_hist[store_hist['date'].dt.year == latest_year]['revenue'].mean()
                # Calculate year-over-year growth
                yearly_avg = store_hist.groupby(store_hist['date'].dt.year)['revenue'].mean()
                if len(yearly_avg) > 1:
                    growth_rates = yearly_avg.pct_change().dropna()
                    avg_growth_rate = growth_rates.mean()
                    # Apply growth rate to latest year's average
                    avg_revenue = latest_year_avg * (1 + avg_growth_rate)
                else:
                    avg_revenue = latest_year_avg
            else:
                avg_revenue = historical_sales['revenue'].mean()
        else:
            avg_revenue = historical_sales['revenue'].mean() if historical_sales is not None else 0
        # Get discount percentage
        if 'discount_pct' in df_X.columns:
            discount_perc = df_X['discount_pct'].values[0] * 100
        elif original_input is not None and 'discount_pct' in original_input.columns:
            discount_perc = original_input['discount_pct'].values[0] * 100
        else:
            discount_perc = 0
        discount_factor = (100 + discount_perc) / 100
        # Calculate fallback
        fallback = avg_revenue * discount_factor
        logger.info(f"[CustomFallback] avg_revenue={avg_revenue}, discount_factor={discount_factor}, fallback={fallback}")
        return fallback

    # --- Enhanced prediction fallback logic ---
    X_pred = df_X[feature_names['all_features']]
    input_sum = X_pred.sum(axis=1).item()
    input_nans = X_pred.isna().sum(axis=1).item()
    
    # Calculate robust fallback values
    store_dayofweek_avg = df_X['store_dayofweek_avg'].values[0] if 'store_dayofweek_avg' in df_X.columns else 0
    store_month_avg_val = df_X['store_month_avg'].values[0] if 'store_month_avg' in df_X.columns else 0
    
    # Use the larger of the two averages as fallback
    fallback_value = max(store_dayofweek_avg, store_month_avg_val)
    if fallback_value == 0 and historical_sales is not None:
        # Calculate fallback based on historical data for the same store, month, and day of week
        store = df_X['store'].values[0] if 'store' in df_X.columns else None
        month = df_X['month'].values[0] if 'month' in df_X.columns else None
        day_of_week = df_X['day_of_week'].values[0] if 'day_of_week' in df_X.columns else None
        if store is not None and month is not None and day_of_week is not None:
            mask = (
                (historical_sales['store'] == store) &
                (historical_sales['date'].dt.month == month) &
                (historical_sales['date'].dt.dayofweek == day_of_week)
            )
            store_hist = historical_sales[mask]
            if not store_hist.empty:
                # Use the most recent year's average as base
                latest_year = store_hist['date'].dt.year.max()
                latest_year_avg = store_hist[store_hist['date'].dt.year == latest_year]['revenue'].mean()
                # Calculate year-over-year growth
                yearly_avg = store_hist.groupby(store_hist['date'].dt.year)['revenue'].mean()
                if len(yearly_avg) > 1:
                    growth_rates = yearly_avg.pct_change().dropna()
                    avg_growth_rate = growth_rates.mean()
                    # Apply growth rate to latest year's average
                    fallback_value = latest_year_avg * (1 + avg_growth_rate)
                else:
                    fallback_value = latest_year_avg
            else:
                fallback_value = historical_sales['revenue'].mean()
        else:
            fallback_value = historical_sales['revenue'].mean()
    
    # Use custom fallback if model output is invalid or zero
    custom_fallback_value = calculate_custom_fallback(df_X, historical_sales, original_input)
    
    # Model prediction with enhanced fallback
    rf_pred = models['median'].predict(X_pred)
    # Compute SHAP values for the median model
    explainer = shap.TreeExplainer(models['median'])
    shap_values = explainer.shap_values(X_pred)
    # Get top 5 features by mean absolute SHAP value
    shap_importance = np.abs(shap_values).mean(axis=0)
    top_5_idx = np.argsort(shap_importance)[::-1][:5]
    top_5_features = [feature_names['all_features'][i] for i in top_5_idx]
    # Define sorted_importance for SHAP audit
    sorted_importance = dict(sorted(zip(feature_names['all_features'], shap_importance), key=lambda x: x[1], reverse=True))
    # If model was trained on log1p of revenue ratio, invert transform and multiply by denominator
    if hasattr(models['median'], 'feature_names_in_'):
        try:
            # 1. Clip log-prediction to prevent overflow
            rf_pred = np.clip(rf_pred, -10, 10)
            # 2. Invert log1p
            ratio_pred = np.expm1(rf_pred)
            # 3. Multiply by denominator (store_dayofweek_avg)
            denominator = df_X['store_dayofweek_avg'].values if 'store_dayofweek_avg' in df_X.columns else 1.0
            revenue_pred = ratio_pred * denominator
            # 4. Fallback if nan or inf
            if np.isnan(revenue_pred).any() or np.isinf(revenue_pred).any():
                logger.warning('Revenue prediction is nan or inf, falling back to denominator.')
                revenue_pred = denominator
            rf_pred = revenue_pred
        except Exception as e:
            logger.warning(f"Could not apply log1p-inverse and scaling to rf_pred: {e}")
            # If transformation fails, use the raw prediction
            rf_pred = np.exp(rf_pred)  # Simple exponential transformation as fallback
    fallback_triggered = False
    
    if (rf_pred is None or np.isnan(rf_pred).any() or np.isinf(rf_pred).any() or 
        (input_nans > len(feature_names['all_features']) * 0.5) or  # More than 50% features are NaN
        (input_sum == 0) or (rf_pred[0] == 0)):
        logger.warning("Model output is invalid or input data is too sparse or zero. Triggering custom fallback.")
        p50 = np.array([custom_fallback_value])
        fallback_triggered = True
    else:
        # If prediction is too low or high compared to fallback, use weighted average
        if abs(rf_pred[0] - fallback_value) > fallback_value * 2:  # If prediction differs by more than 2x
            p50 = np.array([0.5 * fallback_value + 0.5 * rf_pred[0]])  # Equal weight to model and fallback
            logger.info("Using weighted average of model prediction and fallback value")
        else:
            p50 = rf_pred
    logger.debug(f"rf_pred: {rf_pred}, fallback_triggered: {fallback_triggered}")
    
    # Ensure p50 is not too small
    if p50[0] < 1000:  # If prediction is less than 1000
        logger.warning(f"P50 prediction too low ({p50[0]}), using fallback value")
        p50 = np.array([fallback_value])
        fallback_triggered = True
    
    p10 = p50 * 0.5
    p90 = p50 * 1.5

    # Confidence bands: use historical residual std per store+day_of_week
    if historical_sales is not None and 'store' in df_X.columns and 'day_of_week' in df_X.columns:
        p10 = np.zeros_like(p50)
        p90 = np.zeros_like(p50)
        for idx, row in df_X.iterrows():
            store = row['store']
            dow = row['day_of_week']
            mask = (historical_sales['store'] == store) & (historical_sales['date'].dt.dayofweek == dow)
            if not historical_sales[mask].empty:
                actuals = historical_sales[mask]['revenue']
                pred = rf_pred[idx] * row['store_dayofweek_avg']
                residuals = actuals - pred
                std_resid = residuals.std() if not residuals.empty else 0
                p10[idx] = p50[idx] - std_resid
                p90[idx] = p50[idx] + std_resid
            else:
                p10[idx] = p50[idx]
                p90[idx] = p50[idx]
    else:
        p10 = p50.copy()
        p90 = p50.copy()
    # Enforce non-negativity and ordering
    p10 = np.maximum(p10, 0)
    preds = np.vstack([p10, p50, p90])
    p10, p50, p90 = np.sort(preds, axis=0)

    logging.debug(f"Initial p10: {p10}")
    logging.debug(f"Initial p50: {p50}")
    logging.debug(f"Initial p90: {p90}")

    # Apply conformal calibration
    calibrator = ConformalCalibrator(alpha=0.1)  # 90% coverage
    
    # --- Refactor p10 floor logic ---
    if historical_sales is not None:
        for idx, row in df_X.iterrows():
            store = row['store'] if 'store' in row else None
            month = row['month'] if 'month' in row else None
            weekday = row['weekday'] if 'weekday' in row else None
            if store is not None and month is not None and weekday is not None:
                past = historical_sales[
                    (historical_sales['store'] == store) &
                    (historical_sales['date'].dt.month == month) &
                    (historical_sales['date'].dt.dayofweek == weekday)
                ]
                floor = past['revenue'].mean() * 0.7 if not past.empty else 0  # Increased from 0.5 to 0.7
                p10[idx] = np.maximum(p10[idx], floor)

    # --- Fallback if confidence is trash ---
    if 'revenue_std' in df_X.columns:
        if (p90 - p10).mean() > 3 * df_X['revenue_std'].mean():  # Increased threshold from 2x to 3x
            logging.warning('Prediction interval too wide, falling back to historical mean.')
            hist_mean = historical_sales['revenue'].mean() if historical_sales is not None else 0
            p50 = np.full_like(p50, hist_mean)
            p10 = 0.8 * p50
            p90 = 1.2 * p50
            fallback_triggered = True

    # --- Weighted ensemble for p50 ---
    if 'store_month_weekday_avg' in df_X.columns:
        if df_X['store_month_weekday_avg'].isna().all():
            logging.warning('store_month_weekday_avg is all NaN, skipping ensemble adjustment.')
        else:
            p50 = 0.5 * p50 + 0.5 * df_X['store_month_weekday_avg'].fillna(0)  # Equal weight instead of 0.7/0.3
    if np.isnan(p50).any():
        logging.warning('p50 is NaN after ensemble adjustment, setting to p10 or 0.')
        p50 = np.nan_to_num(p50, nan=p10, posinf=p10, neginf=p10)

    # --- Wide interval/low-confidence prediction check ---
    interval_width = p90 - p10
    for i in range(len(p50)):
        if p50[i] < 1e-3 or interval_width[i] > 15 * max(p50[i], 1):  # Increased threshold from 10x to 15x
            logger.warning(f"Prediction interval too wide: width={interval_width[i]:.2f}, p50={p50[i]:.2f}")
            p10[i], p50[i], p90[i] = np.nan, np.nan, np.nan

    # --- Post-processing rules for prediction intervals ---
    if 'store_weekday_avg' in df_X.columns:
        p10 = np.maximum(p10, 0.7 * df_X['store_weekday_avg'])  # Increased from 0.5 to 0.7
    if 'revenue_rolling_mean_3d' in df_X.columns:
        p50 = 0.5 * p50 + 0.5 * df_X['revenue_rolling_mean_3d']  # Equal weight instead of 0.8/0.2
    if 'revenue_volatility_3d' in df_X.columns:
        vol = df_X['revenue_volatility_3d'].fillna(0)
        p10 = np.where(vol > 0.4, p50 - 1.2 * (p50 - p10), p10)
        p90 = np.where(vol > 0.4, p50 + 1.2 * (p90 - p50), p90)
    preds = np.vstack([p10, p50, p90])
    p10, p50, p90 = np.sort(preds, axis=0)
    if 'store_seasonal_index' in df_X.columns:
        upper_clip = df_X['store_seasonal_index'] * 2.0 * p50  # Increased from 1.5x to 2.0x
        p90 = np.minimum(p90, upper_clip)
    shap_sum = np.abs(shap_values).sum(axis=1)
    if 'revenue_rolling_mean_7d' in df_X.columns:
        p50 = np.where(shap_sum < 0.05, df_X['revenue_rolling_mean_7d'], p50)  # Reduced threshold from 0.1 to 0.05
    if 'store_month_revenue_median' in df_X.columns and 'store_month_revenue_std' in df_X.columns:
        upper_bound = df_X['store_month_revenue_median'] + 2.0 * df_X['store_month_revenue_std']  # Increased from 1.5x to 2.0x
        p90 = np.minimum(p90, upper_bound)
    if all([f'revenue_lag_{lag}' in df_X.columns for lag in [7, 14, 30]]):
        rolling_avg = (df_X['revenue_lag_7'] + df_X['revenue_lag_14'] + df_X['revenue_lag_30']) / 3
        p50 = 0.5 * p50 + 0.5 * rolling_avg  # Equal weight instead of 0.75/0.25
    if 'premium_location_discount' in df_X.columns:
        boost = 0.1 * df_X['premium_location_discount']
        p90 += boost
    if 'city_encoded' in df_X.columns and 'revenue_last_3_days' in df_X.columns:
        fallback = 0.5 * df_X['revenue_last_3_days'] + 0.5 * df_X['city_encoded']
        p50 = np.where(shap_sum < 0.05, fallback, p50)
    
    # --- Store-based fallback for p50 ---
    if historical_sales is not None and not historical_sales.empty:
        # Find min historical revenue for this store, weekday, and month
        for idx, row in df_X.iterrows():
            store = row['store'] if 'store' in row else None
            weekday = row['day_of_week'] if 'day_of_week' in row else None
            month = row['month'] if 'month' in row else None
            if store is not None and weekday is not None and month is not None:
                mask = (
                    (historical_sales['store'] == store) &
                    (historical_sales['date'].dt.dayofweek == weekday) &
                    (historical_sales['date'].dt.month == month)
                )
                same_day_hist = historical_sales[mask]
                if not same_day_hist.empty:
                    min_revenue = same_day_hist['revenue'].min()
                    avg_revenue = same_day_hist['revenue'].mean()
                    if p50[idx] < min_revenue:
                        logging.debug(f"p50[{idx}] ({p50[idx]}) < min historical same-day revenue ({min_revenue}), overriding with avg {avg_revenue}")
                        p50[idx] = avg_revenue
    logging.debug(f"Final p10: {p10}")
    logging.debug(f"Final p50: {p50}")
    logging.debug(f"Final p90: {p90}")
    
    # --- SHAP logging ---
    print('Top SHAP features:')
    for i, s in enumerate(np.abs(shap_values[0]).argsort()[::-1][:5]):
        print(f"{i+1}: {feature_names['all_features'][s]} -> {shap_values[0][s]:.2f}")

    # SHAP audit: log top 10 features and check for leakage
    top_10 = list(sorted_importance.items())[:10]
    logging.info('Top 10 SHAP features:')
    for feat, val in top_10:
        logging.info(f'{feat}: {val:.3f}')
    lag_patterns = ['revenue_lag_', 'revenue_rolling_', 'rolling_', 'lag_']
    for feat, _ in top_10:
        if any(pat in feat for pat in lag_patterns):
            logging.warning(f'Potential leakage: lag/rolling feature in top 10: {feat}')
    must_have = ['store', 'store_dayofweek_avg', 'discount_pct']
    for must in must_have:
        if not any(must in feat for feat, _ in top_10):
            logging.warning(f'Expected stable feature missing from top 10: {must}')

    # --- Calibration: skip if too little data ---
    try:
        historical_data = pd.read_json('models/historical_predictions.json')
        if len(historical_data) < 100:
            logging.warning('Too little historical data — skipping conformal calibration.')
        else:
            calibrator.calibrate(
                y_true=historical_data['revenue'].values,
                p10=historical_data['p10'].values,
                p50=historical_data['p50'].values,
                p90=historical_data['p90'].values
            )
            p10, p50, p90 = calibrator.calibrate_predictions(p10, p50, p90)
            p10 = np.maximum(p10, 0)
            preds = np.vstack([p10, p50, p90])
            preds_sorted = np.sort(preds, axis=0)
            p10, p50, p90 = preds_sorted[0], preds_sorted[1], preds_sorted[2]
    except FileNotFoundError:
        logging.warning('Historical predictions not found, skipping calibration')
    
    prediction_is_log_scaled = not fallback_triggered
    # --- Inverse log transform using model metadata ---
    # Load train_pred_mean and target_mean from model metadata
    try:
        with open('models/brandA_model_metadata.json', 'r') as f:
            model_meta = json.load(f)
        train_pred_mean = model_meta['train_pred_mean']
        target_mean = model_meta['target_mean']
    except Exception as e:
        logger.warning(f"Could not load model metadata for inverse transform: {e}")
        train_pred_mean = 0.0
        target_mean = 1.0

    # Model prediction (log space)
    p50_log = models['median'].predict(X_pred)
    p10_log = models['lower'].predict(X_pred)
    p90_log = models['upper'].predict(X_pred)

    # Inverse log transform for all quantiles
    scale = target_mean / np.expm1(train_pred_mean) if np.expm1(train_pred_mean) != 0 else 1.0
    p50 = np.expm1(p50_log) * scale
    p10 = np.expm1(p10_log) * scale
    p90 = np.expm1(p90_log) * scale

    # --- Post-processing rules for prediction intervals ---
    if 'store_weekday_avg' in df_X.columns:
        p10 = np.maximum(p10, 0.7 * df_X['store_weekday_avg'])  # Increased from 0.5 to 0.7
    if 'revenue_rolling_mean_3d' in df_X.columns:
        p50 = 0.5 * p50 + 0.5 * df_X['revenue_rolling_mean_3d']  # Equal weight instead of 0.8/0.2
    if 'revenue_volatility_3d' in df_X.columns:
        vol = df_X['revenue_volatility_3d'].fillna(0)
        p10 = np.where(vol > 0.4, p50 - 1.2 * (p50 - p10), p10)
        p90 = np.where(vol > 0.4, p50 + 1.2 * (p90 - p50), p90)
    preds = np.vstack([p10, p50, p90])
    p10, p50, p90 = np.sort(preds, axis=0)
    if 'store_seasonal_index' in df_X.columns:
        upper_clip = df_X['store_seasonal_index'] * 2.0 * p50  # Increased from 1.5x to 2.0x
        p90 = np.minimum(p90, upper_clip)
    shap_sum = np.abs(shap_values).sum(axis=1)
    if 'revenue_rolling_mean_7d' in df_X.columns:
        p50 = np.where(shap_sum < 0.05, df_X['revenue_rolling_mean_7d'], p50)  # Reduced threshold from 0.1 to 0.05
    if 'store_month_revenue_median' in df_X.columns and 'store_month_revenue_std' in df_X.columns:
        upper_bound = df_X['store_month_revenue_median'] + 2.0 * df_X['store_month_revenue_std']  # Increased from 1.5x to 2.0x
        p90 = np.minimum(p90, upper_bound)
    if all([f'revenue_lag_{lag}' in df_X.columns for lag in [7, 14, 30]]):
        rolling_avg = (df_X['revenue_lag_7'] + df_X['revenue_lag_14'] + df_X['revenue_lag_30']) / 3
        p50 = 0.5 * p50 + 0.5 * rolling_avg  # Equal weight instead of 0.75/0.25
    if 'premium_location_discount' in df_X.columns:
        boost = 0.1 * df_X['premium_location_discount']
        p90 += boost
    if 'city_encoded' in df_X.columns and 'revenue_last_3_days' in df_X.columns:
        fallback = 0.5 * df_X['revenue_last_3_days'] + 0.5 * df_X['city_encoded']
        p50 = np.where(shap_sum < 0.05, fallback, p50)

    # --- Store-based fallback for p50 ---
    if historical_sales is not None and not historical_sales.empty:
        # Find min historical revenue for this store, weekday, and month
        for idx, row in df_X.iterrows():
            store = row['store'] if 'store' in row else None
            weekday = row['day_of_week'] if 'day_of_week' in row else None
            month = row['month'] if 'month' in row else None
            if store is not None and weekday is not None and month is not None:
                mask = (
                    (historical_sales['store'] == store) &
                    (historical_sales['date'].dt.dayofweek == weekday) &
                    (historical_sales['date'].dt.month == month)
                )
                same_day_hist = historical_sales[mask]
                if not same_day_hist.empty:
                    min_revenue = same_day_hist['revenue'].min()
                    avg_revenue = same_day_hist['revenue'].mean()
                    if p50[idx] < min_revenue:
                        logging.debug(f"p50[{idx}] ({p50[idx]}) < min historical same-day revenue ({min_revenue}), overriding with avg {avg_revenue}")
                        p50[idx] = avg_revenue
    logging.debug(f"Final p10: {p10}")
    logging.debug(f"Final p50: {p50}")
    logging.debug(f"Final p90: {p90}")
    
    # --- SHAP logging ---
    print('Top SHAP features:')
    for i, s in enumerate(np.abs(shap_values[0]).argsort()[::-1][:5]):
        print(f"{i+1}: {feature_names['all_features'][s]} -> {shap_values[0][s]:.2f}")

    # SHAP audit: log top 10 features and check for leakage
    top_10 = list(sorted_importance.items())[:10]
    logging.info('Top 10 SHAP features:')
    for feat, val in top_10:
        logging.info(f'{feat}: {val:.3f}')
    lag_patterns = ['revenue_lag_', 'revenue_rolling_', 'rolling_', 'lag_']
    for feat, _ in top_10:
        if any(pat in feat for pat in lag_patterns):
            logging.warning(f'Potential leakage: lag/rolling feature in top 10: {feat}')
    must_have = ['store', 'store_dayofweek_avg', 'discount_pct']
    for must in must_have:
        if not any(must in feat for feat, _ in top_10):
            logging.warning(f'Expected stable feature missing from top 10: {must}')

    # --- CLIP AND VALIDATE TEST INPUTS ---
    try:
        with open('models/brandA_feature_percentiles.json', 'r') as f:
            percentiles = json.load(f)
        for col, bounds in percentiles.items():
            if col in df_X.columns:
                # Only clip if not a fallback zero
                if df_X.at[0, col] == 0.0:
                    continue
                lower, upper = bounds.get('p1', None), bounds.get('p99', None)
                if lower is not None and upper is not None:
                    before = df_X[col].copy()
                    df_X[col] = df_X[col].clip(lower, upper)
                    if (before != df_X[col]).any():
                        warnings.warn(f"Feature '{col}' clipped to [{lower}, {upper}] at prediction time.")
    except Exception as e:
        logging.warning(f"Could not load or apply feature percentiles for clipping: {e}")
    # --- VALIDATE INPUT STORE-DATE PAIR HAS HISTORY ---
    if historical_sales is not None and 'store' in df_X.columns and 'date' in df_X.columns:
        for idx, row in df_X.iterrows():
            store = row['store']
            date = row['date']
            hist = historical_sales[(historical_sales['store'] == store) & (historical_sales['date'] < date)]
            if hist.empty:
                warnings.warn(f"No historical data for store {store} before {date}. Falling back to store median.")
                for col in df_X.columns:
                    if any(pat in col for pat in ['lag', 'rolling', 'revenue_lag_', 'revenue_rolling_']):
                        df_X.at[idx, col] = historical_sales[historical_sales['store'] == store]['revenue'].median() if not historical_sales[historical_sales['store'] == store].empty else 0
    # --- LOG SHAP + PREDICTION CONFIDENCE FOR EVERY INFERENCE ---
    interval_width = p90 - p10
    for i in range(len(p50)):
        logging.info(f"Prediction {i}: p10={p10[i]:.2f}, p50={p50[i]:.2f}, p90={p90[i]:.2f}, interval width={interval_width[i]:.2f}")
        top_shap_idx = np.abs(shap_values[i]).argsort()[::-1][:5]
        top_shap = [(feature_names['all_features'][j], shap_values[i][j]) for j in top_shap_idx]
        logging.info(f"Top SHAP features for prediction {i}: {top_shap}")
        if interval_width[i] > 3 * p50[i]:
            warnings.warn(f"Prediction interval too wide for prediction {i}: width={interval_width[i]:.2f}, p50={p50[i]:.2f}")

    # --- Enforce custom fallback as minimum for all quantiles ---
    p10 = np.maximum(p10, custom_fallback_value)
    p50 = np.maximum(p50, custom_fallback_value)
    p90 = np.maximum(p90, custom_fallback_value)

    return {
        'p10': p10,
        'p50': p50,
        'p90': p90,
        'shap_values': shap_values,
        'top_5_features': top_5_features
    }

def calculate_prediction_metrics(y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> Dict:
    """
    Calculate prediction interval coverage and sharpness metrics.
    
    Args:
        y_true: Array of true values
        p10: Array of 10th percentile predictions
        p50: Array of 50th percentile predictions
        p90: Array of 90th percentile predictions
        
    Returns:
        Dictionary containing:
        - coverage: Percentage of true values falling within prediction intervals
        - sharpness: Average width of prediction intervals (p90 - p10)
        - rmse: Root mean squared error of median predictions
        - mae: Mean absolute error of median predictions
    """
    # Calculate coverage
    in_interval = np.logical_and(y_true >= p10, y_true <= p90)
    coverage = np.mean(in_interval) * 100
    
    # Calculate sharpness
    interval_width = p90 - p10
    sharpness = np.mean(interval_width)
    
    # Calculate RMSE and MAE for median predictions
    rmse = np.sqrt(np.mean((y_true - p50) ** 2))
    mae = np.mean(np.abs(y_true - p50))
    
    return {
        'coverage': coverage,
        'sharpness': sharpness,
        'rmse': rmse,
        'mae': mae
    }
