import joblib
import json
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple
from scipy.special import inv_boxcox
from utils.feature_engineering import apply_cluster_specific_transforms
from utils.conformal_calibration import ConformalCalibrator
import logging

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
    models = joblib.load('models/brandA_models.pkl')
    with open('models/brandA_feature_names.json', 'r') as f:
        feature_names = json.load(f)
    
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

    # Get predictions from each model using appropriate feature sets
    # 1. Median predictions (log scale)
    median_raw = models['median'].predict(df_X[feature_names['all_features']])
    print("DEBUG: Raw median model output:", median_raw)
    
    # 2. Lower tail predictions (residuals)
    lower_residuals = models['lower'].predict(df_X[feature_names['lower_features']])
    
    # 3. Upper tail predictions (residuals)
    upper_residuals = models['upper'].predict(df_X[feature_names['upper_features']])
    
    # Sanity check: Clip raw outputs to reasonable range before expm1
    median_raw = np.clip(median_raw, -10, 20)
    median_pred = np.expm1(median_raw)
    # Cap residuals at ±50% of median_pred
    lower_residuals = np.clip(lower_residuals, 0, 0.5 * median_pred)
    upper_residuals = np.clip(upper_residuals, 0, 0.5 * median_pred)

    logging.debug(f"Raw median_pred: {median_pred}")
    logging.debug(f"Lower residuals (capped): {lower_residuals}")
    logging.debug(f"Upper residuals (capped): {upper_residuals}")

    # Calculate initial prediction intervals
    p10 = np.maximum(median_pred - lower_residuals, 0)  # Ensure non-negative
    p50 = median_pred
    p90 = median_pred + upper_residuals

    logging.debug(f"Initial p10: {p10}")
    logging.debug(f"Initial p50: {p50}")
    logging.debug(f"Initial p90: {p90}")

    # Ensure p10 < p50 < p90 for all predictions
    preds = np.vstack([p10, p50, p90])
    preds_sorted = np.sort(preds, axis=0)
    p10, p50, p90 = preds_sorted[0], preds_sorted[1], preds_sorted[2]
    
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
                floor = past['revenue'].mean() * 0.5 if not past.empty else 0
                p10[idx] = max(p10[idx], floor)

    # --- Fallback if confidence is trash ---
    if 'revenue_std' in df_X.columns:
        if (p90 - p10).mean() > 2 * df_X['revenue_std'].mean():
            logging.warning('Prediction interval too wide, falling back to historical mean.')
            hist_mean = historical_sales['revenue'].mean() if historical_sales is not None else 0
            p10 = p50 = p90 = np.full_like(p10, hist_mean)

    # --- Weighted ensemble for p50 ---
    if 'store_month_weekday_avg' in df_X.columns:
        if df_X['store_month_weekday_avg'].isna().all():
            logging.warning('store_month_weekday_avg is all NaN, skipping ensemble adjustment.')
        else:
            p50 = 0.7 * p50 + 0.3 * df_X['store_month_weekday_avg'].fillna(0)
    if np.isnan(p50).any():
        logging.warning('p50 is NaN after ensemble adjustment, setting to p10 or 0.')
        p50 = np.nan_to_num(p50, nan=p10, posinf=p10, neginf=p10)

    # Calculate SHAP values using median model
    explainer = shap.TreeExplainer(models['median'])
    shap_values = explainer.shap_values(df_X[feature_names['all_features']])
    
    # Get feature importance scores
    feature_importance = np.abs(shap_values).mean(axis=0)
    feature_importance = dict(zip(feature_names['all_features'], feature_importance))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    # Get top 5 features
    top_5_features = list(sorted_importance.items())[:5]
    
    # After all calibration and sorting, invert log1p transformation for all predictions
    print("DEBUG: Final predictions (p10, p50, p90):", p10, p50, p90)
    
    # --- Post-processing rules for prediction intervals ---
    if 'store_weekday_avg' in df_X.columns:
        p10 = np.maximum(p10, 0.5 * df_X['store_weekday_avg'])
    if 'revenue_rolling_mean_3d' in df_X.columns:
        p50 = 0.8 * p50 + 0.2 * df_X['revenue_rolling_mean_3d']
    if 'revenue_volatility_3d' in df_X.columns:
        vol = df_X['revenue_volatility_3d'].fillna(0)
        p10 = np.where(vol > 0.4, p50 - 1.2 * (p50 - p10), p10)
        p90 = np.where(vol > 0.4, p50 + 1.2 * (p90 - p50), p90)
    preds = np.vstack([p10, p50, p90])
    p10, p50, p90 = np.sort(preds, axis=0)
    if 'store_seasonal_index' in df_X.columns:
        upper_clip = df_X['store_seasonal_index'] * 1.5 * p50
        p90 = np.minimum(p90, upper_clip)
    shap_sum = np.abs(shap_values).sum(axis=1)
    if 'revenue_rolling_mean_7d' in df_X.columns:
        p50 = np.where(shap_sum < 0.1, df_X['revenue_rolling_mean_7d'], p50)
    if 'store_month_revenue_median' in df_X.columns and 'store_month_revenue_std' in df_X.columns:
        upper_bound = df_X['store_month_revenue_median'] + 1.5 * df_X['store_month_revenue_std']
        p90 = np.minimum(p90, upper_bound)
    if all([f'revenue_lag_{lag}' in df_X.columns for lag in [7, 14, 30]]):
        rolling_avg = (df_X['revenue_lag_7'] + df_X['revenue_lag_14'] + df_X['revenue_lag_30']) / 3
        p50 = 0.75 * p50 + 0.25 * rolling_avg
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
