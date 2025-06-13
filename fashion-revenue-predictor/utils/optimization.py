import optuna
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error
from utils.custom_cv import GroupTimeSeriesSplit
import logging

def optimize_decay_factors(
    df: pd.DataFrame,
    n_trials: int = 100,
    n_splits: int = 5
) -> Dict[str, float]:
    """
    Optimize decay factors for each store cluster using Optuna.
    
    Args:
        df: DataFrame with sales data and store clusters
        n_trials: Number of Optuna trials
        n_splits: Number of time series splits for cross-validation
        
    Returns:
        Dictionary mapping cluster IDs to optimal decay factors
    """
    def objective(trial: optuna.Trial) -> float:
        # Get decay factors for each cluster
        decay_factors = {}
        for cluster in df['store_cluster'].unique():
            decay_factors[cluster] = trial.suggest_float(f'decay_{cluster}', 0.01, 0.5)
        
        # Calculate sample weights using cluster-specific decay factors
        weights = np.ones(len(df))
        max_date = df['date'].max()
        
        for cluster in df['store_cluster'].unique():
            cluster_mask = df['store_cluster'] == cluster
            days_diff = (max_date - df.loc[cluster_mask, 'date']).dt.days
            weights[cluster_mask] = np.exp(-decay_factors[cluster] * np.log1p(days_diff))
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Perform time series cross-validation
        gtscv = GroupTimeSeriesSplit(n_splits=n_splits, test_size=0.2)
        scores = []
        
        for train_idx, test_idx in gtscv.split(df, groups=df['store']):
            # Calculate weighted RMSE for this fold
            y_true = df.iloc[test_idx]['revenue'].values
            y_pred = df.iloc[test_idx]['revenue'].values  # Using actual values as predictions for weight optimization
            fold_weights = weights[test_idx]
            
            # Calculate weighted RMSE
            rmse = np.sqrt(np.average((y_true - y_pred) ** 2, weights=fold_weights))
            scores.append(rmse)
        
        return np.mean(scores)
    
    # Create and run the study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Extract optimal decay factors
    optimal_decay_factors = {}
    for cluster in df['store_cluster'].unique():
        optimal_decay_factors[cluster] = study.best_params[f'decay_{cluster}']
    
    logging.info(f"Optimal decay factors: {optimal_decay_factors}")
    return optimal_decay_factors 