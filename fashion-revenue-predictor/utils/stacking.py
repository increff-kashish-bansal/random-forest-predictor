import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from utils.custom_cv import GroupTimeSeriesSplit
import logging

class StackingEnsemble:
    def __init__(
        self,
        n_splits: int = 5,
        lgb_params: Dict = None,
        rf_params: Dict = None
    ):
        """
        Initialize stacking ensemble with LightGBM and Random Forest.
        
        Args:
            n_splits: Number of time series splits for cross-validation
            lgb_params: LightGBM parameters
            rf_params: Random Forest parameters
        """
        self.n_splits = n_splits
        self.lgb_params = lgb_params or {
            'objective': 'quantile',
            'alpha': 0.5,  # For median prediction
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        self.rf_params = rf_params or {
            'n_estimators': 100,
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        self.base_models = {}
        self.meta_model = None
        
    def _get_oof_predictions(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        quantile: str
    ) -> np.ndarray:
        """
        Get out-of-fold predictions for base models.
        
        Args:
            X: Feature matrix
            y: Target values
            groups: Group labels for time series split
            quantile: Which quantile to predict ('lower', 'median', or 'upper')
            
        Returns:
            Array of out-of-fold predictions
        """
        oof_preds = np.zeros(len(X))
        gtscv = GroupTimeSeriesSplit(n_splits=self.n_splits, test_size=0.2)
        
        # Adjust LightGBM parameters based on quantile
        lgb_params = self.lgb_params.copy()
        if quantile == 'lower':
            lgb_params['alpha'] = 0.1
        elif quantile == 'upper':
            lgb_params['alpha'] = 0.9
        
        for train_idx, test_idx in gtscv.split(X, groups=groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y[train_idx]
            
            # Train LightGBM
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_model = lgb.train(
                lgb_params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_train],
                early_stopping_rounds=10,
                verbose_eval=False
            )
            
            # Train Random Forest
            rf_model = RandomForestRegressor(**self.rf_params)
            rf_model.fit(X_train, y_train)
            
            # Make predictions
            lgb_pred = lgb_model.predict(X_test)
            rf_pred = rf_model.predict(X_test)
            
            # Average predictions
            oof_preds[test_idx] = 0.5 * (lgb_pred + rf_pred)
            
            # Store base models
            self.base_models[f'lgb_{quantile}_{len(self.base_models)}'] = lgb_model
            self.base_models[f'rf_{quantile}_{len(self.base_models)}'] = rf_model
        
        return oof_preds
    
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        sample_weight: np.ndarray = None
    ) -> None:
        """
        Fit the stacking ensemble.
        
        Args:
            X: Feature matrix
            y: Target values
            groups: Group labels for time series split
            sample_weight: Sample weights
        """
        # Get out-of-fold predictions for each quantile
        oof_lower = self._get_oof_predictions(X, y, groups, 'lower')
        oof_median = self._get_oof_predictions(X, y, groups, 'median')
        oof_upper = self._get_oof_predictions(X, y, groups, 'upper')
        
        # Create meta-features
        meta_features = np.column_stack([oof_lower, oof_median, oof_upper])
        
        # Train meta-model (Random Forest)
        self.meta_model = RandomForestRegressor(**self.rf_params)
        self.meta_model.fit(meta_features, y, sample_weight=sample_weight)
    
    def predict(
        self,
        X: pd.DataFrame,
        quantile: str = 'median'
    ) -> np.ndarray:
        """
        Make predictions using the stacking ensemble.
        
        Args:
            X: Feature matrix
            quantile: Which quantile to predict ('lower', 'median', or 'upper')
            
        Returns:
            Array of predictions
        """
        if not self.meta_model:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get base model predictions
        base_preds = []
        for i in range(self.n_splits):
            lgb_model = self.base_models[f'lgb_{quantile}_{i}']
            rf_model = self.base_models[f'rf_{quantile}_{i}']
            
            lgb_pred = lgb_model.predict(X)
            rf_pred = rf_model.predict(X)
            
            base_preds.append(0.5 * (lgb_pred + rf_pred))
        
        # Average base model predictions
        meta_features = np.column_stack(base_preds)
        
        # Make final predictions using meta-model
        return self.meta_model.predict(meta_features) 