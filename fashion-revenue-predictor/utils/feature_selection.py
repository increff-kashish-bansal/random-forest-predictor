import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Dict, Tuple
import logging
from sklearn.linear_model import LinearRegression
import shap
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator

def remove_near_zero_variance(X: pd.DataFrame, threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with near-zero variance.
    
    Args:
        X: Feature DataFrame
        threshold: Variance threshold (features with variance below this will be removed)
        
    Returns:
        Tuple of (filtered DataFrame, list of removed feature names)
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    
    # Get features to keep
    kept_features = X.columns[selector.get_support()].tolist()
    removed_features = list(set(X.columns) - set(kept_features))
    
    if removed_features:
        logging.info(f"Removed {len(removed_features)} near-zero variance features")
        logging.debug(f"Removed features: {removed_features}")
    
    return X[kept_features], removed_features

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VIF for each feature using a more robust method.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        DataFrame with VIF values
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    
    # Calculate VIF for each feature
    vif_values = []
    for i in range(X.shape[1]):
        # Get the feature to predict
        y = X.iloc[:, i]
        # Get other features
        X_other = X.drop(X.columns[i], axis=1)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X_other, y)
        
        # Calculate R-squared
        r_squared = model.score(X_other, y)
        
        # Calculate VIF
        if r_squared == 1.0:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - r_squared)
        
        vif_values.append(vif)
    
    vif_data["VIF"] = vif_values
    return vif_data

def remove_high_vif(X: pd.DataFrame, threshold: float = 5.0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with high Variance Inflation Factor (VIF).
    
    Args:
        X: Feature DataFrame
        threshold: VIF threshold (features with VIF above this will be removed)
        
    Returns:
        Tuple of (filtered DataFrame, list of removed feature names)
    """
    # Handle infinite and NaN values
    X_clean = X.copy()
    
    # Replace inf with NaN
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with column means
    X_clean = X_clean.fillna(X_clean.mean())
    
    # Calculate VIF for each feature
    vif_data = calculate_vif(X_clean)
    
    # Remove features with high VIF iteratively
    removed_features = []
    while True:
        max_vif = vif_data["VIF"].max()
        if max_vif <= threshold:
            break
            
        # Remove feature with highest VIF
        feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        removed_features.append(feature_to_remove)
        X_clean = X_clean.drop(columns=[feature_to_remove])
        
        # Recalculate VIF
        vif_data = calculate_vif(X_clean)
    
    if removed_features:
        logging.info(f"Removed {len(removed_features)} features with high VIF")
        logging.debug(f"Removed features: {removed_features}")
    
    # Return the original DataFrame with the same columns as X_clean
    return X[X_clean.columns], removed_features

def remove_low_importance(X: pd.DataFrame, feature_importances: Dict[str, float], 
                         importance_threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features with low importance scores.
    
    Args:
        X: Feature DataFrame
        feature_importances: Dictionary of feature importance scores
        importance_threshold: Threshold for feature importance
        
    Returns:
        Tuple of (filtered DataFrame, list of removed feature names)
    """
    # Get features to keep
    kept_features = [f for f, imp in feature_importances.items() if imp >= importance_threshold]
    removed_features = list(set(X.columns) - set(kept_features))
    
    if removed_features:
        logging.info(f"Removed {len(removed_features)} low importance features")
        logging.debug(f"Removed features: {removed_features}")
    
    return X[kept_features], removed_features

def prune_features(X: pd.DataFrame, feature_importances: Dict[str, float] = None,
                  nzv_threshold: float = 0.01, vif_threshold: float = 5.0,
                  importance_threshold: float = 0.01) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Apply all feature pruning steps in sequence.
    
    Args:
        X: Feature DataFrame
        feature_importances: Dictionary of feature importance scores (optional)
        nzv_threshold: Threshold for near-zero variance
        vif_threshold: Threshold for VIF
        importance_threshold: Threshold for feature importance
        
    Returns:
        Tuple of (filtered DataFrame, dictionary of removed features by step)
    """
    removed_features = {}
    
    # 1. Remove near-zero variance features
    X, removed_features['nzv'] = remove_near_zero_variance(X, nzv_threshold)
    
    # 2. Remove high VIF features
    X, removed_features['vif'] = remove_high_vif(X, vif_threshold)
    
    # 3. Remove low importance features (if importances provided)
    if feature_importances is not None:
        # Filter feature_importances to only include features that are still present
        filtered_importances = {k: v for k, v in feature_importances.items() if k in X.columns}
        X, removed_features['importance'] = remove_low_importance(X, filtered_importances, importance_threshold)
    
    # Log summary
    total_removed = sum(len(features) for features in removed_features.values())
    logging.info(f"Total features removed: {total_removed}")
    logging.info(f"Remaining features: {len(X.columns)}")
    
    return X, removed_features

def select_features_by_shap(
    X: pd.DataFrame,
    y: np.ndarray,
    clusters: np.ndarray,
    n_features: int = 20,
    n_trees: int = 100
) -> Dict[str, List[str]]:
    """
    Select features based on SHAP values per cluster.
    
    Args:
        X: Feature matrix
        y: Target values
        clusters: Array of cluster assignments
        n_features: Number of top features to select per cluster
        n_trees: Number of trees to use for SHAP calculation
        
    Returns:
        Dictionary mapping cluster IDs to lists of selected feature names
    """
    logging.info(f"Starting SHAP-based feature selection for {len(np.unique(clusters))} clusters...")
    
    # Initialize dictionary to store selected features per cluster
    selected_features = {}
    
    # Calculate SHAP values for each cluster
    for cluster in np.unique(clusters):
        logging.info(f"Processing cluster {cluster}...")
        
        # Get data for this cluster
        cluster_mask = clusters == cluster
        X_cluster = X[cluster_mask]
        y_cluster = y[cluster_mask]
        
        if len(X_cluster) < 10:  # Skip if too few samples
            logging.warning(f"Cluster {cluster} has too few samples ({len(X_cluster)}), using all features")
            selected_features[cluster] = X.columns.tolist()
            continue
        
        # Train a quick Random Forest for SHAP calculation
        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=5,  # Shallow trees for faster computation
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_cluster, y_cluster)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_cluster)
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Get feature importance scores
        feature_importance = dict(zip(X.columns, mean_abs_shap))
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top N features
        top_features = [f[0] for f in sorted_features[:n_features]]
        selected_features[cluster] = top_features
        
        logging.info(f"Selected {len(top_features)} features for cluster {cluster}")
        logging.info(f"Top 5 features: {top_features[:5]}")
    
    return selected_features

def get_common_features(
    selected_features: Dict[str, List[str]],
    min_clusters: int = 2
) -> List[str]:
    """
    Get features that are selected across multiple clusters.
    
    Args:
        selected_features: Dictionary mapping cluster IDs to selected feature lists
        min_clusters: Minimum number of clusters a feature must appear in
        
    Returns:
        List of common feature names
    """
    # Count how many clusters each feature appears in
    feature_counts = {}
    for features in selected_features.values():
        for feature in features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # Select features that appear in at least min_clusters
    common_features = [
        feature for feature, count in feature_counts.items()
        if count >= min_clusters
    ]
    
    logging.info(f"Found {len(common_features)} common features across clusters")
    return common_features

def remove_correlated_features(X: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features to reduce redundancy.
    
    Args:
        X: Feature DataFrame
        threshold: Correlation threshold (features with correlation above this will be considered redundant)
        
    Returns:
        Tuple of (filtered DataFrame, list of removed feature names)
    """
    logging.info(f"Starting correlation analysis with threshold {threshold}...")
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation above threshold
    to_drop = []
    for column in upper.columns:
        if any(upper[column] > threshold):
            # Find the feature with the highest correlation
            correlated_features = upper[column][upper[column] > threshold].index.tolist()
            if correlated_features:
                # Keep the feature with the highest variance
                variances = X[correlated_features + [column]].var()
                feature_to_keep = variances.idxmax()
                features_to_drop = [f for f in correlated_features + [column] if f != feature_to_keep]
                to_drop.extend(features_to_drop)
    
    # Remove duplicates while preserving order
    to_drop = list(dict.fromkeys(to_drop))
    
    if to_drop:
        logging.info(f"Removing {len(to_drop)} highly correlated features")
        logging.info(f"Removed features: {to_drop}")
        
        # Create correlation report for removed features
        corr_report = []
        for feature in to_drop:
            # Find the feature it was most correlated with
            correlations = corr_matrix[feature].sort_values(ascending=False)
            most_correlated = correlations.index[1]  # Skip self-correlation
            corr_value = correlations.iloc[1]
            corr_report.append(f"{feature} (corr={corr_value:.3f} with {most_correlated})")
        
        logging.info("Correlation report for removed features:")
        for report in corr_report:
            logging.info(f"- {report}")
    
    # Return the original DataFrame with the same columns as X_clean
    return X.drop(columns=to_drop), to_drop

def iterative_feature_pruning(X: pd.DataFrame, y: np.ndarray, model_params: Dict, n_iter: int = 5) -> Tuple[RandomForestRegressor, List[str], List[float]]:
    """
    Iteratively prune features based on importance and correlation.
    
    Args:
        X: Feature matrix
        y: Target values
        model_params: Parameters for RandomForestRegressor
        n_iter: Number of pruning iterations
        
    Returns:
        Tuple of (best model, selected features, R² scores)
    """
    logging.info("Starting iterative feature pruning...")
    
    # Initialize variables
    current_features = X.columns.tolist()
    r2_scores = []
    best_r2 = -np.inf
    best_model = None
    best_features = None
    
    for i in range(n_iter):
        logging.info(f"\nIteration {i+1}/{n_iter}")
        logging.info(f"Current feature count: {len(current_features)}")
        
        # 1. Remove highly correlated features
        X_uncorr, removed_corr = remove_correlated_features(X[current_features], threshold=0.95)
        current_features = X_uncorr.columns.tolist()
        logging.info(f"Features after correlation pruning: {len(current_features)}")
        
        # 2. Remove features with high VIF
        X_low_vif, removed_vif = remove_high_vif(X_uncorr, threshold=5.0)
        current_features = X_low_vif.columns.tolist()
        logging.info(f"Features after VIF pruning: {len(current_features)}")
        
        # 3. Train model and get feature importance
        model = RandomForestRegressor(**model_params)
        model.fit(X_low_vif, y)
        
        # Calculate R² score
        r2 = model.score(X_low_vif, y)
        r2_scores.append(r2)
        logging.info(f"R² score: {r2:.4f}")
        
        # Update best model if current is better
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_features = current_features.copy()
            logging.info("New best model found!")
        
        # 4. Remove least important features
        importances = pd.Series(model.feature_importances_, index=current_features)
        importances = importances.sort_values(ascending=False)
        
        # Remove bottom 10% of features
        n_to_remove = max(1, int(len(current_features) * 0.1))
        removed_features = importances.tail(n_to_remove).index.tolist()
        current_features = importances.head(len(current_features) - n_to_remove).index.tolist()
        
        logging.info(f"Removed {n_to_remove} least important features")
        logging.info(f"Remaining features: {len(current_features)}")
        
        # Stop if we've removed too many features
        if len(current_features) < 10:
            logging.info("Stopping early due to too few features remaining")
            break
    
    logging.info("\nFeature pruning complete")
    logging.info(f"Final feature count: {len(best_features)}")
    logging.info(f"Best R² score: {best_r2:.4f}")
    
    return best_model, best_features, r2_scores 