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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class GroupTimeSeriesSplit:
    """
    Time series cross-validator with group-based splitting.
    This ensures that data from the same group (e.g., store) stays together in splits.
    """
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(1/test_size))
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X: Feature matrix
            y: Target values (optional)
            groups: Group labels (e.g., store IDs)
            
        Yields:
            train_idx, test_idx: Indices for training and test sets
        """
        if groups is None:
            # If no groups provided, use regular TimeSeriesSplit
            yield from self.tscv.split(X, y)
            return
            
        # Get unique groups and their indices
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        # Calculate number of groups for test set
        n_test_groups = max(1, int(n_groups * self.test_size))
        
        # Generate splits
        for i in range(self.n_splits):
            # Calculate test group indices
            test_start = (i * n_test_groups) % n_groups
            test_end = (test_start + n_test_groups) % n_groups
            
            if test_end > test_start:
                test_groups = unique_groups[test_start:test_end]
            else:
                # Handle wrap-around case
                test_groups = np.concatenate([
                    unique_groups[test_start:],
                    unique_groups[:test_end]
                ])
            
            # Get indices for train and test sets
            test_mask = np.isin(groups, test_groups)
            train_mask = ~test_mask
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            yield train_idx, test_idx

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

def select_features_by_global_shap(
    X: pd.DataFrame,
    y: np.ndarray,
    store_ids: np.ndarray,
    n_features: int = 25,
    n_trees: int = 100,
    n_splits: int = 5
) -> List[str]:
    """
    Select features based on SHAP values calculated globally across all CV folds.
    
    Args:
        X: Feature matrix
        y: Target values
        store_ids: Array of store IDs for time series splitting
        n_features: Number of top features to select
        n_trees: Number of trees to use for SHAP calculation
        n_splits: Number of CV splits
        
    Returns:
        List of selected feature names
    """
    logging.info(f"Starting global SHAP-based feature selection with {n_splits} CV folds...")
    
    # Initialize dictionary to store SHAP values across folds
    all_shap_values = {feature: [] for feature in X.columns}
    
    # Initialize GroupTimeSeriesSplit
    gtscv = GroupTimeSeriesSplit(n_splits=n_splits, test_size=0.2)
    
    # Calculate SHAP values for each fold
    for fold, (train_idx, test_idx) in enumerate(gtscv.split(X, groups=store_ids)):
        logging.info(f"Processing fold {fold + 1}/{n_splits}...")
        
        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        
        # Train a quick Random Forest for SHAP calculation
        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=5,  # Shallow trees for faster computation
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_train)
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Store SHAP values for each feature
        for feature, importance in zip(X.columns, mean_abs_shap):
            all_shap_values[feature].append(importance)
    
    # Calculate mean and std of SHAP values across folds
    feature_stability = {}
    for feature, values in all_shap_values.items():
        mean_importance = np.mean(values)
        std_importance = np.std(values)
        # Calculate stability score (mean / std)
        stability_score = mean_importance / (std_importance + 1e-10)  # Add small epsilon to avoid division by zero
        feature_stability[feature] = {
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'stability_score': stability_score
        }
    
    # Sort features by stability score
    sorted_features = sorted(
        feature_stability.items(),
        key=lambda x: x[1]['stability_score'],
        reverse=True
    )
    
    # Remove unstable features (std > 0.5 * mean)
    stable_features = [f for f, stats in feature_stability.items() if stats['std_importance'] <= 0.5 * stats['mean_importance']]
    # Plot top 10 SHAP features with error bars
    top10 = sorted(feature_stability.items(), key=lambda x: x[1]['mean_importance'], reverse=True)[:10]
    features_plot = [f[0] for f in top10]
    means = [f[1]['mean_importance'] for f in top10]
    stds = [f[1]['std_importance'] for f in top10]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(features_plot)), means, yerr=stds, capsize=5)
    plt.xticks(range(len(features_plot)), features_plot, rotation=45, ha='right')
    plt.ylabel('Mean Absolute SHAP Value')
    plt.title('Top 10 SHAP Features (Mean ± Std across folds)')
    plt.tight_layout()
    Path('models').mkdir(exist_ok=True)
    plt.savefig('models/top10_shap_features.png')
    plt.close()
    # Select top N stable features
    selected_features = [f for f in stable_features if f in [f[0] for f in sorted_features[:n_features]]][:n_features]
    logging.info(f"Selected {len(selected_features)} stable features after removing unstable ones.")
    return selected_features

def select_features_by_shap(
    X: pd.DataFrame,
    y: np.ndarray,
    clusters: np.ndarray,
    n_features: int = 20,
    n_trees: int = 100
) -> Dict[str, List[str]]:
    """
    Select features based on SHAP values per cluster.
    This function is kept for backward compatibility but is deprecated.
    Use select_features_by_global_shap instead.
    
    Args:
        X: Feature matrix
        y: Target values
        clusters: Array of cluster assignments
        n_features: Number of top features to select per cluster
        n_trees: Number of trees to use for SHAP calculation
        
    Returns:
        Dictionary mapping cluster IDs to lists of selected feature names
    """
    logging.warning("This function is deprecated. Use select_features_by_global_shap instead.")
    
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

def remove_correlated_features(X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Remove highly correlated features.
    
    Args:
        X: Feature matrix
        threshold: Correlation threshold
        
    Returns:
        List of feature names to keep
    """
    logging.info("Calculating correlation matrix...")
    corr_matrix = X.corr().abs()
    
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        logging.info(f"Found {len(to_drop)} highly correlated features to remove")
        for feat in to_drop:
            corr_feats = upper[feat][upper[feat] > threshold].index.tolist()
            if corr_feats:
                logging.info(f"Feature '{feat}' is highly correlated with: {', '.join(corr_feats)}")
    
    # Return features to keep
    return [col for col in X.columns if col not in to_drop]

def remove_high_vif_features(X: pd.DataFrame, threshold: float = 5.0) -> List[str]:
    """
    Remove features with high Variance Inflation Factor (VIF).
    Handles one-hot encoded categorical variables by grouping them.
    
    Args:
        X: Feature matrix
        threshold: VIF threshold
        
    Returns:
        List of feature names to keep
    """
    logging.info("Calculating VIF values...")
    
    # Identify numeric and categorical columns
    numeric_cols = []
    categorical_cols = []
    categorical_groups = {}
    
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            # Check if this is a one-hot encoded column
            if '_' in col and any(col.startswith(prefix) for prefix in ['city_', 'region_', 'channel_']):
                prefix = col.split('_')[0]
                if prefix not in categorical_groups:
                    categorical_groups[prefix] = []
                categorical_groups[prefix].append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            logging.info(f"Skipping VIF calculation for categorical column: {col}")
    
    if not numeric_cols:
        logging.info("No numeric columns found for VIF calculation")
        return X.columns.tolist()
    
    # Process only numeric columns
    X_numeric = X[numeric_cols].copy()
    
    # Handle infinite values
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with column means
    X_numeric = X_numeric.fillna(X_numeric.mean())
    
    # Verify data is valid
    if not np.all(np.isfinite(X_numeric.values)):
        logging.error("Data still contains non-finite values after cleaning")
        return X.columns.tolist()
    
    try:
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_numeric.columns
        vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
        
        # Sort by VIF
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        # Log VIF values
        logging.info("\nTop 10 highest VIF values:")
        for _, row in vif_data.head(10).iterrows():
            logging.info(f"{row['Variable']}: {row['VIF']:.2f}")
        
        # Remove features with high VIF
        to_keep_numeric = vif_data[vif_data["VIF"] <= threshold]["Variable"].tolist()
        removed = len(X_numeric.columns) - len(to_keep_numeric)
        
        if removed > 0:
            logging.info(f"Removed {removed} numeric features with high VIF")
            removed_features = [col for col in X_numeric.columns if col not in to_keep_numeric]
            logging.info("Removed features:")
            for feat in removed_features:
                vif = vif_data[vif_data["Variable"] == feat]["VIF"].values[0]
                logging.info(f"- {feat}: VIF = {vif:.2f}")
        
        # Handle categorical groups
        to_keep_categorical = []
        for prefix, cols in categorical_groups.items():
            # Keep at least one column from each categorical group
            if any(col in to_keep_numeric for col in cols):
                to_keep_categorical.extend([col for col in cols if col in to_keep_numeric])
            else:
                # If none of the columns are kept, keep the most frequent one
                most_frequent = X[cols].sum().idxmax()
                to_keep_categorical.append(most_frequent)
                logging.info(f"Keeping most frequent category for {prefix}: {most_frequent}")
        
        # Combine kept numeric features with categorical features
        to_keep = to_keep_numeric + to_keep_categorical + categorical_cols
        return to_keep
        
    except Exception as e:
        logging.error(f"Error calculating VIF: {str(e)}")
        logging.info("Skipping VIF-based feature removal")
        return X.columns.tolist()

def group_correlated_features_pca(X: pd.DataFrame, correlation_threshold: float = 0.7) -> List[str]:
    """
    Group correlated features using PCA and correlation analysis.
    
    Args:
        X: Feature matrix
        correlation_threshold: Threshold for correlation grouping
        
    Returns:
        List of selected feature names
    """
    logging.info("Starting PCA-based feature grouping...")
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Find groups of correlated features
    feature_groups = {}
    processed_features = set()
    
    for feature in X.columns:
        if feature in processed_features:
            continue
            
        # Find correlated features
        correlated = corr_matrix[feature][corr_matrix[feature] > correlation_threshold].index.tolist()
        if len(correlated) > 1:  # Only create groups with multiple features
            group_id = f"group_{len(feature_groups)}"
            feature_groups[group_id] = correlated
            processed_features.update(correlated)
    
    # For each group, apply PCA and select representative features
    selected_features = []
    for group_id, features in feature_groups.items():
        if len(features) <= 2:  # Keep all features if group is small
            selected_features.extend(features)
            continue
            
        try:
            # Apply PCA to the feature group
            X_group = X[features].copy()
            
            # Handle any non-numeric columns
            numeric_cols = X_group.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:  # Need at least 2 numeric columns for PCA
                selected_features.extend(features)
                continue
                
            X_group = X_group[numeric_cols]
            
            # Fill any missing values
            X_group = X_group.fillna(X_group.mean())
            
            # Apply PCA
            pca = PCA(n_components=0.95)  # Keep 95% of variance
            X_pca = pca.fit_transform(X_group)
            
            # Calculate feature importance using explained variance ratio
            importance = np.abs(pca.components_) * pca.explained_variance_ratio_[:, np.newaxis]
            feature_importance = np.sum(importance, axis=0)
            
            # Select top 2 features based on importance
            top_indices = np.argsort(feature_importance)[-2:]
            selected_features.extend([numeric_cols[i] for i in top_indices])
            
            logging.info(f"Group {group_id}: Selected {len(top_indices)} features from {len(features)} correlated features")
            
        except Exception as e:
            logging.warning(f"Error processing group {group_id}: {str(e)}")
            # If PCA fails, keep all features from this group
            selected_features.extend(features)
    
    # Add ungrouped features
    ungrouped_features = set(X.columns) - processed_features
    selected_features.extend(list(ungrouped_features))
    
    # Remove any duplicates while preserving order
    selected_features = list(dict.fromkeys(selected_features))
    
    logging.info(f"Selected {len(selected_features)} features from {len(X.columns)} total features")
    return selected_features

def iterative_feature_pruning(X: pd.DataFrame, y: pd.Series, model_params: dict, n_iter: int = 10) -> Tuple[RandomForestRegressor, List[str], List[float]]:
    """
    Iteratively prune features using multiple methods.
    Skips VIF pruning for tree-based models since they can handle correlated features.
    
    Args:
        X: Feature matrix
        y: Target variable
        model_params: Parameters for RandomForestRegressor
        n_iter: Number of iterations
        
    Returns:
        Tuple of (best model, selected features, R² scores)
    """
    logging.info(f"\nStarting iterative feature pruning with {X.shape[1]} initial features...")
    
    # Ensure target variable is numeric
    if not np.issubdtype(y.dtype, np.number):
        logging.warning("Converting target variable to numeric")
        y = pd.to_numeric(y, errors='coerce')
    
    # Handle infinite values in target
    y = pd.Series(y).replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values in target with median
    y = y.fillna(y.median())
    
    best_model = None
    best_score = -np.inf
    best_features = None
    r2_scores = []
    
    # Get initial feature set
    current_features = X.columns.tolist()
    
    for i in range(n_iter):
        logging.info(f"\nIteration {i+1}/{n_iter}")
        logging.info(f"Current feature count: {len(current_features)}")
        
        try:
            # 1. Group correlated features using PCA
            logging.info("Starting PCA-based feature grouping...")
            X_current = X[current_features]
            current_features = group_correlated_features_pca(X_current, correlation_threshold=0.7)
            logging.info(f"Features after PCA grouping: {len(current_features)}")
            
            # 2. Skip VIF pruning for tree-based models
            logging.info("Skipping VIF pruning for tree-based model...")
            
            # 3. Train model and evaluate
            logging.info("Training model with current feature set...")
            model = RandomForestRegressor(**model_params)
            model.fit(X[current_features], y)
            
            # Calculate R² score
            y_pred = model.predict(X[current_features])
            score = np.corrcoef(y, y_pred)[0,1]**2
            r2_scores.append(score)
            logging.info(f"R² score: {score:.4f}")
            
            # Update best model if current score is better
            if score > best_score:
                logging.info("New best model found!")
                best_score = score
                best_model = model
                best_features = current_features.copy()
            
            # 4. Remove least important features
            if len(current_features) > 10:  # Keep at least 10 features
                importances = dict(zip(current_features, model.feature_importances_))
                sorted_features = sorted(importances.items(), key=lambda x: x[1])
                n_to_remove = max(1, len(current_features) // 10)  # Remove 10% of features each iteration
                features_to_remove = [f[0] for f in sorted_features[:n_to_remove]]
                current_features = [f for f in current_features if f not in features_to_remove]
                logging.info(f"Removed {n_to_remove} least important features")
                logging.info(f"Remaining features: {len(current_features)}")
            else:
                logging.info("Reached minimum feature threshold, stopping feature removal")
                break
                
        except Exception as e:
            logging.error(f"Error in iteration {i+1}: {str(e)}")
            logging.info("Continuing with current feature set")
            continue
    
    if best_model is None:
        logging.error("No valid model was found during feature pruning")
        return None, X.columns.tolist(), []
    
    logging.info("\nFeature pruning complete")
    logging.info(f"Final feature count: {len(best_features)}")
    logging.info(f"Best R² score: {best_score:.4f}")
    
    return best_model, best_features, r2_scores 