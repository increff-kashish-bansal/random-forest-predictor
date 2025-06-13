import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import requests
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def add_gaussian_noise(X: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """
    Add Gaussian noise to numeric features for data augmentation.
    
    Args:
        X: Feature matrix
        noise_level: Standard deviation of noise as a fraction of feature std
        
    Returns:
        Feature matrix with added noise
    """
    # Calculate feature-wise standard deviations
    feature_stds = np.std(X, axis=0)
    
    # Generate noise matrix
    noise = np.random.normal(0, noise_level * feature_stds, X.shape)
    
    # Add noise to features
    X_noisy = X + noise
    
    return X_noisy

def enrich_store_data(df_stores: pd.DataFrame, is_prediction: bool = False) -> pd.DataFrame:
    """
    Enrich store data with additional geographical information and clusters.
    
    Args:
        df_stores: DataFrame containing store information
        is_prediction: Whether this is for prediction (True) or training (False)
        
    Returns:
        Enriched DataFrame with additional features
    """
    logging.info("Starting store data enrichment...")
    
    # Create a copy to avoid modifying original
    df = df_stores.copy()
    
    # Ensure store_id exists
    if 'store_id' not in df.columns and 'id' in df.columns:
        df = df.rename(columns={'id': 'store_id'})
    
    # Add temporal interaction features
    # 1. Festival season interaction with discount
    festival_months = [10, 11, 12, 1, 2]  # October to February (Diwali, Christmas, New Year)
    df['is_festival_season'] = df['month'].isin(festival_months).astype(int)
    df['festival_discount_interaction'] = df['is_festival_season'] * df['discount_pct']
    
    # 2. Store cluster interaction with month
    # Create one-hot encoded month features
    month_dummies = pd.get_dummies(df['month'], prefix='month')
    df = pd.concat([df, month_dummies], axis=1)
    
    # Get store cluster dummies
    cluster_dummies = pd.get_dummies(df['store_cluster'], prefix='cluster')
    df = pd.concat([df, cluster_dummies], axis=1)
    
    # Create cluster-month interactions
    for cluster_col in cluster_dummies.columns:
        for month_col in month_dummies.columns:
            interaction_name = f"{cluster_col}_{month_col}_interaction"
            df[interaction_name] = df[cluster_col] * df[month_col]
    
    # 3. Weekend interaction with discount
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)  # Assuming 5,6 are weekend days
    df['weekend_discount_interaction'] = df['is_weekend'] * df['discount_pct']
    
    # 1. Add state information based on city
    city_to_state = {
        'MUMBAI': 'MAHARASHTRA',
        'DELHI': 'DELHI',
        'BANGALORE': 'KARNATAKA',
        'HYDERABAD': 'TELANGANA',
        'CHENNAI': 'TAMIL NADU',
        'KOLKATA': 'WEST BENGAL',
        'PUNE': 'MAHARASHTRA',
        'AHMEDABAD': 'GUJARAT',
        'JAIPUR': 'RAJASTHAN',
        'LUCKNOW': 'UTTAR PRADESH'
    }
    
    df['state'] = df['city'].map(city_to_state)
    
    # 2. Add region classification based on state
    state_to_region = {
        'MAHARASHTRA': 'WEST',
        'DELHI': 'NORTH',
        'KARNATAKA': 'SOUTH',
        'TELANGANA': 'SOUTH',
        'TAMIL NADU': 'SOUTH',
        'WEST BENGAL': 'EAST',
        'GUJARAT': 'WEST',
        'RAJASTHAN': 'NORTH',
        'UTTAR PRADESH': 'NORTH'
    }
    
    df['region_classified'] = df['state'].map(state_to_region)
    
    # 3. Add store tier based on store area with more granular bucketing
    if not is_prediction:
        # During training, create more granular bins using KBinsDiscretizer
        kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
        df['store_area_bucket'] = kbd.fit_transform(df[['store_area']]).astype(int)
        
        # Save bin edges for prediction
        bin_edges = kbd.bin_edges_[0].tolist()
        Path('models').mkdir(exist_ok=True)
        with open('models/store_area_bins.json', 'w') as f:
            json.dump(bin_edges, f)
    else:
        # During prediction, use saved bin edges
        try:
            with open('models/store_area_bins.json', 'r') as f:
                bin_edges = json.load(f)
            
            df['store_area_bucket'] = pd.cut(
                df['store_area'],
                bins=bin_edges,
                labels=False,
                include_lowest=True
            ).fillna(0).astype(int)
        except Exception:
            df['store_area_bucket'] = 2  # Default to middle bucket if binning fails
    
    # 4. Create store clusters using GMM for soft clustering
    if not is_prediction:
        # During training, perform clustering
        
        # Calculate store-month seasonal features with more granular patterns
        store_month_stats = df.groupby(['store', 'month']).agg({
            'revenue': ['mean', 'std', 'min', 'max'],
            'quantity': ['mean', 'std'],
            'discount': ['mean', 'std']
        }).fillna(0)
        store_month_stats.columns = [
            'store_month_revenue_mean', 'store_month_revenue_std',
            'store_month_revenue_min', 'store_month_revenue_max',
            'store_month_quantity_mean', 'store_month_quantity_std',
            'store_month_discount_mean', 'store_month_discount_std'
        ]
        
        # Calculate store-level seasonal features with enhanced metrics
        store_seasonal_features = store_month_stats.groupby('store').agg({
            'store_month_revenue_mean': ['mean', 'std', 'max'],
            'store_month_revenue_std': ['mean', 'std'],
            'store_month_quantity_mean': ['mean', 'std'],
            'store_month_discount_mean': ['mean', 'std']
        }).fillna(0)
        store_seasonal_features.columns = [
            'store_month_revenue_mean_avg', 'store_month_revenue_mean_std',
            'store_month_revenue_mean_max',
            'store_month_revenue_std_avg', 'store_month_revenue_std_std',
            'store_month_quantity_mean_avg', 'store_month_quantity_mean_std',
            'store_month_discount_mean_avg', 'store_month_discount_mean_std'
        ]
        
        # Calculate holiday and seasonal patterns
        holiday_months = [12, 1, 2]  # December, January, February
        summer_months = [4, 5, 6]    # April, May, June
        
        holiday_revenue = df[df['month'].isin(holiday_months)].groupby('store')['revenue'].mean()
        summer_revenue = df[df['month'].isin(summer_months)].groupby('store')['revenue'].mean()
        regular_revenue = df[~df['month'].isin(holiday_months + summer_months)].groupby('store')['revenue'].mean()
        
        holiday_ratio = (holiday_revenue / regular_revenue).fillna(1)
        summer_ratio = (summer_revenue / regular_revenue).fillna(1)
        
        holiday_ratio.name = 'holiday_revenue_ratio'
        summer_ratio.name = 'summer_revenue_ratio'
        
        # Create enhanced location features
        region_avg = df.groupby('region_classified').agg({
            'revenue': ['mean', 'std'],
            'quantity': 'mean',
            'discount': 'mean'
        }).fillna(0)
        region_avg.columns = ['region_revenue_avg', 'region_revenue_std', 
                            'region_quantity_avg', 'region_discount_avg']
        
        city_avg = df.groupby('city').agg({
            'revenue': ['mean', 'std'],
            'quantity': 'mean',
            'discount': 'mean'
        }).fillna(0)
        city_avg.columns = ['city_revenue_avg', 'city_revenue_std',
                          'city_quantity_avg', 'city_discount_avg']
        
        # Normalize location features
        for col in region_avg.columns:
            region_avg[col] = (region_avg[col] - region_avg[col].mean()) / region_avg[col].std()
            city_avg[col] = (city_avg[col] - city_avg[col].mean()) / city_avg[col].std()
        
        # Map averages back to stores
        for col in region_avg.columns:
            df[f'region_{col}'] = df['region_classified'].map(region_avg[col])
            df[f'city_{col}'] = df['city'].map(city_avg[col])
        
        # Combine all features for clustering
        cluster_features = [
            'store_area',
            'store_area_bucket',
            'is_online',
            'region_revenue_avg', 'region_revenue_std',
            'region_quantity_avg', 'region_discount_avg',
            'city_revenue_avg', 'city_revenue_std',
            'city_quantity_avg', 'city_discount_avg'
        ]
        
        # Add seasonal features
        df = df.merge(store_seasonal_features, on='store', how='left')
        df = df.merge(holiday_ratio, on='store', how='left')
        df = df.merge(summer_ratio, on='store', how='left')
        
        cluster_features.extend([
            'store_month_revenue_mean_avg', 'store_month_revenue_mean_std',
            'store_month_revenue_mean_max',
            'store_month_revenue_std_avg', 'store_month_revenue_std_std',
            'store_month_quantity_mean_avg', 'store_month_quantity_mean_std',
            'store_month_discount_mean_avg', 'store_month_discount_mean_std',
            'holiday_revenue_ratio', 'summer_revenue_ratio'
        ])
        
        # Scale features
        scaler = StandardScaler()
        X_cluster = scaler.fit_transform(df[cluster_features])
        
        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        X_cluster_pca = pca.fit_transform(X_cluster)
        
        # Find optimal number of clusters
        optimal_clusters = find_optimal_clusters(X_cluster_pca)
        logging.info(f"Optimal number of clusters determined: {optimal_clusters}")
        
        # Use GMM instead of KMeans for soft clustering
        gmm = GaussianMixture(
            n_components=optimal_clusters,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        
        # Fit GMM and get cluster probabilities
        cluster_probs = gmm.fit_predict_proba(X_cluster_pca)
        
        # Add cluster probabilities as features
        for i in range(optimal_clusters):
            df[f'cluster_{i}_prob'] = cluster_probs[:, i]
        
        # Get the most likely cluster for each store
        df['store_cluster'] = np.argmax(cluster_probs, axis=1)
        
        # Add noise to numeric features for data augmentation
        X_cluster_noisy = add_gaussian_noise(X_cluster, noise_level=0.05)
        X_cluster_pca_noisy = pca.transform(X_cluster_noisy)
        cluster_probs_noisy = gmm.predict_proba(X_cluster_pca_noisy)
        
        # Add noisy cluster probabilities
        for i in range(optimal_clusters):
            df[f'cluster_{i}_prob_noisy'] = cluster_probs_noisy[:, i]
        
        # Save cluster information for prediction
        cluster_info = {
            'gmm_means': gmm.means_.tolist(),
            'gmm_covariances': [cov.tolist() for cov in gmm.covariances_],
            'gmm_weights': gmm.weights_.tolist(),
            'feature_means': [float(x) for x in scaler.mean_],
            'feature_scales': [float(x) for x in scaler.scale_],
            'pca_components': pca.components_.tolist(),
            'pca_mean': pca.mean_.tolist(),
            'cluster_features': cluster_features,
            'n_clusters': optimal_clusters
        }
        with open('models/store_clusters.json', 'w') as f:
            json.dump(cluster_info, f)
    else:
        # During prediction, use GMM probabilities
        try:
            with open('models/store_clusters.json', 'r') as f:
                cluster_info = json.load(f)
            
            # Scale features
            cluster_features = cluster_info['cluster_features']
            X_cluster = (df[cluster_features] - cluster_info['feature_means']) / cluster_info['feature_scales']
            
            # Apply PCA transformation
            pca_components = np.array(cluster_info['pca_components'])
            pca_mean = np.array(cluster_info['pca_mean'])
            X_cluster_pca = np.dot(X_cluster - pca_mean, pca_components.T)
            
            # Calculate GMM probabilities
            gmm_means = np.array(cluster_info['gmm_means'])
            gmm_covariances = np.array(cluster_info['gmm_covariances'])
            gmm_weights = np.array(cluster_info['gmm_weights'])
            
            # Calculate cluster probabilities
            cluster_probs = np.zeros((len(X_cluster_pca), len(gmm_means)))
            for i in range(len(gmm_means)):
                # Calculate multivariate normal probability
                diff = X_cluster_pca - gmm_means[i]
                inv_cov = np.linalg.inv(gmm_covariances[i])
                exponent = -0.5 * np.sum(diff.dot(inv_cov) * diff, axis=1)
                cluster_probs[:, i] = gmm_weights[i] * np.exp(exponent)
            
            # Normalize probabilities
            cluster_probs = cluster_probs / cluster_probs.sum(axis=1, keepdims=True)
            
            # Add cluster probabilities as features
            for i in range(len(gmm_means)):
                df[f'cluster_{i}_prob'] = cluster_probs[:, i]
            
            # Get the most likely cluster
            df['store_cluster'] = np.argmax(cluster_probs, axis=1)
            
        except FileNotFoundError:
            # If no cluster info available, use simple categorization
            df['store_cluster'] = pd.cut(
                df['store_area'],
                bins=3,
                labels=['SMALL_OFFLINE', 'MEDIUM_OFFLINE', 'LARGE_OFFLINE'],
                include_lowest=True
            )
    
    # Map cluster numbers to meaningful labels with more granular categories
    cluster_labels = {
        0: 'SMALL_OFFLINE',
        1: 'MEDIUM_OFFLINE',
        2: 'LARGE_OFFLINE',
        3: 'ONLINE',
        4: 'PREMIUM_OFFLINE',
        5: 'SEASONAL_SPECIALIST',
        6: 'HOLIDAY_SPECIALIST',
        7: 'REGIONAL_LEADER'
    }
    # Only use the labels that correspond to actual clusters
    actual_labels = {i: cluster_labels[i] for i in range(optimal_clusters)}
    df['store_cluster'] = df['store_cluster'].map(actual_labels)
    
    # 5. Add store density features
    # Calculate stores per city
    city_store_counts = df['city'].value_counts().astype(int)
    df['city_store_density'] = df['city'].map(city_store_counts)
    
    # Calculate stores per state
    state_store_counts = df['state'].value_counts().astype(int)
    df['state_store_density'] = df['state'].map(state_store_counts)
    
    # 6. Create one-hot encoded features
    categorical_features = ['city', 'state', 'region_classified', 'store_cluster', 'store_tier']
    for feature in categorical_features:
        dummies = pd.get_dummies(df[feature], prefix=feature)
        df = pd.concat([df, dummies], axis=1)
    
    logging.info("Store data enrichment completed")
    return df

def get_store_features(df_stores: pd.DataFrame) -> List[str]:
    """
    Get list of store-related features for model training.
    
    Args:
        df_stores: DataFrame containing store information
        
    Returns:
        List of feature names
    """
    features = [
        'store_area',
        'is_online',
        'city_store_density',
        'state_store_density'
    ]
    
    # Add one-hot encoded features
    categorical_features = ['city', 'state', 'region_classified', 'store_cluster', 'store_tier']
    for feature in categorical_features:
        prefix = f"{feature}_"
        feature_cols = [col for col in df_stores.columns if col.startswith(prefix)]
        features.extend(feature_cols)
    
    return features

def find_optimal_clusters(X: np.ndarray, max_clusters: int = 10) -> int:
    """
    Find optimal number of clusters using BIC score for GMM.
    
    Args:
        X: Feature matrix
        max_clusters: Maximum number of clusters to try
        
    Returns:
        Optimal number of clusters
    """
    # Calculate BIC scores for different k values
    bic_scores = []
    k_values = range(2, min(max_clusters + 1, len(X)))
    
    for k in k_values:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
    
    # Find the elbow point in BIC scores
    if len(bic_scores) > 2:
        # Calculate second derivative of BIC scores
        second_derivative = np.diff(np.diff(bic_scores))
        optimal_k = np.argmax(second_derivative) + 2  # +2 because we lost 2 points in diff
        
        # Ensure optimal_k is within reasonable bounds
        optimal_k = max(2, min(optimal_k, max_clusters))
    else:
        optimal_k = 2
    
    return optimal_k 