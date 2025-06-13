import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    
    # 3. Add store tier based on store area
    if not is_prediction:
        # During training, create bins using simple categorization
        min_area = df['store_area'].min()
        max_area = df['store_area'].max()
        bin_size = (max_area - min_area) / 3
        
        def get_tier(area):
            if area <= min_area + bin_size:
                return 'TIER_3'
            elif area <= min_area + 2 * bin_size:
                return 'TIER_2'
            else:
                return 'TIER_1'
        
        df['store_tier'] = df['store_area'].apply(get_tier)
        
        # Save bin edges for prediction
        bin_edges = [float(min_area), float(min_area + bin_size), float(min_area + 2 * bin_size), float(max_area)]
        Path('models').mkdir(exist_ok=True)
        with open('models/store_tier_bins.json', 'w') as f:
            json.dump(bin_edges, f)
    else:
        # During prediction, use simple categorization
        try:
            with open('models/store_tier_bins.json', 'r') as f:
                bin_edges = json.load(f)
            
            def get_tier(area):
                if area <= bin_edges[1]:
                    return 'TIER_3'
                elif area <= bin_edges[2]:
                    return 'TIER_2'
                else:
                    return 'TIER_1'
            
            df['store_tier'] = df['store_area'].apply(get_tier)
        except Exception:
            # If binning fails, assign all to TIER_2
            df['store_tier'] = 'TIER_2'
    
    # 4. Create store clusters using multiple features
    if not is_prediction:
        # During training, perform clustering
        
        # Calculate store-month seasonal features
        store_month_stats = df.groupby(['store', 'month']).agg({
            'revenue': ['mean', 'std']
        }).fillna(0)
        store_month_stats.columns = ['store_month_revenue_mean', 'store_month_revenue_std']
        
        # Calculate store-level seasonal features
        store_seasonal_features = store_month_stats.groupby('store').agg({
            'store_month_revenue_mean': ['mean', 'std'],
            'store_month_revenue_std': ['mean', 'std']
        }).fillna(0)
        store_seasonal_features.columns = [
            'store_month_revenue_mean_avg',
            'store_month_revenue_mean_std',
            'store_month_revenue_std_avg',
            'store_month_revenue_std_std'
        ]
        
        # Calculate holiday revenue ratio (assuming holidays are in December)
        holiday_revenue = df[df['month'] == 12].groupby('store')['revenue'].mean()
        non_holiday_revenue = df[df['month'] != 12].groupby('store')['revenue'].mean()
        holiday_ratio = (holiday_revenue / non_holiday_revenue).fillna(1)
        holiday_ratio.name = 'holiday_revenue_ratio'
        
        # Create region/city one-hot averages
        region_avg = df.groupby('region_classified')['revenue'].mean()
        city_avg = df.groupby('city')['revenue'].mean()
        
        # Normalize region and city averages
        region_avg = (region_avg - region_avg.mean()) / region_avg.std()
        city_avg = (city_avg - city_avg.mean()) / city_avg.std()
        
        # Map averages back to stores
        df['region_revenue_avg'] = df['region_classified'].map(region_avg)
        df['city_revenue_avg'] = df['city'].map(city_avg)
        
        # Combine all features for clustering
        cluster_features = [
            'store_area',
            'is_online',
            'region_revenue_avg',
            'city_revenue_avg'
        ]
        
        # Add seasonal features
        df = df.merge(store_seasonal_features, on='store', how='left')
        df = df.merge(holiday_ratio, on='store', how='left')
        cluster_features.extend([
            'store_month_revenue_mean_avg',
            'store_month_revenue_mean_std',
            'store_month_revenue_std_avg',
            'store_month_revenue_std_std',
            'holiday_revenue_ratio'
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
        
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        df['store_cluster'] = kmeans.fit_predict(X_cluster_pca)
        
        # Save cluster information for prediction
        cluster_info = {
            'cluster_centers': [center.tolist() for center in kmeans.cluster_centers_],
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
        # During prediction, use nearest cluster
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
            
            # Find nearest cluster center
            centers = np.array(cluster_info['cluster_centers'])
            # Reshape X_cluster to 2D if it's 1D
            if len(X_cluster_pca.shape) == 1:
                X_cluster_pca = X_cluster_pca.reshape(1, -1)
            # Calculate distances to each center
            distances = np.array([np.linalg.norm(X_cluster_pca - center, axis=1) for center in centers])
            df['store_cluster'] = np.argmin(distances, axis=0)
        except FileNotFoundError:
            # If no cluster info available, use simple categorization
            df['store_cluster'] = pd.cut(
                df['store_area'],
                bins=3,
                labels=['SMALL_OFFLINE', 'MEDIUM_OFFLINE', 'LARGE_OFFLINE'],
                include_lowest=True
            )
    
    # Map cluster numbers to meaningful labels
    cluster_labels = {
        0: 'SMALL_OFFLINE',
        1: 'MEDIUM_OFFLINE',
        2: 'LARGE_OFFLINE',
        3: 'ONLINE',
        4: 'PREMIUM_OFFLINE'  # Added for potential additional clusters
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
    Find optimal number of clusters using elbow method and silhouette score.
    
    Args:
        X: Feature matrix
        max_clusters: Maximum number of clusters to try
        
    Returns:
        Optimal number of clusters
    """
    # Calculate inertia and silhouette scores for different k values
    inertias = []
    silhouette_scores = []
    k_values = range(2, min(max_clusters + 1, len(X)))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score
        if len(X) > k:  # Silhouette score requires more samples than clusters
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(X, labels))
    
    # Find elbow point using second derivative
    if len(inertias) > 2:
        # Calculate second derivative of inertia
        second_derivative = np.diff(np.diff(inertias))
        elbow_k = np.argmax(second_derivative) + 2  # +2 because we lost 2 points in diff
        
        # Find best silhouette score
        best_silhouette_k = k_values[np.argmax(silhouette_scores)]
        
        # Use the average of both methods, rounded to nearest integer
        optimal_k = int(np.round((elbow_k + best_silhouette_k) / 2))
        
        # Ensure optimal_k is within reasonable bounds
        optimal_k = max(2, min(optimal_k, max_clusters))
    else:
        optimal_k = 2
    
    return optimal_k 