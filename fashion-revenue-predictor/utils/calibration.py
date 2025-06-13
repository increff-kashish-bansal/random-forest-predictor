import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.isotonic import IsotonicRegression
import logging
from scipy.stats import norm
import joblib

class ClusterCalibrator:
    """
    Calibrates prediction intervals using isotonic regression per cluster.
    Uses historical prediction residuals to learn the mapping from raw to calibrated intervals.
    """
    
    def __init__(self, n_bins: int = 100):
        """
        Initialize the calibrator.
        
        Args:
            n_bins: Number of bins for isotonic regression
        """
        self.n_bins = n_bins
        self.calibrators = {}  # Dictionary to store calibrators per cluster
        self.cluster_stats = {}  # Store cluster-specific statistics
        
    def _calculate_raw_intervals(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate raw prediction intervals using normal distribution.
        
        Args:
            predictions: Point predictions (p50)
            uncertainties: Uncertainty estimates
            
        Returns:
            Tuple of (p10, p50, p90) arrays
        """
        p50 = predictions
        z_score = norm.ppf(0.9)  # 1.28 for 90% interval
        p10 = p50 - z_score * uncertainties
        p90 = p50 + z_score * uncertainties
        
        return p10, p50, p90
    
    def _calculate_empirical_coverage(
        self,
        y_true: np.ndarray,
        p10: np.ndarray,
        p90: np.ndarray
    ) -> np.ndarray:
        """
        Calculate empirical coverage for each prediction.
        
        Args:
            y_true: True values
            p10: 10th percentile predictions
            p90: 90th percentile predictions
            
        Returns:
            Array of empirical coverage values
        """
        # Calculate normalized position within interval
        in_interval = np.logical_and(y_true >= p10, y_true <= p90)
        position = np.zeros_like(y_true, dtype=float)
        
        # For values within interval, calculate relative position
        mask = in_interval
        position[mask] = (y_true[mask] - p10[mask]) / (p90[mask] - p10[mask])
        
        # For values outside interval, set to 0 or 1
        position[~mask & (y_true < p10)] = 0
        position[~mask & (y_true > p90)] = 1
        
        return position
    
    def fit(
        self,
        y_true: np.ndarray,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        clusters: np.ndarray
    ) -> None:
        """
        Fit calibrators for each cluster using historical data.
        
        Args:
            y_true: True values
            predictions: Point predictions (p50)
            uncertainties: Uncertainty estimates
            clusters: Cluster assignments
        """
        # Calculate raw intervals
        p10_raw, p50_raw, p90_raw = self._calculate_raw_intervals(predictions, uncertainties)
        
        # Fit calibrator for each cluster
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            
            # Get cluster-specific data
            y_cluster = y_true[mask]
            p10_cluster = p10_raw[mask]
            p90_cluster = p90_raw[mask]
            
            # Calculate empirical coverage
            empirical_coverage = self._calculate_empirical_coverage(
                y_cluster, p10_cluster, p90_cluster
            )
            
            # Create bins for isotonic regression
            bins = np.linspace(0, 1, self.n_bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Calculate average coverage per bin
            binned_coverage = np.zeros(len(bin_centers))
            for i in range(len(bin_centers)):
                bin_mask = np.logical_and(
                    empirical_coverage >= bins[i],
                    empirical_coverage < bins[i + 1]
                )
                if np.any(bin_mask):
                    binned_coverage[i] = np.mean(empirical_coverage[bin_mask])
                else:
                    binned_coverage[i] = bin_centers[i]
            
            # Fit isotonic regression
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(bin_centers, binned_coverage)
            
            # Store calibrator and cluster statistics
            self.calibrators[cluster] = calibrator
            self.cluster_stats[cluster] = {
                'mean_uncertainty': np.mean(uncertainties[mask]),
                'std_uncertainty': np.std(uncertainties[mask]),
                'coverage_rate': np.mean(np.logical_and(
                    y_cluster >= p10_cluster,
                    y_cluster <= p90_cluster
                ))
            }
            
            logging.info(f"Fitted calibrator for cluster {cluster}")
            logging.info(f"Coverage rate: {self.cluster_stats[cluster]['coverage_rate']:.3f}")
    
    def calibrate(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        clusters: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calibrate prediction intervals using fitted calibrators.
        
        Args:
            predictions: Point predictions (p50)
            uncertainties: Uncertainty estimates
            clusters: Cluster assignments
            
        Returns:
            Tuple of calibrated (p10, p50, p90) arrays
        """
        p50 = predictions
        p10_calibrated = np.zeros_like(predictions)
        p90_calibrated = np.zeros_like(predictions)
        
        # Calibrate intervals for each cluster
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            if cluster not in self.calibrators:
                logging.warning(f"No calibrator found for cluster {cluster}, using raw intervals")
                p10_raw, _, p90_raw = self._calculate_raw_intervals(
                    predictions[mask], uncertainties[mask]
                )
                p10_calibrated[mask] = p10_raw
                p90_calibrated[mask] = p90_raw
                continue
            
            # Get cluster-specific data
            p50_cluster = p50[mask]
            unc_cluster = uncertainties[mask]
            
            # Calculate raw intervals
            p10_raw, _, p90_raw = self._calculate_raw_intervals(p50_cluster, unc_cluster)
            
            # Calculate normalized positions
            positions = np.linspace(0, 1, len(p50_cluster))
            
            # Apply calibration
            calibrated_positions = self.calibrators[cluster].predict(positions)
            
            # Map calibrated positions back to intervals
            interval_widths = p90_raw - p10_raw
            p10_calibrated[mask] = p50_cluster - calibrated_positions * interval_widths
            p90_calibrated[mask] = p50_cluster + (1 - calibrated_positions) * interval_widths
        
        return p10_calibrated, p50, p90_calibrated
    
    def save(self, path: str) -> None:
        """Save calibrators and statistics to disk."""
        calibration_data = {
            'calibrators': self.calibrators,
            'cluster_stats': self.cluster_stats,
            'n_bins': self.n_bins
        }
        joblib.dump(calibration_data, path)
    
    @classmethod
    def load(cls, path: str) -> 'ClusterCalibrator':
        """Load calibrators and statistics from disk."""
        calibration_data = joblib.load(path)
        calibrator = cls(n_bins=calibration_data['n_bins'])
        calibrator.calibrators = calibration_data['calibrators']
        calibrator.cluster_stats = calibration_data['cluster_stats']
        return calibrator 