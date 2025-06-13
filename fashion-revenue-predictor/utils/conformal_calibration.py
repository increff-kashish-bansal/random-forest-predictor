import numpy as np
from typing import Tuple, Optional
from scipy.stats import norm
import logging

class ConformalCalibrator:
    """
    Implements conformal quantile calibration for prediction intervals.
    Uses nonconformity scores to adjust prediction bounds while maintaining sharpness.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize the calibrator.
        
        Args:
            alpha: Significance level (default: 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile_adjustments = None
        
    def _calculate_nonconformity_scores(
        self,
        y_true: np.ndarray,
        p10: np.ndarray,
        p50: np.ndarray,
        p90: np.ndarray
    ) -> np.ndarray:
        """
        Calculate nonconformity scores for each prediction.
        
        Args:
            y_true: True values
            p10: 10th percentile predictions
            p50: 50th percentile predictions
            p90: 90th percentile predictions
            
        Returns:
            Array of nonconformity scores
        """
        # Calculate normalized distance from median
        normalized_distance = (y_true - p50) / (p90 - p10)
        
        # Calculate nonconformity score based on position relative to interval
        scores = np.zeros_like(y_true)
        
        # For values within interval
        in_interval = np.logical_and(y_true >= p10, y_true <= p90)
        scores[in_interval] = np.abs(normalized_distance[in_interval])
        
        # For values outside interval
        scores[~in_interval] = 1 + np.abs(normalized_distance[~in_interval])
        
        return scores
    
    def _calculate_quantile_adjustments(
        self,
        scores: np.ndarray,
        p10: np.ndarray,
        p50: np.ndarray,
        p90: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate adjustments for prediction bounds based on nonconformity scores.
        Returns scalar adjustments.
        """
        # Calculate empirical quantile of scores
        score_quantile = np.quantile(scores, 1 - self.alpha)
        return score_quantile, score_quantile
    
    def calibrate(
        self,
        y_true: np.ndarray,
        p10: np.ndarray,
        p50: np.ndarray,
        p90: np.ndarray
    ) -> None:
        """
        Fit the calibrator using historical data.
        
        Args:
            y_true: True values
            p10: 10th percentile predictions
            p50: 50th percentile predictions
            p90: 90th percentile predictions
        """
        # Calculate nonconformity scores
        self.calibration_scores = self._calculate_nonconformity_scores(y_true, p10, p50, p90)
        
        # Calculate quantile adjustments
        self.quantile_adjustments = self._calculate_quantile_adjustments(
            self.calibration_scores, p10, p50, p90
        )
        
        # Log calibration statistics
        coverage_before = np.mean(np.logical_and(y_true >= p10, y_true <= p90))
        logging.info(f"Coverage before calibration: {coverage_before:.1%}")
    
    def calibrate_predictions(
        self,
        p10: np.ndarray,
        p50: np.ndarray,
        p90: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply calibration to new predictions.
        """
        if self.quantile_adjustments is None:
            raise ValueError("Calibrator must be fitted before making predictions")
        lower_adjustment, upper_adjustment = self.quantile_adjustments
        interval_width = p90 - p10
        p10_calibrated = p10 - lower_adjustment * interval_width
        p90_calibrated = p90 + upper_adjustment * interval_width
        return p10_calibrated, p50, p90_calibrated 