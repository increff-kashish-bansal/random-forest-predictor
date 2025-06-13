import logging
import numpy as np
from scipy.stats import median_abs_deviation
from conformal_calibration import ConformalCalibrator

# Calculate prediction intervals
logging.info("Calculating prediction intervals...")
p50 = median_pred
p10 = p50 - lower_residuals
p90 = p50 + upper_residuals

# Apply conformal calibration
logging.info("Applying conformal calibration...")
calibrator = ConformalCalibrator(alpha=0.1)  # 90% coverage
calibrator.calibrate(log_y, p10, p50, p90)  # Fit on historical data
p10_calibrated, p50_calibrated, p90_calibrated = calibrator.calibrate_predictions(
    p10, p50, p90
)

# Convert back to original scale using relative modeling
p10_calibrated = np.expm1(p10_calibrated) * avg_day_month_revenue
p50_calibrated = np.expm1(p50_calibrated) * avg_day_month_revenue
p90_calibrated = np.expm1(p90_calibrated) * avg_day_month_revenue

# Create prediction result
prediction = {
    'store': store_id,
    'date': prediction_date,
    'p10': float(p10_calibrated[0]),
    'p50': float(p50_calibrated[0]),
    'p90': float(p90_calibrated[0])
} 