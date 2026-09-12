 
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def regression_metrics(y_true, y_pred):
    """
    Calculate regression evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    within_10 = np.mean(np.abs(y_true - y_pred) <= 10.0)
    within_20 = np.mean(np.abs(y_true - y_pred) <= 20.0)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "within_10_cycles": float(within_10),
        "within_20_cycles": float(within_20),
        "count": int(len(y_true))
    }