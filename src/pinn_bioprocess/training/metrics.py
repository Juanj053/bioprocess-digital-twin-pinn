"""Evaluation metrics for digital twin trajectory validation."""

import numpy as np


def compute_metrics_series(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes R², MAE, and RMSE between ground truth and neural prediction."""
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()

    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))
    rmse = float(np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2)))

    ss_tot = float(np.sum((y_true_flat - np.mean(y_true_flat)) ** 2))
    ss_res = float(np.sum((y_true_flat - y_pred_flat) ** 2))

    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    return {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
    }


def compute_regression_metrics(
    gt: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Evaluates all 4 state trajectories."""
    return {
        "Biomass_X": compute_metrics_series(gt["X"], pred["X"]),
        "Substrate_S": compute_metrics_series(gt["S"], pred["S"]),
        "Oxygen_CL": compute_metrics_series(gt["CL"] * 1000.0, pred["CL"] * 1000.0),  # In mg/L for readability
        "Product_P": compute_metrics_series(gt["P"], pred["P"]),
    }
