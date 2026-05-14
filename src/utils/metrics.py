import numpy as np
import pandas as pd


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Mean Absolute Scaled Error. Returns NaN when naive MAE is zero."""
    naive_mae = np.mean(np.abs(np.diff(y_train)))
    if naive_mae == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / naive_mae)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — stable when y_true=0."""
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    return float(2 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted APE — handles zero-heavy series better than MAPE."""
    total = np.sum(np.abs(y_true))
    if total == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / total)


def evaluate_forecast(
    df_pred: pd.DataFrame,
    df_train: pd.DataFrame,
    model_col: str,
    target_col: str = 'y',
) -> dict:
    """Per-series metrics aggregated to scalar averages."""
    results = []
    for uid, group in df_pred.groupby('unique_id'):
        y_true = group[target_col].to_numpy(dtype=float)
        y_hat = group[model_col].to_numpy(dtype=float)
        y_tr = df_train[df_train['unique_id'] == uid][target_col].to_numpy(dtype=float)
        results.append({
            'unique_id': uid,
            'mase': mase(y_true, y_hat, y_tr),
            'smape': smape(y_true, y_hat),
            'wape': wape(y_true, y_hat),
        })
    df_res = pd.DataFrame(results)
    return {
        'mase': float(df_res['mase'].mean()),
        'smape': float(df_res['smape'].mean()),
        'wape': float(df_res['wape'].mean()),
        'n_series': len(df_res),
    }
