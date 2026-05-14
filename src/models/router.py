from models.smooth import LGBMForecaster
from models.stats import StatsForecaster

# Demand types handled by the ML (LightGBM) path
ML_TYPES = {'smooth'}


def get_model(demand_type: str, **kwargs):
    """
    Factory that returns the right forecaster for a demand type.

    ML_TYPES  → LGBMForecaster  (Optuna + LightGBM global model)
    all else  → StatsForecaster (CV-based stats model selection)

    Extra kwargs are forwarded to the constructor (horizon, n_trials, freq, etc.).
    """
    if demand_type in ML_TYPES:
        ml_kwargs = {k: v for k, v in kwargs.items()
                     if k in ('horizon', 'n_windows', 'n_trials', 'freq', 'seed', 'target_col')}
        return LGBMForecaster(**ml_kwargs)

    stats_kwargs = {k: v for k, v in kwargs.items()
                    if k in ('horizon', 'n_windows', 'season_len', 'freq', 'target_col')}
    return StatsForecaster(demand_type=demand_type, **stats_kwargs)
