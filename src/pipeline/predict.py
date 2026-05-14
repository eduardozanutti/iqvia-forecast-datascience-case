"""
Final forecast generation.

Loads trained artifacts (metadata.json with best_params / model_col),
refits each model on the *full* dataset (no holdout), and generates
H-week ahead predictions saved to predictions/forecast/.

Column schema:
    unique_id | ds | y_pred | demand_type | model | ref_date | run_date

ref_date = last date in the training data used for this run
run_date = calendar date the pipeline executed
"""
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

from config import LoadConfig
from models.base import BaseForecaster
from models.router import ML_TYPES
from models.smooth import LGBMForecaster
from models.stats import StatsForecaster, MODEL_REGISTRY, MODEL_REGISTRY_BY_TYPE
from pipeline.features import FeatureLoader, HOLIDAY_COLS, WEATHER_COLS

warnings.filterwarnings('ignore')

DEMAND_TYPES = ['smooth', 'erratic', 'intermittent', 'lumpy']
TARGET_COL   = 'y'
HORIZON      = 4
SEASON_LEN   = 52
SEED         = 42


def run_final_forecast(
    demand_types: list = None,
    artifact_base: Path = None,
    predictions_base: Path = None,
    horizon: int = HORIZON,
):
    """
    Refit each model on the full dataset using the best configuration found
    during training, then generate H-week ahead forecasts.

    Requires that run_training() has already been executed so that
    artifacts/models/{demand_type}/metadata.json exists.

    Args:
        demand_types: subset of DEMAND_TYPES (default: all four)
        artifact_base: where model artifacts were saved (default: artifacts/models)
        predictions_base: root for prediction files (default: predictions)
        horizon: forecast horizon in weeks

    Returns:
        DataFrame with all predictions concatenated.
    """
    cfg = LoadConfig()
    loader = FeatureLoader(cfg)
    demand_types = demand_types or DEMAND_TYPES
    artifact_base = Path(artifact_base) if artifact_base else Path('artifacts/models')
    predictions_base = Path(predictions_base) if predictions_base else Path('data/gold/forecasting/predictions')

    _ref = cfg.load_forecast('datasets', 'refined', 'smooth')
    freq = FeatureLoader.detect_freq(_ref)
    run_date = date.today()

    print(f'Frequency   : {freq}')
    print(f'Horizon     : {horizon} weeks')
    print(f'Run date    : {run_date}')

    all_preds = []

    for dt in demand_types:
        sep = '=' * 60
        print(f'\n{sep}\nForecasting: {dt.upper()}\n{sep}')

        artifact_dir = artifact_base / dt
        meta = BaseForecaster.load_metadata(artifact_dir)
        if not meta:
            raise FileNotFoundError(
                f'metadata.json not found in {artifact_dir}. '
                f'Run run_training() first.'
            )

        # Full dataset — no holdout split
        df_all = loader.build_dataset(dt)
        ref_date = df_all['ds'].max()
        print(f'History up to : {ref_date.date()}  ({df_all["unique_id"].nunique():,} series)')

        # ── Refit ─────────────────────────────────────────────────────────────
        if dt in ML_TYPES:
            best_params = meta.get('best_params')
            if not best_params:
                raise ValueError(f'best_params missing in {artifact_dir}/metadata.json')

            model = LGBMForecaster(
                horizon=horizon, freq=freq, seed=SEED, target_col=TARGET_COL,
            )
            print('Refitting LGBMForecaster (Optuna skipped)...')
            model.fit(df_all, best_params=best_params)

            avail_hol = [c for c in HOLIDAY_COLS if c in df_all.columns]
            avail_wth = [c for c in WEATHER_COLS if c in df_all.columns]
            preds = model.predict(
                df_static=loader.df_static,
                df_holidays=loader.df_holidays,
                df_weather=loader.df_weather,
                holiday_cols=avail_hol,
                weather_cols=avail_wth,
            )
            pred_col = 'LGBMRegressor'

        else:
            model_col = meta.get('model_col')
            if not model_col:
                raise ValueError(f'model_col missing in {artifact_dir}/metadata.json')

            # AutoARIMA has demand-type variants — try the specific lookup first
            type_registry = MODEL_REGISTRY_BY_TYPE.get(dt, {})
            best_model = type_registry.get(model_col) or MODEL_REGISTRY.get(model_col)
            if best_model is None:
                raise ValueError(
                    f'Model "{model_col}" not found in MODEL_REGISTRY. '
                    f'Available: {list(MODEL_REGISTRY.keys())}'
                )

            model = StatsForecaster(
                demand_type=dt, horizon=horizon, freq=freq,
                season_len=SEASON_LEN, target_col=TARGET_COL,
            )
            print(f'Refitting {model_col} (CV skipped)...')
            model.fit(df_all, best_model=best_model)
            preds = model.predict()   # no df_test → raw StatsForecast output
            pred_col = model_col

        # ── Format output ──────────────────────────────────────────────────────
        preds = preds.rename(columns={pred_col: 'y_pred'})
        preds['demand_type'] = dt
        preds['model']       = pred_col
        preds['ref_date']    = ref_date
        preds['run_date']    = run_date
        all_preds.append(preds)

        print(f'Generated {len(preds):,} forecasts  '
              f'({preds["ds"].min().date()} → {preds["ds"].max().date()})')

    # ── Save ──────────────────────────────────────────────────────────────────
    df_forecast = pd.concat(all_preds, ignore_index=True)

    ref_str = pd.Timestamp(df_forecast['ref_date'].max()).strftime('%Y%m%d')
    run_str = run_date.strftime('%Y%m%d')
    out_path = predictions_base / 'forecast' / f'forecast_ref{ref_str}_run{run_str}.parquet'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_forecast.to_parquet(out_path, index=False)

    sep = '=' * 60
    print(f'\n{sep}')
    print(f'Forecast saved : {out_path}')
    print(f'Total rows     : {len(df_forecast):,}')
    print(df_forecast.groupby('demand_type')[['unique_id']].count().rename(
        columns={'unique_id': 'n_rows'}).to_string())

    return df_forecast
