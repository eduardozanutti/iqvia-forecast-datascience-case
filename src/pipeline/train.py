"""
Full training orchestration.

Trains each demand type model, evaluates on a held-out test split, saves
model artifacts to artifacts/models/ and backtest predictions to
predictions/backtest/.

Usage (from project root, with src/ on sys.path):
    python src/main.py train
    python src/main.py train --types smooth --trials 50
"""
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, CrostonOptimized

from config import LoadConfig
from models.router import get_model, ML_TYPES
from pipeline.features import FeatureLoader, HOLIDAY_COLS, WEATHER_COLS
from utils.metrics import evaluate_forecast

warnings.filterwarnings('ignore')

DEMAND_TYPES = ['smooth', 'erratic', 'intermittent', 'lumpy']
TARGET_COL   = 'y'
HORIZON      = 4
N_WINDOWS    = 6
N_TRIALS     = 50
SEASON_LEN   = 52
SEED         = 42


def train_test_split(df: pd.DataFrame, horizon: int):
    cutoff = df['ds'].max() - pd.Timedelta(weeks=horizon)
    return df[df['ds'] <= cutoff].copy(), df[df['ds'] > cutoff].copy()


def _run_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    demand_type: str,
    freq: str,
):
    if demand_type in ('smooth', 'erratic'):
        models = [SeasonalNaive(season_length=SEASON_LEN)]
        col = 'SeasonalNaive'
    else:
        models = [CrostonOptimized()]
        col = 'CrostonOptimized'

    sf = StatsForecast(models=models, freq=freq, n_jobs=-1)
    sf.fit(df_train[['unique_id', 'ds', TARGET_COL]])
    preds = sf.predict(h=HORIZON)

    # Rank-based alignment avoids weekday-anchor mismatches
    test_r = (
        df_test[['unique_id', 'ds', TARGET_COL]]
        .sort_values(['unique_id', 'ds'])
        .assign(rank=lambda d: d.groupby('unique_id').cumcount())
    )
    pred_r = (
        preds.sort_values(['unique_id', 'ds'])
        .assign(rank=lambda d: d.groupby('unique_id').cumcount())
        .drop(columns='ds')
    )
    merged = test_r.merge(pred_r, on=['unique_id', 'rank'], how='left').drop(columns='rank')
    return merged, col


def run_training(
    demand_types: list = None,
    artifact_base: Path = None,
    predictions_base: Path = None,
    n_trials: int = N_TRIALS,
    horizon: int = HORIZON,
    n_windows: int = N_WINDOWS,
):
    """
    Train models for each demand type, evaluate against baseline, and save artifacts.

    Args:
        demand_types: subset of DEMAND_TYPES to train (default: all four)
        artifact_base: root directory for saved models (default: artifacts/models)
        n_trials: Optuna trials for the LightGBM model
        horizon: forecast horizon in weeks
        n_windows: number of CV windows

    Returns:
        dict mapping demand_type → {baseline: metrics, model: metrics, delta_wape_pct: float}
    """
    cfg = LoadConfig()
    loader = FeatureLoader(cfg)
    demand_types = demand_types or DEMAND_TYPES
    artifact_base    = Path(artifact_base)    if artifact_base    else Path('artifacts/models')
    predictions_base = Path(predictions_base) if predictions_base else Path('data/gold/forecasting/predictions')

    # Detect weekly frequency anchor from the data
    _ref = cfg.load_forecast('datasets', 'refined', 'smooth')
    freq = FeatureLoader.detect_freq(_ref)
    run_date = date.today()

    print(f'Frequência detectada : {freq}')
    print(f'Demand types         : {demand_types}')
    print(f'Horizon              : {horizon} semanas')
    print(f'Artifacts            : {artifact_base.resolve()}')

    all_results  = {}
    all_backtest = []   # accumulated across demand types

    for dt in demand_types:
        sep = '=' * 60
        print(f'\n{sep}\nTreinando: {dt.upper()}\n{sep}')

        df = loader.build_dataset(dt)
        df_train, df_test = train_test_split(df, horizon)

        print(f'Train: {len(df_train):,} rows | {df_train["unique_id"].nunique():,} séries')
        print(f'Test : {len(df_test):,} rows')

        # ── Baseline ──────────────────────────────────────────────────────────
        baseline_preds, baseline_col = _run_baseline(df_train, df_test, dt, freq)
        baseline_metrics = evaluate_forecast(baseline_preds, df_train, baseline_col)
        print(f'Baseline  : MASE={baseline_metrics["mase"]:.4f}  '
              f'SMAPE={baseline_metrics["smape"]:.4f}  '
              f'WAPE={baseline_metrics["wape"]:.4f}')

        # ── Build + fit model ─────────────────────────────────────────────────
        common = dict(horizon=horizon, n_windows=n_windows, freq=freq, target_col=TARGET_COL)
        if dt in ML_TYPES:
            model = get_model(dt, **common, n_trials=n_trials, seed=SEED)
        else:
            model = get_model(dt, **common, season_len=SEASON_LEN)
        model.fit(df_train)

        # ── Predict on test set ───────────────────────────────────────────────
        if dt in ML_TYPES:
            avail_hol = [c for c in HOLIDAY_COLS if c in df_train.columns]
            avail_wth = [c for c in WEATHER_COLS if c in df_train.columns]
            raw_preds = model.predict(
                df_static=loader.df_static,
                df_holidays=loader.df_holidays,
                df_weather=loader.df_weather,
                holiday_cols=avail_hol,
                weather_cols=avail_wth,
            )
            pred_col = 'LGBMRegressor'
            model_preds = df_test[['unique_id', 'ds', TARGET_COL]].merge(
                raw_preds, on=['unique_id', 'ds'], how='left'
            )
        else:
            model_preds = model.predict(df_test=df_test)
            pred_col = model.model_col_

        model_metrics = evaluate_forecast(model_preds, df_train, pred_col)
        print(f'Model     : MASE={model_metrics["mase"]:.4f}  '
              f'SMAPE={model_metrics["smape"]:.4f}  '
              f'WAPE={model_metrics["wape"]:.4f}')

        # ── Delta vs baseline ─────────────────────────────────────────────────
        delta = (baseline_metrics['wape'] - model_metrics['wape']) / baseline_metrics['wape'] * 100
        sign = '+' if delta > 0 else ''
        print(f'WAPE delta: {sign}{delta:.1f}%  '
              f'({baseline_metrics["wape"]:.4f} → {model_metrics["wape"]:.4f})')

        # ── Accumulate backtest predictions ───────────────────────────────────
        bt = (
            model_preds
            .rename(columns={pred_col: 'y_pred', TARGET_COL: 'y'})
            .merge(
                baseline_preds[['unique_id', 'ds', baseline_col]]
                .rename(columns={baseline_col: 'y_baseline'}),
                on=['unique_id', 'ds'], how='left',
            )
        )
        bt['demand_type']    = dt
        bt['model']          = pred_col
        bt['baseline_model'] = baseline_col
        bt['ref_date']       = df_train['ds'].max()
        bt['run_date']       = run_date
        all_backtest.append(bt)

        # ── Save artifact ─────────────────────────────────────────────────────
        extra_meta = {
            'freq': freq,
            'horizon': horizon,
            'n_windows': n_windows,
            'baseline_metrics': baseline_metrics,
            'model_metrics': model_metrics,
            'delta_wape_pct': round(delta, 2),
        }
        if hasattr(model, 'best_params_') and model.best_params_:
            extra_meta['best_params'] = model.best_params_
        if hasattr(model, 'model_col_') and model.model_col_:
            extra_meta['model_col'] = model.model_col_
            extra_meta['cv_wape'] = model.cv_wape_

        artifact_dir = artifact_base / dt
        model.save(artifact_dir, extra_meta=extra_meta)
        print(f'Salvo em  : {artifact_dir}')

        all_results[dt] = {
            'baseline': baseline_metrics,
            'model': model_metrics,
            'delta_wape_pct': round(delta, 2),
        }

    # ── Save consolidated backtest ────────────────────────────────────────────
    df_backtest = pd.concat(all_backtest, ignore_index=True)
    ref_str  = pd.Timestamp(df_backtest['ref_date'].max()).strftime('%Y%m%d')
    run_str  = run_date.strftime('%Y%m%d')
    bt_path  = predictions_base / 'backtest' / f'backtest_ref{ref_str}_run{run_str}.parquet'
    bt_path.parent.mkdir(parents=True, exist_ok=True)
    df_backtest.to_parquet(bt_path, index=False)
    print(f'\nBacktest salvo: {bt_path}  ({len(df_backtest):,} linhas)')

    # ── Summary table ─────────────────────────────────────────────────────────
    sep = '=' * 70
    print(f'\n{sep}\nRESUMO FINAL\n{sep}')
    rows = []
    for dt, res in all_results.items():
        rows.append({'demand_type': dt, 'model': 'Baseline', **res['baseline']})
        rows.append({'demand_type': dt, 'model': 'Trained',  **res['model']})
    df_summary = (
        pd.DataFrame(rows)
        .set_index(['demand_type', 'model'])[['mase', 'smape', 'wape', 'n_series']]
        .round(4)
    )
    print(df_summary.to_string())

    return all_results
