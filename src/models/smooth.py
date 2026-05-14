import numpy as np
import pandas as pd
import optuna
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences, LocalBoxCox, LocalStandardScaler
from mlforecast.lag_transforms import RollingMean
from lightgbm import LGBMRegressor

from models.base import BaseForecaster
from utils.metrics import mase

optuna.logging.set_verbosity(optuna.logging.WARNING)

LAG_PRESETS = {
    'short':  [4, 13],
    'medium': [4, 8, 13, 26],
    'long':   [4, 8, 13, 26, 52],
}

LAG_TRANSFORM_MAP = {
    'rolling_mean_4':  RollingMean(window_size=4),
    'rolling_mean_13': RollingMean(window_size=13),
    'none':            None,
}

TARGET_TRANSFORM_MAP = {
    'none':      [],
    'log1p':     [LocalBoxCox()],
    'detrend':   [Differences([1])],
    'normalize': [LocalStandardScaler()],
}

# Static feature candidates — only those present in df_train are actually used
STATIC_COLS = [
    'product_attribute_1', 'product_attribute_2', 'product_attribute_3',
    'supplier_id', 'region_id',
]
STATIC_FEATURES = STATIC_COLS


class LGBMForecaster(BaseForecaster):
    demand_type = 'smooth'

    def __init__(
        self,
        horizon: int = 4,
        n_windows: int = 6,
        n_trials: int = 50,
        freq: str = 'W-MON',
        seed: int = 42,
        target_col: str = 'y',
    ):
        self.horizon = horizon
        self.n_windows = n_windows
        self.n_trials = n_trials
        self.freq = freq
        self.seed = seed
        self.target_col = target_col

        self.mlf_ = None
        self.best_params_ = None
        self.static_features_ = None
        self.study_ = None

    # ── internal helpers ───────────────────────────────────────────────────────

    def _build(self, trial, df_train: pd.DataFrame):
        lag_preset = trial.suggest_categorical('lag_preset', list(LAG_PRESETS.keys()))
        lags = LAG_PRESETS[lag_preset]

        lag_tf_name = trial.suggest_categorical('lag_transform', list(LAG_TRANSFORM_MAP.keys()))
        lag_tf = LAG_TRANSFORM_MAP[lag_tf_name]
        lag_transforms = {lag: [lag_tf] for lag in lags} if lag_tf is not None else None

        target_tf_name = trial.suggest_categorical('target_transform', list(TARGET_TRANSFORM_MAP.keys()))
        target_transforms = TARGET_TRANSFORM_MAP[target_tf_name] or None

        model_params = {
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves':       trial.suggest_int('num_leaves', 31, 512),
            'n_estimators':     trial.suggest_int('n_estimators', 100, 1000),
            'lambda_l1':        trial.suggest_float('lambda_l1', 0.01, 10.0, log=True),
            'lambda_l2':        trial.suggest_float('lambda_l2', 0.01, 10.0, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq':     1,
            'random_state':     self.seed,
            'verbose':          -1,
        }

        available_static = [c for c in STATIC_FEATURES if c in df_train.columns]

        mlf = MLForecast(
            models=[LGBMRegressor(**model_params)],
            freq=self.freq,
            lags=lags,
            lag_transforms=lag_transforms,
            target_transforms=target_transforms,
            date_features=['month', 'quarter', 'week'],
        )
        return mlf, available_static

    def _cv_loss(self, mlf: MLForecast, df_train: pd.DataFrame, static_features: list) -> float:
        cv_df = mlf.cross_validation(
            df_train,
            h=self.horizon,
            n_windows=self.n_windows,
            step_size=1,
            static_features=static_features,
            fitted=False,
        )
        scores = []
        for uid, group in cv_df.groupby('unique_id'):
            y_true = group[self.target_col].to_numpy(dtype=float)
            y_hat = group['LGBMRegressor'].to_numpy(dtype=float)
            y_tr = df_train[df_train['unique_id'] == uid][self.target_col].to_numpy(dtype=float)
            s = mase(y_true, y_hat, y_tr)
            if not np.isnan(s):
                scores.append(s)
        return float(np.mean(scores)) if scores else np.inf

    # ── public API ─────────────────────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame, best_params: dict = None, **kwargs):
        """
        Train the model.

        If best_params is provided (e.g. loaded from metadata.json), Optuna is
        skipped and the model is fit directly with those parameters — useful for
        refitting on the full dataset without repeating the search.
        """
        if best_params is not None:
            self.best_params_ = best_params
            self._fit_with_params(df_train, best_params)
            return self

        def objective(trial):
            mlf, sf = self._build(trial, df_train)
            return self._cv_loss(mlf, df_train, sf)

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
            study_name=f'lgbm_{self.demand_type}',
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        self.study_ = study
        self.best_params_ = study.best_params
        print(f'Melhor MASE (CV): {study.best_value:.4f}')
        print(f'Params: {study.best_params}')

        self._fit_with_params(df_train, study.best_params)
        return self

    def _fit_with_params(self, df_train: pd.DataFrame, params: dict):
        class _FakeTrial:
            def suggest_categorical(self, name, choices): return params[name]
            def suggest_int(self, name, *a, **k):         return params[name]
            def suggest_float(self, name, *a, **k):       return params[name]

        mlf, sf = self._build(_FakeTrial(), df_train)
        mlf.fit(df_train, static_features=sf)
        self.mlf_ = mlf
        self.static_features_ = sf

    def predict(
        self,
        horizon: int = None,
        df_static: pd.DataFrame = None,
        df_holidays: pd.DataFrame = None,
        df_weather: pd.DataFrame = None,
        holiday_cols: list = None,
        weather_cols: list = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Return predictions DataFrame. Pass df_static/df_holidays/df_weather for exogenous features."""
        h = horizon or self.horizon
        dynamic_cols = (holiday_cols or []) + (weather_cols or [])

        if dynamic_cols and df_static is not None:
            future = (
                self.mlf_.make_future_dataframe(h=h)
                .merge(df_static[['unique_id', 'region_id']], on='unique_id', how='left')
            )
            if holiday_cols and df_holidays is not None:
                avail = [c for c in holiday_cols if c in dynamic_cols]
                if avail:
                    future = future.merge(
                        df_holidays[['ds', 'region_id'] + avail],
                        on=['ds', 'region_id'], how='left',
                    )
            if weather_cols and df_weather is not None:
                avail = [c for c in weather_cols if c in dynamic_cols]
                if avail:
                    future = future.merge(
                        df_weather[['ds', 'region_id'] + avail],
                        on=['ds', 'region_id'], how='left',
                    )
            return self.mlf_.predict(h=h, X_df=future.drop(columns='region_id'))

        return self.mlf_.predict(h=h)
