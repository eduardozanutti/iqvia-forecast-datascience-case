import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoTheta,
    SeasonalExponentialSmoothing,
    TSB,
    ADIDA,
    IMAPA,
    CrostonSBA,
    CrostonOptimized,
)

from models.base import BaseForecaster
from utils.metrics import wape


def _model_name(model) -> str:
    """Return the column name StatsForecast uses for this model."""
    # StatsForecast derives column names from model.__repr__() stripped of params.
    # Easier to just run a tiny dummy predict — instead we maintain this lookup.
    return type(model).__name__.replace('SeasonalExponentialSmoothing', 'SeasonalES')


# Registry used by predict.py to reconstruct a model from its saved column name
# AutoARIMA(season_length=13) e AutoARIMA() produzem a mesma coluna "AutoARIMA"
# no StatsForecast. O registry usa variantes por demand_type para o predict.py
# selecionar a instância correta via MODEL_REGISTRY_BY_TYPE.
MODEL_REGISTRY = {
    'AutoETS':          AutoETS(season_length=52),
    'AutoTheta':        AutoTheta(season_length=52),
    'SeasonalES':       SeasonalExponentialSmoothing(season_length=52, alpha=0.3),
    'CrostonOptimized': CrostonOptimized(),
    'TSB':              TSB(alpha_d=0.3, alpha_p=0.3),
    'ADIDA':            ADIDA(),
    'IMAPA':            IMAPA(),
    'CrostonSBA':       CrostonSBA(),
}

# AutoARIMA tem variantes por demand_type (season_length diferente)
MODEL_REGISTRY_BY_TYPE = {
    'erratic':      {'AutoARIMA': AutoARIMA(season_length=13)},
    'lumpy':        {'AutoARIMA': AutoARIMA()},
    'intermittent': {'AutoARIMA': AutoARIMA()},
}

# Candidates evaluated via CV per demand type
STATS_CANDIDATES = {
    'erratic': [
        # AutoARIMA com sazonalidade trimestral — timing regular favorece ARIMA
        # season_length=13 ao invés de 52 para evitar SARIMA com período longo (muito lento)
        AutoARIMA(season_length=13, approximation=True),
        AutoETS(season_length=52),
        AutoTheta(season_length=52),
        SeasonalExponentialSmoothing(season_length=52, alpha=0.3),
        CrostonOptimized(),
    ],
    'intermittent': [
        TSB(alpha_d=0.3, alpha_p=0.3),
        ADIDA(),
        IMAPA(),
        CrostonSBA(),
        CrostonOptimized(),
    ],
    'lumpy': [
        # AutoETS e AutoARIMA adicionados — lumpy com alguma estrutura pode se beneficiar
        # AutoARIMA sem sazonalidade explícita: séries esparsas raramente têm padrão sazonal estável
        AutoETS(season_length=52),
        AutoARIMA(approximation=True),
        IMAPA(),
        TSB(alpha_d=0.3, alpha_p=0.3),
        CrostonSBA(),
        CrostonOptimized(),
    ],
}


def _pred_col(cv_df: pd.DataFrame, target_col: str) -> str:
    skip = {'unique_id', 'ds', 'cutoff', target_col}
    return next(c for c in cv_df.columns if c not in skip)


def _filter_min_length(df: pd.DataFrame, horizon: int, n_windows: int, season_len: int = 0) -> pd.DataFrame:
    min_obs = horizon * (n_windows + 1) + max(season_len, 0)
    counts = df.groupby('unique_id').size()
    valid = counts[counts >= min_obs].index
    dropped = len(counts) - len(valid)
    if dropped:
        print(f'    (descartadas {dropped} séries curtas — min_obs={min_obs})')
    return df[df['unique_id'].isin(valid)]


class StatsForecaster(BaseForecaster):

    def __init__(
        self,
        demand_type: str,
        horizon: int = 4,
        n_windows: int = 6,
        season_len: int = 52,
        freq: str = 'W-MON',
        target_col: str = 'y',
    ):
        self.demand_type = demand_type
        self.horizon = horizon
        self.n_windows = n_windows
        self.season_len = season_len
        self.freq = freq
        self.target_col = target_col

        self.sf_ = None
        self.model_col_ = None
        self.best_model_ = None
        self.cv_wape_ = None

    # ── model selection ────────────────────────────────────────────────────────

    def _select_best(self, df_train: pd.DataFrame):
        candidates = STATS_CANDIDATES.get(self.demand_type, [CrostonOptimized()])
        results = {}

        for model in candidates:
            name = type(model).__name__
            season = getattr(model, 'season_length', 0) or 0
            df_cv = _filter_min_length(df_train, self.horizon, self.n_windows, season_len=season)

            if df_cv['unique_id'].nunique() == 0:
                print(f'  {name:<45}: sem séries suficientes, pulado')
                continue

            try:
                sf = StatsForecast(models=[model], freq=self.freq, n_jobs=-1)
                cv_df = sf.cross_validation(
                    df=df_cv[['unique_id', 'ds', self.target_col]],
                    h=self.horizon,
                    n_windows=self.n_windows,
                    step_size=self.horizon,
                )
                col = _pred_col(cv_df, self.target_col)
                scores = [
                    wape(
                        g[self.target_col].to_numpy(dtype=float),
                        g[col].to_numpy(dtype=float),
                    )
                    for _, g in cv_df.groupby('unique_id')
                ]
                mean_w = float(np.nanmean(scores)) if scores else np.inf
                results[col] = (model, mean_w)
                n_s = df_cv['unique_id'].nunique()
                print(f'  {col:<45}: CV WAPE = {mean_w:.4f}  ({n_s} séries)')
            except Exception as exc:
                print(f'  {name:<45}: ERRO — {exc}')

        if not results:
            print('  Nenhum modelo completou CV — usando CrostonOptimized como fallback')
            return CrostonOptimized(), 'CrostonOptimized', np.inf

        best_col = min(results, key=lambda k: results[k][1])
        best_mdl, best_w = results[best_col]
        print(f'\n  Vencedor: {best_col}  (CV WAPE = {best_w:.4f})')
        return best_mdl, best_col, best_w

    # ── public API ─────────────────────────────────────────────────────────────

    def fit(self, df_train: pd.DataFrame, best_model=None, **kwargs):
        """
        Select best model via CV then fit on the full training set.

        If best_model is provided (e.g. loaded from _MODEL_REGISTRY using the
        model_col stored in metadata.json), CV selection is skipped — useful for
        refitting on the full dataset without repeating the search.
        """
        if best_model is not None:
            self.best_model_ = best_model
            self.model_col_ = _model_name(best_model)
            self.cv_wape_ = None
            self._fit_model(df_train, best_model)
            return self

        model, col, cv_wape = self._select_best(df_train)
        self.best_model_ = model
        self.model_col_ = col
        self.cv_wape_ = cv_wape
        self._fit_model(df_train, model)
        return self

    def _fit_model(self, df_train: pd.DataFrame, model):
        sf = StatsForecast(models=[model], freq=self.freq, n_jobs=-1)
        sf.fit(df_train[['unique_id', 'ds', self.target_col]])
        self.sf_ = sf

    def predict(self, horizon: int = None, df_test: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        """
        Return predictions aligned to df_test by rank (avoids date-offset issues
        when StatsForecast anchors to a different weekday than the training data).
        If df_test is None, returns the raw StatsForecast predict output.
        """
        h = horizon or self.horizon
        preds = self.sf_.predict(h=h)

        if df_test is not None:
            test_r = (
                df_test[['unique_id', 'ds', self.target_col]]
                .sort_values(['unique_id', 'ds'])
                .assign(rank=lambda d: d.groupby('unique_id').cumcount())
            )
            pred_r = (
                preds.sort_values(['unique_id', 'ds'])
                .assign(rank=lambda d: d.groupby('unique_id').cumcount())
                .drop(columns='ds')
            )
            return test_r.merge(pred_r, on=['unique_id', 'rank'], how='left').drop(columns='rank')

        return preds
