import pandas as pd

STATIC_COLS = [
    'product_attribute_1', 'product_attribute_2', 'product_attribute_3',
    'supplier_id', 'region_id',
]
HOLIDAY_COLS = [
    'is_national_holiday', 'is_holiday',
    'weeks_to_next_holiday', 'weeks_since_last_holiday',
]
WEATHER_COLS = [
    'temp_max', 'temp_min', 'temp_mean',
    'precip', 'wind_max', 'is_rainy',
]
DYNAMIC_COLS = HOLIDAY_COLS + WEATHER_COLS


class FeatureLoader:
    """Lazy-loading cache for all feature tables. Pass a LoadConfig instance."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._df_static = None
        self._df_holidays = None
        self._df_weather = None

    @property
    def df_static(self):
        if self._df_static is None:
            df = self.cfg.load_forecast('features', 'static')
            self._df_static = df[['unique_id'] + STATIC_COLS].drop_duplicates('unique_id')
        return self._df_static

    @property
    def df_holidays(self):
        if self._df_holidays is None:
            try:
                self._df_holidays = self.cfg.load_forecast('features', 'holidays')
            except Exception:
                return None
        return self._df_holidays

    @property
    def df_weather(self):
        if self._df_weather is None:
            try:
                self._df_weather = self.cfg.load_forecast('features', 'weather')
            except Exception:
                return None
        return self._df_weather

    def build_dataset(self, demand_type: str) -> pd.DataFrame:
        """Load refined dataset and merge all available feature tables."""
        df = self.cfg.load_forecast('datasets', 'refined', demand_type)
        df = df.drop(columns=['demand_type'], errors='ignore')
        df = df.merge(self.df_static, on='unique_id', how='left')

        if self.df_holidays is not None:
            df = df.merge(
                self.df_holidays[['ds', 'region_id'] + HOLIDAY_COLS],
                on=['ds', 'region_id'], how='left',
            )
        if self.df_weather is not None:
            df = df.merge(
                self.df_weather[['ds', 'region_id'] + WEATHER_COLS],
                on=['ds', 'region_id'], how='left',
            )
        return df

    @staticmethod
    def detect_freq(df: pd.DataFrame) -> str:
        day_abbrevs = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        day_num = int(df['ds'].dt.dayofweek.mode()[0])
        return f'W-{day_abbrevs[day_num]}'
