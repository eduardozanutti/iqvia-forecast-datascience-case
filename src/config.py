import yaml
import pandas as pd
from pathlib import Path


class LoadConfig:
    def __init__(self):
        config_file = Path(__file__).resolve().parent.parent / 'config.yaml'
        self._config_dir = config_file.parent
        with open(config_file, 'r') as f:
            self._config = yaml.safe_load(f)

    def _resolve(self, relative_path: str) -> Path:
        return self._config_dir / relative_path

    def get_path(self, *keys) -> Path:
        """Navigate config hierarchy and return resolved path.

        Example: cfg.get_path('tables', 'forecasting', 'tags', 'outliers')
        """
        node = self._config
        for key in keys:
            node = node[key]
        if not isinstance(node, str):
            available = list(node.keys()) if isinstance(node, dict) else node
            raise ValueError(f"Key chain {keys!r} resolves to a section. Available keys: {available}")
        return self._resolve(node)

    def load(self, *keys) -> pd.DataFrame:
        """Load parquet by config key chain (lazy — reads only when called)."""
        return pd.read_parquet(self.get_path(*keys))

    # ------------------------------------------------------------------
    # Convenience shortcuts matching the YAML structure
    # ------------------------------------------------------------------

    def load_dw(self, group: str, table: str) -> pd.DataFrame:
        """Load a data_warehouse table. group: 'dimensional' or 'fact'."""
        return self.load('tables', 'data_warehouse', group, table)

    def load_forecast(self, *keys) -> pd.DataFrame:
        """Load a forecasting table by key chain.

        Examples:
            cfg.load_forecast('features', 'static')
            cfg.load_forecast('datasets', 'refined', 'smooth')
            cfg.load_forecast('tags', 'series')
        """
        return self.load('tables', 'forecasting', *keys)
