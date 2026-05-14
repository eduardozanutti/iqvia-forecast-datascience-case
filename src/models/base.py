from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
import json
import joblib


class BaseForecaster(ABC):
    """Common interface for all demand-type forecasters."""

    demand_type: str = ''

    @abstractmethod
    def fit(self, df_train, **kwargs):
        """Train the model on df_train."""

    @abstractmethod
    def predict(self, horizon: int = None, **kwargs):
        """Return a DataFrame with columns [unique_id, ds, <prediction_col>]."""

    def save(self, artifact_dir: Path, extra_meta: dict = None):
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, artifact_dir / 'model.pkl')
        meta = {
            'demand_type': self.demand_type,
            'class': type(self).__name__,
            'saved_at': datetime.now(timezone.utc).isoformat(),
            **(extra_meta or {}),
        }
        (artifact_dir / 'metadata.json').write_text(
            json.dumps(meta, indent=2, default=str)
        )

    @classmethod
    def load(cls, artifact_dir: Path):
        return joblib.load(Path(artifact_dir) / 'model.pkl')

    @staticmethod
    def load_metadata(artifact_dir: Path) -> dict:
        path = Path(artifact_dir) / 'metadata.json'
        return json.loads(path.read_text()) if path.exists() else {}
