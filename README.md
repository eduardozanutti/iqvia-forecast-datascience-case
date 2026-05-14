# IQVIA — Weekly Demand Forecasting

Forecasting pipeline for weekly pharmaceutical product demand, segmented by distributor and geographic region. Covers the full workflow from raw data ingestion to production-ready predictions.

---

## Approach

### 1. Data Pipeline — Medallion Architecture

Raw sales data is processed through three layers before reaching the modeling stage:

```
data/raw/  →  data/bronze/  →  data/silver/  →  data/gold/
              (ingest)         (clean)           (model-ready)
```

### 2. Demand Segmentation

Each time series is classified into one of four demand types using the **Syntetos-Boylan** criteria (coefficient of variation × average inter-demand interval). This drives the entire modeling strategy downstream.

| Type | Characteristic | Series |
|---|---|---|
| Smooth | Regular timing, stable volume | 528 |
| Erratic | Regular timing, high volume variance | 433 |
| Intermittent | Sparse demand, stable volume | 698 |
| Lumpy | Sparse + high volume variance | 555 |

### 3. Feature Engineering

| Feature group | Description | Key |
|---|---|---|
| Static | Product attributes, supplier, region | per `unique_id` |
| Holidays | Brazilian national + regional holidays, proximity features | per `(ds, region_id)` |
| Weather | Temperature, precipitation, wind — **52-week lag** as leakage-free future proxy | per `(ds, region_id)` |

Holiday coverage uses the 5 IBGE macro-regions mapped to their constituent states. The 52-week weather lag replicates seasonal patterns without requiring future data at prediction time.

### 4. Modeling Strategy

Each demand type uses a tailored approach:

| Demand type | Model | Tuning |
|---|---|---|
| Smooth | LightGBM (global, MLForecast) | Optuna — 50 trials, CV MASE |
| Erratic | Best of: AutoETS, AutoTheta, SeasonalES, CrostonOptimized | CV WAPE per candidate |
| Intermittent | Best of: TSB, ADIDA, IMAPA, CrostonSBA, CrostonOptimized | CV WAPE per candidate |
| Lumpy | Best of: IMAPA, TSB, CrostonSBA, CrostonOptimized | CV WAPE per candidate |

**Baseline**: SeasonalNaive (season=52) for smooth/erratic; CrostonOptimized for intermittent/lumpy.

---

## Results

Evaluation on a 4-week held-out test split (time-based, no data leakage).

| Demand type | Model selected | Baseline WAPE | Model WAPE | Improvement |
|---|---|---|---|---|
| Smooth | LightGBM | 0.821 | 0.483 | **+41.2%** |
| Erratic | CrostonOptimized | 2.375 | 1.044 | **+56.1%** |
| Intermittent | TSB | 1.258 | 0.883 | **+29.8%** |
| Lumpy | TSB | 1.200 | 0.970 | **+19.2%** |

Top SHAP features for the smooth LightGBM model: lag features (rolling means), week of year, quarter, region_id, and proximity to holidays.

---

## Project Structure

```
├── config.yaml                        # All data paths (single source of truth)
├── requirements.txt
├── run_train.py                       # Quick training entry point (Shift+Enter)
├── docs/
│   └── case_brief.md
├── notebooks/
│   ├── data_engineering/
│   │   ├── 1-bronze_layer.ipynb      # Raw ingestion
│   │   ├── 2-silver_layer.ipynb      # Cleaning & standardisation
│   │   └── 3-gold_layer.ipynb        # Data warehouse (dim + fact tables)
│   └── data_science/
│       ├── eda.ipynb                  # Exploratory analysis
│       ├── 1-create_base_dataset.ipynb
│       ├── 2-data_preparation.ipynb  # Outlier removal, demand segmentation
│       ├── 3-create_features.ipynb   # Static + holiday features
│       ├── 4-weather_features.ipynb  # Weather features via Open-Meteo API
│       └── experiments/
│           └── model_building.ipynb  # Full model experimentation notebook
├── src/
│   ├── config.py                     # LoadConfig — path resolution from config.yaml
│   ├── main.py                       # CLI entry point (train / predict subcommands)
│   ├── models/
│   │   ├── base.py                   # BaseForecaster ABC (fit / predict / save / load)
│   │   ├── smooth.py                 # LGBMForecaster (Optuna + MLForecast)
│   │   ├── stats.py                  # StatsForecaster (CV-based model selection)
│   │   └── router.py                 # get_model(demand_type) factory
│   ├── pipeline/
│   │   ├── features.py               # FeatureLoader — lazy-cached feature tables
│   │   ├── train.py                  # Training orchestration + backtest saving
│   │   └── predict.py                # Final forecast (refit on full history)
│   └── utils/
│       └── metrics.py                # mase / smape / wape / evaluate_forecast
├── artifacts/
│   └── models/
│       └── {demand_type}/
│           ├── model.pkl             # Serialised forecaster (gitignored)
│           └── metadata.json         # Best params, metrics, delta WAPE
└── data/gold/forecasting/
    └── predictions/
        ├── backtest/                 # Test-split predictions (y + y_pred + y_baseline)
        └── forecast/                 # Future H-week predictions
```

---

## Reproducing the Solution

### Prerequisites

```bash
pip install -r requirements.txt
```

Place the raw data file in `data/raw/`.

### Step 1 — Data Engineering

Run in order:

```
notebooks/data_engineering/1-bronze_layer.ipynb
notebooks/data_engineering/2-silver_layer.ipynb
notebooks/data_engineering/3-gold_layer.ipynb
```

### Step 2 — Data Science

```
notebooks/data_science/eda.ipynb
notebooks/data_science/1-create_base_dataset.ipynb
notebooks/data_science/2-data_preparation.ipynb
notebooks/data_science/3-create_features.ipynb
notebooks/data_science/4-weather_features.ipynb   # optional — requires internet
```

### Step 3 — Train Models

```bash
# All demand types, 50 Optuna trials
python src/main.py train

# Quick smoke test (5 trials, smooth only)
python src/main.py train --types smooth --trials 5
```

Or open [run_train.py](run_train.py) and press **Shift+Enter**.

### Step 4 — Generate Final Forecast

```bash
python src/main.py predict
```

Outputs:
- `data/gold/forecasting/predictions/backtest/backtest_ref{date}_run{date}.parquet`
- `data/gold/forecasting/predictions/forecast/forecast_ref{date}_run{date}.parquet`

---

## CLI Reference

```
python src/main.py train   --types [smooth erratic intermittent lumpy]
                           --trials 50        # Optuna trials (LightGBM only)
                           --horizon 4        # weeks ahead
                           --artifacts artifacts/models
                           --predictions data/gold/forecasting/predictions

python src/main.py predict --types [...]
                           --horizon 4
                           --artifacts artifacts/models
                           --predictions data/gold/forecasting/predictions
```

---

## Loading a Saved Model

```python
import sys; sys.path.insert(0, 'src')
from models.base import BaseForecaster

model = BaseForecaster.load('artifacts/models/smooth')
meta  = BaseForecaster.load_metadata('artifacts/models/smooth')
print(meta['model_metrics'])   # {'mase': ..., 'wape': ..., 'delta_wape_pct': 41.16}
```
