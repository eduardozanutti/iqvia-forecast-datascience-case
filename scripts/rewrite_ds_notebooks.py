"""
Rewrites data science notebooks 1-4 to be professional, English-only,
free of redundancies and emojis/icons.

Strategy: generates each notebook cell-by-cell from scratch using
a dedicated cell() helper so we never embed triple-quoted strings
inside other triple-quoted strings.

Run from project root:
    python scripts/rewrite_ds_notebooks.py
"""

import json
from pathlib import Path
from textwrap import dedent


# ── helpers ────────────────────────────────────────────────────────────────

def C(*lines):
    """Code cell. Pass each source line as a separate argument."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "\n".join(lines),
    }

def M(*lines):
    """Markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": "\n".join(lines),
    }

def notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }

def save(nb, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  Written: {path}  ({len(nb['cells'])} cells)")


# ─────────────────────────────────────────────────────────────────────────────
# 1-create_base_dataset.ipynb
# ─────────────────────────────────────────────────────────────────────────────

def nb1():
    D = '    '   # 4-space indent shorthand
    D2 = '        '  # 8-space

    header = M(
        "# Create Base Dataset",
        "",
        "Builds the denormalised base dataset used as the starting point for all",
        "downstream data science steps.",
        "",
        "**Input:** Star-schema tables from `data/gold/data_warehouse/`",
        "**Output:** `data/gold/forecasting/datasets/base/ds_sales_timeseries.parquet`",
        "",
        "**Steps:**",
        "1. Load dimension and fact tables from the data warehouse.",
        "2. Generate a full weekly spine for every time series (no gaps).",
        "3. Enrich with region, product, supplier, and calendar attributes.",
        "4. Optimise column ordering and data types to reduce memory footprint.",
    )

    imports = C(
        "import pandas as pd",
        "import numpy as np",
        "",
        "pd.set_option('display.max_columns', None)",
    )

    load_data = C(
        "# -- 1. Load data warehouse tables ----------------------------------------",
        "DW = '../../data/gold/data_warehouse'",
        "",
        "df_dim_region      = pd.read_parquet(f'{DW}/dw_dim_region.parquet')",
        "df_dim_product     = pd.read_parquet(f'{DW}/dw_dim_product.parquet')",
        "df_dim_supplier    = pd.read_parquet(f'{DW}/dw_dim_supplier.parquet')",
        "df_dim_calendar    = pd.read_parquet(f'{DW}/dw_dim_weekly_calendar.parquet')",
        "df_dim_time_series = pd.read_parquet(f'{DW}/dw_dim_time_series.parquet')",
        "df_fact_sales      = pd.read_parquet(f'{DW}/dw_fact_sales_weekly.parquet')",
        "",
        "print(f'Time series registry : {len(df_dim_time_series):,} series')",
        "print(f'Fact sales           : {len(df_fact_sales):,} rows')",
    )

    spine_fn = C(
        "# -- 2. Generate full weekly spine (one row per series x week) ---------------",
        "def generate_spines_time_series(",
        "    df_fact_sales: 'pd.DataFrame',",
        "    df_dim_time_series: 'pd.DataFrame',",
        ") -> 'pd.DataFrame':",
        "    rows = []",
        "    for _, r in df_dim_time_series.iterrows():",
        "        start = pd.to_datetime(r['first_week_date'])",
        "        end   = pd.to_datetime(r['last_week_date'])",
        "        if pd.isna(start) or pd.isna(end):",
        "            continue",
        "        for d in pd.date_range(start=start, end=end, freq='W-MON'):",
        "            rows.append({'time_series_id': r['time_series_id'], 'week_date': d})",
        "",
        "    df_spine = pd.DataFrame(rows)",
        "    df_meta  = df_dim_time_series.drop(columns=['first_week_date', 'last_week_date'], errors='ignore')",
        "    df_spine = df_spine.merge(df_meta, on='time_series_id', how='left')",
        "",
        "    df_out = df_spine.merge(",
        "        df_fact_sales[['time_series_id', 'week_date', 'units_sold']],",
        "        on=['time_series_id', 'week_date'],",
        "        how='left',",
        "    )",
        "    df_out['units_sold'] = df_out['units_sold'].fillna(0).astype(int)",
        "    return df_out",
        "",
        "",
        "df = generate_spines_time_series(df_fact_sales, df_dim_time_series)",
        "print(f'Spine: {len(df):,} rows  |  {df[\"time_series_id\"].nunique():,} series')",
    )

    enrich_fn = C(
        "# -- 3. Enrich with dimension attributes ------------------------------------",
        "def bring_dimensions(df, df_dim_region, df_dim_product, df_dim_supplier, df_dim_calendar):",
        "    return (",
        "        df",
        "        .merge(df_dim_region,   on='region_id',   how='left')",
        "        .merge(df_dim_product,  on='product_id',  how='left')",
        "        .merge(df_dim_supplier, on='supplier_id', how='left')",
        "        .merge(df_dim_calendar, on='week_date',   how='left')",
        "    )",
        "",
        "",
        "df = bring_dimensions(df, df_dim_region, df_dim_product, df_dim_supplier, df_dim_calendar)",
        "print(f'Shape after enrichment: {df.shape}')",
    )

    optimize_fn = C(
        "# -- 4. Optimise column order and data types --------------------------------",
        "def optimize_base_dataset(df: 'pd.DataFrame') -> 'pd.DataFrame':",
        "    df = df.copy()",
        "    column_order = [",
        "        'time_series_id', 'week_date',",
        "        'supplier_id', 'region_id', 'product_id',",
        "        'units_sold',",
        "        'week', 'start_date', 'end_date', 'year', 'semester', 'semester_date',",
        "        'semester_name', 'quarter', 'quarter_date', 'quarter_name',",
        "        'month', 'month_name', 'month_date',",
        "        'first_week_date', 'last_week_date', 'total_weeks_length',",
        "        'num_week_with_sales', 'num_week_with_zeros', 'sales_weeks_ratio',",
        "        'sales_units', 'avg_weekly_sales', 'avg_weekly_sales_non_zero',",
        "        'std_weekly_sales', 'std_weekly_sales_non_zero',",
        "        'max_weekly_sales', 'min_weekly_sales',",
        "        'q25_sales', 'q50_sales', 'q75_sales', 'iqr', 'cv',",
        "        'supplier_name', 'region_name', 'product_name',",
        "        'product_attribute_1', 'product_attribute_2', 'product_attribute_3',",
        "    ]",
        "    existing_order = [c for c in column_order if c in df.columns]",
        "    remaining      = [c for c in df.columns if c not in existing_order]",
        "    df = df[existing_order + remaining]",
        "",
        "    if 'time_series_id' in df.columns:",
        "        df['time_series_id'] = df['time_series_id'].astype('int32')",
        "    for col in ['supplier_id', 'region_id', 'product_id']:",
        "        if col in df.columns:",
        "            df[col] = df[col].astype('category')",
        "    for col in ['week_date', 'first_week_date', 'last_week_date',",
        "                'start_date', 'end_date', 'semester_date', 'quarter_date', 'month_date']:",
        "        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):",
        "            df[col] = pd.to_datetime(df[col])",
        "    for col in ['year', 'semester', 'quarter', 'month', 'week']:",
        "        if col in df.columns:",
        "            df[col] = df[col].astype('int16')",
        "    for col in ['semester_name', 'quarter_name', 'month_name']:",
        "        if col in df.columns:",
        "            df[col] = df[col].astype('category')",
        "    for col in ['units_sold', 'sales_units', 'max_weekly_sales', 'min_weekly_sales',",
        "                'q25_sales', 'q50_sales', 'q75_sales',",
        "                'num_week_with_sales', 'num_week_with_zeros', 'total_weeks_length']:",
        "        if col in df.columns:",
        "            df[col] = df[col].astype('Int64')",
        "    for col in ['avg_weekly_sales', 'std_weekly_sales', 'cv', 'iqr', 'sales_weeks_ratio']:",
        "        if col in df.columns:",
        "            df[col] = df[col].astype('float32')",
        "    for col in ['supplier_name', 'region_name', 'product_name',",
        "                'product_attribute_1', 'product_attribute_2', 'product_attribute_3']:",
        "        if col in df.columns:",
        "            df[col] = df[col].astype('category')",
        "",
        "    mem_mb = df.memory_usage(deep=True).sum() / 1024**2",
        "    print(f'Shape  : {df.shape}')",
        "    print(f'Memory : {mem_mb:.1f} MB')",
        "    return df",
        "",
        "",
        "df_base = optimize_base_dataset(df)",
    )

    save_cell = C(
        "# -- 5. Save ----------------------------------------------------------------",
        "import os",
        "",
        "OUT_PATH = '../../data/gold/forecasting/datasets/base/ds_sales_timeseries.parquet'",
        "os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)",
        "df_base.to_parquet(OUT_PATH, index=False)",
        "",
        "print(f'Saved  : {OUT_PATH}')",
        "print(f'Rows   : {len(df_base):,}')",
        "print(f'Series : {df_base[\"time_series_id\"].nunique():,}')",
    )

    return notebook([header, imports, load_data, spine_fn, enrich_fn, optimize_fn, save_cell])


# ─────────────────────────────────────────────────────────────────────────────
# 2-data_preparation.ipynb
# ─────────────────────────────────────────────────────────────────────────────

def nb2():
    header = M(
        "# Data Preparation",
        "",
        "Unified preparation pipeline for modelling. Reads the base dataset and produces",
        "demand-type-segmented files ready for feature engineering.",
        "",
        "**Input:**",
        "- `datasets/base/ds_sales_timeseries.parquet` - full weekly base dataset",
        "- `dw_dim_time_series.parquet` - time series registry",
        "",
        "**Output:**",
        "- `datasets/refined/ds_sales_timeseries_{demand_type}.parquet` - one file per demand type",
        "- `datasets/excluded/ds_sales_timeseries_excluded.parquet` - series filtered out",
        "- `tags/ds_tags.parquet` - per-series metadata (segment, demand type, outlier count)",
        "",
        "**Steps:**",
        "1. Maturity segmentation - filter out cold-start and discontinued series.",
        "2. Outlier detection and capping - rolling median + MAD, non-destructive capping.",
        "3. Demand classification - Syntetos-Boylan (ADI x CV2).",
        "4. Save outputs.",
    )

    imports = C(
        "import os",
        "import pandas as pd",
        "import numpy as np",
        "import warnings",
        "warnings.filterwarnings('ignore')",
    )

    paths = C(
        "# -- Paths -----------------------------------------------------------------",
        "PATH_BASE_DATASET = '../../data/gold/forecasting/datasets/base/ds_sales_timeseries.parquet'",
        "PATH_DIM_TS       = '../../data/gold/data_warehouse/dw_dim_time_series.parquet'",
        "",
        "PATH_REFINED  = '../../data/gold/forecasting/datasets/refined'",
        "PATH_EXCLUDED = '../../data/gold/forecasting/datasets/excluded/ds_sales_timeseries_excluded.parquet'",
        "PATH_TAGS     = '../../data/gold/forecasting/tags/ds_tags.parquet'",
        "",
        "os.makedirs(PATH_REFINED, exist_ok=True)",
        "os.makedirs(os.path.dirname(PATH_EXCLUDED), exist_ok=True)",
        "os.makedirs(os.path.dirname(PATH_TAGS), exist_ok=True)",
        "",
        "df_base   = pd.read_parquet(PATH_BASE_DATASET)",
        "df_dim_ts = pd.read_parquet(PATH_DIM_TS)",
        "",
        "print(f'Base dataset : {len(df_base):,} rows  |  {df_base[\"time_series_id\"].nunique():,} series')",
    )

    md_seg = M("## 1. Maturity Segmentation")

    segmentation = C(
        "def segment_by_maturity(df_dim_ts, reference_date=None, discontinued_weeks=26):",
        "    df = df_dim_ts.copy()",
        "    df['last_week_date'] = pd.to_datetime(df['last_week_date'])",
        "",
        "    if reference_date is None:",
        "        reference_date = df['last_week_date'].max()",
        "    reference_date = pd.to_datetime(reference_date)",
        "",
        "    discontinued_threshold = reference_date - pd.Timedelta(weeks=discontinued_weeks)",
        "    df['weeks_since_last_sale'] = ((reference_date - df['last_week_date']).dt.days / 7).astype(int)",
        "    df['segment'] = 'unknown'",
        "",
        "    df.loc[df['total_weeks_length'] < 26, 'segment'] = 'cold_start'",
        "    df.loc[",
        "        (df['segment'] != 'cold_start') & (df['last_week_date'] < discontinued_threshold),",
        "        'segment'",
        "    ] = 'discontinued'",
        "    df.loc[df['segment'] == 'unknown', 'segment'] = 'valid'",
        "",
        "    return df[['time_series_id', 'segment', 'weeks_since_last_sale']]",
        "",
        "",
        "df_segments = segment_by_maturity(df_dim_ts)",
        "valid_ids   = df_segments.loc[df_segments['segment'] == 'valid', 'time_series_id']",
        "df_valid    = df_base[df_base['time_series_id'].isin(valid_ids)].copy()",
        "",
        "print(f'Valid dataset : {df_valid.shape[0]:,} rows  |  {df_valid[\"time_series_id\"].nunique():,} series')",
        "print()",
        "print(df_segments['segment'].value_counts().to_string())",
    )

    md_out = M("## 2. Outlier Detection and Capping")

    outlier = C(
        "def detect_outliers_rolling_median(ts_data, window=13, cutoff_multiplier=3.0):",
        "    n = len(ts_data)",
        "    outlier_mask     = np.zeros(n, dtype=bool)",
        "    rolling_baseline = np.full(n, np.nan)",
        "    rolling_mad      = np.full(n, np.nan)",
        "",
        "    if (ts_data > 0).sum() < window:",
        "        nan = np.full(n, np.nan)",
        "        return outlier_mask, nan, nan, nan",
        "",
        "    for i in range(n):",
        "        start = max(0, i - window // 2)",
        "        end   = min(n, i + window // 2 + 1)",
        "        w = ts_data[start:end]",
        "        w = w[w > 0]",
        "        if len(w) < 3:",
        "            continue",
        "        baseline            = np.median(w)",
        "        rolling_baseline[i] = baseline",
        "        rolling_mad[i]      = np.median(np.abs(w - baseline))",
        "",
        "    rolling_std = 1.4826 * rolling_mad",
        "    upper       = rolling_baseline + cutoff_multiplier * rolling_std",
        "    lower       = np.zeros(n)",
        "",
        "    for i in range(n):",
        "        if ts_data[i] > 0 and not np.isnan(upper[i]):",
        "            if ts_data[i] > upper[i] or ts_data[i] < lower[i]:",
        "                outlier_mask[i] = True",
        "",
        "    return outlier_mask, rolling_baseline, rolling_std, upper",
        "",
        "",
        "def run_outlier_pipeline(df, window=13, cutoff_multiplier=3.0):",
        "    results = []",
        "    for ts_id, group in df.groupby('time_series_id'):",
        "        group = group.sort_values('week_date').copy()",
        "        y     = group['units_sold'].astype(float).values",
        "        out_mask, baseline, roll_std, upper = detect_outliers_rolling_median(",
        "            y, window=window, cutoff_multiplier=cutoff_multiplier",
        "        )",
        "        y_cap = y.copy()",
        "        valid_upper = ~np.isnan(upper)",
        "        y_cap[out_mask & valid_upper] = upper[out_mask & valid_upper]",
        "",
        "        group['is_outlier']       = out_mask",
        "        group['rolling_baseline'] = baseline",
        "        group['rolling_std']      = roll_std",
        "        group['upper_threshold']  = upper",
        "        group['lower_threshold']  = 0.0",
        "        group['y_capped']         = y_cap.astype(int)",
        "        results.append(group)",
        "    return pd.concat(results, ignore_index=True)",
        "",
        "",
        "df_treated = run_outlier_pipeline(df_valid, window=13, cutoff_multiplier=3.0)",
        "",
        "n_outliers = df_treated['is_outlier'].sum()",
        "n_series   = df_treated['time_series_id'].nunique()",
        "print(f'Outliers detected : {n_outliers:,}  ({n_outliers / len(df_treated) * 100:.2f}% of all observations)')",
        "print(f'Series affected   : {df_treated.groupby(\"time_series_id\")[\"is_outlier\"].any().sum():,} / {n_series:,}')",
    )

    md_cls = M("## 3. Demand Classification (Croston)")

    classify = C(
        "def classify_series(y: 'pd.Series') -> str:",
        "    non_zero = y[y > 0]",
        "    if len(non_zero) == 0:",
        "        return 'lumpy'",
        "    adi = len(y) / len(non_zero)",
        "    cv2 = (non_zero.std(ddof=0) / non_zero.mean()) ** 2 if non_zero.mean() != 0 else 0",
        "    if adi < 1.32:",
        "        return 'smooth' if cv2 < 0.49 else 'erratic'",
        "    return 'intermittent' if cv2 < 0.49 else 'lumpy'",
        "",
        "",
        "demand_type_map = (",
        "    df_treated",
        "    .groupby('time_series_id')['y_capped']",
        "    .apply(classify_series)",
        ")",
        "df_treated['demand_type'] = df_treated['time_series_id'].map(demand_type_map)",
        "",
        "print('Demand classification:')",
        "print(df_treated.groupby('demand_type')['time_series_id'].nunique()",
        "      .rename('n_series').sort_values(ascending=False).to_string())",
    )

    md_save = M("## 4. Save Outputs")

    save_refined = C(
        "# 4.1 Refined datasets - standard time series format: unique_id | ds | y",
        "DEMAND_TYPES = ['smooth', 'erratic', 'intermittent', 'lumpy']",
        "",
        "for demand_type in DEMAND_TYPES:",
        "    df_out = (",
        "        df_treated[df_treated['demand_type'] == demand_type]",
        "        [['time_series_id', 'week_date', 'y_capped', 'demand_type']]",
        "        .rename(columns={'time_series_id': 'unique_id', 'week_date': 'ds', 'y_capped': 'y'})",
        "        .sort_values(['unique_id', 'ds'])",
        "        .reset_index(drop=True)",
        "    )",
        "    path = f'{PATH_REFINED}/ds_sales_timeseries_{demand_type}.parquet'",
        "    df_out.to_parquet(path, index=False)",
        "    print(f'  Saved {demand_type:<15}: {df_out[\"unique_id\"].nunique():,} series  |  {len(df_out):,} rows')",
    )

    save_excluded = C(
        "# 4.2 Excluded series",
        "excluded_ids = df_segments[df_segments['segment'] != 'valid'][",
        "    ['time_series_id', 'segment', 'weeks_since_last_sale']",
        "]",
        "df_excluded = (",
        "    df_base[df_base['time_series_id'].isin(excluded_ids['time_series_id'])]",
        "    .merge(excluded_ids, on='time_series_id', how='left')",
        ")",
        "df_excluded.to_parquet(PATH_EXCLUDED, index=False)",
        "print(f'Excluded : {df_excluded[\"time_series_id\"].nunique():,} series  ->  {PATH_EXCLUDED}')",
    )

    save_tags = C(
        "# 4.3 Tags - per-series metadata for all series (valid and excluded)",
        "outlier_counts = (",
        "    df_treated.groupby('time_series_id')['is_outlier']",
        "    .sum()",
        "    .rename('n_outliers')",
        "    .reset_index()",
        ")",
        "df_tags = (",
        "    df_segments",
        "    .merge(outlier_counts, on='time_series_id', how='left')",
        "    .merge(",
        "        df_treated[['time_series_id', 'demand_type']].drop_duplicates(),",
        "        on='time_series_id', how='left',",
        "    )",
        ")",
        "df_tags.to_parquet(PATH_TAGS, index=False)",
        "print(f'Tags     : {len(df_tags):,} series  ->  {PATH_TAGS}')",
    )

    md_summ = M("## Summary")

    summary = C(
        "print('OUTPUTS GENERATED')",
        "print('=' * 60)",
        "print()",
        "print('  datasets/refined/')",
        "for dt in DEMAND_TYPES:",
        "    n = df_treated[df_treated['demand_type'] == dt]['time_series_id'].nunique()",
        "    print(f'    ds_sales_timeseries_{dt}.parquet   {n:>5,} series')",
        "print()",
        "print('  datasets/excluded/')",
        "print(f'    ds_sales_timeseries_excluded.parquet   '",
        "      f'{df_excluded[\"time_series_id\"].nunique():>5,} series')",
        "print()",
        "print('  tags/')",
        "print(f'    ds_tags.parquet   {len(df_tags):,} total series')",
    )

    return notebook([
        header, imports, paths,
        md_seg, segmentation,
        md_out, outlier,
        md_cls, classify,
        md_save, save_refined, save_excluded, save_tags,
        md_summ, summary,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 3-create_features.ipynb
# ─────────────────────────────────────────────────────────────────────────────

def nb3():
    header = M(
        "# Feature Engineering",
        "",
        "Generates two feature artefacts consumed by the model building step.",
        "",
        "| File | Key | Content |",
        "|---|---|---|",
        "| `features/ds_features.parquet` | `unique_id x ds` | Cyclic calendar encodings, product attributes, entity IDs |",
        "| `features/ds_holidays.parquet` | `ds x region_id` | National and regional holiday flags, type, proximity |",
        "",
        "**Static features** (`ds_features`): constant per `unique_id`, valid for any forecast week.",
        "**Holiday features** (`ds_holidays`): dynamic per `ds`, vary by macro-region (5 regions -> Brazilian states).",
        "",
        "Distribution statistics (cv, iqr, q50) are computed in model building after the train/test split",
        "to avoid data leakage.",
    )

    imports = C(
        "import os",
        "import unicodedata",
        "import pandas as pd",
        "import numpy as np",
        "import warnings",
        "from holidays import country_holidays",
        "warnings.filterwarnings('ignore')",
    )

    paths = C(
        "# -- Paths -----------------------------------------------------------------",
        "PATH_REFINED    = '../../data/gold/forecasting/datasets/refined'",
        "PATH_BASE       = '../../data/gold/forecasting/datasets/base/ds_sales_timeseries.parquet'",
        "PATH_DIM_REGION = '../../data/gold/data_warehouse/dw_dim_region.parquet'",
        "",
        "PATH_FEATURES   = '../../data/gold/forecasting/features/ds_features.parquet'",
        "PATH_HOLIDAYS   = '../../data/gold/forecasting/features/ds_holidays.parquet'",
        "os.makedirs(os.path.dirname(PATH_FEATURES), exist_ok=True)",
        "",
        "DEMAND_TYPES = ['smooth', 'erratic', 'intermittent', 'lumpy']",
    )

    md_load = M("## 1. Load Refined Series")

    load_series = C(
        "df_valid = pd.concat([",
        "    pd.read_parquet(f'{PATH_REFINED}/ds_sales_timeseries_{dt}.parquet')",
        "    for dt in DEMAND_TYPES",
        "], ignore_index=True)",
        "",
        "print(f'Valid series : {df_valid[\"unique_id\"].nunique():,}')",
        "print(f'Total rows   : {len(df_valid):,}')",
    )

    md_attrs = M("## 2. Load Static Attributes")

    load_attrs = C(
        "# Load only attribute columns - stats excluded to prevent leakage",
        "BASE_COLS = [",
        "    'time_series_id', 'week_date',",
        "    'week', 'month', 'quarter', 'year',",
        "    'product_attribute_1', 'product_attribute_2', 'product_attribute_3',",
        "    'supplier_id', 'region_id',",
        "]",
        "df_base = pd.read_parquet(PATH_BASE, columns=BASE_COLS)",
        "print(f'Base attributes loaded: {df_base.shape}')",
    )

    md_build = M("## 3. Build Features")

    build_feats = C(
        "def build_calendar_features(df: 'pd.DataFrame') -> 'pd.DataFrame':",
        "    # Cyclic encoding: preserves circular proximity (month 12 near month 1)",
        "    df['month_sin']   = np.sin(2 * np.pi * df['month']   / 12)",
        "    df['month_cos']   = np.cos(2 * np.pi * df['month']   / 12)",
        "    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)",
        "    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)",
        "    df['week_sin']    = np.sin(2 * np.pi * df['week']    / 52)",
        "    df['week_cos']    = np.cos(2 * np.pi * df['week']    / 52)",
        "    return df.drop(columns=['week', 'month', 'quarter'])  # year kept as numeric",
        "",
        "",
        "def build_product_features(df: 'pd.DataFrame') -> 'pd.DataFrame':",
        "    for col in ['product_attribute_1', 'product_attribute_2', 'product_attribute_3']:",
        "        df[col] = df[col].astype('category').cat.codes",
        "    return df",
        "",
        "",
        "def build_features(df_valid: 'pd.DataFrame', df_base: 'pd.DataFrame') -> 'pd.DataFrame':",
        "    df = df_valid[['unique_id', 'ds']].copy()",
        "    df = df.merge(",
        "        df_base.rename(columns={'time_series_id': 'unique_id', 'week_date': 'ds'}),",
        "        on=['unique_id', 'ds'], how='left',",
        "    )",
        "    df = build_calendar_features(df)",
        "    df = build_product_features(df)",
        "    return df",
        "",
        "",
        "df_features = build_features(df_valid, df_base)",
        "",
        "print(f'Shape   : {df_features.shape}')",
        "print(f'Columns : {df_features.columns.tolist()}')",
        "if df_features.isna().any().any():",
        "    print('NaN (%) :')",
        "    print(df_features.isna().mean().sort_values(ascending=False).head(5))",
    )

    md_save_feats = M("## 4. Save -- Static Features")

    save_feats = C(
        "df_features.to_parquet(PATH_FEATURES, index=False)",
        "",
        "cal_cols  = [c for c in df_features.columns if any(s in c for s in ['sin', 'cos', 'year'])]",
        "prod_cols = [c for c in df_features.columns if 'attribute' in c]",
        "id_cols   = ['supplier_id', 'region_id']",
        "",
        "print(f'Saved   : {PATH_FEATURES}')",
        "print(f'Shape   : {df_features.shape}')",
        "print(f'Calendar: {cal_cols}')",
        "print(f'Product : {prod_cols}')",
        "print(f'IDs     : {id_cols}')",
    )

    md_holidays = M(
        "## 5. Holiday Features",
        "",
        "Holiday features keyed by `(ds, region_id)`.",
        "",
        "**Macro-region to states:**",
        "",
        "| region_id | Region | States |",
        "|---|---|---|",
        "| 1 | Centro-Oeste | DF, GO, MS, MT |",
        "| 2 | Nordeste | AL, BA, CE, MA, PB, PE, PI, RN, SE |",
        "| 3 | Norte | AC, AM, AP, PA, RO, RR, TO |",
        "| 4 | Sudeste | ES, MG, RJ, SP |",
        "| 5 | Sul | PR, RS, SC |",
        "",
        "**Columns generated:**",
        "- `is_national_holiday` - national holiday falls within the week",
        "- `is_regional_holiday` - state-specific holiday exclusive to this region",
        "- `is_holiday` - any holiday (national or regional)",
        "- `n_holidays` - total holidays in the week",
        "- `holiday_type` - dominant type: `none / regional / other / christmas_newyear / easter / carnival`",
        "- `holiday_type_enc` - ordinal encoding by expected demand impact (0=none to 5=carnival)",
        "- `weeks_to_next_holiday` - weeks to the next holiday, capped at 4",
        "- `weeks_since_last_holiday` - weeks since the last holiday, capped at 4",
    )

    build_hols = C(
        "REGION_TO_STATES = {",
        "    1: ['DF', 'GO', 'MS', 'MT'],",
        "    2: ['AL', 'BA', 'CE', 'MA', 'PB', 'PE', 'PI', 'RN', 'SE'],",
        "    3: ['AC', 'AM', 'AP', 'PA', 'RO', 'RR', 'TO'],",
        "    4: ['ES', 'MG', 'RJ', 'SP'],",
        "    5: ['PR', 'RS', 'SC'],",
        "}",
        "",
        "# Priority ordered by expected pharmaceutical demand impact",
        "HOLIDAY_TYPE_KEYWORDS = [",
        "    ('carnival',          ['carnaval', 'quaresma']),",
        "    ('easter',            ['sexta-feira santa', 'corpus christi']),",
        "    ('christmas_newyear', ['natal', 'ano-novo', 'confraterniza']),",
        "    ('other',             []),",
        "]",
        "HOLIDAY_TYPE_ENC = {",
        "    'none': 0, 'regional': 1, 'other': 2,",
        "    'christmas_newyear': 3, 'easter': 4, 'carnival': 5,",
        "}",
        "",
        "",
        "def _strip_accents(text: str) -> str:",
        "    nfkd = unicodedata.normalize('NFD', text.lower())",
        "    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')",
        "",
        "",
        "def classify_holiday_type(names: list) -> str:",
        "    combined = _strip_accents(' '.join(names))",
        "    for htype, keywords in HOLIDAY_TYPE_KEYWORDS:",
        "        if keywords and any(k in combined for k in keywords):",
        "            return htype",
        "    return 'other' if names else 'none'",
        "",
        "",
        "def build_holiday_features(all_weeks: 'pd.Series') -> 'pd.DataFrame':",
        "    years    = all_weeks.dt.year.unique().tolist()",
        "    all_rows = []",
        "",
        "    for region_id, states in REGION_TO_STATES.items():",
        "        national_hols = country_holidays('BR', years=years)",
        "        regional_hols = set()",
        "        for state in states:",
        "            try:",
        "                state_hols = country_holidays('BR', subdiv=state, years=years)",
        "                for d, name in state_hols.items():",
        "                    if d not in national_hols:",
        "                        regional_hols.add((d, name))",
        "            except Exception:",
        "                pass",
        "",
        "        for week_start in all_weeks:",
        "            week_days = pd.date_range(week_start, periods=7, freq='D')",
        "            national_in_week = [(d.date(), n) for d, n in national_hols.items()",
        "                                if d in week_days.date]",
        "            regional_in_week = [(d, n) for d, n in regional_hols",
        "                                if d in week_days.date]",
        "",
        "            is_national = len(national_in_week) > 0",
        "            is_regional = len(regional_in_week) > 0",
        "            is_holiday  = is_national or is_regional",
        "            n_holidays  = len(national_in_week) + len(regional_in_week)",
        "",
        "            if is_national:",
        "                names = [n for _, n in national_in_week]",
        "                htype = classify_holiday_type(names)",
        "            elif is_regional:",
        "                htype = 'regional'",
        "            else:",
        "                htype = 'none'",
        "",
        "            all_rows.append({",
        "                'ds':                  week_start,",
        "                'region_id':           region_id,",
        "                'is_national_holiday': is_national,",
        "                'is_regional_holiday': is_regional,",
        "                'is_holiday':          is_holiday,",
        "                'n_holidays':          n_holidays,",
        "                'holiday_type':        htype,",
        "                'holiday_type_enc':    HOLIDAY_TYPE_ENC.get(htype, 0),",
        "            })",
        "",
        "    df_h = pd.DataFrame(all_rows).sort_values(['ds', 'region_id']).reset_index(drop=True)",
        "",
        "    # Proximity features",
        "    for region_id in df_h['region_id'].unique():",
        "        mask   = df_h['region_id'] == region_id",
        "        df_r   = df_h[mask].copy().reset_index()",
        "        hol_idx = df_r.index[df_r['is_holiday']].tolist()",
        "        to_next    = []",
        "        since_last = []",
        "        for i in range(len(df_r)):",
        "            future = [j - i for j in hol_idx if j >= i]",
        "            past   = [i - j for j in hol_idx if j <= i]",
        "            to_next.append(min(future) if future else 4)",
        "            since_last.append(min(past) if past else 4)",
        "        df_h.loc[mask, 'weeks_to_next_holiday']    = [min(v, 4) for v in to_next]",
        "        df_h.loc[mask, 'weeks_since_last_holiday'] = [min(v, 4) for v in since_last]",
        "",
        "    return df_h",
        "",
        "",
        "all_weeks   = df_valid['ds'].drop_duplicates().sort_values().reset_index(drop=True)",
        "df_holidays = build_holiday_features(all_weeks)",
        "",
        "print(f'Shape   : {df_holidays.shape}')",
        "print(f'Regions : {sorted(df_holidays[\"region_id\"].unique())}')",
        "print(f'Weeks   : {df_holidays[\"ds\"].nunique()}')",
        "print()",
        "print('Holidays by type (region 4 - Sudeste):')",
        "print(df_holidays[df_holidays['region_id'] == 4]",
        "      .groupby('holiday_type')['is_holiday'].sum()",
        "      .sort_values(ascending=False).to_string())",
    )

    md_save_hols = M("## 6. Save -- Holiday Features")

    save_hols = C(
        "df_holidays.to_parquet(PATH_HOLIDAYS, index=False)",
        "",
        "print(f'Saved : {PATH_HOLIDAYS}')",
        "print(f'Shape : {df_holidays.shape}')",
        "print()",
        "print('Column groups:')",
        "print('  Key         : [\"ds\", \"region_id\"]')",
        "print('  Flags       : [\"is_national_holiday\", \"is_regional_holiday\", \"is_holiday\", \"n_holidays\"]')",
        "print('  Type        : [\"holiday_type\", \"holiday_type_enc\"]')",
        "print('  Proximity   : [\"weeks_to_next_holiday\", \"weeks_since_last_holiday\"]')",
    )

    return notebook([
        header, imports, paths,
        md_load, load_series,
        md_attrs, load_attrs,
        md_build, build_feats,
        md_save_feats, save_feats,
        md_holidays, build_hols,
        md_save_hols, save_hols,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 4-weather_features.ipynb
# ─────────────────────────────────────────────────────────────────────────────

def nb4():
    header = M(
        "# Weather Features -- Open-Meteo (52-Week Lag)",
        "",
        "Generates `features/ds_weather.parquet` with historical climate variables per macro-region.",
        "",
        "## Design: why a 52-week lag?",
        "",
        "| Scenario | Problem |",
        "|---|---|",
        "| Real future climate | Unavailable - Open-Meteo forecast covers only **16 days** (< 4-week horizon) |",
        "| Current climate directly | Creates leakage: test and prediction sets would not have this data in production |",
        "| **Climate lagged 52 weeks** | Always available: uses the **same period from the prior year** as a seasonal proxy |",
        "",
        "**Flow:**",
        "```",
        "Dataset range   : Oct/2022 -> Oct/2024",
        "API fetch       : Oct/2021 -> Oct/2023  (52 weeks earlier)",
        "After +52w shift: Oct/2022 -> Oct/2024  (aligned with series)",
        "Future forecast : Nov/2024 uses Nov/2023 climate  (historical data available)",
        "```",
        "",
        "**Limitation:** does not capture inter-annual anomalies (El Nino, exceptional heatwaves).",
        "For pharma demand, the seasonal pattern dominates -- this trade-off is acceptable.",
    )

    imports = C(
        "import os",
        "import time",
        "import warnings",
        "import requests",
        "import pandas as pd",
        "import numpy as np",
        "warnings.filterwarnings('ignore')",
        "",
        "import sys",
        "sys.path.insert(0, '../../src')",
        "from config import LoadConfig",
        "cfg = LoadConfig()",
    )

    config = C(
        "# -- Config ----------------------------------------------------------------",
        "PATH_WEATHER  = '../../data/gold/forecasting/features/ds_weather.parquet'",
        "os.makedirs(os.path.dirname(PATH_WEATHER), exist_ok=True)",
        "",
        "LAG_WEEKS     = 52   # temporal shift applied after fetching",
        "HORIZON_WEEKS = 4    # extra weeks to ensure future coverage",
        "DEMAND_TYPES  = ['smooth', 'erratic', 'intermittent', 'lumpy']",
        "",
        "# Representative coordinates per macro-region (capital or most populous city)",
        "REGION_COORDS = {",
        "    1: {'name': 'Centro-Oeste', 'lat': -15.78, 'lon': -47.93},  # Brasilia/DF",
        "    2: {'name': 'Nordeste',     'lat':  -8.05, 'lon': -34.90},  # Recife/PE",
        "    3: {'name': 'Norte',        'lat':  -3.10, 'lon': -60.02},  # Manaus/AM",
        "    4: {'name': 'Sudeste',      'lat': -23.55, 'lon': -46.63},  # Sao Paulo/SP",
        "    5: {'name': 'Sul',          'lat': -25.43, 'lon': -49.27},  # Curitiba/PR",
        "}",
        "",
        "DAILY_VARS = [",
        "    'temperature_2m_max',",
        "    'temperature_2m_min',",
        "    'temperature_2m_mean',",
        "    'precipitation_sum',",
        "    'windspeed_10m_max',",
        "]",
    )

    md_range = M("## 1. Determine Date Range")

    date_range = C(
        "# Read the actual week dates from the refined series",
        "all_weeks = pd.concat([",
        "    cfg.load_forecast('datasets', 'refined', dt)[['unique_id', 'ds']]",
        "    for dt in DEMAND_TYPES",
        "])['ds'].drop_duplicates().sort_values()",
        "",
        "ds_min = all_weeks.min()",
        "ds_max = all_weeks.max()",
        "",
        "# Fetch range = [ds_min - LAG_WEEKS, ds_max + HORIZON_WEEKS - LAG_WEEKS]",
        "# After the +LAG_WEEKS shift this covers [ds_min, ds_max + HORIZON_WEEKS]",
        "fetch_start = ds_min - pd.Timedelta(weeks=LAG_WEEKS)",
        "fetch_end   = ds_max + pd.Timedelta(weeks=HORIZON_WEEKS) - pd.Timedelta(weeks=LAG_WEEKS)",
        "",
        "print(f'Dataset        : {ds_min.date()}  ->  {ds_max.date()}')",
        "print(f'API fetch      : {fetch_start.date()}  ->  {fetch_end.date()}')",
        "print(f'After +{LAG_WEEKS}w shift : '",
        "      f'{(fetch_start + pd.Timedelta(weeks=LAG_WEEKS)).date()}  ->  '",
        "      f'{(fetch_end + pd.Timedelta(weeks=LAG_WEEKS)).date()}')",
        "print(f'Unique weeks   : {len(all_weeks):,}')",
    )

    md_fetch = M("## 2. Fetch Open-Meteo Historical Archive")

    fetch_fns = C(
        "ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'",
        "",
        "",
        "def fetch_daily_weather(lat, lon, start, end, variables, retries=3):",
        "    # Free service (no API key) with a soft rate limit",
        "    params = {",
        "        'latitude':   lat,",
        "        'longitude':  lon,",
        "        'start_date': start,",
        "        'end_date':   end,",
        "        'daily':      ','.join(variables),",
        "        'timezone':   'America/Sao_Paulo',",
        "    }",
        "    for attempt in range(retries):",
        "        try:",
        "            r = requests.get(ARCHIVE_URL, params=params, timeout=30)",
        "            r.raise_for_status()",
        "            data = r.json()['daily']",
        "            df = pd.DataFrame(data)",
        "            df['date'] = pd.to_datetime(df['time'])",
        "            return df.drop(columns='time').set_index('date')",
        "        except requests.HTTPError:",
        "            if r.status_code == 429:",
        "                time.sleep(2 ** attempt)",
        "            else:",
        "                raise",
        "    raise RuntimeError(f'Failed after {retries} retries')",
        "",
        "",
        "def aggregate_to_weekly(df_daily: 'pd.DataFrame') -> 'pd.DataFrame':",
        "    df = df_daily.resample('W-MON').agg({",
        "        'temperature_2m_max':  'max',",
        "        'temperature_2m_min':  'min',",
        "        'temperature_2m_mean': 'mean',",
        "        'precipitation_sum':   'sum',",
        "        'windspeed_10m_max':   'max',",
        "    }).reset_index().rename(columns={",
        "        'date':                'ds',",
        "        'temperature_2m_max':  'temp_max',",
        "        'temperature_2m_min':  'temp_min',",
        "        'temperature_2m_mean': 'temp_mean',",
        "        'precipitation_sum':   'precip',",
        "        'windspeed_10m_max':   'wind_max',",
        "    })",
        "    df['is_cold']  = (df['temp_mean'] < 15).astype(int)",
        "    df['is_hot']   = (df['temp_mean'] > 30).astype(int)",
        "    df['is_rainy'] = (df['precip'] > 50).astype(int)",
        "    return df",
    )

    fetch_call = C(
        "start_str = fetch_start.strftime('%Y-%m-%d')",
        "end_str   = fetch_end.strftime('%Y-%m-%d')",
        "",
        "raw_by_region = {}",
        "",
        "for region_id, info in REGION_COORDS.items():",
        "    print(f'Fetching region {region_id} - {info[\"name\"]} ({info[\"lat\"]}, {info[\"lon\"]})...', end=' ')",
        "    df_daily  = fetch_daily_weather(info['lat'], info['lon'], start_str, end_str, DAILY_VARS)",
        "    df_weekly = aggregate_to_weekly(df_daily)",
        "    df_weekly['region_id'] = region_id",
        "    raw_by_region[region_id] = df_weekly",
        "    print(f'{len(df_weekly)} weeks OK')",
        "    time.sleep(0.5)  # respect soft rate limit",
        "",
        "df_raw = pd.concat(raw_by_region.values(), ignore_index=True)",
        "print(f'\\nTotal: {len(df_raw):,} rows  |  columns: {df_raw.columns.tolist()}')",
    )

    md_shift = M(
        "## 3. Apply 52-Week Lag Shift",
        "",
        "```",
        "weather_lag52[t] = actual climate at (t - 52 weeks)",
        "```",
        "",
        "Advancing the raw dates forward by 52 weeks aligns each row with the",
        "corresponding period in the model's training and prediction range.",
    )

    shift = C(
        "WEATHER_COLS = ['temp_max', 'temp_min', 'temp_mean', 'precip', 'wind_max',",
        "                'is_cold', 'is_hot', 'is_rainy']",
        "",
        "df_weather = df_raw.copy()",
        "",
        "# Advance dates by 52 weeks -> align with the model period",
        "df_weather['ds'] = df_weather['ds'] + pd.Timedelta(weeks=LAG_WEEKS)",
        "",
        "# Keep only weeks within [ds_min, ds_max + horizon]",
        "valid_range_end = ds_max + pd.Timedelta(weeks=HORIZON_WEEKS)",
        "df_weather = df_weather[",
        "    (df_weather['ds'] >= ds_min) &",
        "    (df_weather['ds'] <= valid_range_end)",
        "].reset_index(drop=True)",
        "",
        "print(f'After shift: {len(df_weather):,} rows')",
        "print(f'Date range : {df_weather[\"ds\"].min().date()}  ->  {df_weather[\"ds\"].max().date()}')",
    )

    md_val = M("## 4. Validation")

    validation = C(
        "import matplotlib.pyplot as plt",
        "",
        "fig, axes = plt.subplots(2, 3, figsize=(16, 7))",
        "",
        "plot_vars = [",
        "    ('temp_mean', 'Mean Temperature (C)', 'tomato'),",
        "    ('precip',    'Precipitation (mm)',    'steelblue'),",
        "    ('wind_max',  'Max Wind Speed (km/h)', 'seagreen'),",
        "]",
        "",
        "for col_idx, (var, label, colour) in enumerate(plot_vars):",
        "    for row_idx, region_id in enumerate([4, 2]):  # Sudeste and Nordeste",
        "        ax  = axes[row_idx][col_idx]",
        "        sub = df_weather[df_weather['region_id'] == region_id].sort_values('ds')",
        "        ax.plot(sub['ds'], sub[var], color=colour, linewidth=0.8)",
        "        region_name = REGION_COORDS[region_id]['name']",
        "        ax.set_title(f'{label} - {region_name}', fontsize=9)",
        "        ax.tick_params(labelsize=8)",
        "",
        "fig.suptitle('Weather Features -- 52-week lag applied', fontsize=11)",
        "plt.tight_layout()",
        "plt.show()",
    )

    md_save = M("## 5. Save")

    save_cell = C(
        "df_weather.to_parquet(PATH_WEATHER, index=False)",
        "",
        "print(f'Saved  : {PATH_WEATHER}')",
        "print(f'Shape  : {df_weather.shape}')",
        "print(f'Columns: {df_weather.columns.tolist()}')",
        "print(f'Regions: {sorted(df_weather[\"region_id\"].unique())}')",
        "print(f'Weeks  : {df_weather[\"ds\"].nunique():,}')",
    )

    return notebook([
        header, imports, config,
        md_range, date_range,
        md_fetch, fetch_fns, fetch_call,
        md_shift, shift,
        md_val, validation,
        md_save, save_cell,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    base = Path('notebooks/data_science')
    print('Rewriting data science notebooks...')
    save(nb1(), base / '1-create_base_dataset.ipynb')
    save(nb2(), base / '2-data_preparation.ipynb')
    save(nb3(), base / '3-create_features.ipynb')
    save(nb4(), base / '4-weather_features.ipynb')
    print('Done.')
