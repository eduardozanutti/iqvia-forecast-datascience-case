"""Rewrites the three data-engineering notebooks with clean English comments."""
import json
from pathlib import Path

BASE = Path('c:/Users/eduar/Documents/iqvia-forecast-datascience-case/notebooks/data_engineering')

def md(source_lines, cell_id):
    return {'cell_type': 'markdown', 'id': cell_id, 'metadata': {}, 'source': source_lines}

def code(source_lines, cell_id):
    return {
        'cell_type': 'code', 'id': cell_id, 'metadata': {},
        'source': source_lines, 'outputs': [], 'execution_count': None,
    }

def nb(cells):
    return {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': cells,
    }

def save(notebook, name):
    path = BASE / name
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f'Saved: {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 1 — BRONZE LAYER
# ══════════════════════════════════════════════════════════════════════════════
bronze = nb([
    md([
        '# Bronze Layer — Raw Ingestion\n',
        '\n',
        '**Objective:** Ingest the raw CSV file, enforce a consistent schema '
        '(column names + data types), and persist the result as Parquet for downstream processing.\n',
        '\n',
        '**Input :** `data/raw/dataset-case-iqvia.csv`  \n',
        '**Output:** `data/bronze/iqvia-sales_data.parquet`',
    ], 'b_00'),

    code(['import pandas as pd\n'], 'b_01'),

    code([
        '# ── 1. Load raw CSV ──────────────────────────────────────────────────────────\n',
        "df = pd.read_csv('../../data/raw/dataset-case-iqvia.csv')\n",
        'print(f"{len(df):,} rows  |  {df.shape[1]} columns")\n',
        'df.head()',
    ], 'b_02'),

    code([
        '# ── 2. Standardise column names ──────────────────────────────────────────────\n',
        'df.columns = [\n',
        "    'week_date', 'supplier_id', 'product_id', 'region',\n",
        "    'units_sold', 'product_attribute_1', 'product_attribute_2', 'product_attribute_3',\n",
        ']\n',
    ], 'b_03'),

    code([
        '# ── 3. Enforce data types ────────────────────────────────────────────────────\n',
        "df['week_date']   = pd.to_datetime(df['week_date'], format='%Y-%m-%d')\n",
        "df['supplier_id'] = df['supplier_id'].astype('object')\n",
        "df['product_id']  = df['product_id'].astype('object')\n",
        "df['units_sold']  = df['units_sold'].astype('int')\n",
        '\n',
        'df.dtypes\n',
    ], 'b_04'),

    code([
        '# ── 4. Persist to bronze ─────────────────────────────────────────────────────\n',
        "df.to_parquet('../../data/bronze/iqvia-sales_data.parquet', index=False)\n",
        "print('Saved: data/bronze/iqvia-sales_data.parquet')\n",
    ], 'b_05'),
])

save(bronze, '1-bronze_layer.ipynb')


# ══════════════════════════════════════════════════════════════════════════════
# 2 — SILVER LAYER
# ══════════════════════════════════════════════════════════════════════════════
silver = nb([
    md([
        '# Silver Layer — Dimensional Modelling\n',
        '\n',
        '**Objective:** Transform the bronze flat file into a star-schema data model.\n',
        'Four dimension tables (calendar, supplier, region, product) and one fact table\n',
        '(weekly sales) are created, cleaned, and saved as Parquet.\n',
        '\n',
        '**Input :** `data/bronze/iqvia-sales_data.parquet`  \n',
        '**Output:** `data/silver/dim_*.parquet` + `data/silver/fact_sales_weekly.parquet`',
    ], 's_00'),

    code(['import pandas as pd\n'], 's_01'),

    code([
        '# ── Load bronze ──────────────────────────────────────────────────────────────\n',
        "df = pd.read_parquet('../../data/bronze/iqvia-sales_data.parquet')\n",
        'print(f"{len(df):,} rows  |  '
        'date range: {df[chr(34)+chr(119)+chr(101)+chr(101)+chr(107)+chr(95)+chr(100)+chr(97)+chr(116)+chr(101)+chr(34)].min().date()} to {df[chr(34)+chr(119)+chr(101)+chr(101)+chr(107)+chr(95)+chr(100)+chr(97)+chr(116)+chr(101)+chr(34)].max().date()}")\n',
    ], 's_02'),

    md([
        '## dim_calendar\n',
        '\n',
        'Daily calendar spanning the full date range, enriched with temporal attributes\n',
        '(year, semester, quarter, month, week). Week dates are anchored to Monday\n',
        'for unambiguous weekly joins.\n',
    ], 's_03'),

    code([
        '# Build daily spine and derive all temporal attributes\n',
        "start_date = df['week_date'].dt.to_period('W').apply(lambda r: r.start_time).min()\n",
        '\n',
        "df_dim_time = pd.DataFrame({'date': pd.date_range(start=start_date, end=df['week_date'].max(), freq='D')})\n",
        "df_dim_time['year']             = df_dim_time['date'].dt.year\n",
        "df_dim_time['semester']         = df_dim_time['date'].dt.month.apply(lambda x: 1 if x <= 6 else 2)\n",
        "df_dim_time['semester_date']    = pd.to_datetime(df_dim_time['date'].dt.to_period('6M').apply(lambda r: r.start_time))\n",
        "df_dim_time['semester_name']    = df_dim_time['semester'].apply(lambda x: 'H1' if x == 1 else 'H2')\n",
        "df_dim_time['quarter']          = df_dim_time['date'].dt.quarter\n",
        "df_dim_time['quarter_date']     = pd.to_datetime(df_dim_time['date'].dt.to_period('Q').apply(lambda r: r.start_time))\n",
        "df_dim_time['quarter_name']     = df_dim_time['quarter'].apply(lambda x: f'Q{x}')\n",
        "df_dim_time['month']            = df_dim_time['date'].dt.month\n",
        "df_dim_time['month_name']       = df_dim_time['date'].dt.month_name()\n",
        "df_dim_time['month_date']       = pd.to_datetime(df_dim_time['date'].dt.to_period('M').apply(lambda r: r.start_time))\n",
        "df_dim_time['day']              = df_dim_time['date'].dt.day\n",
        "df_dim_time['week']             = df_dim_time['date'].dt.isocalendar().week\n",
        "df_dim_time['week_date']        = pd.to_datetime(df_dim_time['date'].dt.to_period('W').apply(lambda r: r.start_time))\n",
        "df_dim_time['day_of_week']      = df_dim_time['date'].dt.dayofweek\n",
        "df_dim_time['day_of_week_name'] = df_dim_time['date'].dt.day_name()\n",
        '\n',
        "df_dim_time.to_parquet('../../data/silver/dim_calendar.parquet', index=False)\n",
        'print(f"dim_calendar: {df_dim_time.shape[0]:,} days  — Saved")\n',
        'df_dim_time.head()',
    ], 's_04'),

    md([
        '## dim_supplier\n',
        '\n',
        'One row per supplier. Supplier names are anonymised (e.g. Supplier A).\n',
    ], 's_05'),

    code([
        "df_dim_supplier = pd.DataFrame({'supplier_id': df['supplier_id'].sort_values().unique()})\n",
        "df_dim_supplier['supplier_name'] = df_dim_supplier['supplier_id'].apply(lambda x: f'Supplier {x}')\n",
        '\n',
        "df_dim_supplier.to_parquet('../../data/silver/dim_supplier.parquet', index=False)\n",
        'print(f"dim_supplier: {len(df_dim_supplier)} suppliers  — Saved")\n',
    ], 's_06'),

    md([
        '## dim_region\n',
        '\n',
        'One row per Brazilian macro-region (IBGE standard). Region IDs are assigned '
        'alphabetically starting at 1.\n',
    ], 's_07'),

    code([
        "df_dim_region = pd.DataFrame(df['region'].sort_values().unique()).reset_index()\n",
        "df_dim_region.columns = ['region_id', 'region_name']\n",
        "df_dim_region['region_id'] += 1  # 1-indexed\n",
        '\n',
        "df_dim_region.to_parquet('../../data/silver/dim_region.parquet', index=False)\n",
        'print(f"dim_region: {len(df_dim_region)} regions  — Saved")\n',
        'df_dim_region\n',
    ], 's_08'),

    md([
        '## dim_product\n',
        '\n',
        'One row per product. Three anonymised hierarchy levels are preserved:\n',
        '`product_attribute_1` (family) → `product_attribute_2` (line) → `product_attribute_3` (class).\n',
    ], 's_09'),

    code([
        "df_dim_product = (\n",
        "    df[['product_id', 'product_attribute_1', 'product_attribute_2', 'product_attribute_3']]\n",
        "    .drop_duplicates()\n",
        "    .sort_values('product_id')\n",
        "    .reset_index(drop=True)\n",
        ")\n",
        "df_dim_product['product_name'] = df_dim_product['product_id'].apply(lambda x: f'Product {x}')\n",
        '\n',
        "df_dim_product.to_parquet('../../data/silver/dim_product.parquet', index=False)\n",
        'print(f"dim_product: {len(df_dim_product)} products  — Saved")\n',
    ], 's_10'),

    md([
        '## fact_sales_weekly\n',
        '\n',
        'Joins the bronze flat table with `dim_region` to replace raw region names with integer IDs '
        'and standardises the week date to Monday.\n',
        'Multiple rows for the same (week, supplier, product, region) are summed — '
        'consistent with weekly sell-out data.\n',
    ], 's_11'),

    code([
        '# Join to get Monday-anchored week_date and integer region_id\n',
        "df_fact_sales = (\n",
        "    df[['week_date', 'supplier_id', 'product_id', 'region', 'units_sold']]\n",
        "    .rename(columns={'week_date': 'week_date_raw'})\n",
        "    .merge(df_dim_time[['date', 'week_date']], left_on='week_date_raw', right_on='date', how='left')\n",
        "    .drop(columns=['date', 'week_date_raw'])\n",
        "    .merge(df_dim_region[['region_id', 'region_name']], left_on='region', right_on='region_name', how='left')\n",
        "    .drop(columns=['region', 'region_name'])\n",
        ")\n",
        '\n',
        '# Aggregate duplicate (week, supplier, product, region) rows by summing units\n',
        "df_fact_sales = (\n",
        "    df_fact_sales\n",
        "    .groupby(['week_date', 'supplier_id', 'product_id', 'region_id'])\n",
        "    .agg(units_sold=('units_sold', 'sum'))\n",
        "    .reset_index()\n",
        ")\n",
        '\n',
        '# Enforce column order and add a surrogate key\n',
        "df_fact_sales = (\n",
        "    df_fact_sales[['week_date', 'supplier_id', 'region_id', 'product_id', 'units_sold']]\n",
        "    .reset_index(drop=True)\n",
        "    .assign(sales_id=lambda d: d.index + 1)\n",
        ")\n",
        '\n',
        "df_fact_sales.to_parquet('../../data/silver/fact_sales_weekly.parquet', index=False)\n",
        'print(f"fact_sales_weekly: {len(df_fact_sales):,} rows  — Saved")\n',
        'df_fact_sales.head()\n',
    ], 's_12'),
])

save(silver, '2-silver_layer.ipynb')


# ══════════════════════════════════════════════════════════════════════════════
# 3 — GOLD LAYER
# ══════════════════════════════════════════════════════════════════════════════
gold = nb([
    md([
        '# Gold Layer — Data Warehouse & Time Series Registry\n',
        '\n',
        '**Objective:** Promote the silver tables to a production-ready data warehouse,\n',
        'aggregate the calendar to weekly granularity, and build a time series registry\n',
        '(`dim_time_series`) that assigns a unique ID to every active (supplier, product, region)\n',
        'combination. Empty series (zero sales across all weeks) are excluded.\n',
        '\n',
        '**Input :** `data/silver/*.parquet`  \n',
        '**Output:** `data/gold/data_warehouse/dw_*.parquet`',
    ], 'g_00'),

    code(['import pandas as pd\nimport numpy as np\n'], 'g_01'),

    code([
        '# ── Load silver tables ───────────────────────────────────────────────────────\n',
        "df_dim_region   = pd.read_parquet('../../data/silver/dim_region.parquet')\n",
        "df_dim_product  = pd.read_parquet('../../data/silver/dim_product.parquet')\n",
        "df_dim_calendar = pd.read_parquet('../../data/silver/dim_calendar.parquet')\n",
        "df_dim_supplier = pd.read_parquet('../../data/silver/dim_supplier.parquet')\n",
        "df_fact_sales   = pd.read_parquet('../../data/silver/fact_sales_weekly.parquet')\n",
        'print("Silver tables loaded.")\n',
    ], 'g_02'),

    md([
        '## Promote dimension tables to gold\n',
        '\n',
        'Region, product, and supplier dimensions require no transformation — '
        'they are written as-is to the gold layer.\n',
    ], 'g_03'),

    code([
        "df_dim_region.to_parquet('../../data/gold/data_warehouse/dw_dim_region.parquet', index=False)\n",
        "df_dim_product.to_parquet('../../data/gold/data_warehouse/dw_dim_product.parquet', index=False)\n",
        "df_dim_supplier.to_parquet('../../data/gold/data_warehouse/dw_dim_supplier.parquet', index=False)\n",
        "print('dim_region / dim_product / dim_supplier  — Saved')\n",
    ], 'g_04'),

    md([
        '## dim_weekly_calendar\n',
        '\n',
        'Collapse the daily calendar to weekly granularity: one row per ISO week.\n',
        'Day-level attributes are dropped; year, semester, quarter, and month attributes\n',
        'are kept using the first day of each week as representative.\n',
    ], 'g_05'),

    code([
        '# Aggregate daily → weekly, drop day-level columns\n',
        'columns = df_dim_calendar.columns[1:]\n',
        'df_dim_weekly_calendar = (\n',
        '    df_dim_calendar\n',
        "    .groupby('week_date', as_index=False)\n",
        '    .agg(\n',
        "        start_date=('date', 'min'),\n",
        "        end_date=('date', 'max'),\n",
        "        **{col: (col, 'first') for col in columns}\n",
        '    )\n',
        "    .drop(columns=[col for col in columns if 'day' in col])\n",
        ')\n',
        '\n',
        "# Reorder: week_date + week first, then remaining temporal attributes\n",
        "weekly_columns = ['week_date', 'week'] + [\n",
        "    c for c in df_dim_weekly_calendar.columns if c not in ('week_date', 'week')\n",
        "]\n",
        "df_dim_weekly_calendar = df_dim_weekly_calendar[[c for c in weekly_columns if c in df_dim_weekly_calendar.columns]]\n",
        '\n',
        "df_dim_weekly_calendar.to_parquet('../../data/gold/data_warehouse/dw_dim_weekly_calendar.parquet', index=False)\n",
        'print(f"dim_weekly_calendar: {len(df_dim_weekly_calendar)} weeks  — Saved")\n',
        'df_dim_weekly_calendar.head()\n',
    ], 'g_06'),

    md([
        '## dim_time_series — Time Series Registry\n',
        '\n',
        'Builds one row per active (supplier, product, region) combination.\n',
        'For each series, a complete weekly spine is created (filling gaps with zeros)\n',
        'to compute statistics that correctly account for zero-demand weeks.\n',
        'Series with zero sales in every week are excluded — they carry no signal.\n',
    ], 'g_07'),

    code([
        'def create_time_series_registry(df_fact_sales: pd.DataFrame) -> pd.DataFrame:\n',
        '    """\n',
        '    Build a registry of active time series with summary statistics.\n',
        '    A complete weekly spine is constructed per series to ensure\n',
        '    zero-demand weeks are counted correctly in all metrics.\n',
        '    """\n',
        '    results = []\n',
        '    groups = df_fact_sales.groupby(["supplier_id", "region_id", "product_id"])\n',
        '    print(f"Building registry for {groups.ngroups:,} series...")\n',
        '\n',
        '    for i, ((supplier, region, product), group) in enumerate(groups):\n',
        '        if (i + 1) % 500 == 0:\n',
        '            print(f"  {i+1}/{groups.ngroups}")\n',
        '\n',
        '        first_week = group["week_date"].min()\n',
        '        last_week  = group["week_date"].max()\n',
        '\n',
        '        # Full weekly spine for the active period\n',
        '        spine = pd.DataFrame({\n',
        '            "supplier_id": supplier, "region_id": region, "product_id": product,\n',
        '            "week_date": pd.date_range(first_week, last_week, freq="W-MON"),\n',
        '        })\n',
        '        series = spine.merge(group[["week_date", "units_sold"]], on="week_date", how="left")\n',
        '        series["units_sold"] = series["units_sold"].fillna(0)\n',
        '        units = series["units_sold"].values\n',
        '\n',
        '        results.append({\n',
        '            "supplier_id": supplier,\n',
        '            "region_id":   region,\n',
        '            "product_id":  product,\n',
        '            "first_week_date":    first_week,\n',
        '            "last_week_date":     last_week,\n',
        '            "total_weeks_length": len(units),\n',
        '            "num_week_with_sales": int((units > 0).sum()),\n',
        '            "num_week_with_zeros": int((units == 0).sum()),\n',
        '            "sales_units":         units.sum(),\n',
        '            "avg_weekly_sales":    units.mean(),\n',
        '            "std_weekly_sales":    units.std(ddof=1) if len(units) > 1 else 0,\n',
        '            "max_weekly_sales":    units.max(),\n',
        '            "min_weekly_sales":    units.min(),\n',
        '            "q25_sales":  np.quantile(units, 0.25),\n',
        '            "q50_sales":  np.quantile(units, 0.50),\n',
        '            "q75_sales":  np.quantile(units, 0.75),\n',
        '        })\n',
        '\n',
        '    df = pd.DataFrame(results)\n',
        '    # Derived features\n',
        '    df["sales_weeks_ratio"] = df["num_week_with_sales"] / df["total_weeks_length"]\n',
        '    df["cv"]  = df["std_weekly_sales"] / (df["avg_weekly_sales"] + 1e-6)\n',
        '    df["iqr"] = df["q75_sales"] - df["q25_sales"]\n',
        '    return df\n',
        '\n',
        'df_dim_time_series = create_time_series_registry(df_fact_sales)\n',
        'print(f"Registry built: {len(df_dim_time_series):,} series")\n',
    ], 'g_08'),

    code([
        '# Inspect and remove series with zero sales across all weeks\n',
        'empty = (df_dim_time_series["num_week_with_sales"] == 0).sum()\n',
        'print(f"Empty series (all zeros): {empty} ({empty / len(df_dim_time_series):.1%})")\n',
        '\n',
        'df_dim_time_series = df_dim_time_series[df_dim_time_series["num_week_with_sales"] > 0].copy()\n',
        'print(f"Active series retained: {len(df_dim_time_series):,}")\n',
    ], 'g_09'),

    code([
        '# Assign surrogate time_series_id and save\n',
        'df_dim_time_series = df_dim_time_series.reset_index(drop=True)\n',
        'df_dim_time_series.insert(0, "time_series_id", df_dim_time_series.index + 1)\n',
        '\n',
        "df_dim_time_series.to_parquet('../../data/gold/data_warehouse/dw_dim_time_series.parquet', index=False)\n",
        'print(f"dim_time_series: {len(df_dim_time_series):,} series  — Saved")\n',
        'df_dim_time_series.head()\n',
    ], 'g_10'),

    md([
        '## dw_fact_sales_weekly\n',
        '\n',
        'Join the fact table with the time series registry to attach `time_series_id`.\n',
        'An inner join naturally excludes any (supplier, product, region) combination\n',
        'not present in the registry (i.e. empty series).\n',
    ], 'g_11'),

    code([
        '(\n',
        '    df_fact_sales\n',
        '    .merge(df_dim_time_series[["supplier_id", "region_id", "product_id", "time_series_id"]],\n',
        '           on=["supplier_id", "region_id", "product_id"], how="inner")\n',
        '    [["week_date", "supplier_id", "region_id", "product_id", "time_series_id", "units_sold"]]\n',
        "    .to_parquet('../../data/gold/data_warehouse/dw_fact_sales_weekly.parquet', index=False)\n",
        ')\n',
        "print('dw_fact_sales_weekly  — Saved')\n",
    ], 'g_12'),
])

save(gold, '3-gold_layer.ipynb')
print('All notebooks rewritten.')
