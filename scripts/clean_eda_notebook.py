"""
Targeted cleanup for notebooks/data_science/eda.ipynb:
- Remove redundant / raw-file cells
- Translate Portuguese strings to English
- Fix typos in section headers

Run from project root:
    python scripts/clean_eda_notebook.py
"""

import json
from pathlib import Path

NB_PATH = Path('notebooks/data_science/eda.ipynb')

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ── Indices to remove (based on original 82-cell structure) ──────────────────
# fmt: off
REMOVE = {
    5,   # df_dataset.shape  — basic check
    8,   # df_dataset.head(5) — redundant
    9,   # df_dataset.tail(5) — redundant
    10,  # df_dataset.dtypes  — redundant
    12,  # pd.read_csv(...).min() — reads raw file!
    13,  # pd.read_csv(...).max() — reads raw file!
    15,  # df_dim_timeseries[id==1] — single-record debug display
    16,  # df_dataset[id==1]        — single-record debug display
    19,  # df_dataset.isna().sum()  — summarised in following markdown
    33,  # df_dataset                — bare DataFrame display
    52,  # df_total_sales_monthly_region.head() — df not yet defined here
    54,  # pd.read_csv(raw) alone  — reads raw file
    55,  # df_raw.columns           — debug display
    56,  # df_raw = pd.read_csv(raw); df_raw.columns — reads raw file
    57,  # df_total_sales_monthly_region.describe() — before df redefined
    74,  # Y_df display             — bare DataFrame display
    75,  # Y_df.isnull().sum()      — basic null check
    80,  # Product 134 analysis v1  — duplicate of cell 81
    81,  # Product 134 analysis v2  — experimental, isolated
}
# fmt: on


def translate_pt(src: str) -> str:
    """Apply all Portuguese-to-English substitutions."""
    replacements = [
        # Cell 0
        ('Explortory Data Analysis', 'Exploratory Data Analysis'),
        # Cell 11
        ('Descritive statistics', 'Descriptive Statistics'),
        ('Descritive Statistics', 'Descriptive Statistics'),
        # Cell 14
        ('Número de semanas:', 'Weeks    :'),
        ('Número de produtos:', 'Products :'),
        ('Número de fornecedores:', 'Suppliers:'),
        ('Número de regiões:', 'Regions  :'),
        # Cell 45
        ('#### 5.1.1\n', ''),
        # Cell 47
        ('# 1. Teste de autocorrelação nos resíduos', '# 1. Autocorrelation test on residuals'),
        ('Teste Ljung-Box (p-value > 0.05 = bom):', 'Ljung-Box test (p-value > 0.05 = good):'),
        ('# Se p-value < 0.05 → ainda há autocorrelação (padrão não capturado)',
         '# p-value < 0.05: residuals still correlated (pattern not fully captured)'),
        ('# Se p-value > 0.05 → resíduos são ruído branco (bom!)',
         '# p-value > 0.05: residuals are white noise (good)'),
        ('# 2. Normalidade dos resíduos', '# 2. Residual normality test'),
        ('Shapiro-Wilk (normalidade):', 'Shapiro-Wilk (normality):'),
        ('# p < 0.05 → não normal (esperado para séries de demanda)',
         '# p < 0.05: non-normal (expected for demand series)'),
        # Cell 73
        ('# 1. Preparar o dataframe no formato esperado (unique_id, ds, y)',
         '# 1. Prepare dataframe in expected format (unique_id, ds, y)'),
        ('# 2. Definir a hierarquia agrupada (spec = do mais agregado ao bottom)',
         '# 2. Define hierarchy spec (from most aggregated to bottom level)'),
        ('"Total" é adicionado automaticamente pela função',
         '"Total" is added automatically by the function'),
        ('# 27 séries', '# 27 series'),
        # Cell 78
        ('# distribuição por nível', '# zero sales distribution by hierarchy level'),
        ('mediana zeros =', 'median zeros ='),
        # Cell 79
        ("f'Nível: {level_name}'", "f'Level: {level_name}'"),
    ]
    for pt, en in replacements:
        src = src.replace(pt, en)
    return src


# ── Apply removals and translations ──────────────────────────────────────────
new_cells = []
for i, cell in enumerate(cells):
    if i in REMOVE:
        continue

    # Apply translation to source
    src = ''.join(cell['source'])
    fixed = translate_pt(src)
    cell = dict(cell)           # shallow copy
    cell['source'] = fixed

    new_cells.append(cell)

nb['cells'] = new_cells

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'EDA notebook: {len(cells)} -> {len(new_cells)} cells')
print(f'Saved: {NB_PATH}')
