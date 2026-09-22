# Task: Implement data cleaning for the WFP price dataset and populate `02_data_cleaning.ipynb`

This task implements the cleaning logic in `preprocessing/cleaning.py` and
`preprocessing/aggregation.py`, then wires it together in a new notebook,
`model/notebooks/02_data_cleaning.ipynb`, which takes the raw WFP dataset and produces a
clean, monthly, model-ready price panel. No climate or fuel price data is used in this
project — only WFP price data.

## Finalized scope (update `config/constants.py` accordingly)

```python
MARKETS = ["Nairobi", "Kisumu", "Mombasa", "Nakuru"]
CROPS = ["Maize", "Beans"]
```

Remove `Eldoret` from `MARKETS`. Update the module docstring in `constants.py` to note the
final crop scope was determined by coverage analysis in `01_eda.ipynb`, merging `Maize` with
`Maize (White)` and `Beans` with `Beans (Dry)` after unit normalization.

## `preprocessing/cleaning.py` — implement the following functions

### `filter_target_scope(df, markets, raw_commodity_map)`
Filters the raw dataframe to the target markets, and relabels commodities according to
`raw_commodity_map`, e.g. `{"Maize (White)": "Maize", "Beans (Dry)": "Beans"}`. Returns the
filtered, relabeled dataframe. Include a docstring noting that market and commodity string
values should already be standardized (stripped, title-cased) before this function runs.

### `normalize_units(df, price_col='price', unit_col='unit')`
Adds a `price_per_kg` column: divides price by 90 where `unit == '90 KG'`, leaves price
unchanged where `unit == 'KG'`, and sets `price_per_kg` to `NaN` for any other unit value,
logging a warning listing any unhandled unit values found. Do not silently drop unhandled
rows — flag them for manual review via a printed summary.

### `resolve_pricetype(df, preferred='Wholesale', pricetype_col='pricetype')`
Filters the dataframe to rows matching `preferred` pricetype only. Before filtering, print a
summary table of record counts per (market, commodity, pricetype) so the proportion of data
being dropped is visible. Return the filtered dataframe.

### `resolve_label_overlap(df, date_col='date', group_cols=('market', 'commodity'))`
After relabeling `Maize (White)` to `Maize` and `Beans (Dry)` to `Beans`, there may be a small
number of dates where both the original and renamed label reported a price for the same
(market, date) combination, if their coverage periods overlap. For any such overlapping date,
average the `price_per_kg` values rather than arbitrarily keeping one. Log how many
(market, commodity, date) combinations required this averaging. Return the deduplicated
dataframe.

### `build_monthly_calendar(markets, crops, start_date, end_date)`
Returns a dataframe containing every combination of (market, crop, month) between
`start_date` and `end_date` at monthly frequency, representing the complete expected panel.
This is used to make missingness explicit rather than implicit.

### `quantify_missingness(observed_df, calendar_df, join_cols=('market', 'commodity', 'month'))`
Left-joins `calendar_df` to `observed_df` and returns a summary dataframe showing the count
and proportion of missing months per (market, commodity) combination. Also print an overall
missingness percentage.

### `forward_fill_short_gaps(df, value_col='price_per_kg', max_gap_months=2, group_cols=('market', 'commodity'))`
Within each (market, commodity) group, forward-fills `value_col` only for gaps of
`max_gap_months` or fewer consecutive missing months. Gaps longer than the threshold remain
`NaN` rather than being filled, since forward-filling a long gap would fabricate a misleading
trend. Return the dataframe with an added boolean column `was_imputed` marking which values
were filled.

### `flag_outliers_iqr(df, value_col='price_per_kg', group_cols=('market', 'commodity'), multiplier=1.5)`
Computes Q1, Q3, and IQR per (market, commodity) group, and adds a boolean column
`is_outlier` marking values outside `[Q1 - multiplier * IQR, Q3 + multiplier * IQR]`. Does
NOT remove or alter flagged values — flagging only, so the decision to exclude or retain them
can be made explicitly in the notebook after visual review.

## `preprocessing/aggregation.py` — implement the following function

### `aggregate_to_monthly(df, date_col='date', value_col='price_per_kg', group_cols=('market', 'commodity'))`
Floors `date_col` to the first of its month and computes the mean `value_col` per
(market, commodity, month) group. This is the final aggregation step producing one row per
market-crop-month, matching the granularity used throughout the rest of the project. Returns
the aggregated dataframe with columns `market`, `commodity`, `month`, `avg_price_per_kg`.

## `model/notebooks/02_data_cleaning.ipynb` — notebook structure

Insert the following cells in order.

### 0. Colab session setup
```python
REPO_URL = "https://github.com/<your-username>/<your-repo>.git"
!git clone {REPO_URL}
%cd <your-repo>/model
!pip install -r requirements.txt
```

```python
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config.constants import MARKETS, CROPS
from preprocessing.cleaning import (
    filter_target_scope,
    normalize_units,
    resolve_pricetype,
    resolve_label_overlap,
    build_monthly_calendar,
    quantify_missingness,
    forward_fill_short_gaps,
    flag_outliers_iqr,
)
from preprocessing.aggregation import aggregate_to_monthly

sns.set_theme(style="whitegrid")
```

### 1. Load raw dataset
```python
raw_df = pd.read_csv("PATH_TO_RAW_WFP_CSV")
raw_df['market'] = raw_df['market'].str.strip().str.title()
raw_df['commodity'] = raw_df['commodity'].str.strip().str.title()
raw_df['date'] = pd.to_datetime(raw_df['date'])
```

### 2. Filter to target markets and relabel commodities
```python
RAW_COMMODITY_MAP = {
    "Maize (White)": "Maize",
    "Beans (Dry)": "Beans",
}
df = filter_target_scope(raw_df, MARKETS, RAW_COMMODITY_MAP)
print(df.shape)
df.groupby(['market', 'commodity']).size()
```

### 3. Normalize units to price per kg
```python
df = normalize_units(df)
df[['commodity', 'unit', 'price', 'price_per_kg']].sample(10)
```

### 4. Verify unit normalization closed the scale gap
```python
comparison = df.groupby(['commodity'])['price_per_kg'].agg(['mean', 'std', 'min', 'max'])
print(comparison)
# Confirm Maize and Beans price_per_kg values are now in a comparable, sensible range
# before proceeding. If not, stop and investigate before continuing.
```

### 5. Resolve pricetype to wholesale only
```python
df = resolve_pricetype(df, preferred='Wholesale')
print(df.shape)
```

### 6. Resolve any remaining label-overlap duplicates
```python
df = resolve_label_overlap(df)
print(df.shape)

# Re-run duplicate check to confirm no (market, commodity, date) duplicates remain
dupes = df[df.duplicated(subset=['market', 'commodity', 'date'], keep=False)]
print(f"Remaining duplicates: {len(dupes)}")
```

### 7. Build the full monthly calendar and quantify missingness
```python
df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
calendar = build_monthly_calendar(MARKETS, CROPS, df['month'].min(), df['month'].max())
missingness_summary = quantify_missingness(df, calendar)
missingness_summary
```

```python
plt.figure(figsize=(8, 5))
sns.heatmap(
    missingness_summary.pivot(index='market', columns='commodity', values='missing_proportion'),
    annot=True, fmt='.1%', cmap='Reds'
)
plt.title('Missing month proportion by market and crop')
plt.tight_layout()
plt.show()
```

### 8. Aggregate to monthly frequency
```python
monthly_df = aggregate_to_monthly(df)
monthly_df.head()
```

### 9. Reindex against the full calendar to expose gaps explicitly
```python
full_panel = calendar.merge(
    monthly_df, on=['market', 'commodity', 'month'], how='left'
)
print(f"Total expected rows: {len(calendar)}")
print(f"Rows with an observed price: {full_panel['avg_price_per_kg'].notna().sum()}")
```

### 10. Forward-fill short gaps only
```python
full_panel = forward_fill_short_gaps(full_panel, value_col='avg_price_per_kg', max_gap_months=2)
print(f"Values imputed: {full_panel['was_imputed'].sum()}")
print(f"Remaining missing after short-gap fill: {full_panel['avg_price_per_kg'].isna().sum()}")
```

### 11. Flag outliers (do not remove yet)
```python
full_panel = flag_outliers_iqr(full_panel, value_col='avg_price_per_kg')
print(f"Flagged outliers: {full_panel['is_outlier'].sum()}")
full_panel[full_panel['is_outlier']]
```

```python
# Visual check on flagged outliers before deciding whether to exclude or retain them
fig, axes = plt.subplots(len(CROPS), 1, figsize=(12, 4 * len(CROPS)))
for ax, crop in zip(axes, CROPS):
    subset = full_panel[full_panel['commodity'] == crop]
    for market in MARKETS:
        m = subset[subset['market'] == market]
        ax.plot(m['month'], m['avg_price_per_kg'], label=market)
        outliers = m[m['is_outlier']]
        ax.scatter(outliers['month'], outliers['avg_price_per_kg'], color='red', zorder=5)
    ax.set_title(f'{crop}: cleaned series with flagged outliers')
    ax.legend()
plt.tight_layout()
plt.show()
```

### 12. Determine final common analysis window
```python
# Based on missingness and available history per market-crop combination,
# set the final chronological window used for training in the next notebook.
coverage_check = (
    full_panel[full_panel['avg_price_per_kg'].notna()]
    .groupby(['market', 'commodity'])['month']
    .agg(['min', 'max', 'count'])
)
coverage_check
```

```python
# TODO: set based on coverage_check output
ANALYSIS_START = None
ANALYSIS_END = None

final_df = full_panel[
    (full_panel['month'] >= ANALYSIS_START) & (full_panel['month'] <= ANALYSIS_END)
].copy()
```

### 13. Save cleaned dataset
```python
# Working cache in Drive
final_df.to_csv('/content/drive/MyDrive/capstone-data/cleaned_monthly_prices.csv', index=False)

# Repo snapshot (commit manually once this is the finalized version used for training)
final_df.to_csv('artifacts/training_data_snapshot.csv', index=False)
```

## Do not

- Do not include any climate (Open-Meteo) or fuel price (EPRA) columns anywhere in this
  notebook — this project's feature set is derived from WFP price data only.
- Do not drop flagged outliers automatically — flagging only, pending explicit review in
  Step 11.
- Do not perform feature engineering (lags, rolling averages, seasonal indicators) in this
  notebook — that belongs in the next stage.