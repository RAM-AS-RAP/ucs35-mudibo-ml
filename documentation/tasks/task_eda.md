# Task: Populate `Tasks/eda/eda.ipynb` with exploratory data analysis cells

Insert the following cells, in order, into `01_eda.ipynb`. Each `##` heading below should
become a markdown cell, followed immediately by the corresponding code cell(s). Do not
collapse steps together — each numbered step should be independently runnable and inspectable.

This notebook explores the raw WFP Kenya food prices dataset, restricted to the five target
markets (Nairobi, Nakuru, Mombasa, Kisumu, Eldoret) defined in `config/constants.py`, ahead of
cleaning, feature engineering, and baseline model training in later notebooks. Climate and fuel
price data are NOT explored here — this notebook is WFP price data only.

---

## 0. Colab session setup

```python
REPO_URL = "https://github.com/Mudibo/AgriSence.git"
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
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from config.constants import MARKETS

sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)
```

## 1. Load raw dataset

```python
# TODO: replace with actual fetch_wfp.py call once ingestion module is implemented in Sprint 2
raw_df = pd.read_csv("https://data.humdata.org/dataset/wfp-food-prices-for-kenya/resource/"
                    "517ee1bf-2437-4f8c-aa1b-cb9925b9d437/download/wfp_food_prices_ken.csv")
print(raw_df.shape)
raw_df.head()
```

## 2. Structural inspection

```python
raw_df.info()
```

```python
for col in raw_df.columns:
    print(f"--- {col} ---")
    print(raw_df[col].dropna().unique()[:15])
    print()
```

```python
# Check unit consistency per commodity — critical before any cross-market comparison
raw_df.groupby('commodity')['unit'].unique()
```

## 3. Standardize market and commodity naming

```python
# TODO: confirm exact market/commodity string values from Step 2, then standardize
raw_df['market'] = raw_df['market'].str.strip().str.title()
raw_df['commodity'] = raw_df['commodity'].str.strip().str.title()
```

## 4. Filter to target markets

```python
market_df = raw_df[raw_df['market'].isin(MARKETS)].copy()
print(f"Rows before market filter: {len(raw_df)}")
print(f"Rows after market filter: {len(market_df)}")
market_df['market'].value_counts()
```

## 5. Crop coverage analysis — data-driven crop selection

```python
# Record count per commodity across the 5 target markets
crop_coverage = (
    market_df.groupby('commodity')
    .agg(total_records=('commodity', 'count'),
         markets_covered=('market', 'nunique'),
         date_min=('date', 'min'),
         date_max=('date', 'max'))
    .sort_values('total_records', ascending=False)
)
crop_coverage.head(20)
```

```python
plt.figure(figsize=(10, 6))
crop_coverage.head(15)['total_records'].plot(kind='barh')
plt.xlabel('Total records')
plt.title('Commodity coverage across the 5 target markets')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

```python
# Coverage breakdown per market for candidate crops — confirms no single market is a
# blind spot for the crops eventually selected
candidate_crops = crop_coverage.head(10).index.tolist()
coverage_matrix = (
    market_df[market_df['commodity'].isin(candidate_crops)]
    .groupby(['commodity', 'market'])
    .size()
    .unstack(fill_value=0)
)

plt.figure(figsize=(10, 6))
sns.heatmap(coverage_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Record count by commodity and market')
plt.tight_layout()
plt.show()
```

```python
# TODO: based on the above, confirm final crop selection.
# If the top 3 by coverage differ from the proposal's tomatoes/kale/onions,
# flag this explicitly for the supervisor before proceeding.
FINAL_CROPS = []  # populate based on coverage_matrix results
```

## 6. Filter to final crop selection

```python
df = market_df[market_df['commodity'].isin(FINAL_CROPS)].copy()
df['date'] = pd.to_datetime(df['date'])
print(df.shape)
df.groupby(['market', 'commodity']).size()
```

## 7. Duplicate detection

```python
duplicates = df[df.duplicated(subset=['market', 'commodity', 'date'], keep=False)]
print(f"Duplicate (market, commodity, date) rows: {len(duplicates)}")
duplicates.sort_values(['market', 'commodity', 'date']).head(20)
```

## 8. Missingness analysis

```python
missing_by_market_crop = (
    df.groupby(['market', 'commodity'])['price']
    .apply(lambda x: x.isna().mean())
    .unstack()
)

plt.figure(figsize=(8, 5))
sns.heatmap(missing_by_market_crop, annot=True, fmt='.2%', cmap='Reds')
plt.title('Proportion of missing prices by market and crop')
plt.tight_layout()
plt.show()
```

```python
# Missingness over time — are gaps concentrated in earlier years?
df['year'] = df['date'].dt.year
missing_by_year = df.groupby('year')['price'].apply(lambda x: x.isna().mean())

plt.figure(figsize=(10, 4))
missing_by_year.plot(kind='bar')
plt.ylabel('Proportion missing')
plt.title('Missing price proportion by year')
plt.tight_layout()
plt.show()
```

## 9. Reporting frequency check

```python
freq_check = (
    df.groupby(['market', 'commodity'])['date']
    .apply(lambda x: x.sort_values().diff().mode()[0] if len(x) > 1 else pd.NaT)
)
freq_check
```

## 10. Outlier and anomaly screening

```python
print("Zero or negative prices:")
print(df[df['price'] <= 0])
```

```python
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='commodity', y='price', hue='market')
plt.title('Price distribution by crop and market')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 11. Time series visualization per market-crop combination

```python
fig, axes = plt.subplots(len(FINAL_CROPS), 1, figsize=(14, 4 * len(FINAL_CROPS)), sharex=True)

for ax, crop in zip(axes, FINAL_CROPS):
    subset = df[df['commodity'] == crop]
    for market in MARKETS:
        market_subset = subset[subset['market'] == market].sort_values('date')
        ax.plot(market_subset['date'], market_subset['price'], label=market)
    ax.set_title(f'{crop} price over time')
    ax.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()
```

## 12. Seasonality check

```python
df['month'] = df['date'].dt.month

fig, axes = plt.subplots(1, len(FINAL_CROPS), figsize=(6 * len(FINAL_CROPS), 5), sharey=False)

for ax, crop in zip(axes, FINAL_CROPS):
    seasonal = df[df['commodity'] == crop].groupby('month')['price'].mean()
    ax.plot(seasonal.index, seasonal.values, marker='o')
    ax.set_title(f'{crop}: average price by calendar month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average price')

plt.tight_layout()
plt.show()
```

## 13. Volatility comparison across crops and markets

```python
volatility = (
    df.groupby(['market', 'commodity'])['price']
    .agg(mean='mean', std='std')
    .assign(coefficient_of_variation=lambda x: x['std'] / x['mean'])
    .sort_values('coefficient_of_variation', ascending=False)
)
volatility
```

```python
plt.figure(figsize=(10, 6))
volatility['coefficient_of_variation'].unstack(level=0).plot(kind='bar')
plt.ylabel('Coefficient of variation')
plt.title('Price volatility by market and crop')
plt.tight_layout()
plt.show()
```

## 14. Stationarity testing (ADF)

```python
adf_results = []

for market in MARKETS:
    for crop in FINAL_CROPS:
        series = (
            df[(df['market'] == market) & (df['commodity'] == crop)]
            .sort_values('date')['price']
            .dropna()
        )
        if len(series) > 10:
            result = adfuller(series)
            adf_results.append({
                'market': market,
                'crop': crop,
                'adf_statistic': result[0],
                'p_value': result[1],
                'stationary': result[1] < 0.05
            })

adf_df = pd.DataFrame(adf_results)
adf_df
```

## 15. Autocorrelation and partial autocorrelation (ACF/PACF)

```python
# Example for one representative market-crop combination — repeat/adjust as needed
example_series = (
    df[(df['market'] == MARKETS[0]) & (df['commodity'] == FINAL_CROPS[0])]
    .sort_values('date')['price']
    .dropna()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(example_series, ax=axes[0], lags=24)
plot_pacf(example_series, ax=axes[1], lags=24)
axes[0].set_title(f'ACF: {FINAL_CROPS[0]} in {MARKETS[0]}')
axes[1].set_title(f'PACF: {FINAL_CROPS[0]} in {MARKETS[0]}')
plt.tight_layout()
plt.show()
```

## 16. Summary of findings

```markdown
## EDA Summary

- Final crop selection: [fill in based on Step 5]
- Markets/crops with notable missingness: [fill in based on Step 8]
- Series requiring differencing (non-stationary per ADF): [fill in based on Step 14]
- Notable seasonal patterns observed: [fill in based on Step 12]
- Any duplicate or anomalous records requiring attention in cleaning: [fill in based on Steps 7, 10]
```

## Do not

- Do not perform any cleaning (forward-fill, outlier removal) in this notebook — that belongs
  in the next stage (`preprocessing/cleaning.py` and its corresponding notebook work).
- Do not fetch or join Open-Meteo or EPRA data in this notebook.
- Do not hardcode `FINAL_CROPS` before Step 5's analysis has actually run and been reviewed.