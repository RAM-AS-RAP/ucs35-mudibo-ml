# Data Cleaning Analysis and Feature-Engineering Implications

## 1. Purpose and scope

This document records the cleaning decisions applied to the WFP Kenya food-price data in `model/notebooks/02_data_cleaning.ipynb`. The objective is to produce a reliable monthly price panel for forecasting without fabricating long stretches of unobserved prices.

The scope is limited to WFP prices. No climate, fuel-price, or external explanatory variables are introduced during cleaning. The finalized scope contains four markets and two commodities:

- Markets: Nairobi, Kisumu, Mombasa, and Nakuru
- Commodities: Maize and Beans

`Maize (White)` is mapped to `Maize`, and `Beans (Dry)` is mapped to `Beans` after unit normalization. The resulting modeling unit is one market-commodity monthly series.

## 2. Cleaning process

### 2.1 Standardization and scope filtering

Raw market and commodity labels are stripped and title-cased before filtering. The raw dataset is then restricted to the four configured markets. Commodity variants are relabeled into the two final crop categories.

This creates consistent categorical keys for grouping, calendar construction, aggregation, lag generation, and model evaluation. The scope filter is intentionally performed before downstream calculations so that missingness and coverage statistics describe only the modeling population.

### 2.2 Unit normalization

The pipeline creates `price_per_kg`:

- `KG`: price is retained
- `90 KG`: price is divided by 90
- Any other unit: `price_per_kg` is set to `NaN` and the unit is printed as unhandled

Unhandled units are not silently dropped. This protects the model from mixing prices expressed at incompatible scales. Unit validation is especially important because the two merged commodity labels have different raw coverage and may otherwise appear to have materially different price levels simply because of packaging or unit definitions.

### 2.3 Price-type resolution

The pipeline prints record counts by market, commodity, and price type, then retains the preferred `Wholesale` observations. This avoids mixing retail and wholesale prices in one target series. The distinction matters because the same market, commodity, and date can contain more than one price-type observation.

### 2.4 Label-overlap resolution

After commodity relabeling, rows can collide on `(market, commodity, date)`. The pipeline averages the normalized price values for such overlaps rather than arbitrarily keeping one row. This produces one observation per market-commodity-date while preserving information from both source labels.

### 2.5 Monthly aggregation

Daily or within-month observations are converted to the first day of the month and averaged by `(market, commodity, month)`. The resulting target is `avg_price_per_kg`.

The monthly frequency matches the observed reporting cadence: the dominant date difference is approximately 31 days. Aggregation also ensures that the downstream model has one intended target granularity rather than a mixture of daily and monthly records.

## 3. Missingness analysis

### 3.1 Complete expected panel

A full market-commodity-month calendar is constructed. The executed notebook reports:

- Expected calendar rows: 1,568
- Calendar rows present in the panel: 1,568
- Initial observed prices: 1,354
- Initial overall missingness: 13.65%

The complete calendar is retained so the absence of a reported price is explicit. Without this step, a missing month could be mistaken for a month that never belonged to the series.

### 3.2 Gap-length strategy

Missing runs are classified before imputation:

- Short gaps: 1-2 months
- Medium gaps: 3-6 months
- Long gaps: more than 6 months

Short gaps are forward-filled from the most recent known value and marked with `was_imputed`. Medium gaps use only the same calendar month from one or two prior years and are marked with `was_interpolated`. Future-year references and linear interpolation are deliberately excluded because they would use information unavailable at the historical prediction date. If neither prior-year reference exists, the value remains missing and is classified as `long_gap_unfilled`. Long gaps remain missing.

The executed counts are:

| Imputation method | Rows | Share |
| --- | ---: | ---: |
| Observed | 1,354 | 86.35% |
| Short-gap forward fill | 31 | 1.98% |
| Medium-gap seasonal interpolation | 17 | 1.08% |
| Long gap unfilled | 166 | 10.59% |

The method is deliberately conservative. A long outage is not a normal missing value: filling it would create a synthetic price trajectory that could be mistaken for ground truth.

### 3.3 Per-series windows

Each market-commodity series receives its own first and last observed month. Leading and trailing coverage gaps are excluded from that series' analysis window, so a series with a shorter reporting history does not force all other series to discard valid recent observations.

The final per-series panel contains:

- Per-series panel rows: 1,450
- Rows with a target price: 1,402
- Internal missing target rows: 48
- Series retained: 8

The series windows are:

| Market | Commodity | Window start | Window end | Rows | Missing targets |
| --- | --- | --- | --- | ---: | ---: |
| Kisumu | Beans | 2006-01 | 2019-10 | 166 | 0 |
| Kisumu | Maize | 2006-01 | 2022-04 | 196 | 22 |
| Mombasa | Beans | 2006-01 | 2020-03 | 171 | 0 |
| Mombasa | Maize | 2006-01 | 2022-04 | 196 | 9 |
| Nairobi | Beans | 2006-01 | 2022-04 | 196 | 8 |
| Nairobi | Maize | 2006-01 | 2022-04 | 196 | 0 |
| Nakuru | Beans | 2008-01 | 2021-02 | 158 | 9 |
| Nakuru | Maize | 2008-02 | 2022-04 | 171 | 0 |

The 48 remaining targets are concentrated in four series:

- Kisumu Maize: 22 months, 2020-04 to 2022-01
- Mombasa Maize: 9 months, 2020-04 to 2020-12
- Nairobi Beans: 8 months, 2020-05 to 2020-12
- Nakuru Beans: 9 months, 2020-04 to 2020-12

These gaps overlap in the 2020 reporting-disruption period and are long, correlated outages rather than isolated missing points. They should not be filled with forward fill, ordinary interpolation, or a fabricated market average.

### 3.4 Training-target policy

The canonical cleaned panel keeps all expected market-commodity-month rows, including the 48 missing targets, for auditability and future data revision. A separate supervised-learning view must exclude missing targets:

```python
training_df = final_df.dropna(subset=['avg_price_per_kg']).copy()
```

The 48 rows must be excluded from model fitting, validation, testing, and metric calculation because no ground-truth price exists for them. They are unavailable labels, not zero prices and not valid evaluation cases.

A useful explicit indicator is:

```python
final_df['target_available'] = final_df['avg_price_per_kg'].notna()
```

### 3.5 Outlier handling

The pipeline flags, but does not remove, IQR outliers. The latest executed output flags 38 rows. Several flagged values occur in multi-market periods such as 2017 maize price increases, which makes them plausible market shocks rather than data errors.

Outlier flags should therefore be retained for analysis and possible feature use. A flagged value should only be altered after source-level review confirms a data-entry problem. Removing genuine shocks would make the model less capable of forecasting unusual but real market conditions.

## 4. Data products for the next stage

The cleaning stage should conceptually produce two related outputs:

### 4.1 Canonical panel

`final_df` is the auditable monthly panel. It retains:

- all expected calendar rows within each series window
- observed and imputed price values
- `impute_method`
- `was_imputed`
- `was_interpolated`
- `is_outlier`
- per-series coverage metadata

This is the correct source for diagnostics and feature construction because it preserves the monthly index and exposes gaps.

### 4.2 Supervised modeling table

A training table is derived from the canonical panel only after features are created and target availability is checked. It contains rows with:

- a non-missing target
- valid required lag and rolling features
- no feature leakage across long missing runs

This table may contain fewer rows than `final_df`, which is expected and preferable to fabricating labels.

## 5. Implications for feature engineering

### 5.1 Compute features within each series

All temporal features must be calculated separately for each `(market, commodity)` pair. Never sort the pooled data by month and compute lags across markets or crops.

```python
series_cols = ['market', 'commodity']
feature_df = final_df.sort_values(series_cols + ['month']).copy()

feature_df['lag_1'] = (
    feature_df.groupby(series_cols)['avg_price_per_kg'].shift(1)
)
feature_df['lag_2'] = (
    feature_df.groupby(series_cols)['avg_price_per_kg'].shift(2)
)
feature_df['lag_3'] = (
    feature_df.groupby(series_cols)['avg_price_per_kg'].shift(3)
)
```

### 5.2 Do not bridge long gaps with lags

A lag is meaningful only when the prior observation is genuinely close in time. For example, a value from March 2020 should not be treated as the one-month lag for a value in February 2022 after a 22-month outage.

Use the explicit monthly panel and validate calendar continuity before accepting a lag. At minimum, create a month-distance check or a `consecutive_observation` indicator. Rows immediately after a long gap should have their affected lag features set to missing rather than carrying stale pre-gap values.

Rows with unavailable required features should be excluded from a particular training design, or the model should use a separate missingness-aware feature policy. The choice must be consistent between training and inference.

### 5.3 Rolling features must exclude the current target

Rolling statistics must be shifted before aggregation so the current target does not leak into its own predictors:

```python
feature_df['rolling_mean_3'] = (
    feature_df.groupby(series_cols)['avg_price_per_kg']
    .transform(lambda values: values.shift(1).rolling(3).mean())
)
```

The same rule applies to rolling standard deviation, minimum, maximum, and exponentially weighted statistics.

### 5.4 Add calendar features

The monthly target supports calendar features such as:

- month number
- quarter
- sine and cosine month encodings
- year or a carefully defined time trend

Calendar features should be created from `month`, not from row position. Row position is unreliable when series have different start dates and internal gaps.

### 5.5 Encode market and commodity identity

For a global model, include `market` and `commodity` as categorical predictors. This lets one model learn shared price dynamics while retaining systematic differences among locations and products.

Use one global model across the eight market-commodity series, then report metrics separately per series. The global model benefits from shared data, while per-series reporting exposes whether one market or commodity is poorly served.

### 5.6 Treat imputation provenance as metadata first

`impute_method` should be retained for audit and may be supplied as a feature only after a controlled experiment. A model can otherwise learn artifacts of the imputation process instead of price dynamics.

Recommended experiments:

1. Train using observed targets plus short- and medium-gap imputations.
2. Train using observed targets only.
3. Compare per-series and aggregate performance under chronological validation.

Never use `long_gap_unfilled` rows as targets. If an imputed value is used as a target, report metrics separately from metrics calculated on genuinely observed targets.

### 5.7 Handle outlier flags carefully

Keep `is_outlier` available as a diagnostic field. It may become a binary regime feature, but only after checking that it is not simply encoding information unavailable at forecast time. Since the current flag is calculated using the full dataset, recompute any production feature using training-only history to avoid leakage.

### 5.8 Avoid target leakage in all preprocessing

Any statistic learned from the target, including scaling parameters, outlier thresholds, imputation references, or rolling aggregates, must be fitted using only the training portion within each walk-forward split. Do not calculate full-dataset thresholds before temporal validation if those thresholds influence features.

## 6. Global-model training design

The recommended primary model is one global supervised model across all eight series, with market and commodity indicators plus group-wise lag and rolling features. Random Forest and XGBoost can be used as base learners, with a linear meta-model for stacking.

The stacking meta-model must receive out-of-fold, time-respecting predictions from the base learners. In-sample base predictions would leak the target into the meta-model.

Use chronological validation within each market-commodity series:

- train on earlier available observations
- validate on the next chronological block
- test on the latest valid observed block for that series

The test periods may end at different dates because the series have different observation windows. This is acceptable for a global model, provided metrics are reported by series and aggregate metrics are clearly weighted.

## 7. Recommended training-table pattern

```python
series_cols = ['market', 'commodity']
feature_df = final_df.sort_values(series_cols + ['month']).copy()

feature_df['target_available'] = feature_df['avg_price_per_kg'].notna()
feature_df['lag_1'] = (
    feature_df.groupby(series_cols)['avg_price_per_kg'].shift(1)
)
feature_df['lag_3'] = (
    feature_df.groupby(series_cols)['avg_price_per_kg'].shift(3)
)
feature_df['rolling_mean_3'] = (
    feature_df.groupby(series_cols)['avg_price_per_kg']
    .transform(lambda values: values.shift(1).rolling(3).mean())
)

model_df = feature_df.dropna(
    subset=['avg_price_per_kg', 'lag_1', 'lag_3', 'rolling_mean_3']
).copy()
```

The exact feature set belongs to the feature-engineering stage. The important contract from cleaning is that target values and temporal continuity are explicit before that stage begins.

## 8. Final recommendations

1. Keep `final_df` as the auditable per-series monthly panel.
2. Exclude the 48 long-gap target rows from supervised training and all evaluation metrics.
3. Do not overwrite missing targets with synthetic values.
4. Compute every lag and rolling feature within a market-commodity series.
5. Prevent lag features from crossing the 2020 reporting outage.
6. Use one global model with market and commodity identifiers, supported by per-series baselines.
7. Use chronological, per-series validation and report metrics per series.
8. Preserve imputation and outlier provenance for audit and controlled ablation experiments.
9. Recompute any target-derived statistic inside each training split to prevent leakage.

The cleaned data is therefore ready for feature engineering once the next stage creates continuity-aware lag features and derives a target-available modeling table from the canonical panel.
