# Data Cleaning Analysis and Feature-Engineering Implications

## 1. Purpose and scope

This document records the cleaning decisions applied to the WFP Kenya food-price data in `notebooks/02_data_cleaning.ipynb`. The objective is to produce a monthly price panel suitable for one-month and two-month-ahead forecasting, while preserving source and imputation provenance.

The target is monthly wholesale price per kilogram. The finalized scope contains four markets and two commodities:

- Markets: Nairobi, Kisumu, Mombasa, and Nakuru
- Commodities: Maize and Beans

`Maize (White)` is mapped to `Maize`, and `Beans (Dry)` is mapped to `Beans` after unit normalization. The resulting modeling unit is one of eight market-commodity monthly series. Climate and other exogenous variables are not introduced by the cleaning notebook; they remain optional feature-engineering extensions.

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

## 3. Coverage and missingness analysis

### 3.1 Complete expected panel

A full market-commodity-month calendar is constructed before imputation. This gives every series an explicit monthly index and prevents missing months from being confused with months outside the reporting window. The panel is then extended with manually exported FEWS NET records through 2025.

After the extension step, all eight market-commodity series have coverage through 2025 and the cleaned dataset has no missing `avg_price_per_kg` targets. This means the resulting table is complete for the defined market, commodity, and month keys; it does not mean every value is an original WFP observation.

### 3.2 Tiered gap handling and external extension

Before the FEWS extension, missing runs are classified as:

- Short gaps: 1-2 months
- Medium gaps: 3-6 months
- Long gaps: more than 6 months

Short gaps are forward-filled from the most recent known value and marked with `short_gap_ffill`. Medium gaps use the same calendar month from one or two prior years and are marked with `medium_gap_seasonal`. The original long gaps are filled from manually saved FEWS NET responses and marked `external_fill_fews`. These steps are deterministic and retain the `impute_method` column for later ablation and reporting.

The extension then loads one FEWS file per market-commodity series, normalizes `start_date` to month start, validates `currency == 'KES'` and `common_unit == 'kg'`, and adds missing months through 2025 with the label `external_fill_fews_extension`. Existing non-null values are never overwritten. Overlap differences greater than 10% are printed for review only.

### 3.3 Final coverage state

The notebook's final coverage summary is grouped by `(market, commodity)` and reports the first and last populated month, total populated months, and counts by `impute_method`. It must be printed sorted by market and commodity and should contain all eight series. Because the extension reaches 2025, `last_record` should be 2025-12 for each series when the complete FEWS exports are present.

The final quality checks are:

- eight market-commodity series are retained;
- `avg_price_per_kg.isna().sum() == 0`;
- each series has a continuous monthly key from its first valid month through 2025-12;
- no duplicate `(market, commodity, month)` keys remain;
- the imputation breakdown is available for audit.

### 3.4 Training-target policy

The canonical cleaned panel is now complete for the defined forecast period. A separate supervised-learning view should still be created explicitly so target construction and feature availability are transparent:

```python
training_df = final_df.dropna(subset=['avg_price_per_kg']).copy()
```

Do not treat FEWS-derived values as equivalent to observed WFP labels without a sensitivity analysis. Report results on all complete targets and, separately, on rows whose `impute_method == 'observed'`. This is essential because the extension supplies a substantial share of the post-WFP-period data.

A useful explicit indicator is:

```python
final_df['target_available'] = final_df['avg_price_per_kg'].notna()
```

### 3.5 Outlier handling

The pipeline flags, but does not remove, IQR outliers. The latest executed output flags 38 rows. Several flagged values occur in multi-market periods such as 2017 maize price increases, which makes them plausible market shocks rather than data errors.

Outlier flags should therefore be retained for analysis and possible feature use. A flagged value should only be altered after source-level review confirms a data-entry problem. Removing genuine shocks would make the model less capable of forecasting unusual but real market conditions.

### 3.6 FEWS extension limitation

The extension-period rows tagged `external_fill_fews_extension` extend the eight series
through 2025 using manually-exported FEWS NET records. These values are filled
unconditionally for months missing from the WFP panel and are not calibrated against WFP;
overlap discrepancies are printed for manual review but do not alter the data. This is a
substantially larger and uncalibrated reliance on a secondary data source than the original
48-row interior-gap backfill tagged `external_fill_fews`. The two limitations must be
reported separately when describing data provenance and model coverage.

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

### 5.9 Build separate targets for the forecast horizons

The forecasting task has two horizons, one month and two months ahead. For each series sorted by `month`, construct:

```python
feature_df['target_h1'] = (
    feature_df.groupby(series_cols)['avg_price_per_kg'].shift(-1)
)
feature_df['target_h2'] = (
    feature_df.groupby(series_cols)['avg_price_per_kg'].shift(-2)
)
```

The predictors at month `t` must use information available at `t`; `target_h1` is the price at `t+1` and `target_h2` is the price at `t+2`. Drop rows with missing horizon-specific targets only after features have been created. The final one-month and two-month datasets will therefore have different usable row counts at the series endpoint, even though the cleaned source panel itself is complete through 2025.

Evaluate each horizon independently. Do not use the one-month prediction as an input to the two-month model unless a recursive forecasting design is deliberately chosen and validated separately. A direct two-model design, one model for each horizon, is simpler and avoids compounding one-month forecast error.

## 6. Candidate-model training design

Train one global model per horizon across all eight series, with market and commodity indicators plus group-wise lag, rolling, and calendar features. Compare a baseline linear regression against Random Forest and XGBoost. Use a stacking ensemble only after the base models have been evaluated under the same time-respecting splits.

### 6.1 Baseline linear regression

Linear regression provides the interpretable benchmark. Use lagged prices, rolling summaries, seasonal encodings, and one-hot encoded market and commodity identifiers. Standardization is optional for tree models but should be applied inside a training-only pipeline for the linear model when feature scales differ. Inspect coefficients and residuals by series and horizon; the baseline should be difficult to beat only if the feature set is weak or the series are dominated by non-linear effects.

### 6.2 Random Forest

Random Forest can capture non-linear price responses and interactions without requiring feature scaling. Tune the number of trees, maximum depth, minimum leaf size, and the feature-sampling strategy using only the training folds. Because tree ensembles do not extrapolate smoothly beyond the training range, inspect predictions near the 2025 boundary and report performance separately for each market-commodity series.

### 6.3 XGBoost

XGBoost is a strong candidate for non-linear tabular forecasting. Tune learning rate, number of estimators, maximum depth, minimum child weight, subsampling, column sampling, and regularization with chronological validation. Use early stopping only against a validation block that occurs after the training block. Keep the feature set and target definition identical to the linear and Random Forest comparisons.

### 6.4 Stacking ensemble

The stacking ensemble should combine out-of-fold predictions from the linear regression, Random Forest, and XGBoost models. Generate those predictions with time-respecting folds; never train the meta-model on in-sample base-model predictions. The meta-model should be fit only on base predictions generated for observations not used to fit the corresponding base model. Build and evaluate separate stacks for the one-month and two-month targets.

Use chronological validation within each market-commodity series and preserve the same folds for every candidate model and horizon:

- train on earlier available observations
- validate on the next chronological block
- test on the latest valid observed block for that series

The test periods should be held out until model selection is complete. The latest test block must represent the intended deployment task: predicting one or two months after the latest information available at forecast time. Report MAE, RMSE, and preferably sMAPE for each horizon and series, plus a clearly defined aggregate. Also report metrics on observed-only targets separately from metrics including FEWS-derived targets.

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
feature_df['target_h1'] = feature_df.groupby(series_cols)['avg_price_per_kg'].shift(-1)
feature_df['target_h2'] = feature_df.groupby(series_cols)['avg_price_per_kg'].shift(-2)

model_df_h1 = feature_df.dropna(
    subset=['target_h1', 'lag_1', 'lag_3', 'rolling_mean_3']
).copy()
model_df_h2 = feature_df.dropna(
    subset=['target_h2', 'lag_1', 'lag_3', 'rolling_mean_3']
).copy()
```

The current `feature_engineering.py` module is a scaffold, so these transformations must be implemented and tested before training. The important contract from cleaning is that monthly continuity, source provenance, and horizon-specific targets are explicit before model fitting begins.

## 8. Final recommendations

1. Keep `final_df` as the auditable per-series monthly panel.
2. Treat `avg_price_per_kg` as complete through 2025, but preserve and report `impute_method`.
3. Train and evaluate separate direct models for one-month and two-month horizons.
4. Compute every lag and rolling feature within a market-commodity series.
5. Validate that feature windows are continuous and do not cross invalid or unsupported history.
6. Use one global model with market and commodity identifiers, supported by per-series diagnostics.
7. Compare linear regression, Random Forest, XGBoost, and leakage-safe stacking under identical chronological folds.
8. Report metrics separately for observed-only targets and all targets, including FEWS-derived values.
9. Recompute any target-derived statistic inside each training split to prevent leakage.

The cleaned data is ready for feature engineering. The immediate next step is to implement the lag, rolling, seasonal, categorical, and horizon-target transformations, then construct reproducible chronological evaluation splits for both forecast horizons.
