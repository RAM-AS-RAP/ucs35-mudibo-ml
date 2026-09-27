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

### 3.1 WFP date range and panel

The notebook filters raw WFP records to dates before November 2019, making October 2019 the inclusive endpoint. It retains earlier WFP history from 2006 (from 2008 for Nakuru) as context for feature warm-up. `MODEL_TARGET_START` is January 2010; it defines the later supervised target window, not a truncation of the canonical feature-history panel.

The executed notebook produced a 1,279-row canonical panel with eight series ending in October 2019. Kisumu, Mombasa, and Nairobi series start in January 2006; Nakuru Beans starts in January 2008 and Nakuru Maize in February 2008. The full eight-series calendar before trimming leading coverage gaps contains 1,328 rows.

### 3.2 WFP-only gap handling

The notebook retains WFP-derived fills and labels their provenance:

- `short_gap_ffill`: forward-fill missing runs of up to two months from a prior WFP price;
- `medium_gap_seasonal`: fill three-to-six-month gaps from the same calendar month one or two years earlier, without using future values;
- `long_gap_unfilled`: leave unresolved gaps missing.

No FEWS gap fill or extension is part of this experiment. In the executed panel there were 1,265 observed WFP rows, 14 short-gap fills, and 49 unresolved leading-coverage rows in the full calendar. The per-series coverage step removes those leading rows, leaving the 1,279-row canonical panel with populated prices. No medium seasonal fills occurred in that run.

### 3.3 Observed coverage and modeling window

Coverage metadata (`first_observed_month`, `last_observed_month`, `observed_months`) is calculated from rows labeled `observed`, before imputations are counted. The executed output has continuous coverage for each retained series between its first and last observed WFP month, through October 2019.

The 2010-01 through 2019-10 modeling target window contains 944 rows (118 months × 8 series): 932 observed WFP targets and 12 short forward-filled targets. The saved canonical panel also carries pre-2010 rows for lag and rolling-feature warm-up. There are no FEWS-derived targets in this experiment.

### 3.4 Outlier diagnostics

The notebook computes IQR bounds separately by market and commodity using observed WFP prices only. It flags values for review and does not remove or modify prices. The latest executed run flagged 16 rows. These flags are diagnostic metadata, not model predictors; if any target-derived threshold is used in a model, it must be recomputed from each training fold only.

### 3.5 Current output and checks

The final notebook cell writes the canonical panel only to Google Drive at `MyDrive/capstone-data/cleaned_monthly_prices_wfp_2010_2019-10.csv`. It checks that there are eight series, unique market-commodity-month keys, recognized WFP imputation methods, complete imputation flags, 944 rows in the target window, and no external-source methods in that window.

The 49 leading-coverage calendar rows left missing before trimming are expected. They must not be treated as zero prices or as supervised targets. The final canonical panel retains monthly continuity within each series' observed coverage window.

## 4. Data products for the next stage

The cleaning stage should conceptually produce two related outputs:

### 4.1 Canonical WFP panel

`final_df` is the auditable monthly WFP panel written to Drive. It retains observed and WFP-imputed prices, the `impute_method` and imputation flags, diagnostic `is_outlier`, and observed-only coverage metadata. It includes pre-2010 history for feature warm-up and ends at October 2019. It is not a feature matrix or a train/test split.

### 4.2 Supervised modeling views

Create horizon-specific supervised tables downstream, after group-wise features and exact-calendar targets are built. The candidate target months are January 2010 through October 2019. Keep `impute_method` for audit; use observed WFP targets for primary scoring and report any short-imputed-target result separately. Drop rows only after feature and target alignment, and only when the needed target or predictors are unavailable.

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
A lag is meaningful only when the prior value is from the intended preceding calendar month. Preserve the explicit monthly index and verify month-to-month distance before accepting lags. If an unresolved gap occurs, do not let a value before the gap silently become a one- or two-month lag after it. At minimum, create a month-distance check or a `consecutive_observation` indicator; set affected lags to missing after a long gap.

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

For the current experiment, the panel contains short-gap WFP fills and no FEWS values; the executed run did not produce medium seasonal fills. Use observed WFP targets for primary metrics. A secondary run may include the 12 short-filled target rows, but label and report it separately. Never use `long_gap_unfilled` rows as targets.

### 5.7 Handle outlier flags carefully

Keep `is_outlier` as diagnostic metadata. The notebook's flags use observed-only WFP values through October 2019, but are calculated over that complete period; do not use them as predictors in historical validation. Any threshold used as a model feature must be fitted within each training fold.

### 5.8 Avoid target leakage in all preprocessing

Follow this sequence independently for H1 and H2. Apply the same month boundaries to all eight series so records from one calendar month cannot land in different folds.

1. **Reserve the final test period first.** Hold out target months in 2019; do not use them to select features, transformations, model families, hyperparameters, or stopping rounds. Use expanding-window folds with training and validation target months from 2010 through 2018 for development and tuning.
2. **Define each fold by forecast origin and target month.** At origin month `t`, H1 targets `t+1` and H2 targets `t+2`. A row is eligible for fitting only when its target month is no later than the fold's training-information cutoff. For an H2 forecast issued at `t`, do not train on a label from `t+1` or `t+2`, since those prices are not yet known at that origin.
3. **Build targets on the intact monthly panel.** Align targets to exact calendar months within each `(market, commodity)` series before filtering missing or imputed targets. Never drop rows first and then use `shift(-1)` or `shift(-2)`, because the next remaining row may be more than one or two months away. Apply the 2010-01 through 2019-10 restriction to target months, while allowing earlier WFP history to supply features when that history would have been available at the forecast origin.
4. **Make each feature causal.** At origin `t`, lag features may use only prices dated at or before `t`. Rolling features must shift the series before calculating the window, so the value at `t` is not included in a predictor for `t`. Verify calendar distance and do not carry a pre-gap value across an unresolved gap as if it were a one-month lag. For block forecasts, do not use validation-period actual prices unless the evaluation explicitly simulates a new forecast origin after each actual becomes available.
5. **Fit learned transformations on training data only.** Inside each fold, fit scaling, statistical imputation, categorical vocabularies/encoders, feature selection, and any target-derived thresholds using that fold's training rows only. Apply the fitted transformations unchanged to validation rows. Calendar encodings such as month-of-year sine/cosine may be calculated deterministically from the date. Do not use the notebook's full-period `is_outlier` flags as predictors; recompute any outlier threshold from that fold's training data, or leave the flag diagnostic-only.
6. **Tune only on development folds.** Evaluate every candidate hyperparameter setting using the same expanding-window folds and horizon-specific target definitions. If early stopping is used, provide a later validation block drawn only from the current training period; never use the final 2019 test block for early stopping. After selecting settings, freeze them before the final test.
7. **Generate stacking meta-training data out of time.** Within each development/training window, fit Random Forest and XGBoost on earlier months and generate predictions only for a later fold they did not train on. Concatenate these chronological out-of-fold predictions and fit Ridge on those predictions and their corresponding training targets only. Do not fit Ridge on in-sample base-model predictions. For final testing, refit the base learners on the full eligible pre-test training data and predict the untouched 2019 test origins; use Ridge trained from out-of-fold predictions generated entirely within that pre-test data.
8. **Open the test once for final scoring.** After model and feature choices are frozen, fit on all eligible pre-2019 data, predict the 2019 origins, and calculate metrics against the corresponding targets. Report H1 and H2 separately, with primary metrics on `impute_method == 'observed'`; show WFP-imputed target scores separately if included. Do not revise the pipeline based on 2019 results and continue to call that set an untouched test.

Any information learned from target values must follow these fold boundaries. This includes scaling parameters, outlier limits, statistical imputers, and feature-selection decisions, not just the estimator itself.

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

The predictors at month `t` must use information available at `t`; `target_h1` is the price in the next calendar month and `target_h2` is the price two calendar months later. The canonical panel retains monthly rows, so build targets before filtering by target provenance or dropping rows; otherwise a later month can be mistaken for the next month. Limit target dates to 2010-01 through 2019-10 and evaluate H1 and H2 independently.

Evaluate each horizon independently. Do not use the one-month prediction as an input to the two-month model unless a recursive forecasting design is deliberately chosen and validated separately. A direct two-model design, one model for each horizon, is simpler and avoids compounding one-month forecast error.

## 6. Candidate-model training design

Train one global model per horizon across all eight series, with market and commodity indicators plus group-wise lag, rolling, and calendar features. Compare Linear Regression, Random Forest, XGBoost, and a stacking ensemble with Random Forest and XGBoost as base learners and Ridge as the meta-model. Fit separate direct models for H1 and H2 and use identical time-respecting splits within each horizon.

### 6.1 Baseline linear regression

Linear regression provides the interpretable benchmark. Use lagged prices, rolling summaries, seasonal encodings, and one-hot encoded market and commodity identifiers. Standardization is optional for tree models but should be applied inside a training-only pipeline for the linear model when feature scales differ. Inspect coefficients and residuals by series and horizon; the baseline should be difficult to beat only if the feature set is weak or the series are dominated by non-linear effects.

### 6.2 Random Forest

Random Forest can capture non-linear price responses and interactions without requiring feature scaling. Tune the number of trees, maximum depth, minimum leaf size, and the feature-sampling strategy using only the training folds. Because tree ensembles do not extrapolate smoothly beyond the training range, inspect predictions in the 2019 holdout and report performance separately for each market-commodity series.

### 6.3 XGBoost

XGBoost is a strong candidate for non-linear tabular forecasting. Tune learning rate, number of estimators, maximum depth, minimum child weight, subsampling, column sampling, and regularization with chronological validation. Use early stopping only against a validation block that occurs after the training block. Keep the feature set and target definition identical to the linear and Random Forest comparisons.

### 6.4 Stacking ensemble

The requested stack uses Random Forest and XGBoost as base learners and Ridge as meta-model. Fit Ridge only on time-ordered out-of-fold base predictions; never use in-sample base predictions to train the meta-model. Build a separate stack for each horizon.

Use expanding-window chronological folds over 2010-2018 for tuning, then keep 2019 targets as the final holdout. Apply each fold by target month across all series so observations from the same month cannot be split between training and validation. Hyperparameter selection must use only development folds; keep the final 2019 test untouched until model choices are fixed. Report MAE, RMSE, and sMAPE (with a defined zero-price policy) separately by horizon and series, plus a clearly defined pooled metric. Primary metrics should use observed WFP targets; report imputed-target metrics separately if included.

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

The current `feature_engineering.py` module is a scaffold, so these transformations must be implemented and tested before training. The cleaning output provides monthly WFP history through October 2019, provenance labels, and observed-only coverage metadata; it does not provide model features, horizon targets, or a train/test split.

## 8. Final recommendations

1. Use the Drive-saved WFP panel through October 2019; retain pre-2010 rows as feature warm-up.
2. Preserve WFP short fills and provenance, but score observed WFP targets as the primary result.
3. Build within-series lag and rolling features, checking calendar distance across missing months.
4. Construct separate exact-calendar H1 and H2 targets before filtering rows.
5. Tune with expanding-window folds over the development period and reserve 2019 target months for final evaluation.
6. Compare Linear Regression, Random Forest, XGBoost, and the RF/XGBoost plus Ridge stack with the same folds.
7. Keep outlier flags diagnostic and fit all target-derived preprocessing within each training fold.

The canonical panel is ready to serve as input to feature engineering for the WFP-only experiment. Feature generation, supervised-table construction, model fitting, hyperparameter tuning, and evaluation remain downstream steps and are not performed by the cleaning notebook.
