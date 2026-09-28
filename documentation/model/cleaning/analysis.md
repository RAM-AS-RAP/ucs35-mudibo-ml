# Data Cleaning Analysis and Feature-Engineering Implications

## 1. Purpose and scope

This document records the cleaning decisions applied to the WFP Kenya food-price data in `notebooks/02_data_cleaning.ipynb`. The objective is to produce a monthly WFP price panel for one-month and two-month-ahead forecasting while preserving source and imputation provenance.

The cleaned observation is monthly wholesale price per kilogram. The forecasting targets are one-month and two-month log price changes, defined in Section 5.9. The finalized scope contains four markets and two commodities:

- Markets: Nairobi, Kisumu, Mombasa, and Nakuru
- Commodities: Maize and Beans

`Maize (White)` is mapped to `Maize`, and `Beans (Dry)` is mapped to `Beans` after unit normalization. The resulting modeling unit is one of eight market-commodity monthly series. WFP is the sole training and evaluation source in this experiment; FEWS NET is reserved for inference-time input only.

## 2. Cleaning process

### 2.1 Standardization and scope filtering

Raw market and commodity labels are stripped and title-cased before filtering. The raw dataset is then restricted to the four configured markets. Commodity variants are relabeled into the two final crop categories.

This creates consistent categorical keys for grouping, calendar construction, aggregation, lag generation, and model evaluation. The scope filter is intentionally performed before downstream calculations so that missingness and coverage statistics describe only the modeling population.

### 2.2 Unit normalization

The pipeline creates `price_per_kg`:

- `KG`: price is retained
- `90 KG`: price is divided by 90
- Any other unit: `price_per_kg` is set to `NaN` and the unit is printed as unhandled

The latest notebook output reports unsupported units `400 G`, `50 KG`, `500 ML`, and `L`; those records receive missing normalized prices and do not enter the selected WFP series. The saved notebook does not report per-kg mean, standard deviation, minimum, and maximum separately for original labels `Maize`, `Maize (White)`, `Beans`, and `Beans (Dry)`: it relabels variants before the current aggregate summary. Add a diagnostic that preserves each source label and computes those four statistics after normalization but before relabeling. The values remain pending a clean run; do not infer them from merged-commodity summaries.

### 2.3 Price-type resolution

The pipeline prints record counts by market, commodity, and price type, then retains the preferred `Wholesale` observations. The executed target-scope data went from 3,681 rows to 3,030 wholesale rows, so **651 rows were removed**. This avoids mixing retail and wholesale prices in one target series.

### 2.4 Label-overlap resolution

After commodity relabeling, rows can collide on `(market, commodity, date)`. The pipeline averages the normalized price values for such overlaps rather than arbitrarily keeping one row. The saved run reported 1,988 rows involved in duplicates and reduced the table from 3,030 to 2,036 rows. Thus 994 distinct `(market, commodity, date)` keys required averaging (1,988 affected rows), leaving one observation per key.

### 2.5 Monthly aggregation

Daily or within-month observations are converted to the first day of the month and averaged by `(market, commodity, month)`. The resulting target is `avg_price_per_kg`.

The monthly frequency matches the observed reporting cadence: the dominant date difference is approximately 31 days. Aggregation also ensures that the downstream model has one intended target granularity rather than a mixture of daily and monthly records.

## 3. Coverage and missingness analysis

### 3.1 WFP date range and panel

The notebook filters raw WFP records to dates before November 2019, making October 2019 the inclusive endpoint. It retains earlier WFP history from 2006 (from 2008 for Nakuru) as context for feature warm-up. `MODEL_TARGET_START` is January 2010; it defines the later supervised target window, not a truncation of the canonical feature-history panel.

The executed notebook produced a 1,279-row canonical panel with eight series ending in October 2019. Kisumu, Mombasa, and Nairobi series start in January 2006; Nakuru Beans starts in January 2008 and Nakuru Maize in February 2008. The full eight-series calendar before trimming leading coverage gaps contains 1,328 rows. The endpoint is set by Kisumu Beans, the shortest-coverage series.

### 3.2 WFP-only gap handling

The notebook retains WFP-derived fills and labels their provenance:

- `short_gap_ffill`: forward-fill missing runs of up to two months from a prior WFP price;
- `medium_gap_seasonal`: fill three-to-six-month gaps from the same calendar month one or two years earlier, without using future values;
- `long_gap_unfilled`: leave unresolved gaps missing.

No FEWS gap fill or extension is part of this experiment. In the executed panel there were 1,265 observed WFP rows, 14 short-gap fills, and 49 unresolved leading-coverage rows in the full calendar. The per-series coverage step removes those leading rows, leaving the 1,279-row canonical panel with populated prices. No three-to-six-month gap occurred in the 2010-2019 WFP window, so the prior-years-only seasonal-fill logic did not trigger in this run; the logic remains present but untested at this scope.

### 3.3 Observed coverage and modeling window

Coverage metadata (`first_observed_month`, `last_observed_month`, `observed_months`) is calculated from rows labeled `observed`, before imputations are counted. The executed output has continuous coverage for each retained series between its first and last observed WFP month, through October 2019.

The January 2010-October 2019 source window contains 944 panel rows (118 months × 8 series): 932 observed WFP prices and 12 short forward-filled prices. This is a source-panel count, not the number of eligible H1/H2 labels; exact target alignment, observed origin/target requirements, and feature warm-up reduce the supervised sample. The canonical panel also carries pre-2010 rows for feature warm-up. FEWS-derived values are not training targets.

### 3.4 Outlier diagnostics

The notebook computes IQR bounds separately by market and commodity using observed WFP prices only. It flags values for review and does not remove or modify prices. The latest executed run flagged 16 rows. These flags are diagnostic metadata, not model predictors; if any target-derived threshold is used in a model, it must be recomputed from each training fold only.



## 4. Data products for the next stage

The cleaning stage should conceptually produce two related outputs:

### 4.1 Canonical WFP panel

`final_df` is the auditable monthly WFP panel written to Drive. It retains observed and WFP-imputed prices, the `impute_method` and imputation flags, diagnostic `is_outlier`, and observed-only coverage metadata. It includes pre-2010 history for feature warm-up and ends at October 2019. It is not a feature matrix or a train/test split.

### 4.2 Supervised modeling views

Create horizon-specific supervised tables downstream, after group-wise features and exact-calendar log-change targets are built. Candidate targets cover January 2010 through October 2019; development targets are January 2010-October 2018 and the final test is November 2018-October 2019. Primary rows require observed WFP prices at both origin and target. Keep imputed months as historical feature inputs, retain `impute_method` for audit, and report any imputed-endpoint sensitivity analysis separately. Count endpoint-provenance exclusions separately from missing-feature exclusions.

## 5. Implications for feature engineering

### 5.1 Compute features within each series

Compute all temporal features separately within each `(market, commodity)` series. The price-history features are log returns at lags 1-3, `log(P[t-k]) - log(P[t-k-1])`; log ratio of current price to the prior three-month mean; and log ratio of current price to the prior twelve-month mean. Add month sine/cosine, a carefully defined year/time trend, and one-hot market and commodity indicators. These price-change and ratio features expose relative movement rather than asking tree models to predict an absolute level outside their training range. They do not guarantee accuracy under an unprecedented regime.

### 5.2 Do not bridge long gaps with lags
A lag is meaningful only when the prior value is from the intended preceding calendar month. Preserve the explicit monthly index and verify month-to-month distance before accepting lags. If an unresolved gap occurs, do not let a value before the gap silently become a one- or two-month lag after it. At minimum, create a month-distance check or a `consecutive_observation` indicator; set affected lags to missing after a long gap.

Rows with unavailable required features should be excluded from a particular training design, or the model should use a separate missingness-aware feature policy. The choice must be consistent between training and inference.

### 5.3 Rolling features must exclude the current target

For the current-price-to-trailing-mean predictors, compute the three- and twelve-month means from prices strictly before month `t`: shift each series by one month, then apply the rolling window. The ratio is `log(P[t]) - log(mean(P[t-1:t-3]))` or `log(P[t]) - log(mean(P[t-1:t-12]))`. The origin price `P[t]` is allowed as an input; it must not be included in its own trailing mean. Apply the same shift-before-aggregation rule to any other rolling statistic.

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

For the current experiment, the panel contains short-gap WFP fills and no FEWS values; the executed run did not produce medium seasonal fills. For primary training and scoring, require both the origin price at month `t` and the target price at month `t+h` to have `impute_method == 'observed'`. A forward-filled endpoint can create a zero-change label by construction and bias the model toward no change. Imputed months may remain in historical feature windows, but may not be the origin or target that defines a primary supervised row. Print candidate rows excluded because one or both endpoints are imputed separately from rows excluded because required features are missing. Any short-imputed-endpoint analysis is secondary and must be reported separately. Never use `long_gap_unfilled` rows as endpoints.

### 5.7 Handle outlier flags carefully

Keep `is_outlier` as diagnostic metadata. The notebook's flags use observed-only WFP values through October 2019, but are calculated over that complete period; do not use them as predictors in historical validation. Any threshold used as a model feature must be fitted within each training fold.

### 5.8 Avoid target leakage in all preprocessing

Follow this sequence independently for H1 and H2. Apply the same month boundaries to all eight series so records from one calendar month cannot land in different folds.

1. **Reserve the final test period first.** Hold out target months November 2018 through October 2019 (12 months); do not use them to select features, transformations, model families, hyperparameters, or stopping rounds. The development/tuning target period is January 2010 through October 2018.
2. **Define each fold by forecast origin and target month.** At origin month `t`, H1 targets `t+1` and H2 targets `t+2`. A row is eligible for fitting only when its target month is no later than the fold's training-information cutoff. For an H2 forecast issued at `t`, do not train on a label from `t+1` or `t+2`, since those prices are not yet known at that origin.
3. **Build targets on the intact monthly panel.** Align targets to exact calendar months within each `(market, commodity)` series before filtering missing or imputed endpoints. Never drop rows first and then use row shifts, because the next remaining row may be more than one or two months away. Restrict development targets to 2010-01 through 2018-10 and final-test targets to 2018-11 through 2019-10, while allowing earlier WFP history to supply features when available at the forecast origin.
4. **Make each feature causal.** At origin `t`, lag features may use only prices dated at or before `t`. Rolling features must shift the series before calculating the window, so the value at `t` is not included in a predictor for `t`. Verify calendar distance and do not carry a pre-gap value across an unresolved gap as if it were a one-month lag. For block forecasts, do not use validation-period actual prices unless the evaluation explicitly simulates a new forecast origin after each actual becomes available.
5. **Fit learned transformations on training data only.** Inside each fold, fit scaling, statistical imputation, categorical vocabularies/encoders, feature selection, and any target-derived thresholds using that fold's training rows only. Apply the fitted transformations unchanged to validation rows. Calendar encodings such as month-of-year sine/cosine may be calculated deterministically from the date. Do not use the notebook's full-period `is_outlier` flags as predictors; recompute any outlier threshold from that fold's training data, or leave the flag diagnostic-only.
6. **Tune only on development folds.** Evaluate every candidate hyperparameter setting using the same expanding-window folds and horizon-specific target definitions. If early stopping is used, provide a later validation block drawn only from the current training period; never use the November 2018-October 2019 final test block for early stopping. After selecting settings, freeze them before the final test.
7. **Generate stacking meta-training data out of time.** Within each development/training window, fit Random Forest and XGBoost on earlier months and generate predictions only for a later fold they did not train on. Concatenate these chronological out-of-fold predictions and fit Ridge on those predictions and their corresponding training targets only. Do not fit Ridge on in-sample base-model predictions. For final testing, refit the base learners on the full eligible pre-test training data and predict the untouched 2019 test origins; use Ridge trained from out-of-fold predictions generated entirely within that pre-test data.
8. **Open the test once for final scoring.** After model and feature choices are frozen, fit on eligible development data through October 2018, predict origins corresponding to November 2018-October 2019 target months, and score matching observed endpoint pairs. Report H1 and H2 separately. Per-series scores over 12 months are indicative given the small sample and must be read alongside the pooled cross-series metric. Do not revise the pipeline based on final-test results and continue to call that set an untouched test.

Any information learned from target values must follow these fold boundaries. This includes scaling parameters, outlier limits, statistical imputers, and feature-selection decisions, not just the estimator itself.

### 5.9 Build separate targets for the forecast horizons

Train on cumulative log price changes, not absolute prices. For origin month `t`, define `target_h1 = log(P[t+1]) - log(P[t])` and `target_h2 = log(P[t+2]) - log(P[t])`, where `t+1` and `t+2` mean exact calendar months in the same series. Build the labels on the intact monthly panel before filtering rows. Use an exact `(market, commodity, target_month)` calendar join, or first assert a contiguous monthly index before using row shifts. Do not drop missing or imputed rows before target alignment.

The model predicts log changes. For KES-scale reporting and farmer-facing display only, reconstruct `predicted_price = origin_price * exp(predicted_change)`. Reconstructed prices are not training targets. Fit separate direct H1 and H2 models; do not feed an H1 prediction into H2 unless a recursive design is separately specified and validated.

## 6. Candidate-model training design

Train one global model per horizon across the eight series, with the ratio/return features and market/crop indicators. Use identical time-respecting folds for every candidate model and horizon.

### 6.1 Persistence baseline

The naive baseline predicts zero log change for both horizons, equivalent to `predicted_price = origin_price`. Evaluate it on exactly the same eligible endpoint pairs and with the same metrics as every learned model. Report it first in the Chapter 5 model-comparison table; a model that cannot outperform persistence has not demonstrated forecasting value.

The Linear Regression baseline is the interpretable learned benchmark. Fit it to the same log-change targets and features, with scaling inside each training-fold pipeline. Inspect coefficients and residuals by horizon and series.

### 6.2 Random Forest

Random Forest captures nonlinearities and interactions without requiring feature scaling. Tune tree count, depth, leaf size, and feature sampling only on development folds. The log-change target is not bounded by historical absolute price levels, so reconstruction can produce a forecast above the largest training price. This avoids a hard level ceiling by design but does not guarantee accuracy when the change itself is unprecedented.

### 6.3 XGBoost

XGBoost is a nonlinear tabular candidate. Tune learning rate, estimator count, depth, child weight, subsampling, column sampling, and regularization with chronological development folds. Any early-stopping block must be later than the corresponding training block and earlier than the final test. Use the same change target and features as the other candidates.

### 6.4 Metrics and final evaluation

For each horizon, report two metric groups on the same observed origin/target pairs:

- On reconstructed KES prices, compute MAPE, MAE, and RMSE. **MAPE is the primary percentage metric** for consistency with Chapter 3 and NFR-02.
- On the log-change target before reconstruction, compute MAE and RMSE.
- Report directional accuracy: the proportion of rows where predicted and actual directions agree as rise, fall, or no change. Define and freeze a small no-change threshold before opening the final test.

The development/tuning target period is January 2010 through October 2018. The final test target period is November 2018 through October 2019 (12 months). Use expanding-window folds by target month, grouped across all series. Per-series metrics over 12 test months are indicative because of the small sample; read them alongside the pooled cross-series metric. Freeze all choices before evaluating the final test once.

### 6.5 Stacking ensemble

The specified stack uses Random Forest and XGBoost as base learners and Ridge as the meta-model. For each horizon, fit Ridge only on chronological out-of-fold base predictions from the development/training period. Never train the meta-model on in-sample base predictions. Tune base-model settings and any Ridge regularization using development folds only. Evaluate the stack on the same final-test endpoint pairs as persistence and the individual models.

### 6.6 FEWS NET at inference only

FEWS NET values are excluded from model training, tuning, and primary scoring. At inference, select a FEWS product variant that matches the merged WFP commodity definition (white maize, wholesale) and use `common_unit_price`, normalized to the same KES/kg convention. Training and the live inference pipeline must call the same shared feature-engineering module so feature definitions cannot diverge between WFP and FEWS inputs. Describe any WFP-versus-FEWS overlap comparison or out-of-time evaluation on FEWS data factually if it is actually performed; do not imply such validation has occurred by default.

## 7. Recommended training-table pattern

```python
import numpy as np
import pandas as pd

series_cols = ['market', 'commodity']
feature_df = final_df.sort_values(series_cols + ['month']).copy()
feature_df['month'] = pd.to_datetime(feature_df['month'])

# Row shifts below are valid only after confirming each series is contiguous.
for _, series in feature_df.groupby(series_cols, sort=False):
    expected = pd.date_range(series['month'].min(), series['month'].max(), freq='MS')
    assert series['month'].reset_index(drop=True).equals(pd.Series(expected))

feature_df['log_price'] = np.log(feature_df['avg_price_per_kg'])
grouped_log = feature_df.groupby(series_cols, sort=False)['log_price']
for lag in (1, 2, 3):
    feature_df[f'log_return_lag_{lag}'] = (
        grouped_log.shift(lag) - grouped_log.shift(lag + 1)
    )

grouped_price = feature_df.groupby(series_cols, sort=False)['avg_price_per_kg']
mean_3m = grouped_price.transform(
    lambda values: values.shift(1).rolling(3, min_periods=3).mean()
)
mean_12m = grouped_price.transform(
    lambda values: values.shift(1).rolling(12, min_periods=12).mean()
)
feature_df['log_ratio_trailing_3m'] = feature_df['log_price'] - np.log(mean_3m)
feature_df['log_ratio_trailing_12m'] = feature_df['log_price'] - np.log(mean_12m)
month_number = feature_df['month'].dt.month
feature_df['month_sin'] = np.sin(2 * np.pi * month_number / 12)
feature_df['month_cos'] = np.cos(2 * np.pi * month_number / 12)
feature_df['time_index'] = (
    (feature_df['month'].dt.year - 2010) * 12 + month_number - 1
)

# Join labels by exact target calendar month, not by the next surviving row.
target_lookup = feature_df[
    series_cols + ['month', 'log_price', 'avg_price_per_kg', 'impute_method']
].rename(columns={
    'month': 'target_month',
    'log_price': 'target_log_price',
    'avg_price_per_kg': 'target_price',
    'impute_method': 'target_method',
})
feature_cols = [
    'log_return_lag_1', 'log_return_lag_2', 'log_return_lag_3',
    'log_ratio_trailing_3m', 'log_ratio_trailing_12m',
    'month_sin', 'month_cos', 'time_index', 'market', 'commodity',
]

for horizon in (1, 2):
    rows = feature_df.copy()
    rows['origin_price'] = rows['avg_price_per_kg']
    rows['target_month'] = rows['month'] + pd.DateOffset(months=horizon)
    rows = rows.merge(target_lookup, on=series_cols + ['target_month'], how='left')
    rows[f'target_h{horizon}'] = rows['target_log_price'] - rows['log_price']
    in_target_window = rows['target_month'].between('2010-01-01', '2019-10-01')
    endpoint_observed = (
        rows['impute_method'].eq('observed')
        & rows['target_method'].eq('observed')
    )
    candidates = rows.loc[in_target_window]
    excluded_endpoint_count = int((~endpoint_observed.loc[in_target_window]).sum())
    primary_rows = rows.loc[in_target_window & endpoint_observed].copy()
    missing_feature_count = int(primary_rows[feature_cols].isna().any(axis=1).sum())
    model_rows = primary_rows.dropna(subset=feature_cols).copy()
    print(
        f'H{horizon}: excluded for imputed/missing endpoints={excluded_endpoint_count}; '
        f'excluded for missing features={missing_feature_count}; '
        f'eligible rows={len(model_rows)}'
    )
```

One-hot encoding and scaling are fitted inside each training fold, never on the full panel. Reconstruct only for evaluation or display using `origin_price * exp(predicted_log_change)`. The current `feature_engineering.py` is a scaffold; implement and test these transformations downstream from cleaning. Persist the fitted feature/model pipeline so offline evaluation and live inference use identical definitions.

## 8. Final recommendations

1. Use the WFP panel through October 2019, retaining earlier WFP history for feature warm-up; FEWS is inference-only.
2. Build causal log-return and current-to-trailing-mean ratio features within each series, with the month index validated as contiguous.
3. Construct H1/H2 cumulative log-change targets by exact calendar-month alignment before filtering rows.
4. For primary training and scoring, require observed WFP prices at both the origin and target; report excluded endpoint rows separately from rows dropped for missing features.
5. Use zero-log-change persistence as the first benchmark and report MAPE on reconstructed KES prices as the primary percentage metric, alongside price/log-change error and directional accuracy.
6. Tune with expanding folds over January 2010-October 2018; reserve November 2018-October 2019 for the single final test. Report pooled and per-series metrics, interpreting 12-month per-series results cautiously.
7. Compare Linear Regression, Random Forest, XGBoost, and the RF/XGBoost stack with Ridge meta-model on identical folds. Fit stacking only from chronological out-of-fold base predictions.
8. Keep outlier flags diagnostic, fit all learned preprocessing inside each training fold, and use the same shared feature module at WFP training and FEWS inference.

The canonical WFP panel provides the source history for feature engineering. Feature generation, supervised-view construction, fitting, tuning, and evaluation remain downstream processes; they are not performed by the cleaning notebook.
