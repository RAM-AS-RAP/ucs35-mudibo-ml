# Feature Engineering Analysis

## 1. Purpose and scope

This document records the feature, target, eligibility, and fold-generation steps implemented for the WFP crop-price experiment. The input is the canonical monthly panel produced by `notebooks/02_data_cleaning.ipynb` and stored at `results/artifacts/cleaned_monthly_prices_wfp_2010_2019-10.csv`.

The panel covers four markets (Nairobi, Kisumu, Mombasa, Nakuru) and two commodities (Beans, Maize). It retains WFP price history before 2010 for feature warm-up and ends in October 2019. No estimator, scaler, encoder, or hyperparameter search is fit in this feature-engineering task.

## 2. Pipeline order

The notebook in `notebooks/04_feature_engineering.ipynb` runs the stages in this order:

1. Resolve the repository and input paths; load the canonical WFP panel.
2. Parse `month` as a month-start timestamp, sort within each market-commodity series, and validate the panel keys and monthly continuity.
3. Build price-history, calendar, and identity features on the full panel, including pre-2010 rows.
4. Build the H1 label by exact calendar-month lookup on the intact panel.
5. Apply the calendar-distance guard to temporal features and targets.
6. Build the observed-endpoint H1 supervised table, with origins in January 2010-October 2019.
7. Create origin-month development/test masks and expanding-window validation boundaries.
8. Save the supervised tables, fold definitions, and summary to `results/artifacts/feature_engineering/`; when run in Colab, mirror the outputs to Drive under `MyDrive/capstone-data/feature_engineering/`.

The ordering prevents dropped rows from changing the meaning of the one-month horizon. All feature/target generation happens before the supervised rows are filtered.

## 3. Panel validation and warm-up history

The executed run loaded **1,279 rows** across eight unique market-commodity-month series, from January 2006 or 2008 through October 2019. The panel contains 1,265 observed prices and 14 `short_gap_ffill` prices. The observed coverage windows are contiguous monthly sequences; duplicate series-month keys are rejected.

Pre-2010 history is retained because three-month and twelve-month features need prior observations for early 2010 origins. Restricting the panel to 2010 before feature construction would discard that warm-up history and cause avoidable missing features at the start of the supervised window.

The panel loader requires `market`, `commodity`, `month`, `avg_price_per_kg`, and `impute_method`. Prices must be positive wherever present because their logarithms are used.

## 4. Engineered features

All temporal features are computed independently within each `(market, commodity)` series. This prevents values from one market or crop being used as another series' history.

### 4.1 Log price and lagged log returns

The base transformation is:

```text
log_price[t] = log(price[t])
log_return_lag_k[t] = log_price[t-k] - log_price[t-k-1], k in {1, 2, 3}
```

These features express recent proportional price movements on an additive log scale. Including three different historical return intervals gives models information about recent direction and momentum without using the current month’s future label.

### 4.2 Log ratios to trailing averages

For each origin month, the code shifts log price back one month, then calculates rolling means over the prior 3 or 12 values:

```text
log_ratio_3m[t]  = log_price[t-1] - mean(log_price[t-1], ..., log_price[t-3])
log_ratio_12m[t] = log_price[t-1] - mean(log_price[t-1], ..., log_price[t-12])
```

These are deviations from recent and annual historical log-price levels. Shifting before the rolling operation ensures the rolling reference excludes the current month. The implemented formula compares the previous month's log price to a mean of previous log prices; it is not a raw-price average and should be interpreted accordingly.

### 4.3 Calendar and series-identity features

- `month_sin` and `month_cos` encode month-of-year cyclically. December and January are therefore adjacent in the representation rather than opposite ends of a linear 1-12 scale.
- `year` supplies a coarse time trend so models can distinguish long-run temporal regimes.
- Fixed one-hot indicators identify the four configured markets and two commodities. The categories are known from project constants; no encoder is fitted on the full dataset.

The feature list is shared across all series. Market/crop indicators allow a pooled model to learn common patterns while retaining systematic differences between locations and commodities.

## 5. One-month target

The target is the cumulative log price change from the origin price to the exact next calendar month:

```text
target_h1[t] = log_price[t+1 calendar month] - log_price[t]
```

`build_targets` uses `(market, commodity, target_month)` lookups rather than selecting the next surviving row. This matters when a calendar month is absent: a row two months later must not silently become an H1 target. The target-method column retains provenance for the target endpoint.

The labels are log changes, not absolute prices. For reporting in KES or displaying a forecast, reconstruct the future price as:

```text
predicted_price[t+1] = origin_price[t] * exp(predicted_log_change[t])
```

Reconstructed prices are used for evaluation/display only; they are not the training labels.

## 6. Calendar-distance guard

`apply_calendar_distance_guard` checks the actual calendar months needed by each lagged return and trailing window. A lagged return is valid only if both source months exist at the expected offsets; a trailing ratio is valid only if every month in its 3- or 12-month history exists and has a price. If a dependency is missing, the derived feature is set to `NaN` rather than treating a pre-gap value as a recent lag.

H1 targets are re-looked up at their exact target months and set missing when either endpoint cannot be aligned. The boolean `consecutive_observation` marks that the current price and all required lag/rolling source prices are present at the expected calendar positions. It is a temporal-availability flag, **not** a provenance flag: an imputed price is non-missing and may satisfy the calendar check. Observed-vs-imputed endpoint eligibility is enforced separately.

In the saved run, 96 panel rows failed at least one historical-input distance/availability check; this includes warm-up rows near series starts as well as rows affected by gaps. The guard left 8 H1 target values missing, largely at panel endpoints where the future target month is outside the panel.

## 7. Observed endpoint eligibility and supervised tables

Primary supervised rows require both:

- origin month `impute_method == 'observed'`, and
- target month `target_method_h1 == 'observed'`.

This avoids training on or scoring a zero-change label created by forward filling. Imputed months can still appear in historical feature windows if their values were available at the forecast origin; they cannot be the origin or label endpoint for the primary experiment. `build_supervised_table` prints endpoint exclusions separately from rows dropped for missing required features.

The executed run restricted origins to January 2010-October 2019. Results:

| Candidate origin rows in window | Excluded for non-observed/missing endpoints | Excluded for missing features | Final supervised rows |
|---:|---:|---:|---:|
| 944 | 31 | 0 | 913 |

The counts are pooled over all eight series. `model_df_h1.csv` preserves the engineered columns and provenance needed for the next modeling task.

## 8. Development and test folds

`build_expanding_window_folds` operates on the H1 supervised table. It assigns rows by **origin month**: development origins end October 2018; final-test origins run November 2018 through October 2019 where a valid observed target exists. Every series sharing an origin month receives the same assignment. The builder creates four annual expanding validation blocks in development, beginning in 2015 after a five-year initial history.

For each validation block, training origins expand from January 2010 through the last origin whose one-month target is available by the validation origin. The output JSON stores row-position masks, source-panel indices, and the train/validation date boundaries.

| Supervised rows | Development origins | Final-test origins | Expanding folds |
|---:|---:|---:|---:|
| 913 | 825 | 88 | 4 |

**Test-window interpretation:** the fold masks split on origin month, as implemented. Consequently test origins span November 2018-September 2019 (their target months are December 2018-October 2019). These are 11 origin months, not 12 complete origin months. If the evaluation requirement is exactly November 2018-October 2019 **target months**, the fold masks must instead be defined using `target_month`; the current output should not be described as 12 target months.

## 9. Artifacts and execution status

The notebook writes the following outputs under `results/artifacts/feature_engineering/` and mirrors them to Drive when run in Colab:

- `model_df_h1.csv`: the observed-endpoint supervised table with a `fold_split` label. The committed file was generated before the two-month horizon was removed and still contains unused `target_month_h2`, `target_method_h2`, and `target_h2` columns; rerunning the notebook omits them without changing any H1 row, feature, or target.
- `folds_h1.json`: development/test masks and expanding validation boundaries aligned to table row order.
- `feature_engineering_split_summary.csv`: supervised, development, test, outside-window, and fold counts.

The saved run produced all five files. Feature construction, target building, distance guarding, eligibility filtering, and fold generation are complete. Model fitting, scaling/encoding pipelines, hyperparameter tuning, and evaluation are intentionally not performed in this task.

## 10. Rationale and next step

The sequence is designed to preserve temporal meaning and data provenance: full history supplies warm-up; relative features and log-change targets express price dynamics; exact-calendar lookup and the distance guard protect horizon alignment; observed endpoints protect label quality; and shared chronological folds prevent random temporal leakage. The resulting tables and fold definitions are the inputs to the next task, which will fit and tune persistence, linear, tree, and stacking models without changing these feature definitions.
