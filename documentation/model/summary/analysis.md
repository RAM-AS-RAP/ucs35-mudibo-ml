# End-to-End Training Process Summary

## 1. Overview

This document summarizes every stage of the AgriSence model-training process: data cleaning, feature engineering, model training, stacking, and final evaluation. It also explains how data leakage was prevented and what is reported. Each stage has a detailed analysis, linked in its section; this summary records what was done and why, with the key numbers.

**Forecasting task.** Predict the one-month-ahead change in the monthly wholesale price per kilogram of maize and beans in four Kenyan markets (Nairobi, Kisumu, Mombasa, Nakuru). This gives eight market-commodity series, modelled together by one global model.

**Target.** The cumulative log-price change from the origin month $t$ to the next calendar month:

$$
y_t = \log(P_{t+1}) - \log(P_t).
$$

A predicted change is converted to a price for display and price-based evaluation:

$$
\hat P_{t+1} = P_t \exp(\hat y_t).
$$

**Final model.** A stacking ensemble in which Random Forest and XGBoost forecasts are combined by a Ridge meta-model. A scaled Linear Regression is the comparison baseline.

**Data.** WFP Kenya market prices only, ending October 2019. FEWS NET prices are used only as live inputs when the deployed pipeline makes forecasts; they are never used for training, tuning, or evaluation.

| Stage | Notebook | Detailed analysis |
| --- | --- | --- |
| Data cleaning | [02_data_cleaning.ipynb](../../../notebooks/02_data_cleaning.ipynb) | [cleaning/analysis.md](../cleaning/analysis.md) |
| Feature engineering | [04_feature_engineering.ipynb](../../../notebooks/04_feature_engineering.ipynb) | [feature_engineering/analysis.md](../feature_engineering/analysis.md) |
| Model training and stacking | [05_model_training.ipynb](../../../notebooks/05_model_training.ipynb) | [model_training/analysis.md](../model_training/analysis.md) |
| Final evaluation | [06_final_evaluation.ipynb](../../../notebooks/06_final_evaluation.ipynb) | [model_evaluation/analysis.md](../model_evaluation/analysis.md) |
| Explainability | [07_shap_explainability.ipynb](../../../notebooks/07_shap_explainability.ipynb) | [shap/analysis.md](../shap/analysis.md) |

## 2. Data Cleaning

**Input.** The raw WFP Kenya food-price dataset.

**Activities.**

1. **Standardized labels and scope.** Market and commodity names were trimmed and title-cased, then restricted to the four markets. `Maize (White)` was merged into `Maize` and `Beans (Dry)` into `Beans`.
2. **Normalized units to KES per kilogram.** `KG` prices were kept and `90 KG` prices divided by 90. Other units (`400 G`, `50 KG`, `500 ML`, `L`) were set to missing and did not enter the series.
3. **Kept wholesale prices only.** Retail and wholesale prices were not mixed in one series. This removed 651 rows (3,681 → 3,030).
4. **Resolved label overlaps.** Merging commodity variants created rows sharing the same market, commodity, and date. Their prices were averaged rather than one being picked arbitrarily (3,030 → 2,036 rows).
5. **Aggregated to monthly.** Observations were averaged per market, commodity, and calendar month, matching the roughly monthly WFP reporting cadence. The result is `avg_price_per_kg`.
6. **Set the date range.** Data end in October 2019. Earlier history (from 2006, or 2008 for Nakuru) was kept for feature warm-up, even though supervised examples start in January 2010.
7. **Handled gaps using WFP data only, with provenance recorded.**
   - Gaps of up to two months were forward-filled (`short_gap_ffill`; 14 rows).
   - Three-to-six-month gaps would be filled from the same month in earlier years; none occurred.
   - Longer gaps stay missing (`long_gap_unfilled`).
   - Every row records its `impute_method`, so filled prices can be excluded as forecast origins or targets later.
8. **Flagged outliers without changing them.** IQR bounds per series flagged 16 rows for review. Prices were not modified, and the flags are not model inputs.

**Output.** A canonical monthly panel of 1,279 rows across the eight series (1,265 observed prices and 14 short-gap fills), saved as `results/artifacts/cleaned_monthly_prices_wfp_2010_2019-10.csv`. The January 2010–October 2019 window contains 944 of these rows.

## 3. Feature Engineering

**Input.** The canonical panel.

**Activities.**

1. **Validated the panel.** Duplicate series-month keys were rejected, and prices had to be positive because they are log-transformed.
2. **Built 14 predictors within each series.** Every temporal feature is computed separately per market-commodity series, so one series never supplies another's history.

   | Group | Features | Definition |
   | --- | --- | --- |
   | Recent momentum | `log_return_lag_1`, `_2`, `_3` | $L_{t-k} - L_{t-k-1}$ for $k = 1, 2, 3$, where $L$ is log price |
   | Position vs. recent average | `log_ratio_3m`, `log_ratio_12m` | Previous month's log price minus the mean of the previous 3 or 12 log prices |
   | Seasonality | `month_sin`, `month_cos` | Cyclic month-of-year encoding |
   | Trend | `year` | Calendar year |
   | Identity | 4 market and 2 commodity indicators | Fixed one-hot columns from the project's market and crop lists |

3. **Built the target by exact calendar lookup.** `target_h1` was looked up at exactly the next calendar month on the intact panel, before any rows were filtered. If a month is missing, the target is missing; the next available row is never used as "next month".
4. **Applied a calendar-distance guard.** Lags, trailing ratios, and targets were set to missing whenever a required source month was absent, so a value from before a gap can never pose as a recent lag.
5. **Kept only rows with observed endpoints.** Both the origin price and the target price had to be genuinely observed, not filled. A forward-filled target would create an artificial zero-change label. This excluded 31 of the 944 origins in the January 2010–October 2019 window. No rows were lost to missing features.
6. **Assigned development and test sets by origin month.** Every series at the same origin month falls on the same side of a boundary.

   | Set | Origin months | Rows |
   | --- | --- | ---: |
   | Development | January 2010 – October 2018 | 825 |
   | Final test | November 2018 – September 2019 | 88 (11 per series) |

   October 2019 origins have no observed next-month target, so the test set has 11 origin months.

7. **Built expanding-window validation folds within development.** After a five-year initial span, four annual validation blocks were created. Training always starts in January 2010 and ends at the last origin whose one-month target falls no later than the first validation month.

   | Fold | Training origins | Training rows | Validation origins | Validation rows |
   | --- | --- | ---: | --- | ---: |
   | 1 | Jan 2010 – Dec 2014 | 459 | Jan – Dec 2015 | 96 |
   | 2 | Jan 2010 – Dec 2015 | 555 | Jan – Dec 2016 | 94 |
   | 3 | Jan 2010 – Dec 2016 | 649 | Jan – Dec 2017 | 96 |
   | 4 | Jan 2010 – Dec 2017 | 745 | Jan – Oct 2018 | 80 |

**Output.** `model_df_h1.csv` (913 supervised rows) and `folds_h1.json` (development/test masks and fold boundaries) in `results/artifacts/feature_engineering/`. No model, scaler, or encoder is fitted at this stage.

## 4. Model Training

All training uses the 825 development rows only. The 88 test rows are counted but never used.

### 4.1 Baselines

- **Persistence** predicts zero change (next month's price equals this month's). It has no parameters and is used as a development-stage reference.
- **Linear Regression** is a `StandardScaler` + `LinearRegression` pipeline. A fresh pipeline is fitted inside each fold, so the scaler only sees that fold's training rows. It is not tuned.

### 4.2 Tuning Random Forest and XGBoost

Both base learners were tuned with an exhaustive grid search over the four chronological folds, minimizing mean squared error. `random_state=42` was fixed.

| Model | Grid size | Selected settings |
| --- | ---: | --- |
| Random Forest | 24 combinations (96 fits) | 400 trees, max depth 6, min leaf size 10, `sqrt` features |
| XGBoost | 288 combinations (1,152 fits) | 200 trees, max depth 3, learning rate 0.01, min child weight 1, subsample 1.0, column sample 1.0, L2 penalty 5 |

Both selections favour shallow, heavily regularized models, which suits a panel of under a thousand rows.

### 4.3 Development Results

These are development-period diagnostics in log-change units, not final test results.

| Model | MAE | RMSE | How it was scored |
| --- | ---: | ---: | --- |
| Persistence | 0.059214 | 0.079878 | Mean over the four folds |
| Linear Regression | 0.058371 | 0.076907 | Mean over the four folds |
| Random Forest | 0.056913 | 0.076619 | Pooled out-of-fold predictions |
| XGBoost | 0.056986 | 0.076524 | Pooled out-of-fold predictions |
| Ridge stack | 0.056209 | 0.076063 | Fit diagnostic on the rows Ridge was trained on |

The rows are scored in different ways, so they show broad agreement rather than a strict ranking. The stack row is in-sample for the meta-model.

## 5. How the Stacking Ensemble Was Trained

The ensemble predicts:

$$
\hat y_{\text{stack}} = b + w_{RF}\,\hat y_{RF} + w_{XGB}\,\hat y_{XGB}.
$$

Ridge learns the intercept $b$ and weights $w$ from the two base-model forecasts only, not from the 14 original features.

**Step 1: generate out-of-fold (OOF) base predictions.** For each of the four development folds, a fresh Random Forest and XGBoost (with the selected settings) were fitted on the fold's training rows and asked to predict only that fold's later validation rows. Concatenating the four validation blocks gives 366 OOF rows (2015 to October 2018). Each row has an RF forecast and an XGBoost forecast from models that never saw it. Rows from 2010–2014 receive no OOF predictions, because they only ever appear in training.

**Step 2: tune and fit Ridge on the OOF predictions.** Ridge was fitted to the two OOF prediction columns against the actual targets. Its penalty `alpha` was chosen from {0.1, 1.0, 10.0} using a time-ordered split of the OOF months, keeping all rows from a month together. The selected `alpha` was **0.1**. The RF and XGBoost OOF forecasts were highly correlated (Pearson 0.927), which limits how much a stack can gain from combining them.

**Step 3: final refit before testing.** With all settings frozen, the final evaluation:

1. regenerated the OOF predictions and refitted Ridge with `alpha` = 0.1;
2. refitted Random Forest and XGBoost on all 825 development rows;
3. refitted the Linear Regression baseline on the same rows;
4. saved everything as one bundle (`stacking_h1.pkl`) with the feature order.

The final Ridge combiner is:

$$
\hat y_{\text{stack}} = -0.00759 + 0.3786\,\hat y_{RF} + 0.4544\,\hat y_{XGB}.
$$

The weights are not constrained to sum to one. They sum to about 0.83, so Ridge slightly shrinks the combined forecast toward its intercept.

**Step 4: predict.** At prediction time, the refitted Random Forest and XGBoost each forecast from the 14 features, and Ridge combines the two forecasts with the weights above.

## 6. Final Evaluation

**Procedure.** The frozen, refitted models predicted the 88 test rows once. Before predicting, a closure marker (`test_closed_h1.json`) was written. The evaluation code refuses to run again while the marker or saved results exist, so the test set cannot be reused for tuning. Metrics can be recomputed later from the saved predictions (`test_predictions_h1.csv`) without refitting or predicting again.

**What is reported.** Linear Regression and the stacking ensemble only. Random Forest and XGBoost are components of the stack and are not reported separately; persistence is not part of the final comparison. Metrics are pooled over all 88 test rows rather than reported per series, because 11 rows per series are too few for stable per-series estimates.

**Metrics.**

- **Reconstructed prices (KES/kg):** MAE, MSE, RMSE, MAPE.

**Results: reconstructed prices**

| Model | MAE (KES/kg) | MSE ((KES/kg)²) | RMSE (KES/kg) | MAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 4.921290 | 51.371490 | 7.167391 | 8.734540 |
| Stacking ensemble | **4.847952** | **51.105719** | **7.148826** | **8.648759** |

**Interpretation.**

- Stacking is better on all four price metrics.
- In practical terms, the stacking ensemble's typical error is about 4.85 KES/kg, or 8.6% of the actual price, about 1.5% lower than Linear Regression's price MAE.
- The margins are small, and no significance tests or uncertainty intervals were computed. The results describe this single test year.
- Test errors are clearly higher than development errors. The test year was unusual: prices rose on average about 4% a month, against roughly 0.1% across development origins. Models trained on the earlier, flatter period could not anticipate this.

## 7. How Data Leakage Was Prevented

Leakage means information from the future, or from the test set, reaching the model during training. The safeguards below apply at every stage.

### 7.1 Held-Out Test Set

- The test period (origins November 2018 – September 2019) was fixed before any modelling and used for nothing except one final prediction.
- Feature choices, model families, hyperparameters, Ridge `alpha`, and OOF generation used development rows only.
- The closure marker blocks a second evaluation run. Re-scoring reads saved predictions only.

### 7.2 Features Use Only Past Information

- Lags use prices strictly before the origin month, and trailing means are shifted one month before averaging, so the window excludes month $t$.
- Features are computed within each series, so one market's later data cannot leak into another's history.
- The calendar-distance guard stops a value from before a gap being treated as a recent lag.
- Calendar features come from the date, not row position.
- The same feature code is used in training and in live inference, so definitions cannot drift apart.

### 7.3 Targets Are Clean and Exactly Aligned

- Targets are looked up by exact calendar month on the intact panel, before filtering, so a dropped month can never shift a label.
- Only observed origin and target prices define a supervised row. A forward-filled target would create an artificial "no change" label.

### 7.4 Nothing Is Fitted on Data It Will Be Scored On

- `StandardScaler` is fitted inside each fold's training rows. Market and crop indicators are fixed from the project lists, so no encoder is fitted on the full data.
- Outlier flags were computed over the whole period, so they are kept as diagnostics and never used as features.

### 7.5 Time-Ordered Validation

- Expanding-window folds always train on earlier origins and validate on later ones.
- The training cutoff is set so every training target is known by the first validation month.
- All series at the same origin month stay on the same side of each boundary.

### 7.6 Stacking Without In-Sample Leakage

- Ridge is trained only on out-of-fold base predictions. If it were trained on predictions made by base models that had already seen those rows, it would learn to trust their overconfident in-sample fit.
- Ridge's own `alpha` search splits OOF rows by month in time order.
- For the final test, the base learners are refitted on development data only, and Ridge uses OOF predictions generated entirely within development data.

### 7.7 Remaining Caveats

- **Split boundary.** Rows are assigned by origin month. The last development origin (October 2018) has a November 2018 target, which overlaps the first test month. Whether that price would be known when a November forecast is issued depends on the issue-time convention. The results should not be described as having a strict target-time embargo.
- **Post-selection OOF scores.** Hyperparameters were chosen on the same folds that produce the OOF predictions. Development OOF scores are therefore slightly optimistic diagnostics, not nested cross-validation estimates. This does not affect the final test, which was untouched.
- **Ridge time splits.** The Ridge `alpha` search does not apply an explicit horizon embargo. For a one-month target this matches the base-fold convention (the latest training target equals the first validation month), so it has no practical effect.

## 8. Reporting and Saved Artifacts

| Stage | Artifact | Location |
| --- | --- | --- |
| Cleaning | Canonical monthly panel | `results/artifacts/cleaned_monthly_prices_wfp_2010_2019-10.csv` |
| Feature engineering | Supervised table, fold definitions, split summary | `results/artifacts/feature_engineering/` (`model_df_h1.csv`, `folds_h1.json`, `feature_engineering_split_summary.csv`) |
| Training | Tuned settings, development metrics, Ridge and Linear Regression fits | `MyDrive/capstone-data/model_training/` (`tuning_h1.json`, `oof_metrics_h1.json`, and related files) |
| Evaluation | Pooled price results table, test predictions, manifest, closure marker | `MyDrive/capstone-data/final_evaluation/` (`results_h1_price.csv`, `test_predictions_h1.csv`, `evaluation_manifest_h1.json`, `test_closed_h1.json`) |
| Evaluation | Complete fitted bundle | `MyDrive/capstone-data/models/stacking_h1.pkl` |
| Deployment | Portable model files used by the live pipeline | `results/artifacts/models/` (`xgb_h1.json`, `stack_h1_sklearn.joblib`, verified by `MODEL_CHECKSUMS.json` and a golden-prediction parity test) |
| Explainability | SHAP values, summaries, grouped farmer-facing explanations | `results/artifacts/` and `MyDrive/capstone-data/explainability/` |

The evaluation manifest records the row counts, tuned settings, Ridge `alpha`, fold boundaries, metric units, reported models, evaluation scope, and library versions used.

**From training to deployment.** The live forecast pipeline loads the checksum-verified portable files, builds features from stored prices with the same feature-engineering code, and runs the same RF → XGBoost → Ridge prediction. Each forecast is published with an exact SHAP explanation that adds up to the predicted change. The pipeline never retrains or changes the model.

## 9. Limitations

- **Single test year.** All final results come from 88 rows in one unusual year of rising prices. They do not establish performance in other periods or statistical significance.
- **Small data.** About 900 supervised rows across eight series limit model complexity and the stability of any per-series breakdown.
- **WFP only.** The model was trained and evaluated on WFP prices up to October 2019. Live forecasts use FEWS NET prices, and no formal comparison of the two sources has been performed.
- **No uncertainty estimates.** Forecasts are point predictions without prediction intervals.
- **Limited ensemble diversity.** Random Forest and XGBoost produce highly correlated forecasts, which limits the stack's advantage over its components and over Linear Regression.
