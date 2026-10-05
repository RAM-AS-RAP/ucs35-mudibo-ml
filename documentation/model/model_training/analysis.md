# Model Training Analysis

## 1. Scope and Evidence

This report describes the development-stage training workflow implemented in [05_model_training.ipynb](../../../notebooks/05_model_training.ipynb). It uses the current training modules and the retained outputs of that notebook and [04_feature_engineering.ipynb](../../../notebooks/04_feature_engineering.ipynb). Results below are recorded development results, not a new execution or final held-out test evaluation.

The task is to forecast monthly wholesale crop-price changes for Beans and Maize in Nairobi, Kisumu, Mombasa, and Nakuru. The eight market-commodity series are pooled into one model per forecast horizon, with identity indicators distinguishing them. Two independent workflows are trained:

- **H1:** cumulative log-price change from the origin month to one month ahead.
- **H2:** cumulative log-price change from the origin month to two months ahead.

The supervised panel is WFP-only and ends in October 2019. FEWS NET and climate variables are not inputs to this training run. The initial panel contains 1,279 rows, including pre-2010 history retained for feature warm-up. The final test subset is counted but not scored.

## 2. Runtime and Input Loading

The setup cell detects whether execution is local or in Google Colab. In Colab, it mounts Google Drive, loads environment variables, and locates or clones the repository on the configured `feature/model-training` branch. Authentication is read from environment variables or Colab Secrets; it is not part of the model artifacts.

The repository's `src` directory is added to the Python import path. The notebook imports the persistence baseline, scaled linear regression, RF/XGBoost tuning functions, OOF generation, Ridge fitting, and chronological fold utilities.

In Colab, supervised tables are loaded from `MyDrive/capstone-data/feature_engineering/` when the H1 table is available there; otherwise the repository's `results/artifacts/feature_engineering/` is used. Local execution uses the repository directory. Both `model_df_h1.csv` and `model_df_h2.csv` must exist in the selected directory. Origin and target dates are parsed as dates.

## 3. Targets, Features, and Eligibility

### 3.1 Direct Horizon Targets

Let $P_t$ denote the price per kilogram at origin month $t$. The target for horizon $h$ is:

$$
y_t^{(h)} = \log(P_{t+h}) - \log(P_t), \qquad h \in \{1,2\}.
$$

H2 is a direct two-month forecast, not a recursive forecast using the H1 prediction. The code looks up exact calendar target months rather than assuming the next available row is the next month.

Both the origin and target prices must have `impute_method == 'observed'`. Rows with missing targets or required features are excluded. Historical predictor values are not necessarily all observed: the endpoint filter applies to the origin and target, while the history guard checks calendar alignment and nonmissing prices. This distinction matters when assessing upstream imputation effects.

The retained feature-engineering output reports 944 origin rows within the modeling window. Endpoint eligibility excludes 31 H1 rows and 40 H2 rows; no additional feature-missing rows are excluded within this window.

### 3.2 Predictor Construction

The training notebook selects columns beginning with `log_return_lag_`, `log_ratio_`, `month_`, `market_`, or `commodity_`, plus `year`. Under the current feature implementation this produces 14 predictors:

| Feature Group | Predictors | Purpose |
| --- | --- | --- |
| Prior log returns | `log_return_lag_1`, `log_return_lag_2`, `log_return_lag_3` | Recent price-change history |
| Trailing log ratios | `log_ratio_3m`, `log_ratio_12m` | Recent price position relative to short/long trailing history |
| Seasonal calendar | `month_sin`, `month_cos` | Cyclic month-of-year effects |
| Trend | `year` | Broad temporal trend |
| Market identity | Four market indicators | Differences between Nairobi, Kisumu, Mombasa, and Nakuru |
| Commodity identity | Two commodity indicators | Differences between Beans and Maize |

For $L_t = \log(P_t)$, lag feature $k$ is $L_{t-k} - L_{t-k-1}$. It does not include the current-origin return $L_t-L_{t-1}$. For trailing window $w$, the log-ratio feature is:

$$
q_{t,w} = L_{t-1} - \frac{1}{w}\sum_{j=1}^{w} L_{t-j}, \qquad w \in \{3,12\}.
$$

Thus, these are ratios relative to a trailing geometric price mean, expressed in log space. Calendar-distance checks prevent gaps from being mistaken for consecutive history. Pre-2010 rows provide warm-up history but are not supervised training examples. Raw prices, `log_price`, target columns, and provenance columns are not selected as predictors.

## 4. Chronological Development and Test Splits

The notebook rebuilds folds from each supervised table using [folds.py](../../../src/agrisence/training/folds.py), rather than loading the saved fold JSON files. Default masks are based on **origin months**:

- Development: January 2010 through October 2018.
- Final test: November 2018 through October 2019, subject to target availability and row eligibility.

| Horizon | Supervised Rows | Development Rows | Reserved Test Rows | Expanding Folds |
| --- | ---: | ---: | ---: | ---: |
| H1 | 913 | 825 | 88 | 4 |
| H2 | 904 | 824 | 80 | 4 |

The cutoff means late origins cannot be retained when their future target is beyond October 2019. Therefore, the reserved test mask spans a nominal 12-month origin window but does not contain eligible rows for every month at both horizons.

After an initial five-year span, training expands while validation advances through annual blocks. All market-commodity rows at an origin month stay on the same side of a boundary.

| Fold | Training Start | H1 Training Origin End | H2 Training Origin End | Validation Origins |
| --- | --- | --- | --- | --- |
| 1 | January 2010 | December 2014 | November 2014 | January-December 2015 |
| 2 | January 2010 | December 2015 | November 2015 | January-December 2016 |
| 3 | January 2010 | December 2016 | November 2016 | January-December 2017 |
| 4 | January 2010 | December 2017 | November 2017 | January-October 2018 |

The horizon-aware boundary ensures a training target is no later than the first validation origin month. This assumes the first-origin monthly price is available when the forecast is issued. Models remain fixed within each annual validation block, although each later origin receives its own historical predictors. This is not monthly retraining within a block.

## 5. Baseline Training

### 5.1 Persistence

Persistence predicts zero log change for every validation row:

$$
\hat y_t^{(h)} = 0.
$$

In price space, this means retaining the origin price. It has no fitted or tuned parameters and provides the essential no-change benchmark.

### 5.2 Linear Regression

The linear baseline is a scikit-learn pipeline containing `StandardScaler` followed by `LinearRegression`. A fresh pipeline is fitted on each fold's training rows and evaluated on its validation rows. Scaling is fitted within the fold, avoiding a full-data scaling fit before validation. The baseline uses default estimator settings, including an intercept, and is not tuned.

For each baseline, MAE and RMSE are calculated separately per validation fold, then averaged with equal weight across folds. A separate linear pipeline is subsequently fitted on all development rows and retained for serialization. Its development training-fit score is not the reported baseline score.

## 6. Random Forest and XGBoost Tuning

Both base models use exhaustive `GridSearchCV` with the same four explicit chronological splits and `scoring='neg_mean_squared_error'`. Scikit-learn maximizes the negative score; the notebook reverses its sign to display positive CV MSE. The selected score is the equal-weight mean of fold MSE values.

Both searches use `refit=False`: the searches return selected parameters and scores, not a final fitted estimator trained on all development data. `random_state=42` is fixed, and parallel execution uses `n_jobs=-1`. XGBoost uses `objective='reg:squarederror'` and `eval_metric='rmse'`; the selection criterion remains CV MSE. No early-stopping configuration is supplied.

### 6.1 Search Spaces

| Model | Hyperparameter | Candidate Values |
| --- | --- | --- |
| Random Forest | `n_estimators` | 200, 400 |
| Random Forest | `max_depth` | 4, 6, 8 |
| Random Forest | `min_samples_leaf` | 5, 10 |
| Random Forest | `max_features` | `sqrt`, 0.7 |
| XGBoost | `n_estimators` | 200, 400 |
| XGBoost | `max_depth` | 3, 4, 5 |
| XGBoost | `learning_rate` | 0.01, 0.05, 0.1 |
| XGBoost | `min_child_weight` | 1, 5 |
| XGBoost | `subsample` | 0.7, 1.0 |
| XGBoost | `colsample_bytree` | 0.7, 1.0 |
| XGBoost | `reg_lambda` | 1, 5 |

RF has 24 combinations and XGBoost has 288. With four folds, that is 96 RF fits and 1,152 XGBoost fits per horizon, or 2,496 base-model grid fits across both horizons, excluding OOF fits and Ridge tuning. Other estimator settings remain library defaults and are not explicitly captured by the compact tuning payload.

### 6.2 Selected Random Forest Settings

| Hyperparameter | H1 | H2 |
| --- | --- | --- |
| `n_estimators` | 400 | 200 |
| `max_depth` | 6 | 6 |
| `min_samples_leaf` | 10 | 5 |
| `max_features` | `sqrt` | `sqrt` |
| Fixed `random_state` | 42 | 42 |

Both configurations limit tree depth and leaf size, constraining complexity in a relatively small panel. H1 selects more trees and a larger minimum leaf size; these are search outcomes, not evidence by themselves of why one horizon is easier.

### 6.3 Selected XGBoost Settings

| Hyperparameter | H1 | H2 |
| --- | --- | --- |
| `n_estimators` | 200 | 200 |
| `max_depth` | 3 | 3 |
| `learning_rate` | 0.01 | 0.01 |
| `min_child_weight` | 1 | 5 |
| `subsample` | 1.0 | 1.0 |
| `colsample_bytree` | 1.0 | 1.0 |
| `reg_lambda` | 5 | 5 |
| Fixed `random_state` | 42 | 42 |

The selected configurations favor shallow trees, a low learning rate, and the larger tested L2 penalty. H2 selects the larger child-weight threshold. No stochastic row/column subsampling is selected. Several settings lie at grid boundaries; this does not establish that the grid contains the globally optimal configuration.

## 7. Out-of-Fold Predictions and Stacking

### 7.1 Base Predictions

For each development fold, fresh Random Forest and XGBoost estimators are fitted using that horizon's selected settings. They predict only the later validation rows. The OOF table stores the original row index, origin month, `rf_oof_pred`, `xgb_oof_pred`, and actual target. Tables are concatenated and sorted by origin month.

The earliest training-only years do not receive OOF predictions. Each OOF prediction comes from a model not fitted on that row; however, hyperparameters were selected using those same development validation folds. Consequently, these are post-selection development diagnostics, not nested-CV estimates untouched by tuning.

### 7.2 Ridge Meta-Model

The stacking ensemble is **Random Forest + XGBoost + Ridge**. Ridge uses only the two base-prediction columns, not the original 14 predictors:

$$
\hat y_{\mathrm{stack}} = b + w_{\mathrm{RF}}\hat y_{\mathrm{RF}} + w_{\mathrm{XGB}}\hat y_{\mathrm{XGB}}.
$$

Its coefficients are learned with L2 regularization and are not constrained to be nonnegative or sum to one. This is not necessarily a simple average. The current outputs do not report the fitted coefficients or intercept, so numerical ensemble weights cannot be stated from the retained results.

Ridge searches `alpha` in `[0.1, 1.0, 10.0]`. OOF origin months are sorted, distinct months are split with `TimeSeriesSplit`, and all rows from a given month remain together. Up to four splits are used. Selection uses negative MSE, and `refit=True` fits the winning Ridge estimator on all OOF rows. Selected alpha is **0.1 for H1** and **1.0 for H2**.

Unlike the base-fold utility, this meta-CV splitter does not explicitly embargo training target months by horizon. For H2, adjacent train/validation origin blocks can leave training outcomes extending beyond the first validation origin. This should be reviewed before interpreting Ridge CV as a fully deployment-faithful estimate.

### 7.3 What the Displayed Stack Score Means

After fitting Ridge, the notebook predicts the same OOF rows used in that fit. The displayed **Ridge stack (fit on OOF)** MAE/RMSE is therefore an in-sample meta-model diagnostic, despite its OOF inputs. The separate Ridge CV MSE is a different quantity, produced while selecting alpha. Neither is a final held-out ensemble test score.

## 8. Key Metrics and Recorded Results

For actual log-change targets $y_i$ and predictions $\hat y_i$:

$$
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|,
\qquad
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2}.
$$

All errors below are in **log-change units**, not currency per kilogram or literal percentage-point errors. Lower is better. A later price-space forecast would use $\hat P_{t+h}=P_t\exp(\hat y_t^{(h)})$, but the notebook does not evaluate reconstructed prices or apply a retransformation-bias correction.

### 8.1 Development MAE and RMSE

Values are rounded to six decimal places from the retained training output.

| Horizon | Model | MAE | RMSE | Evaluation Scope |
| --- | --- | ---: | ---: | --- |
| H1 | Persistence | 0.059214 | 0.079878 | Equal-weight mean of fold metrics |
| H1 | Linear Regression | 0.058371 | 0.076907 | Equal-weight mean of fold metrics |
| H1 | Random Forest | 0.056913 | 0.076619 | Pooled OOF rows after tuning |
| H1 | XGBoost | 0.056986 | 0.076524 | Pooled OOF rows after tuning |
| H1 | Ridge stack | 0.056209 | 0.076063 | Fit diagnostic on meta-training rows |
| H2 | Persistence | 0.092882 | 0.121140 | Equal-weight mean of fold metrics |
| H2 | Linear Regression | 0.091752 | 0.114029 | Equal-weight mean of fold metrics |
| H2 | Random Forest | 0.088559 | 0.114114 | Pooled OOF rows after tuning |
| H2 | XGBoost | 0.090501 | 0.116288 | Pooled OOF rows after tuning |
| H2 | Ridge stack | 0.087762 | 0.114869 | Fit diagnostic on meta-training rows |

Baseline fold averaging and pooled OOF evaluation weight observations differently, especially because the 2018 validation block is shorter. A pooled RMSE is also not the same as an average of fold RMSEs. Small differences across these rows should not be treated as a controlled ranking without a common evaluation aggregation.

### 8.2 Selected Cross-Validation Scores

| Horizon | Model | Best CV Negative MSE | Positive CV MSE |
| --- | --- | ---: | ---: |
| H1 | Random Forest | -0.005962772192 | 0.005962772192 |
| H1 | XGBoost | -0.005950363578 | 0.005950363578 |
| H1 | Ridge meta-model | -0.006357958062 | 0.006357958062 |
| H2 | Random Forest | -0.013266763967 | 0.013266763967 |
| H2 | XGBoost | -0.013693835197 | 0.013693835197 |
| H2 | Ridge meta-model | -0.015046535770 | 0.015046535770 |

RF/XGBoost CV scores use base-feature development folds. Ridge CV scores use later OOF month splits with a different training/evaluation structure. These scores are not directly comparable as a three-model leaderboard. Persistence and Linear Regression are not tuned, so their parameter-table CV MSE entries are `Not tuned`.

### 8.3 Base-Prediction Correlation

| Horizon | Pearson Correlation Between RF and XGBoost OOF Predictions |
| --- | ---: |
| H1 | 0.926951 |
| H2 | 0.893366 |

Both correlations are high, indicating similar prediction patterns. This can limit ensemble diversity, but prediction correlation is not residual-error correlation and does not by itself determine stacking benefit.

### 8.4 Interpretation

- H1 has lower absolute errors than H2 across all displayed models, consistent with the longer cumulative target being harder; the horizons also have different eligible samples and target distributions.
- For H1, RF has slightly lower pooled OOF MAE than XGBoost, while XGBoost has slightly lower pooled OOF RMSE. The difference is small.
- For H2, RF has lower pooled OOF MAE and RMSE than XGBoost. Linear Regression's displayed fold-mean RMSE is also competitive, but differs in aggregation.
- Ridge has the lowest displayed MAE at both horizons, but its scoring is in-sample at the meta level. H2 Ridge's fit RMSE is higher than RF's pooled OOF RMSE. Neither observation establishes held-out ensemble superiority.
- No final test MAE/RMSE, price-space error, MAPE, R-squared, directional accuracy, prediction-interval coverage, uncertainty estimate, or per-market/per-commodity breakdown is reported. These values must not be inferred from the aggregate development table.

## 9. Checkpoints and Saved Artifacts

The training loop now saves tuning checkpoints immediately after RF tuning, updates them after XGBoost tuning, and updates them again after Ridge fitting. A partial run may therefore contain only the completed models. This preserves completed tuning work if a later stage fails; it does not implement automatic resume or skipping previously tuned models.

The final save cell writes six artifacts per horizon:

| Filename Pattern | Contents |
| --- | --- |
| `tuning_{horizon}.json` | Horizon, feature columns, development/test row counts, RF/XGBoost best parameters and scores, Ridge alpha/CV score, OOF row count |
| `persistence_dev_metrics_{horizon}.json` | Development scope and fold-mean baseline MAE/RMSE |
| `linreg_dev_metrics_{horizon}.json` | Development scope and fold-mean linear MAE/RMSE |
| `oof_metrics_{horizon}.json` | Baseline and OOF/stack diagnostics, prediction correlation, tuning scores, row counts, and score-scope caveats |
| `linear_regression_{horizon}.joblib` | Linear pipeline fitted on all development rows |
| `ridge_meta_{horizon}.joblib` | Ridge fitted on all OOF rows, including selected-alpha/score attributes |

Local execution writes to `<repo_root>/results/artifacts/model_training/`. The recorded Colab run writes its local copy under `/content/AgriSence_model_training/results/artifacts/model_training/` and mirrors artifacts to `/content/drive/MyDrive/capstone-data/model_training/`.

The tuned hyperparameter files in Drive are:

- `MyDrive/capstone-data/model_training/tuning_h1.json`
- `MyDrive/capstone-data/model_training/tuning_h2.json`

Retained notebook outputs report successful saves and `exists=True` for all 12 mirrored files. At report preparation, the Windows repository's artifact folder was empty, and the Drive files were not independently opened from this environment. Colab's local checkout is temporary runtime storage, not the Windows checkout; notebook save messages do not imply those files were synchronized to Windows.

The fitted RF/XGBoost fold estimators and full OOF table are not serialized by this notebook. A saved Ridge model alone is not a deployable stacking ensemble: it requires fitted base models that produce the same two input predictions. The reported OOF row count is included in generated JSON but is not printed in the retained training output, so no numerical count is asserted here.

