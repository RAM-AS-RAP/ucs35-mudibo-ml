# Final Model Evaluation Analysis

## Executive Summary

The evaluation refits models on eligible development data and scores each forecast horizon once on held-out observations. This analysis focuses on two pooled comparison entries: the Linear Regression baseline and a stacking ensemble that combines Random Forest and XGBoost forecasts with a Ridge meta-model. It reports log-change MAE/RMSE and reconstructed-price MAPE/MAE/RMSE.

Against Linear Regression, stacking is mixed at H1 and worse on all pooled metrics at H2. At H1 stacking has lower log MAE and all three price errors, but higher log RMSE. At H2 Linear Regression has lower error on all five metrics. The margins are modest and describe this test period only.

These are results for this particular held-out period and aggregation. They do not establish statistical significance or guarantee performance in future periods.

## Evaluation Design

The implementation is in [evaluate.py](../../../src/agrisence/training/evaluate.py), with final refitting and held-out predictions in [stacking.py](../../../src/agrisence/training/ensemble/stacking.py). The orchestration, saved outputs, and recorded result tables are in [06_final_evaluation.ipynb](../../../notebooks/06_final_evaluation.ipynb).

### Targets and Samples

For origin price $P_t$, the direct horizon target is cumulative log-price change:

$$
y_t^{(h)} = \log(P_{t+h}) - \log(P_t), \qquad h \in \{1,2\}.
$$

H2 is predicted directly from its features; it is not produced by chaining H1 forecasts. A predicted log change $\hat y_t^{(h)}$ is converted to a price by:

$$
\hat P_{t+h}=P_t\exp(\hat y_t^{(h)}).
$$

The WFP-only data combine four Kenyan markets (Kisumu, Mombasa, Nairobi, Nakuru) and two commodities (Beans, Maize). The evaluation uses eligible test origins from November 2018 through October 2019, subject to the availability of an observed target endpoint. The actual tables contain:

| Horizon | Development Rows | Pooled Test Rows | Rows per Market-Commodity Series | Number of Series |
| --- | ---: | ---: | ---: | ---: |
| H1 | 825 | 88 | 11 | 8 |
| H2 | 824 | 80 | 10 | 8 |

Thus, “November through October” describes the nominal origin window; it does not mean every series contributes 12 eligible rows. Final-test rows are passed once to `predict_test_set`; the fitted models and Ridge meta-model are trained only from pre-test development rows.

### Final Refit and Models

The hyperparameters and feature order are frozen. For each horizon, final refitting:

1. Generates chronological out-of-fold predictions from the two stacking base learners during the development period.
2. Fits Ridge on those two OOF prediction columns, using the saved horizon-specific alpha.
3. Refits both stacking base learners on all eligible development rows using their frozen settings.
4. Fits the Linear Regression pipeline on all eligible development rows.
5. Predicts the eligible held-out rows once and combines the base-model predictions with Ridge for the stack output.

The reported comparison contains the stacking ensemble and the Linear Regression baseline. The saved stack bundle contains both base estimators, the Ridge meta-model, Linear Regression, and feature order.

The refit uses the hyperparameters selected during training; no tuning is performed on the final test set.

| Model Setting | H1 | H2 |
| --- | --- | --- |
| Ridge meta-model alpha | 0.1 | 1.0 |
| Linear Regression | `StandardScaler` + `LinearRegression`, fitted on development rows | `StandardScaler` + `LinearRegression`, fitted on development rows |

Both comparisons use the frozen 14-feature order: three lagged log returns, two trailing log ratios, month sine/cosine, year, four market indicators, and two commodity indicators. The stacking base learners consume these features; Ridge consumes their two OOF prediction columns.

### Metrics

The reported metrics are:

| Metric | Definition / Unit | Better |
| --- | --- | --- |
| `mae_log` | Mean absolute error of cumulative log change; log-change units | Lower |
| `rmse_log` | Root mean squared error of cumulative log change; log-change units | Lower |
| `mape_price_pct` | Mean absolute percentage error between reconstructed and actual prices; percent | Lower |
| `mae_price` | Mean absolute reconstructed-price error; KES/kg | Lower |
| `rmse_price` | Root mean squared reconstructed-price error; KES/kg | Lower |

MAPE is calculated on positive reconstructed prices, not on log changes. Applying MAPE directly to log changes would be unstable near zero. No directional-accuracy metric is included.

### Pooled Aggregation

Each pooled row is computed directly from all test observations for that model, rather than by averaging the eight displayed series rows. Since each series has the same row count within each horizon (11 for H1, 10 for H2), pooled MAE and MAPE correspond to equal-series-weight averages here. RMSE is formed by pooling squared observation errors and then taking the square root; it is not the arithmetic mean of the eight series RMSEs.

## Pooled Results: H1

All results below are taken from the retained H1 table in the final evaluation notebook. Error values are rounded to six decimal places.

| Model | Log MAE | Log RMSE | Price MAPE (%) | Price MAE (KES/kg) | Price RMSE (KES/kg) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Regression | 0.092726 | 0.125886 | 8.734540 | 4.921290 | 7.167391 |
| Stacking ensemble | 0.092434 | 0.128219 | 8.648759 | 4.847952 | 7.148826 |

At H1 the difference is small. Stacking has lower log MAE and price MAPE/MAE/RMSE, while Linear Regression has lower log RMSE.

## Pooled Results: H2

| Model | Log MAE | Log RMSE | Price MAPE (%) | Price MAE (KES/kg) | Price RMSE (KES/kg) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Regression | 0.154149 | 0.194964 | 14.029162 | 8.070475 | 10.861196 |
| Stacking ensemble | 0.156914 | 0.202914 | 14.205292 | 8.166092 | 10.955661 |

At H2 Linear Regression has lower error on all five pooled metrics than stacking.

## Per-Series Results

The following tables compare the stacking ensemble and Linear Regression across all eight market-commodity series. Price MAE is in KES/kg.

### H1 Price MAE by Series

| Market | Commodity | Linear Regression | Stacking ensemble |
| --- | --- | ---: | ---: |
| Kisumu | Beans | 4.9997 | 5.2419 |
| Kisumu | Maize | 3.4359 | 3.4431 |
| Mombasa | Beans | 6.6436 | 6.0935 |
| Mombasa | Maize | 2.8772 | 2.8930 |
| Nairobi | Beans | 8.7440 | 8.1963 |
| Nairobi | Maize | 3.6617 | 3.5512 |
| Nakuru | Beans | 5.8729 | 6.1290 |
| Nakuru | Maize | 3.1354 | 3.2356 |

### H2 Price MAE by Series

| Market | Commodity | Linear Regression | Stacking ensemble |
| --- | --- | ---: | ---: |
| Kisumu | Beans | 8.5069 | 8.8678 |
| Kisumu | Maize | 6.8687 | 6.7569 |
| Mombasa | Beans | 11.1398 | 10.6350 |
| Mombasa | Maize | 4.9196 | 4.5095 |
| Nairobi | Beans | 12.8256 | 12.0354 |
| Nairobi | Maize | 4.8995 | 5.0242 |
| Nakuru | Beans | 9.9474 | 11.7629 |
| Nakuru | Maize | 5.4563 | 5.7371 |

These per-series tables show price MAE; the pooled tables report all five metrics for both comparison models.

## Cross-Horizon Comparison

H2 errors are higher than H1 errors for every model and every reported metric. This is consistent with forecasting cumulative change across a longer horizon being more difficult in this sample. The horizons also have different eligible rows and target values, so their absolute errors should not be interpreted as a controlled experiment changing horizon alone.

## Saved Artifacts and Interpretation Limits

The notebook saves results tables, prediction CSVs, manifests, and complete fitted-model bundles. The local output root is `results/artifacts/`; in Colab, result files are mirrored to `MyDrive/capstone-data/final_evaluation/` and model bundles to `MyDrive/capstone-data/models/`. Model bundle files are `stacking_h1.pkl` and `stacking_h2.pkl` and retain the feature order for downstream use.

Persistence and standalone base-learner scores are intentionally outside the scope of this comparison. Test data are closed to further tuning. This analysis uses the notebook's retained results and does not rerun predictions.

### Split-Boundary Caveat

Rows are assigned by **origin month**: development origins run through October 2018, and held-out origins start in November 2018. The final refit therefore includes every eligible row with a development origin, without separately restricting its target month to October 2018. Consequently, an H1 development row originating in October 2018 targets November 2018, while an H2 row from that origin targets December 2018. These target months overlap the calendar range of held-out origins. Whether those labels would have been available at the intended production cutoff depends on the forecast issue-time convention. The reported results follow the saved origin-based protocol; they should not be presented as a stricter target-time-embargoed estimate.

## Pooled Comparison Summary

- **H1:** Stacking is slightly better than Linear Regression on four pooled metrics: log MAE, price MAPE, price MAE, and price RMSE. Linear Regression is better on log RMSE.
- **H2:** Linear Regression is better than stacking on all five pooled metrics.

The observed differences are descriptive of this fixed test period. No uncertainty intervals or significance tests were computed, so the rankings do not establish statistical significance. Do not revise the model choice based on this closed test set.

## Stacking Versus Linear Regression: Pooled Metrics

The final comparison of pooled metrics is shown below. Lower values indicate better performance.

| Horizon | Metric | Linear Regression | Stacking | Better |
| --- | --- | ---: | ---: | --- |
| H1 | Log MAE | 0.092726 | 0.092434 | Stacking |
| H1 | Log RMSE | 0.125886 | 0.128219 | Linear Regression |
| H1 | Price MAPE (%) | 8.734540 | 8.648759 | Stacking |
| H1 | Price MAE (KES/kg) | 4.921290 | 4.847952 | Stacking |
| H1 | Price RMSE (KES/kg) | 7.167391 | 7.148826 | Stacking |
| H2 | Log MAE | 0.154149 | 0.156914 | Linear Regression |
| H2 | Log RMSE | 0.194964 | 0.202914 | Linear Regression |
| H2 | Price MAPE (%) | 14.029162 | 14.205292 | Linear Regression |
| H2 | Price MAE (KES/kg) | 8.070475 | 8.166092 | Linear Regression |
| H2 | Price RMSE (KES/kg) | 10.861196 | 10.955661 | Linear Regression |

At H1, stacking is better on four of five pooled metrics, but worse on log RMSE. The gains are small: price MAE falls by about 1.49%, and price RMSE by about 0.26% relative to Linear Regression. At H2, Linear Regression is better on all five metrics; stacking's price MAE is about 1.18% higher and its price RMSE about 0.87% higher. This supports a horizon-dependent descriptive result, not a claim that stacking inherently helps H1 or inherently harms H2.
