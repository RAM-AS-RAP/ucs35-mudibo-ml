# Final Model Evaluation Analysis

## Executive Summary

The evaluation refits models on eligible development data and scores the one-month-ahead (H1) forecast once on held-out observations. It compares two models: the Linear Regression baseline and a stacking ensemble that combines Random Forest and XGBoost forecasts with a Ridge meta-model. Results are pooled over all held-out rows and reported in two tables: one for the predicted log change (MAE, MSE, RMSE) and one for the reconstructed price (MAE, MSE, RMSE, MAPE).

The comparison is mixed. Stacking has lower log-change MAE and lower error on all four price metrics. Linear Regression has lower log-change MSE and RMSE. The margins are small and describe this test period only.

These are results for this particular held-out period. They do not establish statistical significance or guarantee performance in future periods.

## Evaluation Design

The implementation is in [evaluate.py](../../../src/agrisence/training/evaluate.py), with final refitting and held-out predictions in [stacking.py](../../../src/agrisence/training/ensemble/stacking.py). The orchestration and saved outputs are in [06_final_evaluation.ipynb](../../../notebooks/06_final_evaluation.ipynb).

### Target and Samples

For origin price $P_t$, the target is the cumulative log-price change one calendar month ahead:

$$
y_t = \log(P_{t+1}) - \log(P_t).
$$

A predicted log change $\hat y_t$ is converted to a price by:

$$
\hat P_{t+1}=P_t\exp(\hat y_t).
$$

The WFP-only data combine four Kenyan markets (Kisumu, Mombasa, Nairobi, Nakuru) and two commodities (Beans, Maize). The evaluation uses eligible test origins from November 2018 through October 2019, subject to the availability of an observed target endpoint.

| Development Rows | Pooled Test Rows | Market-Commodity Series |
| ---: | ---: | ---: |
| 825 | 88 | 8 |

Each series contributes 11 eligible test rows, so “November through October” describes the nominal origin window rather than 12 rows per series. Final-test rows are passed once to `predict_test_set`; the fitted models and Ridge meta-model are trained only from pre-test development rows.

### Final Refit and Models

The hyperparameters and feature order are frozen. Final refitting:

1. Generates chronological out-of-fold predictions from the two stacking base learners during the development period.
2. Fits Ridge on those two OOF prediction columns, using the saved alpha (0.1).
3. Refits both stacking base learners on all eligible development rows using their frozen settings.
4. Fits the Linear Regression pipeline (`StandardScaler` + `LinearRegression`) on all eligible development rows.
5. Predicts the eligible held-out rows once and combines the base-model predictions with Ridge for the stack output.

No tuning is performed on the final test set. The saved stack bundle contains both base estimators, the Ridge meta-model, Linear Regression, and feature order.

Both models use the frozen 14-feature order: three lagged log returns, two trailing log ratios, month sine/cosine, year, four market indicators, and two commodity indicators. The stacking base learners consume these features; Ridge consumes their two OOF prediction columns. Random Forest and XGBoost are components of the stack and are not reported as separate models.

### Metrics

All metrics are pooled: each is computed once over all 88 held-out rows for a model, not averaged across series.

**Log-change table** (`results_h1_log_change.csv`):

| Metric | Definition / Unit | Better |
| --- | --- | --- |
| `mae` | Mean absolute error of the log change; log-change units | Lower |
| `mse` | Mean squared error of the log change; squared log-change units | Lower |
| `rmse` | Square root of `mse`; log-change units | Lower |

**Reconstructed-price table** (`results_h1_price.csv`):

| Metric | Definition / Unit | Better |
| --- | --- | --- |
| `mae` | Mean absolute reconstructed-price error; KES/kg | Lower |
| `mse` | Mean squared reconstructed-price error; (KES/kg)² | Lower |
| `rmse` | Square root of `mse`; KES/kg | Lower |
| `mape` | Mean absolute percentage error between reconstructed and actual prices; percent | Lower |

MAPE is reported only for prices. Actual log changes can be zero (two held-out rows are exactly zero), which makes MAPE on log changes undefined.

On the same rows, MSE and RMSE always rank models identically, because RMSE is the square root of MSE. They describe sensitivity to large errors, while MAE describes the typical error size.

## Results

### Log-Change Predictions

| Model | MAE | MSE | RMSE |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.092726 | **0.015847** | **0.125886** |
| Stacking ensemble | **0.092434** | 0.016440 | 0.128219 |

Stacking has a slightly lower MAE (about 0.3%). Linear Regression has a lower RMSE (about 1.8%), so stacking makes slightly smaller typical errors but somewhat larger occasional misses.

### Reconstructed Prices

| Model | MAE (KES/kg) | MSE ((KES/kg)²) | RMSE (KES/kg) | MAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 4.921290 | 51.371490 | 7.167391 | 8.734540 |
| Stacking ensemble | **4.847952** | **51.105719** | **7.148826** | **8.648759** |

Stacking is better on all four price metrics. The gains are small: price MAE falls by about 1.49% and price RMSE by about 0.26% relative to Linear Regression.

The two tables rank RMSE differently because errors are weighted differently once log changes are converted to prices. A log-change error is scaled by the origin price, so errors on higher-priced series (mainly beans) count for more in price terms.

All values were computed by the re-scoring cell in the final evaluation notebook directly from the saved `test_predictions_h1.csv`.

The held-out year was unusual: the mean monthly log change was about 0.041 (roughly +4% a month), against about 0.001 across development origins. Models trained on the development period expected roughly flat prices, which should be kept in mind when reading the error levels.

## Saved Artifacts and Interpretation Limits

The notebook saves the two results tables, the prediction CSV, a manifest, and the fitted model bundle. The local output root is `results/artifacts/`; in Colab, result files are mirrored to `MyDrive/capstone-data/final_evaluation/` and the model bundle to `MyDrive/capstone-data/models/`. The bundle file is `stacking_h1.pkl` and retains the feature order for downstream use.

Persistence and standalone base-learner scores are intentionally outside the scope of this comparison. Test data are closed to further tuning; re-scoring recomputes metrics from saved predictions and does not refit or predict.

### Split-Boundary Caveat

Rows are assigned by **origin month**: development origins run through October 2018, and held-out origins start in November 2018. The final refit therefore includes every eligible row with a development origin, without separately restricting its target month to October 2018. Consequently, a development row originating in October 2018 targets November 2018, which overlaps the calendar range of held-out origins. Whether that label would have been available at the intended production cutoff depends on the forecast issue-time convention. The reported results follow the saved origin-based protocol; they should not be presented as a stricter target-time-embargoed estimate.

## Comparison Summary

| Scale | Metric | Linear Regression | Stacking | Better |
| --- | --- | ---: | ---: | --- |
| Log change | MAE | 0.092726 | 0.092434 | Stacking |
| Log change | MSE | 0.015847 | 0.016440 | Linear Regression |
| Log change | RMSE | 0.125886 | 0.128219 | Linear Regression |
| Price | MAE (KES/kg) | 4.921290 | 4.847952 | Stacking |
| Price | MSE ((KES/kg)²) | 51.371490 | 51.105719 | Stacking |
| Price | RMSE (KES/kg) | 7.167391 | 7.148826 | Stacking |
| Price | MAPE (%) | 8.734540 | 8.648759 | Stacking |

Stacking is better on five of the seven metrics, including every price metric. Linear Regression is better on log-change MSE and RMSE, which share one underlying quantity. The differences are descriptive of this fixed test period. No uncertainty intervals or significance tests were computed, so the rankings do not establish statistical significance. Do not revise the model choice based on this closed test set.
