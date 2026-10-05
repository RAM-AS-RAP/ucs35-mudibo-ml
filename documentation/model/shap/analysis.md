# Stacked Forecast SHAP Analysis

## Executive Summary

This analysis explains the final Random Forest/XGBoost/Ridge stacking forecasts for one-month (H1) and direct two-month (H2) cumulative log-price changes. It uses the saved held-out feature rows and saved final predictions for the same observations; no model fitting, parameter tuning, or forecast generation is performed.

The notebook produced grouped explanations for **88 H1 rows** and **80 H2 rows** across the eight market-commodity series. In both horizons, **Time of year** and **Long-run trend** have the largest mean absolute grouped log contributions. The other groups, recent average, recent momentum, and market/crop identity, have smaller average contributions in these test observations. These are model attributions, not causal evidence about why prices moved.

## Method

### Exact SHAP Values for the Stack

The fitted stack predicts with a linear Ridge meta-model over the two tree model predictions:

$$
\hat y = b + w_{RF}\hat y_{RF} + w_{XGB}\hat y_{XGB}.
$$

Here, $\hat y$ is the stack's predicted cumulative log-price change for one market, commodity, and horizon. $b$ is Ridge's fitted intercept; $w_{RF}$ and $w_{XGB}$ are its fitted coefficients for Random Forest and XGBoost. $\hat y_{RF}$ and $\hat y_{XGB}$ are the corresponding base-model predictions.

The base learners' TreeExplainer SHAP values are combined with Ridge's fitted coefficients. Their expected values are combined in the same way, including Ridge's intercept. This is an exact additive decomposition for a linear Ridge meta-model; the same method would not be valid for a nonlinear meta-model. No model-agnostic explainer is used.

### Shapley Value Algorithm

For a model $f$, input row $x$, and feature set $F$ containing $M$ features, the Shapley value for feature $i$ is the weighted average of that feature's marginal effect across every possible subset $S$ of the other features:

$$
\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}}
\frac{|S|!(M-|S|-1)!}{M!}
\left[v_x(S \cup \{i\}) - v_x(S)\right].
$$

Here, $x$ is the held-out row being explained, $F$ is its complete set of features, $M=|F|$ is the number of features, $i$ is the feature being attributed, and $S$ is a subset of the other features. $|S|$ is the number of features in that subset, and $!$ denotes factorial. $v_x(S)$ is the model's expected output when features in $S$ are known from row $x$ and the remaining features are marginalized according to the explainer's reference distribution. The factorial weight assigns the appropriate share to each possible feature ordering. Thus, $\phi_i(x)$ is the average marginal change in output when feature $i$ is added across all coalitions.

The implementation invokes `shap.TreeExplainer` separately on the fitted Random Forest and XGBoost models, with no explicit background dataset passed. In the current SHAP API, this selects tree-path-dependent expectations: training samples' tree-path cover counts provide the reference for unknown features. TreeSHAP uses a tree-specific dynamic-programming algorithm to compute the Shapley values for these tree models exactly under that convention, rather than sampling feature permutations. The chosen reference convention matters because Shapley values explain differences from a model baseline under that convention.

For each feature $i$, the final stacked attribution and baseline are then:

$$
\phi_i^{stack} = w_{RF}\phi_i^{RF} + w_{XGB}\phi_i^{XGB},
\qquad
\phi_0^{stack} = b + w_{RF}\phi_0^{RF} + w_{XGB}\phi_0^{XGB},
$$

Here, $i$ indexes a feature; $\phi_i^{RF}$ and $\phi_i^{XGB}$ are that feature's SHAP values for the Random Forest and XGBoost base models. $w_{RF}$ and $w_{XGB}$ are `ridge_meta.coef_[0]` and `ridge_meta.coef_[1]`; $b$ is the Ridge intercept. $\phi_0^{RF}$ and $\phi_0^{XGB}$ are the base models' TreeExplainer expected values, and $\phi_0^{stack}$ is the stack's combined base value. By linearity of Shapley values, the combined base value plus the combined feature attributions reconstructs the Ridge stack prediction. The implementation checks this additivity against the saved final forecast for every held-out row.

For each observation, the grouped explanation is:

$$
\hat y = b_{stack} + \sum_g c_g,
$$

Here, $b_{stack}$ is the stack's base value, $G$ is the set of feature groups, $g$ identifies one group, and $c_g$ is the sum of the exact feature SHAP values assigned to that group. The grouping code verifies that every feature belongs to exactly one group and that the grouped contributions reconcile with the full SHAP vector within an absolute tolerance of $10^{-6}$ before presentation rounding.

### Feature Groups

| Group | Included Features | Interpretation Label |
| --- | --- | --- |
| Recent price momentum | `log_return_lag_1`, `log_return_lag_2`, `log_return_lag_3` | Recent changes in the price series |
| Price compared with its recent average | `log_ratio_3m`, `log_ratio_12m` | Price relative to its trailing average level |
| Time of year | `month_sin`, `month_cos` | Cyclical calendar seasonality |
| Long-run trend | `year` | Broad time trend |
| Market and crop | All `market_` and `commodity_` indicators | Market and commodity identity |

The prefix-based market/crop group captures all identity indicators, so the implementation does not depend on a fixed list of one-hot market or commodity names.

### Price and Direction Presentation

For origin price $P_t$ and grouped predicted log change $\hat y$:

$$
\hat P_{t+h}=P_t\exp(\hat y), \qquad
\mathrm{percentChange} = (\exp(\hat y)-1)\times100.
$$

Here, $t$ is the forecast origin month, $h$ is the horizon in months ($1$ for H1 or $2$ for H2), $P_t$ is the observed origin price in KES/kg, and $\hat y$ is the predicted cumulative log-price change. $\hat P_{t+h}$ is the predicted future price in KES/kg. $\exp$ is the natural exponential function, and `percentChange` is the corresponding predicted percentage change relative to the origin price. The multiplier $100$ converts the proportion to percent.

For display, origin and predicted prices and percentage values are rounded to two decimals; log values and log contributions are rounded to four decimals. A group's displayed percentage is calculated as $(\exp(c_g)-1)\times100$. Group percentages are **not additive** and do not sum to the overall expected percentage change, because percentage changes compound. The grouped log contributions are the additive quantities.

Direction is assigned from the unrounded group log contribution using the neutral band $[-0.001, 0.001]$: values above it are `increases`, below it are `decreases`, and values within it are `neutral`. Groups in each row are sorted by descending absolute log contribution. These labels describe the direction of a model attribution, not an independently measured causal effect.

## Data and Validation

The input feature matrices are the same eligible held-out origin rows used for final evaluation. H1 contains 88 rows (11 observations for each of eight market-commodity series); H2 contains 80 rows (10 per series). For every horizon the notebook:

1. Loads the saved fitted stack and saved final prediction CSV.
2. Selects held-out feature rows using the persisted fold mask and the model bundle's exact feature order.
3. Loads the existing per-row SHAP CSV instead of recomputing TreeExplainer values.
4. Checks horizon, row count, base values, market/commodity/month alignment, and origin-price/prediction alignment.
5. Builds all five group contributions for every held-out row, prints three complete examples, and computes mean absolute log contribution by group.
6. Saves grouped JSON locally and mirrors it to Google Drive.

The grouped reconciliation comparison is made against the saved stack prediction after `predicted_log_change` has been rounded to four decimal places. Therefore its maximum observed difference is dominated by display rounding, not by an omitted feature:

| Horizon | Held-Out Rows | Maximum Difference: Rounded Explanation vs Saved Stack Prediction |
| --- | ---: | ---: |
| H1 | 88 | 0.000050 |
| H2 | 80 | 0.000048 |

The unrounded group sum is validated internally against the full SHAP reconstruction to an absolute tolerance of $10^{-6}$.

## Mean Absolute Group Contribution

Mean absolute contribution is calculated by taking the absolute log contribution for each row, then averaging within group across the horizon's held-out rows. It indicates how much a group typically moves the model prediction away from its expected stack output; it does not indicate whether the group generally increases or decreases forecasts.

| Feature Group | H1 Mean Absolute Log Contribution | H2 Mean Absolute Log Contribution | Relative Pattern |
| --- | ---: | ---: | --- |
| Time of year | **0.017235** | **0.022142** | Largest group for both horizons |
| Long-run trend | 0.010912 | 0.019095 | Second largest for both horizons |
| Price compared with its recent average | 0.004799 | 0.005366 | Smaller than calendar/trend groups |
| Recent price momentum | 0.003293 | 0.003612 | Smaller average contribution |
| Market and crop | 0.001074 | 0.001638 | Smallest average group contribution |

H2's mean absolute values are larger for every group, consistent with the larger scale of the two-month cumulative target in this held-out sample. Because SHAP values are in target log-change units, group magnitudes should be compared within the model/horizon context; they are not price changes in KES.

## Example Row Interpretations

Examples below summarize the complete structured outputs retained in the notebook. Group log contributions are rounded to four decimal places and percentages to two decimals.

### H1: Kisumu Beans, November 2018

- Origin price: KES 71.23/kg; stacking forecast: KES 70.56/kg.
- Predicted cumulative log change: -0.0095, or -0.95% expected price change.
- Largest grouped contributions: Long-run trend -0.0112 (decreases); Time of year +0.0059 (increases); price versus recent average +0.0017 (increases).
- Recent momentum and market/crop are each approximately +0.0005 and classified neutral under the $\pm0.001$ threshold.

### H1: Nakuru Maize, September 2019

- Origin price: KES 33.28/kg; stacking forecast: KES 32.30/kg.
- Predicted cumulative log change: -0.0299, or -2.95% expected price change.
- Time of year (-0.0173) and Long-run trend (-0.0081) are the largest decreasing contributions. Recent-average position (+0.0027), market/crop identity (+0.0022), and momentum (-0.0027) partially offset them.

### H2: Nakuru Maize, August 2019

- Origin price: KES 40.87/kg; stacking forecast: KES 38.14/kg.
- Predicted cumulative log change: -0.0692, or -6.69% expected price change.
- Time of year (-0.0260), Long-run trend (-0.0167), and recent-average position (-0.0123) are the largest decreasing contributions; momentum and market/crop contribute smaller decreases.

The notebook emits full JSON-style grouped explanations for three sample rows per horizon. The examples are selected from the beginning, middle, and end of the saved prediction row order; they are illustrative, not selected as representative or extreme cases.

## Interpretation Limits

- SHAP attributes the model's forecast relative to the model's average output. It does not establish that a feature caused an observed price change.
- Group-level positive or negative contribution indicates how that feature group shifts this particular forecast from the base value, holding the additive decomposition fixed; it is not a causal intervention.
- Mean absolute contribution reports magnitude but removes direction. A feature group can have large absolute importance while increasing some row forecasts and decreasing others.
- Percentage contributions are individual transformed log contributions and are not additive; only the unrounded log contributions reconcile exactly.
- Feature grouping communicates model inputs more compactly but hides variation among member features, such as individual months, markets, or commodities.
- Explanations cover only the held-out rows and the saved final models. They do not quantify uncertainty, describe training-set behavior, or justify changing the already-evaluated model.

## Saved Outputs

The notebook saves the grouped explanations and mean absolute group summaries inside each JSON file:

- `results/artifacts/shap_grouped_h1.json`
- `results/artifacts/shap_grouped_h2.json`
- `MyDrive/capstone-data/explainability/shap_grouped_h1.json`
- `MyDrive/capstone-data/explainability/shap_grouped_h2.json`

Each file records the horizon, held-out row count, post-rounding reconciliation maximum, group summary, and a complete explanation object per row. Existing per-feature SHAP arrays, per-feature importance CSVs, and beeswarm plots are left unchanged by the grouped addendum.
