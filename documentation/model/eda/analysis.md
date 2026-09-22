# Exploratory Data Analysis Findings

## Scope and method

This analysis summarizes the executed cells in `model/notebooks/01_detailed_eda.ipynb`. The notebook examines the raw WFP Kenya food-price export after standardizing market and commodity labels. It focuses on market prices only; no cleaning, forward filling, outlier removal, climate data, or fuel-price joins are performed here.

The raw export contains 27,774 rows and 16 columns. The analysis filters the data to the markets listed in the project configuration and then evaluates commodity coverage, duplicate keys, missingness, reporting frequency, price anomalies, time-series behavior, and volatility.

## Market coverage

The market filter retained 3,881 rows. Four markets had observations in the executed dataset:

| Market | Records |
| --- | ---: |
| Nairobi | 1,762 |
| Kisumu | 962 |
| Mombasa | 616 |
| Nakuru | 541 |


Nairobi contributes approximately 45% of the retained rows, while Nakuru contributes approximately 14%. Consequently, an unweighted pooled analysis would be dominated by Nairobi and should not be interpreted as an equal-market comparison.

## Commodity coverage and selection

The coverage analysis identified four commodities with substantial observations across all four populated markets:

| Commodity | Records | Markets covered | Date range |
| --- | ---: | ---: | --- |
| Maize | 648 | 4 | 2006-01-15 to 2022-04-15 |
| Beans | 576 | 4 | 2006-01-15 to 2021-11-15 |
| Maize (White) | 563 | 4 | 2006-01-15 to 2020-03-15 |
| Beans (Dry) | 555 | 4 | 2006-01-15 to 2020-08-15 |

The notebook selected `Maize`, `Beans`, `Beans (Dry)`, and `Maize (White)`, producing a working subset of 2,342 rows. The coverage matrix shows that every selected commodity has observations in each of the four populated markets, but coverage is uneven. Nakuru has only 62 observations for both `Beans (Dry)` and `Maize (White)`, compared with 155 to 171 observations for those commodities in the other markets.

The selected labels should not automatically be treated as interchangeable products. `Maize` and `Maize (White)` may differ by product definition, unit, or market convention; the same concern applies to `Beans` and `Beans (Dry)`. The unit-consistency inspection should therefore be reviewed before comparing levels across commodities. The large differences in observed means also indicate that cross-commodity price levels are not directly comparable without accounting for units and product definitions.

## Time coverage and reporting frequency

The selected observations span dates from 2006 through 2022, but the end date differs by commodity. Maize has the longest coverage, ending in April 2022. Beans ends in November 2021, Beans (Dry) ends in August 2020, and Maize (White) ends in March 2020. Models trained on all four commodities will therefore operate on unbalanced time horizons unless a common analysis window is deliberately chosen.

The modal interval between observations is 31 days for every market-commodity combination. This is consistent with a monthly reporting process. The series should be regularized against an explicit monthly calendar during preprocessing, while preserving the distinction between genuinely missing observations and months that were never reported.

## Duplicate records

The duplicate check found two rows with the same `(market, commodity, date)` key. Both are Nairobi Maize observations dated 2020-08-15: one is a retail record priced at KES 52.00 and the other is a wholesale record priced at KES 32.91.

These are not identical duplicate measurements. They represent different `pricetype` values and should not be dropped blindly. The cleaning and modeling stages should either retain `pricetype` as a modeling dimension, select one price type consistently, or define an aggregation rule. A simple deduplication by market, commodity, and date would discard meaningful information and could introduce an arbitrary bias.

## Missingness

The coverage counts already identify the main structural risk: Nakuru has notably shorter histories for Beans (Dry) and Maize (White), so missingness and sparse coverage are likely to be more consequential for those series. Missing values must be handled in the preprocessing stage. Any imputation should be performed within each market-commodity series and should respect the monthly time index.

## Price anomalies and distribution

No zero or negative prices were found in the selected data. This removes one obvious data-quality failure mode, but it does not establish that all remaining values are valid. 

The selected commodities have materially different price scales. The executed volatility table reports means around 27 to 63 for Maize and Beans, versus roughly 2,497 to 6,344 for Maize (White) and Beans (Dry). This scale difference reinforces the need to model products separately or normalize features before pooled comparisons. It also suggests that unit and product-label validation is a prerequisite for interpreting price levels.

## Volatility

The coefficient of variation (standard deviation divided by mean) ranges from 0.147 to 0.345 across the 16 market-commodity combinations.

The highest observed relative volatility is:

- Mombasa Maize: CV 0.345
- Kisumu Maize (White): CV 0.336
- Mombasa Maize (White): CV 0.325
- Kisumu Maize: CV 0.323

The lowest observed relative volatility is Nakuru Beans, with a CV of 0.147, followed by Nakuru Beans (Dry) at 0.178. Relative volatility is therefore not uniform across markets: Mombasa and Kisumu are among the more variable locations for maize products, while Nakuru is comparatively stable for beans in this sample. These comparisons are descriptive and should be revisited after confirming units, price type, and the treatment of missing values.

## Seasonality and time-series structure

The notebook produced monthly seasonal plots and market-specific time-series plots. These visualizations should be interpreted as exploratory signals rather than evidence of causal seasonality. The monthly reporting interval and uneven date coverage can create apparent patterns if months are not represented consistently across markets and commodities.

## Implications for preprocessing and modeling

1. Confirm the units and definitions for Maize versus Maize (White), and Beans versus Beans (Dry), before making cross-series level comparisons.
2. Decide how to represent retail and wholesale observations. .
3. Create an explicit monthly calendar for each market-commodity series and quantify missingness numerically before choosing an imputation policy.
4. Treat the shorter histories for Nakuru Beans (Dry) and Nakuru Maize (White) as a data-availability limitation during train/test splitting.
5. Model commodities separately or use scale-aware transformations because price levels differ substantially across products.


## Conclusion

The dataset provides a usable monthly panel for four populated markets and well-covered commodity labels, with 2,342 selected observations. Its strongest modeling opportunities are also its main risks: product coverage is broad but unbalanced, price scales differ sharply, and two price types share a nominal key. The next stage should resolve those schema and coverage questions, quantify missingness, and establish a consistent market-commodity-price-type representation before training baselines or applying forecasting models.
