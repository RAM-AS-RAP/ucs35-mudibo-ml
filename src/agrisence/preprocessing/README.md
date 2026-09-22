# Preprocessing

This folder contains all data-cleaning and feature-generation logic needed before a forecasting model is trained. It includes missing-value handling, monthly rollups, and the engineered variables used in the baselines and the ensemble stack.

## Files

- `cleaning.py` — applies forward-fill and IQR-based outlier detection rules.
- `aggregation.py` — aggregates raw observations into the monthly modeling view.
- `feature_engineering.py` — creates lag, rolling, seasonal, and optional climate features.
- `__init__.py` — marks the package for import.
