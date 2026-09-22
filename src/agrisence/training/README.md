# Training

This folder contains the modeling workflow for the crop price forecasting project, including statistical baselines, the ensemble learners, and model evaluation. It is the main package where feature sets are trained and scored before deployment into inference or reporting.

## Files

- `train.py` — orchestrates the full training workflow from feature loading to serialization.
- `evaluate.py` — computes MAE, RMSE, and MAPE metrics for model comparison.
- `baselines/` — linear regression and SARIMA benchmark models.
- `ensemble/` — Random Forest, XGBoost, and stacking ensemble components.
- `__init__.py` — marks the training package for import.
