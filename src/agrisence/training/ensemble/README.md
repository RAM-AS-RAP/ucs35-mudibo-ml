# Training / Ensemble

This folder contains the main model candidates for the final forecasting system. The ensemble layer is designed to combine Random Forest and XGBoost learners under a linear regression meta-learner, while the project also evaluates simpler statistical baselines.

## Files

- `random_forest.py` — trains the random forest learner with optional tuning.
- `xgboost_model.py` — trains the XGBoost learner with optional tuning.
- `stacking.py` — assembles the stacked ensemble using the base models and meta-learner.
- `__init__.py` — marks the package for import.
