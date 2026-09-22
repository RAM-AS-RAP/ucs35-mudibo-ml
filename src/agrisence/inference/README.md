# Inference

This folder contains the model-loading and prediction routines used after training is complete. It is designed for batch-oriented forecast generation across all market, crop, and horizon combinations.

## Files

- `model_loader.py` — loads a trained model artifact from disk for the current forecast run.
- `batch_forecast.py` — generates forecast records across market, crop, and horizon combinations.
- `__init__.py` — marks the package for import.
