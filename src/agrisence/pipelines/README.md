# Pipelines

This folder contains the top-level workflow entrypoints for the project’s scheduled model jobs. These modules are intended to be used as the GitHub Actions execution layer once the project moves from prototype scaffolding to automated run schedules.

## Files

- `run_data_pipeline.py` — orchestrates the daily ingestion and preprocessing pipeline.
- `run_forecast_pipeline.py` — orchestrates the nightly batch forecasting and explainability workflow.
- `__init__.py` — marks the package for import.
