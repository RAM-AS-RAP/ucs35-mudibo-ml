# AgriSence Model

This folder contains the machine learning component of the capstone project: a daily data ingestion and feature engineering pipeline, a nightly forecast generation workflow with SHAP explainability, and a stacking ensemble benchmarked against statistical baselines. The project is intentionally organized to support scheduled execution from GitHub Actions later, while keeping the current state focused on clean, testable scaffolding.

## Subfolder overview

- `config/` — fixed project constants and environment-based configuration.
- `ingestion/` — source data fetchers for WFP, climate, and EPRA inputs.
- `preprocessing/` — cleaning, aggregation, and feature generation utilities.
- `training/` — baseline and ensemble model training logic plus evaluation.
- `explainability/` — SHAP-based interpretation utilities.
- `inference/` — model loading and batch forecasting helpers.
- `pipelines/` — orchestrator entrypoints for the data and forecast workflows.
- `storage/` — Supabase data-write stubs pending schema creation.
- `artifacts/` — serialized model outputs and checkpoint storage.
- `notebooks/` — exploratory and model comparison notebooks.
- `tests/` — pytest placeholders for core model modules.

> Supabase integration is intentionally stubbed for now; the database schema will be created through a later migration step in a separate task.
