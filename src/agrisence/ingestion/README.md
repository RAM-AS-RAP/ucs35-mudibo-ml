# Ingestion

This folder contains raw data acquisition utilities for the forecasting pipeline. It is intended to collect WFP price data, climate context from Open-Meteo, and EPRA fuel inputs before they are cleaned and aligned.

## Files

- `fetch_wfp.py` — downloads and normalizes the WFP Kenya food price dataset.
- `fetch_openmeteo.py` — fetches historical climate features for a market latitude/longitude pair.
- `fetch_epra.py` — loads monthly EPRA fuel price series for the forecasting feature set.
- `market_coordinates.py` — maps each market to a placeholder coordinate pair for climate lookups.
- `__init__.py` — marks the ingestion package for imports.
