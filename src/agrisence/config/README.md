# Config

This folder contains the fixed scope definitions and environment-based configuration used throughout the forecasting project. It holds the agreed market, crop, and horizon constants together with simple settings loaded from a local `.env` file.

## Files

- `constants.py` — defines the project markets, crop list, forecast horizons, and derived total combinations.
- `settings.py` — loads environment-based settings such as API keys and Supabase placeholders from `.env`.
- `__init__.py` — marks the config package for import.
