"""Climate data ingestion utilities using the Open-Meteo Historical Archive API."""

from __future__ import annotations

import pandas as pd


def fetch_climate_data(
    market: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily rainfall and temperature data for a market window.

    The data source is the Open-Meteo Historical Archive API, which provides rainfall and
    temperature time series needed for climate-aware price forecasting.
    """
    # TODO: call the Open-Meteo API, parse JSON responses, and return a DataFrame.
    return pd.DataFrame()
