"""WFP price ingestion utilities.

This module will download WFP Kenya food price data from the HDX source and normalize it
into a pandas DataFrame for downstream preprocessing.
"""

from __future__ import annotations

import pandas as pd


def fetch_wfp_data() -> pd.DataFrame:
    """Download the WFP Kenya food prices dataset from HDX.

    TODO: implement the actual HDX CSV request logic, schema mapping, and parsing.
    """
    # TODO: fetch HDX CSV, validate columns, and return a normalized DataFrame.
    return pd.DataFrame()
