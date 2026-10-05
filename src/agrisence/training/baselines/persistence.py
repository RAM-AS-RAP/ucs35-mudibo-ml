"""Persistence baseline for log-change forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_persistence(df: pd.DataFrame) -> np.ndarray:
    """Predict zero log change for every supplied origin row."""
    return np.zeros(len(df), dtype=float)
