"""Batch forecast generation utilities for the model output layer."""

from __future__ import annotations

import pandas as pd

from agrisence.config.constants import CROPS, HORIZONS, MARKETS


def generate_all_forecasts(model, explainer, features_df: pd.DataFrame) -> list[dict]:
    """Generate model forecasts for all market, crop, and horizon combinations.

    The function conceptually loops over the Cartesian product of MARKETS x CROPS x HORIZONS
    to generate a final forecast table for each combination.
    """
    # TODO: iterate over all combination values and produce forecast records.
    # for market in MARKETS:
    #     for crop in CROPS:
    #         for horizon in HORIZONS:
    #             ...
    return []
