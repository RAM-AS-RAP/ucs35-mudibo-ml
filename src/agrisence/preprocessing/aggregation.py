"""Aggregation utilities for converting WFP observations to monthly views."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def aggregate_to_monthly(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "price_per_kg",
    group_cols: tuple[str, ...] = ("market", "commodity"),
) -> pd.DataFrame:
    """Return mean price per kilogram for each market-crop-month."""
    result = df.copy()
    def month_start(value):
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        return datetime(value.year, value.month, 1)

    result["_month_key"] = result[date_col].map(month_start).map(lambda value: value.strftime("%Y-%m-%d"))
    aggregated = (
        result.groupby([*group_cols, "_month_key"], as_index=False)[value_col]
        .mean()
        .rename(columns={"_month_key": "month", value_col: "avg_price_per_kg"})
    )
    aggregated["month"] = pd.to_datetime(aggregated["month"])
    return aggregated
