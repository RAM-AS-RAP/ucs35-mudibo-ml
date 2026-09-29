"""Chronological development and final-test folds for direct forecasts."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd


class FoldBoundary(TypedDict):
    train_origin_start: pd.Timestamp
    train_origin_end: pd.Timestamp
    train_target_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


class ExpandingWindowSplit(TypedDict):
    test_mask: pd.Series
    development_mask: pd.Series
    folds: list[FoldBoundary]


def build_expanding_window_folds(
    df: pd.DataFrame,
    dev_end: str = "2018-10-01",
    test_start: str = "2018-11-01",
    test_end: str = "2019-10-01",
) -> ExpandingWindowSplit:
    """Create shared origin-month masks and expanding annual validation boundaries.

    Input tables are horizon-specific outputs of ``build_supervised_table`` and must
    include ``month``, ``target_month``, and ``forecast_horizon``. Fold validation blocks are
    calendar-year blocks within development, after a five-year initial training span.
    The training-label cutoff is embargoed by the horizon so every training target
    would have been observable at the first forecast origin in its validation block.
    """
    required = {"month", "target_month", "forecast_horizon"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Supervised table is missing fold columns: {sorted(missing)}")

    target_dates = pd.to_datetime(df["target_month"], errors="coerce")
    if target_dates.isna().any():
        raise ValueError("target_month must contain valid dates for every supervised row")
    target_month = pd.to_datetime(
        {"year": target_dates.dt.year, "month": target_dates.dt.month, "day": 1}
    )
    origin_dates = pd.to_datetime(df["month"], errors="coerce")
    if origin_dates.isna().any():
        raise ValueError("month must contain valid origin dates for every supervised row")
    origin_month = pd.to_datetime(
        {"year": origin_dates.dt.year, "month": origin_dates.dt.month, "day": 1}
    )

    horizons = set(df["forecast_horizon"].dropna().astype(str).unique())
    if len(horizons) != 1 or not horizons.issubset({"h1", "h2"}):
        raise ValueError("Input must contain rows for exactly one forecast_horizon ('h1' or 'h2')")
    horizon_months = int(next(iter(horizons))[1:])

    raw_boundaries = [pd.Timestamp(value) for value in (dev_end, test_start, test_end)]
    development_end, final_test_start, final_test_end = [
        pd.Timestamp(year=value.year, month=value.month, day=1)
        for value in raw_boundaries
    ]
    if development_end >= final_test_start:
        raise ValueError("dev_end must be earlier than test_start")
    if final_test_start > final_test_end:
        raise ValueError("test_start must be on or before test_end")

    development_start = target_month.min()
    development_start = origin_month.min()
    development_mask = origin_month.between(development_start, development_end)
    test_mask = origin_month.between(final_test_start, final_test_end)
    overlap = development_mask & test_mask
    if overlap.any():
        raise ValueError("Development and final-test origin months must not overlap")

    first_validation_year = (development_start + pd.DateOffset(years=5)).year
    first_validation_start = pd.Timestamp(year=first_validation_year, month=1, day=1)
    folds: list[FoldBoundary] = []
    validation_start = first_validation_start
    while validation_start <= development_end:
        validation_end = min(
            pd.Timestamp(year=validation_start.year, month=12, day=1),
            development_end,
        )
        train_target_end = validation_start - pd.DateOffset(months=horizon_months)
        train_target_end = validation_start
        train_origin_end = train_target_end - pd.DateOffset(months=horizon_months)
        if train_origin_end >= development_start:
            folds.append(
                {
                    "train_origin_start": development_start,
                    "train_origin_end": train_origin_end,
                    "train_target_end": train_target_end,
                    "validation_start": validation_start,
                    "validation_end": validation_end,
                }
            )
        validation_start += pd.DateOffset(years=1)

    return {
        "test_mask": pd.Series(test_mask.to_numpy(), index=df.index, name="is_test"),
        "development_mask": pd.Series(
            development_mask.to_numpy(), index=df.index, name="is_development"
        ),
        "folds": folds,
    }
