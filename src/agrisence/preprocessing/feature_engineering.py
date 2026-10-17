"""Feature engineering utilities for crop price modeling.

This module creates lag, rolling, and seasonal features for a multivariate forecasting task.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from agrisence.config.constants import CROPS, MARKETS


SERIES_COLUMNS = ("market", "commodity")
PRICE_COLUMN = "avg_price_per_kg"
FEATURE_COLUMNS = (
    "log_return_lag_1",
    "log_return_lag_2",
    "log_return_lag_3",
    "log_ratio_3m",
    "log_ratio_12m",
    "month_sin",
    "month_cos",
    "year",
)


def _prepare_panel(
    df: pd.DataFrame,
    series_cols: Sequence[str],
) -> tuple[pd.DataFrame, pd.Index]:
    required = {*series_cols, "month", PRICE_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")

    original_index = df.index
    panel = df.copy().reset_index(drop=True)
    panel["_feature_row_order"] = np.arange(len(panel))
    dates = pd.to_datetime(panel["month"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Panel month values must all be valid dates")
    panel["month"] = pd.to_datetime(
        {"year": dates.dt.year, "month": dates.dt.month, "day": 1}
    )

    duplicate_keys = panel.duplicated([*series_cols, "month"], keep=False)
    if duplicate_keys.any():
        raise ValueError("Panel must have unique series-month keys")

    prices = pd.to_numeric(panel[PRICE_COLUMN], errors="coerce")
    if (prices.dropna() <= 0).any():
        raise ValueError(f"{PRICE_COLUMN} must be positive wherever present")
    panel[PRICE_COLUMN] = prices
    panel = panel.sort_values([*series_cols, "month"], kind="stable").reset_index(drop=True)
    return panel, original_index


def _restore_order(panel: pd.DataFrame, original_index: pd.Index) -> pd.DataFrame:
    panel = panel.sort_values("_feature_row_order", kind="stable")
    panel = panel.drop(columns="_feature_row_order").reset_index(drop=True)
    panel.index = original_index
    return panel


def _add_identity_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    categories = {"market": MARKETS, "commodity": CROPS}
    for column, known_values in categories.items():
        if column not in panel.columns:
            continue
        unknown = set(panel[column].dropna().unique()).difference(known_values)
        if unknown:
            raise ValueError(f"Unexpected {column} values: {sorted(unknown)}")
        for value in known_values:
            panel[f"{column}_{value}"] = panel[column].eq(value).astype("int8")
    return panel


def build_features(
    df: pd.DataFrame,
    series_cols: tuple[str, ...] = SERIES_COLUMNS,
) -> pd.DataFrame:
    """Append causal log-return, trailing-log-ratio, calendar, and identity features."""
    panel, original_index = _prepare_panel(df, series_cols)
    panel["log_price"] = np.log(panel[PRICE_COLUMN])
    grouped_log = panel.groupby(list(series_cols), sort=False)["log_price"]

    for lag in (1, 2, 3):
        panel[f"log_return_lag_{lag}"] = (
            grouped_log.shift(lag) - grouped_log.shift(lag + 1)
        )

    for window in (3, 12):
        panel[f"log_ratio_{window}m"] = grouped_log.transform(
            lambda values: values.shift(1)
            - values.shift(1).rolling(window, min_periods=window).mean()
        )

    month_number = panel["month"].dt.month
    panel["month_sin"] = np.sin(2 * np.pi * (month_number - 1) / 12)
    panel["month_cos"] = np.cos(2 * np.pi * (month_number - 1) / 12)
    panel["year"] = panel["month"].dt.year
    panel = _add_identity_indicators(panel)
    return _restore_order(panel, original_index)


def build_targets(
    df: pd.DataFrame,
    series_cols: tuple[str, ...] = SERIES_COLUMNS,
) -> pd.DataFrame:
    """Append the exact-calendar cumulative log-change target one month ahead."""
    panel, original_index = _prepare_panel(df, series_cols)
    if "log_price" not in panel.columns:
        panel["log_price"] = np.log(panel[PRICE_COLUMN])
    if "impute_method" not in panel.columns:
        raise ValueError("Panel must include impute_method to retain target provenance")

    key_columns = [*series_cols, "month"]
    log_lookup = panel.set_index(key_columns)["log_price"]
    method_lookup = panel.set_index(key_columns)["impute_method"]

    for horizon in (1,):
        target_month = panel["month"] + pd.DateOffset(months=horizon)
        requested = panel[[*series_cols]].copy()
        requested["target_month"] = target_month.to_numpy()
        keys = pd.MultiIndex.from_frame(requested[[*series_cols, "target_month"]])
        target_log_price = log_lookup.reindex(keys).to_numpy()
        target_method = method_lookup.reindex(keys).to_numpy()

        panel[f"target_month_h{horizon}"] = target_month.to_numpy()
        panel[f"target_method_h{horizon}"] = target_method
        panel[f"target_h{horizon}"] = target_log_price - panel["log_price"].to_numpy()

    return _restore_order(panel, original_index)


def apply_calendar_distance_guard(
    df: pd.DataFrame,
    series_cols: tuple[str, ...] = SERIES_COLUMNS,
) -> pd.DataFrame:
    """Mask temporal features and targets whose source months are missing or misaligned."""
    panel, original_index = _prepare_panel(df, series_cols)
    if "log_price" not in panel.columns:
        panel["log_price"] = np.log(panel[PRICE_COLUMN])

    history_valid = pd.Series(True, index=panel.index)
    for _, indices in panel.groupby(list(series_cols), sort=False).groups.items():
        series = panel.loc[indices]
        price_by_month = series.set_index("month")["log_price"]
        for row_index, row in series.iterrows():
            month = row["month"]
            row_valid = pd.notna(row["log_price"])

            for lag in (1, 2, 3):
                column = f"log_return_lag_{lag}"
                if column not in panel.columns:
                    continue
                earlier = month - pd.DateOffset(months=lag)
                earlier_previous = month - pd.DateOffset(months=lag + 1)
                valid = (
                    earlier in price_by_month.index
                    and earlier_previous in price_by_month.index
                    and pd.notna(price_by_month.get(earlier))
                    and pd.notna(price_by_month.get(earlier_previous))
                )
                if not valid:
                    panel.at[row_index, column] = np.nan
                    row_valid = False

            for window in (3, 12):
                column = f"log_ratio_{window}m"
                if column not in panel.columns:
                    continue
                history_months = [
                    month - pd.DateOffset(months=offset)
                    for offset in range(1, window + 1)
                ]
                valid = all(
                    history_month in price_by_month.index
                    and pd.notna(price_by_month.get(history_month))
                    for history_month in history_months
                )
                if not valid:
                    panel.at[row_index, column] = np.nan
                    row_valid = False

            history_valid.at[row_index] = row_valid

    panel["consecutive_observation"] = history_valid.to_numpy(dtype=bool)

    key_columns = [*series_cols, "month"]
    log_lookup = panel.set_index(key_columns)["log_price"]
    method_lookup = (
        panel.set_index(key_columns)["impute_method"]
        if "impute_method" in panel.columns
        else None
    )
    for horizon in (1,):
        target_column = f"target_h{horizon}"
        if target_column not in panel.columns:
            continue
        expected_month = panel["month"] + pd.DateOffset(months=horizon)
        requested = panel[[*series_cols]].copy()
        requested["target_month"] = expected_month.to_numpy()
        keys = pd.MultiIndex.from_frame(requested[[*series_cols, "target_month"]])
        target_log_price = log_lookup.reindex(keys).to_numpy()
        valid_target = pd.notna(target_log_price) & panel["log_price"].notna().to_numpy()
        panel[target_column] = np.where(
            valid_target,
            target_log_price - panel["log_price"].to_numpy(),
            np.nan,
        )
        month_column = f"target_month_h{horizon}"
        if month_column in panel.columns:
            panel[month_column] = expected_month.to_numpy()
        method_column = f"target_method_h{horizon}"
        if method_column in panel.columns and method_lookup is not None:
            panel[method_column] = method_lookup.reindex(keys).to_numpy()

    return _restore_order(panel, original_index)


def filter_observed_eligible(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Keep rows with observed WFP prices at both origin and horizon target."""
    if horizon != "h1":
        raise ValueError("horizon must be 'h1'")
    target_column = f"target_{horizon}"
    target_method_column = f"target_method_{horizon}"
    required = {"impute_method", target_column, target_method_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Build targets before filtering; missing columns: {sorted(missing)}")

    eligible = (
        df["impute_method"].eq("observed")
        & df[target_method_column].eq("observed")
        & df[target_column].notna()
    )
    excluded = int((~eligible).sum())
    print(f"{horizon}: rows excluded for non-observed/missing endpoints: {excluded}")
    return df.loc[eligible].copy()


def build_supervised_table(
    df: pd.DataFrame,
    horizon: str,
    window_start: str = "2010-01-01",
    window_end: str = "2019-10-01",
) -> pd.DataFrame:
    """Build the observed-endpoint supervised table for the one-month horizon."""
    if horizon != "h1":
        raise ValueError("horizon must be 'h1'")
    if "month" not in df.columns:
        raise ValueError("Panel must include month")
    start = pd.Timestamp(window_start)
    end = pd.Timestamp(window_end)
    origin_dates = pd.to_datetime(df["month"], errors="coerce")
    if origin_dates.isna().any():
        raise ValueError("Panel month values must all be valid dates")
    origin_month = pd.to_datetime(
        {"year": origin_dates.dt.year, "month": origin_dates.dt.month, "day": 1}
    )
    in_window = origin_month.between(start, end)
    window_rows = df.loc[in_window].copy()
    eligible_rows = filter_observed_eligible(window_rows, horizon)
    target_month_column = f"target_month_{horizon}"
    if target_month_column not in eligible_rows.columns:
        raise ValueError(f"Build targets before supervised tables; missing {target_month_column}")
    eligible_rows["target_month"] = pd.to_datetime(eligible_rows[target_month_column])

    identity_features = [
        column for column in df.columns
        if column.startswith("market_") or column.startswith("commodity_")
    ]
    required_features = [*FEATURE_COLUMNS, *identity_features]
    missing_columns = set(required_features + [f"target_{horizon}"]).difference(eligible_rows.columns)
    if missing_columns:
        raise ValueError(f"Build features before supervised tables; missing: {sorted(missing_columns)}")

    missing_feature_mask = eligible_rows[required_features].isna().any(axis=1)
    missing_feature_count = int(missing_feature_mask.sum())
    supervised = eligible_rows.loc[~missing_feature_mask].copy()
    supervised["forecast_horizon"] = horizon
    print(
        f"{horizon}: origins in window={len(window_rows)}; "
        f"excluded for missing features={missing_feature_count}; "
        f"final supervised rows={len(supervised)}"
    )
    return supervised


def build_feature_set(df: pd.DataFrame, include_climate: bool = True) -> pd.DataFrame:
    """Backward-compatible wrapper for callers of the former scaffold API."""
    del include_climate
    return build_features(df)
