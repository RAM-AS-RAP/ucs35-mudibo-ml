"""Cleaning utilities for the WFP price dataset."""

from __future__ import annotations

import pandas as pd
import logging
from datetime import datetime
from itertools import product


LOGGER = logging.getLogger(__name__)


def filter_target_scope(df: pd.DataFrame, markets: list[str], raw_commodity_map: dict[str, str]) -> pd.DataFrame:
    """Filter markets and relabel commodities.

    Market and commodity values should already be standardized by stripping
    whitespace and title-casing before this function runs.
    """
    filtered = df[df["market"].isin(markets)].copy()
    filtered["commodity"] = filtered["commodity"].replace(raw_commodity_map)
    return filtered


def normalize_units(df: pd.DataFrame, price_col: str = "price", unit_col: str = "unit") -> pd.DataFrame:
    """Add ``price_per_kg`` for supported WFP units and flag others."""
    cleaned = df.copy()
    cleaned["price_per_kg"] = pd.NA
    kg_mask = cleaned[unit_col].eq("KG")
    ninety_kg_mask = cleaned[unit_col].eq("90 KG")
    cleaned.loc[kg_mask, "price_per_kg"] = cleaned.loc[kg_mask, price_col]
    cleaned.loc[ninety_kg_mask, "price_per_kg"] = cleaned.loc[ninety_kg_mask, price_col] / 90
    cleaned["price_per_kg"] = pd.to_numeric(cleaned["price_per_kg"], errors="coerce")
    unhandled = sorted(cleaned.loc[~(kg_mask | ninety_kg_mask), unit_col].dropna().unique())
    if unhandled:
        message = f"Unhandled unit values; price_per_kg set to NaN: {unhandled}"
        LOGGER.warning(message)
        print(message)
    return cleaned


def resolve_pricetype(df: pd.DataFrame, preferred: str = "Wholesale", pricetype_col: str = "pricetype") -> pd.DataFrame:
    """Print pricetype counts and retain only the preferred pricetype."""
    print(df.groupby(["market", "commodity", pricetype_col]).size().rename("records"))
    return df[df[pricetype_col].eq(preferred)].copy()


def resolve_label_overlap(df: pd.DataFrame, date_col: str = "date", group_cols: tuple[str, ...] = ("market", "commodity")) -> pd.DataFrame:
    """Average overlapping rows after commodity labels have been merged."""
    keys = [*group_cols, date_col]
    duplicate_mask = df.duplicated(keys, keep=False)
    print(f"Overlapping (market, commodity, date) rows averaged: {int(duplicate_mask.sum())}")
    if not duplicate_mask.any():
        return df.copy()
    rows = []
    for _, group in df.groupby(keys, sort=False, dropna=False):
        row = group.iloc[0].copy()
        if "price_per_kg" in group:
            row["price_per_kg"] = group["price_per_kg"].mean()
        if "price" in group:
            row["price"] = group["price"].mean()
        rows.append(row)
    return pd.DataFrame(rows, columns=df.columns).reset_index(drop=True)


def build_monthly_calendar(markets: list[str], crops: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Return every market-crop combination for each month in the range."""
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)
    start_month = datetime(start_date.year, start_date.month, 1)
    end_month = datetime(end_date.year, end_date.month, 1)
    months = []
    current_month = start_month
    while current_month <= end_month:
        months.append(current_month)
        next_year = current_month.year + (current_month.month == 12)
        next_month = 1 if current_month.month == 12 else current_month.month + 1
        current_month = datetime(next_year, next_month, 1)
    return pd.DataFrame(
        product(markets, crops, months),
        columns=["market", "commodity", "month"],
    )


def quantify_missingness(observed_df: pd.DataFrame, calendar_df: pd.DataFrame, join_cols: tuple[str, ...] = ("market", "commodity", "month")) -> pd.DataFrame:
    """Summarize missing calendar months by market and commodity."""
    observed = observed_df[list(join_cols)].drop_duplicates().assign(_observed=True)
    panel = calendar_df[list(join_cols)].merge(observed, on=list(join_cols), how="left")
    summary = panel.assign(_missing=panel["_observed"].isna()).groupby(list(join_cols[:-1]))["_missing"].agg(missing_months="sum", expected_months="size")
    summary["missing_proportion"] = summary["missing_months"] / summary["expected_months"]
    print(f"Overall missingness: {panel['_observed'].isna().mean():.2%}")
    return summary.reset_index()


def forward_fill_short_gaps(df: pd.DataFrame, value_col: str = "price_per_kg", max_gap_months: int = 2, group_cols: tuple[str, ...] = ("market", "commodity")) -> pd.DataFrame:
    """Fill only complete missing runs no longer than ``max_gap_months``."""
    result = df.copy().sort_values([*group_cols, "month"]).reset_index(drop=True)
    result["was_imputed"] = False
    for _, index in result.groupby(list(group_cols), sort=False).groups.items():
        labels = list(index)
        position = 0
        while position < len(labels):
            label = labels[position]
            if pd.notna(result.at[label, value_col]):
                position += 1
                continue
            gap_start = position
            while position < len(labels) and pd.isna(result.at[labels[position], value_col]):
                position += 1
            gap_length = position - gap_start
            if gap_length <= max_gap_months and gap_start > 0:
                previous_value = result.at[labels[gap_start - 1], value_col]
                if pd.notna(previous_value):
                    for gap_position in range(gap_start, position):
                        result.at[labels[gap_position], value_col] = previous_value
                        result.at[labels[gap_position], "was_imputed"] = True
    return result


def diagnose_missing_gap_lengths(df: pd.DataFrame, value_col: str = "price_per_kg", group_cols: tuple[str, ...] = ("market", "commodity")) -> pd.DataFrame:
    """Summarize consecutive-missing-month run lengths per group.

    Returns one row per missing run with its length, so short/medium/long
    gap thresholds can be chosen from evidence rather than guessed.
    """
    ordered = df.copy().sort_values([*group_cols, "month"]).reset_index(drop=True)
    runs = []
    for group_key, index in ordered.groupby(list(group_cols), sort=False).groups.items():
        labels = list(index)
        position = 0
        while position < len(labels):
            label = labels[position]
            if pd.notna(ordered.at[label, value_col]):
                position += 1
                continue
            gap_start = position
            while position < len(labels) and pd.isna(ordered.at[labels[position], value_col]):
                position += 1
            runs.append(
                {
                    **dict(zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,))),
                    "gap_start_month": ordered.at[labels[gap_start], "month"],
                    "gap_length_months": position - gap_start,
                    "is_edge_gap": gap_start == 0 or position == len(labels),
                }
            )
    return pd.DataFrame(runs)


def list_missing_rows(
    df: pd.DataFrame,
    value_col: str = "price_per_kg",
    group_cols: tuple[str, ...] = ("market", "commodity"),
    month_col: str = "month",
) -> pd.DataFrame:
    """Return every row whose ``value_col`` is missing, annotated with gap context.

    Each row carries the length of the missing run it belongs to, its position
    within that run, and whether the run is leading, trailing, or interior to
    the series, which separates coverage gaps from genuine reporting gaps.
    """
    ordered = df.copy().sort_values([*group_cols, month_col]).reset_index(drop=True)
    records = []
    for _, index in ordered.groupby(list(group_cols), sort=False).groups.items():
        labels = list(index)
        position = 0
        while position < len(labels):
            if pd.notna(ordered.at[labels[position], value_col]):
                position += 1
                continue
            gap_start = position
            while position < len(labels) and pd.isna(ordered.at[labels[position], value_col]):
                position += 1
            gap_length = position - gap_start
            if gap_start == 0 and position == len(labels):
                gap_position = "entire_series"
            elif gap_start == 0:
                gap_position = "leading"
            elif position == len(labels):
                gap_position = "trailing"
            else:
                gap_position = "interior"
            for offset, gap_index in enumerate(range(gap_start, position), start=1):
                record = ordered.loc[labels[gap_index]].to_dict()
                record["gap_length_months"] = gap_length
                record["month_in_gap"] = offset
                record["gap_position"] = gap_position
                records.append(record)
    columns = [*ordered.columns, "gap_length_months", "month_in_gap", "gap_position"]
    return pd.DataFrame(records, columns=columns)


def seasonal_interpolate_medium_gaps(
    df: pd.DataFrame,
    value_col: str = "price_per_kg",
    group_cols: tuple[str, ...] = ("market", "commodity"),
    min_gap_months: int = 3,
    max_gap_months: int = 6,
) -> pd.DataFrame:
    """Fill missing runs strictly between ``min_gap_months`` and ``max_gap_months``.

    Each missing month is filled only from the same calendar month one or two
    years earlier. Future-year values and forward interpolation are deliberately
    excluded to preserve walk-forward, no-future-information discipline. If no
    prior-year seasonal reference exists, the value remains missing. Gaps shorter
    than ``min_gap_months`` or longer than ``max_gap_months`` are left untouched
    so callers can combine this with ``forward_fill_short_gaps`` and leave
    genuinely unfillable gaps missing.
    """
    result = df.copy().sort_values([*group_cols, "month"]).reset_index(drop=True)
    if "was_interpolated" not in result.columns:
        result["was_interpolated"] = False

    for _, index in result.groupby(list(group_cols), sort=False).groups.items():
        labels = list(index)
        position = 0
        while position < len(labels):
            label = labels[position]
            if pd.notna(result.at[label, value_col]):
                position += 1
                continue
            gap_start = position
            while position < len(labels) and pd.isna(result.at[labels[position], value_col]):
                position += 1
            gap_length = position - gap_start
            if not (min_gap_months <= gap_length <= max_gap_months):
                continue

            for gap_position in range(gap_start, position):
                gap_label = labels[gap_position]
                gap_month = result.at[gap_label, "month"]
                seasonal_value = None
                for year_offset in (-1, -2):
                    candidate_month = pd.Timestamp(
                        year=gap_month.year + year_offset, month=gap_month.month, day=1
                    )
                    mask = result["month"] == candidate_month
                    for col in group_cols:
                        mask &= result[col] == result.at[gap_label, col]
                    match = result.loc[mask, value_col]
                    if len(match) and pd.notna(match.iloc[0]):
                        seasonal_value = match.iloc[0]
                        break
                if seasonal_value is not None:
                    result.at[gap_label, value_col] = seasonal_value
                    result.at[gap_label, "was_interpolated"] = True
    return result


def flag_outliers_iqr(df: pd.DataFrame, value_col: str = "price_per_kg", group_cols: tuple[str, ...] = ("market", "commodity"), multiplier: float = 1.5) -> pd.DataFrame:
    """Add ``is_outlier`` using group-wise IQR bounds without removing values."""
    result = df.copy()
    grouped = result.groupby(list(group_cols))[value_col]
    q1 = grouped.transform("quantile", 0.25)
    q3 = grouped.transform("quantile", 0.75)
    iqr = q3 - q1
    result["is_outlier"] = result[value_col].notna() & ((result[value_col] < q1 - multiplier * iqr) | (result[value_col] > q3 + multiplier * iqr))
    return result


def forward_fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for short-gap filling."""
    return forward_fill_short_gaps(df)


def calibrate_fews_source(
    wfp_df: pd.DataFrame, fews_df: pd.DataFrame, market: str, commodity: str
) -> dict:
    """Compare FEWS and WFP prices on months observed by both sources."""
    fews_by_month = {}
    for row in fews_df[["month", "price_per_kg"]].itertuples(index=False):
        fews_by_month.setdefault(row.month, []).append(row.price_per_kg)
    wfp_values = []
    fews_values = []
    for row in wfp_df[["month", "avg_price_per_kg"]].itertuples(index=False):
        for fews_value in fews_by_month.get(row.month, []):
            if pd.notna(row.avg_price_per_kg) and pd.notna(fews_value):
                wfp_values.append(row.avg_price_per_kg)
                fews_values.append(fews_value)
    n_overlap = len(wfp_values)
    if n_overlap:
        differences = [abs(wfp - fews) for wfp, fews in zip(wfp_values, fews_values)]
        mean_abs_diff = float(sum(differences) / n_overlap)
        correlation = float(pd.Series(wfp_values).corr(pd.Series(fews_values)))
    else:
        mean_abs_diff = float("nan")
        correlation = float("nan")
    review_note = ""
    if n_overlap < 6:
        review_note = "; fewer than 6 overlaps, manual review required"
    print(
        f"FEWS calibration {market} {commodity}: n={n_overlap}, "
        f"mean_abs_diff={mean_abs_diff:.4f}, correlation={correlation:.4f}"
        f"{review_note}"
    )
    # The correlation threshold is provisional and must be confirmed against
    # the notebook's visual overlay before treating it as final.
    return {
        "is_calibrated": bool(correlation > 0.9),
        "n_overlap_months": n_overlap,
        "mean_abs_diff": mean_abs_diff,
        "correlation": correlation,
    }


def fill_gaps_from_fews(
    df: pd.DataFrame,
    fews_data: dict,
    calibration_results: dict,
    group_cols: tuple[str, ...] = ("market", "commodity"),
) -> pd.DataFrame:
    """Fill missing targets from calibrated FEWS series with explicit provenance."""
    result = df.copy()
    if "impute_method" not in result:
        result["impute_method"] = "observed"

    original_missing = {
        key: int(
            result.loc[
                (result[group_cols[0]] == key[0])
                & (result[group_cols[1]] == key[1]),
                "avg_price_per_kg",
            ].isna().sum()
        )
        for key in fews_data
    }
    filled_counts = {key: 0 for key in fews_data}

    for key, fews_df in fews_data.items():
        calibration = calibration_results.get(key, {"is_calibrated": False})
        if not calibration.get("is_calibrated", False):
            print(
                f"WARNING: {key[0]} {key[1]} is not calibrated; manual review "
                "is required before applying a FEWS fill."
            )
            continue

        series_mask = (result[group_cols[0]] == key[0]) & (result[group_cols[1]] == key[1])
        fews_by_month = fews_df.groupby("month")["price_per_kg"].agg(list)
        for row_index in result.index[series_mask & result["avg_price_per_kg"].isna()]:
            month = result.at[row_index, "month"]
            values = fews_by_month.get(month, [])
            if len(values) == 1 and pd.notna(values[0]):
                result.at[row_index, "avg_price_per_kg"] = values[0]
                result.at[row_index, "impute_method"] = "external_fill_fews"
                filled_counts[key] += 1
            elif len(values) > 1:
                print(
                    f"WARNING: skipping ambiguous FEWS month {key[0]} {key[1]} "
                    f"{month}; multiple product variants require manual filtering."
                )

    for key in fews_data:
        remaining = original_missing[key] - filled_counts[key]
        print(
            f"FEWS fill summary {key[0]} {key[1]}: "
            f"{filled_counts[key]} of {original_missing[key]} originally missing "
            f"months filled; {remaining} remain unfilled"
        )
    return result


