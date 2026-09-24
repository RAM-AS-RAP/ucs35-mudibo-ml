"""Cleaning utilities for the WFP price dataset.

Limitation: 48 long-gap rows (Kisumu/Mombasa Maize, Nairobi/Nakuru Beans) are filled from
a manually-sourced, one-time FEWS NET export (see load_fews_gap_files) because the FEWS API
is blocked when called programmatically. These rows have no overlapping WFP observation, so
there is no calibration check confirming FEWS and WFP prices agree; this must be stated as
an explicit limitation in the project documentation. Rows tagged
``external_fill_fews_extension`` represent a substantially larger, uncalibrated reliance on
a secondary data source than the original interior-gap backfill. This must be stated
explicitly and separately in the project's data limitations section, not folded into the
same sentence as the smaller 48-row backfill.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def fill_gaps_from_fews(
    df: pd.DataFrame,
    fews_fill_data: dict,
    group_cols: tuple[str, ...] = ("market", "commodity"),
) -> pd.DataFrame:
    """Fill missing targets directly from FEWS, with no WFP overlap to calibrate against."""
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
        for key in fews_fill_data
    }
    filled_counts = {key: 0 for key in fews_fill_data}

    for key, fews_df in fews_fill_data.items():
        series_mask = (result[group_cols[0]] == key[0]) & (result[group_cols[1]] == key[1])
        missing_months = {
            row.month
            for row in result.loc[series_mask, ["month", "avg_price_per_kg"]].itertuples(index=False)
            if pd.isna(row.avg_price_per_kg)
        }
        fews_by_month: dict = {}
        for row in fews_df[["month", "price_per_kg"]].itertuples(index=False):
            fews_by_month.setdefault(row.month, []).append(row.price_per_kg)
        for month, values in fews_by_month.items():
            if month not in missing_months:
                print(
                    f"WARNING: FEWS month {key[0]} {key[1]} {month} does not match "
                    "any missing row in the panel; skipping."
                )
                continue
            if len(values) > 1:
                print(
                    f"WARNING: skipping ambiguous FEWS month {key[0]} {key[1]} "
                    f"{month}; multiple product variants require manual filtering."
                )
                continue
            row_mask = series_mask & (result["month"] == month)
            result.loc[row_mask, "avg_price_per_kg"] = values[0]
            result.loc[row_mask, "impute_method"] = "external_fill_fews"
            filled_counts[key] += 1

    for key in fews_fill_data:
        remaining = original_missing[key] - filled_counts[key]
        print(
            f"FEWS fill summary {key[0]} {key[1]}: "
            f"{filled_counts[key]} of {original_missing[key]} originally missing "
            f"months filled; {remaining} remain unfilled"
        )
    return result


def load_fews_gap_files(file_map: dict) -> dict:
    """Load raw Postman-exported FEWS NET responses for the known missing gap windows."""
    fews_gap_data = {}
    for (market, commodity), file_path in file_map.items():
        path = Path(file_path)
        records = json.loads(path.read_text())
        raw = pd.DataFrame(records)

        variants = raw[["product", "cpcv2"]].drop_duplicates()
        if len(variants) > 1:
            print(
                f"WARNING: {path} contains multiple product variants for {market} "
                f"{commodity}: {variants.to_dict('records')}. Re-export from Postman "
                "with the correct filter before using this file."
            )

        if not raw["currency"].eq("KES").all():
            raise ValueError(f"{path}: all records must have currency == 'KES'")
        if not raw["common_unit"].eq("kg").all():
            raise ValueError(f"{path}: all records must have common_unit == 'kg'")

        fews_df = pd.DataFrame(
            {
                "month": pd.to_datetime(raw["start_date"]),
                "price_per_kg": pd.to_numeric(raw["common_unit_price"]),
            }
        ).reset_index(drop=True)
        fews_gap_data[(market, commodity)] = fews_df

        print(
            f"{market} {commodity}: loaded {len(fews_df)} records from {path}, "
            f"covering {fews_df['month'].min()} to {fews_df['month'].max()}"
        )
    return fews_gap_data


def fill_gaps_from_fews_files(
    df: pd.DataFrame,
    fews_gap_data: dict,
    group_cols: tuple[str, ...] = ("market", "commodity"),
) -> pd.DataFrame:
    """Fill missing targets from manually-loaded FEWS gap files; never overwrite non-missing rows."""
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
        for key in fews_gap_data
    }
    filled_counts = {key: 0 for key in fews_gap_data}

    for key, fews_df in fews_gap_data.items():
        series_mask = (result[group_cols[0]] == key[0]) & (result[group_cols[1]] == key[1])
        for row in fews_df[["month", "price_per_kg"]].itertuples(index=False):
            row_mask = series_mask & (result["month"] == row.month)
            if not row_mask.any():
                print(
                    f"WARNING: FEWS month {key[0]} {key[1]} {row.month} does not match "
                    "any row in the panel; skipping."
                )
                continue
            missing_mask = row_mask & result["avg_price_per_kg"].isna()
            if not missing_mask.any():
                print(
                    f"WARNING: FEWS month {key[0]} {key[1]} {row.month} already has a "
                    "WFP value; not overwriting."
                )
                continue
            result.loc[missing_mask, "avg_price_per_kg"] = row.price_per_kg
            result.loc[missing_mask, "impute_method"] = "external_fill_fews"
            filled_counts[key] += 1

    for key in fews_gap_data:
        remaining = original_missing[key] - filled_counts[key]
        print(
            f"FEWS file fill summary {key[0]} {key[1]}: "
            f"{filled_counts[key]} of {original_missing[key]} originally missing "
            f"months filled; {remaining} remain unfilled"
        )
    return result


def load_fews_extension_files(file_map: dict) -> dict:
    """Load manually-exported FEWS NET responses for extension-period coverage."""
    fews_extension_data = {}
    for (market, commodity), file_path in file_map.items():
        path = Path(file_path)
        records = json.loads(path.read_text())
        raw = pd.DataFrame(records)

        variants = raw[["product", "cpcv2"]].drop_duplicates()
        if len(variants) > 1:
            print(
                f"WARNING: {path} contains multiple product variants for {market} "
                f"{commodity}: {variants.to_dict('records')}. Re-export from Postman "
                "with the correct filter before using this file."
            )

        if not raw["currency"].eq("KES").all():
            raise ValueError(f"{path}: all records must have currency == 'KES'")
        if not raw["common_unit"].eq("kg").all():
            raise ValueError(f"{path}: all records must have common_unit == 'kg'")

        start_dates = pd.to_datetime(raw["start_date"])
        fews_df = pd.DataFrame(
            {
                "month": start_dates.map(lambda value: pd.Timestamp(value.year, value.month, 1)),
                "price_per_kg": pd.to_numeric(raw["common_unit_price"]),
            }
        ).reset_index(drop=True)
        fews_extension_data[(market, commodity)] = fews_df

        print(
            f"{market} {commodity}: loaded {len(fews_df)} records from {path}, "
            f"covering {fews_df['month'].min()} to {fews_df['month'].max()}"
        )
    return fews_extension_data


def extend_panel_with_fews(
    df: pd.DataFrame,
    fews_ext_data: dict,
    group_cols: tuple[str, ...] = ("market", "commodity"),
    tolerance: float = 0.10,
) -> pd.DataFrame:
    """Extend a panel with FEWS values without overwriting populated observations."""
    result = df.copy()
    result["month"] = pd.to_datetime(result["month"]).map(
        lambda value: pd.Timestamp(value.year, value.month, 1)
    )
    if "impute_method" not in result.columns:
        result["impute_method"] = "observed"

    for key, fews_df in fews_ext_data.items():
        series_mask = (result[group_cols[0]] == key[0]) & (result[group_cols[1]] == key[1])
        months_added = 0
        months_already_present = 0
        discrepancies = 0
        ambiguous_months = set()
        grouped_fews = fews_df.groupby("month", sort=True)

        for month, month_rows in grouped_fews:
            if len(month_rows) > 1:
                ambiguous_months.add(month)
                continue
            fews_value = month_rows.iloc[0]["price_per_kg"]
            row_mask = series_mask & result["month"].eq(month)
            populated_mask = row_mask & result["avg_price_per_kg"].notna()
            if populated_mask.any():
                months_already_present += 1
                existing_value = float(result.loc[populated_mask, "avg_price_per_kg"].iloc[0])
                if existing_value == 0:
                    differs = float(fews_value) != 0
                else:
                    differs = abs(existing_value - float(fews_value)) / abs(existing_value) > tolerance
                if differs:
                    discrepancies += 1
                    print(
                        f"FEWS overlap discrepancy {key[0]} {key[1]} {month}: "
                        f"existing={existing_value}, fews={fews_value}"
                    )
                continue

            if row_mask.any():
                result.loc[row_mask, "avg_price_per_kg"] = fews_value
                result.loc[row_mask, "impute_method"] = "external_fill_fews_extension"
            else:
                new_row = {column: None for column in result.columns}
                new_row.update(
                    {
                        group_cols[0]: key[0],
                        group_cols[1]: key[1],
                        "month": month,
                        "avg_price_per_kg": fews_value,
                        "impute_method": "external_fill_fews_extension",
                    }
                )
                result.loc[len(result)] = new_row
            months_added += 1

        if ambiguous_months:
            print(
                f"WARNING: skipping ambiguous FEWS months for {key[0]} {key[1]}: "
                f"{sorted(ambiguous_months)}; re-export with the correct product filter."
            )
        series_result = result[
            (result[group_cols[0]] == key[0]) & (result[group_cols[1]] == key[1])
        ]
        last_month = series_result["month"].max()
        print(
            f"FEWS extension summary {key[0]} {key[1]}: {months_added} months added, "
            f"{months_already_present} months already present, "
            f"{discrepancies} discrepancies, last record {last_month}"
        )

    return result.sort_values([*group_cols, "month"]).reset_index(drop=True)


