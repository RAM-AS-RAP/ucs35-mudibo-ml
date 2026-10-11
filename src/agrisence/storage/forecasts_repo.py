"""Database reads and idempotent writes for generated forecast rows only."""

from decimal import Decimal, ROUND_HALF_UP
import json

import pandas as pd

from agrisence.storage.prices_repo import redact_exception


LIVE_UPSERT_SQL = """
insert into forecasts (market, crop, horizon, origin_month, target_month, origin_price,
    predicted_log_change, predicted_price, shap_values, is_backfill)
values (%s, %s::crop_type, %s::forecast_horizon, %s, %s, %s, %s, %s, %s, false)
on conflict (market, crop, horizon, origin_month) do update
  set target_month = excluded.target_month,
      origin_price = excluded.origin_price,
      predicted_log_change = excluded.predicted_log_change,
      predicted_price = excluded.predicted_price,
      shap_values = excluded.shap_values,
      generated_at = now()
  where forecasts.is_backfill = false
    and (forecasts.origin_price is distinct from excluded.origin_price
      or forecasts.predicted_log_change is distinct from excluded.predicted_log_change
      or forecasts.predicted_price is distinct from excluded.predicted_price
      or forecasts.shap_values is distinct from excluded.shap_values)
"""
BACKFILL_INSERT_SQL = """
insert into forecasts (market, crop, horizon, origin_month, target_month, origin_price,
    predicted_log_change, predicted_price, shap_values, is_backfill)
values (%s, %s::crop_type, %s::forecast_horizon, %s, %s, %s, %s, %s, %s, true)
on conflict (market, crop, horizon, origin_month) do nothing
"""


def fetch_prices(conn, markets, crops, start_month) -> pd.DataFrame:
    """Read observed market_prices for the enabled inference scope."""
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "select market, crop::text, month, avg_price_per_kg from market_prices "
                "where market = any(%s) and crop::text = any(%s) and month >= %s "
                "and avg_price_per_kg > 0 order by market, crop, month",
                (list(markets), list(crops), start_month),
            )
            records = cursor.fetchall()
        return pd.DataFrame(records, columns=["market", "crop", "month", "avg_price_per_kg"]).assign(
            avg_price_per_kg=lambda frame: pd.to_numeric(frame["avg_price_per_kg"], errors="raise").astype(float),
        )
    except Exception as exception:
        raise RuntimeError(f"Could not read inference prices: {redact_exception(exception)}") from None


def fetch_existing_forecasts(conn, markets, crops) -> dict:
    """Read forecast rows for idempotency and backfill planning."""
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "select market, crop::text, horizon::text, origin_month, target_month, origin_price, "
                "predicted_log_change, predicted_price, shap_values, is_backfill from forecasts "
                "where market = any(%s) and crop::text = any(%s)",
                (list(markets), list(crops)),
            )
            rows = cursor.fetchall()
        columns = (
            "market", "crop", "horizon", "origin_month", "target_month", "origin_price",
            "predicted_log_change", "predicted_price", "shap_values", "is_backfill",
        )
        return {
            (row[0], row[1], row[2], row[3]): dict(zip(columns, row))
            for row in rows
        }
    except Exception as exception:
        raise RuntimeError(f"Could not read existing forecasts: {redact_exception(exception)}") from None


def _quantized(value, places: str) -> Decimal:
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("Forecast numeric values must be finite")
    return number.quantize(Decimal(places), rounding=ROUND_HALF_UP)


def _json_key(value) -> str:
    if isinstance(value, str):
        value = json.loads(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def _forecast_values_differ(stored, incoming) -> bool:
    for field, places in (
        ("origin_price", "0.01"),
        ("predicted_log_change", "0.0001"),
        ("predicted_price", "0.01"),
    ):
        if _quantized(stored[field], places) != _quantized(incoming[field], places):
            return True
    return _json_key(stored["shap_values"]) != _json_key(incoming["shap_values"])


def _classify_forecast(existing, row, is_backfill, plan):
    key = (row["market"], row["crop"], row["horizon"], row["origin_month"])
    stored = existing.get(key)
    if is_backfill:
        (plan["backfill_inserts"] if stored is None else plan["unchanged"]).append(row)
    elif stored is None:
        plan["live_inserts"].append(row)
    elif stored["is_backfill"]:
        plan["unchanged"].append(row)
        plan["warnings"].append(f"Preserved stored backfill row for {key}")
    elif _forecast_values_differ(stored, row):
        plan["live_updates"].append(row)
    else:
        plan["unchanged"].append(row)


def plan_forecast_writes(existing, live_rows, backfill_rows) -> dict:
    """Classify live rows and write-once backfills without mutating storage."""
    plan = {
        "live_inserts": [], "live_updates": [], "backfill_inserts": [],
        "unchanged": [], "warnings": [],
    }
    seen = set()
    for is_backfill, rows in ((False, live_rows), (True, backfill_rows)):
        for row in rows:
            key = (row["market"], row["crop"], row["horizon"], row["origin_month"])
            if key in seen:
                raise ValueError(f"Duplicate forecast row in plan: {key}")
            seen.add(key)
            _classify_forecast(existing, row, is_backfill, plan)
    return plan


def _parameters(row, jsonb_factory):
    return (
        row["market"], row["crop"], row["horizon"], row["origin_month"],
        row["target_month"], _quantized(row["origin_price"], "0.01"),
        _quantized(row["predicted_log_change"], "0.0001"),
        _quantized(row["predicted_price"], "0.01"), jsonb_factory(row["shap_values"]),
    )


def apply_forecast_writes(conn, plan):
    """Atomically upsert changed live rows and insert missing backfills only."""
    try:
        from psycopg.types.json import Jsonb

        with conn.transaction(), conn.cursor() as cursor:
            live_rows = [*plan["live_inserts"], *plan["live_updates"]]
            if live_rows:
                cursor.executemany(LIVE_UPSERT_SQL, [_parameters(row, Jsonb) for row in live_rows])
            if plan["backfill_inserts"]:
                cursor.executemany(
                    BACKFILL_INSERT_SQL,
                    [_parameters(row, Jsonb) for row in plan["backfill_inserts"]],
                )
    except Exception as exception:
        raise RuntimeError(f"Could not write forecast rows: {redact_exception(exception)}") from None