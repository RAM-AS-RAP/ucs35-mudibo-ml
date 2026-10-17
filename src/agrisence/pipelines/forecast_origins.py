"""Pure live/backfill origin planning shared by forecast orchestration."""

from datetime import date
from collections.abc import Mapping

from agrisence.pipelines.series_gate import evaluate_series


def _month(value: date) -> date:
    return value.replace(day=1)


def _previous_month(value: date) -> date:
    return date(value.year - 1, 12, 1) if value.month == 1 else value.replace(month=value.month - 1)


def consecutive_months_ending_at(months: list[date], origin: date) -> int:
    """Count uninterrupted observed calendar months backwards from origin."""
    available = {_month(value) for value in months}
    current = _month(origin)
    count = 0
    while current in available:
        count += 1
        if current.year == 1 and current.month == 1:
            break
        current = _previous_month(current)
    return count


def _existing_horizons(existing_origin_horizons, origin):
    if isinstance(existing_origin_horizons, Mapping):
        for key, horizons in existing_origin_horizons.items():
            if isinstance(key, tuple) and key and _month(key[0]) == origin:
                return set(horizons) if isinstance(horizons, (set, list, tuple)) else {key[1]}
            if isinstance(key, date) and _month(key) == origin:
                return set(horizons)
        return set()
    return {
        horizon for existing_origin, horizon in existing_origin_horizons
        if _month(existing_origin) == origin
    }


def plan_origins(months_with_price, existing_origin_horizons, today: date) -> dict:
    """Plan one live origin and eligible backfill origins without a one-month forecast."""
    months = sorted({_month(value) for value in months_with_price})
    gate = evaluate_series(months, today)
    if not gate["eligible"]:
        return {
            "eligible": False, "reason": gate["reason"],
            "live_origin": None, "backfill_origins": [],
        }
    latest = gate["latest_month"]
    backfills = []
    for origin in months:
        if origin >= latest:
            continue
        if consecutive_months_ending_at(months, origin) < 13:
            continue
        if "one_month" in _existing_horizons(existing_origin_horizons, origin):
            continue
        backfills.append(origin)
    return {
        "eligible": True, "reason": "", "live_origin": latest,
        "backfill_origins": backfills,
    }


def target_month(origin: date, horizon: str) -> date:
    """Return the first day of the calendar month after an origin."""
    if horizon != "one_month":
        raise ValueError("horizon must be 'one_month'")
    serial = origin.year * 12 + origin.month
    return date(serial // 12, serial % 12 + 1, 1)