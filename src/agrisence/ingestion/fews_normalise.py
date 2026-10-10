"""Pure FEWS normalization, discovery, and diagnostic functions."""

from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


def record_month(value) -> date:
    """Parse the price coverage date and return its calendar month."""
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    else:
        parsed = date.fromisoformat(str(value)[:10])
    return parsed.replace(day=1)


def _consecutive(previous: date, current: date) -> bool:
    return current.year * 12 + current.month - (previous.year * 12 + previous.month) == 1


def _validated_record(record, allowed, crop, market, current_month, diagnostics):
    variety = (record.get("product"), record.get("cpcv2"))
    if variety not in allowed or record.get("price_type") != "Wholesale":
        diagnostics["filtered_records"] += 1
        return None
    context = f"market={market}, crop={crop}, product={variety[0]}, date={record.get('start_date')}"
    if record.get("currency") != "KES" or record.get("common_unit") != "kg":
        raise ValueError(f"FEWS requires currency=KES and common_unit=kg: {context}")
    try:
        month = record_month(record.get("start_date"))
    except (ValueError, TypeError) as exception:
        raise ValueError(f"Invalid FEWS start_date: {context}") from exception
    if not str(record.get("start_date")).startswith(month.isoformat()):
        diagnostics["warnings"].append(f"Floored non-first-day start_date to {month}: {context}")
    if month > current_month:
        diagnostics["future_records"] += 1
        return None
    try:
        price = Decimal(str(record.get("common_unit_price")))
    except (InvalidOperation, ValueError):
        diagnostics["invalid_prices"] += 1
        return None
    if not price.is_finite() or price <= 0:
        diagnostics["invalid_prices"] += 1
        return None
    return month, variety, price


def normalise_records(records, allowed, crop, market, today) -> tuple[list[dict], dict]:
    """Average records within varieties, then equally across varieties per month."""
    grouped = defaultdict(list)
    diagnostics = {
        "market": market, "crop": crop, "records_received": len(records),
        "filtered_records": 0, "invalid_prices": 0, "future_records": 0,
        "duplicate_variety_month_records": 0, "warnings": [], "months": [],
    }
    current_month = today.replace(day=1)
    for record in records:
        validated = _validated_record(record, allowed, crop, market, current_month, diagnostics)
        if validated is not None:
            month, variety, price = validated
            grouped[(month, variety)].append(price)

    by_month = defaultdict(list)
    for (month, variety), prices in grouped.items():
        diagnostics["duplicate_variety_month_records"] += max(0, len(prices) - 1)
        by_month[month].append((variety, sum(prices) / len(prices), len(prices)))
    rows = []
    for month, varieties in sorted(by_month.items()):
        price = (sum(item[1] for item in varieties) / len(varieties)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        if price <= 0:
            raise ValueError(f"Price rounds to a non-positive value: market={market}, crop={crop}, month={month}")
        rows.append({"market": market, "crop": crop, "month": month, "avg_price_per_kg": price})
        diagnostics["months"].append({
            "market": market, "crop": crop, "month": month,
            "record_count": sum(item[2] for item in varieties),
            "variety_count": len(varieties),
            "varieties": [
                {"product": variety[0], "cpcv2": variety[1]}
                for variety, _, _ in sorted(varieties)
            ],
        })
    return rows, diagnostics


def variety_warnings(monthly_diagnostics) -> list[str]:
    """Warn when the variety set changes between consecutive calendar months."""
    warnings = []
    series = defaultdict(list)
    for item in monthly_diagnostics:
        series[(item["market"], item["crop"])].append(item)
    for (market, crop), items in series.items():
        ordered = sorted(items, key=lambda item: item["month"])
        for previous, current in zip(ordered, ordered[1:]):
            previous_set = {(item["product"], item["cpcv2"]) for item in previous["varieties"]}
            current_set = {(item["product"], item["cpcv2"]) for item in current["varieties"]}
            if _consecutive(previous["month"], current["month"]) and previous_set != current_set:
                warnings.append(f"Variety change for {market}/{crop}: {previous['month']} to {current['month']}")
    return warnings


def move_warnings(rows, threshold) -> list[str]:
    """Warn about absolute log moves between consecutive calendar months."""
    warnings = []
    series = defaultdict(list)
    for row in rows:
        series[(row["market"], row["crop"])].append(row)
    for (market, crop), items in series.items():
        ordered = sorted(items, key=lambda item: item["month"])
        for previous, current in zip(ordered, ordered[1:]):
            log_change = (Decimal(str(current["avg_price_per_kg"])) / Decimal(str(previous["avg_price_per_kg"]))).ln()
            if _consecutive(previous["month"], current["month"]) and abs(log_change) > Decimal(str(threshold)):
                warnings.append(f"Large monthly log move for {market}/{crop} at {current['month']}: {log_change:.4f}")
    return warnings


def discover_products(records) -> list[dict]:
    """Summarize raw product groups without assuming units or approval."""
    fields = ("product", "cpcv2", "unit", "price_type", "currency", "product_source")
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record.get(field) for field in fields)].append(record_month(record.get("start_date")))
    return [
        {**dict(zip(fields, key)), "record_count": len(months), "first_month": min(months), "last_month": max(months)}
        for key, months in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0]))
    ]