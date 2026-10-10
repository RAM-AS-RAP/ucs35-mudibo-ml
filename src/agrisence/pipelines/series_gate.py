"""Stable, pure continuity gate shared by data and future inference stages."""

from datetime import date

from agrisence.config.pipeline_config import GATE_MAX_AGE_MONTHS, GATE_MIN_CONSECUTIVE_MONTHS


def evaluate_series(months_with_price: list[date], today: date) -> dict:
    """Count calendar months backwards from latest without bridging gaps.

    The signature and output are the public contract for the inference stage.
    Empty series have no latest month or age and are never eligible.
    """
    months = {month.replace(day=1) for month in months_with_price}
    if not months:
        return {"latest_month": None, "consecutive_months": 0, "age_months": None, "eligible": False, "reason": "no months with prices"}
    latest = max(months)
    age = (today.year - latest.year) * 12 + today.month - latest.month
    cursor = latest
    consecutive = 0
    while cursor in months:
        consecutive += 1
        if cursor.year == 1 and cursor.month == 1:
            break
        cursor = date(cursor.year - 1, 12, 1) if cursor.month == 1 else cursor.replace(month=cursor.month - 1)
    reasons = []
    if consecutive < GATE_MIN_CONSECUTIVE_MONTHS:
        reasons.append(f"only {consecutive} consecutive months ending {latest:%Y-%m}")
    if age > GATE_MAX_AGE_MONTHS:
        reasons.append(f"latest month {latest:%Y-%m} is {age} months old")
    if age < 0:
        reasons.append(f"latest month {latest:%Y-%m} is in the future")
    return {
        "latest_month": latest, "consecutive_months": consecutive, "age_months": age,
        "eligible": not reasons, "reason": "; ".join(reasons),
    }