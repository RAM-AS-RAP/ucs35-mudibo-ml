"""Pure forecast-origin calendar and live/backfill planning tests."""

from datetime import date

import pytest

from agrisence.pipelines.forecast_origins import (
    consecutive_months_ending_at,
    plan_origins,
    target_month,
)


def _months(end, count):
    serial = end.year * 12 + end.month - 1
    return [date((serial - offset) // 12, (serial - offset) % 12 + 1, 1) for offset in reversed(range(count))]


def test_thirteen_months_have_live_only():
    months = _months(date(2026, 10, 1), 13)
    result = plan_origins(months, {}, date(2026, 10, 10))
    assert result["eligible"]
    assert result["live_origin"] == date(2026, 10, 1)
    assert result["backfill_origins"] == []


def test_fourteen_months_have_live_and_one_backfill():
    months = _months(date(2026, 10, 1), 14)
    result = plan_origins(months, {}, date(2026, 10, 10))
    assert result["live_origin"] == date(2026, 10, 1)
    assert result["backfill_origins"] == [date(2026, 9, 1)]


def test_twelve_months_are_ineligible():
    result = plan_origins(_months(date(2026, 10, 1), 12), {}, date(2026, 10, 10))
    assert not result["eligible"]
    assert result["live_origin"] is None
    assert result["backfill_origins"] == []


def test_gap_removes_pre_gap_backfill_origins():
    months = _months(date(2026, 10, 1), 30)
    months.remove(date(2025, 6, 1))
    result = plan_origins(months, {}, date(2026, 10, 10))
    assert result["eligible"]
    assert result["backfill_origins"] == [
        date(2025, 5, 1), date(2026, 7, 1), date(2026, 8, 1), date(2026, 9, 1),
    ]
    assert all(consecutive_months_ending_at(months, origin) >= 13 for origin in result["backfill_origins"])
    assert date(2025, 4, 1) not in result["backfill_origins"]


def test_age_three_skips_the_whole_series():
    months = _months(date(2026, 7, 1), 13)
    result = plan_origins(months, {}, date(2026, 10, 10))
    assert not result["eligible"]
    assert not result["backfill_origins"]
    assert "3 months old" in result["reason"]


def test_origin_with_both_horizons_is_not_backfilled():
    months = _months(date(2026, 10, 1), 14)
    existing = {date(2026, 9, 1): {"one_month", "two_month"}}
    result = plan_origins(months, existing, date(2026, 10, 10))
    assert result["backfill_origins"] == []


@pytest.mark.parametrize("horizon,expected", [
    ("one_month", date(2026, 1, 1)),
    ("two_month", date(2026, 2, 1)),
])
def test_target_month_crosses_year_boundary(horizon, expected):
    assert target_month(date(2025, 12, 1), horizon) == expected


def test_unknown_horizon_is_rejected():
    with pytest.raises(ValueError, match="horizon"):
        target_month(date(2025, 12, 1), "h3")