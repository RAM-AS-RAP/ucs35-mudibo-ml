"""Synthetic calendar continuity gate tests."""

from datetime import date

import pytest

from agrisence.pipelines.series_gate import evaluate_series


def months_ending(latest, count=13):
    serial = latest.year * 12 + latest.month - 1
    return [date((serial - offset) // 12, (serial - offset) % 12 + 1, 1) for offset in range(count)]


def test_thirteen_current_months_are_eligible():
    today = date(2026, 10, 10)
    result = evaluate_series(months_ending(date(2026, 10, 1)), today)
    assert result["eligible"]
    assert result["consecutive_months"] == 13
    assert result["age_months"] == 0


def test_middle_gap_stops_backward_count():
    months = months_ending(date(2026, 10, 1))
    months.remove(date(2026, 5, 1))
    result = evaluate_series(months, date(2026, 10, 10))
    assert result["consecutive_months"] == 5
    assert not result["eligible"]
    assert "only 5 consecutive months" in result["reason"]


@pytest.mark.parametrize("latest,eligible,age", [(date(2026, 8, 1), True, 2), (date(2026, 7, 1), False, 3)])
def test_age_boundary(latest, eligible, age):
    result = evaluate_series(months_ending(latest), date(2026, 10, 10))
    assert result["eligible"] is eligible
    assert result["age_months"] == age


def test_empty_list_is_handled():
    result = evaluate_series([], date(2026, 10, 10))
    assert result["latest_month"] is None
    assert result["age_months"] is None
    assert result["consecutive_months"] == 0
    assert not result["eligible"]


def test_year_boundary_and_duplicate_months():
    months = months_ending(date(2025, 12, 1))
    result = evaluate_series(months + months, date(2026, 2, 10))
    assert result["age_months"] == 2
    assert result["consecutive_months"] == 13
    assert result["eligible"]