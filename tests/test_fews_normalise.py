"""Synthetic FEWS normalization tests; no network or database."""

from datetime import date
from decimal import Decimal

import pytest

from agrisence.ingestion.fews_normalise import discover_products, move_warnings, normalise_records, variety_warnings

TODAY = date(2026, 10, 10)
ALLOWED = {("White maize", "white"), ("Yellow maize", "yellow")}


def record(**changes):
    return {
        "product": "White maize", "cpcv2": "white", "price_type": "Wholesale",
        "currency": "KES", "common_unit": "kg", "common_unit_price": "100",
        "start_date": "2026-09-01", "unit": "90 kg", "product_source": "Local",
        **changes,
    }


def normalize(records):
    return normalise_records(records, ALLOWED, "maize", "Nairobi", TODAY)


def test_allow_list_and_wholesale_filter():
    rows, diagnostics = normalize([record(), record(product="Unknown"), record(price_type="Retail")])
    assert len(rows) == 1
    assert rows[0]["avg_price_per_kg"] == Decimal("100.00")
    assert diagnostics["filtered_records"] == 2


def test_off_allow_list_products_are_reported_with_record_counts():
    _, diagnostics = normalize([
        record(product="Unreviewed", cpcv2="other"),
        record(product="Unreviewed", cpcv2="other"),
        record(price_type="Retail"),
    ])
    assert diagnostics["ignored_products"] == [
        {"product": "Unreviewed", "cpcv2": "other", "record_count": 2},
    ]


@pytest.mark.parametrize("field,value", [("currency", "USD"), ("common_unit", "bag")])
def test_invalid_currency_or_common_unit_raises(field, value):
    with pytest.raises(ValueError, match="Nairobi.*maize.*White maize.*2026-09-01"):
        normalize([record(**{field: value})])


def test_month_flooring_warns_and_future_month_drops():
    rows, diagnostics = normalize([record(start_date="2026-09-17"), record(start_date="2026-11-01")])
    assert [row["month"] for row in rows] == [date(2026, 9, 1)]
    assert diagnostics["warnings"]
    assert diagnostics["future_records"] == 1


@pytest.mark.parametrize("price", [None, 0, -1, "NaN", "Infinity"])
def test_null_nonpositive_and_nonfinite_prices_drop(price):
    rows, diagnostics = normalize([record(common_unit_price=price)])
    assert not rows
    assert diagnostics["invalid_prices"] == 1


def test_equal_variety_weight_after_within_variety_average():
    rows, diagnostics = normalize([
        record(common_unit_price="80"), record(common_unit_price="120"),
        record(product="Yellow maize", cpcv2="yellow", common_unit_price="200"),
    ])
    assert rows[0]["avg_price_per_kg"] == Decimal("150.00")
    assert diagnostics["duplicate_variety_month_records"] == 1
    month = diagnostics["months"][0]
    assert month["record_count"] == 3
    assert month["variety_count"] == 2
    assert {item["product"] for item in month["varieties"]} == {"White maize", "Yellow maize"}


def test_decimal_half_up_rounding():
    rows, _ = normalize([record(common_unit_price="10.005")])
    assert rows[0]["avg_price_per_kg"] == Decimal("10.01")


def test_variety_changes_warn_only_for_adjacent_months():
    _, diagnostics = normalize([
        record(start_date="2026-08-01"),
        record(start_date="2026-09-01", product="Yellow maize", cpcv2="yellow"),
    ])
    assert len(variety_warnings(diagnostics["months"])) == 1
    diagnostics["months"][0]["month"] = date(2026, 7, 1)
    assert not variety_warnings(diagnostics["months"])


@pytest.mark.parametrize("next_price,expected", [(135, True), (120, False)])
def test_log_move_warning(next_price, expected):
    rows = [
        {"market": "Nairobi", "crop": "maize", "month": date(2026, 8, 1), "avg_price_per_kg": Decimal(100)},
        {"market": "Nairobi", "crop": "maize", "month": date(2026, 9, 1), "avg_price_per_kg": Decimal(next_price)},
    ]
    assert bool(move_warnings(rows, 0.30)) is expected


def test_discovery_preserves_raw_group_fields_and_month_ranges():
    products = discover_products([record(start_date="2026-08-01"), record(), record(unit="kg")])
    assert len(products) == 2
    bag = next(item for item in products if item["unit"] == "90 kg")
    assert bag["record_count"] == 2
    assert bag["first_month"] == date(2026, 8, 1)
    assert bag["last_month"] == date(2026, 9, 1)