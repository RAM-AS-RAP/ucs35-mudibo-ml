"""Pure idempotent live/backfill forecast write classification."""

from datetime import date

from agrisence.storage.forecasts_repo import plan_forecast_writes


def forecast(**changes):
    return {
        "market": "Nairobi", "crop": "maize", "horizon": "one_month",
        "origin_month": date(2026, 9, 1), "target_month": date(2026, 10, 1),
        "origin_price": 100.0, "predicted_log_change": 0.05,
        "predicted_price": 105.13, "shap_values": {"groups": []},
        "is_backfill": False, **changes,
    }


def _key(row):
    return row["market"], row["crop"], row["horizon"], row["origin_month"]


def test_live_insert_update_and_unchanged_after_decimal_quantization():
    unchanged = forecast()
    update = forecast(market="Kisumu")
    insert = forecast(market="Mombasa")
    existing = {
        _key(unchanged): {**unchanged, "origin_price": 100.004, "predicted_log_change": 0.05001,
                          "predicted_price": 105.134, "shap_values": {"groups": []}},
        _key(update): {**update, "predicted_price": 107.0},
    }
    plan = plan_forecast_writes(existing, [unchanged, update, insert], [])
    assert plan["live_inserts"] == [insert]
    assert plan["live_updates"] == [update]
    assert plan["unchanged"] == [unchanged]


def test_existing_backfill_is_never_updated_and_live_over_backfill_is_preserved():
    old_backfill = forecast(is_backfill=True)
    recomputed_live = forecast(predicted_price=120.0)
    existing = {_key(old_backfill): old_backfill}
    live_plan = plan_forecast_writes(existing, [recomputed_live], [])
    assert not live_plan["live_updates"]
    assert live_plan["unchanged"] == [recomputed_live]
    assert len(live_plan["warnings"]) == 1
    backfill_plan = plan_forecast_writes(existing, [], [old_backfill])
    assert not backfill_plan["backfill_inserts"]
    assert backfill_plan["unchanged"] == [old_backfill]


def test_new_backfill_is_inserted_once():
    backfill = forecast(is_backfill=True)
    plan = plan_forecast_writes({}, [], [backfill])
    assert plan["backfill_inserts"] == [backfill]