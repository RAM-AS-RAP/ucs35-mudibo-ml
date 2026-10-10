"""Pure upsert planning and mocked transactional write tests."""

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from agrisence.storage.prices_repo import apply_upserts, plan_upserts, redact_exception


def row(market="Nairobi", price="100", month=date(2026, 9, 1)):
    return {"market": market, "crop": "maize", "month": month, "avg_price_per_kg": price}


def test_insert_update_unchanged_and_source_change():
    month = date(2026, 9, 1)
    existing = {
        ("Nairobi", "maize", month): (Decimal("100.00"), "fews"),
        ("Kisumu", "maize", month): (Decimal("90.00"), "fews"),
        ("Mombasa", "maize", month): (Decimal("100.00"), "wfp"),
    }
    incoming = [row(price=100.000000001), row("Kisumu"), row("Mombasa"), row("Nakuru")]
    inserts, updates, unchanged = plan_upserts(existing, incoming)
    assert [item["market"] for item in inserts] == ["Nakuru"]
    assert [item["market"] for item in updates] == ["Kisumu", "Mombasa"]
    assert [item["market"] for item in unchanged] == ["Nairobi"]
    assert unchanged[0]["avg_price_per_kg"] == Decimal("100.00")
    assert incoming[0]["avg_price_per_kg"] == 100.000000001


def test_duplicate_keys_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        plan_upserts({}, [row(), row()])


def test_write_failure_exits_transaction_with_exception():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.executemany.side_effect = RuntimeError("simulated write failure")
    with pytest.raises(RuntimeError, match="simulated write failure"):
        apply_upserts(conn, [row()], [])
    conn.transaction.assert_called_once()
    assert conn.transaction.return_value.__exit__.call_args.args[0] is RuntimeError
    conn.commit.assert_not_called()


def test_database_secrets_are_redacted(monkeypatch):
    secret = uuid4().hex
    url = f"postgresql://pipeline_writer:{secret}@example.invalid/database"
    monkeypatch.setenv("PIPELINE_DATABASE_URL", url)
    message = redact_exception(RuntimeError(f"failure {url}, password={secret}"))
    assert url not in message
    assert secret not in message