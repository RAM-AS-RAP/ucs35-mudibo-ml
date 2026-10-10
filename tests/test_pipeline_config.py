"""Allow-list configuration is deliberately fail-closed."""

from contextlib import contextmanager
from datetime import date
from decimal import Decimal
import json
from unittest.mock import MagicMock

import pytest

from agrisence.config.pipeline_config import load_allowed_products


@pytest.mark.parametrize("empty_crop", ["maize", "beans"])
def test_empty_allow_list_names_the_crop(tmp_path, empty_crop):
    path = tmp_path / "products.yaml"
    entries = {}
    for crop in ("maize", "beans"):
        entries[crop] = "[]" if crop == empty_crop else '[{product: "Reviewed", cpcv2: "Code"}]'
    path.write_text(
        f"maize:\n  allowed: {entries['maize']}\nbeans:\n  allowed: {entries['beans']}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=empty_crop):
        load_allowed_products(path)


def test_reviewed_allow_list_loads_pairs(tmp_path):
    path = tmp_path / "products.yaml"
    path.write_text(
        'maize:\n  allowed: [{product: "White", cpcv2: "M"}]\n'
        'beans:\n  allowed: [{product: "Mixed", cpcv2: "B"}]\n',
        encoding="utf-8",
    )
    assert load_allowed_products(path) == {"maize": {("White", "M")}, "beans": {("Mixed", "B")}}


def _configure_pipeline(monkeypatch, tmp_path):
    from agrisence.config import pipeline_config
    from agrisence.ingestion import fetch_fews
    from agrisence.pipelines import run_data_pipeline

    monkeypatch.setattr(pipeline_config, "load_allowed_products", lambda: {
        crop: {("Reviewed", "Code")} for crop in pipeline_config.CROPS
    })
    monkeypatch.setattr(run_data_pipeline, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(fetch_fews, "fetch_fews_records", lambda *args: [{
        "product": "Reviewed", "cpcv2": "Code", "price_type": "Wholesale",
        "currency": "KES", "common_unit": "kg", "common_unit_price": "100",
        "start_date": "2025-07-01",
    }])
    return run_data_pipeline, fetch_fews


def test_no_db_never_connects_and_writes_preview(monkeypatch, tmp_path):
    from agrisence.storage import prices_repo

    pipeline, _ = _configure_pipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(prices_repo, "connect_prices", MagicMock(side_effect=AssertionError("Unexpected database connection")))
    assert pipeline.main(["--no-db"]) == 0
    prices_repo.connect_prices.assert_not_called()
    assert (tmp_path / "artifacts" / "fews_monthly_preview.csv").exists()
    report = json.loads((tmp_path / "artifacts" / "run_report.json").read_text(encoding="utf-8"))
    assert report["rows_fetched"] == 4
    assert report["inserted"] == report["updated"] == 0


def test_failed_series_prevents_all_outputs(monkeypatch, tmp_path):
    pipeline, fetcher = _configure_pipeline(monkeypatch, tmp_path)
    original = fetcher.fetch_fews_records
    monkeypatch.setattr(fetcher, "fetch_fews_records", lambda market_id, *args: [] if market_id == "57999" else original(market_id, *args))
    assert pipeline.main(["--no-db"]) == 1
    assert not (tmp_path / "artifacts").exists()


def test_empty_configuration_fails_before_fetch(monkeypatch, tmp_path):
    from agrisence.config import pipeline_config

    pipeline, fetcher = _configure_pipeline(monkeypatch, tmp_path)
    def empty():
        raise ValueError("maize allow-list is empty")
    monkeypatch.setattr(pipeline_config, "load_allowed_products", empty)
    mock_fetch = MagicMock(side_effect=AssertionError("Unexpected fetch"))
    monkeypatch.setattr(fetcher, "fetch_fews_records", mock_fetch)
    assert pipeline.main(["--no-db"]) == 2
    mock_fetch.assert_not_called()


@pytest.mark.parametrize("dry_run", [True, False])
def test_database_stage_modes_and_transaction(monkeypatch, tmp_path, dry_run):
    from agrisence.config import pipeline_config
    from agrisence.storage import prices_repo

    pipeline, _ = _configure_pipeline(monkeypatch, tmp_path)
    conn = MagicMock()
    @contextmanager
    def connect():
        yield conn
    monkeypatch.setattr(prices_repo, "connect_prices", connect)
    monkeypatch.setattr(prices_repo, "fetch_existing_prices", lambda *args: {})
    monkeypatch.setattr(prices_repo, "fetch_months_with_price", lambda *args: [date(2025, 7, 1)])
    mock_apply = MagicMock()
    monkeypatch.setattr(prices_repo, "apply_upserts", mock_apply)
    incoming = [{"market": "Nairobi", "crop": "maize", "month": date(2025, 7, 1), "avg_price_per_kg": Decimal("100.00")}]
    pipeline._database_stage(pipeline_config, incoming, date(2025, 7, 1), date(2026, 10, 10), dry_run)
    assert mock_apply.call_count == (0 if dry_run else 1)
    conn.transaction.assert_called_once()


def test_gate_read_failure_rolls_back_enclosing_transaction(monkeypatch, tmp_path):
    from agrisence.config import pipeline_config
    from agrisence.storage import prices_repo

    pipeline, _ = _configure_pipeline(monkeypatch, tmp_path)
    conn = MagicMock()
    @contextmanager
    def connect():
        yield conn
    monkeypatch.setattr(prices_repo, "connect_prices", connect)
    monkeypatch.setattr(prices_repo, "fetch_existing_prices", lambda *args: {})
    monkeypatch.setattr(prices_repo, "apply_upserts", MagicMock())
    monkeypatch.setattr(prices_repo, "fetch_months_with_price", MagicMock(side_effect=RuntimeError("Gate read failed")))
    with pytest.raises(RuntimeError, match="Gate read failed"):
        pipeline._database_stage(pipeline_config, [], date(2025, 7, 1), date(2026, 10, 10), False)
    assert conn.transaction.return_value.__exit__.call_args.args[0] is RuntimeError