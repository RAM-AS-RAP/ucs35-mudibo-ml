"""CI summary formatting from representative pipeline reports."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ci_summary.py"
SPEC = importlib.util.spec_from_file_location("ci_summary", SCRIPT_PATH)
ci_summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci_summary)


@pytest.mark.ci
def test_summary_formats_pipeline_reports_and_missing_optional_fields():
    data_report = {
        "series": [{
            "market": "Nairobi", "crop": "maize", "latest_month": "2026-10-01",
            "consecutive_months": 16, "age_months": 0, "eligible": True, "reason": "",
        }],
        "inserted": 1, "updated": 2, "unchanged": 3,
    }
    forecast_report = {
        "summary": {"rows_inserted": 4, "rows_updated": 5, "rows_unchanged": 6},
        "series": [{
            "market": "Kisumu", "crop": "beans", "live_origin": "2026-10-01",
            "eligible": False, "reason": "only 4 consecutive months ending 2026-10",
        }],
    }

    markdown = ci_summary.render_summary(
        data_report, forecast_report, "workflow_dispatch", "dry-run", "2026-10-11",
    )

    assert "# Daily pipeline summary - 2026-10-11 UTC" in markdown
    assert "Workflow: workflow_dispatch | Mode: dry-run" in markdown
    assert "Nairobi | maize | 2026-10-01 | 16 | 0 | eligible" in markdown
    assert "Kisumu | beans | 2026-10-01 | - | - | skipped: only 4 consecutive months ending 2026-10" in markdown
    assert "Prices | 1 | 2 | 3 | not recorded in report" in markdown
    assert "Forecasts | 4 | 5 | 6 | not recorded in report" in markdown