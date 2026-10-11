"""Render the data and forecast reports as a GitHub Actions step summary."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA_REPORT = ROOT / "artifacts" / "run_report.json"
FORECAST_REPORT = ROOT / "artifacts" / "forecast_run_report.json"
NOT_REPORTED = "not reported"


def _read_report(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        return value
    except (OSError, json.JSONDecodeError):
        return None


def _cell(value) -> str:
    if value is None or value == "":
        return "-"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _valid_series_items(report: dict | None):
    if not isinstance(report, dict):
        return []
    rows = report.get("series", [])
    if not isinstance(rows, list):
        return []
    return [
        item for item in rows
        if isinstance(item, dict) and item.get("market") is not None and item.get("crop") is not None
    ]


def _merge_series_item(merged: dict, item: dict) -> None:
    market, crop = item["market"], item["crop"]
    row = merged.setdefault((market, crop), {"market": market, "crop": crop})
    fields = ("latest_month", "live_origin", "consecutive_months", "age_months", "eligible", "reason")
    for key in fields:
        if item.get(key) is not None:
            row[key] = item[key]


def _series_rows(data_report: dict | None, forecast_report: dict | None) -> list[dict]:
    merged = {}
    for report in (data_report, forecast_report):
        for item in _valid_series_items(report):
            _merge_series_item(merged, item)
    return [merged[key] for key in sorted(merged)]


def _series_status(row: dict) -> str:
    if row.get("eligible") is True:
        return "eligible"
    reason = row.get("reason")
    if reason:
        return f"skipped: {reason}"
    if row.get("eligible") is False:
        return "skipped"
    return NOT_REPORTED


def _counts(report: dict | None, stage: str) -> tuple[object, object, object]:
    if not report:
        return NOT_REPORTED, NOT_REPORTED, NOT_REPORTED
    if stage == "prices":
        source = report
        keys = ("inserted", "updated", "unchanged")
    else:
        summary = report.get("summary", {})
        source = summary if isinstance(summary, dict) else {}
        keys = ("rows_inserted", "rows_updated", "rows_unchanged")
    return tuple(source.get(key, NOT_REPORTED) for key in keys)


def _exit_code(report: dict | None, stage: str) -> str:
    if not report:
        return NOT_REPORTED
    stage_codes = report.get("exit_codes")
    if isinstance(stage_codes, dict) and stage in stage_codes:
        return str(stage_codes[stage])
    if "exit_code" in report:
        return str(report["exit_code"])
    return "not recorded in report"


def render_summary(
    data_report: dict | None,
    forecast_report: dict | None,
    event: str,
    mode: str,
    utc_date: str | None = None,
) -> str:
    """Render reports into safe Markdown; environment variable values are not read here."""
    today = utc_date or datetime.now(timezone.utc).date().isoformat()
    displayed_mode = mode or ("apply (scheduled)" if event == "schedule" else "not supplied")
    lines = [
        f"# Daily pipeline summary - {today} UTC",
        "",
        f"Workflow: {_cell(event or 'unknown')} | Mode: {_cell(displayed_mode)}",
        "",
    ]
    for report, path in ((data_report, DATA_REPORT), (forecast_report, FORECAST_REPORT)):
        if report is None:
            lines.append(f"Report missing or unreadable: `{path.relative_to(ROOT)}`")
    lines.extend([
        "",
        "## Series status",
        "",
        "| Market | Crop | Latest month | Consecutive months | Age (months) | Status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ])
    rows = _series_rows(data_report, forecast_report)
    for row in rows:
        latest = row.get("latest_month", row.get("live_origin"))
        lines.append("| " + " | ".join(_cell(value) for value in (
            row.get("market"), row.get("crop"), latest,
            row.get("consecutive_months"), row.get("age_months"), _series_status(row),
        )) + " |")
    if not rows:
        lines.append("| - | - | - | - | - | No series data reported |")
    lines.extend([
        "",
        "## Writes",
        "",
        "| Stage | Inserted | Updated | Unchanged | Exit code |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for stage, report, label in (
        ("prices", data_report, "Prices"),
        ("forecasts", forecast_report, "Forecasts"),
    ):
        inserted, updated, unchanged = _counts(report, stage)
        lines.append("| " + " | ".join(_cell(value) for value in (
            label, inserted, updated, unchanged, _exit_code(report, stage),
        )) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    data_report = _read_report(DATA_REPORT)
    forecast_report = _read_report(FORECAST_REPORT)
    markdown = render_summary(
        data_report,
        forecast_report,
        os.environ.get("EVENT", ""),
        os.environ.get("MODE", ""),
    )
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())