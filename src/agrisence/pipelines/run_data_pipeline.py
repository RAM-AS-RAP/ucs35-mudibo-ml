"""FEWS monthly data-stage CLI; no features, forecasting, or model loading."""

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import sys

from agrisence.ingestion.fews_normalise import move_warnings, normalise_records, variety_warnings
from agrisence.storage.prices_repo import redact_exception

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts"


def _start_month(value: str) -> date:
    try:
        month = date.fromisoformat(value)
    except ValueError as exception:
        raise argparse.ArgumentTypeError("--start-month must be an ISO date") from exception
    if month.day != 1:
        raise argparse.ArgumentTypeError("--start-month must be the first day of a month")
    return month


def _collect(config, allowed, start, today):
    from agrisence.ingestion.fetch_fews import fetch_fews_records

    incoming, diagnostics, warnings = [], [], []
    fetched = 0
    for market in config.ENABLED_MARKETS:
        for crop in config.CROPS:
            records = fetch_fews_records(config.FEWS_MARKET_IDS[market], crop, start, today)
            fetched += len(records)
            rows, details = normalise_records(records, allowed[crop], crop, market, today)
            rows = [row for row in rows if start <= row["month"] <= today.replace(day=1)]
            if not rows:
                raise ValueError(f"No usable FEWS rows for enabled series {market}/{crop}")
            incoming.extend(rows)
            diagnostics.append(details)
            warnings.extend(details["warnings"])
            warnings.extend(variety_warnings(details["months"]))
            warnings.extend(move_warnings(rows, config.MAX_MONTHLY_MOVE))
    return incoming, diagnostics, warnings, fetched


def _gate_report(config, incoming, today, conn=None, projected=False):
    from agrisence.pipelines.series_gate import evaluate_series
    from agrisence.storage.prices_repo import fetch_months_with_price

    report = []
    for market in config.ENABLED_MARKETS:
        for crop in config.CROPS:
            incoming_months = [row["month"] for row in incoming if row["market"] == market and row["crop"] == crop]
            if conn is None:
                months = incoming_months
            else:
                months = fetch_months_with_price(conn, market, crop)
                if projected:
                    months = list(set(months) | set(incoming_months))
            report.append({"market": market, "crop": crop, **evaluate_series(months, today)})
    return report


def _database_stage(config, incoming, start, today, dry_run):
    from agrisence.storage.prices_repo import apply_upserts, connect_prices, fetch_existing_prices, plan_upserts

    with connect_prices() as conn, conn.transaction():
        existing = fetch_existing_prices(conn, config.ENABLED_MARKETS, config.CROPS, start)
        inserts, updates, unchanged = plan_upserts(existing, incoming)
        print(f"Planned: inserts={len(inserts)}, updates={len(updates)}, unchanged={len(unchanged)}")
        if not dry_run:
            apply_upserts(conn, inserts, updates)
        gates = _gate_report(config, incoming, today, conn, projected=dry_run)
    return inserts, updates, unchanged, gates


def _write_outputs(incoming, report, no_db):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if no_db:
        with (ARTIFACT_DIR / "fews_monthly_preview.csv").open("w", newline="", encoding="utf-8") as preview:
            writer = csv.DictWriter(preview, fieldnames=["market", "crop", "month", "avg_price_per_kg"])
            writer.writeheader()
            writer.writerows(incoming)
    (ARTIFACT_DIR / "run_report.json").write_text(
        json.dumps(report, default=str, indent=2) + "\n", encoding="utf-8",
    )


def _select_run_mode(config, incoming, args, today):
    if args.no_db:
        gates = _gate_report(config, incoming, today)
        return [], [], [], gates, "no-db"
    inserts, updates, unchanged, gates = _database_stage(
        config, incoming, args.start_month, today, args.dry_run,
    )
    return inserts, updates, unchanged, gates, "dry-run" if args.dry_run else "apply"


def _summarize_ignored_products(diagnostics):
    ignored_counts = {}
    for detail in diagnostics:
        series = f"{detail['market']}/{detail['crop']}"
        for item in detail["ignored_products"]:
            key = (series, item["product"], item["cpcv2"])
            ignored_counts[key] = ignored_counts.get(key, 0) + item["record_count"]
    return [
        {"series": series, "product": product, "cpcv2": cpcv2, "record_count": count}
        for (series, product, cpcv2), count in sorted(ignored_counts.items())
    ]


def _run(config, allowed, args, today):
    incoming, diagnostics, warnings, fetched = _collect(config, allowed, args.start_month, today)
    for warning in warnings:
        print(f"WARNING: {warning}")
    inserts, updates, unchanged, gates, mode = _select_run_mode(config, incoming, args, today)
    print("market | crop | latest_month | consecutive_months | age_months | eligible | reason")
    for gate in gates:
        print(" | ".join(str(gate[key]) for key in (
            "market", "crop", "latest_month", "consecutive_months", "age_months", "eligible", "reason",
        )))
    eligible = sum(gate["eligible"] for gate in gates)
    ignored_products = _summarize_ignored_products(diagnostics)
    if ignored_products:
        print("Ignored products (not allow-listed):")
        for item in ignored_products:
            print(
                f"  {item['series']}: {item['product']} ({item['cpcv2']}) "
                f"records={item['record_count']}"
            )
    else:
        print("Ignored products: none")
    summary = {
        "rows_fetched": fetched, "monthly_rows": len(incoming),
        "inserted": len(inserts) if mode == "apply" else 0,
        "updated": len(updates) if mode == "apply" else 0,
        "unchanged": len(unchanged), "series_eligible": eligible,
        "series_skipped": len(gates) - eligible, "ignored_products": ignored_products,
    }
    gate_basis = {"no-db": "incoming only", "dry-run": "projected", "apply": "stored after upsert"}[mode]
    report = {
        "mode": mode, "start_month": args.start_month, "today": today,
        **summary, "planned_inserts": len(inserts), "planned_updates": len(updates),
        "gate_basis": gate_basis, "series": gates, "warnings": warnings, "diagnostics": diagnostics,
    }
    _write_outputs(incoming, report, args.no_db)
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in summary.items()))


def main(argv=None) -> int:
    try:
        from agrisence.config import pipeline_config as config
    except (ValueError, ImportError) as exception:
        print(f"Configuration error: {redact_exception(exception)}", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--no-db", action="store_true", help="Fetch/normalize only; write a preview CSV")
    modes.add_argument("--dry-run", action="store_true", help="Read and plan without database changes")
    parser.add_argument("--start-month", type=_start_month, default=config.START_MONTH)
    args = parser.parse_args(argv)
    today = date.today()
    if args.start_month > today:
        parser.error("--start-month must not be in the future")
    try:
        allowed = config.load_allowed_products()
    except Exception as exception:
        print(f"Configuration error: {redact_exception(exception)}", file=sys.stderr)
        return 2
    try:
        _run(config, allowed, args, today)
    except Exception as exception:
        print(f"Pipeline failed: {type(exception).__name__}: {redact_exception(exception)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
