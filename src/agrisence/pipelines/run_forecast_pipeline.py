"""Generate, explain, plan, and optionally write live and backfill forecasts."""

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd

from agrisence.config import pipeline_config
from agrisence.inference.forecast_engine import build_origin_features, predict_and_explain, prepare_series_panel
from agrisence.inference.model_loader import ModelLoadError, load_all_models
from agrisence.pipelines.forecast_origins import plan_origins, target_month
from agrisence.storage.prices_repo import connect_prices, redact_exception
from agrisence.storage.forecasts_repo import (
    apply_forecast_writes,
    fetch_existing_forecasts,
    fetch_prices,
    plan_forecast_writes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "artifacts"
HORIZONS = ("one_month", "two_month")


def _today(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exception:
        raise argparse.ArgumentTypeError("--today must be an ISO date (YYYY-MM-DD)") from exception


def _existing_by_series(existing):
    result = {}
    for market, crop, horizon, origin_month in existing:
        result.setdefault((market, crop), {}).setdefault(origin_month, set()).add(horizon)
    return result


def _series_candidates(models, prices_df, market, crop, series_prices, origin_plan):
    candidates = {horizon: [] for horizon in HORIZONS}
    origins = [(origin_plan["live_origin"], False)] + [
        (origin, True) for origin in origin_plan["backfill_origins"]
    ]
    price_by_month = {
        pd.Timestamp(row.month).date(): float(row.avg_price_per_kg)
        for row in series_prices.itertuples(index=False)
    }
    for origin, is_backfill in origins:
        if origin not in price_by_month:
            raise ValueError(f"No stored origin price for {market}/{crop} at {origin}")
        panel = prepare_series_panel(prices_df, market, crop, origin)
        for horizon_key, model_horizon in (("h1", "one_month"), ("h2", "two_month")):
            features = build_origin_features(panel, origin, models[horizon_key]["feature_cols"])
            candidates[model_horizon].append({
                "market": market, "crop": crop, "origin_month": origin,
                "origin_price": price_by_month[origin], "is_backfill": is_backfill,
                "features": features,
            })
    return candidates


def _build_candidates(models, prices_df, existing, today):
    existing_by_series = _existing_by_series(existing)
    plans = []
    candidates = {horizon: [] for horizon in HORIZONS}
    skipped = 0
    for market in pipeline_config.ENABLED_MARKETS:
        for crop in pipeline_config.CROPS:
            series_prices = prices_df.loc[
                prices_df["market"].eq(market) & prices_df["crop"].astype(str).str.lower().eq(crop)
            ]
            months = [pd.Timestamp(value).date() for value in series_prices["month"]]
            origin_plan = plan_origins(months, existing_by_series.get((market, crop), {}), today)
            plans.append({"market": market, "crop": crop, **origin_plan})
            if not origin_plan["eligible"]:
                skipped += 1
                print(f"Skipped {market}/{crop}: {origin_plan['reason']}")
                continue
            series_rows = _series_candidates(models, prices_df, market, crop, series_prices, origin_plan)
            for horizon in HORIZONS:
                candidates[horizon].extend(series_rows[horizon])
    return plans, candidates, skipped


def _forecast_rows(models, candidates):
    live_rows, backfill_rows = [], []
    for horizon in HORIZONS:
        batch = candidates[horizon]
        if not batch:
            continue
        model_horizon = "h1" if horizon == "one_month" else "h2"
        X = pd.concat([item["features"] for item in batch], ignore_index=True)
        origin_prices = [item["origin_price"] for item in batch]
        predictions = predict_and_explain(models[model_horizon], X, origin_prices)
        for item, prediction in zip(batch, predictions, strict=True):
            row = {
                "market": item["market"], "crop": item["crop"], "horizon": horizon,
                "origin_month": item["origin_month"],
                "target_month": target_month(item["origin_month"], horizon),
                "origin_price": item["origin_price"],
                "predicted_log_change": prediction["predicted_log_change"],
                "predicted_price": prediction["predicted_price"],
                "shap_values": prediction["shap_values"],
                "is_backfill": item["is_backfill"],
            }
            (backfill_rows if item["is_backfill"] else live_rows).append(row)
    return live_rows, backfill_rows


def _report_table(plans, write_plan):
    actions = {}
    for category, action in (
        ("live_inserts", "insert"), ("live_updates", "update"),
        ("backfill_inserts", "insert"), ("unchanged", "unchanged"),
    ):
        for row in write_plan[category]:
            key = (row["market"], row["crop"], row["horizon"], row["origin_month"])
            actions[key] = action
    table = []
    for row in [*write_plan["live_inserts"], *write_plan["live_updates"], *write_plan["backfill_inserts"], *write_plan["unchanged"]]:
        table.append({
            **{key: row[key] for key in (
                "market", "crop", "origin_month", "horizon", "origin_price",
                "predicted_price", "predicted_log_change", "is_backfill",
            )},
            "change_percent": round((row["predicted_price"] / row["origin_price"] - 1) * 100, 2),
            "action": actions[(row["market"], row["crop"], row["horizon"], row["origin_month"])],
        })
    for plan in plans:
        if not plan["eligible"]:
            for horizon in HORIZONS:
                table.append({
                    "market": plan["market"], "crop": plan["crop"], "origin_month": None,
                    "horizon": horizon, "origin_price": None, "predicted_price": None,
                    "predicted_log_change": None, "change_percent": None,
                    "is_backfill": None, "action": "skipped", "reason": plan["reason"],
                })
    return sorted(table, key=lambda row: (row["market"], row["crop"], row["origin_month"] or date.min, row["horizon"]))


def _write_preview(table, report):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "forecasts_preview.json").write_text(
        json.dumps(report, default=str, indent=2) + "\n", encoding="utf-8",
    )
    fields = [
        "market", "crop", "origin_month", "horizon", "origin_price", "predicted_price",
        "predicted_log_change", "change_percent", "is_backfill", "action", "reason",
    ]
    with (ARTIFACT_DIR / "forecasts_preview.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)


def _execute(args, today, models):
    start = pipeline_config.START_MONTH
    existing = {}
    if args.no_db:
        prices_df = pd.read_csv(args.prices_csv, parse_dates=["month"])
    else:
        with connect_prices() as conn:
            prices_df = fetch_prices(conn, pipeline_config.ENABLED_MARKETS, pipeline_config.CROPS, start)
            existing = fetch_existing_forecasts(conn, pipeline_config.ENABLED_MARKETS, pipeline_config.CROPS)
    required = {"market", "crop", "month", "avg_price_per_kg"}
    missing = required.difference(prices_df.columns)
    if missing:
        raise ValueError(f"Price input is missing columns: {sorted(missing)}")
    prices_df["month"] = pd.to_datetime(prices_df["month"], errors="raise").dt.date
    prices_df = prices_df.loc[
        prices_df["market"].isin(pipeline_config.ENABLED_MARKETS)
        & prices_df["crop"].astype(str).str.lower().isin(pipeline_config.CROPS)
        & (prices_df["month"] >= start)
    ].copy()

    plans, candidates, skipped = _build_candidates(models, prices_df, existing, today)
    live_rows, backfill_rows = _forecast_rows(models, candidates)
    write_plan = plan_forecast_writes(existing, live_rows, backfill_rows)
    for warning in write_plan["warnings"]:
        print(f"WARNING: {warning}")
    table = _report_table(plans, write_plan)
    summary = {
        "series_eligible": len(plans) - skipped,
        "series_skipped": skipped,
        "rows_inserted": len(write_plan["live_inserts"]) + len(write_plan["backfill_inserts"]),
        "rows_updated": len(write_plan["live_updates"]),
        "rows_unchanged": len(write_plan["unchanged"]),
    }
    report = {
        "today": today, "summary": summary, "series": plans, "forecasts": table,
        "warnings": write_plan["warnings"],
    }
    if not args.no_db and not args.dry_run:
        with connect_prices() as conn:
            apply_forecast_writes(conn, write_plan)
    if args.no_db:
        _write_preview(table, report)
    print("market | crop | origin_month | horizon | origin_price | predicted_price | change_percent | is_backfill | action")
    for row in table:
        print(" | ".join(str(row.get(key, "")) for key in (
            "market", "crop", "origin_month", "horizon", "origin_price", "predicted_price",
            "change_percent", "is_backfill", "action",
        )))
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in summary.items()))
    if not args.dry_run:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "forecast_run_report.json").write_text(
            json.dumps(report, default=str, indent=2) + "\n", encoding="utf-8",
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--no-db", action="store_true", help="Use a CSV and write preview files only")
    mode.add_argument("--dry-run", action="store_true", help="Read database state and print the plan without writes")
    parser.add_argument("--prices-csv", type=Path)
    parser.add_argument("--today", type=_today)
    args = parser.parse_args(argv)
    if args.no_db and args.prices_csv is None:
        parser.error("--no-db requires --prices-csv PATH")
    if not args.no_db and args.prices_csv is not None:
        parser.error("--prices-csv is only valid with --no-db")
    today = args.today or date.today()
    try:
        models = load_all_models()
    except (ModelLoadError, OSError) as exception:
        print(f"Configuration/model error: {redact_exception(exception)}", file=sys.stderr)
        return 2
    try:
        _execute(args, today, models)
    except Exception as exception:
        message = redact_exception(exception)
        if "PIPELINE_DATABASE_URL is required" in message:
            print(f"Configuration error: {message}", file=sys.stderr)
            return 2
        print(f"Forecast pipeline failed: {type(exception).__name__}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
