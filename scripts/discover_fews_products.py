"""Discover FEWS products to fill and review config/fews_products.yaml.

The missing-month table shows whether each series can pass the 13-month gate.
This CLI needs no database or secrets. It does not normalize or approve products.
"""

import argparse
from datetime import date
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agrisence.config.pipeline_config import CROPS, ENABLED_MARKETS, FEWS_MARKET_IDS, START_MONTH
from agrisence.ingestion.fetch_fews import fetch_fews_records
from agrisence.ingestion.fews_normalise import discover_products, record_month


def parse_start_month(value: str) -> date:
    month = date.fromisoformat(value)
    if month.day != 1:
        raise argparse.ArgumentTypeError("--start-month must be the first day of a month")
    return month


def calendar_months(start: date, end: date) -> list[date]:
    months = []
    current = start.replace(day=1)
    while current <= end.replace(day=1):
        months.append(current)
        current = date(current.year + 1, 1, 1) if current.month == 12 else current.replace(month=current.month + 1)
    return months


def _discover_series(market, crop, start, today):
    records = fetch_fews_records(FEWS_MARKET_IDS[market], crop, start, today)
    products = discover_products(records)
    print(f"\n{market}/{crop}: product | cpcv2 | unit | price_type | currency | product_source | count | first | last")
    for group in products:
        print(" | ".join(str(group[key]) for key in (
            "product", "cpcv2", "unit", "price_type", "currency", "product_source", "record_count", "first_month", "last_month",
        )))
    window = calendar_months(start, today)
    returned_months = {record_month(record.get("start_date")) for record in records}
    present = sorted(set(window) & returned_months)
    missing = [month for month in window if month not in returned_months]
    latest = max(present) if present else None
    age = (today.year - latest.year) * 12 + today.month - latest.month if latest else None
    print(f"Present months: {', '.join(month.isoformat() for month in present) or 'none'}")
    print(f"Missing months: {', '.join(month.isoformat() for month in missing) or 'none'}")
    print(f"Latest month: {latest}; age_months: {age}")
    return {
        "market": market, "crop": crop, "products": products,
        "present_months": present, "missing_months": missing,
        "latest_month": latest, "age_months": age,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-markets", action="store_true")
    parser.add_argument("--start-month", type=parse_start_month, default=START_MONTH)
    args = parser.parse_args()
    today = date.today()
    if args.start_month > today:
        parser.error("--start-month must not be in the future")
    markets = list(FEWS_MARKET_IDS) if args.all_markets else ENABLED_MARKETS
    report = {"start_month": args.start_month, "today": today, "series": []}
    try:
        for market in markets:
            for crop in CROPS:
                report["series"].append(_discover_series(market, crop, args.start_month, today))
    except Exception as exception:
        print(f"Discovery failed: {type(exception).__name__}: {exception}", file=sys.stderr)
        return 1
    output = REPO_ROOT / "artifacts" / "fews_discovery.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, default=str, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())