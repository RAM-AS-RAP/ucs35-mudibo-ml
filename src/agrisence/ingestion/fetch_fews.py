"""Fetch FEWS NET independently collected monthly wholesale price data.

The FEWS series is used to fill specific known-missing WFP months directly,
with no overlapping WFP observations available to calibrate against. It is an
independently observed measurement, not a statistical imputation.
"""

from __future__ import annotations

import requests
import pandas as pd


FEWS_MARKET_IDS = {
    "Kisumu": "57999",
    # Mombasa's ID must be re-verified against the maize product specifically.
    "Mombasa": "58073",
    "Nairobi": "58100",
    "Nakuru": "254212",
}

FEWS_MARKET_PRICE_FACTS_URL = "https://fdw.fews.net/api/marketpricefacts/"


def fetch_fews_prices(
    market_id: str, product: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch and normalize FEWS NET wholesale prices for one market window."""
    response = requests.get(
        FEWS_MARKET_PRICE_FACTS_URL,
        params={
            "product": product,
            "country": "Kenya",
            "market": market_id,
            "price_type": "Wholesale",
            "start_date": start_date,
            "end_date": end_date,
            "format": "json",
        },
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept": "application/json",
        },
    )
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(
            f"FEWS request failed for market={market_id}, product={product}: "
            f"{response.text}"
        )
        raise
    payload = response.json()
    records = payload.get("results", []) if isinstance(payload, dict) else payload
    if not records:
        return pd.DataFrame(columns=["month", "product", "cpcv2", "price_per_kg"])

    raw = pd.DataFrame(records)
    variants = raw[["product", "cpcv2"]].drop_duplicates()
    if len(variants) > 1:
        variant_names = variants.to_dict("records")
        print(
            f"WARNING: FEWS returned multiple product variants for market "
            f"{market_id}, product {product}: {variant_names}. Manual filtering "
            "is required before filling."
        )

    if not raw["currency"].eq("KES").all():
        raise AssertionError(
            f"FEWS currency must be KES for market {market_id}, product {product}"
        )
    if not raw["common_unit"].eq("kg").all():
        raise AssertionError(
            f"FEWS common_unit must be kg for market {market_id}, product {product}"
        )

    return pd.DataFrame(
        {
            "month": pd.to_datetime(raw["start_date"]),
            "product": raw["product"],
            "cpcv2": raw["cpcv2"],
            "price_per_kg": pd.to_numeric(raw["common_unit_price"]),
        }
    ).reset_index(drop=True)