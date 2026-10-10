"""Configuration for the FEWS monthly data stage."""

from datetime import date
from pathlib import Path

START_MONTH = date(2025, 7, 1)
ENABLED_MARKETS = ["Nairobi", "Kisumu"]
CROPS = ["maize", "beans"]
FEWS_MARKET_IDS = {
    "Nairobi": "58100",
    "Kisumu": "57999",
    "Mombasa": "58073",
    "Nakuru": "254212",
}
# Mombasa's ID has only been checked for beans; verify maize with
# scripts/discover_fews_products.py --all-markets before enabling it.
GATE_MIN_CONSECUTIVE_MONTHS = 13
GATE_MAX_AGE_MONTHS = 2
MAX_MONTHLY_MOVE = 0.30
FEWS_BASE_URL = "https://fdw.fews.net/api/marketpricefacts/"
PRODUCTS_PATH = Path(__file__).with_name("fews_products.yaml")

if any(market not in FEWS_MARKET_IDS for market in ENABLED_MARKETS):
    raise ValueError("Every enabled market must have a FEWS market ID")
if any(crop not in {"maize", "beans"} for crop in CROPS):
    raise ValueError("Pipeline crops must be maize or beans")


def load_allowed_products(path: Path = PRODUCTS_PATH) -> dict[str, set[tuple[str, str]]]:
    """Load reviewed product/code pairs; an empty crop list blocks the run."""
    import yaml

    with Path(path).open(encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file)
    if not isinstance(document, dict):
        raise ValueError("Product configuration must contain crop mappings")
    allowed = {}
    for crop in CROPS:
        crop_config = document.get(crop)
        entries = crop_config.get("allowed") if isinstance(crop_config, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Allowed FEWS product list for {crop} is empty or invalid; run discovery and review it")
        pairs = set()
        for entry in entries:
            if not isinstance(entry, dict) or not all(
                isinstance(entry.get(key), str) and entry[key].strip()
                for key in ("product", "cpcv2")
            ):
                raise ValueError(f"Invalid FEWS product entry for {crop}: expected product and cpcv2 strings")
            pairs.add((entry["product"], entry["cpcv2"]))
        allowed[crop] = pairs
    return allowed