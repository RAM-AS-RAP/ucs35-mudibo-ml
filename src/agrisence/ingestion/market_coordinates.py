"""Market coordinates placeholders for climate lookup and station mapping."""

from __future__ import annotations

from agrisence.config.constants import MARKETS

MARKET_COORDINATES: dict[str, tuple[float, float]] = {
    market: (0.0, 0.0) for market in MARKETS
}

# TODO: replace these placeholder lat/lon pairs with real coordinates for Nairobi, Nakuru,
# Mombasa, Kisumu, and Eldoret.
