"""Limited-role synchronous price reads and transactional FEWS upserts."""

from contextlib import contextmanager
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[3]
UPSERT_SQL = """
insert into market_prices (market, crop, month, avg_price_per_kg, source)
values (%s, %s::crop_type, %s, %s, 'fews'::price_source)
on conflict (market, crop, month) do update
  set avg_price_per_kg = excluded.avg_price_per_kg,
      source = excluded.source,
      fetched_at = now()
  where market_prices.avg_price_per_kg is distinct from excluded.avg_price_per_kg
     or market_prices.source is distinct from excluded.source
"""


def redact_exception(exception) -> str:
    """Strip connection URLs and passwords before any failure is reported."""
    message = str(exception)
    url = os.getenv("PIPELINE_DATABASE_URL", "")
    if url:
        message = message.replace(url, "[REDACTED DATABASE URL]")
        try:
            password = urlsplit(url).password
            if password:
                for secret in (password, unquote(password)):
                    message = message.replace(secret, "[REDACTED]")
        except ValueError:
            pass
    message = re.sub(r"(?i)postgres(?:ql)?://[^\s\"']+", "[REDACTED DATABASE URL]", message)
    return re.sub(r"(?i)(password\s*[=:]\s*)(?:'[^']*'|\"[^\"]*\"|[^\s,;]+)", r"\1[REDACTED]", message)


@contextmanager
def _safe_errors():
    try:
        yield
    except Exception as exception:
        raise RuntimeError(f"{type(exception).__name__}: {redact_exception(exception)}") from None


@contextmanager
def connect_prices():
    """Read only PIPELINE_DATABASE_URL and disable prepared statements."""
    with _safe_errors():
        from dotenv import load_dotenv
        import psycopg

        env_path = REPO_ROOT / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)
        url = os.getenv("PIPELINE_DATABASE_URL")
        if not url:
            raise ValueError("PIPELINE_DATABASE_URL is required for database modes")
        with psycopg.connect(url, prepare_threshold=None, autocommit=True) as conn:
            yield conn


def fetch_existing_prices(conn, markets, crops, start_month) -> dict[tuple[str, str, date], tuple[Decimal, str]]:
    """Read the scoped stored prices and sources, keyed by market/crop/month."""
    with _safe_errors(), conn.cursor() as cursor:
        cursor.execute(
            "select market, crop::text, month, avg_price_per_kg, source::text "
            "from market_prices where market = any(%s) and crop::text = any(%s) and month >= %s",
            (list(markets), list(crops), start_month),
        )
        return {(market, crop, month): (Decimal(str(price)), source) for market, crop, month, price, source in cursor.fetchall()}


def _rounded_price(value) -> Decimal:
    price = Decimal(str(value))
    if not price.is_finite() or price <= 0:
        raise ValueError("Prices must be finite and positive")
    rounded = price.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if rounded <= 0:
        raise ValueError("Prices must remain positive after rounding")
    return rounded


def plan_upserts(existing, incoming) -> tuple[list[dict], list[dict], list[dict]]:
    """Classify incoming FEWS observations without changing inputs or storage."""
    inserts, updates, unchanged = [], [], []
    seen = set()
    for row in incoming:
        key = (row["market"], row["crop"], row["month"])
        if key in seen:
            raise ValueError(f"Duplicate incoming price key: {key}")
        seen.add(key)
        if row["crop"] not in {"maize", "beans"} or row["month"].day != 1:
            raise ValueError(f"Invalid crop or non-month-start price key: {key}")
        normalized = {**row, "avg_price_per_kg": _rounded_price(row["avg_price_per_kg"]), "source": "fews"}
        stored = existing.get(key)
        if stored is None:
            inserts.append(normalized)
        elif _rounded_price(stored[0]) == normalized["avg_price_per_kg"] and stored[1] == "fews":
            unchanged.append(normalized)
        else:
            updates.append(normalized)
    return inserts, updates, unchanged


def apply_upserts(conn, inserts, updates):
    """Commit one transaction, or join the caller's transaction via a savepoint.

    Any write failure rolls back this batch. An enclosing transaction can also
    roll back the batch if its post-write gate read fails.
    """
    with _safe_errors(), conn.transaction(), conn.cursor() as cursor:
        parameters = [
            (row["market"], row["crop"], row["month"], _rounded_price(row["avg_price_per_kg"]))
            for row in [*inserts, *updates]
        ]
        if parameters:
            cursor.executemany(UPSERT_SQL, parameters)


def fetch_months_with_price(conn, market, crop) -> list[date]:
    """Read observed months on the same connection after an upsert."""
    with _safe_errors(), conn.cursor() as cursor:
        cursor.execute(
            "select month from market_prices where market = %s and crop = %s::crop_type "
            "and avg_price_per_kg > 0 order by month",
            (market, crop),
        )
        return [row[0] for row in cursor.fetchall()]