"""Fetch raw FEWS wholesale records; product approval is a separate step."""

from datetime import date
from urllib.parse import urljoin, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agrisence.config.pipeline_config import FEWS_BASE_URL


def _response_records(payload, market_id, product_query):
    if isinstance(payload, list):
        if len(payload) in {100, 250, 500, 1000}:
            print(f"WARNING: FEWS response may be truncated ({len(payload)} records), market={market_id}, crop={product_query}")
        batch, next_url = payload, None
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        batch, next_url = payload["results"], payload.get("next")
    else:
        raise ValueError("FEWS response must be a list or a mapping with a results list")
    if not all(isinstance(record, dict) for record in batch):
        raise ValueError("FEWS response records must be mappings")
    return batch, next_url


def _next_page(url, next_url):
    if not isinstance(next_url, str):
        raise ValueError("FEWS next page must be a URL string")
    resolved = urljoin(url, next_url)
    parsed = urlsplit(resolved)
    base = urlsplit(FEWS_BASE_URL)
    if parsed.scheme != "https" or parsed.netloc != base.netloc or parsed.username or parsed.password:
        raise ValueError("FEWS next page must remain on the FEWS HTTPS host without credentials")
    return resolved


def fetch_fews_records(
    market_id: str, product_query: str, start: date, end: date,
) -> list[dict]:
    """Fetch at most 20 pages with three total attempts for retryable statuses."""
    if product_query not in {"maize", "beans"}:
        raise ValueError("FEWS product_query must be maize or beans")
    if start > end:
        raise ValueError("FEWS start date must not be after end date")
    retry = Retry(
        total=2, connect=0, read=0, other=0, status=2, redirect=0,
        backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"GET"}), raise_on_status=False,
        respect_retry_after_header=False,
    )
    params = {
        "product": product_query, "country": "Kenya", "market": market_id,
        "price_type": "Wholesale", "start_date": start.isoformat(),
        "end_date": end.isoformat(), "format": "json",
    }
    records = []
    url = FEWS_BASE_URL
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; AgriSence monthly price ingestion)",
            "Accept": "application/json",
        })
        for page in range(20):
            response = session.get(url, params=params, timeout=(10, 60), allow_redirects=False)
            if not 200 <= response.status_code < 300:
                raise RuntimeError(
                    f"FEWS HTTP {response.status_code} for market={market_id}, "
                    f"crop={product_query}: {response.text[:300]}"
                )
            batch, next_url = _response_records(response.json(), market_id, product_query)
            records.extend(batch)
            if not next_url:
                return records
            if page == 19:
                raise ValueError("FEWS pagination exceeded 20 pages; refusing partial data")
            url = _next_page(url, next_url)
            params = None
    raise RuntimeError("FEWS pagination did not finish")