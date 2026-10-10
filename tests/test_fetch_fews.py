"""Mock requests and its transport while exercising the configured Retry policy."""

from collections import deque
from datetime import date
from types import SimpleNamespace

import pytest

from agrisence.ingestion import fetch_fews


class MockSession:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.headers = {}
        self.calls = []
        self.attempts = 0
        self.adapter = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def mount(self, prefix, adapter):
        self.adapter = adapter

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        retry = self.adapter.max_retries
        while True:
            self.attempts += 1
            status, payload = self.responses.popleft()
            if retry.is_retry("GET", status, has_retry_after=True) and retry.total:
                retry = retry.increment(method="GET", response=SimpleNamespace(
                    status=status, get_redirect_location=lambda: None,
                ))
                continue
            return SimpleNamespace(status_code=status, text="response body", json=lambda: payload)


def fetch(monkeypatch, responses):
    session = MockSession(responses)
    monkeypatch.setattr(fetch_fews.requests, "Session", lambda: session)
    return session


def run_fetch():
    return fetch_fews.fetch_fews_records("58100", "maize", date(2025, 7, 1), date(2026, 10, 10))


def test_plain_list_and_request_parameters(monkeypatch):
    session = fetch(monkeypatch, [(200, [{"product": "maize"}])])
    assert run_fetch() == [{"product": "maize"}]
    kwargs = session.calls[0][1]
    assert kwargs["timeout"] == (10, 60)
    assert kwargs["params"]["country"] == "Kenya"
    assert kwargs["params"]["product"] == "maize"
    assert kwargs["params"]["price_type"] == "Wholesale"
    assert kwargs["params"]["start_date"] == "2025-07-01"
    assert session.headers["Accept"] == "application/json"


def test_results_and_next_page_are_followed(monkeypatch):
    session = fetch(monkeypatch, [
        (200, {"results": [{"page": 1}], "next": "?page=2"}),
        (200, {"results": [{"page": 2}], "next": None}),
    ])
    assert run_fetch() == [{"page": 1}, {"page": 2}]
    assert len(session.calls) == 2
    assert session.calls[1][1]["params"] is None


@pytest.mark.parametrize("status", [403, 400, 401, 404, 413])
def test_other_4xx_never_retry_even_with_retry_after(monkeypatch, status):
    session = fetch(monkeypatch, [(status, {})])
    with pytest.raises(RuntimeError, match=f"HTTP {status}.*response body"):
        run_fetch()
    assert session.attempts == 1
    assert not session.adapter.max_retries.is_retry("GET", status, has_retry_after=True)


def test_503_retries_and_succeeds(monkeypatch):
    session = fetch(monkeypatch, [(503, {}), (200, [])])
    assert run_fetch() == []
    assert session.attempts == 2
    assert session.adapter.max_retries.backoff_factor > 0


def test_retry_limit_is_three_total_attempts(monkeypatch):
    session = fetch(monkeypatch, [(503, {}), (503, {}), (503, {})])
    with pytest.raises(RuntimeError, match="HTTP 503"):
        run_fetch()
    assert session.attempts == 3


def test_100_record_list_warns(monkeypatch, capsys):
    fetch(monkeypatch, [(200, [{} for _ in range(100)])])
    assert len(run_fetch()) == 100
    assert "may be truncated" in capsys.readouterr().out


def test_twenty_page_limit_rejects_partial_data(monkeypatch):
    session = fetch(monkeypatch, [(200, {"results": [], "next": "?page=2"})] * 20)
    with pytest.raises(ValueError, match="20 pages"):
        run_fetch()
    assert len(session.calls) == 20


def test_external_next_url_is_rejected(monkeypatch):
    session = fetch(monkeypatch, [(200, {"results": [], "next": "https://example.com/"})])
    with pytest.raises(ValueError, match="FEWS HTTPS host"):
        run_fetch()
    assert len(session.calls) == 1