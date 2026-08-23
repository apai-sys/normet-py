"""Unit tests for the shared HTTP retry helper — no real network access."""

import pytest

from normet.io import _http


class _FakeHTTPError(Exception):
    """Mimics requests.HTTPError: carries a .response with a status_code."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.response = type("R", (), {"status_code": status})()


class _FakeResp:
    def __init__(self, status=200, json_data=None, text="", content=b"", headers=None):
        self.status_code = status
        self._json = json_data if json_data is not None else {}
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _FakeHTTPError(self.status_code)


class _FakeRequests:
    """Stand-in for the ``requests`` module returning queued responses."""

    def __init__(self, queue):
        self._queue = list(queue)
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def patched(monkeypatch):
    """Patch require()->fake requests and disable real sleeping."""

    def _install(queue):
        fake = _FakeRequests(queue)
        monkeypatch.setattr(_http, "require", lambda *a, **k: fake)
        monkeypatch.setattr(_http.time, "sleep", lambda _s: None)
        return fake

    return _install


def test_retries_on_429_then_succeeds(patched):
    fake = patched(
        [
            _FakeResp(429, headers={"Retry-After": "0"}),
            _FakeResp(200, json_data={"ok": 1}),
        ]
    )
    assert _http.get_json("http://x", source="T") == {"ok": 1}
    assert fake.calls == 2


def test_retries_on_500_then_succeeds(patched):
    fake = patched([_FakeResp(500), _FakeResp(200, json_data={"ok": 2})])
    assert _http.get_json("http://x", source="T") == {"ok": 2}
    assert fake.calls == 2


def test_client_error_4xx_does_not_retry(patched):
    fake = patched([_FakeResp(404), _FakeResp(200, json_data={"never": True})])
    with pytest.raises(RuntimeError):
        _http.get_json("http://x", source="T")
    assert fake.calls == 1  # 404 is deterministic — no retry


def test_exhausts_retries_and_raises(patched):
    fake = patched([_FakeResp(503), _FakeResp(503), _FakeResp(503)])
    with pytest.raises(RuntimeError):
        _http.request_with_retry("http://x", retries=3, source="T")
    assert fake.calls == 3


def test_retry_after_header_parsing():
    resp = _FakeResp(429, headers={"Retry-After": "1.5"})
    assert _http._retry_after(resp, backoff=1.0, attempt=2) == 1.5
    # Falls back to exponential backoff when header is absent/invalid.
    assert _http._retry_after(_FakeResp(429), backoff=1.0, attempt=2) == 4.0
    assert _http._retry_after(_FakeResp(429, headers={"Retry-After": "soon"}), 1.0, 1) == 2.0
