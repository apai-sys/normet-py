# src/normet/io/_http.py
"""Shared HTTP helper for the data adapters.

Centralises timeout, retry, exponential backoff, and rate-limit (HTTP 429)
handling so each adapter (OpenAQ, EEA, DEFRA) does not reimplement it. ERA5 is
excluded — it goes through the ``cdsapi`` client, which has its own retry logic.
"""

from __future__ import annotations

import time
from typing import Any

from ..utils._lazy import require
from ..utils.logging import get_logger

log = get_logger(__name__)

# Transient server/infrastructure statuses worth retrying.
_RETRY_STATUS = frozenset({500, 502, 503, 504})


def _retry_after(resp: Any, backoff: float, attempt: int) -> float:
    """Seconds to wait on a 429: honour the ``Retry-After`` header, else backoff."""
    header = resp.headers.get("Retry-After") if getattr(resp, "headers", None) else None
    if header:
        try:
            return float(header)
        except (TypeError, ValueError):
            pass
    return backoff * (2**attempt)


def request_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.0,
    source: str = "HTTP",
) -> Any:
    """GET ``url`` and return the ``requests.Response`` on success.

    Retries on connection errors, timeouts, HTTP 429 (rate limit), and 5xx
    responses using exponential backoff (honouring ``Retry-After`` on 429).
    Client errors (4xx other than 429) fail immediately — retrying cannot fix a
    malformed request.

    Parameters
    ----------
    source : str
        Human-readable adapter name used in log and error messages.

    Raises
    ------
    RuntimeError
        If all attempts fail.
    """
    requests = require("requests", hint="pip install requests")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = _retry_after(resp, backoff, attempt)
                log.warning("%s rate-limited (429); sleeping %.1fs.", source, wait)
                last_err = RuntimeError("rate-limited (429)")
                time.sleep(wait)
                continue
            if resp.status_code in _RETRY_STATUS:
                wait = backoff * (2**attempt)
                log.warning(
                    "%s returned %d; retrying in %.1fs (attempt %d/%d).",
                    source,
                    resp.status_code,
                    wait,
                    attempt + 1,
                    retries,
                )
                last_err = RuntimeError(f"server error {resp.status_code}")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:  # connection errors, timeouts, 4xx, etc.
            last_err = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                break  # deterministic client error — do not retry
            if attempt < retries - 1:
                wait = backoff * (2**attempt)
                log.debug("%s request error: %s; retrying in %.1fs.", source, e, wait)
                time.sleep(wait)
    raise RuntimeError(f"{source} request to {url} failed after {retries} attempts: {last_err}")


def get_json(url: str, **kwargs: Any) -> Any:
    """GET ``url`` and return parsed JSON (see :func:`request_with_retry`)."""
    return request_with_retry(url, **kwargs).json()
