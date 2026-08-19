import logging
import time

import pytest

from normet.utils.logging import _progress_str, enable_default_logging, get_logger


def test_get_logger_namespaced():
    logger = get_logger("foo.bar")
    assert logger.name == "normet.foo.bar"
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_get_logger_root():
    logger = get_logger()
    assert logger.name == "normet"


def test_progress_str_basic():
    t0 = time.time()
    result = _progress_str(5, 10, t0)
    assert "5/10" in result
    assert "ETA" in result


def test_progress_str_elapsed():
    """Progress after all steps — ETA should be 0 or very small."""
    t0 = time.time() - 2.0
    result = _progress_str(10, 10, t0)
    assert "10/10" in result


def test_enable_default_logging_idempotent():
    logger = get_logger()
    old_handlers = list(logger.handlers)
    try:
        # Start from a clean slate: strip any non-Null handlers another test may
        # have left on the shared "normet" logger, so the assertions below are
        # deterministic regardless of test order.
        logger.handlers[:] = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]
        # First call adds a handler
        enable_default_logging("INFO")
        non_null = [h for h in logger.handlers if not isinstance(h, logging.NullHandler)]
        assert len(non_null) == 1
        first_count = len(logger.handlers)

        # Second call must not add another handler
        enable_default_logging("DEBUG")
        assert len(logger.handlers) == first_count
    finally:
        logger.handlers[:] = old_handlers


def test_enable_default_logging_no_rich():
    logger = get_logger()
    # Remove all non-Null handlers to get a clean state
    logger.handlers[:] = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]
    try:
        enable_default_logging("WARNING", prefer_rich=False)
        plain = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(plain) >= 1
    finally:
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
