"""resolve_config: keyword overrides, None semantics and object ownership."""

from __future__ import annotations

from dataclasses import dataclass

from normet.utils._config import dispatch_batch_size, resolve_config


@dataclass
class _Cfg:
    a: int = 1
    b: str = "x"


def test_a_none_keyword_leaves_the_config_value_in_place():
    cfg = _Cfg(a=2)
    out = resolve_config(_Cfg, cfg, a=None, b="y")
    assert out == _Cfg(a=2, b="y")


def test_the_result_is_always_a_new_instance():
    """Derived values written onto the result must not reach the caller's config.

    ``normalise`` resolves auto-detected features onto the config, and
    ``do_all``/``decompose`` resolve a backend-dependent ``n_samples``; if
    ``resolve_config`` handed back the caller's own object, those writes
    stuck, and a config reused with another model or backend carried the
    first call's values.
    """
    cfg = _Cfg(a=2)
    out = resolve_config(_Cfg, cfg)
    assert out is not cfg
    out.a = 99
    assert cfg.a == 2


def test_dispatch_batch_size_spreads_tasks_evenly_over_workers():
    """One batch per worker, rounded up, never zero.

    The process backend pickles each task's closure once per dispatch, so the
    batch size is the count of frames that cross to the workers. Uneven
    division must not leave a worker idle on a remainder.
    """
    assert dispatch_batch_size(40, 4) == 10
    assert dispatch_batch_size(5, 2) == 3
    assert dispatch_batch_size(2, 8) == 1
    assert dispatch_batch_size(0, 4) == 1  # no tasks still needs a legal size
    assert dispatch_batch_size(4, 0) == 4  # a zero worker count must not divide by zero
