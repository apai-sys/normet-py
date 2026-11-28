"""The macOS guard against two OpenMP runtimes in one process.

PyTorch, LightGBM and scikit-learn each link their own libomp. LLVM's runtime
tolerates the second copy only under ``KMP_DUPLICATE_LIB_OK``, which it warns
"may cause crashes" -- and on this project's macOS environment it does: a
Chronos-2 forward pass after a LightGBM fit segfaults inside an OpenMP region,
which takes ``tests/test_foundation_embedder.py`` down when the suite reaches it
after a LightGBM test. ``normet.utils._omp`` caps torch's intra-op pool at one
thread in that situation; these tests pin the decision logic, since the crash
itself needs torch and a real second runtime to reproduce.
"""

from __future__ import annotations

import os
import sys

import pytest

from normet.utils import _omp


class _FakeTorch:
    """Records what the guard asked of torch, without importing it."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def set_num_threads(self, n: int) -> None:
        self.calls.append(int(n))


@pytest.fixture(autouse=True)
def _fresh_guard(monkeypatch):
    """Every test starts with the guard un-run, on macOS, with no override."""
    monkeypatch.setattr(_omp, "_CAPPED", False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv(_omp.THREADS_ENV, raising=False)


# ------------------------------------------------------------------ images


def test_openmp_runtime_images_lists_loaded_libomp_copies():
    """The dynamic loader's image list is what the guard consults."""
    if sys.platform != "darwin":
        pytest.skip("the image query is macOS-only")
    images = _omp.openmp_runtime_images()
    assert isinstance(images, list)
    for path in images:
        assert os.path.basename(path).startswith(("libomp", "lib_omp", "libiomp")), path


def test_openmp_runtime_images_is_empty_off_macos(monkeypatch):
    """Linux and Windows wheels ship no private libomp, so nothing is listed."""
    monkeypatch.setattr(sys, "platform", "linux")
    assert _omp.openmp_runtime_images() == []


def test_torch_runtime_is_recognised_by_its_path():
    assert _omp._is_torch_runtime("/x/python3.11/site-packages/torch/lib/libomp.dylib")
    assert not _omp._is_torch_runtime("/opt/homebrew/lib/libomp.dylib")
    assert not _omp._is_torch_runtime("/x/sklearn/.dylibs/libomp.dylib")


# ------------------------------------------------------------------- guard


def test_caps_torch_when_a_second_runtime_is_already_loaded(monkeypatch):
    """A LightGBM/sklearn runtime in the process is the crash condition."""
    foreign = "/x/sklearn/.dylibs/libomp.dylib"
    monkeypatch.setattr(
        _omp, "openmp_runtime_images", lambda: [foreign, "/y/torch/lib/libomp.dylib"]
    )
    torch = _FakeTorch()

    assert _omp._cap_torch_threads(torch, [foreign]) is True
    assert torch.calls == [1]


def test_caps_even_when_torch_was_imported_by_someone_else(monkeypatch):
    """A caller-loaded pipeline (the test fixture, a notebook) skips import_torch."""
    foreign = "/opt/homebrew/lib/libomp.dylib"
    monkeypatch.setattr(
        _omp, "openmp_runtime_images", lambda: [foreign, "/y/torch/lib/libomp.dylib"]
    )
    torch = _FakeTorch()

    # foreign is empty because the caller imported torch first; the post-import
    # image list still shows the other runtime.
    assert _omp._cap_torch_threads(torch, []) is True
    assert torch.calls == [1]


def test_leaves_torch_alone_with_a_single_runtime(monkeypatch):
    monkeypatch.setattr(_omp, "openmp_runtime_images", lambda: ["/y/torch/lib/libomp.dylib"])
    torch = _FakeTorch()

    assert _omp._cap_torch_threads(torch, []) is False
    assert torch.calls == []


def test_caps_only_once(monkeypatch):
    foreign = "/x/sklearn/.dylibs/libomp.dylib"
    monkeypatch.setattr(
        _omp, "openmp_runtime_images", lambda: [foreign, "/y/torch/lib/libomp.dylib"]
    )
    torch = _FakeTorch()

    assert _omp._cap_torch_threads(torch, [foreign]) is True
    assert _omp._cap_torch_threads(torch, [foreign]) is False
    assert torch.calls == [1]


def test_explicit_thread_count_wins(monkeypatch):
    monkeypatch.setenv(_omp.THREADS_ENV, "3")
    torch = _FakeTorch()

    assert _omp._cap_torch_threads(torch, ["/x/libomp.dylib"]) is True
    assert torch.calls == [3]


def test_zero_threads_turns_the_guard_off(monkeypatch):
    monkeypatch.setenv(_omp.THREADS_ENV, "0")
    torch = _FakeTorch()

    assert _omp._cap_torch_threads(torch, ["/x/libomp.dylib"]) is False
    assert torch.calls == []


def test_a_non_integer_override_is_warned_about_and_ignored(monkeypatch, caplog):
    monkeypatch.setenv(_omp.THREADS_ENV, "many")
    torch = _FakeTorch()

    with caplog.at_level("WARNING"):
        _omp._cap_torch_threads(torch, ["/x/libomp.dylib"])
    assert any(_omp.THREADS_ENV in record.getMessage() for record in caplog.records)


def test_off_macos_nothing_is_capped(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        _omp, "openmp_runtime_images", lambda: ["/x/libomp.dylib", "/y/lib/libomp.dylib"]
    )
    torch = _FakeTorch()

    assert _omp._cap_torch_threads(torch, ["/x/libomp.dylib"]) is False
    assert torch.calls == []


def test_import_torch_returns_the_module(monkeypatch):
    """``import_torch`` is a drop-in ``import torch``; the guard is a side effect."""
    pytest.importorskip("torch")
    monkeypatch.setattr(_omp, "_cap_torch_threads", lambda torch, foreign: False)

    torch = _omp.import_torch()
    assert getattr(torch, "__name__", "") == "torch"
