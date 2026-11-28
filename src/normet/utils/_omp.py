# src/normet/utils/_omp.py
"""Keeping a second OpenMP runtime from crashing the process (macOS).

PyTorch, LightGBM and scikit-learn each link their own copy of LLVM's OpenMP
runtime. LLVM's libomp initialises only the first copy it finds and, when a
second one starts, aborts with ``OMP: Error #15`` -- *unless*
``KMP_DUPLICATE_LIB_OK=TRUE`` is set, which lets it continue and warns that
this "may cause crashes or silently produce incorrect results".

``normet`` has always set that variable before loading either runtime (see
``backends.lgb_backend._guard_macos_openmp`` and
``foundation.estimator._import_foundation``). On this project's macOS
development environment the warning is not academic: after a LightGBM fit, a
Chronos-2 forward pass segfaults inside an OpenMP parallel region, and the
whole test suite hangs or dies in ``tests/test_foundation_embedder.py``.
Measured here with three libomp copies in one process (torch, sklearn's and
Homebrew's), the failure is order-dependent and reproduces every time the
suite reaches the foundation tests after a LightGBM one.

Capping PyTorch's intra-op pool at one thread is enough: a one-thread "team"
does not cross the second runtime's barrier, and both libraries then run --
``normalise``/``do_all`` (LightGBM) followed by Chronos-2 (torch) included.
Re-importing torch is a no-op; ``normet`` therefore imports torch through
:func:`import_torch` rather than calling ``import torch`` itself, so the cap is
applied before the first parallel region wherever the first import happens.

The cap is one line of behaviour with a performance cost -- a one-thread
Chronos-2 forward pass is slower than a multi-threaded one -- so it is applied
only when a *second* OpenMP runtime is actually mapped into the process, and
can be overridden:

``NORMET_TORCH_THREADS``
    Intra-op threads for torch, as-is. ``0`` disables the cap.

On every platform other than macOS nothing is touched: the Linux and Windows
wheels do not ship private libomp copies.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .logging import get_logger

log = get_logger(__name__)

#: Environment variable that overrides the intra-op thread count, or turns the
#: cap off with ``0``.
THREADS_ENV = "NORMET_TORCH_THREADS"


def _on_macos() -> bool:
    """Whether this process is on macOS.

    Asked at call time rather than resolved once, so a caller (or a test that
    monkeypatches ``sys.platform``) sees the current answer. A plain
    ``sys.platform == "darwin"`` at each use site would do the same at runtime,
    but mypy narrows that comparison to the checking host and then calls the
    darwin-only bodies unreachable on a Linux runner; a ``bool`` return does
    not narrow.
    """
    return sys.platform == "darwin"


#: Set once torch has been capped. Left false while no second runtime is loaded,
#: so that a torch-first import followed by a LightGBM fit can still be capped by
#: the next ``import_torch`` call.
_CAPPED = False


def openmp_runtime_images() -> list[str]:
    """Paths of the OpenMP runtimes mapped into this process.

    Read from the dynamic loader's own image list, which is the only reliable
    answer on macOS and needs no third-party package (``psutil`` is not a
    dependency). Returns an empty list off macOS, where the question does not
    arise.
    """
    if not _on_macos():
        return []
    try:
        import ctypes

        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        images: list[str] = []
        for index in range(int(dyld._dyld_image_count())):
            raw = dyld._dyld_get_image_name(index)
            if not raw:
                continue
            path = raw.decode("utf-8", "replace")
            if os.path.basename(path).startswith(("libomp", "lib_omp", "libiomp")):
                images.append(path)
        return images
    except (OSError, AttributeError, ValueError):  # pragma: no cover - libc oddities
        return []


def _is_torch_runtime(path: str) -> bool:
    """Whether an OpenMP image is the one that shipped inside torch."""
    return "torch" in path.replace("\\", "/").split("/")


def _cap_torch_threads(torch: Any, foreign: list[str]) -> bool:
    """Cap torch's intra-op pool when a second OpenMP runtime is loaded.

    Returns whether anything was changed. ``foreign`` is the set of OpenMP
    images already mapped *before* torch was imported, which is what identifies
    the multi-runtime case even on a build that maps its own libomp lazily.
    """
    global _CAPPED
    if _CAPPED:
        return False
    if not _on_macos():
        return False

    requested = os.environ.get(THREADS_ENV)
    if requested is not None:
        try:
            threads = int(requested)
        except ValueError:
            log.warning("%s=%r is not an integer; ignored", THREADS_ENV, requested)
        else:
            if threads <= 0:
                return False
            torch.set_num_threads(threads)
            _CAPPED = True
            return True

    others = sorted(
        {
            *foreign,
            *(p for p in openmp_runtime_images() if not _is_torch_runtime(p)),
        }
    )
    if not others:
        # Only torch's own runtime, or none yet: nothing to collide with.
        return False
    torch.set_num_threads(1)
    _CAPPED = True
    log.info(
        "More than one OpenMP runtime is loaded on macOS: %s. PyTorch's "
        "intra-op pool is capped at one thread so the two runtimes do not "
        "collide; set %s to choose the count, or 0 to leave torch alone.",
        ", ".join(os.path.basename(p) for p in others),
        THREADS_ENV,
    )
    return True


def import_torch() -> Any:
    """Import torch, with the macOS multi-OpenMP guard applied.

    Every ``import torch`` inside ``normet`` goes through this function: the
    guard has to run before the first parallel region, and a plain
    ``import torch`` is a no-op that cannot be hooked after the fact. The cost
    is one ``import`` plus one dynamic-loader query on the first call, and
    nothing once the guard has run.
    """
    global _CAPPED
    foreign = (
        [] if _CAPPED else [path for path in openmp_runtime_images() if not _is_torch_runtime(path)]
    )
    import torch

    _cap_torch_threads(torch, foreign)
    return torch
