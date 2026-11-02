from __future__ import annotations

import os
import warnings
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

#: Default random seed used across modelling, resampling, and causal routines.
#: Centralised here so a single change propagates to every public default.
DEFAULT_SEED = 7654321


def _cgroup_cores() -> int | None:
    """Cores implied by the cgroup CPU quota, or None where no quota is set."""

    def ratio(quota: float, period: float) -> int | None:
        # NaN fails the first comparison, which is how "unquotaed" arrives here.
        if not (quota > 0 and period > 0):
            return None
        return max(1, int(quota // period))

    def read_num(path: Path) -> float:
        try:
            return float(path.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            return float("nan")

    try:
        fields = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        # "max <period>" means no quota is set; anything else is "<quota> <period>".
        if len(fields) == 2 and fields[0] != "max":
            out = ratio(float(fields[0]), float(fields[1]))
            if out is not None:
                return out
    except (OSError, ValueError):
        pass

    v1_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    v1_period = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if v1_quota.exists() and v1_period.exists():
        # A quota of -1 is v1's "unlimited"; ratio() rejects it along with NaN.
        out = ratio(read_num(v1_quota), read_num(v1_period))
        if out is not None:
            return out
    return None


def available_cores() -> int:
    """Number of CPU cores this process may actually run on.

    ``os.cpu_count()`` reports the *machine*, not the allocation: on a 4-core
    Slurm job carved out of a 32-core node it returns 32, so a pool sized from
    it asks for 31 workers on 4 cores. This takes the minimum over the CPU
    affinity mask, the scheduler's allocation, the cgroup CPU quota and the
    machine count, mirroring normet-r's ``.nm_available_cores()``.

    ``OMP_NUM_THREADS`` is deliberately not consulted: it caps threads *per
    process*, and is routinely set to 1 to stop BLAS oversubscribing
    underneath exactly the kind of worker pool this sizes.
    """
    limits: list[int] = []

    # Affinity mask: Slurm's cpuset binding, taskset, docker --cpuset-cpus.
    # Linux-only; absent on macOS/Windows, where the machine count stands alone.
    try:
        limits.append(len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

    # Schedulers that allocate without binding leave the mask wide open, so
    # their own environment variables are the only signal.
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "NSLOTS",
                "PBS_NUM_PPN", "NCPUS", "LSB_DJOB_NUMPROC"):
        try:
            val = int(os.environ.get(var, ""))
        except ValueError:
            continue
        if val >= 1:
            limits.append(val)

    quota = _cgroup_cores()
    if quota is not None:
        limits.append(quota)

    total = os.cpu_count()
    if total:
        limits.append(total)

    return max(1, min(limits)) if limits else 1


def default_workers() -> int:
    """Default worker-pool size: the whole allocation, minus one for the parent."""
    return max(1, available_cores() - 1)


def resolve_workers(n_cores: int | None) -> int:
    """Worker count for a parallel section: explicit value (>=1), else the default."""
    return default_workers() if n_cores is None else max(1, int(n_cores))


def resolve_config(cls: type[T], config: T | None = None, **kwargs: Any) -> T:
    """Build a dataclass config from an optional instance and keyword overrides.

    Unknown kwargs emit a UserWarning; None-valued kwargs do not override an
    existing config field (so callers can pass optional params without clobbering
    explicit config values).
    """
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(kwargs) - known
    if unknown:
        warnings.warn(
            f"Unknown config field(s): {sorted(unknown)}. Known fields: {sorted(known)}",
            UserWarning,
            stacklevel=3,
        )
    if config is None:
        return cls(**{k: v for k, v in kwargs.items() if k in known})
    overrides = {k: v for k, v in kwargs.items() if k in known and v is not None}
    return replace(config, **overrides) if overrides else config  # type: ignore[type-var]
