"""Worker-count resolution: the allocation, never the machine.

`os.cpu_count()` reports the node, so on a 4-core Slurm job carved out of a
32-core host it returns 32 and every parallel section asked for 31 workers.
`available_cores()` takes the minimum of the affinity mask, the scheduler's
allocation, the cgroup quota and the machine count.
"""

from __future__ import annotations

import os

from normet.utils import available_cores, default_workers
from normet.utils._config import resolve_workers


def test_reports_a_usable_count() -> None:
    assert available_cores() >= 1
    assert default_workers() >= 1
    assert available_cores() <= (os.cpu_count() or 1)


def test_never_exceeds_the_machine() -> None:
    # Whatever the environment claims, we never invent cores the box lacks.
    assert available_cores() <= (os.cpu_count() or 1)


def test_scheduler_allocation_caps_the_machine(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert available_cores() <= 2
    assert default_workers() == 1


def test_explicit_worker_count_wins(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert resolve_workers(5) == 5
    assert resolve_workers(1) == 1
    # A nonsense explicit value is clamped, not propagated.
    assert resolve_workers(0) == 1
    assert resolve_workers(None) == default_workers()
