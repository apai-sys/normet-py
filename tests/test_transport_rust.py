"""Parity tests for the optional Rust Sirois-Bottenheim angle kernel.

``normet_rust`` is built from ``rust/normet-python`` (see ``AGENTS.md``) and
is opt-in: ``cluster_trajectories(..., metric="angle", backend="rust")``.
The contract is bit-identity -- switching backends must not move a distance,
a cluster label or a centroid -- so every test below compares exact floats.
The module skips itself when the extension is not installed, so the suite
still passes from a source checkout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.analysis.transport import _angle_distance_matrix, cluster_trajectories
from normet.core.exceptions import ConfigError
from normet.utils._rust import rust_available

pytest.importorskip("normet_rust")


def _corridor_endpoints(n_traj: int = 36):
    """North-east-west corridor trajectories, three regimes as in the suite."""
    rng = np.random.default_rng(42)
    lat0, lon0 = 51.5, 0.0
    times = pd.date_range("2023-01-01", periods=n_traj, freq="6h")
    rows = []
    series_rows = []
    for i, ts in enumerate(times):
        regime = i % 3
        series_rows.append(
            {"date": ts, "pm25": {0: 80.0, 1: 15.0, 2: 20.0}[regime] + float(rng.normal(0, 1.5))}
        )
        for age in (0, -3, -6, -9, -12):
            step = abs(age) / 3.0
            if regime == 0:
                la, lo, h = lat0 + 0.6 * step, lon0 + float(rng.normal(0, 0.05)), 250.0
            elif regime == 1:
                la, lo, h = lat0 + float(rng.normal(0, 0.05)), lon0 + 0.8 * step, 300.0
            else:
                la, lo, h = lat0 - 0.6 * step, lon0 + float(rng.normal(0, 0.05)), 300.0
            rows.append(
                {
                    "receptor_date": ts,
                    "start_lat": lat0,
                    "start_lon": lon0,
                    "start_height": 250.0,
                    "traj_id": 1,
                    "datetime": ts + pd.Timedelta(hours=age),
                    "age_h": float(age),
                    "lat": la,
                    "lon": lo,
                    "height": h,
                    "blh": 800.0,
                    "rainfall": 0.0,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(series_rows)


def _assert_matrices_equal(dx: np.ndarray, dy: np.ndarray) -> None:
    with np.errstate(invalid="ignore"):
        ref = _angle_distance_matrix(dx, dy)
        got = _angle_distance_matrix(dx, dy, backend="rust")
    assert got.shape == ref.shape
    # Bit-for-bit, so NaN payloads and the sign of zero are compared too.
    assert np.array_equal(got.view(np.uint64), ref.view(np.uint64))


def test_rust_available() -> None:
    assert rust_available() is True


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (4, 3),
        (25, 9),
        (60, 30),
        (150, 49),
        # Longer than numpy's 8192-element ufunc buffer, so the per-pair
        # mean crosses a reduction-chunk boundary. The seed makes the
        # single-buffer tree differ from numpy's chunked reduction here.
        (2, 9002),
    ],
)
def test_random_matrices_are_identical(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(sum(shape))
    _assert_matrices_equal(rng.normal(size=shape), rng.normal(size=shape))


def test_stagnant_nan_and_partial_steps_are_identical() -> None:
    """A pair with no valid step is exactly zero; NaN steps propagate per pair."""
    rng = np.random.default_rng(7)
    dx = rng.normal(size=(12, 8))
    dy = rng.normal(size=(12, 8))
    dx[3] = 0.0
    dy[3] = 0.0
    dx[5] = np.nan
    dy[5] = np.nan
    dx[7, 2] = np.nan
    dy[7, 2] = np.nan
    dx[9, 1] = np.inf  # inf/inf gives a NaN cosine, so a pair distance is NaN
    _assert_matrices_equal(dx, dy)


def test_exact_parallel_and_opposite_directions_are_identical() -> None:
    dx = np.array([[0.0, 1.0, 1.0], [0.0, -2.0, -2.0], [0.0, 0.0, 0.0]], dtype=float)
    dy = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ref = _angle_distance_matrix(dx, dy)
    got = _angle_distance_matrix(dx, dy, backend="rust")
    assert np.array_equal(got, ref)
    assert ref[0, 1] == 1.0  # antiparallel, exact after clipping
    assert ref[0, 2] == 0.5  # perpendicular
    assert ref[2, 0] == 0.5


def test_clustering_is_identical_end_to_end() -> None:
    """Labels, centroids, TSV and summary all derive from the distance matrix."""
    endpoints, _ = _corridor_endpoints(36)
    ref = cluster_trajectories(endpoints, metric="angle", n_clusters=3)
    got = cluster_trajectories(endpoints, metric="angle", n_clusters=3, backend="rust")
    assert got.n_clusters == ref.n_clusters
    assert np.array_equal(got.labels.to_numpy(), ref.labels.to_numpy())
    pd.testing.assert_frame_equal(got.centroids, ref.centroids, check_exact=True)
    pd.testing.assert_frame_equal(got.tsv, ref.tsv, check_exact=True)
    pd.testing.assert_frame_equal(got.summary, ref.summary, check_exact=True)


def test_backend_validation() -> None:
    endpoints, _ = _corridor_endpoints(6)
    with pytest.raises(ConfigError, match="must be 'python' or 'rust'"):
        cluster_trajectories(endpoints, metric="angle", backend="numba")
    with pytest.raises(ConfigError, match="only `metric='angle'`"):
        cluster_trajectories(endpoints, backend="rust")
