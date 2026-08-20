"""Tests for the Chronos-2 embedder.

Structural tests exercise the checkpoint guard, the context window and the
coverage rail, and need no model weights. Inference tests are gated on
``chronos-forecasting`` plus ``torch`` and share the session-loaded checkpoint
from ``conftest``.

The test that matters most is
``test_embedding_is_independent_of_batch_composition``. Chronos-2 left-pads a
ragged batch, so the number of patches an encoder emits — and therefore the mean
over the patch axis — depends on which *other* series happened to share the
batch. Measured on this checkpoint, the same 300-point series pooled to
21 patches alone and 34 patches next to a 512-point series, and the two mean
vectors differed by 0.30 in max absolute value. A station's embedding must be a
property of the station, not of its neighbours in the batch, or the clustering
it feeds is an artefact of iteration order.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from normet.foundation import ChronosEmbedder, InsufficientContextError
from normet.foundation.chronos import EMBED_DIM


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


needs_chronos = pytest.mark.skipif(
    not (_has("chronos") and _has("torch")),
    reason="chronos-forecasting/torch not installed",
)

CONTEXT = 256


@pytest.fixture
def embedder(chronos2_pipeline, chronos_device) -> ChronosEmbedder:
    """Embedder wired to the session-loaded pipeline."""
    emb = ChronosEmbedder(device=chronos_device)
    emb._pipeline = chronos2_pipeline
    return emb


def _series(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (20.0 + 10.0 * np.sin(2 * np.pi * t / 24.0) + rng.normal(0, 1.0, n)).astype("float32")


# --------------------------------------------------------------- structural


def test_rejects_chronos_one_checkpoints():
    """Chronos-1/Bolt embeddings have a different shape and are not comparable."""
    with pytest.raises(ValueError, match="Chronos-1/Bolt"):
        ChronosEmbedder(model_name="amazon/chronos-t5-base")
    with pytest.raises(ValueError, match="Chronos-1/Bolt"):
        ChronosEmbedder(model_name="amazon/chronos-bolt-small")


def test_window_truncates_to_the_most_recent_points():
    emb = ChronosEmbedder()
    arr = np.arange(1000, dtype="float32")
    window = emb._window(arr, 256)
    assert window.shape == (256,)
    assert window[-1] == 999.0
    assert window[0] == 744.0


def test_window_left_pads_short_series_with_nan():
    """Padding must be maskable, not a fabricated flat stretch.

    The pre-Chronos-2 implementation padded with ``np.pad(mode="edge")``, which
    repeats the first observation. Chronos-2 masks NaN but reads a repeated
    value as real signal, so edge-padding invents a calm period that never
    happened.
    """
    emb = ChronosEmbedder()
    window = emb._window(np.array([1.0, 2.0, 3.0], dtype="float32"), 10)
    assert window.shape == (10,)
    assert np.isnan(window[:7]).all()
    np.testing.assert_allclose(window[7:], [1.0, 2.0, 3.0])


def test_window_rejects_an_empty_series():
    emb = ChronosEmbedder()
    with pytest.raises(ValueError, match="empty"):
        emb._window(np.array([], dtype="float32"), 10)


def test_coverage_guard_rejects_a_mostly_missing_context():
    """Chronos-2 masks NaNs, so an empty context otherwise embeds silently."""
    emb = ChronosEmbedder(min_coverage=0.5)
    window = np.full(100, np.nan, dtype="float32")
    window[-10:] = 1.0
    with pytest.raises(InsufficientContextError, match="10.0% of the 100-point context"):
        emb._check_coverage(window, "series")


def test_coverage_guard_accepts_a_populated_context():
    emb = ChronosEmbedder(min_coverage=0.5)
    emb._check_coverage(_series(100), "series")


def test_cluster_embeddings_returns_coords_and_labels():
    rng = np.random.default_rng(0)
    mat = np.vstack([rng.normal(0, 1, (10, 8)), rng.normal(5, 1, (10, 8))])
    coords, labels = ChronosEmbedder.cluster_embeddings(mat, n_clusters=2)
    assert coords.shape == (20, 2)
    assert labels.shape == (20,)
    assert set(np.unique(labels)) == {0, 1}


def test_cluster_embeddings_survives_a_handful_of_sites():
    """Three sites must cluster, not crash.

    UMAP's spectral init asks scipy for ``n_components + 1`` eigenvectors of an
    N x N graph and ``eigsh`` refuses once ``k >= N``, so a three-row matrix used
    to raise a ``TypeError`` -- which the ``except ImportError`` around the UMAP
    import did not catch. It only ever surfaced where umap-learn was installed,
    which is why the CPU test environment never saw it.

    ``n_clusters`` above the number of embeddings is clamped for the same reason:
    asking for four regimes across three stations is a thing users do.
    """
    rng = np.random.default_rng(0)
    mat = rng.normal(0, 1, (3, 8))
    coords, labels = ChronosEmbedder.cluster_embeddings(mat, n_clusters=4)
    assert coords.shape == (3, 2)
    assert labels.shape == (3,)
    assert len(set(labels)) <= 3


def test_cluster_embeddings_handles_a_single_site():
    """One embedding has no second component; the caller still gets an (x, y)."""
    coords, labels = ChronosEmbedder.cluster_embeddings(np.ones((1, 8)), n_clusters=4)
    assert coords.shape == (1, 2)
    assert labels.shape == (1,)


def test_continuous_gradients_track_a_known_slope():
    ramp = np.arange(200, dtype="float64") * 0.5
    deriv = ChronosEmbedder.compute_continuous_gradients(ramp)
    np.testing.assert_allclose(deriv, 0.5, rtol=1e-3)


# ---------------------------------------------------------------- inference


@needs_chronos
def test_embedding_width(embedder):
    """Pins ``EMBED_DIM``; downstream graph node features assume it."""
    vec = embedder.embed_series(_series(CONTEXT), context_length=CONTEXT)
    assert vec.shape == (EMBED_DIM,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()


@needs_chronos
def test_embedding_is_independent_of_batch_composition(embedder):
    """A station's embedding must not shift when a longer station joins the batch.

    Regression test for Chronos-2's ragged-batch left-padding; see the module
    docstring.
    """
    short, long = _series(300, seed=1), _series(900, seed=2)

    solo = embedder.embed_stations({"A": short}, context_length=CONTEXT)
    mixed = embedder.embed_stations({"A": short, "B": long}, context_length=CONTEXT)

    np.testing.assert_allclose(solo["A"], mixed["A"], atol=1e-5)


@needs_chronos
def test_embed_stations_matches_embed_series(embedder):
    """The batched path and the single-series path must agree."""
    s = _series(CONTEXT, seed=3)
    np.testing.assert_allclose(
        embedder.embed_series(s, context_length=CONTEXT),
        embedder.embed_stations({"A": s}, context_length=CONTEXT)["A"],
        atol=1e-5,
    )


@needs_chronos
def test_embed_stations_accepts_a_frame(embedder):
    """A time-indexed frame and the equivalent dict give the same vectors."""
    idx = pd.date_range("2020-01-01", periods=CONTEXT, freq="h")
    frame = pd.DataFrame({"A": _series(CONTEXT, 4), "B": _series(CONTEXT, 5)}, index=idx)

    from_frame = embedder.embed_stations(frame, context_length=CONTEXT)
    from_dict = embedder.embed_stations(
        {c: frame[c].to_numpy() for c in frame.columns}, context_length=CONTEXT
    )

    assert set(from_frame) == {"A", "B"}
    for st in ("A", "B"):
        np.testing.assert_allclose(from_frame[st], from_dict[st], atol=1e-6)


@needs_chronos
def test_distinct_regimes_separate_in_embedding_space(embedder):
    """Two clearly different signals must not collapse onto the same vector."""
    diurnal = _series(CONTEXT, seed=6)
    flat = np.full(CONTEXT, 20.0, dtype="float32")

    out = embedder.embed_stations({"diurnal": diurnal, "flat": flat}, context_length=CONTEXT)
    a, b = out["diurnal"], out["flat"]
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cosine < 0.99


@needs_chronos
def test_empty_station_mapping_returns_empty_dict(embedder):
    assert embedder.embed_stations({}, context_length=CONTEXT) == {}


@needs_chronos
def test_all_missing_station_is_rejected_not_embedded(embedder):
    """The checkpoint returns a finite vector for an all-NaN series; we must not."""
    with pytest.raises(InsufficientContextError, match="station 'dead'"):
        embedder.embed_stations(
            {"dead": np.full(CONTEXT, np.nan, dtype="float32")}, context_length=CONTEXT
        )
