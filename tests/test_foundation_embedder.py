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


def test_continuous_gradients_track_a_known_slope():
    ramp = np.arange(200, dtype="float64") * 0.5
    deriv = ChronosEmbedder.compute_continuous_gradients(ramp)
    np.testing.assert_allclose(deriv, 0.5, rtol=1e-3)


def test_embed_patches_and_patch_regime_shifts_torch_free():
    """Patch embeddings preserve time order and peak cosine distance at regime boundary."""
    n_patches, d_model = 12, 8
    patch_mat = np.zeros((1, n_patches, d_model), dtype=np.float32)
    patch_mat[0, :6, 0] = 1.0
    patch_mat[0, 6:, 1] = 1.0

    class _FakeEmbedPipe:
        def embed(self, windows, batch_size=32):
            return [patch_mat for _ in windows], None

    emb = ChronosEmbedder()
    emb._pipeline = _FakeEmbedPipe()

    idx = pd.date_range("2024-01-01", periods=192, freq="h")
    s = pd.Series(np.linspace(10.0, 20.0, 192), index=idx)

    patches = emb.embed_patches(s, context_length=192)
    assert patches.shape == (n_patches, d_model)
    assert patches.dtype == np.float32

    shifts = emb.patch_regime_shifts(s, context_length=192, window=2)
    assert list(shifts.columns) == ["date", "patch_idx", "position", "cosine_distance", "z_score"]
    peak = shifts.loc[shifts["cosine_distance"].idxmax()]
    assert int(peak["patch_idx"]) == 6
    assert float(peak["cosine_distance"]) == pytest.approx(1.0)
    assert float(peak["z_score"]) > 1.5

    # Array input has no date column; window<1 raises
    arr_shifts = emb.patch_regime_shifts(s.to_numpy(), context_length=192, window=2)
    assert list(arr_shifts.columns) == ["patch_idx", "position", "cosine_distance", "z_score"]
    with pytest.raises(ValueError, match="window"):
        emb.patch_regime_shifts(s, window=0)


def test_special_token_trimming_reg_pooling_and_multivariate_torch_free():
    """Trim trailing special tokens, support REG pooling, and encode multi-variates jointly."""

    class _FakeSpecialPipe:
        n_special_tokens = 2
        use_reg_token = True

        def embed(self, windows, batch_size=32):
            out = []
            for w in windows:
                arr = np.asarray(w, dtype=np.float32)
                n_v = 1 if arr.ndim == 1 else arr.shape[0]
                # 4 context patches filled with 1.0, [REG] with 7.0 + n_v, output patch with -5.0
                toks = np.zeros((n_v, 6, 8), dtype=np.float32)
                toks[:, :4, :] = 1.0
                toks[:, 4, :] = 7.0 + float(n_v)
                toks[:, 5, :] = -5.0
                out.append(toks)
            return out, None

    emb = ChronosEmbedder()
    emb._pipeline = _FakeSpecialPipe()
    s = np.linspace(1.0, 10.0, 64, dtype=np.float32)

    trimmed = emb.embed_patches(s, context_length=64)
    full = emb.embed_patches(s, context_length=64, include_special_tokens=True)
    assert trimmed.shape == (4, 8)
    assert full.shape == (6, 8)
    np.testing.assert_allclose(trimmed, 1.0)

    reg_vec = emb.embed_series(s, context_length=64, pooling="reg")
    mean_vec = emb.embed_series(s, context_length=64, pooling="mean")
    assert reg_vec.shape == (8,) and np.allclose(reg_vec, 8.0)
    assert not np.allclose(reg_vec, mean_vec)

    station_reg = emb.embed_stations({"A": s, "B": s * 2.0}, context_length=64, pooling="reg")
    assert set(station_reg) == {"A", "B"}
    np.testing.assert_allclose(station_reg["A"], 8.0)

    df_mv = pd.DataFrame({"pm25": s, "no2": s * 0.5, "ws": np.ones(64, dtype=np.float32)})
    mv_vec = emb.embed_multivariate(
        df_mv, cols=["pm25", "no2", "ws"], context_length=64, pooling="reg"
    )
    assert mv_vec.shape == (8,) and np.allclose(mv_vec, 10.0)

    with pytest.raises(ValueError, match="pooling must be"):
        emb.embed_series(s, context_length=64, pooling="cls")

    # Pipeline with no [REG] token refuses pooling="reg"
    class _NoRegPipe(_FakeSpecialPipe):
        n_special_tokens = 1
        use_reg_token = False

    emb._pipeline = _NoRegPipe()
    with pytest.raises(ValueError, match="use_reg_token"):
        emb.embed_series(s, context_length=64, pooling="reg")


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


@needs_chronos
def test_embed_patches_reg_pooling_and_multivariate_real(embedder):
    """Real Chronos-2 trims [REG]+output patches cleanly, exposes [REG] and multivariate."""
    s = _series(CONTEXT, seed=11)
    patches = embedder.embed_patches(s, context_length=CONTEXT)
    with_spec = embedder.embed_patches(s, context_length=CONTEXT, include_special_tokens=True)
    # 256 / 16 == 16 context patches; +2 for [REG] and masked output patch
    assert patches.shape == (16, EMBED_DIM)
    assert with_spec.shape == (18, EMBED_DIM)

    vec_mean = embedder.embed_series(s, context_length=CONTEXT, pooling="mean")
    vec_reg = embedder.embed_series(s, context_length=CONTEXT, pooling="reg")
    assert vec_reg.shape == (EMBED_DIM,) and np.isfinite(vec_reg).all()
    assert not np.allclose(vec_mean, vec_reg)

    df = pd.DataFrame(
        {
            "pm25": _series(CONTEXT, seed=12),
            "no2": _series(CONTEXT, seed=13),
            "ws": _series(CONTEXT, seed=14),
        }
    )
    mv = embedder.embed_multivariate(df, cols=["pm25", "no2", "ws"], context_length=CONTEXT)
    assert mv.shape == (EMBED_DIM,) and np.isfinite(mv).all()
