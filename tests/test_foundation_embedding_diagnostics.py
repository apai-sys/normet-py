"""Tests for the embedding diagnostics (silhouette curve, projection, plot).

These run without the foundation extra: the diagnostics operate on plain
arrays, and the heavy imports (sklearn, umap-learn) are lazy, so the tests
only need numpy and -- for the silhouette and the isolation-style paths --
scikit-learn, skipped when absent.
"""

import importlib.util

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

sklearn_available = importlib.util.find_spec("sklearn") is not None
umap_available = importlib.util.find_spec("umap") is not None

needs_sklearn = pytest.mark.skipif(not sklearn_available, reason="scikit-learn not installed")


def _two_blobs(n_per: int = 40, dim: int = 16, seed: int = 0) -> dict:
    """Two well-separated gaussian blobs, keyed by site name like
    embed_multisite returns."""
    rng = np.random.default_rng(seed)
    a = 4.0 * np.ones(dim)
    pts = np.vstack([rng.normal(0.0, 0.5, (n_per, dim)), rng.normal(0.0, 0.5, (n_per, dim)) + a])
    return {f"site_{i}": pts[i] for i in range(pts.shape[0])}


# ---------------------------------------------------------------------------
# silhouette_curve
# ---------------------------------------------------------------------------


@needs_sklearn
def test_silhouette_curve_peaks_at_the_true_k():
    from normet.foundation.embedding_diagnostics import silhouette_curve

    curve = silhouette_curve(_two_blobs(), k_range=range(2, 9), seed=0)
    assert list(curve.columns) == ["k", "silhouette"]
    assert list(curve["k"]) == [2, 3, 4, 5, 6, 7, 8]
    finite = curve[np.isfinite(curve["silhouette"])]
    # Two clean blobs: the best single cut is k=2, clearly.
    assert finite.loc[finite["silhouette"].idxmax(), "k"] == 2
    assert finite["silhouette"].max() > 0.6


@needs_sklearn
def test_silhouette_curve_is_deterministic_under_seed():
    from normet.foundation.embedding_diagnostics import silhouette_curve

    a = silhouette_curve(_two_blobs(), k_range=range(2, 6), seed=1)
    b = silhouette_curve(_two_blobs(), k_range=range(2, 6), seed=1)
    pd.testing.assert_frame_equal(a, b)


@needs_sklearn
def test_silhouette_curve_nan_rows_for_infeasible_k():
    from normet.foundation.embedding_diagnostics import silhouette_curve

    pts = _two_blobs(n_per=3)  # n = 6
    curve = silhouette_curve(pts, k_range=[2, 5, 6, 7], seed=0)
    assert np.isfinite(curve.loc[curve["k"] == 2, "silhouette"]).all()
    assert np.isfinite(curve.loc[curve["k"] == 5, "silhouette"]).all()
    # k == n and k > n cannot be formed: NaN rows, not dropped rows.
    assert np.isnan(curve.loc[curve["k"] == 6, "silhouette"]).all()
    assert np.isnan(curve.loc[curve["k"] == 7, "silhouette"]).all()


@needs_sklearn
def test_silhouette_curve_accepts_bare_arrays():
    from normet.foundation.embedding_diagnostics import silhouette_curve

    blobs = _two_blobs()
    as_array = np.vstack(list(blobs.values()))
    a = silhouette_curve(as_array, k_range=range(2, 5), seed=0)
    b = silhouette_curve(blobs, k_range=range(2, 5), seed=0)
    pd.testing.assert_frame_equal(a, b)


def test_silhouette_curve_input_validation():
    from normet.foundation.embedding_diagnostics import silhouette_curve

    with pytest.raises(ValueError, match="at least two"):
        silhouette_curve({"a": np.ones(4)})
    with pytest.raises(ValueError, match="empty mapping"):
        silhouette_curve({})
    with pytest.raises(ValueError, match="non-finite"):
        silhouette_curve({"a": np.ones(4), "b": np.array([1.0, np.nan, 1.0, 1.0])})
    with pytest.raises(TypeError, match="2-D array"):
        silhouette_curve([1.0, 2.0])


# ---------------------------------------------------------------------------
# embedding_projection
# ---------------------------------------------------------------------------


def test_embedding_projection_pca_shape_and_determinism():
    from normet.foundation.embedding_diagnostics import embedding_projection

    blobs = _two_blobs()
    p1 = embedding_projection(blobs)
    p2 = embedding_projection(blobs)
    assert p1.shape == (80, 2)
    assert np.allclose(p1, p2)


def test_embedding_projection_pca_preserves_centre():
    """A translation of the vectors translates the projection, and the first
    component lines up with the data's dominant direction."""
    from normet.foundation.embedding_diagnostics import embedding_projection

    blobs = _two_blobs()
    base = embedding_projection(blobs)
    shifted = {k: v + 100.0 * np.ones(len(v)) for k, v in blobs.items()}
    moved = embedding_projection(shifted)
    # Centre invariance: the centred projections are identical.
    assert np.allclose(base - base.mean(axis=0), moved - moved.mean(axis=0), atol=1e-8)
    # The two blobs separate on some component: the between-blob spread
    # exceeds the within-blob spread in the projection.
    half = len(base) // 2
    within = np.std(base[:half] - base[:half].mean(axis=0), axis=0).max()
    between = np.linalg.norm(base[:half].mean(axis=0) - base[half:].mean(axis=0))
    assert between > 3.0 * within


def test_embedding_projection_umap():
    from normet.foundation.embedding_diagnostics import embedding_projection

    if not umap_available:
        # The lazy import must fail with the installation hint, not a bare
        # ModuleNotFoundError from deep inside.
        with pytest.raises(ImportError, match="umap-learn"):
            embedding_projection(_two_blobs(), method="umap")
        return
    p = embedding_projection(_two_blobs(), method="umap", seed=0)
    assert p.shape == (80, 2)
    assert np.isfinite(p).all()


def test_embedding_projection_rejects_unknown_method():
    from normet.foundation.embedding_diagnostics import embedding_projection

    with pytest.raises(ValueError, match="'pca' or 'umap'"):
        embedding_projection(_two_blobs(), method="tsne")


# ---------------------------------------------------------------------------
# plot_embedding_diagnostics
# ---------------------------------------------------------------------------


def test_plot_embedding_diagnostics_smoke():
    import matplotlib.figure

    from normet.foundation.embedding_diagnostics import plot_embedding_diagnostics

    fig = plot_embedding_diagnostics(_two_blobs(n_per=5))
    assert isinstance(fig, matplotlib.figure.Figure)
    axes = fig.axes
    assert len(axes) >= 2
    # The style left-aligns its titles (matplotlib 3.10's titlelocation).
    assert axes[0].get_title("left") == "PCA projection"
    assert "silhouette by k" in axes[1].get_title("left")
    # Small input: the sites are annotated by name on the scatter.
    texts = {t.get_text() for t in axes[0].texts}
    assert "site_0" in texts and "site_9" in texts


@needs_sklearn
def test_plot_embedding_diagnostics_with_user_groups():
    from normet.foundation.embedding_diagnostics import plot_embedding_diagnostics

    blobs = _two_blobs(n_per=4)
    groups = ["A"] * 4 + ["B"] * 4
    fig = plot_embedding_diagnostics(blobs, groups=groups, title="my title")
    assert fig._suptitle.get_text() == "my title"
    # The user's grouping is what gets coloured: a legend with their labels.
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["A", "B"]


def test_plot_embedding_diagnostics_groups_length_mismatch():
    from normet.foundation.embedding_diagnostics import plot_embedding_diagnostics

    with pytest.raises(ValueError, match="groups"):
        plot_embedding_diagnostics(_two_blobs(n_per=3), groups=["A", "B"])
