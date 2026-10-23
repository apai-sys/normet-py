# src/normet/foundation/embedding_diagnostics.py
"""Diagnostics for choosing how to use multi-site embeddings.

:func:`normet.embed_multisite` returns one 768-D vector per site and stops
there: ``cluster_multisite`` was deliberately removed because the package
should not make the grouping decision for the user. What the user needs to
make that decision is *diagnostics* -- how separated the sites are from one
another in the embedding space, and what the space looks like -- not labels.
This module supplies exactly that, and no more: it scores the silhouette as
a function of the number of groups and projects the vectors to two
dimensions for the eye, but it emits no cluster memberships and no
recommended k. Where a clustering is needed as a yardstick (silhouette), it
is computed and thrown away before the score, so nothing downstream can
read a membership off the result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .. import style as S
from ..style import styled
from ..utils._lazy import require
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["embedding_projection", "plot_embedding_diagnostics", "silhouette_curve"]


def _as_matrix(vectors: Any, *, name: str = "vectors") -> tuple[np.ndarray, list[str]]:
    """Accept the module's input forms and return (n x D) plus row labels.

    The natural input is the ``{site: vector}`` mapping
    :func:`normet.embed_multisite` returns; a bare 2-D array or DataFrame is
    also accepted, for vectors that come from elsewhere.
    """
    if isinstance(vectors, Mapping):
        if not vectors:
            raise ValueError(f"`{name}` needs at least two vectors; got an empty mapping.")
        keys = list(vectors.keys())
        mat = np.vstack([np.asarray(v, dtype=float) for v in vectors.values()])
        labels = [str(k) for k in keys]
    elif isinstance(vectors, (np.ndarray, pd.DataFrame)):
        mat = np.asarray(vectors, dtype=float)
        labels = [str(i) for i in range(mat.shape[0] if mat.ndim == 2 else 0)]
    else:
        raise TypeError(
            f"`{name}` must be a mapping of site to vector (as embed_multisite returns) "
            "or a 2-D array/DataFrame of vectors, "
            f"got {type(vectors).__name__}."
        )
    if mat.ndim != 2:
        raise ValueError(f"`{name}` must be 2-D (one row per site), got shape {mat.shape}.")
    n, d = mat.shape
    if n < 2:
        raise ValueError(f"`{name}` needs at least two vectors to diagnose; got {n}.")
    if not np.isfinite(mat).all():
        n_bad = int(np.isnan(mat).sum() + np.isinf(mat).sum())
        raise ValueError(f"`{name}` carries {n_bad} non-finite value(s); clean the vectors first.")
    return mat, labels


def silhouette_curve(
    vectors: Any,
    *,
    k_range: Sequence[int] = range(2, 12),
    metric: str = "euclidean",
    seed: int = 0,
) -> pd.DataFrame:
    """The mean silhouette as a function of the number of groups ``k``.

    The curve answers the question the grouping decision starts from -- how
    many separated groups of behaviour are there in this embedding space,
    roughly -- without answering it *for* you: this function deliberately
    emits no labels and no recommended k, because that recommendation is the
    user's to make (see the module docstring). KMeans is the yardstick's
    internal mechanism only: its assignments are discarded before the
    silhouette is scored, so the result carries no membership information.

    Parameters
    ----------
    vectors : mapping of site to vector, or (n, D) array
        The output of :func:`normet.embed_multisite`, or any stack of
        vectors.
    k_range : sequence of int, default range(2, 12)
        The group counts to score. A ``k`` that cannot be formed (``k < 2``
        or ``k >= n``, or one that collapses to a single cluster) gets a row
        with NaN rather than being skipped, so the curve is honest about
        what was not scored.
    metric : {"euclidean", "cosine", ...}, default "euclidean"
        The distance the silhouette is measured in. ``"cosine"`` is the
        usual choice when the vectors have been normalised; the default is
        sklearn's own default, for predictability.
    seed : int, default 0
        KMeans' ``random_state``; the curve is deterministic under it.

    Returns
    -------
    pandas.DataFrame
        One row per ``k`` in ``k_range`` (sorted), columns ``k`` and
        ``silhouette`` (the mean over the scored points, NaN where ``k`` is
        infeasible).

    Notes
    -----
    Needs scikit-learn (``pip install scikit-learn``), imported lazily so
    ``import normet`` is unaffected when it is absent. For more than 1000
    points the silhouette is scored on a deterministic 1000-point subsample
    (sklearn's ``sample_size``), which keeps the curve interactive without
    changing its reading.
    """
    mat, _ = _as_matrix(vectors)
    n = mat.shape[0]
    cluster = require("sklearn.cluster", hint="pip install scikit-learn")
    metrics = require("sklearn.metrics", hint="pip install scikit-learn")

    rows: list[dict[str, float | int]] = []
    for k in sorted({int(k) for k in k_range}):
        if k < 2 or k >= n:
            rows.append({"k": k, "silhouette": np.nan})
            continue
        labels = cluster.KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(mat)
        if len(np.unique(labels)) < 2:
            # A cluster collapsed (duplicate points): the silhouette is
            # undefined, and the NaN row says so.
            rows.append({"k": k, "silhouette": np.nan})
            continue
        sample_size = min(n, 1000)
        s = metrics.silhouette_score(
            mat,
            labels,
            sample_size=sample_size,
            random_state=seed,
            metric=metric,
        )
        rows.append({"k": k, "silhouette": float(s)})
    return pd.DataFrame(rows, columns=["k", "silhouette"])


def embedding_projection(
    vectors: Any,
    *,
    method: str = "pca",
    n_components: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Project the vectors onto ``n_components`` dimensions, for the eye.

    Parameters
    ----------
    vectors : mapping of site to vector, or (n, D) array
        As in :func:`silhouette_curve`.
    method : {"pca", "umap"}, default "pca"
        ``"pca"`` is pure numpy (centre, then truncated SVD) and adds no
        dependency. ``"umap"`` needs ``umap-learn`` (``pip install
        umap-learn``), imported lazily; it keeps local neighbourhoods better
        at the price of a global layout that is not comparable between runs
        beyond their shared seed.
    n_components : int, default 2
        The projection's dimension; 2 is what the diagnostic plot wants.
    seed : int, default 0
        ``random_state`` for UMAP (PCA is deterministic up to sign).

    Returns
    -------
    numpy.ndarray
        Array of shape (n, n_components), rows in the input's order.

    Notes
    -----
    PCA component *signs* are not stable across platforms or builds (the
    SVD is), so two runs may show mirrored layouts; the geometry --
    distances and neighbourhoods -- is the same. The projection is a view of
    the vectors, not a statistic to compare numerically.
    """
    mat, _ = _as_matrix(vectors)
    if method == "pca":
        centred = mat - mat.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        return centred @ vt[:n_components].T
    if method == "umap":
        umap_mod = require("umap.umap_", hint="pip install umap-learn")
        n_neigh = max(2, min(15, mat.shape[0] - 1))
        reducer = umap_mod.UMAP(n_components=n_components, n_neighbors=n_neigh, random_state=seed)
        return reducer.fit_transform(mat)
    raise ValueError(f"method must be 'pca' or 'umap', got {method!r}")


@styled
def plot_embedding_diagnostics(
    vectors: Any,
    *,
    method: str = "pca",
    k_range: Sequence[int] = range(2, 12),
    metric: str = "euclidean",
    seed: int = 0,
    groups: Sequence[str] | None = None,
    title: str | None = None,
) -> Any:
    """The two diagnostic panels: the projection, and the silhouette curve.

    The left panel is the vectors projected by :func:`embedding_projection`
    -- the sites' positions in the embedding space, at a glance. Coloured by
    ``groups`` when you supply them (your own grouping, drawn as you chose
    it), a single colour otherwise: the module itself does not assign any.
    With 20 or fewer sites each point is annotated with its site name.

    The right panel is :func:`silhouette_curve`: the mean silhouette against
    the number of groups. Its peak's position is a reading, not a verdict --
    the figure deliberately marks no k.

    Parameters
    ----------
    vectors : mapping of site to vector, or (n, D) array
        As in :func:`silhouette_curve`.
    method, k_range, metric, seed
        Forwarded to :func:`embedding_projection` and :func:`silhouette_curve`.
    groups : sequence of str, optional
        One label per vector, in the input's order. When given, the scatter
    is coloured by it; when absent, every point shares one colour.
    title : str, optional
        Figure title; defaults to "Site embedding diagnostics".

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the two panels (``fig.axes[0]`` the scatter,
        ``fig.axes[1]`` the curve).
    """
    import matplotlib.pyplot as plt

    mat, labels = _as_matrix(vectors)
    n = mat.shape[0]
    if groups is not None and len(groups) != n:
        raise ValueError(f"`groups` has {len(groups)} entries; the input has {n} vectors.")

    fig, (ax_scatter, ax_curve) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")

    pts = embedding_projection(vectors, method=method, seed=seed)
    if groups is None:
        ax_scatter.scatter(pts[:, 0], pts[:, 1], color=S.PALETTE[0], alpha=0.8, s=28)
    else:
        uniq = sorted(set(groups))
        for i, g in enumerate(uniq):
            m = np.asarray(groups) == g
            ax_scatter.scatter(
                pts[m, 0], pts[m, 1], color=S.PALETTE[i % len(S.PALETTE)], alpha=0.8, s=28, label=g
            )
        S.legend(ax_scatter)
    if n <= 20:
        for i, lab in enumerate(labels):
            ax_scatter.annotate(lab, (pts[i, 0], pts[i, 1]), fontsize=7, alpha=0.8)
    ax_scatter.set_xlabel("component 1")
    ax_scatter.set_ylabel("component 2")
    ax_scatter.set_title(f"{method.upper()} projection")
    S.grid(ax_scatter)

    curve = silhouette_curve(vectors, k_range=k_range, metric=metric, seed=seed)
    S.line(ax_curve, curve["k"], curve["silhouette"], marker="o", ms=4, lw=1.0)
    ax_curve.set_xlabel("k (number of groups)")
    ax_curve.set_ylabel("mean silhouette")
    ax_curve.set_title("silhouette by k")
    S.grid(ax_curve)

    fig.suptitle(title or "Site embedding diagnostics")
    return fig
