"""Chronos-2 embeddings for environmental time-series.

Provides :class:`ChronosEmbedder`, a zero-shot latent representation extractor
for atmospheric observation networks — station clustering, transfer learning,
and node features for the graph models — plus noise-robust time derivatives for
PDE loss constraints.

Chronos-2 only
--------------
This module used to run on ``amazon/chronos-t5-base`` through
:class:`chronos.ChronosPipeline`. The two model families do not share an
``embed`` contract: Chronos-1 returns a single stacked
``(batch, context_length, d_model)`` tensor, whereas Chronos-2 is patch-based
and returns a *list* of ``(n_variates, n_patches, d_model)`` tensors, one per
input series. Embeddings from the two families are not comparable either — they
are different encoders — so a clustering built from a mix of both is
meaningless. Chronos-1/Bolt checkpoints are therefore rejected at construction
rather than silently producing a differently-shaped result, matching
:class:`~normet.foundation.estimator.Chronos2Estimator`.

``d_model`` is 768 for ``amazon/chronos-2``, the same width the T5-base encoder
produced, so downstream code that assumed 768-D vectors is unaffected.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .estimator import (
    DEFAULT_MODEL,
    InsufficientContextError,
    _load_pipeline_on,
    resolve_device,
)

log = logging.getLogger(__name__)

#: Encoder width of ``amazon/chronos-2``. Pinned by ``test_embedding_width``.
EMBED_DIM = 768


class ChronosEmbedder:
    """Zero-shot environmental context embeddings from Chronos-2.

    Extracts a 768-D temporal representation per station for unsupervised
    clustering, transfer learning and graph node features.

    Parameters
    ----------
    model_name : str, default ``"amazon/chronos-2"``
        Hugging Face identifier. ``chronos-t5-*`` / ``chronos-bolt-*``
        checkpoints are rejected — see the module docstring.
    device : str, optional
        ``"cuda"``/``"cpu"``. Auto-detected when omitted.
    min_coverage : float, default 0.1
        Minimum fraction of the context window that must carry observed values.
        Chronos-2 masks NaNs rather than failing, so an all-missing series
        otherwise yields a confident-looking embedding of nothing. The bar is
        lower than :class:`~normet.foundation.estimator.Chronos2Estimator`'s
        0.25 because an embedding only has to summarise the history it was
        given, while a forecast has to extrapolate from it.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        min_coverage: float = 0.1,
    ) -> None:
        if "chronos-t5" in model_name or "chronos-bolt" in model_name:
            raise ValueError(
                f"{model_name!r} is a Chronos-1/Bolt checkpoint. Its embed() returns a "
                "single stacked tensor rather than Chronos-2's per-series patch "
                "embeddings, and vectors from the two encoders are not comparable. "
                f"Use {DEFAULT_MODEL!r}."
            )
        self.model_name = model_name
        self.device = device
        self.min_coverage = float(min_coverage)
        self._pipeline: Any = None

    def _load_model(self) -> Any:
        """Lazy-load the Chronos-2 pipeline on first use."""
        if self._pipeline is not None:
            return self._pipeline

        auto = self.device is None
        self.device = resolve_device(self.device)
        log.info("Loading %s on %s", self.model_name, self.device)
        self._pipeline, self.device = _load_pipeline_on(self.model_name, self.device, auto)
        return self._pipeline

    # ------------------------------------------------------------- windowing

    def _window(self, series: pd.Series | np.ndarray, context_length: int) -> np.ndarray:
        """Cut or left-pad a series to exactly ``context_length`` points.

        Every series in a batch is normalised to the same length on purpose.
        Chronos-2 left-pads ragged batches itself, which makes the patch count
        — and therefore the mean over the patch axis — depend on which *other*
        series happened to share the batch. Fixing the length up front makes a
        station's embedding a property of that station alone.

        Short series are padded with NaN rather than the old edge-repeat: NaN is
        masked by the model, whereas a repeated first value is a fabricated flat
        stretch the encoder reads as real signal.
        """
        arr = np.asarray(series, dtype=np.float32).ravel()
        if arr.size == 0:
            raise ValueError("Input series is empty.")
        if arr.size >= context_length:
            return arr[-context_length:]
        pad = np.full(context_length - arr.size, np.nan, dtype=np.float32)
        return np.concatenate([pad, arr])

    def _check_coverage(self, window: np.ndarray, label: str) -> None:
        """Refuse to embed a context that is (almost) entirely missing."""
        coverage = float(np.isfinite(window).mean())
        if coverage < self.min_coverage:
            raise InsufficientContextError(
                f"{label}: only {coverage:.1%} of the {window.size}-point context is "
                f"observed (minimum {self.min_coverage:.0%}). Chronos-2 masks missing "
                "values, so this would return an embedding of an empty series."
            )

    @staticmethod
    def _pool(embedding: Any) -> np.ndarray:
        """Mean-pool one ``(n_variates, n_patches, d_model)`` tensor to ``(d_model,)``."""
        import torch

        pooled = embedding.mean(dim=(0, 1))
        return np.asarray(pooled.detach().cpu().to(torch.float32).numpy(), dtype=np.float32)

    # -------------------------------------------------------------- embedding

    def embed_series(
        self,
        series: pd.Series | np.ndarray,
        context_length: int = 2048,
    ) -> np.ndarray:
        """Extract a single context embedding vector for one time series.

        Parameters
        ----------
        series : pandas.Series or numpy.ndarray
            1D history of concentrations. NaNs are masked by the model, not
            interpolated.
        context_length : int, default 2048
            Most recent points to encode. 2048 is Chronos-2's native context.

        Returns
        -------
        numpy.ndarray
            1D array of shape ``(768,)``.
        """
        import torch

        pipeline = self._load_model()
        window = self._window(series, context_length)
        self._check_coverage(window, "series")

        with torch.no_grad():
            embeddings, _ = pipeline.embed([window])
        return self._pool(embeddings[0])

    def embed_stations(
        self,
        station_data: dict[str, pd.Series | np.ndarray] | pd.DataFrame,
        context_length: int = 2048,
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        """Extract context embeddings for many stations in one batched pass.

        Parameters
        ----------
        station_data : dict or pandas.DataFrame
            ``{station_id: series}``, or a frame indexed by time with one
            column per station.
        context_length : int, default 2048
            Context window length, applied identically to every station.
        batch_size : int, default 32
            Forwarded to Chronos-2, which does its own batching.

        Returns
        -------
        dict
            ``{station_id: 768-D array}``.
        """
        import torch

        pipeline = self._load_model()

        items: list[tuple[str, pd.Series | np.ndarray]]
        if isinstance(station_data, pd.DataFrame):
            items = [(str(c), station_data[c].to_numpy()) for c in station_data.columns]
        else:
            items = [(str(k), v) for k, v in station_data.items()]

        if not items:
            return {}

        stations = [st for st, _ in items]
        windows = []
        for st, series in items:
            window = self._window(series, context_length)
            self._check_coverage(window, f"station {st!r}")
            windows.append(window)

        log.info("Embedding %d stations (batch_size=%d)", len(stations), batch_size)
        with torch.no_grad():
            embeddings, _ = pipeline.embed(windows, batch_size=batch_size)

        return {st: self._pool(emb) for st, emb in zip(stations, embeddings, strict=True)}

    # ------------------------------------------------------------ derivatives

    @staticmethod
    def compute_continuous_gradients(
        series: pd.Series | np.ndarray,
        window_length: int = 15,
        polyorder: int = 3,
        smooth_method: str = "savgol",
    ) -> np.ndarray:
        """Compute continuous, noise-robust time derivatives (dC/dt) for PDE loss constraints.

        Args:
            series: Hourly concentration series.
            window_length: Kernel smoothing window size (odd integer).
            polyorder: Polynomial order for Savitzky-Golay filtering.
            smooth_method: 'savgol' or 'gaussian'.

        Returns:
            1D numpy array of continuous derivatives dC/dt (ug m^-3 hr^-1).
        """
        arr = np.asarray(series, dtype=np.float64)
        s_clean = pd.Series(arr).interpolate().bfill().ffill().to_numpy()

        if smooth_method == "savgol":
            from scipy.signal import savgol_filter

            w = window_length if window_length % 2 == 1 else window_length + 1
            if len(s_clean) <= w:
                return np.gradient(s_clean)
            deriv = savgol_filter(s_clean, window_length=w, polyorder=polyorder, deriv=1, delta=1.0)
            return np.asarray(deriv, dtype=np.float32)
        elif smooth_method == "gaussian":
            from scipy.ndimage import gaussian_filter1d

            smoothed = gaussian_filter1d(s_clean, sigma=2.0)
            return np.asarray(np.gradient(smoothed), dtype=np.float32)
        else:
            return np.asarray(np.gradient(s_clean), dtype=np.float32)

    # -------------------------------------------------------------- clustering

    @staticmethod
    def cluster_embeddings(
        embeddings: dict[str, np.ndarray] | np.ndarray,
        n_clusters: int = 4,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform unsupervised dimensionality reduction (UMAP) and KMeans clustering on embeddings.

        Args:
            embeddings: Dict of {station: embedding_768d} or matrix [N, 768].
            n_clusters: Number of latent environmental regimes (e.g. 4).
            random_state: Random seed for reproducibility.

        Returns:
            coords_2d: [N, 2] 2D projection coordinates.
            labels: [N] cluster assignment integer labels.
        """
        from sklearn.cluster import KMeans

        if isinstance(embeddings, dict):
            mat = np.array(list(embeddings.values()))
        else:
            mat = np.asarray(embeddings)

        n = len(mat)
        if n == 0:
            return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=int)

        coords_2d = _reduce_to_2d(mat, random_state)

        k = min(int(n_clusters), n)
        if k != n_clusters:
            log.warning(
                "n_clusters=%d exceeds the %d embeddings on hand; clustering into %d.",
                n_clusters,
                n,
                k,
            )
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(mat)

        return coords_2d, labels


def _reduce_to_2d(mat: np.ndarray, random_state: int) -> np.ndarray:
    """Project embeddings onto two dimensions, preferring UMAP but not needing it.

    UMAP's spectral initialisation asks scipy for ``n_components + 1`` eigenvectors
    of an N x N graph, and ``eigsh`` refuses outright once ``k >= N``; ``n_neighbors``
    likewise has to stay below N. A three-site frame is an ordinary thing to want
    clustered, so small N takes the PCA path instead of raising -- the previous
    code hard-coded ``n_neighbors=15`` and caught only ``ImportError``, so three
    sites crashed with a scipy ``TypeError`` wherever umap-learn happened to be
    installed.

    Any other UMAP failure also degrades to PCA rather than propagating: a 2-D
    projection is a diagnostic view, and no view is a worse outcome than a
    slightly plainer one.
    """
    n = len(mat)
    if n > 3:  # spectral init needs N > n_components + 1
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, n - 1),
                min_dist=0.1,
                random_state=random_state,
            )
            return np.asarray(reducer.fit_transform(mat))
        except ImportError:
            log.warning("umap-learn not found; falling back to 2D PCA.")
        except Exception:
            log.warning("UMAP failed on %d embeddings; falling back to 2D PCA.", n, exc_info=True)

    from sklearn.decomposition import PCA

    coords = np.asarray(
        PCA(n_components=min(2, n, mat.shape[1]), random_state=random_state).fit_transform(mat)
    )
    if coords.shape[1] < 2:  # a single embedding has no second component to report
        coords = np.hstack([coords, np.zeros((n, 2 - coords.shape[1]))])
    return coords
