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

import contextlib
import logging
import sys
from collections.abc import Sequence
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


def _no_grad_ctx() -> Any:
    """Return ``torch.no_grad()`` if torch is loaded, else a no-op context."""
    torch = sys.modules.get("torch")
    if torch is not None:
        return torch.no_grad()
    return contextlib.nullcontext()


def _to_numpy_f32(tensor_or_array: Any) -> np.ndarray:
    """Convert a torch Tensor or array-like object into a float32 ndarray."""
    if isinstance(tensor_or_array, np.ndarray):
        return np.asarray(tensor_or_array, dtype=np.float32)
    if hasattr(tensor_or_array, "detach"):
        import torch

        return np.asarray(
            tensor_or_array.detach().cpu().to(torch.float32).numpy(), dtype=np.float32
        )
    return np.asarray(tensor_or_array, dtype=np.float32)


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
    def _validate_pooling(pooling: str) -> str:
        """Validate embedding pooling strategy (``'mean'`` or ``'reg'``)."""
        if pooling not in ("mean", "reg"):
            raise ValueError(f"pooling must be 'mean' or 'reg', got {pooling!r}")
        return pooling

    @staticmethod
    def _special_token_info(pipeline: Any) -> tuple[int, bool]:
        """Return ``(n_trailing_special_tokens, has_reg_token)`` emitted by ``embed``."""
        model = getattr(pipeline, "model", None)
        cfg = getattr(model, "chronos_config", None)
        if cfg is not None:
            use_reg = bool(getattr(cfg, "use_reg_token", True))
            return (2 if use_reg else 1, use_reg)
        n_spec = int(getattr(pipeline, "n_special_tokens", 0))
        use_reg = bool(getattr(pipeline, "use_reg_token", n_spec >= 2))
        return n_spec, use_reg

    @staticmethod
    def _pool(
        embedding: Any,
        *,
        pooling: str = "mean",
        use_reg_token: bool = True,
    ) -> np.ndarray:
        """Pool ``(n_variates, n_patches, d_model)`` into ``(d_model,)``."""
        mode = ChronosEmbedder._validate_pooling(pooling)
        arr = _to_numpy_f32(embedding)
        if mode == "mean":
            if arr.ndim == 3:
                return np.asarray(arr.mean(axis=(0, 1)), dtype=np.float32)
            if arr.ndim == 2:
                return np.asarray(arr.mean(axis=0), dtype=np.float32)
            return np.asarray(arr, dtype=np.float32)

        if not use_reg_token:
            raise ValueError(
                "pooling='reg' requires a Chronos-2 checkpoint with use_reg_token=True"
            )
        if arr.ndim == 3:
            if arr.shape[1] < 2:
                raise ValueError(
                    f"pooling='reg' requires >= 2 patch tokens ([REG] + output), got {arr.shape[1]}"
                )
            return np.asarray(arr[:, -2, :].mean(axis=0), dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] < 2:
                raise ValueError(
                    f"pooling='reg' requires >= 2 patch tokens ([REG] + output), got {arr.shape[0]}"
                )
            return np.asarray(arr[-2, :], dtype=np.float32)
        raise ValueError(
            f"pooling='reg' requires a 2D or 3D embedding tensor, got shape {arr.shape}"
        )

    # -------------------------------------------------------------- embedding

    def embed_patches(
        self,
        series: pd.Series | np.ndarray,
        context_length: int = 2048,
        *,
        include_special_tokens: bool = False,
    ) -> np.ndarray:
        """Extract unpooled patch embeddings ``(n_patches, d_model)`` for ``series``.

        Chronos-2 appends ``[REG]`` and a masked output patch token after the
        context patches. By default those trailing special tokens are stripped so
        the row count equals the true context patch count; pass
        ``include_special_tokens=True`` to retain them.
        """
        pipeline = self._load_model()
        window = self._window(series, context_length)
        self._check_coverage(window, "series")
        with _no_grad_ctx():
            embeddings, _ = pipeline.embed([window])
        arr = _to_numpy_f32(embeddings[0])
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        n_special, _ = self._special_token_info(pipeline)
        if not include_special_tokens and n_special > 0 and arr.shape[0] > n_special:
            arr = arr[:-n_special]
        return np.asarray(arr, dtype=np.float32)

    def patch_regime_shifts(
        self,
        series: pd.Series | np.ndarray,
        context_length: int = 2048,
        window: int = 2,
    ) -> pd.DataFrame:
        """Detect structural regime transitions in latent Chronos-2 patch trajectories."""
        win = int(window)
        if win < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        patches = self.embed_patches(series, context_length=context_length)
        n_patches = int(patches.shape[0])
        if n_patches < 2 * win:
            raise ValueError(f"need at least {2 * win} patches for window={win}, got {n_patches}")

        raw_len = len(series)
        effective_len = min(raw_len, int(context_length))
        tail_offset = max(0, raw_len - effective_len)
        has_dates = isinstance(series, pd.Series) and isinstance(series.index, pd.DatetimeIndex)

        rows: list[dict[str, Any]] = []
        dists: list[float] = []
        for i in range(win, n_patches - win + 1):
            before = patches[i - win : i].mean(axis=0)
            after = patches[i : i + win].mean(axis=0)
            denom = float(np.linalg.norm(before) * np.linalg.norm(after))
            if denom <= 0.0:
                cos_dist = 0.0
            else:
                cos_sim = float(np.dot(before, after) / denom)
                cos_dist = float(np.clip(1.0 - cos_sim, 0.0, 2.0))
            dists.append(cos_dist)
            pos = min(int(round(i * effective_len / n_patches)), max(effective_len - 1, 0))
            row: dict[str, Any] = {
                "patch_idx": int(i),
                "position": int(pos),
                "cosine_distance": cos_dist,
            }
            if has_dates:
                assert isinstance(series, pd.Series)
                row["date"] = series.index[tail_offset + pos]
            rows.append(row)

        d_arr = np.asarray(dists, dtype=float)
        med = float(np.median(d_arr))
        mad = float(np.median(np.abs(d_arr - med)))
        scale = max(1.4826 * mad, float(np.std(d_arr)), 1e-12)
        z_arr = (d_arr - med) / scale
        for row, z in zip(rows, z_arr, strict=True):
            row["z_score"] = float(z)

        cols = (["date"] if has_dates else []) + [
            "patch_idx",
            "position",
            "cosine_distance",
            "z_score",
        ]
        return pd.DataFrame(rows, columns=cols)

    def embed_series(
        self,
        series: pd.Series | np.ndarray,
        context_length: int = 2048,
        *,
        pooling: str = "mean",
    ) -> np.ndarray:
        """Extract a single context embedding vector for one time series.

        Parameters
        ----------
        series : pandas.Series or numpy.ndarray
            1D history of concentrations. NaNs are masked by the model, not
            interpolated.
        context_length : int, default 2048
            Most recent points to encode. 2048 is Chronos-2's native context.
        pooling : {"mean", "reg"}, default "mean"
            ``"mean"`` averages across patch tokens; ``"reg"`` extracts Chronos-2's
            ``[REG]`` summary token embedding.

        Returns
        -------
        numpy.ndarray
            1D array of shape ``(768,)``.
        """
        self._validate_pooling(pooling)
        pipeline = self._load_model()
        _, use_reg = self._special_token_info(pipeline)
        window = self._window(series, context_length)
        self._check_coverage(window, "series")

        with _no_grad_ctx():
            embeddings, _ = pipeline.embed([window])
        return self._pool(embeddings[0], pooling=pooling, use_reg_token=use_reg)

    def embed_multivariate(
        self,
        df: pd.DataFrame,
        cols: Sequence[str] | None = None,
        *,
        context_length: int = 2048,
        pooling: str = "mean",
    ) -> np.ndarray:
        """Extract a joint multi-variate Chronos-2 embedding from aligned columns.

        Unlike embedding columns separately, Chronos-2 attends across all
        stacked variates via group self-attention so co-varying pollutant and
        meteorology dynamics are encoded jointly.
        """
        self._validate_pooling(pooling)
        if cols is not None:
            selected = [str(c) for c in cols]
            missing = [c for c in selected if c not in df.columns]
            if missing:
                raise KeyError(f"columns absent from frame: {missing}")
        else:
            selected = [str(c) for c in df.select_dtypes(include="number").columns]
        if not selected:
            raise ValueError("embed_multivariate requires at least one numeric column")

        pipeline = self._load_model()
        _, use_reg = self._special_token_info(pipeline)
        windows = []
        for col in selected:
            window = self._window(df[col], context_length)
            self._check_coverage(window, f"column {col!r}")
            windows.append(window)
        stacked = np.stack(windows, axis=0).astype(np.float32, copy=False)

        with _no_grad_ctx():
            embeddings, _ = pipeline.embed([stacked])
        return self._pool(embeddings[0], pooling=pooling, use_reg_token=use_reg)

    def embed_stations(
        self,
        station_data: dict[str, pd.Series | np.ndarray] | pd.DataFrame,
        context_length: int = 2048,
        batch_size: int = 32,
        *,
        pooling: str = "mean",
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
        pooling : {"mean", "reg"}, default "mean"
            Patch pooling strategy (mean over tokens vs Chronos-2 ``[REG]`` token).

        Returns
        -------
        dict
            ``{station_id: 768-D array}``.
        """
        self._validate_pooling(pooling)
        pipeline = self._load_model()
        _, use_reg = self._special_token_info(pipeline)

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
        with _no_grad_ctx():
            embeddings, _ = pipeline.embed(windows, batch_size=batch_size)

        return {
            st: self._pool(emb, pooling=pooling, use_reg_token=use_reg)
            for st, emb in zip(stations, embeddings, strict=True)
        }

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
