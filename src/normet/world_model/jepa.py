"""Action-Conditioned JEPA (AC-JEPA) for latent deweathered dynamics and SCM.

Implements a lightweight, deterministic-by-default Joint Embedding Predictive
Architecture tailored to atmospheric monitoring time series:

- Encodes contiguous trajectory patches into latent structural states (using
  either the built-in multi-scale spectral/lag encoder in pure NumPy or a
  zero-shot :class:`~normet.foundation.ChronosEmbedder`).
- Conditions state transitions on external meteorological actions
  :math:`f_\\theta(s_{p-1}, c_p, w_p) \\to \\hat{s}_p` rather than predicting
  raw noisy concentrations directly.
- Uses whitened latent innovation residuals for meteorology-invariant anomaly
  detection and weather-removed latent trajectories for leak-free synthetic
  control donor matching.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..analysis.diagnostics import counterfactual_separability_audit
from ..analysis.events import _intervals_from_mask
from ..causal._common import newey_west_se_mean, solve_simplex_weights_fast
from ..utils._config import DEFAULT_SEED
from ._common import (
    _blocked_folds,
    _build_calendar_basis,
    _build_met_design,
    _calendar_basis_state,
    _ensure_datetime_frame,
    _met_feature_matrix,
    _ridge_cv_fit,
    _ridge_predict,
    _standardize_apply,
    _standardize_fit,
)

__all__ = [
    "JEPAResult",
    "JEPASCMResult",
    "LatentJEPA",
    "fit_latent_jepa",
    "jepa_anomaly_scores",
    "detect_jepa_events",
    "jepa_select_donors",
    "jepa_scm",
]


@dataclass
class JEPAResult:
    """Outputs of a forward pass through :class:`LatentJEPA`."""

    embeddings: pd.DataFrame
    predicted_embeddings: pd.DataFrame
    latent_residuals: pd.DataFrame
    patch_index: pd.DatetimeIndex
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class JEPASCMResult:
    """Causal policy evaluation outputs from JEPA-deweathered synthetic control."""

    synthetic: pd.DataFrame
    weights: pd.Series
    donor_table: pd.DataFrame
    cutoff_date: pd.Timestamp
    att: float
    att_pct: float
    att_hac_se: float
    pre_rmspe: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _encode_spectral_lag_windows(
    Y: np.ndarray,
    patch_length: int,
    stride: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Encode windows of shape ``(L, M)`` into multi-scale spectral/lag features."""
    n, m_dim = Y.shape
    L = max(2, int(patch_length))
    st = max(1, int(stride))
    if n < L:
        raise ValueError(f"Need at least patch_length={L} timestamps, got {n}.")

    t_grid = np.linspace(-1.0, 1.0, L)
    t_denom = float(np.sum(t_grid**2)) or 1.0
    ang_24 = 2.0 * np.pi * np.arange(L, dtype=float) / max(L, 4)
    sin_b, cos_b = np.sin(ang_24), np.cos(ang_24)

    # Windows start at 0, stride, 2*stride, ...; when (n - L) is not a
    # multiple of the stride they stop short of the newest rows, which then sit
    # in no patch at all and would keep an older timestamp's score.  A final
    # window ending at ``n`` covers that tail (it overlaps the previous patch
    # by (n - L) % stride rows; the per-timestamp score takes the max).
    starts = list(range(0, n - L + 1, st))
    if starts[-1] != n - L:
        starts.append(n - L)

    feats: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    for s in starts:
        e = s + L
        seg = Y[s:e]
        col_vec: list[float] = []
        for j in range(m_dim):
            v = seg[:, j]
            fin = np.isfinite(v)
            if not np.any(fin):
                col_vec.extend([0.0] * 9)
                continue
            vf = np.where(fin, v, float(np.nanmean(v)))
            mu = float(np.mean(vf))
            sd = float(np.std(vf))
            q25, q75 = np.quantile(vf, [0.25, 0.75])
            slope = float(np.dot(t_grid, vf - mu) / t_denom)
            diffs = np.diff(vf)
            rms_diff = float(np.sqrt(np.mean(diffs**2))) if diffs.size else 0.0
            end_jump = float(vf[-1] - vf[0])
            sin_p = float(np.mean((vf - mu) * sin_b))
            cos_p = float(np.mean((vf - mu) * cos_b))
            col_vec.extend(
                [mu, sd, float(q25), float(q75), slope, rms_diff, end_jump, sin_p, cos_p]
            )
        if m_dim >= 2:
            m0 = max(float(np.nanmean(seg[:, 0])), 0.0)
            for j in range(1, m_dim):
                mj = max(float(np.nanmean(seg[:, j])), 0.0)
                col_vec.append(float(np.log1p(m0) - np.log1p(mj)))
        feats.append(np.asarray(col_vec, dtype=float))
        bounds.append((s, e))
    return np.vstack(feats), bounds


def _encode_via_external_embedder(
    Y: np.ndarray,
    encoder: Any,
    context_length: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Encode target channels via a ChronosEmbedder-compatible ``embed_patches`` interface."""
    n, m_dim = Y.shape
    eff_ctx = min(n, max(16, int(context_length)))
    tail_offset = max(0, n - eff_ctx)
    ch_patches: list[np.ndarray] = []
    for j in range(m_dim):
        p_j = np.asarray(encoder.embed_patches(Y[:, j], context_length=eff_ctx), dtype=float)
        if p_j.ndim != 2:
            raise ValueError(f"Expected 2D patch array from embed_patches, got {p_j.shape}.")
        ch_patches.append(p_j)
    min_p = min(p.shape[0] for p in ch_patches)
    raw_patches = np.mean(np.stack([p[:min_p] for p in ch_patches], axis=0), axis=0)
    bounds: list[tuple[int, int]] = []
    for i in range(min_p):
        s = tail_offset + int(round(i * eff_ctx / min_p))
        e = tail_offset + max(int(round((i + 1) * eff_ctx / min_p)), s - tail_offset + 1)
        bounds.append((min(s, n - 1), min(e, n)))
    return raw_patches, bounds


class LatentJEPA:
    """Action-Conditioned Joint Embedding Predictive Architecture for atmospheric series."""

    def __init__(
        self,
        *,
        patch_length: int = 6,
        stride: int | None = None,
        latent_dim: int = 8,
        met_degree: int = 2,
        n_harmonics: int = 2,
        alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
        encoder: str | Any = "spectral_lag",
        context_length: int = 2048,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.patch_length = max(2, int(patch_length))
        self.stride = self.patch_length if stride is None else max(1, int(stride))
        self.latent_dim = max(1, int(latent_dim))
        self.met_degree = max(1, int(met_degree))
        self.n_harmonics = max(1, int(n_harmonics))
        self.alphas = tuple(float(a) for a in alphas)
        self.encoder = encoder
        self.context_length = max(16, int(context_length))
        self.seed = int(seed)

        self._target_cols: list[str] = []
        self._met_cols: list[str] = []
        self._pca_mu: np.ndarray = np.empty(0, dtype=float)
        self._pca_sd: np.ndarray = np.empty(0, dtype=float)
        self._pca_proj: np.ndarray = np.empty((0, 0), dtype=float)
        self._met_stats: tuple[np.ndarray, np.ndarray] = (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )
        self._trans_model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._stat_model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._n_cal_cols: int = 0
        self._cal_basis: dict[str, Any] | None = None
        self._prec: np.ndarray = np.empty((0, 0), dtype=float)
        self._med_raw_norm: float = 0.0
        self._scale_raw_norm: float = 1.0
        self.diagnostics_: dict[str, Any] = {}

    def _raw_patches(
        self,
        df_s: pd.DataFrame,
        target_cols: Sequence[str],
        met_cols: Sequence[str],
        *,
        fit_stats: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list[tuple[int, int]]]:
        Y = (
            df_s[list(target_cols)]
            .apply(pd.to_numeric, errors="coerce")
            .ffill()
            .bfill()
            .to_numpy(dtype=float)
        )
        if self.encoder == "spectral_lag":
            raw_S, bounds = _encode_spectral_lag_windows(Y, self.patch_length, self.stride)
        elif hasattr(self.encoder, "embed_patches"):
            raw_S, bounds = _encode_via_external_embedder(Y, self.encoder, self.context_length)
        else:
            raise ValueError(
                "`encoder` must be 'spectral_lag' or an object exposing .embed_patches()."
            )

        if met_cols:
            if fit_stats:
                X_met_full, self._met_stats = _met_feature_matrix(
                    df_s, met_cols, degree=self.met_degree
                )
            else:
                X_met_full, _ = _met_feature_matrix(
                    df_s, met_cols, degree=self.met_degree, fit_stats=self._met_stats
                )
        else:
            X_met_full = np.empty((len(df_s), 0), dtype=float)

        idx_cal = pd.DatetimeIndex(df_s.index)
        if fit_stats or self._cal_basis is None:
            # Pin the fit window's basis; the fit's calendar design is what the
            # ridge coefficients (and their standardisation) are on.
            self._cal_basis = _calendar_basis_state(idx_cal, n_harmonics=self.n_harmonics)
        X_cal_full = _build_calendar_basis(idx_cal, state=self._cal_basis)
        idx_vals: list[pd.Timestamp] = []
        W_list: list[np.ndarray] = []
        C_list: list[np.ndarray] = []
        for s, e in bounds:
            mid = min((s + e - 1) // 2, len(df_s) - 1)
            idx_vals.append(pd.Timestamp(df_s.index[mid]))
            W_list.append(
                np.mean(X_met_full[s:e], axis=0)
                if X_met_full.shape[1] > 0
                else np.empty(0, dtype=float)
            )
            C_list.append(np.mean(X_cal_full[s:e], axis=0))
        W_p = (
            np.vstack(W_list)
            if X_met_full.shape[1] > 0
            else np.empty((len(bounds), 0), dtype=float)
        )
        C_p = np.vstack(C_list)
        return raw_S, W_p, C_p, pd.DatetimeIndex(idx_vals), bounds

    def _project_fit(self, raw_S: np.ndarray) -> np.ndarray:
        Z, mu, sd = _standardize_fit(raw_S)
        self._pca_mu = mu
        self._pca_sd = sd
        k = min(self.latent_dim, raw_S.shape[1], max(1, raw_S.shape[0] - 1))
        _, _, vt = np.linalg.svd(Z, full_matrices=False)
        self._pca_proj = vt[:k].T
        return Z @ self._pca_proj

    def _project_apply(self, raw_S: np.ndarray) -> np.ndarray:
        Z = _standardize_apply(raw_S, self._pca_mu, self._pca_sd)
        return Z @ self._pca_proj

    def fit(
        self,
        df: pd.DataFrame,
        target_cols: str | Sequence[str],
        met_cols: Sequence[str],
    ) -> LatentJEPA:
        """Fit latent state projection and action-conditioned transition predictor."""
        df_s = _ensure_datetime_frame(df)
        tcols = [target_cols] if isinstance(target_cols, str) else list(target_cols)
        mcols = list(met_cols)
        if not tcols:
            raise ValueError("`target_cols` must be non-empty.")
        for c in [*tcols, *mcols]:
            if c not in df_s.columns:
                raise KeyError(f"Column {c!r} not found in DataFrame.")

        self._target_cols = tcols
        self._met_cols = mcols
        raw_S, W_p, C_p, _p_idx, _bounds = self._raw_patches(df_s, tcols, mcols, fit_stats=True)
        S = self._project_fit(raw_S)
        n_p, d_lat = S.shape
        if n_p < 6:
            raise ValueError(
                f"Need at least 6 patches to fit LatentJEPA dynamics, got {n_p}. "
                "Use smaller patch_length/stride or a longer series."
            )

        self._n_cal_cols = C_p.shape[1]
        X_static = np.hstack([C_p, W_p]) if W_p.shape[1] > 0 else C_p
        s_coef, s_int, s_mu, s_sd, s_alpha = _ridge_cv_fit(X_static, S, alphas=self.alphas)
        self._stat_model = (s_coef, s_int, s_mu, s_sd)

        X_trans = (
            np.hstack([S[:-1], W_p[:-1], C_p[1:], W_p[1:]])
            if W_p.shape[1] > 0
            else np.hstack([S[:-1], C_p[1:]])
        )
        t_coef, t_int, t_mu, t_sd, t_alpha = _ridge_cv_fit(X_trans, S[1:], alphas=self.alphas)
        self._trans_model = (t_coef, t_int, t_mu, t_sd)

        S_pred = _ridge_predict(X_trans, t_coef, t_int, t_mu, t_sd)
        E = S[1:] - S_pred
        cov_e = np.cov(E, rowvar=False) if d_lat > 1 else np.array([[float(np.var(E))]])
        cov_e = np.atleast_2d(cov_e)
        diag_mean = max(float(np.mean(np.diag(cov_e))), 1e-6)
        cov_shrunk = 0.85 * cov_e + 0.15 * diag_mean * np.eye(d_lat, dtype=float)
        cov_shrunk += 1e-5 * np.eye(d_lat, dtype=float)
        self._prec = np.linalg.inv(cov_shrunk)

        raw_norms = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", E, self._prec, E), 0.0))
        med = float(np.median(raw_norms))
        mad = float(np.median(np.abs(raw_norms - med)))
        self._med_raw_norm = med
        self._scale_raw_norm = max(1.4826 * mad, float(np.std(raw_norms)), 1e-6)

        audit = (
            counterfactual_separability_audit(
                S[1:, 0],
                S_pred[:, 0],
                W_p[1:],
                folds=_blocked_folds(len(S) - 1, n_splits=3),
                nonlinear=False,
                n_boot=0,
                seed=self.seed,
            )
            if W_p.shape[1] > 0 and len(S) >= 12
            else {}
        )
        sst = float(np.var(S[1:]))
        mse = float(np.mean(E**2))
        latent_r2 = float(1.0 - mse / sst) if sst > 1e-12 else 0.0
        self.diagnostics_ = {
            "n_patches": int(n_p),
            "latent_dim": int(d_lat),
            "static_alpha": float(s_alpha),
            "transition_alpha": float(t_alpha),
            "latent_r2": latent_r2,
            "latent_rc_score": float(audit.get("rc_score_linear", float("nan"))),
            "latent_rho_lb": float(audit.get("interaction_rho_lb_raw", float("nan"))),
        }
        return self

    def _require_fitted(
        self,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        if self._trans_model is None or self._stat_model is None:
            raise RuntimeError("LatentJEPA must be fitted via .fit() before prediction.")
        return self._trans_model, self._stat_model

    def predict_latents(self, df: pd.DataFrame) -> JEPAResult:
        """Compute observed patches, action-conditioned latent predictions, and innovations."""
        t_mod, s_mod = self._require_fitted()
        df_s = _ensure_datetime_frame(df)
        raw_S, W_p, C_p, p_idx, bounds = self._raw_patches(
            df_s, self._target_cols, self._met_cols, fit_stats=False
        )
        S = self._project_apply(raw_S)
        S_pred = np.empty_like(S)

        X_static0 = np.hstack([C_p[:1], W_p[:1]]) if W_p.shape[1] > 0 else C_p[:1]
        S_pred[0] = _ridge_predict(X_static0, *s_mod)[0]
        if len(S) > 1:
            X_trans = (
                np.hstack([S[:-1], W_p[:-1], C_p[1:], W_p[1:]])
                if W_p.shape[1] > 0
                else np.hstack([S[:-1], C_p[1:]])
            )
            S_pred[1:] = _ridge_predict(X_trans, *t_mod)

        E = S - S_pred
        cols = [f"z_{i}" for i in range(S.shape[1])]
        return JEPAResult(
            embeddings=pd.DataFrame(S, index=p_idx, columns=cols),
            predicted_embeddings=pd.DataFrame(S_pred, index=p_idx, columns=cols),
            latent_residuals=pd.DataFrame(E, index=p_idx, columns=cols),
            patch_index=p_idx,
            diagnostics={**self.diagnostics_, "bounds": bounds},
        )

    def anomaly_scores(self, df: pd.DataFrame) -> pd.Series:
        """Return per-timestamp whitened AC-JEPA innovation z-scores aligned to ``df.index``.

        Every timestamp carries its own score: the patch windows cover the whole
        frame, the last one overlapping the previous patch when the frame length
        is not a multiple of the stride (see
        :func:`_encode_spectral_lag_windows`), and a row inside several patches
        takes their maximum. ``ffill``/``bfill`` only stand in for rows a
        context-limited external encoder leaves outside its patches.
        """
        df_s = _ensure_datetime_frame(df)
        res = self.predict_latents(df_s)
        E = res.latent_residuals.to_numpy(dtype=float)
        raw_norms = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", E, self._prec, E), 0.0))
        if len(raw_norms) > 0:
            raw_norms[0] = min(raw_norms[0], self._med_raw_norm)
        z_patch = np.maximum(0.0, (raw_norms - self._med_raw_norm) / self._scale_raw_norm)

        n = len(df_s)
        acc = np.zeros(n, dtype=float)
        cnt = np.zeros(n, dtype=float)
        bounds: list[tuple[int, int]] = res.diagnostics["bounds"]
        for p, (s, e) in enumerate(bounds):
            acc[s:e] = np.maximum(acc[s:e], z_patch[p])
            cnt[s:e] += 1.0
        mapped = np.where(cnt > 0.0, acc, np.nan)
        ser = (
            pd.Series(mapped, index=df_s.index, name="jepa_anomaly_score")
            .ffill()
            .bfill()
            .fillna(0.0)
        )
        if not isinstance(df.index, pd.DatetimeIndex):
            return pd.Series(ser.to_numpy(dtype=float), index=df.index, name="jepa_anomaly_score")
        return ser.reindex(df.index)

    def deweathered_state_summary(self, df: pd.DataFrame) -> np.ndarray:
        """Project out estimated meteorological action effects from patch embeddings."""
        _t_mod, (s_coef, _s_int, s_mu, s_sd) = self._require_fitted()
        df_s = _ensure_datetime_frame(df)
        raw_S, W_p, _C_p, _p_idx, _bounds = self._raw_patches(
            df_s, self._target_cols, self._met_cols, fit_stats=False
        )
        S = self._project_apply(raw_S)
        if W_p.shape[1] == 0:
            return S
        w_mu = self._met_stats[0]
        # Compute met action contribution in standardized static feature space
        met_mu_s = s_mu[self._n_cal_cols :]
        met_sd_s = s_sd[self._n_cal_cols :]
        W_std = _standardize_apply(W_p, met_mu_s, met_sd_s)
        met_coef = s_coef[self._n_cal_cols :]
        if w_mu.size == 0:
            return S
        met_effect = W_std @ met_coef
        return S - (met_effect - np.mean(met_effect, axis=0, keepdims=True))


def fit_latent_jepa(
    df: pd.DataFrame,
    target_cols: str | Sequence[str],
    met_cols: Sequence[str],
    **kwargs: Any,
) -> LatentJEPA:
    """Convenience wrapper fitting a :class:`LatentJEPA` model."""
    return LatentJEPA(**kwargs).fit(df, target_cols, met_cols)


def jepa_anomaly_scores(
    df: pd.DataFrame,
    target_cols: str | Sequence[str],
    met_cols: Sequence[str],
    *,
    jepa: LatentJEPA | None = None,
    **kwargs: Any,
) -> pd.Series:
    """Compute AC-JEPA meteorology-invariant anomaly scores for ``df``."""
    model = jepa if jepa is not None else fit_latent_jepa(df, target_cols, met_cols, **kwargs)
    return model.anomaly_scores(df)


def detect_jepa_events(
    df: pd.DataFrame,
    target_cols: str | Sequence[str],
    met_cols: Sequence[str],
    *,
    threshold: float = 3.0,
    min_length: int = 1,
    jepa: LatentJEPA | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Detect consecutive deweathered anomaly episodes from AC-JEPA innovations."""
    score = jepa_anomaly_scores(df, target_cols, met_cols, jepa=jepa, **kwargs)
    arr = score.to_numpy(dtype=float)
    mask = (arr > float(threshold)) & np.isfinite(arr)
    idx = (
        score.index
        if isinstance(score.index, pd.DatetimeIndex)
        else pd.to_datetime(_ensure_datetime_frame(df).index)
    )
    events = _intervals_from_mask(pd.DatetimeIndex(idx), mask, arr)
    if events.empty:
        return events
    return events[events["n"] >= int(min_length)].reset_index(drop=True)


def jepa_select_donors(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    unit_col: str = "ID",
    outcome_col: str = "value",
    met_cols: Sequence[str],
    treated_unit: str,
    cutoff_date: str | pd.Timestamp,
    donors: Sequence[str] | None = None,
    patch_length: int = 6,
    stride: int | None = None,
    latent_dim: int = 6,
    met_degree: int = 2,
    top_k: int | None = None,
    min_similarity: float | None = None,
    return_weights: bool = True,
    l2_penalty: float = 0.05,
) -> pd.DataFrame:
    """Select synthetic-control donors by pre-cutoff deweathered JEPA latent dynamics.

    Removes each site's local meteorological confounding strictly before
    ``cutoff_date`` so stations sharing weather regimes are not mistaken for
    emission-matched donors.
    """
    if top_k is not None and int(top_k) < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if date_col not in df.columns:
        raise KeyError(f"`date_col` {date_col!r} not found in DataFrame.")
    if unit_col not in df.columns:
        raise KeyError(f"`unit_col` {unit_col!r} not found in DataFrame.")
    if outcome_col not in df.columns:
        raise KeyError(f"`outcome_col` {outcome_col!r} not found in DataFrame.")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    cutoff_ts = pd.Timestamp(cutoff_date)
    if work[date_col].dt.tz is not None and cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize(work[date_col].dt.tz)
    elif work[date_col].dt.tz is None and cutoff_ts.tzinfo is not None:
        cutoff_ts = cutoff_ts.tz_localize(None)

    pre = work.loc[work[date_col] < cutoff_ts].copy()
    if pre.empty:
        raise ValueError("No pre-cutoff observations available before `cutoff_date`.")

    units_all = [str(u) for u in pre[unit_col].unique()]
    if treated_unit not in units_all:
        raise ValueError(f"Treated unit {treated_unit!r} absent in pre-cutoff period.")
    donor_list = (
        [u for u in units_all if u != treated_unit]
        if donors is None
        else [str(u) for u in donors if str(u) in units_all and str(u) != treated_unit]
    )
    if not donor_list:
        raise ValueError("No valid candidate donors available in pre-cutoff period.")

    st = patch_length if stride is None else max(1, int(stride))
    mcols = [c for c in met_cols if c in pre.columns]

    raw_patch_map: dict[str, np.ndarray] = {}
    w_patch_map: dict[str, np.ndarray] = {}
    c_patch_map: dict[str, np.ndarray] = {}
    obs_vec_map: dict[str, pd.Series] = {}

    # Extract shared deterministic spectral/lag patch representations per unit
    all_raw_blocks: list[np.ndarray] = []
    for u in [treated_unit, *donor_list]:
        sub = (
            pre.loc[pre[unit_col] == u]
            .sort_values(date_col)
            .drop_duplicates(subset=[date_col])
            .set_index(date_col)
        )
        y_u = pd.to_numeric(sub[outcome_col], errors="coerce").ffill().bfill()
        if len(y_u) < max(patch_length * 4, patch_length + 4):
            continue
        raw_s, bounds = _encode_spectral_lag_windows(
            y_u.to_numpy(dtype=float)[:, None], patch_length, st
        )
        X_met_u, _ = _met_feature_matrix(sub, mcols, degree=met_degree)
        X_cal_u = _build_calendar_basis(pd.DatetimeIndex(sub.index), n_harmonics=2)
        w_u = (
            np.vstack([np.mean(X_met_u[s:e], axis=0) for s, e in bounds])
            if X_met_u.shape[1] > 0
            else np.empty((len(bounds), 0), dtype=float)
        )
        c_u = np.vstack([np.mean(X_cal_u[s:e], axis=0) for s, e in bounds])
        raw_patch_map[u] = raw_s
        w_patch_map[u] = w_u
        c_patch_map[u] = c_u
        obs_vec_map[u] = y_u
        all_raw_blocks.append(raw_s)

    if treated_unit not in raw_patch_map:
        raise ValueError(f"Treated unit {treated_unit!r} has too few pre-cutoff observations.")

    pooled = np.vstack(all_raw_blocks)
    Z_pool, p_mu, p_sd = _standardize_fit(pooled)
    k = min(int(latent_dim), pooled.shape[1], max(1, pooled.shape[0] - 1))
    _, _, vt = np.linalg.svd(Z_pool, full_matrices=False)
    proj = vt[:k].T

    # Remove each unit's local meteorological effect in the shared latent space
    deweathered: dict[str, np.ndarray] = {}
    for u, raw_s in raw_patch_map.items():
        S_u = _standardize_apply(raw_s, p_mu, p_sd) @ proj
        W_u = w_patch_map[u]
        C_u = c_patch_map[u]
        if W_u.shape[1] > 0 and len(S_u) >= 6:
            X_st = np.hstack([C_u, W_u])
            coef, _y_mu, _x_mu, _x_sd, _a = _ridge_cv_fit(X_st, S_u, alphas=(0.1, 1.0, 10.0, 50.0))
            Z_st, _, _ = _standardize_fit(X_st)
            n_c = C_u.shape[1]
            met_part = Z_st[:, n_c:] @ coef[n_c:]
            S_dew = S_u - (met_part - np.mean(met_part, axis=0, keepdims=True))
        else:
            S_dew = S_u
        deweathered[u] = S_dew

    # Align lengths smoothly via minimum common patch count (trailing pre-cutoff window)
    valid_donors = [d for d in donor_list if d in deweathered]
    if not valid_donors:
        raise ValueError("No candidate donors have sufficient pre-cutoff patches.")
    min_len = min(len(deweathered[u]) for u in [treated_unit, *valid_donors])

    def _flat_centered(mat: np.ndarray) -> np.ndarray:
        sub_m = mat[-min_len:]
        return (sub_m - np.mean(sub_m, axis=0, keepdims=True)).ravel()

    t_lat = _flat_centered(deweathered[treated_unit])
    t_norm = float(np.linalg.norm(t_lat))
    t_obs = obs_vec_map[treated_unit]

    scored: list[tuple[str, float, float]] = []
    for d in valid_donors:
        d_lat = _flat_centered(deweathered[d])
        denom = t_norm * float(np.linalg.norm(d_lat))
        sim = float(np.dot(t_lat, d_lat) / denom) if denom > 0.0 else 0.0

        common_idx = t_obs.index.intersection(obs_vec_map[d].index)
        if len(common_idx) >= 5:
            a_raw = t_obs.loc[common_idx].to_numpy(dtype=float)
            b_raw = obs_vec_map[d].loc[common_idx].to_numpy(dtype=float)
            sa, sb = float(np.std(a_raw)), float(np.std(b_raw))
            raw_sim = float(np.corrcoef(a_raw, b_raw)[0, 1]) if sa > 1e-12 and sb > 1e-12 else 0.0
        else:
            raw_sim = float("nan")
        scored.append((d, sim, raw_sim))

    scored.sort(key=lambda r: (-r[1], r[0]))
    rows: list[dict[str, Any]] = []
    selected_units: list[str] = []
    for rank, (unit, sim, raw_sim) in enumerate(scored, start=1):
        sel = True
        if min_similarity is not None and sim < float(min_similarity):
            sel = False
        if top_k is not None and rank > int(top_k):
            sel = False
        if sel:
            selected_units.append(unit)
        rows.append(
            {
                "unit": unit,
                "similarity": sim,
                "raw_similarity": raw_sim,
                "rank": rank,
                "selected": sel,
                "weight": 0.0,
            }
        )

    out_df = pd.DataFrame(rows)
    if return_weights and selected_units:
        X_mat = np.column_stack([_flat_centered(deweathered[u]) for u in selected_units])
        w_sol = solve_simplex_weights_fast(X_mat, t_lat, l2_penalty=float(l2_penalty))
        w_map = dict(zip(selected_units, w_sol, strict=True))
        out_df["weight"] = out_df["unit"].map(lambda u: float(w_map.get(u, 0.0)))
    return out_df


def jepa_scm(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    unit_col: str = "ID",
    outcome_col: str = "value",
    met_cols: Sequence[str],
    treated_unit: str,
    cutoff_date: str | pd.Timestamp,
    donors: Sequence[str] | None = None,
    patch_length: int = 6,
    stride: int | None = None,
    latent_dim: int = 6,
    met_head: str = "gam",
    n_knots: int = 6,
    top_k: int | None = None,
    min_similarity: float | None = None,
    l2_penalty: float = 0.05,
) -> JEPASCMResult:
    """Run JEPA-deweathered Synthetic Control across pre- and post-intervention periods.

    1. Ranks donors and solves simplex weights strictly before ``cutoff_date`` in
       weather-invariant AC-JEPA latent space (:func:`jepa_select_donors`).
    2. Fits each station's local meteorological action response (GAM splines by default)
       using *only* pre-cutoff data to avoid absorbing post-intervention shifts.
    3. Constructs the deweathered counterfactual synthetic trajectory, ATT, %ATT,
       HAC standard error, and pre-period RMSPE.
    """
    donor_tbl = jepa_select_donors(
        df,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        met_cols=met_cols,
        treated_unit=treated_unit,
        cutoff_date=cutoff_date,
        donors=donors,
        patch_length=patch_length,
        stride=stride,
        latent_dim=latent_dim,
        top_k=top_k,
        min_similarity=min_similarity,
        return_weights=True,
        l2_penalty=l2_penalty,
    )

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    cutoff_ts = pd.Timestamp(cutoff_date)
    if work[date_col].dt.tz is not None and cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize(work[date_col].dt.tz)
    elif work[date_col].dt.tz is None and cutoff_ts.tzinfo is not None:
        cutoff_ts = cutoff_ts.tz_localize(None)

    active = donor_tbl.loc[donor_tbl["weight"] > 1e-8]
    if active.empty:
        active = donor_tbl.iloc[:1]
    sel_donors = active["unit"].astype(str).tolist()
    w_vec: npt.NDArray[np.float64] = active["weight"].to_numpy(dtype=float)
    w_sum = float(np.sum(w_vec))
    w_vec = w_vec / w_sum if w_sum > 0.0 else np.full(len(sel_donors), 1.0 / len(sel_donors))
    weights_ser = pd.Series(w_vec, index=sel_donors, name="weight")

    mcols = [c for c in met_cols if c in work.columns]

    def _deweather_unit(unit_id: str) -> tuple[pd.Series, pd.Series]:
        sub = (
            work.loc[work[unit_col] == unit_id]
            .sort_values(date_col)
            .drop_duplicates(subset=[date_col])
            .set_index(date_col)
        )
        idx_u = pd.DatetimeIndex(sub.index)
        y_u = pd.to_numeric(sub[outcome_col], errors="coerce").ffill().bfill().to_numpy(dtype=float)
        pre_m = np.asarray(idx_u < cutoff_ts, dtype=bool)
        if not mcols or int(pre_m.sum()) < 10:
            ser = pd.Series(y_u, index=idx_u, name=unit_id)
            return ser, ser

        sub_pre = sub.iloc[pre_m]
        y_pre = y_u[pre_m]
        # One calendar design for the whole record, sliced for the fit rows:
        # the width (quadratic past 90 d, annual past 180 d) and the trend
        # scale then match between the pre-cutoff fit and its application, and
        # the pre-only ridge fit still sees pre rows only.
        C_full = _build_calendar_basis(idx_u, n_harmonics=2)
        C_pre = C_full[pre_m]
        M_pre, met_state = _build_met_design(
            sub_pre, mcols, met_head=met_head, degree=1, n_knots=n_knots
        )
        X_pre = np.hstack([C_pre, M_pre])
        coef, _y_mu, x_mu, x_sd, _ = _ridge_cv_fit(X_pre, y_pre, alphas=(0.1, 1.0, 10.0, 50.0))

        M_full, _ = _build_met_design(sub, mcols, state=met_state)
        X_full = np.hstack([C_full, M_full])
        Z_full = _standardize_apply(X_full, x_mu, x_sd)
        n_c = C_full.shape[1]
        met_part = Z_full[:, n_c:] @ coef[n_c:, 0]
        y_dew = y_u - met_part
        return pd.Series(y_u, index=idx_u, name=unit_id), pd.Series(
            y_dew, index=idx_u, name=unit_id
        )

    obs_tr, dew_tr = _deweather_unit(treated_unit)
    dew_donors: dict[str, pd.Series] = {}
    for d in sel_donors:
        _, dew_d = _deweather_unit(d)
        dew_donors[d] = dew_d

    common_idx = dew_tr.index
    for d_ser in dew_donors.values():
        common_idx = common_idx.intersection(d_ser.index)
    common_idx = common_idx.sort_values()

    y_tr_al = dew_tr.reindex(common_idx).to_numpy(dtype=float)
    obs_tr_al = obs_tr.reindex(common_idx).to_numpy(dtype=float)
    D_mat = np.column_stack(
        [dew_donors[d].reindex(common_idx).to_numpy(dtype=float) for d in sel_donors]
    )
    synth_raw = D_mat @ w_vec

    pre_mask = np.asarray(pd.DatetimeIndex(common_idx) < cutoff_ts, dtype=bool)
    post_mask = ~pre_mask
    if not np.any(pre_mask):
        raise ValueError("No aligned pre-cutoff timestamps across treated and selected donors.")

    shift = float(np.mean(y_tr_al[pre_mask] - synth_raw[pre_mask]))
    synth = synth_raw + shift
    effect = y_tr_al - synth
    pre_rmspe = float(np.sqrt(np.mean(effect[pre_mask] ** 2)))

    if np.any(post_mask):
        post_eff = effect[post_mask]
        att = float(np.mean(post_eff))
        mean_synth_post = float(np.mean(synth[post_mask]))
        att_pct = float(100.0 * att / max(abs(mean_synth_post), 1e-12))
        hac_se, _ = newey_west_se_mean(post_eff) if int(post_mask.sum()) >= 4 else (float("nan"), 0)
    else:
        att = float("nan")
        att_pct = float("nan")
        hac_se = float("nan")

    synth_df = pd.DataFrame(
        {
            "observed": obs_tr_al,
            "deweathered": y_tr_al,
            "synthetic": synth,
            "effect": effect,
        },
        index=common_idx,
    )
    return JEPASCMResult(
        synthetic=synth_df,
        weights=weights_ser,
        donor_table=donor_tbl,
        cutoff_date=cutoff_ts,
        att=att,
        att_pct=att_pct,
        att_hac_se=float(hac_se) if np.isfinite(hac_se) else float("nan"),
        pre_rmspe=pre_rmspe,
        diagnostics={
            "n_pre": int(pre_mask.sum()),
            "n_post": int(post_mask.sum()),
            "n_donors_selected": len(sel_donors),
            "met_head": met_head,
            "pre_level_alignment": shift,
        },
    )
