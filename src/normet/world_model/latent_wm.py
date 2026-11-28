"""Separable Latent Atmospheric World Model (JEPA-inspired counterfactual engine).

Learns a bijective latent representation :math:`S_t = \\phi(Y_t)` (shifted power
coordinates coupled with an invertible unit-determinant linear mixing layer for
multi-species panels, in either ``warp_then_mix`` or ``mix_then_warp`` order)
in which meteorological action effects :math:`g(w_t)` and latent emission
disturbances :math:`U_t` become additively separable:

.. math::
    S_t = \\mu_{\\text{base}}(c_t) + g_{\\text{met}}(w_t) + U_t.

Because :math:`\\phi` is strictly invertible, counterfactual weather draws and
trajectory-level What-If rollouts are carried out in the separable latent space
and decoded back to exact physical concentrations with accompanying factual
separability certificates.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..analysis.diagnostics import counterfactual_separability_audit
from ..utils._config import DEFAULT_SEED
from ._common import (
    _build_calendar_basis,
    _build_fold_projector,
    _build_met_design,
    _calendar_basis_state,
    _calendar_basis_width,
    _day_interleaved_folds,
    _ensure_datetime_frame,
    _fast_oof_separability_proj,
    _FoldProjector,
    _inv_shifted_power,
    _ridge_cv_fit,
    _ridge_fit,
    _ridge_predict,
    _shifted_power,
    _standardize_apply,
    _standardize_fit,
)

__all__ = [
    "WorldModelConfig",
    "WorldModelNormaliseResult",
    "TrajectoryRolloutResult",
    "SeparableWorldModel",
    "fit_world_model",
]


@dataclass
class WorldModelConfig:
    """Configuration for :class:`SeparableWorldModel`."""

    backend: str = "separable"  # "separable" | "torch"
    met_head: str = "gam"  # "gam" | "linear" | "poly" | "tree"
    n_knots: int = 6
    met_degree: int = 1
    n_harmonics: int = 2
    coordinate_order: str = "auto"  # "auto" | "warp_then_mix" | "mix_then_warp"
    lambda_rc: float = 1.0
    lambda_het: float = 2.0
    lambda_reg: float = 0.05
    ar_persistence: bool = True
    carryover_min_phi: float = 0.15
    alphas: tuple[float, ...] = (0.5, 2.0, 10.0, 50.0, 200.0)
    fixed_gamma: float | None = None
    learn_mixing: bool = True
    torch_epochs: int = 250
    torch_lr: float = 2e-2
    torch_hidden: int = 32
    device: str = "auto"
    seed: int = DEFAULT_SEED


@dataclass
class WorldModelNormaliseResult:
    """De-weathered outputs and counterfactual separability certificates."""

    df: pd.DataFrame
    species: list[str]
    latent_audit: dict[str, Any]
    raw_audit: dict[str, Any]
    separability_gain: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_normet_frame(self, species: str | None = None) -> pd.DataFrame:
        """Return standard ``observed/normalised/factual`` frame for a species."""
        sp = self.species[0] if species is None else species
        if sp not in self.species:
            raise KeyError(f"Species {sp!r} not found in {self.species}.")
        if len(self.species) == 1 and "normalised" in self.df.columns:
            cols = [c for c in ("observed", "normalised", "factual") if c in self.df.columns]
            q_cols = [c for c in self.df.columns if c.startswith("q") and c[1:].isdigit()]
            return self.df[[*cols, *q_cols]].copy()
        out = pd.DataFrame(
            {
                "observed": self.df[f"{sp}_observed"],
                "normalised": self.df[f"{sp}_normalised"],
                "factual": self.df[f"{sp}_factual"],
            },
            index=self.df.index,
        )
        prefix = f"{sp}_q"
        for c in self.df.columns:
            if c.startswith(prefix):
                out[c[len(sp) + 1 :]] = self.df[c]
        return out


@dataclass
class TrajectoryRolloutResult:
    """Outputs of a trajectory-level What-If counterfactual rollout."""

    start: pd.Timestamp
    end: pd.Timestamp
    species: list[str]
    actual: pd.DataFrame
    counterfactual: pd.DataFrame
    difference: pd.DataFrame
    latent_disturbance: pd.DataFrame
    separability_certificate: dict[str, Any]
    summary_metrics: dict[str, Any] = field(default_factory=dict)
    counterfactual_p10: pd.DataFrame | None = None
    counterfactual_p90: pd.DataFrame | None = None


def _lu_matrix(theta: np.ndarray, m_dim: int) -> np.ndarray:
    """Build invertible unit-determinant ``W = L @ U`` from ``M*(M-1)`` params."""
    L = np.eye(m_dim, dtype=float)
    U = np.eye(m_dim, dtype=float)
    k = 0
    for i in range(1, m_dim):
        for j in range(i):
            L[i, j] = float(theta[k])
            k += 1
    for i in range(m_dim):
        for j in range(i + 1, m_dim):
            U[i, j] = float(theta[k])
            k += 1
    return L @ U


def _serialize_met_state(state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"met_head": state.get("met_head", "gam")}
    if "degree" in state:
        out["degree"] = int(state["degree"])
    if "stats" in state:
        mu, sd = state["stats"]
        out["stats"] = [np.asarray(mu, dtype=float).tolist(), np.asarray(sd, dtype=float).tolist()]
    if "gam_state" in state:
        gs = state["gam_state"]
        out["gam_state"] = {
            "mu": np.asarray(gs["mu"], dtype=float).tolist(),
            "sd": np.asarray(gs["sd"], dtype=float).tolist(),
            "knots": [np.asarray(k, dtype=float).tolist() for k in gs["knots"]],
        }
    return out


def _serialize_calendar_basis(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """JSON form of a :func:`_calendar_basis_state` dict (``t0`` as ISO text)."""
    if state is None:
        return None
    t0 = state.get("t0")
    return {
        "t0": None if t0 is None else str(pd.Timestamp(t0)),
        "span_sec": float(state["span_sec"]),
        "span_days": float(state["span_days"]),
        "n_harmonics": int(state["n_harmonics"]),
        "n_cols": int(state["n_cols"]),
    }


def _deserialize_calendar_basis(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Inverse of :func:`_serialize_calendar_basis`; ``None`` for old archives."""
    if payload is None:
        return None
    t0_raw = payload.get("t0")
    return {
        "t0": None if t0_raw is None else pd.Timestamp(t0_raw),
        "span_sec": float(payload["span_sec"]),
        "span_days": float(payload["span_days"]),
        "n_harmonics": int(payload["n_harmonics"]),
        "n_cols": int(payload["n_cols"]),
    }


def _deserialize_met_state(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"met_head": str(payload["met_head"])}
    if "degree" in payload:
        out["degree"] = int(payload["degree"])
    if "stats" in payload:
        mu, sd = payload["stats"]
        out["stats"] = (np.asarray(mu, dtype=float), np.asarray(sd, dtype=float))
    if "gam_state" in payload:
        gs = payload["gam_state"]
        out["gam_state"] = {
            "mu": np.asarray(gs["mu"], dtype=float),
            "sd": np.asarray(gs["sd"], dtype=float),
            "knots": [np.asarray(k, dtype=float) for k in gs["knots"]],
        }
    return out


class SeparableWorldModel:
    """Bijective separable latent atmospheric world model."""

    def __init__(self, config: WorldModelConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            self.config = WorldModelConfig(**kwargs)
        else:
            if kwargs:
                cfg_d = dict(config.__dict__)
                cfg_d.update(kwargs)
                self.config = WorldModelConfig(**cfg_d)
            else:
                self.config = config

        if self.config.backend not in ("separable", "torch"):
            raise ValueError(
                f"backend must be 'separable' or 'torch', got {self.config.backend!r}."
            )
        if self.config.met_head not in ("gam", "linear", "poly", "tree"):
            raise ValueError(
                f"met_head must be 'gam', 'linear', 'poly', or 'tree', got {self.config.met_head!r}."
            )
        if self.config.coordinate_order not in ("auto", "warp_then_mix", "mix_then_warp"):
            raise ValueError(
                "coordinate_order must be 'auto', 'warp_then_mix', or 'mix_then_warp', "
                f"got {self.config.coordinate_order!r}."
            )

        self._species: list[str] = []
        self._met_cols: list[str] = []
        self._coord_order: str = "warp_then_mix"
        self._gammas: np.ndarray = np.empty(0, dtype=float)
        self._z_mu: np.ndarray = np.empty(0, dtype=float)
        self._z_sd: np.ndarray = np.empty(0, dtype=float)
        self._W: np.ndarray = np.empty((0, 0), dtype=float)
        self._W_inv: np.ndarray = np.empty((0, 0), dtype=float)

        self._met_design_state: dict[str, Any] = {}
        self._cal_stats: tuple[np.ndarray, np.ndarray] = (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )
        self._coef_cal: np.ndarray = np.empty((0, 0), dtype=float)
        self._coef_met: np.ndarray = np.empty((0, 0), dtype=float)
        self._intercept: np.ndarray = np.empty(0, dtype=float)
        self._x_mu: np.ndarray = np.empty(0, dtype=float)
        self._x_sd: np.ndarray = np.empty(0, dtype=float)
        self._n_cal: int = 0
        self._cal_basis: dict[str, Any] | None = None
        self._phi: np.ndarray = np.empty(0, dtype=float)
        self._train_resid_centered: np.ndarray = np.empty((0, 0), dtype=float)

        self._tree_model: Any = None
        self._tree_met_mean: np.ndarray = np.empty(0, dtype=float)
        self._torch_net: Any = None
        self._torch_device: str = "cpu"
        self._train_met_mean_latent: np.ndarray = np.empty(0, dtype=float)
        self._is_fitted: bool = False
        self.fit_metadata_: dict[str, Any] = {}

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("SeparableWorldModel must be fitted via .fit() before use.")

    def _calendar_basis(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Calendar design for ``index`` on the basis pinned at fit time.

        The basis width (quadratic past 90 d, annual past 180 d) and the trend
        normalisation both come from the fitted window, so any evaluation
        window -- a rollout episode, a normalisation subset, a persistence
        pre-window -- gets the same design the coefficients were fitted on.
        """
        state = self._cal_basis
        if state is None:
            state = self._legacy_calendar_basis(index)
        return _build_calendar_basis(index, n_harmonics=self.config.n_harmonics, state=state)

    def _legacy_calendar_basis(self, index: pd.DatetimeIndex) -> dict[str, Any]:
        """Best-effort basis for an archive saved before the spec was serialised.

        The fit epoch and span are not recoverable from a v1 payload that
        predates ``calendar_basis``, so the evaluation window keeps its own epoch
        and span; only the 90 d / 180 d bucket is inferred back from the fitted
        column count, which keeps the design width (and so the model runnable)
        without inventing a span. Re-``save()`` such an archive to pin the spec.
        """
        state = _calendar_basis_state(index, n_harmonics=self.config.n_harmonics)
        width = int(self._n_cal)
        if width <= 0:
            return state
        k = max(1, int(self.config.n_harmonics))
        if width == _calendar_basis_width(0.0, k):
            level = 0
        elif width == _calendar_basis_width(90.0, k):
            level = 1
        elif width == _calendar_basis_width(180.0, k):
            level = 2
        else:
            return state
        span_days = float(state["span_days"])
        if level == 0:
            span_days = min(span_days, 89.0)
        elif level == 1:
            span_days = min(max(span_days, 90.0), 179.0)
        else:
            span_days = max(span_days, 180.0)
        return {**state, "span_days": span_days, "n_cols": width}

    def _apply_power_encode(self, Y: np.ndarray) -> np.ndarray:
        Z = np.empty_like(Y, dtype=float)
        for m in range(Y.shape[1]):
            Z[:, m] = _shifted_power(Y[:, m], float(self._gammas[m]))
        return Z

    def _apply_power_decode(self, Z: np.ndarray) -> np.ndarray:
        Y = np.empty_like(Z, dtype=float)
        for m in range(Z.shape[1]):
            Y[:, m] = _inv_shifted_power(Z[:, m], float(self._gammas[m]))
        return Y

    def encode(self, Y: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Map physical observations ``(N, M)`` bijectively into latent coordinates."""
        self._require_fitted()
        Y_arr = np.asarray(Y, dtype=float)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr[:, None]
        if self.config.backend == "torch":
            return self._encode_torch(Y_arr)
        if self._coord_order == "mix_then_warp":
            C = Y_arr @ self._W.T
            Z = self._apply_power_encode(C)
            return (Z - self._z_mu[None, :]) / self._z_sd[None, :]
        Z = self._apply_power_encode(Y_arr)
        Zn = (Z - self._z_mu[None, :]) / self._z_sd[None, :]
        return Zn @ self._W.T

    def decode(self, S: np.ndarray) -> np.ndarray:
        """Map latent coordinates ``(N, M)`` inversely back to physical concentrations."""
        self._require_fitted()
        S_arr = np.asarray(S, dtype=float)
        if S_arr.ndim == 1:
            S_arr = S_arr[:, None]
        if self.config.backend == "torch":
            return self._decode_torch(S_arr)
        if self._coord_order == "mix_then_warp":
            Z = S_arr * self._z_sd[None, :] + self._z_mu[None, :]
            C = self._apply_power_decode(Z)
            return C @ self._W_inv.T
        Zn = S_arr @ self._W_inv.T
        Z = Zn * self._z_sd[None, :] + self._z_mu[None, :]
        return self._apply_power_decode(Z)

    def _build_met_features(
        self, df: pd.DataFrame, *, fit_state: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X_met_design, X_met_raw)`` aligned to ``df``."""
        X_raw = (
            df[self._met_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if self._met_cols
            else np.empty((len(df), 0), dtype=float)
        )
        if fit_state:
            design_head = "gam" if self.config.met_head == "tree" else self.config.met_head
            X_des, st = _build_met_design(
                df,
                self._met_cols,
                met_head=design_head,
                degree=self.config.met_degree,
                n_knots=self.config.n_knots,
            )
            self._met_design_state = st
            return X_des, X_raw
        X_des, _ = _build_met_design(df, self._met_cols, state=self._met_design_state)
        return X_des, X_raw

    def _predict_latent_components(
        self, X_cal: np.ndarray, X_met: np.ndarray, X_met_raw: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(base_cal, met_effect)`` in latent space, zero-centered on training met."""
        if self.config.backend == "torch":
            return self._predict_latent_torch(X_cal, X_met)
        if self.config.met_head == "tree" and self._tree_model is not None:
            cal_mu, cal_sd = self._cal_stats
            Zc = _standardize_apply(X_cal, cal_mu, cal_sd)
            base_cal = Zc @ self._coef_cal + self._intercept[None, :]
            if X_met_raw is not None and X_met_raw.shape[1] > 0:
                raw_fill = np.nan_to_num(X_met_raw, nan=0.0)
                preds = np.asarray(self._tree_model.predict(raw_fill), dtype=float)
                if preds.ndim == 1:
                    preds = preds[:, None]
                met_eff = preds - self._tree_met_mean[None, :]
            else:
                met_eff = np.zeros_like(base_cal)
            return base_cal, met_eff

        X_full = np.hstack([X_cal, X_met]) if X_met.shape[1] > 0 else X_cal
        X_std = _standardize_apply(X_full, self._x_mu, self._x_sd)
        base_cal = X_std[:, : self._n_cal] @ self._coef_cal + self._intercept[None, :]
        met_eff = (
            X_std[:, self._n_cal :] @ self._coef_met
            if X_met.shape[1] > 0
            else np.zeros_like(base_cal)
        )
        return base_cal, met_eff

    def fit(
        self,
        df: pd.DataFrame,
        target_cols: str | Sequence[str],
        met_cols: Sequence[str],
    ) -> SeparableWorldModel:
        """Fit bijective separable coordinate map and additive action dynamics."""
        df_s = _ensure_datetime_frame(df)
        tcols = [target_cols] if isinstance(target_cols, str) else list(target_cols)
        mcols = list(met_cols)
        if not tcols:
            raise ValueError("`target_cols` must be non-empty.")
        for c in [*tcols, *mcols]:
            if c not in df_s.columns:
                raise KeyError(f"Column {c!r} not found in DataFrame.")

        Y_df = df_s[tcols].apply(pd.to_numeric, errors="coerce")
        valid: npt.NDArray[np.bool_] = np.all(np.isfinite(Y_df.to_numpy(dtype=float)), axis=1)
        if mcols:
            X_m_chk = df_s[mcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            valid &= np.all(np.isfinite(X_m_chk), axis=1)
        df_tr = df_s.loc[valid]
        Y = Y_df.iloc[valid].to_numpy(dtype=float)
        n, m_dim = Y.shape
        if n < 15:
            raise ValueError(f"Need at least 15 finite rows to fit SeparableWorldModel, got {n}.")

        self._species = tcols
        self._met_cols = mcols

        idx_tr = pd.DatetimeIndex(df_tr.index)
        self._cal_basis = _calendar_basis_state(idx_tr, n_harmonics=self.config.n_harmonics)
        X_cal = _build_calendar_basis(idx_tr, state=self._cal_basis)
        X_met, X_met_raw = self._build_met_features(df_tr, fit_state=True)
        self._n_cal = X_cal.shape[1]

        if self.config.backend == "torch":
            self._coord_order = "warp_then_mix"
            X_met_torch = (
                _standardize_fit(X_met_raw)[0]
                if X_met_raw.shape[1] > 0
                else np.empty((len(Y), 0), dtype=float)
            )
            self._met_design_state = {
                "met_head": "linear",
                "stats": _standardize_fit(X_met_raw)[1:]
                if X_met_raw.shape[1] > 0
                else (np.empty(0), np.empty(0)),
            }
            self._fit_torch(Y, X_cal, X_met_torch)
            X_met = X_met_torch
        else:
            self._fit_separable_numpy(Y, X_cal, X_met, X_met_raw)

        self._is_fitted = True

        S_tr = self.encode(Y)
        base_tr, met_tr = self._predict_latent_components(X_cal, X_met, X_met_raw)
        R_tr = S_tr - (base_tr + met_tr)
        self._train_resid_centered = R_tr - np.mean(R_tr, axis=0, keepdims=True)
        phis = np.zeros(m_dim, dtype=float)
        if self.config.ar_persistence and n >= 16:
            for j in range(m_dim):
                a, b = R_tr[:-1, j], R_tr[1:, j]
                if float(np.std(a)) > 1e-12 and float(np.std(b)) > 1e-12:
                    phis[j] = float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 0.98))
        self._phi = phis
        self.fit_metadata_ = {
            "n_train": int(n),
            "backend": self.config.backend,
            "met_head": self.config.met_head,
            "coordinate_order": self._coord_order,
            "gammas": [float(g) for g in self._gammas],
            "latent_phi1": [float(p) for p in phis],
        }
        return self

    def _optimize_single_gammas(
        self,
        Y: np.ndarray,
        proj: _FoldProjector,
        *,
        has_met: bool,
    ) -> np.ndarray:
        from scipy.optimize import minimize_scalar

        m_dim = Y.shape[1]
        gammas = np.ones(m_dim, dtype=float)
        if self.config.fixed_gamma is not None:
            fg = float(self.config.fixed_gamma)
            for j in range(m_dim):
                gammas[j] = fg if float(np.min(Y[:, j])) >= -1e-9 else 1.0
            return gammas
        if not has_met:
            return gammas

        gamma_grid = (1.0, 0.75, 0.5, 0.25, 0.0)
        for j in range(m_dim):
            yj = Y[:, j]
            if float(np.min(yj)) < -1e-9:
                gammas[j] = 1.0
                continue

            def _loss_g(g: float, col: np.ndarray = yj) -> float:
                z = _shifted_power(col, g)[:, None]
                zn, _, _ = _standardize_fit(z)
                rc, het, fvu = _fast_oof_separability_proj(zn, proj)
                return self.config.lambda_rc * rc + self.config.lambda_het * het + 0.10 * fvu

            scores = [_loss_g(g) for g in gamma_grid]
            best_idx = int(np.argmin(scores))
            best_g = gamma_grid[best_idx]
            lo_b = max(0.0, best_g - 0.25)
            hi_b = min(1.0, best_g + 0.25)
            if hi_b > lo_b + 1e-6:
                opt = minimize_scalar(
                    _loss_g, bounds=(lo_b, hi_b), method="bounded", options={"maxiter": 10}
                )
                if opt.success and float(opt.fun) < scores[best_idx] - 1e-4:
                    best_g = float(opt.x)
            gammas[j] = float(best_g)
        return gammas

    def _fit_separable_numpy(
        self,
        Y: np.ndarray,
        X_cal: np.ndarray,
        X_met: np.ndarray,
        X_met_raw: np.ndarray,
    ) -> None:
        from scipy.optimize import minimize

        n, m_dim = Y.shape
        folds = _day_interleaved_folds(n, block_len=24, n_splits=3)
        proj = _build_fold_projector(X_cal, X_met, X_met_raw, folds)
        has_met = bool(X_met.shape[1] > 0)
        order_cfg = self.config.coordinate_order

        best_loss = float("inf")
        best_state: tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

        if order_cfg in ("auto", "warp_then_mix") or m_dim == 1 or not self.config.learn_mixing:
            gammas_w = self._optimize_single_gammas(Y, proj, has_met=has_met)
            Z_w = np.column_stack(
                [_shifted_power(Y[:, j], float(gammas_w[j])) for j in range(m_dim)]
            )
            Zn_w, z_mu_w, z_sd_w = _standardize_fit(Z_w)
            if m_dim >= 2 and self.config.learn_mixing and has_met:
                n_params = m_dim * (m_dim - 1)

                def _loss_wtm(theta: np.ndarray) -> float:
                    t_c = np.clip(theta, -3.0, 3.0)
                    W = _lu_matrix(t_c, m_dim)
                    S_try = Zn_w @ W.T
                    rc, het, fvu = _fast_oof_separability_proj(S_try, proj)
                    reg = self.config.lambda_reg * float(np.sum(t_c**2))
                    return (
                        self.config.lambda_rc * rc + self.config.lambda_het * het + 0.10 * fvu + reg
                    )

                theta0 = np.zeros(n_params, dtype=float)
                l0 = _loss_wtm(theta0)
                res = minimize(
                    _loss_wtm, theta0, method="Powell", options={"maxfev": 25 * m_dim, "ftol": 1e-4}
                )
                t_best = np.clip(res.x, -3.0, 3.0) if float(res.fun) < l0 else theta0
                W_w = _lu_matrix(t_best, m_dim)
                loss_wtm = float(min(l0, res.fun))
            else:
                W_w = np.eye(m_dim, dtype=float)
                rc0, het0, fvu0 = _fast_oof_separability_proj(Zn_w, proj)
                loss_wtm = self.config.lambda_rc * rc0 + self.config.lambda_het * het0 + 0.10 * fvu0
            best_loss = loss_wtm
            best_state = ("warp_then_mix", gammas_w, z_mu_w, z_sd_w, W_w)

        can_mtw = (
            order_cfg in ("auto", "mix_then_warp")
            and m_dim >= 2
            and self.config.learn_mixing
            and self.config.fixed_gamma is None
            and has_met
            and float(np.min(Y)) >= -1e-9
        )
        if can_mtw:
            n_params = m_dim * (m_dim - 1)

            def _eval_mtw_theta(theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
                t_pos = np.clip(theta, 0.0, 2.5)
                W = _lu_matrix(t_pos, m_dim)
                C = np.maximum(Y, 0.0) @ W.T
                g_m = np.ones(m_dim, dtype=float)
                Z_m = np.empty_like(C)
                for j in range(m_dim):
                    best_gj, best_lj, best_col = 1.0, float("inf"), C[:, j]
                    for g_cand in (1.0, 0.5, 0.0):
                        col_p = _shifted_power(C[:, j], g_cand)[:, None]
                        zn_j, _, _ = _standardize_fit(col_p)
                        rc_j, het_j, fvu_j = _fast_oof_separability_proj(zn_j, proj)
                        l_j = (
                            self.config.lambda_rc * rc_j
                            + self.config.lambda_het * het_j
                            + 0.10 * fvu_j
                        )
                        if l_j < best_lj:
                            best_lj, best_gj, best_col = l_j, g_cand, col_p[:, 0]
                    g_m[j] = best_gj
                    Z_m[:, j] = best_col
                Zn_m, _, _ = _standardize_fit(Z_m)
                rc_all, het_all, fvu_all = _fast_oof_separability_proj(Zn_m, proj)
                reg = self.config.lambda_reg * float(np.sum(t_pos**2))
                total = (
                    self.config.lambda_rc * rc_all
                    + self.config.lambda_het * het_all
                    + 0.10 * fvu_all
                    + reg
                )
                return float(total), W, g_m

            theta_zero = np.zeros(n_params, dtype=float)
            loss_z, W_z, _ = _eval_mtw_theta(theta_zero)
            theta0_m = np.full(n_params, 0.25, dtype=float)
            res_m = minimize(
                lambda th: _eval_mtw_theta(th)[0],
                theta0_m,
                method="Powell",
                options={"maxfev": 20 * m_dim, "ftol": 1e-4},
            )
            loss_opt, W_opt, _g_init = _eval_mtw_theta(res_m.x)
            if loss_opt < loss_z:
                loss_mtw, W_m = loss_opt, W_opt
            else:
                loss_mtw, W_m = loss_z, W_z
            if order_cfg == "mix_then_warp" or loss_mtw < best_loss - 1e-4:
                C_best = np.maximum(Y, 0.0) @ W_m.T
                g_refined = self._optimize_single_gammas(C_best, proj, has_met=has_met)
                Z_m_best = np.column_stack(
                    [_shifted_power(C_best[:, j], float(g_refined[j])) for j in range(m_dim)]
                )
                _, z_mu_m, z_sd_m = _standardize_fit(Z_m_best)
                best_state = ("mix_then_warp", g_refined, z_mu_m, z_sd_m, W_m)
                best_loss = loss_mtw

        assert best_state is not None
        coord_ord, gammas_f, z_mu_f, z_sd_f, W_f = best_state
        self._coord_order = coord_ord
        self._gammas = gammas_f
        self._z_mu = z_mu_f
        self._z_sd = z_sd_f
        self._W = W_f
        self._W_inv = np.linalg.inv(W_f)

        if coord_ord == "mix_then_warp":
            Z_fin = self._apply_power_encode(Y @ self._W.T)
            S = (Z_fin - self._z_mu[None, :]) / self._z_sd[None, :]
        else:
            Z_fin = self._apply_power_encode(Y)
            S = ((Z_fin - self._z_mu[None, :]) / self._z_sd[None, :]) @ self._W.T

        if self.config.met_head == "tree":
            self._fit_tree_action_head(S, X_cal, X_met_raw)
        else:
            X_full = np.hstack([X_cal, X_met]) if X_met.shape[1] > 0 else X_cal
            coef, y_mu, x_mu, x_sd, _ = _ridge_cv_fit(X_full, S, alphas=self.config.alphas)
            self._coef_cal = coef[: self._n_cal]
            self._coef_met = coef[self._n_cal :]
            self._intercept = y_mu
            self._x_mu = x_mu
            self._x_sd = x_sd

    def _fit_tree_action_head(
        self, S: np.ndarray, X_cal: np.ndarray, X_met_raw: np.ndarray
    ) -> None:
        from sklearn.ensemble import ExtraTreesRegressor

        _, cal_mu, cal_sd = _standardize_fit(X_cal)
        self._cal_stats = (cal_mu, cal_sd)
        c_coef, c_int, _, _ = _ridge_fit(X_cal, S, alpha=1.0)
        base0 = _ridge_predict(X_cal, c_coef, c_int, cal_mu, cal_sd)
        if X_met_raw.shape[1] == 0:
            self._coef_cal = c_coef
            self._intercept = c_int
            self._tree_model = None
            self._tree_met_mean = np.zeros(S.shape[1], dtype=float)
            return

        X_clean = np.nan_to_num(X_met_raw, nan=0.0)
        et = ExtraTreesRegressor(
            n_estimators=60,
            min_samples_leaf=5,
            random_state=self.config.seed,
            n_jobs=1,
        )
        R_cal = S - base0
        et.fit(X_clean, R_cal if S.shape[1] > 1 else R_cal[:, 0])
        g0 = np.asarray(et.predict(X_clean), dtype=float)
        if g0.ndim == 1:
            g0 = g0[:, None]
        g0_c = g0 - np.mean(g0, axis=0, keepdims=True)
        c_coef, c_int, _, _ = _ridge_fit(X_cal, S - g0_c, alpha=1.0)
        base1 = _ridge_predict(X_cal, c_coef, c_int, cal_mu, cal_sd)
        R_cal1 = S - base1
        et.fit(X_clean, R_cal1 if S.shape[1] > 1 else R_cal1[:, 0])
        g1 = np.asarray(et.predict(X_clean), dtype=float)
        if g1.ndim == 1:
            g1 = g1[:, None]

        self._coef_cal = c_coef
        self._intercept = c_int
        self._tree_model = et
        self._tree_met_mean = np.mean(g1, axis=0)

    def _fit_torch(self, Y: np.ndarray, X_cal: np.ndarray, X_met: np.ndarray) -> None:
        from ..utils._omp import import_torch

        torch = import_torch()
        import torch.nn as nn

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        dev_req = str(self.config.device).lower()
        if dev_req == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        else:
            dev = dev_req
        self._torch_device = dev

        n, m_dim = Y.shape
        can_power = np.array([float(np.min(Y[:, j])) >= -1e-9 for j in range(m_dim)], dtype=bool)
        if self.config.fixed_gamma is not None:
            init_g = np.clip(float(self.config.fixed_gamma), 1e-3, 1.0 - 1e-3)
            train_gamma = False
        else:
            init_g = 0.5
            train_gamma = True

        z0 = np.column_stack(
            [_shifted_power(Y[:, j], init_g if can_power[j] else 1.0) for j in range(m_dim)]
        )
        _, z_mu0, z_sd0 = _standardize_fit(z0)
        self._z_mu = z_mu0
        self._z_sd = z_sd0

        Zc_np, cal_mu, cal_sd = _standardize_fit(X_cal)
        self._cal_stats = (cal_mu, cal_sd)

        t_Y = torch.tensor(Y, dtype=torch.float32, device=dev)
        t_C = torch.tensor(Zc_np, dtype=torch.float32, device=dev)
        t_W = (
            torch.tensor(X_met, dtype=torch.float32, device=dev)
            if X_met.shape[1] > 0
            else torch.empty((n, 0), dtype=torch.float32, device=dev)
        )
        t_can_pow = torch.tensor(can_power, dtype=torch.bool, device=dev)
        t_z_mu = torch.tensor(z_mu0, dtype=torch.float32, device=dev)
        t_z_sd = torch.tensor(z_sd0, dtype=torch.float32, device=dev)

        hidden = int(self.config.torch_hidden)
        n_cal = Zc_np.shape[1]
        n_met = X_met.shape[1]
        learn_mix = bool(m_dim >= 2 and self.config.learn_mixing and n_met > 0)

        class _SeparableNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                init_logit = float(np.log(init_g / (1.0 - init_g)))
                self.theta_g = nn.Parameter(
                    torch.full((m_dim,), init_logit, dtype=torch.float32),
                    requires_grad=train_gamma,
                )
                n_tri = (m_dim * (m_dim - 1)) // 2
                self.theta_l = nn.Parameter(
                    torch.zeros(n_tri, dtype=torch.float32), requires_grad=learn_mix
                )
                self.theta_u = nn.Parameter(
                    torch.zeros(n_tri, dtype=torch.float32), requires_grad=learn_mix
                )
                self.cal_head = nn.Linear(n_cal, m_dim)
                if n_met > 0:
                    self.met_net: nn.Module | None = nn.Sequential(
                        nn.Linear(n_met, hidden),
                        nn.SiLU(),
                        nn.Linear(hidden, m_dim),
                    )
                else:
                    self.met_net = None

            def gammas_tensor(self) -> Any:
                g = torch.sigmoid(self.theta_g)
                return torch.where(t_can_pow, g, torch.ones_like(g))

            def mix_matrix(self) -> Any:
                if not learn_mix:
                    return torch.eye(m_dim, dtype=torch.float32, device=t_Y.device)
                L = torch.eye(m_dim, dtype=torch.float32, device=t_Y.device)
                U = torch.eye(m_dim, dtype=torch.float32, device=t_Y.device)
                k = 0
                for i in range(1, m_dim):
                    for j in range(i):
                        L[i, j] = torch.clamp(self.theta_l[k], -3.0, 3.0)
                        k += 1
                k = 0
                for i in range(m_dim):
                    for j in range(i + 1, m_dim):
                        U[i, j] = torch.clamp(self.theta_u[k], -3.0, 3.0)
                        k += 1
                return L @ U

            def encode_t(self, y_t: Any) -> Any:
                g = self.gammas_tensor()[None, :]
                x = torch.clamp(y_t, min=0.0) + 1.0
                log_x = torch.log(x)
                small = torch.abs(g) < 1e-4
                box = (torch.pow(x, g) - 1.0) / torch.where(small, torch.ones_like(g), g)
                taylor = log_x * (1.0 + 0.5 * g * log_x)
                pow_y = torch.where(small, taylor, box)
                z = torch.where(t_can_pow[None, :], pow_y, y_t)
                zn = (z - t_z_mu[None, :]) / t_z_sd[None, :]
                W_mat = self.mix_matrix()
                return zn @ W_mat.t()

            def forward_actions(self, c_t: Any, w_t: Any) -> tuple[Any, Any]:
                base = self.cal_head(c_t)
                if self.met_net is None or w_t.shape[1] == 0:
                    return base, torch.zeros_like(base)
                m_eff = self.met_net(w_t)
                m_eff = m_eff - m_eff.mean(dim=0, keepdim=True)
                return base, m_eff

        net = _SeparableNet().to(dev)
        n_epochs = max(10, int(self.config.torch_epochs))
        opt = torch.optim.AdamW(net.parameters(), lr=float(self.config.torch_lr), weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=max(float(self.config.torch_lr) * 0.05, 1e-4)
        )

        def _corr2_loss(A: Any, B: Any) -> Any:
            if B.shape[1] == 0:
                return torch.tensor(0.0, device=dev)
            Ac = A - A.mean(dim=0, keepdim=True)
            Bc = B - B.mean(dim=0, keepdim=True)
            An = Ac / torch.sqrt(torch.sum(Ac * Ac, dim=0, keepdim=True) + 1e-8)
            Bn = Bc / torch.sqrt(torch.sum(Bc * Bc, dim=0, keepdim=True) + 1e-8)
            corr = An.t() @ Bn
            return torch.mean(corr * corr)

        net.train()
        for _ in range(n_epochs):
            opt.zero_grad()
            S = net.encode_t(t_Y)
            base, m_eff = net.forward_actions(t_C, t_W)
            S_hat = base + m_eff
            R = S - S_hat
            mse_loss = torch.mean(R * R) / torch.clamp(torch.mean(S * S), min=1e-4)
            rc_loss = _corr2_loss(R, t_W)
            het_loss = _corr2_loss(R * R, t_W) + _corr2_loss(R * R, m_eff.detach())
            reg_loss = self.config.lambda_reg * (
                torch.sum(net.theta_l**2) + torch.sum(net.theta_u**2)
            )
            loss = (
                mse_loss
                + self.config.lambda_rc * rc_loss
                + self.config.lambda_het * het_loss
                + reg_loss
            )
            loss.backward()
            opt.step()
            sched.step()

        net.eval()
        with torch.no_grad():
            self._gammas = net.gammas_tensor().detach().cpu().numpy().astype(float)
            self._W = net.mix_matrix().detach().cpu().numpy().astype(float)
            self._W_inv = np.linalg.inv(self._W)
            if net.met_net is not None and t_W.shape[1] > 0:
                m_raw = net.met_net(t_W).detach().cpu().numpy().astype(float)
                self._train_met_mean_latent = m_raw.mean(axis=0)
            else:
                self._train_met_mean_latent = np.zeros(m_dim, dtype=float)
        self._torch_net = net

    def _encode_torch(self, Y: np.ndarray) -> np.ndarray:
        Z = self._apply_power_encode(Y)
        Zn = (Z - self._z_mu[None, :]) / self._z_sd[None, :]
        return Zn @ self._W.T

    def _decode_torch(self, S: np.ndarray) -> np.ndarray:
        Zn = S @ self._W_inv.T
        Z = Zn * self._z_sd[None, :] + self._z_mu[None, :]
        return self._apply_power_decode(Z)

    def _predict_latent_torch(
        self, X_cal: np.ndarray, X_met: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        from ..utils._omp import import_torch

        torch = import_torch()

        cal_mu, cal_sd = self._cal_stats
        Zc = _standardize_apply(X_cal, cal_mu, cal_sd)
        t_C = torch.tensor(Zc, dtype=torch.float32, device=self._torch_device)
        net = self._torch_net
        with torch.no_grad():
            base = net.cal_head(t_C).detach().cpu().numpy().astype(float)
            if net.met_net is not None and X_met.shape[1] > 0:
                t_W = torch.tensor(X_met, dtype=torch.float32, device=self._torch_device)
                m_raw = net.met_net(t_W).detach().cpu().numpy().astype(float)
                m_eff = m_raw - self._train_met_mean_latent[None, :]
            else:
                m_eff = np.zeros_like(base)
        return base, m_eff

    def audit(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run factual separability audit in raw vs learned latent coordinates."""
        self._require_fitted()
        df_s = _ensure_datetime_frame(df)
        Y_df = df_s[self._species].apply(pd.to_numeric, errors="coerce")
        valid: npt.NDArray[np.bool_] = np.all(np.isfinite(Y_df.to_numpy(dtype=float)), axis=1)
        if self._met_cols:
            X_chk = df_s[self._met_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            valid &= np.all(np.isfinite(X_chk), axis=1)
        df_v = df_s.loc[valid]
        Y = Y_df.iloc[valid].to_numpy(dtype=float)

        X_cal = self._calendar_basis(pd.DatetimeIndex(df_v.index))
        X_met, X_met_raw = self._build_met_features(df_v, fit_state=False)

        S = self.encode(Y)
        base_lat, met_lat = self._predict_latent_components(X_cal, X_met, X_met_raw)
        S_fact = base_lat + met_lat
        Y_fact = self.decode(S_fact)

        raw_per_sp: dict[str, dict[str, Any]] = {}
        lat_per_dim: dict[str, dict[str, Any]] = {}
        folds = _day_interleaved_folds(len(df_v), block_len=24, n_splits=3)
        for j, sp in enumerate(self._species):
            raw_per_sp[sp] = counterfactual_separability_audit(
                Y[:, j],
                Y_fact[:, j],
                X_met_raw,
                folds=folds,
                nonlinear=False,
                met_names=self._met_cols,
                n_boot=0,
                seed=self.config.seed,
            )
            lat_per_dim[f"latent_{j}"] = counterfactual_separability_audit(
                S[:, j],
                S_fact[:, j],
                X_met_raw,
                folds=folds,
                nonlinear=False,
                met_names=self._met_cols,
                n_boot=0,
                seed=self.config.seed,
            )

        raw_rho = float(np.nanmean([v["interaction_rho_lb_raw"] for v in raw_per_sp.values()]))
        lat_rho = float(np.nanmean([v["interaction_rho_lb_raw"] for v in lat_per_dim.values()]))
        raw_rc = float(np.nanmean([v["rc_score_linear"] for v in raw_per_sp.values()]))
        lat_rc = float(np.nanmean([v["rc_score_linear"] for v in lat_per_dim.values()]))
        raw_w2 = float(np.nanmean([v["interaction_w2_lb_raw"] for v in raw_per_sp.values()]))
        lat_w2 = float(np.nanmean([v["interaction_w2_lb_raw"] for v in lat_per_dim.values()]))

        gain = {
            "mean_rho_lb_raw": raw_rho,
            "mean_rho_lb_latent": lat_rho,
            "rho_lb_reduction": float(raw_rho - lat_rho) if np.isfinite(raw_rho) else float("nan"),
            "mean_rc_raw": raw_rc,
            "mean_rc_latent": lat_rc,
            "mean_w2_raw": raw_w2,
            "mean_w2_latent": lat_w2,
        }
        return {"raw_audit": raw_per_sp, "latent_audit": lat_per_dim, "separability_gain": gain}

    def operating_window(
        self,
        df: pd.DataFrame,
        *,
        horizons: Sequence[int] = (1, 6, 12, 24, 48, 72),
        max_rc: float = 0.10,
        max_rho_lb: float = 0.15,
    ) -> pd.DataFrame:
        """Audit predictive accuracy and factual separability across rollout lead horizons."""
        self._require_fitted()
        df_s = _ensure_datetime_frame(df)
        Y = df_s[self._species].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        fin: npt.NDArray[np.bool_] = np.all(np.isfinite(Y), axis=1)
        if self._met_cols:
            Xm_chk = (
                df_s[self._met_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            )
            fin &= np.all(np.isfinite(Xm_chk), axis=1)
        df_v = df_s.loc[fin]
        Yv = Y[fin]
        n = len(df_v)

        X_cal = self._calendar_basis(pd.DatetimeIndex(df_v.index))
        X_met, X_met_raw = self._build_met_features(df_v, fit_state=False)
        base_lat, met_lat = self._predict_latent_components(X_cal, X_met, X_met_raw)
        S_fact = base_lat + met_lat
        S_obs = self.encode(Yv)
        R_lat = S_obs - S_fact

        rows: list[dict[str, Any]] = []
        for h_raw in horizons:
            h = int(h_raw)
            if h < 1 or h >= n - 15:
                continue
            decay = (self._phi**h)[None, :]
            S_pred_h = S_fact[h:] + decay * R_lat[:-h]
            Y_pred_h = self.decode(S_pred_h)
            yo = Yv[h:, 0]
            yf = Y_pred_h[:, 0]
            Xm_h = X_met_raw[h:]

            rmse = float(np.sqrt(np.mean((yo - yf) ** 2)))
            var_y = float(np.var(yo))
            r2 = float(1.0 - np.mean((yo - yf) ** 2) / var_y) if var_y > 1e-12 else float("nan")
            audit_h = (
                counterfactual_separability_audit(
                    yo,
                    yf,
                    Xm_h,
                    folds=_day_interleaved_folds(len(yo), block_len=24, n_splits=3),
                    nonlinear=False,
                    met_names=self._met_cols,
                    n_boot=0,
                    seed=self.config.seed,
                )
                if Xm_h.shape[1] > 0
                else {}
            )
            rc = float(audit_h.get("rc_score_linear", 0.0))
            rho_raw = float(audit_h.get("interaction_rho_lb_raw", 0.0))
            rho_log = float(audit_h.get("interaction_rho_lb_log", 0.0))
            sep = bool(
                np.isfinite(rc) and rc <= max_rc and np.isfinite(rho_raw) and rho_raw <= max_rho_lb
            )
            rows.append(
                {
                    "horizon": h,
                    "n_obs": len(yo),
                    "rmse": rmse,
                    "r2": r2,
                    "rc_score_linear": rc,
                    "interaction_rho_lb_raw": rho_raw,
                    "interaction_rho_lb_log": rho_log,
                    "separable": sep,
                }
            )
        cols = [
            "horizon",
            "n_obs",
            "rmse",
            "r2",
            "rc_score_linear",
            "interaction_rho_lb_raw",
            "interaction_rho_lb_log",
            "separable",
        ]
        return pd.DataFrame(rows, columns=cols)

    def normalise(
        self,
        df: pd.DataFrame,
        *,
        n_samples: int = 100,
        paired: bool = True,
        pool_df: pd.DataFrame | None = None,
        quantiles: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> WorldModelNormaliseResult:
        """De-weather all modeled species via counterfactual weather resampling in latent space."""
        self._require_fitted()
        df_s = _ensure_datetime_frame(df)
        n = len(df_s)
        pool_src = _ensure_datetime_frame(pool_df) if pool_df is not None else df_s

        Y_df = df_s[self._species].apply(pd.to_numeric, errors="coerce")
        Y_arr = Y_df.to_numpy(dtype=float)
        fin_y = np.all(np.isfinite(Y_arr), axis=1)

        X_cal = self._calendar_basis(pd.DatetimeIndex(df_s.index))
        X_met_fact, X_met_raw_fact = self._build_met_features(df_s, fit_state=False)
        X_met_pool, X_met_raw_pool = self._build_met_features(pool_src, fit_state=False)

        base_lat, met_fact = self._predict_latent_components(X_cal, X_met_fact, X_met_raw_fact)
        S_fact = base_lat + met_fact
        Y_fact = self.decode(S_fact)

        S_obs = np.full_like(S_fact, np.nan)
        if np.any(fin_y):
            S_obs[fin_y] = self.encode(Y_arr[fin_y])

        _, met_pool = self._predict_latent_components(
            np.zeros((len(pool_src), self._n_cal), dtype=float), X_met_pool, X_met_raw_pool
        )
        rng = np.random.default_rng(self.config.seed if seed is None else int(seed))
        m_dim = len(self._species)
        ns = max(2, int(n_samples))
        draws_phys = np.full((ns, n, m_dim), np.nan, dtype=float)
        resid_lat = S_obs - S_fact

        for k in range(ns):
            idx_draw = rng.integers(0, len(pool_src), size=n)
            S_cf = base_lat + met_pool[idx_draw]
            if paired:
                S_k = S_cf + resid_lat
                valid_k = np.all(np.isfinite(S_k), axis=1)
                if np.any(valid_k):
                    draws_phys[k, valid_k] = self.decode(S_k[valid_k])
            else:
                draws_phys[k] = self.decode(S_cf)

        mean_phys = np.nanmean(draws_phys, axis=0)
        out_cols: dict[str, np.ndarray] = {}
        if m_dim == 1:
            out_cols["observed"] = Y_arr[:, 0]
            out_cols["normalised"] = mean_phys[:, 0]
            out_cols["factual"] = Y_fact[:, 0]
            if quantiles:
                for q in quantiles:
                    q_tag = f"q{int(round(1000 * float(q))):03d}"
                    out_cols[q_tag] = np.nanquantile(draws_phys[:, :, 0], float(q), axis=0)
        else:
            for j, sp in enumerate(self._species):
                out_cols[f"{sp}_observed"] = Y_arr[:, j]
                out_cols[f"{sp}_normalised"] = mean_phys[:, j]
                out_cols[f"{sp}_factual"] = Y_fact[:, j]
                if quantiles:
                    for q in quantiles:
                        q_tag = f"{sp}_q{int(round(1000 * float(q))):03d}"
                        out_cols[q_tag] = np.nanquantile(draws_phys[:, :, j], float(q), axis=0)

        out_df = pd.DataFrame(out_cols, index=df_s.index)
        audit_res = self.audit(df_s)
        return WorldModelNormaliseResult(
            df=out_df,
            species=list(self._species),
            latent_audit=audit_res["latent_audit"],
            raw_audit=audit_res["raw_audit"],
            separability_gain=audit_res["separability_gain"],
            metadata={
                **self.fit_metadata_,
                "n_samples": ns,
                "paired": bool(paired),
            },
        )

    def counterfactual_rollout(
        self,
        df: pd.DataFrame,
        *,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        weather_scenario: pd.DataFrame | None = None,
        met_shifts: Mapping[str, float | tuple[str, float]] | None = None,
        disturbance_scale: float = 1.0,
        species_scale: Mapping[str, float] | None = None,
        ar_carryover: bool = True,
        n_boot: int = 0,
        block_hours: int = 24,
        seed: int | None = None,
    ) -> TrajectoryRolloutResult:
        """Execute a trajectory-level What-If rollout over ``[start, end]``."""
        self._require_fitted()
        df_s = _ensure_datetime_frame(df)
        idx = pd.DatetimeIndex(df_s.index)

        def _align_ts(val: str | pd.Timestamp) -> pd.Timestamp:
            ts = pd.Timestamp(val)
            if idx.tz is None:
                return ts.tz_localize(None) if ts.tzinfo is not None else ts
            return ts.tz_localize(idx.tz) if ts.tzinfo is None else ts.tz_convert(idx.tz)

        ep_s = _align_ts(start)
        ep_e = _align_ts(end)
        if ep_e < ep_s:
            raise ValueError(f"`end` ({ep_e}) precedes `start` ({ep_s}).")

        ep_mask = np.asarray((idx >= ep_s) & (idx <= ep_e), dtype=bool)
        if not np.any(ep_mask):
            raise ValueError("No timestamps fall inside the requested rollout window.")

        df_ep = df_s.iloc[ep_mask].copy()
        n_ep = len(df_ep)
        Y_ep = df_ep[self._species].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        X_cal_ep = self._calendar_basis(pd.DatetimeIndex(df_ep.index))
        X_met_fact_ep, X_met_raw_fact_ep = self._build_met_features(df_ep, fit_state=False)

        df_met_cf = df_ep[self._met_cols].copy()
        if weather_scenario is not None:
            scen = weather_scenario.copy()
            ep_dti = pd.DatetimeIndex(df_ep.index)
            if (
                isinstance(scen.index, pd.DatetimeIndex)
                and len(scen.index.intersection(ep_dti)) == n_ep
            ):
                scen_al = scen.reindex(ep_dti)
                for c in self._met_cols:
                    if c in scen_al.columns:
                        df_met_cf[c] = pd.to_numeric(scen_al[c], errors="coerce").to_numpy(
                            dtype=float
                        )
            elif len(scen) >= n_ep:
                for c in self._met_cols:
                    if c in scen.columns:
                        df_met_cf[c] = pd.to_numeric(scen[c], errors="coerce").to_numpy(
                            dtype=float
                        )[:n_ep]
            else:
                raise ValueError("`weather_scenario` must overlap or match rollout length.")

        if met_shifts:
            for col, spec in met_shifts.items():
                if col not in df_met_cf.columns:
                    raise KeyError(f"Shift column {col!r} not in met_cols {self._met_cols}.")
                cur = pd.to_numeric(df_met_cf[col], errors="coerce").to_numpy(dtype=float)
                if isinstance(spec, tuple):
                    op, val = spec
                    if op == "*":
                        df_met_cf[col] = cur * float(val)
                    elif op == "+":
                        df_met_cf[col] = cur + float(val)
                    else:
                        raise ValueError(f"Unsupported met shift op {op!r}; expected '*' or '+'.")
                else:
                    df_met_cf[col] = cur + float(spec)

        X_met_cf_ep, X_met_raw_cf_ep = self._build_met_features(df_met_cf, fit_state=False)

        S_obs = self.encode(Y_ep)
        base_ep, met_fact_ep = self._predict_latent_components(
            X_cal_ep, X_met_fact_ep, X_met_raw_fact_ep
        )
        S_fact_ep = base_ep + met_fact_ep
        R_ep = S_obs - S_fact_ep

        _, met_cf_ep = self._predict_latent_components(X_cal_ep, X_met_cf_ep, X_met_raw_cf_ep)
        S_cf_ep = base_ep + met_cf_ep + float(disturbance_scale) * R_ep

        pre_mask = np.asarray(idx < ep_s, dtype=bool)
        if ar_carryover and abs(float(disturbance_scale)) < 1e-9 and np.any(pre_mask):
            df_pre = df_s.iloc[pre_mask].tail(24)
            Y_pre = (
                df_pre[self._species].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            )
            fin_pre: npt.NDArray[np.bool_] = np.all(np.isfinite(Y_pre), axis=1)
            if np.any(fin_pre):
                df_pre = df_pre.loc[fin_pre]
                Y_pre = Y_pre[fin_pre]
                X_cal_pre = self._calendar_basis(pd.DatetimeIndex(df_pre.index))
                X_met_pre, X_met_raw_pre = self._build_met_features(df_pre, fit_state=False)
                b_pre, m_pre = self._predict_latent_components(X_cal_pre, X_met_pre, X_met_raw_pre)
                r_tail = self.encode(Y_pre) - (b_pre + m_pre)
                r0 = np.mean(r_tail[-min(6, len(r_tail)) :], axis=0)
                steps = np.arange(1, n_ep + 1, dtype=float)[:, None]
                for j, phi_j in enumerate(self._phi):
                    if phi_j >= self.config.carryover_min_phi:
                        S_cf_ep[:, j] += r0[j] * (phi_j ** steps[:, 0])

        if species_scale:
            Y_base_phys = self.decode(S_fact_ep)
            Y_scaled_phys = Y_base_phys.copy()
            for sp, fac in species_scale.items():
                if sp not in self._species:
                    raise KeyError(f"Species {sp!r} not in modeled species {self._species}.")
                j = self._species.index(sp)
                Y_scaled_phys[:, j] = np.maximum(0.0, Y_base_phys[:, j] * float(fac))
            delta_s = self.encode(Y_scaled_phys) - self.encode(Y_base_phys)
            S_cf_ep = S_cf_ep + delta_s

        Y_cf_ep = self.decode(S_cf_ep)

        p10_df: pd.DataFrame | None = None
        p90_df: pd.DataFrame | None = None
        if int(n_boot) > 0 and len(self._train_resid_centered) >= 4:
            n_tr, m_dim = self._train_resid_centered.shape
            b_len = max(1, min(int(block_hours), n_tr // 2))
            n_starts = n_tr - b_len + 1
            n_blocks = int(np.ceil(n_ep / b_len))
            rng_b = np.random.default_rng(self.config.seed if seed is None else int(seed))
            nb = max(10, int(n_boot))
            boot_phys = np.empty((nb, n_ep, m_dim), dtype=float)
            offsets = np.arange(b_len)[None, :]
            for bi in range(nb):
                starts = rng_b.integers(0, n_starts, size=n_blocks)[:, None]
                r_idx = (starts + offsets).ravel()[:n_ep]
                boot_phys[bi] = self.decode(S_cf_ep + self._train_resid_centered[r_idx])
            p10_df = pd.DataFrame(
                np.nanquantile(boot_phys, 0.10, axis=0), index=df_ep.index, columns=self._species
            )
            p90_df = pd.DataFrame(
                np.nanquantile(boot_phys, 0.90, axis=0), index=df_ep.index, columns=self._species
            )

        actual_df = pd.DataFrame(Y_ep, index=df_ep.index, columns=self._species)
        cf_df = pd.DataFrame(Y_cf_ep, index=df_ep.index, columns=self._species)
        diff_df = actual_df - cf_df
        lat_cols = [f"latent_{j}" for j in range(len(self._species))]
        dist_df = pd.DataFrame(R_ep, index=df_ep.index, columns=lat_cols)

        audit_full = self.audit(df_s)
        gain = audit_full["separability_gain"]
        cert = {
            "latent_rc_score": gain["mean_rc_latent"],
            "latent_rho_lb": gain["mean_rho_lb_latent"],
            "raw_rho_lb": gain["mean_rho_lb_raw"],
            "rho_lb_reduction": gain["rho_lb_reduction"],
            "reliable": bool(
                np.isfinite(gain["mean_rc_latent"])
                and gain["mean_rc_latent"] <= 0.12
                and gain["mean_rho_lb_latent"] <= 0.20
            ),
        }
        primary = self._species[0]
        m_act = float(actual_df[primary].mean())
        m_cf = float(cf_df[primary].mean())
        summary: dict[str, Any] = {
            "primary_species": primary,
            "n_steps": int(n_ep),
            "mean_actual": m_act,
            "mean_counterfactual": m_cf,
            "mean_diff": float(m_act - m_cf),
            "pct_change": float(100.0 * (m_cf - m_act) / max(abs(m_act), 1e-12)),
            "peak_actual": float(actual_df[primary].max()),
            "peak_counterfactual": float(cf_df[primary].max()),
        }
        return TrajectoryRolloutResult(
            start=pd.Timestamp(df_ep.index[0]),
            end=pd.Timestamp(df_ep.index[-1]),
            species=list(self._species),
            actual=actual_df,
            counterfactual=cf_df,
            difference=diff_df,
            latent_disturbance=dist_df,
            separability_certificate=cert,
            summary_metrics=summary,
            counterfactual_p10=p10_df,
            counterfactual_p90=p90_df,
        )

    def save(self, path: str | Path) -> Path:
        """Persist a fitted separable world model to a gzip-compressed JSON file."""
        self._require_fitted()
        if self.config.backend == "torch" or self.config.met_head == "tree":
            raise NotImplementedError(
                "JSON persistence supports pure-NumPy/GAM/linear/poly heads only "
                f"(got backend={self.config.backend!r}, met_head={self.config.met_head!r})."
            )
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_d = dict(self.config.__dict__)
        cfg_d["alphas"] = list(self.config.alphas)
        payload = {
            "version": 1,
            "config": cfg_d,
            "species": self._species,
            "met_cols": self._met_cols,
            "coord_order": self._coord_order,
            "gammas": self._gammas.tolist(),
            "z_mu": self._z_mu.tolist(),
            "z_sd": self._z_sd.tolist(),
            "W": self._W.tolist(),
            "W_inv": self._W_inv.tolist(),
            "met_design_state": _serialize_met_state(self._met_design_state),
            "coef_cal": self._coef_cal.tolist(),
            "coef_met": self._coef_met.tolist(),
            "intercept": self._intercept.tolist(),
            "x_mu": self._x_mu.tolist(),
            "x_sd": self._x_sd.tolist(),
            "n_cal": int(self._n_cal),
            "calendar_basis": _serialize_calendar_basis(self._cal_basis),
            "phi": self._phi.tolist(),
            "train_resid_centered": self._train_resid_centered.tolist(),
            "fit_metadata": self.fit_metadata_,
        }
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, allow_nan=False)
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> SeparableWorldModel:
        """Reload a fitted :class:`SeparableWorldModel` saved with :meth:`save`."""
        with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        if int(payload.get("version", -1)) != 1:
            raise ValueError(f"Unsupported world model archive version: {payload.get('version')!r}")
        cfg_raw = dict(payload["config"])
        cfg_raw["alphas"] = tuple(float(a) for a in cfg_raw["alphas"])
        model = cls(WorldModelConfig(**cfg_raw))
        model._species = [str(s) for s in payload["species"]]
        model._met_cols = [str(c) for c in payload["met_cols"]]
        model._coord_order = str(payload["coord_order"])
        model._gammas = np.asarray(payload["gammas"], dtype=float)
        model._z_mu = np.asarray(payload["z_mu"], dtype=float)
        model._z_sd = np.asarray(payload["z_sd"], dtype=float)
        model._W = np.asarray(payload["W"], dtype=float)
        model._W_inv = np.asarray(payload["W_inv"], dtype=float)
        model._met_design_state = _deserialize_met_state(payload["met_design_state"])
        model._coef_cal = np.asarray(payload["coef_cal"], dtype=float)
        model._coef_met = np.asarray(payload["coef_met"], dtype=float)
        model._intercept = np.asarray(payload["intercept"], dtype=float)
        model._x_mu = np.asarray(payload["x_mu"], dtype=float)
        model._x_sd = np.asarray(payload["x_sd"], dtype=float)
        model._n_cal = int(payload["n_cal"])
        # Optional: absent from archives written before the basis was serialised.
        model._cal_basis = _deserialize_calendar_basis(payload.get("calendar_basis"))
        model._phi = np.asarray(payload["phi"], dtype=float)
        model._train_resid_centered = np.asarray(payload["train_resid_centered"], dtype=float)
        model.fit_metadata_ = dict(payload.get("fit_metadata", {}))
        model._is_fitted = True
        return model


def fit_world_model(
    df: pd.DataFrame,
    target_cols: str | Sequence[str],
    met_cols: Sequence[str],
    *,
    config: WorldModelConfig | None = None,
    **kwargs: Any,
) -> SeparableWorldModel:
    """Convenience constructor fitting :class:`SeparableWorldModel`."""
    return SeparableWorldModel(config=config, **kwargs).fit(df, target_cols, met_cols)
