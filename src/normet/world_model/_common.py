"""Shared numerical, calendar, ridge, and coordinate utilities for ``normet.world_model``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _ensure_datetime_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a time-sorted copy with a valid :class:`pandas.DatetimeIndex`."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "date" in out.columns:
            out = out.set_index(pd.to_datetime(out["date"]))
        else:
            raise ValueError("Input DataFrame must have a DatetimeIndex or a 'date' column.")
    if out.index.hasnans:
        raise ValueError("Input DataFrame index contains NaT timestamps.")
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    return out


def _calendar_basis_width(span_days: float, n_harmonics: int) -> int:
    """Number of columns :func:`_build_calendar_basis` gives a span."""
    k = max(1, int(n_harmonics))
    width = 3 + 2 * k  # trend + diurnal harmonics + weekly pair
    if float(span_days) >= 90.0:
        width += 1  # quadratic trend
    if float(span_days) >= 180.0:
        width += 2 * k  # annual harmonics
    return width


def _align_timestamp(ts: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """``ts`` on ``index``'s clock, so a stored epoch subtracts cleanly."""
    tz = index.tz
    if tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def _calendar_basis_state(index: pd.DatetimeIndex, *, n_harmonics: int = 2) -> dict[str, Any]:
    """The calendar basis a window pins: epoch, span, bucket and design width.

    :func:`_build_calendar_basis` derives both its *width* (the quadratic and
    annual columns appear only past 90 d / 180 d) and its *trend scale* from
    the span of the frame it is handed. A model fitted on one window and
    evaluated on another therefore has to reuse the **fit** window's spec, or
    the design width changes (broadcast ``ValueError``) and the fitted trend
    coefficient is applied on a different scale. Capture the spec at fit time
    with this function and pass it back as ``state`` on every later call.
    """
    h = max(1, int(n_harmonics))
    n = len(index)
    if n == 0:
        return {
            "t0": None,
            "span_sec": 3600.0,
            "span_days": 3600.0 / 86400.0,
            "n_harmonics": h,
            "n_cols": 0,
        }
    t_sec = (index - index[0]).total_seconds().to_numpy(dtype=np.float64)
    span_sec = max(float(t_sec[-1]) if n > 1 else 1.0, 3600.0)
    span_days = span_sec / 86400.0
    return {
        "t0": pd.Timestamp(index[0]),
        "span_sec": span_sec,
        "span_days": span_days,
        "n_harmonics": h,
        "n_cols": _calendar_basis_width(span_days, h),
    }


def _build_calendar_basis(
    index: pd.DatetimeIndex,
    *,
    n_harmonics: int = 2,
    state: dict[str, Any] | None = None,
) -> np.ndarray:
    """Smooth trend + diurnal/weekly (and annual when span >= 180d) harmonic basis.

    With ``state`` (a :func:`_calendar_basis_state` dict) the basis is pinned to
    a fitted window's epoch, span and column set, so an evaluation window of any
    length gets exactly the design the model was fitted on.
    """
    n = len(index)
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if state is None:
        state = _calendar_basis_state(index, n_harmonics=n_harmonics)
    k_max = max(1, int(state["n_harmonics"]))
    span_sec = float(state["span_sec"])
    span_days = float(state["span_days"])
    ref_raw = state.get("t0")
    ref = index[0] if ref_raw is None else _align_timestamp(pd.Timestamp(ref_raw), index)
    t_sec = (index - ref).total_seconds().to_numpy(dtype=np.float64)
    t_norm = t_sec / span_sec - 0.5

    hr = index.hour.to_numpy(dtype=np.float64) + index.minute.to_numpy(dtype=np.float64) / 60.0
    doy = index.dayofyear.to_numpy(dtype=np.float64)
    dw = index.dayofweek.to_numpy(dtype=np.float64)

    cols: list[np.ndarray] = [t_norm]
    if span_days >= 90.0:
        cols.append(t_norm**2)
    for k in range(1, k_max + 1):
        ang_d = 2.0 * np.pi * k * hr / 24.0
        cols.extend([np.sin(ang_d), np.cos(ang_d)])
        if span_days >= 180.0:
            ang_a = 2.0 * np.pi * k * doy / 365.25
            cols.extend([np.sin(ang_a), np.cos(ang_a)])
    ang_w = 2.0 * np.pi * dw / 7.0
    cols.extend([np.sin(ang_w), np.cos(ang_w)])
    return np.column_stack(cols).astype(float)


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(X, dtype=float)
    if arr.shape[1] == 0:
        return arr, np.empty(0, dtype=float), np.empty(0, dtype=float)
    mu = np.nanmean(arr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    filled = np.where(np.isfinite(arr), arr, mu[None, :])
    sd = np.std(filled, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (filled - mu[None, :]) / sd[None, :], mu, sd


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.shape[1] == 0:
        return arr
    filled = np.where(np.isfinite(arr), arr, mu[None, :])
    return (filled - mu[None, :]) / sd[None, :]


def _ridge_fit(
    X: np.ndarray, Y: np.ndarray, *, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multi-output ridge regression with feature standardization and unpenalized intercept."""
    Y_arr = np.asarray(Y, dtype=float)
    squeeze = Y_arr.ndim == 1
    if squeeze:
        Y_arr = Y_arr[:, None]
    Z, mu, sd = _standardize_fit(X)
    y_mu = np.mean(Y_arr, axis=0)
    Yc = Y_arr - y_mu[None, :]
    p = Z.shape[1]
    if p == 0:
        coef = np.zeros((0, Y_arr.shape[1]), dtype=float)
        return (coef[:, 0] if squeeze else coef), y_mu, mu, sd
    gram = Z.T @ Z + max(float(alpha), 1e-8) * np.eye(p, dtype=float)
    rhs = Z.T @ Yc
    try:
        coef = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    return (coef[:, 0] if squeeze else coef), y_mu, mu, sd


def _ridge_predict(
    X: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    if X.shape[1] == 0:
        if intercept.ndim == 0 or len(intercept) == 1:
            return np.full(len(X), float(np.ravel(intercept)[0]), dtype=float)
        return np.repeat(intercept[None, :], len(X), axis=0)
    Z = _standardize_apply(X, mu, sd)
    return Z @ coef + intercept


def _ridge_cv_fit(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    alphas: Sequence[float] = (1e-2, 1e-1, 1.0, 10.0, 100.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Select multi-output ridge penalty via exact SVD leave-one-out CV."""
    Y_arr = np.asarray(Y, dtype=float)
    if Y_arr.ndim == 1:
        Y_arr = Y_arr[:, None]
    n, p = X.shape
    if p == 0 or n < 5:
        alpha0 = float(alphas[0]) if alphas else 1.0
        coef, y_mu, mu, sd = _ridge_fit(X, Y_arr, alpha=alpha0)
        return coef, y_mu, mu, sd, alpha0

    Z, mu, sd = _standardize_fit(X)
    y_mu = np.mean(Y_arr, axis=0)
    Yc = Y_arr - y_mu[None, :]
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Yc
    U2 = U**2
    eps = np.finfo(float).eps

    best_alpha = float(alphas[0])
    best_mse = float("inf")
    for a in alphas:
        alpha = max(float(a), 1e-8)
        d = s2 / (s2 + alpha)
        resid = Yc - U @ (d[:, None] * UtY)
        denom = np.maximum(1.0 - 1.0 / n - (U2 @ d), eps)[:, None]
        mse = float(np.mean((resid / denom) ** 2))
        if np.isfinite(mse) and mse < best_mse:
            best_mse = mse
            best_alpha = alpha

    scale = np.where(s2 + best_alpha > 1e-12, s / (s2 + best_alpha), 0.0)
    coef = (Vt.T * scale[None, :]) @ UtY
    return coef, y_mu, mu, sd, best_alpha


def _blocked_folds(n: int, n_splits: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    if n < 9:
        idx = np.arange(n)
        return [(idx, idx)]
    k = max(2, min(int(n_splits), n // 5))
    edges = np.linspace(0, n, k + 1, dtype=int)
    all_idx = np.arange(n)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for s_i, e_i in zip(edges[:-1], edges[1:], strict=True):
        te = np.arange(s_i, e_i)
        tr = np.concatenate([all_idx[:s_i], all_idx[e_i:]])
        if len(tr) > 0 and len(te) > 0:
            folds.append((tr, te))
    return folds or [(all_idx, all_idx)]


def _day_interleaved_folds(
    n: int, *, block_len: int = 24, n_splits: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """24h-block interleaved folds so events and trends appear across all folds."""
    b = max(1, int(block_len))
    k = max(2, int(n_splits))
    if n < b * k * 2:
        return _blocked_folds(n, n_splits=k)
    idx = np.arange(n)
    block_ids = (idx // b) % k
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(k):
        te = idx[block_ids == f]
        tr = idx[block_ids != f]
        if len(tr) > 0 and len(te) > 0:
            folds.append((tr, te))
    return folds or _blocked_folds(n, n_splits=k)


def _met_feature_matrix(
    df: pd.DataFrame,
    met_cols: Sequence[str],
    *,
    degree: int = 1,
    fit_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Build linear or degree-2 standardized meteorological design matrix."""
    cols = list(met_cols)
    if not cols:
        empty = np.empty((len(df), 0), dtype=float)
        return empty, (np.empty(0, dtype=float), np.empty(0, dtype=float))
    X_raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if fit_stats is None:
        Z, mu, sd = _standardize_fit(X_raw)
    else:
        mu, sd = fit_stats
        Z = _standardize_apply(X_raw, mu, sd)
    if int(degree) <= 1 or Z.shape[1] == 0:
        return Z, (mu, sd)
    k = Z.shape[1]
    quad_cols: list[np.ndarray] = [Z]
    # Monotone stagnation hinges + bounded tanh interactions prevent U-turn
    # extrapolation under large counterfactual weather shifts (e.g., ws * 1.8).
    k_inter = min(k, 8)
    Z_sub = Z[:, :k_inter]
    Z_clip = np.clip(Z_sub, -6.0, 6.0)
    quad_cols.append(np.log1p(np.exp(-Z_clip)))
    Z_tan = np.tanh(0.5 * Z_sub)
    inter_terms: list[np.ndarray] = []
    for i in range(k_inter):
        for j in range(i + 1, k_inter):
            inter_terms.append(Z_tan[:, i] * Z_tan[:, j])
    if inter_terms:
        quad_cols.append(np.column_stack(inter_terms))
    return np.hstack(quad_cols), (mu, sd)


def _dedup_knots(q: np.ndarray, *, min_gap: float = 1e-5) -> np.ndarray:
    kept: list[float] = []
    for val in np.sort(q):
        v = float(val)
        if not np.isfinite(v):
            continue
        if not kept or v - kept[-1] >= min_gap:
            kept.append(v)
    return np.asarray(kept, dtype=float)


def _gam_basis_fit(X: np.ndarray, *, n_knots: int = 6) -> tuple[np.ndarray, dict[str, Any]]:
    """Build truncated-power linear spline basis ``[z, max(z - q_k, 0)]`` with interior quantile knots."""
    Z, mu, sd = _standardize_fit(X)
    n, p = Z.shape
    if p == 0:
        return np.empty((n, 0), dtype=float), {"mu": mu, "sd": sd, "knots": []}

    q_probs = np.linspace(0.10, 0.90, max(2, int(n_knots)))
    knots_list: list[np.ndarray] = []
    for j in range(p):
        q = _dedup_knots(np.quantile(Z[:, j], q_probs))
        knots_list.append(q)

    state: dict[str, Any] = {"mu": mu, "sd": sd, "knots": knots_list}
    return _gam_basis_apply(X, state), state


def _gam_basis_apply(X: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    """Evaluate fitted truncated-power linear spline basis on new observations."""
    mu: np.ndarray = state["mu"]
    sd: np.ndarray = state["sd"]
    knots_list: list[np.ndarray] = state["knots"]
    Z = _standardize_apply(X, mu, sd)
    n, p = Z.shape
    if p == 0:
        return np.empty((n, 0), dtype=float)

    cols: list[np.ndarray] = []
    for j in range(p):
        z = Z[:, j]
        cols.append(z)
        for qk in knots_list[j]:
            cols.append(np.maximum(z - float(qk), 0.0))
    return np.column_stack(cols)


def _build_met_design(
    df: pd.DataFrame,
    met_cols: Sequence[str],
    *,
    met_head: str = "gam",
    degree: int = 1,
    n_knots: int = 6,
    state: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build meteorological action design matrix for ``gam``, ``linear``, or ``poly`` heads."""
    cols = list(met_cols)
    head_eff = str(state["met_head"] if state is not None else met_head).lower().strip()
    if head_eff not in ("gam", "linear", "poly", "tree"):
        raise ValueError(
            f"`met_head` must be 'gam', 'linear', 'poly', or 'tree', got {head_eff!r}."
        )
    if not cols:
        empty = np.empty((len(df), 0), dtype=float)
        return empty, {
            "met_head": head_eff,
            "stats": (np.empty(0, dtype=float), np.empty(0, dtype=float)),
            "gam_state": {
                "mu": np.empty(0, dtype=float),
                "sd": np.empty(0, dtype=float),
                "knots": [],
            },
        }

    if head_eff in ("linear", "tree"):
        fit_stats = state["stats"] if state is not None else None
        X_mat, stats = _met_feature_matrix(df, cols, degree=1, fit_stats=fit_stats)
        return X_mat, {"met_head": head_eff, "stats": stats}
    if head_eff == "poly":
        fit_stats = state["stats"] if state is not None else None
        deg_eff = int(state["degree"]) if state is not None else max(2, int(degree))
        X_mat, stats = _met_feature_matrix(df, cols, degree=deg_eff, fit_stats=fit_stats)
        return X_mat, {"met_head": head_eff, "stats": stats, "degree": deg_eff}

    X_raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if state is None:
        X_gam, gam_state = _gam_basis_fit(X_raw, n_knots=n_knots)
        return X_gam, {"met_head": "gam", "gam_state": gam_state}
    X_gam = _gam_basis_apply(X_raw, state["gam_state"])
    return X_gam, state


def _shifted_power(y: np.ndarray, gamma: float) -> np.ndarray:
    """Bijective shifted Box-Cox transform (gamma=0 -> log1p, gamma=1 -> identity)."""
    arr = np.asarray(y, dtype=float)
    g = float(gamma)
    if abs(g - 1.0) <= 1e-6:
        return arr.copy()
    pos = np.maximum(arr, 0.0)
    if abs(g) <= 1e-6:
        return np.log1p(pos)
    return (np.power(pos + 1.0, g) - 1.0) / g


def _inv_shifted_power(s: np.ndarray, gamma: float) -> np.ndarray:
    """Exact inverse of :func:`_shifted_power` on valid latent domain."""
    arr = np.asarray(s, dtype=float)
    g = float(gamma)
    if abs(g - 1.0) <= 1e-6:
        return arr.copy()
    if abs(g) <= 1e-6:
        return np.maximum(np.expm1(np.clip(arr, -30.0, 30.0)), 0.0)
    inner = np.maximum(1.0 + g * arr, 1e-12)
    return np.maximum(np.power(inner, 1.0 / g) - 1.0, 0.0)


def _fast_oof_separability(
    S: np.ndarray,
    X_met: np.ndarray,
    X_cal: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    alpha: float = 2.0,
) -> tuple[float, float, np.ndarray, float]:
    """Fast OOF multi-output RC-score, heteroscedastic rho_lb, residuals, and FVU."""
    n, m_dim = S.shape
    if n < 12 or m_dim == 0 or X_met.shape[1] == 0:
        return 0.0, 0.0, np.zeros_like(S), 0.0

    X_full = np.hstack([X_cal, X_met]) if X_cal.shape[1] > 0 else X_met
    S_hat = np.empty_like(S)
    for tr, te in folds:
        coef, y_mu, mu, sd = _ridge_fit(X_full[tr], S[tr], alpha=alpha)
        S_hat[te] = _ridge_predict(X_full[te], coef, y_mu, mu, sd)

    R = S - S_hat
    R2 = R**2
    var_s = np.maximum(np.var(S, axis=0), 1e-12)
    fvu = float(np.mean(np.mean(R2, axis=0) / var_s))

    # Out-of-fold linear residual correction and variance explained by meteorology
    C_pred = np.empty_like(R)
    V_pred = np.empty_like(R2)
    for tr, te in folds:
        c_coef, c_mu, xm_mu, xm_sd = _ridge_fit(X_met[tr], R[tr], alpha=alpha)
        C_pred[te] = _ridge_predict(X_met[te], c_coef, c_mu, xm_mu, xm_sd)
        v_coef, v_mu, _, _ = _ridge_fit(X_met[tr], R2[tr], alpha=alpha)
        V_pred[te] = _ridge_predict(X_met[te], v_coef, v_mu, xm_mu, xm_sd)

    rc_vals: list[float] = []
    rho_vals: list[float] = []
    for j in range(m_dim):
        rj = R[:, j]
        cj = C_pred[:, j]
        e0 = float(np.mean(rj**2))
        if e0 <= 1e-12:
            rc_vals.append(0.0)
            rho_vals.append(0.0)
            continue
        e1 = float(np.mean((rj - cj) ** 2))
        rc_vals.append(float(max(0.0, (e0 - e1) / e0)))

        r2j = R2[:, j]
        vj = V_pred[:, j]
        sst = float(np.var(r2j))
        if sst <= 1e-12:
            rho_vals.append(0.0)
            continue
        r2_var = 1.0 - float(np.mean((r2j - vj) ** 2)) / sst
        if r2_var <= 0.0:
            rho_vals.append(0.0)
            continue
        sigma_hat = np.sqrt(np.maximum(vj, 0.0))
        rho_j = float(np.clip(2.0 * np.var(sigma_hat) / e0, 0.0, 2.0))
        rho_vals.append(rho_j)

    return float(np.mean(rc_vals)), float(np.mean(rho_vals)), R, fvu


@dataclass
class _FoldProjector:
    """Precomputed fold ridge projection matrices for sub-millisecond coordinate scoring."""

    folds: list[tuple[np.ndarray, np.ndarray]]
    full_tr: list[np.ndarray]
    full_te: list[np.ndarray]
    met_tr: list[np.ndarray]
    met_te: list[np.ndarray]
    met_action_te: list[np.ndarray]
    n_cal: int


def _build_fold_projector(
    X_cal: np.ndarray,
    X_met_design: np.ndarray,
    X_met_audit: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    alpha_mean: float = 1.0,
    alpha_audit: float = 2.0,
) -> _FoldProjector:
    """Precompute standardization and linear ridge operators across CV folds."""
    X_full = np.hstack([X_cal, X_met_design]) if X_met_design.shape[1] > 0 else X_cal
    n_c = int(X_cal.shape[1])
    full_tr: list[np.ndarray] = []
    full_te: list[np.ndarray] = []
    met_tr: list[np.ndarray] = []
    met_te: list[np.ndarray] = []
    met_act_te: list[np.ndarray] = []

    for tr, te in folds:
        Z_tr, f_mu, f_sd = _standardize_fit(X_full[tr])
        Z_te = _standardize_apply(X_full[te], f_mu, f_sd)
        p_f = Z_tr.shape[1]
        if p_f > 0:
            gram_f = Z_tr.T @ Z_tr + max(float(alpha_mean), 1e-8) * np.eye(p_f, dtype=float)
            try:
                A_f = np.linalg.solve(gram_f, Z_tr.T)
            except np.linalg.LinAlgError:
                A_f = np.linalg.pinv(gram_f) @ Z_tr.T
        else:
            A_f = np.empty((0, len(tr)), dtype=float)
        full_tr.append(A_f)
        full_te.append(Z_te)
        met_act_te.append(Z_te[:, n_c:])

        if X_met_audit.shape[1] > 0:
            Zm_tr, m_mu, m_sd = _standardize_fit(X_met_audit[tr])
            Zm_te = _standardize_apply(X_met_audit[te], m_mu, m_sd)
            p_m = Zm_tr.shape[1]
            gram_m = Zm_tr.T @ Zm_tr + max(float(alpha_audit), 1e-8) * np.eye(p_m, dtype=float)
            try:
                A_m = np.linalg.solve(gram_m, Zm_tr.T)
            except np.linalg.LinAlgError:
                A_m = np.linalg.pinv(gram_m) @ Zm_tr.T
        else:
            A_m = np.empty((0, len(tr)), dtype=float)
            Zm_te = np.empty((len(te), 0), dtype=float)
        met_tr.append(A_m)
        met_te.append(Zm_te)

    return _FoldProjector(
        folds=list(folds),
        full_tr=full_tr,
        full_te=full_te,
        met_tr=met_tr,
        met_te=met_te,
        met_action_te=met_act_te,
        n_cal=n_c,
    )


def _fast_oof_separability_proj(
    S: np.ndarray,
    proj: _FoldProjector,
) -> tuple[float, float, float]:
    """Return ``(rc_mean, het_excess_mean, fvu_mean)`` using precomputed fold operators."""
    n, m_dim = S.shape
    if n < 12 or m_dim == 0 or (proj.met_te and proj.met_te[0].shape[1] == 0):
        return 0.0, 0.0, 0.0

    S_hat = np.empty_like(S)
    m_hat = np.empty_like(S)
    for k, (tr, te) in enumerate(proj.folds):
        y_tr = S[tr]

        mu_y = np.mean(y_tr, axis=0)
        if proj.full_tr[k].shape[0] > 0:
            coef = proj.full_tr[k] @ (y_tr - mu_y[None, :])
            S_hat[te] = mu_y[None, :] + proj.full_te[k] @ coef
            m_hat[te] = proj.met_action_te[k] @ coef[proj.n_cal :]
        else:
            S_hat[te] = mu_y[None, :]
            m_hat[te] = 0.0

    R = S - S_hat
    R2 = R * R
    C_pred = np.empty_like(R)
    for k, (tr, te) in enumerate(proj.folds):
        r_tr = R[tr]

        mu_r = np.mean(r_tr, axis=0)
        if proj.met_tr[k].shape[0] > 0:
            c_coef = proj.met_tr[k] @ (r_tr - mu_r[None, :])
            C_pred[te] = mu_r[None, :] + proj.met_te[k] @ c_coef
        else:
            C_pred[te] = mu_r[None, :]

    R_orth = R - C_pred
    fvu = float(np.mean(np.mean(R2, axis=0) / np.maximum(np.var(S, axis=0), 1e-12)))

    rc_vals: list[float] = []
    het_vals: list[float] = []
    for j in range(m_dim):
        e0 = float(np.mean(R2[:, j]))
        if e0 <= 1e-12:
            rc_vals.append(0.0)
            het_vals.append(0.0)
            continue
        e1 = float(np.mean(R_orth[:, j] ** 2))
        rc_vals.append(float(max(0.0, (e0 - e1) / e0)))

        ra = np.abs(R_orth[:, j])
        mh = m_hat[:, j]
        if float(np.std(ra)) > 1e-12 and float(np.std(mh)) > 1e-12:
            corr_c = float(np.corrcoef(ra, mh)[0, 1])
            het_j = max(0.0, corr_c * corr_c - 0.015)
        else:
            het_j = 0.0
        het_vals.append(het_j)

    return float(np.mean(rc_vals)), float(np.mean(het_vals)), fvu
