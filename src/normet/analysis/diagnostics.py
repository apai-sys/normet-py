# src/normet/analysis/diagnostics.py
"""Quality diagnostics for meteorological normalisation ("deweathering").

Evaluates whether a de-weathered series genuinely removed meteorological
confounding without over-correcting emission relationships, extrapolating
outside the meteorological resampling pool, or absorbing events via
in-sample overfitting:

- :func:`support_audit`           : Marginal A2 range support and multivariate
                                    kNN distance extrapolation check against a pool.
- :func:`diagnose_normalisation`  : Linear & nonlinear Weather Removal Fraction (WRF),
                                    per-variable Spearman attenuation & sign-flip
                                    over-correction detection, support audit, MC SE
                                    summary, and out-of-sample residual checks.
- :func:`counterfactual_separability_audit` :
                                    Factual-only audit of residual-reuse validity:
                                    the Eq. 66 residual-correction score, the
                                    interaction lower bound, and floor saturation,
                                    with a recommended coordinate (raw / log1p /
                                    conserved-or-rank).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils._config import DEFAULT_SEED
from .normalise import _is_time_feature

__all__ = [
    "NormalisationDiagnostics",
    "counterfactual_separability_audit",
    "support_audit",
    "diagnose_normalisation",
]


def _align_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy indexed by sorted ``pd.DatetimeIndex`` named ``date``."""
    work = frame.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.loc[work["date"].notna()].set_index("date")
    elif isinstance(work.index, pd.DatetimeIndex):
        work.index.name = "date"
    else:
        raise ValueError("Input frame must contain a 'date' column or DatetimeIndex.")
    return work.sort_index()


def support_audit(
    df: pd.DataFrame,
    pool: pd.DataFrame,
    met_cols: Sequence[str],
    *,
    k: int = 10,
    quantile: float = 0.99,
) -> dict[str, Any]:
    """Audit marginal range support and multivariate kNN extrapolation against *pool*.

    Parameters
    ----------
    df : pandas.DataFrame
        Target frame whose meteorological rows are audited.
    pool : pandas.DataFrame
        Reference meteorological pool (e.g. ``resample_df``).
    met_cols : sequence of str
        Meteorological covariate columns present in both frames.
    k : int, default 10
        Neighbour order used for multivariate distance thresholding.
    quantile : float, default 0.99
        Upper quantile of pool internal ``k``-NN distances defining the in-support
        envelope; target rows exceeding this threshold are flagged as extrapolations.

    Returns
    -------
    dict
        Keys ``support_fraction``, ``per_col``, ``extrapolation_rate``,
        ``knn_threshold``, ``n_target``, ``n_pool``.
    """
    use_cols = [
        c
        for c in met_cols
        if c in df.columns
        and c in pool.columns
        and pd.api.types.is_numeric_dtype(df[c])
        and pd.api.types.is_numeric_dtype(pool[c])
    ]
    n_target = len(df)
    n_pool = len(pool)
    if not use_cols or n_target == 0 or n_pool == 0:
        return {
            "support_fraction": 1.0 if n_target else 0.0,
            "per_col": {},
            "extrapolation_rate": 0.0,
            "knn_threshold": float("nan"),
            "n_target": n_target,
            "n_pool": n_pool,
        }

    per_col: dict[str, float] = {}
    inside_total = 0
    cell_total = 0
    for c in use_cols:
        p_vals = pd.to_numeric(pool[c], errors="coerce").to_numpy(dtype=float)
        t_vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        p_fin = p_vals[np.isfinite(p_vals)]
        if p_fin.size == 0:
            per_col[c] = 0.0
            cell_total += len(t_vals)
            continue
        lo = float(np.min(p_fin))
        hi = float(np.max(p_fin))
        ok = np.isfinite(t_vals) & (t_vals >= lo) & (t_vals <= hi)
        per_col[c] = float(ok.mean()) if len(ok) else 0.0
        inside_total += int(ok.sum())
        cell_total += len(ok)

    support_fraction = (inside_total / cell_total) if cell_total else 1.0

    extrap_rate = 0.0
    knn_thresh = float("nan")
    if n_pool >= 2:
        from scipy.spatial import cKDTree

        x_pool = pool[use_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        x_tgt = df[use_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        mu = np.nanmean(x_pool, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.nanstd(x_pool, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        z_pool = np.where(np.isfinite(x_pool), (x_pool - mu) / sd, 0.0)
        z_tgt = np.where(np.isfinite(x_tgt), (x_tgt - mu) / sd, 0.0)

        k_eff = min(max(1, int(k)), n_pool - 1)
        tree = cKDTree(z_pool)
        pool_dists, _ = tree.query(z_pool, k=k_eff + 1)
        pool_k = np.asarray(pool_dists, dtype=float).reshape(n_pool, k_eff + 1)[:, -1]
        knn_thresh = float(np.quantile(pool_k, float(quantile)))

        tgt_dists, _ = tree.query(z_tgt, k=k_eff)
        tgt_k = np.asarray(tgt_dists, dtype=float).reshape(n_target, k_eff)[:, -1]
        extrap_rate = float(np.mean(tgt_k > knn_thresh))

    return {
        "support_fraction": float(support_fraction),
        "per_col": per_col,
        "extrapolation_rate": float(extrap_rate),
        "knn_threshold": float(knn_thresh),
        "n_target": int(n_target),
        "n_pool": int(n_pool),
    }


def _oof_ridge_predict(
    X: np.ndarray, y: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Out-of-fold ridge-stabilised OLS predictions."""
    n = len(y)
    oof_pred = np.full(n, np.nan, dtype=float)
    if n < 3 or X.shape[1] == 0:
        return oof_pred
    eval_folds = list(folds) if folds else [(np.arange(n), np.arange(n))]
    for tr_idx, te_idx in eval_folds:
        x_tr, y_tr = X[tr_idx], y[tr_idx]
        x_te = X[te_idx]
        mu = x_tr.mean(axis=0)
        sd = x_tr.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        z_tr = (x_tr - mu) / sd
        z_te = (x_te - mu) / sd
        design_tr = np.column_stack([np.ones(len(z_tr)), z_tr])
        design_te = np.column_stack([np.ones(len(z_te)), z_te])
        gram = design_tr.T @ design_tr
        reg = np.eye(gram.shape[0]) * 1e-6
        reg[0, 0] = 0.0
        coef = np.linalg.solve(gram + reg, design_tr.T @ y_tr)
        oof_pred[te_idx] = design_te @ coef
    return oof_pred


def _oof_r2_raw(y: np.ndarray, oof_pred: np.ndarray) -> float | None:
    """Unclamped OOF R^2 from aligned target and prediction arrays."""
    mask = np.isfinite(oof_pred) & np.isfinite(y)
    if not np.any(mask):
        return None
    ym = y[mask]
    pm = oof_pred[mask]
    sst = float(np.sum((ym - np.mean(ym)) ** 2))
    if sst <= 1e-12:
        return 0.0
    sse = float(np.sum((ym - pm) ** 2))
    return float(1.0 - sse / sst)


def _oof_r2_linear(
    X: np.ndarray, y: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]]
) -> float:
    """Out-of-fold ridge-stabilised OLS R^2 bounded below at 0."""
    oof_pred = _oof_ridge_predict(X, y, folds)
    r2 = _oof_r2_raw(y, oof_pred)
    return float(max(0.0, r2)) if r2 is not None else 0.0


def _oof_extratrees_predict(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> np.ndarray | None:
    """Out-of-fold ExtraTrees predictions (returns None if sklearn absent or too small)."""
    if len(y) < 15 or X.shape[1] == 0 or not folds:
        return None
    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except ImportError:
        return None

    oof_pred = np.full(len(y), np.nan, dtype=float)
    for idx, (tr_idx, te_idx) in enumerate(folds):
        if len(tr_idx) < 5 or len(te_idx) == 0:
            continue
        reg = ExtraTreesRegressor(
            n_estimators=40,
            max_depth=8,
            min_samples_leaf=3,
            random_state=int(seed) + idx,
            n_jobs=1,
        )
        reg.fit(X[tr_idx], y[tr_idx])
        oof_pred[te_idx] = reg.predict(X[te_idx])
    return oof_pred


def _oof_r2_nonlinear(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> float | None:
    """Out-of-fold ExtraTrees R^2 bounded below at 0 (returns None if sklearn absent)."""
    oof_pred = _oof_extratrees_predict(X, y, folds, seed=seed)
    if oof_pred is None:
        return None
    r2 = _oof_r2_raw(y, oof_pred)
    return float(max(0.0, r2)) if r2 is not None else None


def _power_transform(y: np.ndarray, gamma: float) -> np.ndarray:
    """Shifted Box-Cox power coordinate on non-negative values (gamma=0 matches log1p)."""
    x = np.maximum(np.asarray(y, dtype=float), 0.0) + 1.0
    if abs(gamma) <= 1e-6:
        return np.log(x)
    return (np.power(x, gamma) - 1.0) / float(gamma)


def _rc_from_arrays(r: np.ndarray, c: np.ndarray) -> float:
    m = np.isfinite(r) & np.isfinite(c)
    if not np.any(m):
        return 0.0
    e0 = float(np.mean(r[m] ** 2))
    if e0 <= 1e-12:
        return 0.0
    e1 = float(np.mean((r[m] - c[m]) ** 2))
    return float(max(0.0, (e0 - e1) / max(e0, 1e-12)))


def _oof_residual_correction_details(
    X: np.ndarray,
    r: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    nonlinear: bool,
    seed: int,
) -> tuple[float, float | None, np.ndarray]:
    """Appendix I Eq. (66) residual-correction score plus OOF linear correction predictions."""
    e0 = float(np.mean(r**2))
    if e0 <= 1e-12:
        return 0.0, (0.0 if nonlinear else None), np.zeros(len(r), dtype=float)

    c_lin = _oof_ridge_predict(X, r, folds)
    rc_lin = _rc_from_arrays(r, c_lin)

    rc_nl: float | None = None
    if nonlinear and len(r) >= 30 and folds and X.shape[1] > 0:
        c_nl = _oof_extratrees_predict(X, r, folds, seed=seed)
        if c_nl is not None:
            rc_nl = _rc_from_arrays(r, c_nl)
    return rc_lin, rc_nl, c_lin


def _oof_residual_correction_score(
    X: np.ndarray,
    r: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    nonlinear: bool,
    seed: int,
) -> tuple[float, float | None]:
    """Appendix I Eq. (66) residual-correction score: fraction of MSE explained by meteorology."""
    rc_lin, rc_nl, _ = _oof_residual_correction_details(X, r, folds, nonlinear=nonlinear, seed=seed)
    return rc_lin, rc_nl


def _interaction_rho_lower_bound(
    X: np.ndarray,
    r: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    nonlinear: bool,
    seed: int,
) -> tuple[float, float]:
    """Factual lower bound on action-disturbance interaction ratio rho_lb (Section 3.2)."""
    if len(r) < 3 or X.shape[1] == 0:
        return 0.0, 0.0
    r2_target = r**2
    v_lin = _oof_ridge_predict(X, r2_target, folds)
    r2_lin = _oof_r2_raw(r2_target, v_lin)
    best_r2 = float(r2_lin) if r2_lin is not None else float("-inf")
    best_v = v_lin

    if nonlinear and len(r) >= 30 and folds:
        v_nl = _oof_extratrees_predict(X, r2_target, folds, seed=seed)
        if v_nl is not None:
            r2_nl = _oof_r2_raw(r2_target, v_nl)
            if r2_nl is not None and float(r2_nl) > best_r2:
                best_r2 = float(r2_nl)
                best_v = v_nl

    if not np.isfinite(best_r2) or best_r2 <= 0.0:
        return 0.0, 0.0

    mask = np.isfinite(best_v) & np.isfinite(r)
    if not np.any(mask):
        return 0.0, 0.0
    v_hat = np.maximum(best_v[mask], 0.0)
    sigma_hat = np.sqrt(v_hat)
    denom = max(float(np.mean(r[mask] ** 2)), 1e-12)
    rho_lb = float(np.clip(2.0 * np.var(sigma_hat) / denom, 0.0, 2.0))
    return rho_lb, float(best_r2)


def _interaction_w2_lower_bound(
    X: np.ndarray,
    r: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    nonlinear: bool,
    seed: int,
) -> float:
    """Distribution-free 1D Wasserstein-2 factual lower bound on action-disturbance coupling."""
    n = len(r)
    if n < 20 or X.shape[1] == 0:
        return 0.0

    r2_target = r**2
    v_lin = _oof_ridge_predict(X, r2_target, folds)
    r2_lin = _oof_r2_raw(r2_target, v_lin)
    best_r2 = float(r2_lin) if r2_lin is not None else float("-inf")
    score = v_lin

    if nonlinear and n >= 30 and folds:
        v_nl = _oof_extratrees_predict(X, r2_target, folds, seed=seed)
        if v_nl is not None:
            r2_nl = _oof_r2_raw(r2_target, v_nl)
            if r2_nl is not None and float(r2_nl) > best_r2:
                best_r2 = float(r2_nl)
                score = v_nl

    # Fallback to an OOF meteorological principal score when variance regression is flat.
    if not np.isfinite(best_r2) or best_r2 <= 0.0 or not np.any(np.isfinite(score)):
        score = np.full(n, np.nan, dtype=float)
        for tr, te in folds:
            if len(tr) == 0 or len(te) == 0:
                continue
            X_tr = X[tr]
            mu = np.mean(X_tr, axis=0)
            sd = np.std(X_tr, axis=0)
            sd = np.where(sd > 1e-9, sd, 1.0)
            Z_tr = (X_tr - mu) / sd
            Z_te = (X[te] - mu) / sd
            try:
                _, _, vt = np.linalg.svd(Z_tr, full_matrices=False)
                score[te] = Z_te @ vt[0]
            except np.linalg.LinAlgError:
                score[te] = Z_te[:, 0]

    ok = np.isfinite(score) & np.isfinite(r)
    if int(ok.sum()) < 20:
        return 0.0
    sc = score[ok]
    rv = r[ok]
    rv_c = rv - float(np.mean(rv))
    denom = max(float(np.mean(rv_c**2)), 1e-12)

    n_bins = min(6, max(2, len(rv_c) // 25))
    order = np.argsort(sc, kind="mergesort")
    bins = [b for b in np.array_split(order, n_bins) if len(b) >= 5]
    if len(bins) < 2:
        return 0.0

    q_grid = np.linspace(0.05, 0.95, 19)

    def _grouped_w2(vals: np.ndarray) -> float:
        q_bar = np.quantile(vals, q_grid)
        total = 0.0
        w_sum = 0.0
        for b in bins:
            q_b = np.quantile(vals[b], q_grid)
            total += float(len(b)) * float(np.mean((q_b - q_bar) ** 2))
            w_sum += float(len(b))
        return total / max(w_sum, 1.0)

    w2_obs = _grouped_w2(rv_c)
    rng = np.random.default_rng(seed)
    n_perm = 8
    w2_null = 0.0
    for _ in range(n_perm):
        w2_null += _grouped_w2(rng.permutation(rv_c))
    w2_null /= float(n_perm)

    w2_debiased = max(0.0, w2_obs - w2_null)
    return float(np.clip(2.0 * w2_debiased / denom, 0.0, 2.0))


def _block_bootstrap_rc_upper(
    r: np.ndarray,
    c_lin: np.ndarray,
    *,
    block_length: int,
    n_boot: int,
    seed: int,
) -> float:
    m = np.isfinite(r) & np.isfinite(c_lin)
    r_m = r[m]
    c_m = c_lin[m]
    n_m = len(r_m)
    if n_m < 10 or n_boot < 1:
        return float("nan")
    b = max(1, min(int(block_length), n_m // 2))
    n_starts = n_m - b + 1
    k = int(np.ceil(n_m / b))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    offsets = np.arange(b)[None, :]
    for i in range(n_boot):
        starts = rng.integers(0, n_starts, size=k)[:, None]
        idx = (starts + offsets).ravel()[:n_m]
        boots[i] = _rc_from_arrays(r_m[idx], c_m[idx])
    return float(np.nanpercentile(boots, 95.0))


def counterfactual_separability_audit(
    y_obs: np.ndarray,
    y_fact: np.ndarray,
    X_met: np.ndarray,
    *,
    folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    nonlinear: bool = True,
    met_names: Sequence[str] | None = None,
    n_boot: int = 200,
    block_length: int = 24,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Audit residual-reuse validity from factual residuals and meteorology alone.

    Implements factual diagnostics grounded in Balestriero, LeCun & Varaku (2026):

    1. **Residual-correction test** (``rc_score_linear``, ``rc_score_nonlinear``,
       ``rc_score_linear_upper95``, ``rc_score_last_block``; Appendix I Eq. 66):
       out-of-fold fraction of factual residual energy
       :math:`R_t = Y_t - \\hat{\\mu}(H_t, w_t)` explained by meteorology.
    2. **Factual interaction lower bounds** (``interaction_rho_lb_raw``,
       ``interaction_rho_lb_log``, ``interaction_w2_lb_raw``, ``interaction_w2_lb_log``,
       ``optimal_power_gamma``, ``interaction_rho_lb_optimal_power``; Section 3.2):
       heteroscedastic spread and 1D Wasserstein-2 variation across meteorology plus
       continuous shifted-power coordinate selection.
    3. **Boundary saturation & persistence** (``floor_saturation_rate``,
       ``ceiling_saturation_rate``, ``residual_fresh_share``; Section 4.3):
       fraction collapsed on lower/upper bounds and non-persistent residual share.
    4. **Per-feature attribution** (``rc_by_feature``, ``rho_lb_by_feature``,
       ``top_rc_feature``, ``top_interaction_feature``): single-covariate RC and spread
       lower bounds identifying which meteorological channels drive violation.
    """
    y_o = np.asarray(y_obs, dtype=float)
    y_f = np.asarray(y_fact, dtype=float)
    X_arr = np.asarray(X_met, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]

    n_features = X_arr.shape[1] if X_arr.ndim == 2 else 0
    if met_names is not None and len(met_names) == n_features:
        feat_names = [str(nm) for nm in met_names]
    else:
        feat_names = [f"x{j}" for j in range(n_features)]

    ok = np.isfinite(y_o) & np.isfinite(y_f)
    if X_arr.shape[0] != len(y_o):
        raise ValueError(
            f"Length mismatch between y_obs ({len(y_o)}) and X_met ({X_arr.shape[0]})."
        )
    if X_arr.shape[1] > 0:
        ok &= np.all(np.isfinite(X_arr), axis=1)

    n_ok = int(ok.sum())
    if n_ok < 10:
        return {
            "n_obs": n_ok,
            "rc_score_linear": float("nan"),
            "rc_score_nonlinear": float("nan") if nonlinear else None,
            "rc_score_linear_upper95": float("nan"),
            "rc_score_last_block": float("nan"),
            "interaction_rho_lb_raw": float("nan"),
            "interaction_rho_lb_log": float("nan"),
            "interaction_w2_lb_raw": float("nan"),
            "interaction_w2_lb_log": float("nan"),
            "optimal_power_gamma": float("nan"),
            "interaction_rho_lb_optimal_power": float("nan"),
            "spread_r2_raw": float("nan"),
            "spread_r2_log": float("nan"),
            "floor_saturation_rate": float("nan"),
            "ceiling_saturation_rate": float("nan"),
            "residual_fresh_share": float("nan"),
            "rc_by_feature": {},
            "rho_lb_by_feature": {},
            "top_rc_feature": None,
            "top_interaction_feature": None,
            "recommended_coordinate": "insufficient_data",
        }

    yo = y_o[ok]
    yf = y_f[ok]
    Xm = X_arr[ok]

    use_folds: list[tuple[np.ndarray, np.ndarray]]
    if folds is not None and bool(np.all(ok)):
        use_folds = list(folds)
    elif n_ok >= 15:
        n_splits = min(3, n_ok // 5)
        edges = np.linspace(0, n_ok, n_splits + 1, dtype=int)
        all_idx = np.arange(n_ok)
        use_folds = []
        for s_i, e_i in zip(edges[:-1], edges[1:], strict=True):
            te = np.arange(s_i, e_i)
            tr = np.concatenate([all_idx[:s_i], all_idx[e_i:]])
            if len(tr) > 0 and len(te) > 0:
                use_folds.append((tr, te))
        if not use_folds:
            use_folds = [(all_idx, all_idx)]
    else:
        all_idx = np.arange(n_ok)
        use_folds = [(all_idx, all_idx)]

    r_raw = yo - yf
    rc_lin, rc_nl, c_lin = _oof_residual_correction_details(
        Xm, r_raw, use_folds, nonlinear=nonlinear, seed=seed
    )
    rc_upper95 = _block_bootstrap_rc_upper(
        r_raw, c_lin, block_length=block_length, n_boot=n_boot, seed=seed
    )
    if len(use_folds) >= 2:
        _, te_last = use_folds[-1]
        rc_last = _rc_from_arrays(r_raw[te_last], c_lin[te_last]) if len(te_last) > 0 else rc_lin
    else:
        rc_last = rc_lin

    rho_raw, spread_r2_raw = _interaction_rho_lower_bound(
        Xm, r_raw, use_folds, nonlinear=nonlinear, seed=seed
    )
    w2_raw = _interaction_w2_lower_bound(Xm, r_raw, use_folds, nonlinear=nonlinear, seed=seed)

    if bool(np.all(yo >= -1e-9)) and bool(np.all(yf >= -1e-9)):
        r_log = np.log1p(np.maximum(yo, 0.0)) - np.log1p(np.maximum(yf, 0.0))
        rho_log, spread_r2_log = _interaction_rho_lower_bound(
            Xm, r_log, use_folds, nonlinear=nonlinear, seed=seed
        )
        w2_log = _interaction_w2_lower_bound(Xm, r_log, use_folds, nonlinear=nonlinear, seed=seed)

        gammas = (0.0, 0.25, 0.5, 0.75, 1.0)
        best_gamma = 0.0
        best_rho_gamma = float("inf")
        for g in gammas:
            r_g = _power_transform(yo, g) - _power_transform(yf, g)
            rho_g, _ = _interaction_rho_lower_bound(
                Xm, r_g, use_folds, nonlinear=nonlinear, seed=seed
            )
            if rho_g < best_rho_gamma - 1e-12:
                best_rho_gamma = rho_g
                best_gamma = g
        opt_gamma = float(best_gamma)
        rho_opt = float(best_rho_gamma)
    else:
        rho_log = float("nan")
        spread_r2_log = float("nan")
        w2_log = float("nan")
        opt_gamma = 1.0
        rho_opt = float(rho_raw)

    hi = float(np.nanmax(yo))
    lo = float(np.nanmin(yo))
    rng_y = hi - lo

    if float(np.nanmin(yo)) >= -1e-9:
        med_abs = float(np.median(np.abs(yo)))
        tol = max(1e-6, 0.01 * med_abs) if med_abs > 1e-6 else 1e-6
        floor_rate = float(np.mean(yo <= tol))
        if floor_rate == 0.0 and rng_y > 1e-12:
            tol_lo = max(1e-7, 1e-5 * rng_y)
            n_lo = int(np.sum(yo <= lo + tol_lo))
            if n_lo >= max(3, int(0.02 * n_ok)):
                floor_rate = float(n_lo / n_ok)
    else:
        floor_rate = 0.0

    if rng_y > 1e-12:
        tol_hi = max(1e-7, 1e-5 * rng_y)
        n_hi = int(np.sum(yo >= hi - tol_hi))
        ceiling_rate = 0.0 if n_hi <= 1 else float(n_hi / n_ok)
    else:
        ceiling_rate = 1.0

    ac1 = _lag_autocorr(r_raw, 1)
    fresh_share = (
        float(np.clip(1.0 - max(0.0, ac1), 0.0, 1.0)) if np.isfinite(ac1) else float("nan")
    )

    rc_by_feature: dict[str, float] = {}
    rho_lb_by_feature: dict[str, float] = {}
    for j, fname in enumerate(feat_names):
        X_j = Xm[:, [j]]
        rc_j, _ = _oof_residual_correction_score(X_j, r_raw, use_folds, nonlinear=False, seed=seed)
        rho_j, _ = _interaction_rho_lower_bound(X_j, r_raw, use_folds, nonlinear=False, seed=seed)
        rc_by_feature[fname] = float(rc_j)
        rho_lb_by_feature[fname] = float(rho_j)

    top_rc_feature = max(rc_by_feature, key=lambda k: rc_by_feature[k]) if rc_by_feature else None
    top_interaction_feature = (
        max(rho_lb_by_feature, key=lambda k: rho_lb_by_feature[k]) if rho_lb_by_feature else None
    )

    if np.isfinite(rho_log):
        if rho_raw >= 0.15 and rho_log >= 0.15:
            rec_coord = "conserved_or_rank"
        elif rho_raw >= 0.05 and rho_log < 0.6 * rho_raw:
            rec_coord = "log1p"
        elif rho_raw < 0.05 and rho_log < 0.05:
            rec_coord = "raw_or_log"
        else:
            rec_coord = "log1p" if rho_log < rho_raw else "raw"
    else:
        rec_coord = "conserved_or_rank" if rho_raw >= 0.15 else "raw"

    return {
        "n_obs": n_ok,
        "rc_score_linear": float(rc_lin),
        "rc_score_nonlinear": float(rc_nl) if rc_nl is not None else None,
        "rc_score_linear_upper95": float(rc_upper95),
        "rc_score_last_block": float(rc_last),
        "interaction_rho_lb_raw": float(rho_raw),
        "interaction_rho_lb_log": float(rho_log),
        "interaction_w2_lb_raw": float(w2_raw),
        "interaction_w2_lb_log": float(w2_log),
        "optimal_power_gamma": float(opt_gamma),
        "interaction_rho_lb_optimal_power": float(rho_opt),
        "spread_r2_raw": float(spread_r2_raw),
        "spread_r2_log": float(spread_r2_log),
        "floor_saturation_rate": float(floor_rate),
        "ceiling_saturation_rate": float(ceiling_rate),
        "residual_fresh_share": float(fresh_share),
        "rc_by_feature": rc_by_feature,
        "rho_lb_by_feature": rho_lb_by_feature,
        "top_rc_feature": top_rc_feature,
        "top_interaction_feature": top_interaction_feature,
        "recommended_coordinate": rec_coord,
    }


def _lag_autocorr(arr: np.ndarray, lag: int) -> float:
    if len(arr) <= lag + 2:
        return float("nan")
    a = arr[:-lag]
    b = arr[lag:]
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class NormalisationDiagnostics:
    """Comprehensive diagnostic report for a de-weathered time series."""

    wrf_linear: float
    wrf_nonlinear: float | None
    r2_met_observed_linear: float
    r2_met_normalised_linear: float
    r2_met_observed_nonlinear: float | None
    r2_met_normalised_nonlinear: float | None
    variable_correlations: pd.DataFrame
    overcorrection_flags: list[str]
    support_fraction: float | None
    support_per_col: dict[str, float]
    extrapolation_rate: float | None
    mc_se_summary: dict[str, float] | None = None
    residual_diagnostics: dict[str, float] | None = None
    counterfactual_diagnostics: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Return a tidy two-column (``metric``, ``value``) summary table."""
        rows: list[tuple[str, Any]] = [
            ("wrf_linear", self.wrf_linear),
            ("wrf_nonlinear", self.wrf_nonlinear),
            ("r2_met_observed_linear", self.r2_met_observed_linear),
            ("r2_met_normalised_linear", self.r2_met_normalised_linear),
            ("r2_met_observed_nonlinear", self.r2_met_observed_nonlinear),
            ("r2_met_normalised_nonlinear", self.r2_met_normalised_nonlinear),
            ("support_fraction", self.support_fraction),
            ("extrapolation_rate", self.extrapolation_rate),
            ("n_overcorrection_flags", len(self.overcorrection_flags)),
        ]
        if self.mc_se_summary:
            for k, v in self.mc_se_summary.items():
                rows.append((f"mc_{k}", v))
        if self.residual_diagnostics:
            for k, v in self.residual_diagnostics.items():
                rows.append((f"residual_{k}", v))
        if self.counterfactual_diagnostics:
            for k, v in self.counterfactual_diagnostics.items():
                if isinstance(v, dict):
                    continue
                rows.append((f"cf_{k}", v))
        return pd.DataFrame(rows, columns=["metric", "value"])

    def __str__(self) -> str:
        wrf_nl = (
            f"{100 * self.wrf_nonlinear:.1f}%"
            if self.wrf_nonlinear is not None and np.isfinite(self.wrf_nonlinear)
            else "n/a"
        )
        sup = f"{100 * self.support_fraction:.1f}%" if self.support_fraction is not None else "n/a"
        ext = (
            f"{100 * self.extrapolation_rate:.1f}%"
            if self.extrapolation_rate is not None
            else "n/a"
        )
        lines = [
            f"NormalisationDiagnostics: WRF(lin)={100 * self.wrf_linear:.1f}%, WRF(nl)={wrf_nl}",
            f"  support={sup}, extrapolation_rate={ext}, overcorrection_flags={self.overcorrection_flags or 'none'}",
        ]
        if self.warnings:
            lines.append("  warnings: " + "; ".join(self.warnings))
        return "\n".join(lines)


def diagnose_normalisation(
    df_norm: pd.DataFrame,
    df_met: pd.DataFrame,
    *,
    met_cols: Sequence[str] | None = None,
    normalised_col: str = "normalised",
    observed_col: str = "observed",
    resample_df: pd.DataFrame | None = None,
    model: object | None = None,
    nonlinear_wrf: bool = True,
    seed: int = DEFAULT_SEED,
) -> NormalisationDiagnostics:
    """Run meteorological-removal, over-correction, support, and residual audits.

    Parameters
    ----------
    df_norm : pandas.DataFrame
        Output DataFrame from :func:`normalise` / :func:`do_all` / :func:`crossfit_normalise`.
    df_met : pandas.DataFrame
        DataFrame carrying meteorological predictors aligned by timestamp.
    met_cols : sequence of str, optional
        Meteorological columns to evaluate. When omitted, inferred from numeric
        non-time columns in ``df_met``.
    normalised_col : str, default "normalised"
        De-weathered series column in ``df_norm`` (e.g. ``"normalised"`` or ``"paired"``).
    observed_col : str, default "observed"
        Observed target series column in ``df_norm``.
    resample_df : pandas.DataFrame, optional
        Reference resampling pool for A2 support and kNN extrapolation checks.
        Defaults to the aligned ``df_met`` frame.
    model : object, optional
        Optional fitted model used to compute factual residuals when ``"factual"``
        is not already in ``df_norm``.
    nonlinear_wrf : bool, default True
        Compute out-of-fold ExtraTrees nonlinear WRF in addition to linear WRF.
    seed : int, default ``DEFAULT_SEED``
        Random seed used by the nonlinear WRF cross-validator.

    Returns
    -------
    NormalisationDiagnostics
    """
    from scipy.stats import spearmanr

    from ..utils.cv import blocked_time_cv

    norm_idx = _align_by_date(df_norm)
    met_idx = _align_by_date(df_met)

    for req_col in (normalised_col, observed_col):
        if req_col not in norm_idx.columns:
            raise ValueError(f"Column {req_col!r} not found in `df_norm`.")

    if met_cols is None:
        excluded = {
            "date",
            "value",
            "set",
            observed_col,
            normalised_col,
            "normalised",
            "observed",
            "paired",
            "factual",
            "weather_effect",
            "mc_se",
            "fold",
        }
        inferred = [
            c
            for c in met_idx.columns
            if c not in excluded
            and not _is_time_feature(c)
            and pd.api.types.is_numeric_dtype(met_idx[c])
        ]
    else:
        inferred = [
            c
            for c in met_cols
            if c in met_idx.columns and pd.api.types.is_numeric_dtype(met_idx[c])
        ]
    if not inferred:
        raise ValueError("No numeric meteorological columns found for diagnostics.")

    # Align by timestamp (if either side has duplicate timestamps, deduplicate per timestamp)
    if not norm_idx.index.is_unique:
        norm_idx = norm_idx.groupby(level=0).mean(numeric_only=True)
    if not met_idx.index.is_unique:
        num_cols = met_idx.select_dtypes(include=[np.number]).columns
        agg_met = met_idx[num_cols].groupby(level=0).mean()
        if "set" in met_idx.columns:
            agg_met["set"] = met_idx["set"].groupby(level=0).first()
        met_idx = agg_met

    common = norm_idx.index.intersection(met_idx.index)
    if common.empty:
        raise ValueError("`df_norm` and `df_met` share no common timestamps.")

    norm_al = norm_idx.loc[common]
    met_al = met_idx.loc[common]
    y_obs = pd.to_numeric(norm_al[observed_col], errors="coerce").to_numpy(dtype=float)
    y_norm = pd.to_numeric(norm_al[normalised_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_obs) & np.isfinite(y_norm)
    if int(valid.sum()) < 5:
        raise ValueError("Need at least 5 finite paired observations for diagnostics.")

    norm_al = norm_al.iloc[valid]
    met_al = met_al.iloc[valid]
    y_obs = y_obs[valid]
    y_norm = y_norm[valid]

    X_df = met_al[inferred].apply(pd.to_numeric, errors="coerce")
    col_means = X_df.mean(axis=0).fillna(0.0)
    X_mat = X_df.fillna(col_means).to_numpy(dtype=np.float64)

    n = len(y_obs)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    if n >= 15:
        n_splits = min(3, n // 5)
        if n_splits >= 2:
            folds = list(
                blocked_time_cv(pd.DataFrame({"date": norm_al.index}), n_splits=n_splits, embargo=0)
            )

    r2_obs_lin = _oof_r2_linear(X_mat, y_obs, folds)
    r2_norm_lin = _oof_r2_linear(X_mat, y_norm, folds)
    wrf_lin = float(1.0 - r2_norm_lin / r2_obs_lin) if r2_obs_lin > 1e-6 else float("nan")

    r2_obs_nl: float | None = None
    r2_norm_nl: float | None = None
    wrf_nl: float | None = None
    if nonlinear_wrf and n >= 30 and folds:
        r2_obs_nl = _oof_r2_nonlinear(X_mat, y_obs, folds, seed=seed)
        r2_norm_nl = _oof_r2_nonlinear(X_mat, y_norm, folds, seed=seed)
        if r2_obs_nl is not None and r2_norm_nl is not None:
            wrf_nl = float(1.0 - r2_norm_nl / r2_obs_nl) if r2_obs_nl > 1e-6 else float("nan")

    corr_rows: list[dict[str, Any]] = []
    over_flags: list[str] = []
    for c in inferred:
        xc = X_df[c].to_numpy(dtype=float)
        ok = np.isfinite(xc)
        if int(ok.sum()) < 5 or float(np.std(xc[ok])) <= 1e-12:
            r_o, p_o, r_n, p_n = float("nan"), float("nan"), float("nan"), float("nan")
        else:
            res_o = spearmanr(xc[ok], y_obs[ok])
            res_n = spearmanr(xc[ok], y_norm[ok])
            r_o, p_o = float(res_o.statistic), float(res_o.pvalue)
            r_n, p_n = float(res_n.statistic), float(res_n.pvalue)
        sign_flip = bool(
            np.isfinite(r_o)
            and np.isfinite(r_n)
            and abs(r_o) >= 0.1
            and abs(r_n) >= 0.05
            and np.sign(r_o) != np.sign(r_n)
        )
        if sign_flip:
            over_flags.append(c)
        corr_rows.append(
            {
                "variable": c,
                "spearman_observed": r_o,
                "p_observed": p_o,
                "spearman_normalised": r_n,
                "p_normalised": p_n,
                "abs_reduction": abs(r_o) - abs(r_n)
                if np.isfinite(r_o) and np.isfinite(r_n)
                else float("nan"),
                "sign_flip": sign_flip,
            }
        )
    corr_df = pd.DataFrame(corr_rows)

    pool_use = _align_by_date(resample_df) if resample_df is not None else met_al
    sup = support_audit(met_al, pool_use, inferred)

    mc_se_summary: dict[str, float] | None = None
    if "mc_se" in norm_al.columns:
        se = pd.to_numeric(norm_al["mc_se"], errors="coerce").to_numpy(dtype=float)
        ok_se = np.isfinite(se)
        if np.any(ok_se):
            rse = se[ok_se] / np.maximum(np.abs(y_norm[ok_se]), 1e-12)
            mc_se_summary = {
                "median_se": float(np.median(se[ok_se])),
                "p95_se": float(np.percentile(se[ok_se], 95.0)),
                "median_rse": float(np.median(rse)),
                "p95_rse": float(np.percentile(rse, 95.0)),
            }

    residual_diag: dict[str, float] | None = None
    y_fact: np.ndarray | None = None
    if "factual" in norm_al.columns:
        cand_f = pd.to_numeric(norm_al["factual"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(cand_f).sum() >= 5:
            y_fact = cand_f
    elif model is not None:
        try:
            from ..model.predict import ml_predict

            cand_f = ml_predict(model, met_al, num_threads=1)
            if len(cand_f) == len(y_obs):
                y_fact = cand_f
        except Exception:
            y_fact = None

    cf_diag: dict[str, Any] | None = None
    if y_fact is not None:
        ok_f = np.isfinite(y_fact)
        if int(ok_f.sum()) >= 5:
            resid = y_obs[ok_f] - y_fact[ok_f]
            residual_diag = {
                "rmse": float(np.sqrt(np.mean(resid**2))),
                "mae": float(np.mean(np.abs(resid))),
                "lag1_acf": _lag_autocorr(resid, 1),
                "lag24_acf": _lag_autocorr(resid, 24),
            }
            if "set" in met_al.columns:
                set_arr = met_al.loc[met_al.index[ok_f], "set"].to_numpy()
                tr_m = set_arr == "training"
                te_m = set_arr == "testing"
                if np.any(tr_m) and np.any(te_m):
                    tr_rmse = float(np.sqrt(np.mean(resid[tr_m] ** 2)))
                    te_rmse = float(np.sqrt(np.mean(resid[te_m] ** 2)))
                    residual_diag["train_rmse"] = tr_rmse
                    residual_diag["test_rmse"] = te_rmse
                    residual_diag["overfit_ratio"] = te_rmse / max(tr_rmse, 1e-12)
        if int(ok_f.sum()) >= 10:
            n_cf = int(ok_f.sum())
            if bool(np.all(ok_f)) and folds:
                sub_folds: list[tuple[np.ndarray, np.ndarray]] | None = folds
            elif n_cf >= 15 and min(3, n_cf // 5) >= 2:
                sub_folds = list(
                    blocked_time_cv(
                        pd.DataFrame({"date": norm_al.index[ok_f]}),
                        n_splits=min(3, n_cf // 5),
                        embargo=0,
                    )
                )
            else:
                sub_folds = None
            cf_diag = counterfactual_separability_audit(
                y_obs[ok_f],
                y_fact[ok_f],
                X_mat[ok_f],
                folds=sub_folds,
                nonlinear=nonlinear_wrf,
                met_names=inferred,
                seed=seed,
            )

    warn_list: list[str] = []
    if np.isfinite(wrf_lin) and wrf_lin < 0.5:
        warn_list.append(
            f"Low linear WRF ({wrf_lin:.2f} < 0.50): substantial meteorological signal remains."
        )
    if wrf_nl is not None and np.isfinite(wrf_nl) and wrf_nl < 0.5:
        warn_list.append(
            f"Low nonlinear WRF ({wrf_nl:.2f} < 0.50): nonlinear meteorological signal remains."
        )
    if over_flags:
        warn_list.append(
            f"Potential meteorological over-correction (sign flip) on: {', '.join(over_flags)}."
        )
    if sup["support_fraction"] < 0.9:
        warn_list.append(
            f"Low marginal support fraction ({sup['support_fraction']:.2f} < 0.90) vs pool."
        )
    if sup["extrapolation_rate"] > 0.1:
        warn_list.append(
            f"High multivariate kNN extrapolation rate ({sup['extrapolation_rate']:.2f} > 0.10) vs pool."
        )
    if mc_se_summary is not None and mc_se_summary["p95_rse"] > 0.05:
        warn_list.append(
            f"Monte Carlo P95 RSE is {mc_se_summary['p95_rse']:.3f} (> 0.05); increase n_samples or use sampling='balanced'."
        )
    if residual_diag is not None and residual_diag.get("overfit_ratio", 0.0) > 1.5:
        warn_list.append(
            f"Train/test residual ratio {residual_diag['overfit_ratio']:.2f} > 1.50 suggests in-sample overfitting."
        )
    if cf_diag is not None:
        rc_lin = float(cf_diag.get("rc_score_linear", float("nan")))
        rc_nl = cf_diag.get("rc_score_nonlinear")
        if (np.isfinite(rc_lin) and rc_lin > 0.10) or (
            rc_nl is not None and np.isfinite(float(rc_nl)) and float(rc_nl) > 0.15
        ):
            warn_list.append(
                f"Factual residual retains meteorological predictability (RC-score={rc_lin:.2f}); "
                "action-contrast bias may affect counterfactuals (consider debias_icn_residuals or refit)."
            )
        rho_raw = float(cf_diag.get("interaction_rho_lb_raw", float("nan")))
        rho_log = float(cf_diag.get("interaction_rho_lb_log", float("nan")))
        if np.isfinite(rho_raw) and rho_raw > 0.15:
            if np.isfinite(rho_log) and rho_log < 0.6 * rho_raw:
                warn_list.append(
                    f"Raw-coordinate residual spread varies strongly with meteorology "
                    f"(rho_lb_raw={rho_raw:.2f} vs rho_lb_log={rho_log:.2f}); "
                    "recommend log_transform=True / paired='log_ratio'."
                )
            else:
                warn_list.append(
                    f"Non-separable meteorology-emission interaction detected in both raw and log "
                    f"coordinates (rho_lb_raw={rho_raw:.2f}); recommend normalise_ox_invariant or paired_mode='rank'."
                )
        floor_rate = float(cf_diag.get("floor_saturation_rate", float("nan")))
        if np.isfinite(floor_rate) and floor_rate > 0.10:
            warn_list.append(
                f"Lower-boundary saturation rate is {100 * floor_rate:.1f}% (> 10%); "
                "disturbance separability weakens near the floor."
            )
        ceiling_rate = float(cf_diag.get("ceiling_saturation_rate", float("nan")))
        if np.isfinite(ceiling_rate) and ceiling_rate > 0.10:
            warn_list.append(
                f"Upper-boundary saturation rate is {100 * ceiling_rate:.1f}% (> 10%); "
                "ceiling clipping weakens disturbance separability."
            )

    return NormalisationDiagnostics(
        wrf_linear=float(wrf_lin),
        wrf_nonlinear=wrf_nl,
        r2_met_observed_linear=float(r2_obs_lin),
        r2_met_normalised_linear=float(r2_norm_lin),
        r2_met_observed_nonlinear=r2_obs_nl,
        r2_met_normalised_nonlinear=r2_norm_nl,
        variable_correlations=corr_df,
        overcorrection_flags=over_flags,
        support_fraction=float(sup["support_fraction"]),
        support_per_col=dict(sup["per_col"]),
        extrapolation_rate=float(sup["extrapolation_rate"]),
        mc_se_summary=mc_se_summary,
        residual_diagnostics=residual_diag,
        counterfactual_diagnostics=cf_diag,
        warnings=warn_list,
    )
