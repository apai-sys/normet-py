# src/normet/causal/variants.py
"""
Alternative synthetic-control / counterfactual estimators.

These complement the ridge-augmented :func:`scm` in ``scm.py``:

- :func:`scm_abadie` — classic Abadie/Hainmueller SCM on raw outcomes
  (simplex weights, no ridge augmentation).
- :func:`did_baseline` — difference-in-differences with the donor pool as
  the control group (parallel-trends counterfactual).
- :func:`scm_mcnnm` — Matrix Completion with Nuclear-Norm Minimisation
  (Athey, Bayati, Doudchenko, Imbens, Khosravi 2021), with optional
  cross-validated regularisation and a randomized-SVD fast path.
- :func:`scm_robust` — Robust Synthetic Control (Amjad, Shah & Shen 2018):
  HSVT de-noising of the donor matrix followed by (optionally ridge) regression.

All return a dict with keys ``synthetic`` (DataFrame indexed by date with
columns ``observed`` / ``synthetic`` / ``effect``) and ``weights`` (Series of
donor weights — for DiD it is uniform; for MC-NNM it is filled with NaN, since
weights are not the natural parameterisation).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ._common import pivot_panel, solve_simplex_weights

log = get_logger(__name__)

__all__ = ["scm_abadie", "did_baseline", "scm_mcnnm", "scm_robust"]


def _pivot_panel(
    df: pd.DataFrame,
    date_col: str,
    unit_col: str,
    outcome_col: str,
    treated_unit: str,
    donors: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Thin positional adapter over :func:`._common.pivot_panel`."""
    return pivot_panel(
        df,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        donors=donors,
    )


# ---------------------------------------------------------------------------
# Classic Abadie SCM
# ---------------------------------------------------------------------------
def scm_abadie(
    df: pd.DataFrame,
    date_col: str = "date",
    unit_col: str = "code",
    outcome_col: str = "poll",
    treated_unit: str | None = None,
    cutoff_date: str | None = None,
    donors: list[str] | None = None,
    allow_negative_weights: bool = False,
) -> dict[str, Any]:
    """
    Classic Abadie/Diamond/Hainmueller SCM with simplex weights.

    Solves ``min_w || y_pre - X_pre w ||²`` with ``w >= 0, sum(w) = 1``.
    """
    if treated_unit is None or cutoff_date is None:
        raise ValueError("`treated_unit` and `cutoff_date` are required.")
    panel, donors = _pivot_panel(df, date_col, unit_col, outcome_col, treated_unit, donors)
    cutoff_ts = pd.to_datetime(cutoff_date)

    pre = panel[panel.index < cutoff_ts][donors + [treated_unit]].dropna(how="any")
    if pre.shape[0] < 2:
        raise ValueError("Not enough complete pre-treatment rows.")

    X_pre = pre[donors].to_numpy()
    y_pre = pre[treated_unit].to_numpy()

    w = solve_simplex_weights(X_pre, y_pre, allow_negative=allow_negative_weights)
    weights = pd.Series(w, index=donors, name="weight")
    syn_full = panel[donors].to_numpy() @ w
    out = pd.DataFrame(
        {"observed": panel[treated_unit].to_numpy(), "synthetic": syn_full},
        index=panel.index,
    )
    out["effect"] = out["observed"] - out["synthetic"]
    return {"synthetic": out, "weights": weights}


# ---------------------------------------------------------------------------
# Difference-in-Differences baseline
# ---------------------------------------------------------------------------
def did_baseline(
    df: pd.DataFrame,
    date_col: str = "date",
    unit_col: str = "code",
    outcome_col: str = "poll",
    treated_unit: str | None = None,
    cutoff_date: str | None = None,
    donors: list[str] | None = None,
) -> dict[str, Any]:
    """
    Parallel-trends counterfactual: a simple two-way fixed-effects baseline.

    For each date ``t``, ``synthetic(t) = mean(treated, pre) + (mean(donors, t)
    − mean(donors, pre))``. Useful as a sanity check against more elaborate SCMs.
    """
    if treated_unit is None or cutoff_date is None:
        raise ValueError("`treated_unit` and `cutoff_date` are required.")
    panel, donors = _pivot_panel(df, date_col, unit_col, outcome_col, treated_unit, donors)
    cutoff_ts = pd.to_datetime(cutoff_date)
    pre = panel[panel.index < cutoff_ts]

    treated_pre_mean = float(np.nanmean(pre[treated_unit].to_numpy()))
    donor_pre_mean = float(np.nanmean(pre[donors].to_numpy()))
    donor_mean_t = panel[donors].mean(axis=1)

    syn = treated_pre_mean + (donor_mean_t - donor_pre_mean)
    out = pd.DataFrame(
        {"observed": panel[treated_unit], "synthetic": syn},
        index=panel.index,
    )
    out["effect"] = out["observed"] - out["synthetic"]

    weights = pd.Series(1.0 / len(donors), index=donors, name="weight")
    return {"synthetic": out, "weights": weights}


# ---------------------------------------------------------------------------
# MC-NNM (Matrix Completion with Nuclear-Norm Minimisation)
# ---------------------------------------------------------------------------
def _soft_threshold_svd(
    M: np.ndarray,
    lam: float,
    max_rank: int | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Singular-value soft-thresholding of ``M`` by ``lam``.

    With ``max_rank`` set (and smaller than ``min(M.shape)``), uses a truncated
    *randomized* SVD over only the leading ``max_rank`` singular triplets rather
    than a full dense SVD — much faster on large panels, where the
    soft-thresholded result is low rank anyway. If every retained singular value
    still exceeds ``lam`` the truncation may have dropped components above the
    threshold; this is logged at debug level.
    """
    if max_rank is not None and 0 < int(max_rank) < min(M.shape):
        from sklearn.utils.extmath import randomized_svd

        U, s, Vt = randomized_svd(M, n_components=int(max_rank), random_state=random_state)
        if s.size and float(s.min()) > lam:
            log.debug(
                "soft-threshold: all %d randomized singular values exceed lam=%.3g; "
                "max_rank may be too small.",
                s.size,
                lam,
            )
    else:
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_thr = np.maximum(s - lam, 0.0)
    return (U * s_thr) @ Vt


def _mcnnm_core(
    Y_full: np.ndarray,
    M_obs: np.ndarray,
    lam: float,
    *,
    max_iter: int,
    tol: float,
    with_unit_fe: bool,
    with_time_fe: bool,
    max_rank: int | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Soft-Impute MC-NNM with two-way FE for one observation mask and ``lam``.

    Returns the completed matrix ``Y_hat = L + FE``.
    """
    T, N = Y_full.shape
    unit_fe = np.zeros(N)
    time_fe = np.zeros(T)
    L = np.zeros((T, N))

    prev_diff = np.inf
    for it in range(int(max_iter)):
        R = Y_full - L
        if with_unit_fe:
            R = R - time_fe[:, None]
            with np.errstate(invalid="ignore"):
                denom = M_obs.sum(axis=0)
                numer = np.where(M_obs, R, 0.0).sum(axis=0)
                unit_fe = np.where(denom > 0, numer / np.maximum(denom, 1), 0.0)
            R = R - unit_fe[None, :]
        if with_time_fe:
            with np.errstate(invalid="ignore"):
                denom = M_obs.sum(axis=1)
                numer = np.where(M_obs, R, 0.0).sum(axis=1)
                time_fe = np.where(denom > 0, numer / np.maximum(denom, 1), 0.0)

        FE = np.zeros_like(L)
        if with_unit_fe:
            FE = FE + unit_fe[None, :]
        if with_time_fe:
            FE = FE + time_fe[:, None]
        target = np.where(M_obs, Y_full - FE, L)  # fill-in unobserved with current L
        L_new = _soft_threshold_svd(target, lam, max_rank=max_rank, random_state=random_state)

        diff = float(np.linalg.norm(L_new - L) / max(1.0, float(np.linalg.norm(L))))
        L = L_new
        if diff < tol:
            log.debug("MC-NNM converged at iter %d (diff=%.2e)", it, diff)
            break
        prev_diff = diff
    else:
        log.warning(
            "MC-NNM did not converge in %d iterations (last diff=%.2e).", max_iter, prev_diff
        )

    FE = np.zeros_like(L)
    if with_unit_fe:
        FE = FE + unit_fe[None, :]
    if with_time_fe:
        FE = FE + time_fe[:, None]
    return L + FE


def _mcnnm_cv_lambda(
    Y_full: np.ndarray,
    M_obs: np.ndarray,
    lam_grid: Sequence[float],
    *,
    cv: int,
    max_iter: int,
    tol: float,
    with_unit_fe: bool,
    with_time_fe: bool,
    max_rank: int | None,
    random_state: int | None,
) -> tuple[float, np.ndarray]:
    """Select ``lam`` minimising held-out MSE over ``cv`` random cell splits.

    Each fold hides a ``1/cv`` slice of the observed cells, refits MC-NNM on the
    remainder for every candidate ``lam``, and scores reconstruction error on the
    held-out cells. Returns the best ``lam`` and the per-grid mean CV error.
    """
    rng = np.random.default_rng(random_state)
    obs_idx = np.argwhere(M_obs)  # (n_obs, 2)
    n_obs = len(obs_idx)
    errs = np.zeros(len(lam_grid))
    n_used = 0
    for _ in range(cv):
        hold = rng.permutation(n_obs)[: max(1, n_obs // cv)]
        val = obs_idx[hold]
        M_train = M_obs.copy()
        M_train[val[:, 0], val[:, 1]] = False
        if not M_train.any():
            continue
        y_val = Y_full[val[:, 0], val[:, 1]]
        for li, lam in enumerate(lam_grid):
            Y_hat = _mcnnm_core(
                Y_full,
                M_train,
                float(lam),
                max_iter=max_iter,
                tol=tol,
                with_unit_fe=with_unit_fe,
                with_time_fe=with_time_fe,
                max_rank=max_rank,
                random_state=random_state,
            )
            pred = Y_hat[val[:, 0], val[:, 1]]
            errs[li] += float(np.mean((y_val - pred) ** 2))
        n_used += 1
    errs /= max(n_used, 1)
    return float(lam_grid[int(np.argmin(errs))]), errs


def scm_mcnnm(
    df: pd.DataFrame,
    date_col: str = "date",
    unit_col: str = "code",
    outcome_col: str = "poll",
    treated_unit: str | None = None,
    cutoff_date: str | None = None,
    donors: list[str] | None = None,
    *,
    lam: float | None = None,
    cv: int = 0,
    lam_grid: Sequence[float] | None = None,
    max_iter: int = 300,
    tol: float = 1e-5,
    with_unit_fe: bool = True,
    with_time_fe: bool = True,
    max_rank: int | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Matrix Completion with Nuclear-Norm Minimisation (Athey et al. 2021).

    Treats the (date × unit) outcome panel as a low-rank matrix with two-way
    fixed effects. The treated unit's post-period entries are masked out and
    imputed by iterative singular-value soft-thresholding plus alternating FE
    updates.

    Parameters
    ----------
    lam : float, optional
        Nuclear-norm regularisation strength. If given, used directly (disables
        CV). If ``None`` and ``cv <= 1``, defaults to the ``0.1 * sigma_max``
        heuristic of the observed matrix.
    cv : int, default 0
        If ``> 1`` and ``lam is None``, choose ``lam`` by ``cv``-fold
        cross-validation over held-out observed cells (recommended — the fixed
        heuristic otherwise picks an essentially arbitrary rank).
    lam_grid : sequence of float, optional
        Candidate ``lam`` values for CV. Defaults to 8 log-spaced values from
        ``1e-3 * sigma_max`` up to ``sigma_max``.
    max_iter : int, default 300
    tol : float, default 1e-5
        Frobenius-norm relative change tolerance for convergence.
    with_unit_fe, with_time_fe : bool, default True
        Include row/column (unit/time) fixed effects alongside the low-rank part.
    max_rank : int, optional
        If set, use a truncated randomized SVD with this many components inside
        the soft-threshold step (faster on large, low-rank panels).
    random_state : int, optional
        Seed for the CV splits and randomized SVD (reproducibility).

    Notes
    -----
    Returned ``weights`` is filled with NaN (MC-NNM is not weight-based).
    The reconstructed low-rank component for the treated unit is used as
    ``synthetic``.
    """
    if treated_unit is None or cutoff_date is None:
        raise ValueError("`treated_unit` and `cutoff_date` are required.")
    panel, donors = _pivot_panel(df, date_col, unit_col, outcome_col, treated_unit, donors)
    cutoff_ts = pd.to_datetime(cutoff_date)

    cols_order = donors + [treated_unit]
    Y_full = panel[cols_order].to_numpy(dtype=float)
    T, N = Y_full.shape
    treated_idx = N - 1

    # Mask: observed everywhere except (treated, post) which we want to predict.
    pre_mask = np.asarray(panel.index < cutoff_ts)
    M_obs = ~np.isnan(Y_full)
    M_obs[~pre_mask, treated_idx] = False  # hide post treated values

    if not M_obs.any():
        raise ValueError("Observation mask is empty; nothing to fit.")

    # --- Choose lam: explicit > CV > heuristic ---
    if lam is None:
        Y0 = np.where(M_obs, Y_full, 0.0)
        try:
            sigma_max = float(np.linalg.svd(Y0, compute_uv=False)[0])
        except Exception:
            sigma_max = 1.0
        sigma_max = sigma_max if sigma_max > 0 else 1.0
        if cv and cv > 1:
            grid = (
                np.geomspace(1e-3 * sigma_max, sigma_max, 8)
                if lam_grid is None
                else np.asarray(lam_grid, dtype=float)
            )
            lam, _cv_errs = _mcnnm_cv_lambda(
                Y_full,
                M_obs,
                list(grid),
                cv=int(cv),
                max_iter=max_iter,
                tol=tol,
                with_unit_fe=with_unit_fe,
                with_time_fe=with_time_fe,
                max_rank=max_rank,
                random_state=random_state,
            )
            log.info("MC-NNM selected lam=%.4g via %d-fold CV.", lam, int(cv))
        else:
            lam = 0.1 * sigma_max

    Y_hat = _mcnnm_core(
        Y_full,
        M_obs,
        float(lam),
        max_iter=max_iter,
        tol=tol,
        with_unit_fe=with_unit_fe,
        with_time_fe=with_time_fe,
        max_rank=max_rank,
        random_state=random_state,
    )
    syn_treated = Y_hat[:, treated_idx]

    out = pd.DataFrame(
        {"observed": panel[treated_unit].to_numpy(), "synthetic": syn_treated},
        index=panel.index,
    )
    out["effect"] = out["observed"] - out["synthetic"]

    weights = pd.Series(np.nan, index=donors, name="weight")
    return {"synthetic": out, "weights": weights, "rank_lambda": float(lam)}


# ---------------------------------------------------------------------------
# Robust SCM (HSVT de-noising + regression)
# ---------------------------------------------------------------------------
def _hsvt(
    M: np.ndarray, rank: int | None = None, energy: float = 0.95
) -> tuple[np.ndarray, int, np.ndarray]:
    """Hard singular-value thresholding (HSVT).

    Keep the top ``rank`` singular values (or the fewest values capturing
    ``energy`` of the squared-singular-value spectrum) and zero the rest,
    returning the de-noised matrix, the retained rank, and the full spectrum.
    """
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    if s.size == 0:
        return M.copy(), 0, s
    if rank is not None:
        k = int(np.clip(rank, 1, s.size))
    else:
        # smallest k whose cumulative energy reaches `energy`
        e = np.cumsum(s**2) / np.sum(s**2)
        k = int(np.searchsorted(e, energy) + 1)
        k = int(np.clip(k, 1, s.size))
    s_keep = s.copy()
    s_keep[k:] = 0.0
    return (U * s_keep) @ Vt, k, s


def scm_robust(
    df: pd.DataFrame,
    date_col: str = "date",
    unit_col: str = "code",
    outcome_col: str = "poll",
    treated_unit: str | None = None,
    cutoff_date: str | None = None,
    donors: list[str] | None = None,
    *,
    rank: int | None = None,
    energy: float = 0.95,
    alpha: float = 0.0,
    rescale_missing: bool = True,
) -> dict[str, Any]:
    """
    Robust Synthetic Control (Amjad, Shah & Shen 2018).

    De-noises the donor outcome matrix via hard singular-value thresholding
    (HSVT), then learns unconstrained (optionally ridge-regularised) donor
    weights by regressing the treated pre-period outcome on the de-noised
    donors. The synthetic series is the de-noised donor matrix projected
    through those weights. Unlike :func:`scm_abadie`, weights are **not**
    simplex-constrained — the SVD de-noising is what controls overfitting.

    Parameters
    ----------
    rank : int, optional
        Number of singular values to retain in HSVT. If ``None`` (default),
        chosen automatically as the smallest rank capturing ``energy`` of the
        spectral energy.
    energy : float, default 0.95
        Target cumulative spectral energy for automatic rank selection
        (ignored when ``rank`` is given).
    alpha : float, default 0.0
        Ridge penalty on donor weights (the intercept is never penalised).
        ``0.0`` gives ordinary least squares on the de-noised donors.
    rescale_missing : bool, default True
        Divide the recovered low-rank component by the observed fraction ``p``
        to debias HSVT under (approximately) missing-at-random gaps, following
        the robust-SCM construction.

    Returns
    -------
    dict
        A dictionary with the structure::

            {"synthetic": DataFrame[observed, synthetic, effect],
             "weights": Series, "rank": int, "intercept": float}
    """
    if treated_unit is None or cutoff_date is None:
        raise ValueError("`treated_unit` and `cutoff_date` are required.")
    panel, donors = _pivot_panel(df, date_col, unit_col, outcome_col, treated_unit, donors)
    cutoff_ts = pd.to_datetime(cutoff_date)

    X = panel[donors].to_numpy(dtype=float)  # (T, J)
    y = panel[treated_unit].to_numpy(dtype=float)  # (T,)
    pre_mask = np.asarray(panel.index < cutoff_ts)

    # --- De-noise the donor matrix: centre per donor, fill gaps with 0,
    #     HSVT, debias by 1/p, then add the means back. ---
    obs = np.isfinite(X)
    col_mean = np.where(obs.any(axis=0), np.nanmean(np.where(obs, X, np.nan), axis=0), 0.0)
    Xc = np.where(obs, X - col_mean, 0.0)
    p = float(obs.mean()) if rescale_missing else 1.0
    p = p if p > 0 else 1.0
    Xc_hat, k, _ = _hsvt(Xc, rank=rank, energy=energy)
    X_hat = Xc_hat / p + col_mean

    # --- Regress treated pre-period outcome on de-noised donors (with intercept) ---
    fit_rows = pre_mask & np.isfinite(y)
    if fit_rows.sum() < 2:
        raise ValueError("Not enough observed pre-treatment rows to fit robust SCM.")
    A = np.hstack([np.ones((int(fit_rows.sum()), 1)), X_hat[fit_rows]])  # (T_pre, J+1)
    b = y[fit_rows]

    J = len(donors)
    if alpha and alpha > 0:
        # Ridge in closed form; do not penalise the intercept column.
        pen = np.eye(J + 1) * float(alpha)
        pen[0, 0] = 0.0
        coef = np.linalg.solve(A.T @ A + pen, A.T @ b)
    else:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)

    intercept = float(coef[0])
    w = coef[1:]
    weights = pd.Series(w, index=donors, name="weight")

    syn = intercept + X_hat @ w
    out = pd.DataFrame(
        {"observed": panel[treated_unit].to_numpy(), "synthetic": syn},
        index=panel.index,
    )
    out["effect"] = out["observed"] - out["synthetic"]

    log.info("Robust SCM | retained rank=%d | donors=%d | p_obs=%.3f", k, J, p)
    return {"synthetic": out, "weights": weights, "rank": int(k), "intercept": intercept}
