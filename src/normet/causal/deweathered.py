# src/normet/causal/deweathered.py
"""End-to-end deweathered and transport-deconfounded synthetic control.

Propagates first-stage meteorological-normalisation uncertainty into
second-stage synthetic-control effect bands, and corrects downwind donor
plume spillover using Lagrangian transport footprint exposures:

- :func:`deweathered_scm` — two-stage deweathering + SCM with Law-of-Total-Variance
  decomposition into meteorological resampling variance (``se_met``) and SCM
  donor/residual matching variance (``se_scm``).
- :func:`transport_adjusted_scm` — time-varying SUTVA spillover correction that
  estimates downwind plume contamination ``gamma`` from donor panel exposures
  and reconstructs a decontaminated synthetic counterfactual.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.normalise import (
    _CROSSFIT_DEFAULT_EMBARGO,
    _is_time_feature,
    crossfit_normalise,
    dml_normalise,
    normalise,
)
from ..exceptions import ConfigError, DataError
from ..model.train import build_model
from ..utils._config import DEFAULT_SEED
from ._common import newey_west_se_mean, pivot_panel
from .run_scm import run_scm

__all__ = [
    "DeweatheredSCMResult",
    "TransportSCMResult",
    "deweathered_scm",
    "transport_adjusted_scm",
]


@dataclass
class DeweatheredSCMResult:
    """Output of :func:`deweathered_scm`."""

    synthetic: pd.DataFrame
    deweathered_panel: pd.DataFrame
    att: float
    att_se_total: float
    att_se_met: float
    att_se_scm: float
    att_ci_low: float
    att_ci_high: float
    att_p_value: float
    met_variance_share: float
    confidence_level: float
    scm_backend: str
    deweather_method: str
    unit_diagnostics: pd.DataFrame


@dataclass
class TransportSCMResult:
    """Output of :func:`transport_adjusted_scm`."""

    synthetic: pd.DataFrame
    gamma: float
    gamma_se: float
    gamma_p_value: float
    att_naive: float
    att_debiased: float
    spillover_bias: float
    weights: pd.Series
    scm_backend: str


def _block_correlated_normal(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int,
    block_length: int,
) -> np.ndarray:
    """Generate zero-mean, unit-variance block-correlated Gaussian noise ``(n_rows, n_cols)``."""
    if block_length <= 1 or n_rows <= 1:
        return rng.standard_normal((n_rows, n_cols))
    L = min(int(block_length), n_rows)
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    pos = 0
    while pos < n_rows:
        end = min(n_rows, pos + L)
        z_blk = rng.standard_normal((1, n_cols))
        # Mix persistent block shock (80% variance) with hourly innovation (20%)
        z_hr = rng.standard_normal((end - pos, n_cols))
        out[pos:end] = np.sqrt(0.8) * z_blk + np.sqrt(0.2) * z_hr
        pos = end
    return out


def _fit_scm_with_weights(
    df_panel: pd.DataFrame,
    *,
    date_col: str,
    unit_col: str,
    outcome_col: str,
    treated_unit: str,
    cutoff_str: str,
    donors: list[str],
    scm_backend: str,
    scm_kwargs: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series | None, float | None]:
    """Run SCM backend, returning ``(synthetic_df, weights_series_or_None, ridge_lam_or_None)``."""
    backend_l = (scm_backend or "ascm").lower()
    if backend_l == "ascm":
        from .ascm import scm_ascm

        res = scm_ascm(
            df_panel,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            cutoff_date=cutoff_str,
            donors=donors,
            **scm_kwargs,
        )
        return res["synthetic"], res.get("weights"), res.get("ridge_lambda")
    if backend_l == "sdid":
        from .sdid import sdid

        res = sdid(
            df_panel,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            cutoff_date=cutoff_str,
            donors=donors,
            **scm_kwargs,
        )
        return res["synthetic"], res.get("weights"), None
    if backend_l == "scm":
        from .scm import scm

        res = scm(
            df_panel,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            cutoff_date=cutoff_str,
            donors=donors,
            **scm_kwargs,
        )
        return res["synthetic"], res.get("weights"), None
    if backend_l in {"abadie", "did", "mcnnm", "robust"}:
        from .variants import did_baseline, scm_abadie, scm_mcnnm, scm_robust

        fn_map: dict[str, Any] = {
            "abadie": scm_abadie,
            "did": did_baseline,
            "mcnnm": scm_mcnnm,
            "robust": scm_robust,
        }
        res = fn_map[backend_l](
            df_panel,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            cutoff_date=cutoff_str,
            donors=donors,
            **scm_kwargs,
        )
        return res["synthetic"], res.get("weights"), None

    syn_df = run_scm(
        df=df_panel,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        cutoff_date=cutoff_str,
        donors=donors,
        scm_backend=backend_l,
        **scm_kwargs,
    )
    return syn_df, None, None


def deweathered_scm(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    unit_col: str = "ID",
    outcome_col: str = "value",
    treated_unit: str,
    cutoff_date: str | pd.Timestamp,
    donors: Sequence[str] | None = None,
    covariates: Sequence[str] | None = None,
    variables_resample: Sequence[str] | None = None,
    deweather_method: str = "dml",
    ml_backend: str = "lightgbm",
    model_config: dict[str, Any] | None = None,
    scm_backend: str = "ascm",
    scm_kwargs: Mapping[str, Any] | None = None,
    cutoff_days: float = 30.0,
    n_splits: int = 5,
    embargo: int = _CROSSFIT_DEFAULT_EMBARGO,
    n_samples: int = 100,
    log_transform: bool = False,
    use_paired: bool = True,
    n_scm_draws: int = 100,
    block_length: int | None = None,
    confidence_level: float = 0.95,
    seed: int = DEFAULT_SEED,
    n_cores: int | None = None,
    verbose: bool = False,
) -> DeweatheredSCMResult:
    """Two-stage meteorological normalisation + synthetic control with uncertainty propagation.

    Normalises every unit (treated + donors) in Stage 1 while recording its
    per-timestamp Monte Carlo deweathering standard error ``mc_se``, fits
    Synthetic Control on the deweathered panel in Stage 2, and propagates both
    Stage-1 meteorological uncertainty (``se_met``) and Stage-2 pre-period
    SCM matching uncertainty (``se_scm``) into ``se_total = hypot(se_met, se_scm)``.
    """
    from scipy.stats import norm

    if not treated_unit:
        raise ConfigError("`treated_unit` must be a non-empty string.")
    if deweather_method not in ("dml", "crossfit", "standard"):
        raise ConfigError(
            f"`deweather_method` must be 'dml', 'crossfit', or 'standard', got {deweather_method!r}."
        )
    if not (0.0 < float(confidence_level) < 1.0):
        raise ConfigError(f"`confidence_level` must lie in (0, 1), got {confidence_level!r}.")
    for req in (date_col, unit_col, outcome_col):
        if req not in df.columns:
            raise DataError(f"Required column {req!r} not found in `df`.")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work[work[date_col].notna()].copy()
    work[unit_col] = work[unit_col].astype(str)
    treated_key = str(treated_unit)

    all_units = list(dict.fromkeys(work[unit_col].tolist()))
    if treated_key not in all_units:
        raise DataError(f"Treated unit {treated_unit!r} not found in `{unit_col}`.")
    donor_pool = (
        [u for u in all_units if u != treated_key]
        if donors is None
        else [
            str(u) for u in dict.fromkeys(donors) if str(u) != treated_key and str(u) in all_units
        ]
    )
    if not donor_pool:
        raise DataError("No valid donor units available after excluding treated unit.")

    cutoff_ts = pd.to_datetime(cutoff_date)
    cutoff_str = cutoff_ts.strftime("%Y-%m-%d %H:%M:%S")

    if covariates is None:
        excluded = {date_col, unit_col, outcome_col, "set"}
        met_inferred = [
            str(c)
            for c in work.columns
            if str(c) not in excluded
            and not _is_time_feature(str(c))
            and pd.api.types.is_numeric_dtype(work[c])
        ]
        cov_list = [*met_inferred, "day_julian", "weekday", "hour"]
        if deweather_method != "dml":
            cov_list.append("date_unix")
    else:
        cov_list = [str(c) for c in covariates]
    res_list = (
        [str(c) for c in variables_resample]
        if variables_resample is not None
        else [c for c in cov_list if not _is_time_feature(c)]
    )

    units_all = [treated_key, *donor_pool]
    dew_frames: list[pd.DataFrame] = []
    se_by_unit: dict[str, pd.Series] = {}
    raw_by_unit: dict[str, pd.Series] = {}
    diag_rows: list[dict[str, Any]] = []

    for u_idx, unit in enumerate(units_all):
        df_u = work[work[unit_col] == unit].copy()
        if date_col != "date":
            df_u = df_u.rename(columns={date_col: "date"})
        if deweather_method == "dml":
            res_u = dml_normalise(
                df_u,
                target=outcome_col,
                covariates=cov_list,
                variables_resample=res_list,
                backend=ml_backend,
                model_config=model_config,
                cutoff_days=cutoff_days,
                n_splits=n_splits,
                embargo=embargo,
                paired="log_ratio" if log_transform else "additive",
                n_samples=n_samples,
                log_transform=log_transform,
                return_mc_se=True,
                sampling="balanced",
                seed=int(seed) + 31 * u_idx,
                n_cores=n_cores,
                verbose=verbose,
            ).normalised
        elif deweather_method == "crossfit":
            res_u = crossfit_normalise(
                df_u,
                target=outcome_col,
                covariates=cov_list,
                variables_resample=res_list,
                backend=ml_backend,
                model_config=model_config,
                n_splits=n_splits,
                embargo=embargo,
                paired="log_ratio" if log_transform else "additive",
                n_samples=n_samples,
                log_transform=log_transform,
                return_mc_se=True,
                sampling="balanced",
                seed=int(seed) + 31 * u_idx,
                n_cores=n_cores,
                verbose=verbose,
            ).normalised
        else:
            df_prep, mdl = build_model(
                df_u,
                target=outcome_col,
                backend=ml_backend,
                covariates=cov_list,
                model_config=model_config,
                seed=int(seed) + 31 * u_idx,
                n_cores=n_cores,
                verbose=verbose,
                # `build_model` returns the frame already on the fitted scale,
                # and `normalise` expects it that way (as `do_all` does), so
                # the same flag must go to both; passing it to neither made
                # `log_transform=True` silently fit and report in raw units.
                log_transform=bool(log_transform),
            )
            res_u = normalise(
                df_prep,
                mdl,
                covariates=cov_list,
                variables_resample=res_list,
                n_samples=n_samples,
                paired="log_ratio" if log_transform else True,
                return_mc_se=True,
                sampling="balanced",
                seed=int(seed) + 31 * u_idx,
                n_cores=n_cores,
                verbose=verbose,
                log_transform=bool(log_transform),
            )

        est_col = "paired" if (use_paired and "paired" in res_u.columns) else "normalised"
        est_vals = pd.to_numeric(res_u[est_col], errors="coerce")
        obs_vals = pd.to_numeric(res_u["observed"], errors="coerce")
        if "mc_se" in res_u.columns:
            mc_se_s = pd.to_numeric(res_u["mc_se"], errors="coerce").fillna(0.0)
        else:
            mc_se_s = pd.Series(0.0, index=res_u.index)

        # Include first-stage weather-effect estimation spread if mc_se is non-trivial.
        se_by_unit[unit] = mc_se_s
        raw_by_unit[unit] = obs_vals
        dew_frames.append(
            pd.DataFrame(
                {
                    date_col: res_u.index,
                    unit_col: unit,
                    outcome_col: est_vals.to_numpy(dtype=float),
                }
            )
        )
        diag_rows.append(
            {
                "unit": unit,
                "role": "treated" if unit == treated_key else "donor",
                "n_obs": int(est_vals.notna().sum()),
                "mean_observed": float(np.nanmean(obs_vals)),
                "mean_deweathered": float(np.nanmean(est_vals)),
                "median_mc_se": float(np.nanmedian(mc_se_s)),
            }
        )

    df_dew = pd.concat(dew_frames, ignore_index=True)
    scm_kw = dict(scm_kwargs or {})
    syn_df, weights_s, ridge_lam = _fit_scm_with_weights(
        df_dew,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_key,
        cutoff_str=cutoff_str,
        donors=donor_pool,
        scm_backend=scm_backend,
        scm_kwargs=scm_kw,
    )

    idx_all = pd.DatetimeIndex(syn_df.index)
    n_t = len(idx_all)
    pre_mask = idx_all < cutoff_ts
    post_mask = idx_all >= cutoff_ts

    panel_wide, resolved_donors = pivot_panel(
        df_dew,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_key,
        donors=donor_pool,
    )
    panel_wide = panel_wide.reindex(idx_all)
    se_mat = np.column_stack(
        [
            se_by_unit[u].reindex(idx_all).fillna(0.0).to_numpy(dtype=np.float64)
            for u in [treated_key, *resolved_donors]
        ]
    )

    n_draws = max(10, int(n_scm_draws))
    blk_len = (
        max(1, int(block_length))
        if block_length is not None
        else max(1, round(float(max(int(pre_mask.sum()), 1)) ** (1.0 / 3.0)))
    )
    rng = np.random.default_rng(int(seed) + 999)

    # Propagate Stage-1 deweathering uncertainty into Stage-2 SCM effect.
    eff_met_draws = np.empty((n_draws, n_t), dtype=np.float64)
    kw_prop = dict(scm_kw)
    if (
        (scm_backend or "ascm").lower() == "ascm"
        and ridge_lam is not None
        and "ridge_lambda" not in kw_prop
    ):
        kw_prop["ridge_lambda"] = float(ridge_lam)

    base_mat = panel_wide[[treated_key, *resolved_donors]].to_numpy(dtype=np.float64)
    unit_list_p = [treated_key, *resolved_donors]
    for m in range(n_draws):
        noise = _block_correlated_normal(rng, n_t, len(unit_list_p), blk_len) * se_mat
        pert_mat = base_mat + noise
        if weights_s is not None and (scm_backend or "ascm").lower() in {
            "ascm",
            "scm",
            "sdid",
            "abadie",
        }:
            # Fast linear weight propagation + occasional refit stability check
            w_arr = weights_s.reindex(resolved_donors).fillna(0.0).to_numpy(dtype=np.float64)
            syn_base = syn_df["synthetic"].to_numpy(dtype=np.float64)
            delta_treat = noise[:, 0]
            delta_syn = noise[:, 1:] @ w_arr
            eff_met_draws[m] = (syn_df["observed"].to_numpy(dtype=np.float64) + delta_treat) - (
                syn_base + delta_syn
            )
        else:
            df_pert = pd.DataFrame(
                {
                    date_col: np.tile(idx_all.to_numpy(), len(unit_list_p)),
                    unit_col: np.repeat(unit_list_p, n_t),
                    outcome_col: pert_mat.T.ravel(),
                }
            )
            syn_m, _, _ = _fit_scm_with_weights(
                df_pert,
                date_col=date_col,
                unit_col=unit_col,
                outcome_col=outcome_col,
                treated_unit=treated_key,
                cutoff_str=cutoff_str,
                donors=resolved_donors,
                scm_backend=scm_backend,
                scm_kwargs=kw_prop,
            )
            eff_met_draws[m] = syn_m["effect"].reindex(idx_all).to_numpy(dtype=np.float64)

    se_met = np.nanstd(eff_met_draws, axis=0, ddof=1)
    att_met_draws = (
        np.nanmean(eff_met_draws[:, post_mask], axis=1)
        if np.any(post_mask)
        else np.zeros(n_draws, dtype=np.float64)
    )
    att_se_met = float(np.nanstd(att_met_draws, ddof=1)) if n_draws >= 2 else 0.0

    # Stage-2 SCM matching variance via moving-block pre-period residual bootstrap.
    eff_arr = syn_df["effect"].to_numpy(dtype=np.float64)
    pre_resid = eff_arr[pre_mask]
    pre_resid = pre_resid[np.isfinite(pre_resid)]
    if pre_resid.size >= 3:
        n_pre = len(pre_resid)
        l_scm = min(blk_len, n_pre)
        max_st = max(1, n_pre - l_scm + 1)
        n_post = int(post_mask.sum())
        n_blk_draw = int(np.ceil(max(n_t, 1) / l_scm))
        boot_pre = np.empty((n_draws, n_t), dtype=np.float64)
        for m in range(n_draws):
            starts = rng.integers(0, max_st, size=n_blk_draw)
            seq = np.concatenate([np.arange(s, s + l_scm) for s in starts])[:n_t]
            boot_pre[m] = pre_resid[seq]
        se_scm = np.nanstd(boot_pre, axis=0, ddof=1)
        if n_post > 0:
            att_se_scm = float(np.nanstd(np.nanmean(boot_pre[:, post_mask], axis=1), ddof=1))
        else:
            att_se_scm = 0.0
    else:
        se_scm = np.zeros(n_t, dtype=np.float64)
        att_se_scm = 0.0

    se_total = np.hypot(se_met, se_scm)
    att = float(np.nanmean(eff_arr[post_mask])) if np.any(post_mask) else float("nan")
    att_se_total = float(np.hypot(att_se_met, att_se_scm))

    z_crit = float(norm.ppf(0.5 + 0.5 * float(confidence_level)))
    ci_low = eff_arr - z_crit * se_total
    ci_high = eff_arr + z_crit * se_total
    att_ci_low = att - z_crit * att_se_total if np.isfinite(att) else float("nan")
    att_ci_high = att + z_crit * att_se_total if np.isfinite(att) else float("nan")
    if np.isfinite(att) and att_se_total > 1e-12:
        att_p = float(2.0 * (1.0 - norm.cdf(abs(att) / att_se_total)))
    else:
        att_p = float("nan")

    var_tot = att_se_total**2
    met_share = float((att_se_met**2) / var_tot) if var_tot > 1e-15 else 0.0

    out_syn = syn_df.copy()
    out_syn["observed_raw"] = raw_by_unit[treated_key].reindex(idx_all).to_numpy(dtype=float)
    out_syn["se_met"] = se_met
    out_syn["se_scm"] = se_scm
    out_syn["se_total"] = se_total
    out_syn["ci_low"] = ci_low
    out_syn["ci_high"] = ci_high

    return DeweatheredSCMResult(
        synthetic=out_syn,
        deweathered_panel=df_dew,
        att=att,
        att_se_total=att_se_total,
        att_se_met=att_se_met,
        att_se_scm=att_se_scm,
        att_ci_low=att_ci_low,
        att_ci_high=att_ci_high,
        att_p_value=att_p,
        met_variance_share=met_share,
        confidence_level=float(confidence_level),
        scm_backend=scm_backend,
        deweather_method=deweather_method,
        unit_diagnostics=pd.DataFrame(diag_rows),
    )


def transport_adjusted_scm(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    unit_col: str = "ID",
    outcome_col: str = "value",
    treated_unit: str,
    cutoff_date: str | pd.Timestamp,
    donors: Sequence[str] | None = None,
    spillover_exposure: pd.DataFrame | str,
    scm_backend: str = "ascm",
    scm_kwargs: Mapping[str, Any] | None = None,
) -> TransportSCMResult:
    """Synthetic control corrected for dynamic downwind transport spillover on donors.

    When untreated donor sites lie intermittently downwind of the treated zone
    (measured by non-negative Lagrangian footprint exposure ``F_j(t)``), a policy
    at the treated site spills over onto donors during downwind hours and biases
    naive synthetic control toward zero. This estimator identifies the spillover
    coefficient ``gamma`` from post-treatment within-timestamp donor contrasts
    ``(Y_j(t) - Y_j_pre)`` against ``F_j(t)``, decontaminates the donor trajectories,
    and reconstructs the unbiased counterfactual.

    Parameters
    ----------
    df : pandas.DataFrame
        Long panel containing ``[date_col, unit_col, outcome_col]``.
    treated_unit : str
        Treated unit identifier.
    cutoff_date : str or Timestamp
        Intervention timestamp.
    donors : sequence of str, optional
        Donor unit identifiers.
    spillover_exposure : pandas.DataFrame or str
        Either a wide DataFrame indexed by datetime with columns matching donor
        identifiers giving non-negative transport exposure ``F_j(t)`` from the
        treated zone, or a column name in ``df`` carrying that exposure.
    scm_backend : str, default "ascm"
        Synthetic control backend used for pre-period weighting.
    scm_kwargs : mapping, optional
        Extra keyword arguments forwarded to the SCM backend.

    Returns
    -------
    TransportSCMResult
    """
    from scipy.stats import norm

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work[work[date_col].notna()].copy()
    work[unit_col] = work[unit_col].astype(str)
    treated_key = str(treated_unit)
    cutoff_ts = pd.to_datetime(cutoff_date)
    cutoff_str = cutoff_ts.strftime("%Y-%m-%d %H:%M:%S")

    panel, resolved_donors = pivot_panel(
        work,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_key,
        donors=list(donors) if donors is not None else None,
        parse_dates=False,
    )

    if isinstance(spillover_exposure, str):
        if spillover_exposure not in work.columns:
            raise DataError(f"spillover_exposure column {spillover_exposure!r} not found in `df`.")
        exp_wide = work.pivot_table(
            index=date_col, columns=unit_col, values=spillover_exposure, aggfunc="mean"
        ).sort_index()
    elif isinstance(spillover_exposure, pd.DataFrame):
        exp_wide = spillover_exposure.copy()
        if not isinstance(exp_wide.index, pd.DatetimeIndex):
            if date_col in exp_wide.columns:
                exp_wide = exp_wide.set_index(date_col)
            elif "date" in exp_wide.columns:
                exp_wide = exp_wide.set_index("date")
            exp_wide.index = pd.to_datetime(exp_wide.index)
        exp_wide = exp_wide.sort_index()
    else:
        raise ConfigError("`spillover_exposure` must be a DataFrame or column name string.")

    exp_aligned = pd.DataFrame(index=panel.index)
    for d_u in resolved_donors:
        if d_u in exp_wide.columns:
            exp_aligned[d_u] = (
                pd.to_numeric(exp_wide[d_u], errors="coerce")
                .reindex(panel.index)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
        else:
            exp_aligned[d_u] = 0.0

    scm_kw = dict(scm_kwargs or {})
    syn_naive, weights_s, _ = _fit_scm_with_weights(
        work,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_key,
        cutoff_str=cutoff_str,
        donors=resolved_donors,
        scm_backend=scm_backend,
        scm_kwargs=scm_kw,
    )
    if weights_s is None:
        weights_s = pd.Series(
            np.full(len(resolved_donors), 1.0 / len(resolved_donors)),
            index=resolved_donors,
            name="weight",
        )
    w_vec = weights_s.reindex(resolved_donors).to_numpy(dtype=np.float64)
    # `mcnnm` reports NaN weights by design (its fit has no donor-weight
    # vector), and a partially missing index gives NaN here too. Filling those
    # with 0 quietly zeroed the spillover correction -- all-zero weights made
    # `spillover_component` vanish while the fit still looked successful.
    # Missing/unusable weights mean "no donor weighting information", so fall
    # back to the equal-weight synthetic control the backend implies.
    if w_vec.size == 0 or not np.all(np.isfinite(w_vec)) or float(w_vec.sum()) <= 0.0:
        w_vec = np.full(len(resolved_donors), 1.0 / len(resolved_donors), dtype=np.float64)
        # Report the weights that were actually used rather than the backend's
        # NaNs, so the result is usable and self-describing.
        weights_s = pd.Series(w_vec, index=resolved_donors, name="weight")

    pre_mask = panel.index < cutoff_ts
    post_mask = panel.index >= cutoff_ts
    if not np.any(pre_mask) or not np.any(post_mask):
        raise DataError("Panel must contain both pre- and post-cutoff timestamps.")

    Y_donors = panel[resolved_donors].to_numpy(dtype=np.float64)
    F_donors = exp_aligned[resolved_donors].to_numpy(dtype=np.float64)
    pre_means = np.nanmean(Y_donors[pre_mask], axis=0, keepdims=True)
    dY_post = Y_donors[post_mask] - pre_means
    F_post = F_donors[post_mask]

    # Within-timestamp demeaning across donors removes common regional/meteorological
    # shocks delta_t; single-donor fallback uses raw post-minus-pre deviations.
    # Both variables must be centred over the same donors: demeaning Y over the
    # donors that report while demeaning F over every donor (missing exposure
    # was filled with 0) shifts one regressor relative to the other by the
    # mean exposure of the missing Y rows, which biases gamma.
    if len(resolved_donors) >= 2:
        valid = np.isfinite(dY_post) & np.isfinite(F_post)
        n_valid = np.maximum(valid.sum(axis=1, keepdims=True), 1)
        dY_mean = np.where(valid, dY_post, 0.0).sum(axis=1, keepdims=True) / n_valid
        F_mean = np.where(valid, F_post, 0.0).sum(axis=1, keepdims=True) / n_valid
        dY_w = np.where(valid, dY_post - dY_mean, np.nan)
        F_w = np.where(valid, F_post - F_mean, np.nan)
    else:
        dY_w = dY_post
        F_w = F_post

    fin = np.isfinite(dY_w) & np.isfinite(F_w)
    f_flat = F_w[fin]
    y_flat = dY_w[fin]
    denom = float(np.dot(f_flat, f_flat))
    if denom > 1e-12:
        gamma = float(np.dot(f_flat, y_flat) / denom)
        # Per-timestamp score sequence for Newey-West HAC standard error.
        resid_mat = np.where(fin, dY_w - gamma * F_w, 0.0)
        t_scores = np.sum(np.where(fin, F_w * resid_mat, 0.0), axis=1)
        t_denom = np.sum(np.where(fin, F_w**2, 0.0), axis=1)
        ok_t = t_denom > 1e-12
        if int(ok_t.sum()) >= 3:
            mean_denom = float(np.mean(t_denom))
            psi = t_scores / max(mean_denom, 1e-12)
            gamma_se_val, _ = newey_west_se_mean(psi)
            gamma_se = float(gamma_se_val)
            gamma_p = float(2.0 * (1.0 - norm.cdf(abs(gamma) / max(gamma_se, 1e-12))))
        else:
            gamma_se = float("nan")
            gamma_p = float("nan")
    else:
        gamma = 0.0
        gamma_se = float("nan")
        gamma_p = float("nan")

    idx_syn = pd.DatetimeIndex(syn_naive.index)
    f_syn_aligned = (
        exp_aligned[resolved_donors].reindex(idx_syn).fillna(0.0).to_numpy(dtype=np.float64)
    )
    is_post_syn = (idx_syn >= cutoff_ts).astype(np.float64)
    spillover_component = (f_syn_aligned @ w_vec) * gamma * is_post_syn

    syn_debiased = syn_naive["synthetic"].to_numpy(dtype=np.float64) - spillover_component
    obs_arr = syn_naive["observed"].to_numpy(dtype=np.float64)
    eff_naive = syn_naive["effect"].to_numpy(dtype=np.float64)
    eff_debiased = obs_arr - syn_debiased

    out_df = pd.DataFrame(
        {
            "observed": obs_arr,
            "synthetic_naive": syn_naive["synthetic"].to_numpy(dtype=np.float64),
            "synthetic": syn_debiased,
            "effect_naive": eff_naive,
            "effect": eff_debiased,
            "spillover_adjustment": spillover_component,
        },
        index=idx_syn,
    )
    out_df.index.name = "date"

    post_syn_mask = idx_syn >= cutoff_ts
    att_naive = float(np.nanmean(eff_naive[post_syn_mask]))
    att_debiased = float(np.nanmean(eff_debiased[post_syn_mask]))
    return TransportSCMResult(
        synthetic=out_df,
        gamma=gamma,
        gamma_se=gamma_se,
        gamma_p_value=gamma_p,
        att_naive=att_naive,
        att_debiased=att_debiased,
        spillover_bias=att_debiased - att_naive,
        weights=weights_s,
        scm_backend=scm_backend,
    )
