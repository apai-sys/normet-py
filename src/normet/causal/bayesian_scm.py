# src/normet/causal/bayesian_scm.py
"""
Bayesian Synthetic Control Method.

Places a symmetric Dirichlet prior on the donor simplex weights and fits with
NUTS (via PyMC). Output includes posterior samples and credible bands on the
synthetic counterfactual.

``pymc`` is a heavy optional dependency — lazy-imported.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pandas as pd

from ..utils._lazy import require
from ..utils.logging import get_logger
from .variants import _pivot_panel

log = get_logger(__name__)

__all__ = ["bayesian_scm"]

#: A C++ program that needs the standard library headers and the linker, as
#: pytensor's compiled ops do.
_PROBE = "#include <vector>\nint main() { std::vector<int> v(1); return v[0]; }\n"


@functools.cache
def _compiler_broken() -> bool:
    """Whether pytensor has a C++ compiler that cannot build its ops.

    A compiler that is found but fails (stale headers, a linker flag the toolchain
    no longer takes) stops every sample; with none at all pytensor already runs its
    ops in Python."""
    import pytensor
    from pytensor.link.c.cmodule import GCC_compiler

    if not pytensor.config.cxx:
        return False
    try:
        return not GCC_compiler.try_compile_tmp(  # type: ignore[no-untyped-call]
            _PROBE, tmp_prefix="normet_probe_"
        )
    except Exception:  # noqa: BLE001 - any failure to build means the compiler is unusable
        return True


def _sample(pm: Any, model: Any, **kwargs: Any) -> Any:
    """``pm.sample(model=model, **kwargs)``, without pytensor's C++ compiler if that
    compiler cannot build the model's ops: slower, but the sampler runs.

    The probe catches a compiler that builds nothing; one that builds a plain program
    but not every op pytensor writes (a newer clang rejecting its narrowing casts)
    shows only when the model's own ops are built, before any draw is taken."""
    from pytensor.link.c.exceptions import CompileError

    if not _compiler_broken():
        try:
            return pm.sample(model=model, **kwargs)
        except CompileError:
            pass
    import pytensor

    log.warning(
        "pytensor's C++ compiler (%s) cannot build its ops here; sampling in Python "
        "instead, which is slower. Repair the compiler (e.g. reinstall the Xcode "
        "Command Line Tools) for full speed.",
        pytensor.config.cxx,
    )
    with pytensor.config.change_flags(cxx=""):
        return pm.sample(model=model, **kwargs)


def bayesian_scm(
    df: pd.DataFrame,
    date_col: str = "date",
    unit_col: str = "code",
    outcome_col: str = "poll",
    treated_unit: str | None = None,
    cutoff_date: str | None = None,
    donors: list[str] | None = None,
    *,
    dirichlet_alpha: float = 1.0,
    sigma_prior: float = 1.0,
    include_intercept: bool = False,
    intercept_prior: float | None = None,
    ar_order: int = 0,
    predictive_band: bool = True,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.9,
    n_cores: int | None = None,
    seed: int = 7_654_321,
    ci_level: float = 0.95,
    progressbar: bool = False,
) -> dict[str, Any]:
    """
    Bayesian simplex SCM with Dirichlet prior, optional intercept, AR(1) likelihood,
    posterior predictive bands, and MCMC convergence diagnostics.

    Model
    -----
    ::

        w ~ Dirichlet(alpha * 1_J)
        sigma ~ HalfNormal(sigma_prior)
        intercept ~ Normal(0, intercept_prior)   [when include_intercept=True]
        mu = [intercept +] X_pre @ w
        y_pre ~ Normal(mu, sigma)  or  stationary AR(1)(mu, rho, sigma)

    By default (``predictive_band=True``), credible bands incorporate both
    posterior parameter uncertainty (``w``, ``intercept``) and observation noise
    (``sigma``, plus ``rho`` when ``ar_order=1``); parameter-mean-only bands
    are also retained in ``synthetic_mean_low`` / ``synthetic_mean_high``.

    Parameters
    ----------
    df, date_col, unit_col, outcome_col, treated_unit, cutoff_date, donors :
        Same as :func:`normet.causal.variants.scm_abadie`.
    dirichlet_alpha : float, default 1.0
        Symmetric Dirichlet concentration. ``<1`` favours sparse weights;
        ``>1`` favours diffuse weights.
    sigma_prior : float, default 1.0
        HalfNormal scale for the observation noise standard deviation.
    include_intercept : bool, default False
        Add an unconstrained additive offset parameter to handle level
        differences between the treated unit and donors.
    intercept_prior : float, optional
        Prior SD for ``intercept`` (defaults to ``max(2 * std(y_pre), 1.0)``).
    ar_order : {0, 1}, default 0
        Residual temporal dependence: ``0`` for i.i.d. Normal, ``1`` for
        stationary AR(1) residuals with uniform ``rho in (-0.99, 0.99)``.
    predictive_band : bool, default True
        Use full posterior predictive draws (parameter + innovation noise) for
        ``synthetic_low``/``synthetic_high`` and ``effect_low``/``effect_high``.
    draws, tune, chains, target_accept, n_cores, seed :
        Forwarded to :func:`pymc.sample`.
    ci_level : float, default 0.95
        Two-sided credible interval level for the synthetic and effect bands.
    progressbar : bool, default False
        Pass-through to PyMC.

    Returns
    -------
    dict
        ``{"synthetic": DataFrame, "weights": Series, "weights_summary": DataFrame,
        "posterior_samples": ndarray, "idata": InferenceData,
        "diagnostics": dict}``.
    """
    if treated_unit is None:
        raise ValueError("`treated_unit` is required.")
    if int(ar_order) not in (0, 1):
        raise ValueError(f"ar_order must be 0 or 1, got {ar_order}")

    panel, donors = _pivot_panel(df, date_col, unit_col, outcome_col, treated_unit, donors)

    if cutoff_date is None:
        from ..analysis.events import detect_events

        events = detect_events(panel[treated_unit], method="iqr", k=3.0)
        if events.empty:
            raise ValueError(
                "No `cutoff_date` provided, and no anomalies detected in the treated unit outcome series."
            )
        # Select the earliest event start date
        earliest = events.sort_values("start").iloc[0]["start"]
        cutoff_date = str(pd.to_datetime(earliest).date())
        log.info("Automatically detected cutoff date from anomaly events: %s", cutoff_date)

    pm = require("pymc", hint="pip install pymc")
    np_module = require("numpy")  # always available — sanity
    cutoff_ts = pd.to_datetime(cutoff_date)
    pre = panel[panel.index < cutoff_ts][donors + [treated_unit]].dropna(how="any")
    if pre.shape[0] < 5:
        raise ValueError("Not enough complete pre-treatment rows for Bayesian SCM.")

    X_pre = pre[donors].to_numpy(dtype=float)
    y_pre = pre[treated_unit].to_numpy(dtype=float)
    J = len(donors)
    int_prior_eff = (
        float(intercept_prior)
        if intercept_prior is not None
        else max(float(np.std(y_pre, ddof=1)) * 2.0, 1.0)
    )

    log.info(
        "Fitting Bayesian SCM | T_pre=%d | J=%d | ar=%d | intercept=%s | draws=%d × chains=%d",
        X_pre.shape[0],
        J,
        int(ar_order),
        include_intercept,
        draws,
        chains,
    )

    with pm.Model() as model:
        w = pm.Dirichlet("w", a=np_module.full(J, float(dirichlet_alpha)))
        sigma = pm.HalfNormal("sigma", sigma=float(sigma_prior))
        if include_intercept:
            intercept = pm.Normal("intercept", mu=0.0, sigma=int_prior_eff)
            mu = intercept + pm.math.dot(X_pre, w)
        else:
            mu = pm.math.dot(X_pre, w)

        if int(ar_order) == 0:
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_pre)
        else:
            rho = pm.Uniform("rho", lower=-0.99, upper=0.99)
            pm.Normal(
                "y_obs_0",
                mu=mu[0],
                sigma=sigma / pm.math.sqrt(1.0 - rho**2),
                observed=y_pre[0],
            )
            pm.Normal(
                "y_obs_rest",
                mu=mu[1:] + rho * (y_pre[:-1] - mu[:-1]),
                sigma=sigma,
                observed=y_pre[1:],
            )

    idata = _sample(
        pm,
        model,
        draws=int(draws),
        tune=int(tune),
        chains=int(chains),
        target_accept=float(target_accept),
        cores=n_cores,
        random_seed=int(seed),
        progressbar=progressbar,
    )

    posterior = idata.posterior["w"].stack(sample=("chain", "draw")).transpose("sample", ...).values
    sigma_s = idata.posterior["sigma"].stack(sample=("chain", "draw")).values.astype(float)
    w_mean = posterior.mean(axis=0)
    alpha = (1.0 - float(ci_level)) / 2.0

    X_full = panel[donors].to_numpy(dtype=float)
    mean_samples = posterior @ X_full.T
    if include_intercept:
        int_s = idata.posterior["intercept"].stack(sample=("chain", "draw")).values.astype(float)
        mean_samples = mean_samples + int_s[:, None]
    else:
        int_s = None

    syn_mean_lo = np.nanquantile(mean_samples, alpha, axis=0)
    syn_mean_hi = np.nanquantile(mean_samples, 1.0 - alpha, axis=0)

    obs = panel[treated_unit].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed) + 1)

    if int(ar_order) == 0:
        eps = rng.normal(0.0, sigma_s[:, None], size=mean_samples.shape)
        pred_samples = mean_samples + eps
        rho_s = None
    else:
        rho_s = idata.posterior["rho"].stack(sample=("chain", "draw")).values.astype(float)
        n_s, n_t = mean_samples.shape
        eps = np.empty((n_s, n_t), dtype=float)
        eps[:, 0] = rng.normal(0.0, sigma_s / np.sqrt(1.0 - rho_s**2))
        for t_idx in range(1, n_t):
            eps[:, t_idx] = rho_s * eps[:, t_idx - 1] + rng.normal(0.0, sigma_s)
        pred_samples = mean_samples + eps

    band_samples = pred_samples if predictive_band else mean_samples

    syn_mean = mean_samples.mean(axis=0)
    syn_lo = np.nanquantile(band_samples, alpha, axis=0)
    syn_hi = np.nanquantile(band_samples, 1.0 - alpha, axis=0)

    eff_mean = obs - syn_mean
    eff_samples = obs[np.newaxis, :] - band_samples
    eff_lo = np.nanquantile(eff_samples, alpha, axis=0)
    eff_hi = np.nanquantile(eff_samples, 1.0 - alpha, axis=0)

    # MCMC convergence diagnostics via arviz.
    max_r_hat = float("nan")
    min_ess_bulk = float("nan")
    min_ess_tail = float("nan")
    n_divergences = 0
    try:
        az = require("arviz", hint="pip install arviz")
        diag_vars = ["w", "sigma"]
        if include_intercept:
            diag_vars.append("intercept")
        if int(ar_order) == 1:
            diag_vars.append("rho")
        summ = az.summary(idata, var_names=diag_vars)
        if "r_hat" in summ.columns and summ["r_hat"].notna().any():
            max_r_hat = float(summ["r_hat"].max())
        if "ess_bulk" in summ.columns and summ["ess_bulk"].notna().any():
            min_ess_bulk = float(summ["ess_bulk"].min())
        if "ess_tail" in summ.columns and summ["ess_tail"].notna().any():
            min_ess_tail = float(summ["ess_tail"].min())
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
            n_divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    except Exception as e:
        log.debug("Could not compute ArviZ MCMC diagnostics: %s", e)

    converged = bool((np.isnan(max_r_hat) or max_r_hat <= 1.05) and n_divergences == 0)
    if (not np.isnan(max_r_hat) and max_r_hat > 1.05) or n_divergences > 0:
        log.warning(
            "Bayesian SCM convergence warning | max_r_hat=%.3f | min_ess_bulk=%.1f | divergences=%d",
            max_r_hat,
            min_ess_bulk,
            n_divergences,
        )

    mcmc_diag: dict[str, Any] = {
        "max_r_hat": max_r_hat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "n_divergences": n_divergences,
        "converged": converged,
        "ar_order": int(ar_order),
        "include_intercept": bool(include_intercept),
        "predictive_band": bool(predictive_band),
        "sigma_mean": float(np.mean(sigma_s)),
    }
    if int_s is not None:
        mcmc_diag["intercept_mean"] = float(np.mean(int_s))
    if rho_s is not None:
        mcmc_diag["rho_mean"] = float(np.mean(rho_s))

    out_df = pd.DataFrame(
        {
            "observed": obs,
            "synthetic": syn_mean,
            "synthetic_low": syn_lo,
            "synthetic_high": syn_hi,
            "synthetic_mean_low": syn_mean_lo,
            "synthetic_mean_high": syn_mean_hi,
            "effect": eff_mean,
            "effect_low": eff_lo,
            "effect_high": eff_hi,
        },
        index=panel.index,
    )
    out_df.attrs["mcmc_diagnostics"] = mcmc_diag

    w_lo = np.nanquantile(posterior, alpha, axis=0)
    w_hi = np.nanquantile(posterior, 1.0 - alpha, axis=0)
    weights_summary = pd.DataFrame(
        {"mean": w_mean, "ci_low": w_lo, "ci_high": w_hi},
        index=donors,
    )

    return {
        "synthetic": out_df,
        "weights": pd.Series(w_mean, index=donors, name="weight"),
        "weights_summary": weights_summary,
        "posterior_samples": posterior,
        "idata": idata,
        "diagnostics": mcmc_diag,
    }
