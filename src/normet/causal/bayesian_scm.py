# src/normet/causal/bayesian_scm.py
"""
Bayesian Synthetic Control Method.

Places a symmetric Dirichlet prior on the donor simplex weights and fits with
NUTS (via PyMC). Output includes posterior samples and credible bands on the
synthetic counterfactual.

``pymc`` is a heavy optional dependency — lazy-imported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils._lazy import require
from ..utils.logging import get_logger
from .variants import _pivot_panel

log = get_logger(__name__)

__all__ = ["bayesian_scm"]


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
    Bayesian simplex SCM with Dirichlet prior + Normal likelihood.

    Model
    -----
    ::

        w ~ Dirichlet(alpha · 1_J)
        sigma ~ HalfNormal(sigma_prior)
        y_pre = X_pre @ w + epsilon,  epsilon ~ Normal(0, sigma)

    The fitted posterior over ``w`` is propagated through the full panel
    (pre + post) to produce a credible band on the synthetic series.

    Parameters
    ----------
    df, date_col, unit_col, outcome_col, treated_unit, cutoff_date, donors :
        Same as :func:`normet.causal.scm.scm_abadie`.
    dirichlet_alpha : float, default 1.0
        Symmetric Dirichlet concentration. ``<1`` favours sparse weights;
        ``>1`` favours diffuse weights.
    sigma_prior : float, default 1.0
        HalfNormal scale for the noise standard deviation.
    draws, tune, chains, target_accept, n_cores, seed :
        Forwarded to :func:`pymc.sample`.
    ci_level : float, default 0.95
        Two-sided credible interval level for the synthetic and effect bands.
    progressbar : bool, default False
        Pass-through to PyMC.

    Returns
    -------
    dict
        A dictionary with the structure::

            {
                "synthetic": DataFrame[observed, synthetic_mean, synthetic_low,
                                       synthetic_high, effect_mean, effect_low,
                                       effect_high],
                "weights": Series(mean weights),
                "weights_summary": DataFrame[mean, ci_low, ci_high],
                "posterior_samples": ndarray (n_samples, J),
                "idata": arviz.InferenceData,
            }
    """
    if treated_unit is None:
        raise ValueError("`treated_unit` is required.")

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

    log.info(
        "Fitting Bayesian SCM | T_pre=%d | J=%d | draws=%d × chains=%d",
        X_pre.shape[0],
        J,
        draws,
        chains,
    )

    with pm.Model() as model:  # noqa: F841 — handle is used implicitly by the sampler
        w = pm.Dirichlet("w", a=np_module.full(J, float(dirichlet_alpha)))
        sigma = pm.HalfNormal("sigma", sigma=float(sigma_prior))
        mu = pm.math.dot(X_pre, w)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_pre)
        idata = pm.sample(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            target_accept=float(target_accept),
            cores=n_cores,
            random_seed=int(seed),
            progressbar=progressbar,
        )

    posterior = idata.posterior["w"].stack(sample=("chain", "draw")).transpose("sample", ...).values
    w_mean = posterior.mean(axis=0)
    alpha = (1.0 - float(ci_level)) / 2.0

    X_full = panel[donors].to_numpy(dtype=float)
    # (n_samples, T)
    synth_samples = posterior @ X_full.T

    syn_mean = synth_samples.mean(axis=0)
    syn_lo = np.nanquantile(synth_samples, alpha, axis=0)
    syn_hi = np.nanquantile(synth_samples, 1.0 - alpha, axis=0)

    obs = panel[treated_unit].to_numpy(dtype=float)
    eff_samples = obs[np.newaxis, :] - synth_samples
    eff_mean = eff_samples.mean(axis=0)
    eff_lo = np.nanquantile(eff_samples, alpha, axis=0)
    eff_hi = np.nanquantile(eff_samples, 1.0 - alpha, axis=0)

    out_df = pd.DataFrame(
        {
            "observed": obs,
            "synthetic": syn_mean,
            "synthetic_low": syn_lo,
            "synthetic_high": syn_hi,
            "effect": eff_mean,
            "effect_low": eff_lo,
            "effect_high": eff_hi,
        },
        index=panel.index,
    )

    # Equal-tailed credible interval on the weights, computed from the posterior
    # samples with numpy for stability across arviz versions (consistent with
    # the synthetic/effect bands above).
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
    }
