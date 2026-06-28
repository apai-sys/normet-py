# src/normet/causal/run_scm.py
"""Validate inputs and dispatch a single synthetic-control run to a backend."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, cast

import pandas as pd

from ..exceptions import ExperimentalWarning
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["run_scm"]


def run_scm(
    *,
    df: pd.DataFrame,
    date_col: str,
    unit_col: str,
    outcome_col: str,
    treated_unit: str,
    cutoff_date: str,
    donors: list[str] | None,
    scm_backend: str = "scm",
    **kwargs,
) -> pd.DataFrame:
    """Validate inputs and dispatch to the selected synthetic-control backend.

    Parameters
    ----------
    df : pandas.DataFrame
        Long panel with at least [date_col, unit_col, outcome_col].
    date_col : str
        Datetime column (convertible to datetime).
    unit_col : str
        Unit identifier column.
    outcome_col : str
        Outcome/response column.
    treated_unit : str
        Unit to treat as "treated".
    cutoff_date : str
        Treatment cutoff in "YYYY-MM-DD" (flexible parsing accepted).
    donors : List[str] | None
        Donor pool. If None, use all units except the treated unit.
    scm_backend : {"scm","mlscm","abadie","did","mcnnm","robust"}
        Which synthetic-control backend to use.
    **kwargs
        Forwarded to the selected backend function.

        If scm_backend == "mlscm":
          - backend: {"flaml"} (ML AutoML backend, default "flaml")
          - model_config: dict (AutoML settings)
          - seed: int, etc.

        If scm_backend == "scm":
          - pre_covariates: List[str]
          - allow_negative_weights: bool

    Returns
    -------
    pandas.DataFrame
        Indexed by date with columns ["observed","synthetic","effect"].
    """
    # ---- Basic validation
    missing_cols = [c for c in (date_col, unit_col, outcome_col) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in df: {missing_cols}")
    if not treated_unit:
        raise ValueError("`treated_unit` must be a non-empty string.")

    # Normalize backend selector
    scm_backend = (scm_backend or "scm").lower()

    # Units & donors
    all_units = list(pd.unique(df[unit_col]))
    if treated_unit not in all_units:
        raise ValueError(f"Treated unit '{treated_unit}' not present in `{unit_col}`.")

    base_pool = donors if donors is not None else all_units

    donor_pool = [u for u in dict.fromkeys(base_pool) if u != treated_unit]
    if not donor_pool:
        raise ValueError("No donors available after excluding the treated unit.")

    # Robust cutoff parsing → standard string
    try:
        cutoff_ts = pd.to_datetime(cutoff_date)
    except Exception as e:
        raise ValueError("`cutoff_date` must be parseable to a date.") from e
    cutoff_str = cutoff_ts.strftime("%Y-%m-%d")

    # ---- Route to requested backend
    if scm_backend == "scm":
        from ..causal.scm import scm

        log.info(
            "Running synthetic control | scm_backend=%s | treated=%s | donors=%d | cutoff=%s",
            scm_backend,
            treated_unit,
            len(donor_pool),
            cutoff_str,
        )
        out = scm(
            df=df,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            cutoff_date=cutoff_str,
            donors=donor_pool,
            **kwargs,
        )
        # SCM returns a dict with key "synthetic"
        return out["synthetic"]

    if scm_backend == "mlscm":
        from ..causal.mlscm import mlscm

        kw = dict(kwargs)  # avoid mutating original kwargs
        ml_backend = (kw.pop("backend", "flaml") or "flaml").lower()
        log.info(
            "Running synthetic control | scm_backend=%s (backend=%s) | treated=%s | donors=%d | cutoff=%s",
            scm_backend,
            ml_backend,
            treated_unit,
            len(donor_pool),
            cutoff_str,
        )
        # mlscm() warns ExperimentalWarning on every call; run_scm fans out to
        # mlscm() many times per placebo/bootstrap/batch run, so suppress the
        # repeat noise here. Direct nm.mlscm() calls still warn as documented.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ExperimentalWarning)
            return mlscm(
                df=df,
                date_col=date_col,
                unit_col=unit_col,
                outcome_col=outcome_col,
                treated_unit=treated_unit,
                cutoff_date=cutoff_str,
                donors=donor_pool,
                backend=ml_backend,
                **kw,  # model_config, seed, etc.
            )

    # Newer variants (Abadie classic / DiD baseline / MC-NNM / Robust SCM): same signature.
    if scm_backend in {"abadie", "did", "mcnnm", "robust"}:
        from ..causal.variants import did_baseline, scm_abadie, scm_mcnnm, scm_robust

        fn = cast(
            "Callable[..., Any]",
            {
                "abadie": scm_abadie,
                "did": did_baseline,
                "mcnnm": scm_mcnnm,
                "robust": scm_robust,
            }[scm_backend],
        )
        log.info(
            "Running synthetic control | scm_backend=%s | treated=%s | donors=%d | cutoff=%s",
            scm_backend,
            treated_unit,
            len(donor_pool),
            cutoff_str,
        )
        out = fn(
            df=df,
            date_col=date_col,
            unit_col=unit_col,
            outcome_col=outcome_col,
            treated_unit=treated_unit,
            cutoff_date=cutoff_str,
            donors=donor_pool,
            **kwargs,
        )
        return out["synthetic"]

    raise ValueError(f"Unsupported scm_backend: {scm_backend}")
