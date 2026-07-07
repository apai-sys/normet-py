# src/normet/causal/batch.py
"""Batch runner: :func:`scm_all` runs multiple SCM specs in parallel."""

from __future__ import annotations

import os

import pandas as pd
from joblib import Parallel, delayed

from ..utils.logging import get_logger
from .run_scm import run_scm

log = get_logger(__name__)

__all__ = ["scm_all"]


def scm_all(
    df: pd.DataFrame,
    *,
    date_col: str,
    outcome_col: str,
    unit_col: str,
    donors: list[str] | None,
    cutoff_date: str,
    scm_backend: str = "scm",
    n_cores: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run synthetic-control for many candidate treated units (in parallel).

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel with at least {date_col, unit_col, outcome_col}.
    date_col : str
        Column name for dates (convertible to datetime).
    outcome_col : str
        Name of the outcome variable.
    unit_col : str
        Column containing unit identifiers.
    donors : Optional[List[str]]
        Donor pool. If None, use all units except the treated unit.
    cutoff_date : str
        Treatment cutoff in "YYYY-MM-DD" format.
    scm_backend : {"scm","mlscm"}, optional
        Synthetic-control backend to use:
          - "scm"   : Augmented SCM.
          - "mlscm" : ML-augmented SCM (requires ML backend via kwargs).
    n_cores : int | None, optional
        Number of parallel workers. Defaults to all cores - 1.
    **kwargs
        Forwarded to `run_scm`. Examples:
          - If `scm_backend="mlscm"`:
              * backend="flaml"         (ML backend; default "flaml")
              * model_config={...}      (AutoML config)
              * seed=123                (random seed)
          - If `scm_backend="scm"`:
              * pre_covariates=[...]
              * allow_negative_weights=True

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with columns:
          - date_col
          - observed, synthetic, effect
          - unit_col (treated unit ID)
    """
    units = sorted(pd.unique(df[unit_col]))
    n_cores_eff = max(1, n_cores if n_cores is not None else (os.cpu_count() or 2) - 1)

    ml_backend_for_log = kwargs.get("backend") if scm_backend == "mlscm" else None
    log.info(
        "scm_all: scm_backend=%s | ml_backend=%s | cutoff=%s | units=%d | n_cores=%d",
        scm_backend,
        ml_backend_for_log,
        cutoff_date,
        len(units),
        n_cores_eff,
    )

    def _one(code: str) -> pd.DataFrame | None:
        try:
            donors_u = donors if donors is not None else [u for u in units if u != code]
            syn = run_scm(
                df=df,
                date_col=date_col,
                outcome_col=outcome_col,
                unit_col=unit_col,
                treated_unit=code,
                donors=donors_u,
                cutoff_date=cutoff_date,
                scm_backend=scm_backend,
                **kwargs,  # may include backend="flaml" if mlscm
            )
            out = syn.copy()
            out[unit_col] = code
            out = out.reset_index().rename(columns={syn.index.name or "index": date_col})
            return out
        except Exception as e:
            log.warning("%s failed for unit %s: %s", scm_backend, code, e)
            return None

    pieces = Parallel(n_jobs=n_cores_eff)(delayed(_one)(code) for code in units)
    pieces = [p for p in pieces if p is not None]
    if not pieces:
        raise RuntimeError("All synthetic-control runs failed.")

    return pd.concat(pieces, ignore_index=True)
