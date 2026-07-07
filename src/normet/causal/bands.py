# src/normet/causal/bands.py
"""Effect bands and uncertainty quantification for synthetic-control results."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils._config import DEFAULT_SEED
from ..utils.logging import get_logger
from .run_scm import run_scm

log = get_logger(__name__)


def _eff_cores(n_cores: int | None) -> int:
    """Resolve worker count: explicit value, else ``cpu_count() - 1`` (min 1)."""
    return max(1, int(n_cores) if n_cores is not None else (os.cpu_count() or 2) - 1)


__all__ = [
    "effect_bands_space",
    "effect_bands_time",
    "uncertainty_bands",
    "plot_effect_with_bands",
    "plot_uncertainty_bands",
]


def effect_bands_space(
    placebo_space_out: dict, level: float = 0.95, method: str = "quantile"
) -> pd.DataFrame:
    """Build per-time uncertainty bands for the treated effect.

    Uses placebo-in-space output (works for SCM or ML-SCM). Bands are formed
    around the treated effect by adding placebo distribution summaries at
    each timestamp: effect(t) + Q_{α}(placebo_t) and effect(t) + Q_{1-α}(placebo_t),
    or effect(t) ± z * std(placebo_t).

    Parameters
    ----------
    placebo_space_out : dict
        Output from a backend-agnostic placebo-in-space routine (e.g. `placebo_in_space`):
          - "treated":  DataFrame (index=date) with at least column "effect".
          - "placebos": dict[str -> DataFrame] where each value has a single placebo
                        effect column (often named by the donor or 'effect') indexed by date.
    level : float, default=0.95
        Confidence level for bands (e.g., 0.95 → 2.5% / 97.5% tails).
    method : {"quantile","std"}, default="quantile"
        - "quantile": use placebo_t quantiles.
        - "std":      use mean(placebo_t) ± z * std(placebo_t), z = normal critical value.

    Returns
    -------
    pandas.DataFrame
        Indexed by date with columns:
          - "effect"  : treated effect
          - "lower"   : lower band
          - "upper"   : upper band
        Plus helpful extras:
          - if method="quantile": "plc_q_low","plc_q_high","plc_p10","plc_p90" (placebo quantiles)
          - if method="std":      "plc_mean","plc_std","z"

    Notes
    -----
    If no placebo series are available, returns the treated effect with NaN lower/upper.
    """
    # --- treated effect ---
    df_true = placebo_space_out.get("treated")
    if df_true is None or "effect" not in df_true.columns:
        raise ValueError(
            "`placebo_space_out['treated']` must be a DataFrame with an 'effect' column."
        )
    effect = df_true["effect"]

    # --- placebo matrix aligned to treated index ---
    plc_dict = placebo_space_out.get("placebos", {}) or {}
    if not plc_dict:
        out = pd.DataFrame(index=df_true.index)
        out["effect"] = effect
        out["lower"] = np.nan
        out["upper"] = np.nan
        return out

    cols = []
    for key, df_d in plc_dict.items():
        if key in df_d.columns:
            ser = df_d[key]
        elif "effect" in df_d.columns:
            ser = df_d["effect"].rename(key)
        else:  # fallback to the first (and should be only) column
            ser = df_d.iloc[:, 0].rename(key)
        cols.append(ser)

    plc_mat = pd.concat(cols, axis=1).reindex(df_true.index)
    out = pd.DataFrame(index=df_true.index)
    out["effect"] = effect

    if method == "quantile":
        alpha = (1.0 - level) / 2.0
        q_low = plc_mat.quantile(alpha, axis=1)
        q_high = plc_mat.quantile(1.0 - alpha, axis=1)
        # bands around treated effect
        out["lower"] = effect + q_low
        out["upper"] = effect + q_high
        # helpful diagnostics (raw placebo quantiles, not shifted)
        out["plc_q_low"] = q_low
        out["plc_q_high"] = q_high
        out["plc_p10"] = plc_mat.quantile(0.10, axis=1)
        out["plc_p90"] = plc_mat.quantile(0.90, axis=1)

    elif method == "std":
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + level / 2.0))
        mu = plc_mat.mean(axis=1)
        sd = plc_mat.std(axis=1, ddof=1)
        out["lower"] = effect + (mu - z * sd)
        out["upper"] = effect + (mu + z * sd)
        out["plc_mean"] = mu
        out["plc_std"] = sd
        out["z"] = z
    else:
        raise ValueError("`method` must be 'quantile' or 'std'.")

    return out


def effect_bands_time(
    placebo_time_out: dict,
    *,
    level: float = 0.95,
    method: str = "quantile",  # {"quantile","std"}
    horizon: int | None = None,  # if None, use the largest length common to all placebo segments
    return_segments: bool = False,
) -> dict:
    """
    Build event-time uncertainty bands from placebo-in-time results.

    This consumes the output of :func:`placebo_in_time`, i.e. a dict like::

        {
          "treated":  DataFrame,
          "placebos": dict[cutoff_timestamp -> DataFrame(effect segment)],
          "p_value": ...,
          "ref_band_event_time": ...,
          "placebo_stats": ...,
        }

    It produces bands over event time k = 0,1,2,... by aligning each placebo's effect
    series to its own cutoff date.

    Parameters
    ----------
    placebo_time_out : dict
        Output of `placebo_in_time` (as defined above).
    level : float, default=0.95
        Confidence level for the bands (e.g., 0.95 -> two-sided 2.5%/97.5%).
    method : {"quantile","std"}, default="quantile"
        - "quantile": lower/upper are empirical quantiles across placebos.
        - "std":      lower/upper are mean ± z * std (normal critical value).
    horizon : int | None, default=None
        Event-time window length K. If None, use the largest length that
        is common to all placebo segments (from their cutoff onward).
    return_segments : bool, default=False
        If True, also return the aligned placebo segments (event-time × placebo).

    Returns
    -------
    dict
        A dictionary with the structure::

            {
              "bands":   pandas.DataFrame,  # index=event_time (0..K-1)
                                            # - method="quantile": cols ["lower","upper","p10","p90"]
                                            # - method="std":      cols ["lower","upper","mean","std"]
              "segments": pandas.DataFrame | None,  # (K, P): placebo effects by event_time
              "cutoffs":  List[str],                # segment column labels (cutoff repr)
            }
    """
    plc_dict: dict[Any, pd.DataFrame] = placebo_time_out.get("placebos") or {}
    if not isinstance(plc_dict, dict) or len(plc_dict) == 0:
        raise ValueError(
            "`placebo_time_out['placebos']` must be a non-empty dict of cutoff -> DataFrame."
        )

    # Collect aligned placebo segments (as numpy arrays first)
    segments: list[np.ndarray] = []
    labels: list[str] = []
    lengths: list[int] = []

    # Each value is already a *post-cutoff* segment in your earlier implementation
    for cutoff, df_eff in plc_dict.items():
        if not isinstance(df_eff, pd.DataFrame) or df_eff.empty:
            continue

        # Identify the effect column
        if "effect" in df_eff.columns:
            seg = df_eff["effect"].to_numpy()
        else:
            seg = df_eff.iloc[:, 0].to_numpy()

        if seg.size < 2:
            continue

        segments.append(seg)
        labels.append(str(cutoff))
        lengths.append(seg.size)

    if not segments:
        raise ValueError(
            "No valid placebo segments could be constructed from `placebo_in_time` output."
        )

    # Determine event-time horizon K
    if horizon is None:
        K = int(min(lengths))  # common length across all placebos
    else:
        K = int(horizon)
        if K <= 0:
            raise ValueError("`horizon` must be a positive integer.")

    # Truncate segments to K and stack: shape (P, K)
    P = len(segments)
    M = np.vstack([seg[:K] for seg in segments])  # P x K
    if M.shape != (P, K):
        raise RuntimeError("Failed to assemble placebo matrix with the requested horizon.")

    # Compute bands over placebos at each event_time k
    idx_k = pd.RangeIndex(start=0, stop=K, step=1, name="event_time")
    out = pd.DataFrame(index=idx_k)

    if method == "quantile":
        alpha = (1.0 - float(level)) / 2.0
        out["lower"] = np.nanquantile(M, alpha, axis=0)
        out["upper"] = np.nanquantile(M, 1.0 - alpha, axis=0)
        out["p10"] = np.nanpercentile(M, 10, axis=0)
        out["p90"] = np.nanpercentile(M, 90, axis=0)
    elif method == "std":
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + float(level) / 2.0))
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0, ddof=1)
        out["mean"] = mu
        out["std"] = sd
        out["lower"] = mu - z * sd
        out["upper"] = mu + z * sd
    else:
        raise ValueError("`method` must be 'quantile' or 'std'.")

    seg_df = None
    if return_segments:
        seg_df = pd.DataFrame(M.T, index=idx_k, columns=labels)

    return {"bands": out, "segments": seg_df, "cutoffs": labels}


def uncertainty_bands(
    df: pd.DataFrame,
    date_col: str,
    unit_col: str,
    outcome_col: str,
    treated_unit: str,
    cutoff_date: str,
    donors: list[str] | None = None,
    *,
    scm_backend: str = "scm",
    method: str = "jackknife",
    B: int = 200,
    random_state: int = DEFAULT_SEED,
    donor_frac: float = 0.8,
    time_block_days: int | None = None,
    ci_level: float = 0.95,
    n_cores: int | None = None,
    **kwargs,
) -> dict:
    """Construct uncertainty bands for synthetic-control treatment effects.

    Uses either nonparametric bootstrap or leave-one-donor-out jackknife,
    wrapping :func:`run_scm` to repeatedly refit synthetic control models
    under resampling or jackknife schemes, then computes pointwise
    confidence bands for the estimated treatment effect series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input panel dataset in long format. Must include a time column, a unit
        identifier column, and the outcome column.
    date_col : str
        Name of the datetime column. Values must be parseable to pandas datetime.
    unit_col : str
        Name of the column identifying cross-sectional units (treated and donors).
    outcome_col : str
        Name of the outcome variable to be modeled.
    treated_unit : str
        Identifier of the treated unit.
    cutoff_date : str
        Intervention cutoff date (parseable by pandas). Pre- vs post-treatment
        periods are split here.
    donors : list of str, optional
        List of donor units to include. If None, all units except the treated unit
        are used.
    scm_backend : str, default="scm"
        Backend used in `run_scm` (e.g. "scm", "abadie", "mlscm", "robust").
    method : {"bootstrap", "jackknife"}, default="jackknife"
        Uncertainty estimation method:
          - "bootstrap": donor subsampling (and optional block time resampling).
          - "jackknife": leave-one-donor-out jackknife variance estimation.
    B : int, default=200
        Number of bootstrap replications (ignored if `method="jackknife"`).
    random_state : int, default=7654321
        Random seed for reproducibility in bootstrap sampling.
    donor_frac : float, default=0.8
        Fraction of donor pool to sample in each bootstrap replicate.
    time_block_days : int, optional
        If provided, pre-treatment time is resampled in blocks of this length
        (days) during bootstrap. Post-treatment period is always kept intact.
        Ignored if `method="jackknife"`.
    ci_level : float, default=0.95
        Confidence interval level (e.g., 0.95 for 95% bands).
    n_cores : int, optional
        Worker processes for the bootstrap/jackknife refits (via joblib).
        Defaults to ``cpu_count() - 1``.
    **kwargs :
        Additional arguments passed to `run_scm`.

    Returns
    -------
    dict
        Dictionary with keys:
          - "treated" : pandas.DataFrame
              Reference synthetic-control fit for the treated unit.
              Must contain column "effect".
          - "low" : pandas.Series
              Lower confidence band (aligned with treated index).
          - "high" : pandas.Series
              Upper confidence band (aligned with treated index).
          - "jackknife_effects" : pandas.DataFrame, optional
              Only returned if `method="jackknife"`. Each column is one
              leave-one-donor-out effect path.

    Notes
    -----
    - Bootstrap is more flexible and can incorporate both donor and time resampling,
      but is computationally intensive when B is large.
    - Jackknife is faster (requires only one fit per donor) and provides a direct
      donor-sensitivity analysis, but may underestimate uncertainty if donors are
      highly correlated.
    - The function silently skips failed fits (e.g. singular donor configurations).

    Examples
    --------
    >>> out = uncertainty_bands(
    ...     df, date_col="date", unit_col="ID",
    ...     outcome_col="SO2", treated_unit="CityA",
    ...     cutoff_date="2015-10-23", scm_backend="scm",
    ...     method="jackknife", ci_level=0.95
    ... )
    >>> out["low"].head()
    >>> out["high"].head()
    """
    from joblib import Parallel, delayed
    from scipy.stats import norm

    rng = np.random.default_rng(random_state)
    n_jobs = _eff_cores(n_cores)
    df = df.copy()

    # Ensure datetime-like
    if not np.issubdtype(pd.Series(df[date_col]).dtype, np.datetime64):  # type: ignore[arg-type]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise ValueError("Non-parseable dates found; clean 'date_col' first.")

    cutoff_ts = pd.to_datetime(cutoff_date)

    # Full donor pool
    all_units = sorted(df[unit_col].unique())
    base_donors = [u for u in (donors if donors is not None else all_units) if u != treated_unit]
    if len(base_donors) < 3:
        raise ValueError("Need at least 3 donors in the pool.")

    # --- Reference effect path ---
    df_true = run_scm(
        df=df,
        date_col=date_col,
        unit_col=unit_col,
        outcome_col=outcome_col,
        treated_unit=treated_unit,
        cutoff_date=cutoff_date,
        donors=base_donors,
        scm_backend=scm_backend,
        **kwargs,
    )
    effect_index = df_true.index

    # --------------------------------------------------
    # Method 1: Bootstrap
    # --------------------------------------------------
    if method.lower() == "bootstrap":
        dates = df[date_col]
        pre_mask = dates < cutoff_ts
        pre_days = np.asarray(pd.to_datetime(df.loc[pre_mask, date_col]).dt.normalize().unique())
        pre_days = np.sort(pre_days)

        # Per-replicate child seeds so the parallel run is reproducible.
        seeds = rng.integers(0, 2**32 - 1, size=int(B))

        def _one_bootstrap(seed: int) -> np.ndarray | None:
            r = np.random.default_rng(seed)
            k = max(3, int(round(len(base_donors) * float(donor_frac))))
            k = min(k, len(base_donors))
            replace_flag = k > len(base_donors)
            sub_donors = r.choice(base_donors, size=k, replace=replace_flag)

            df_b = df
            if time_block_days and time_block_days > 0 and len(pre_days) >= time_block_days:
                df_b = df.copy()
                n_blocks = max(1, len(pre_days) // time_block_days)
                if len(pre_days) == time_block_days:
                    starts = np.array([0])
                else:
                    starts = r.integers(0, len(pre_days) - time_block_days + 1, size=n_blocks)

                boot_days: list[pd.Timestamp] = []
                for s in starts:
                    boot_days.extend(pre_days[s : s + time_block_days])
                boot_days_ser = pd.to_datetime(pd.Series(boot_days)).dt.normalize()

                is_pre = df_b[date_col] < cutoff_ts
                keep_pre = df_b[date_col].dt.normalize().isin(boot_days_ser)
                df_b = pd.concat(
                    [df_b.loc[is_pre & keep_pre], df_b.loc[~is_pre]], ignore_index=True
                )

            try:
                out_b = run_scm(
                    df=df_b,
                    date_col=date_col,
                    unit_col=unit_col,
                    outcome_col=outcome_col,
                    treated_unit=treated_unit,
                    cutoff_date=cutoff_date,
                    donors=list(sub_donors),
                    scm_backend=scm_backend,
                    **kwargs,
                )
                return out_b["effect"].reindex(effect_index).to_numpy()
            except Exception as e:
                log.debug("bootstrap replicate failed: %s", e)
                return None

        results = Parallel(n_jobs=n_jobs)(delayed(_one_bootstrap)(int(s)) for s in seeds)
        eff_paths = [r for r in results if r is not None]
        n_fail = int(B) - len(eff_paths)
        if n_fail:
            emit = log.warning if n_fail > int(B) // 2 else log.debug
            emit("%d/%d bootstrap replicates failed and were skipped.", n_fail, int(B))

        if not eff_paths:
            return {"treated": df_true, "low": None, "high": None}

        M = np.vstack(eff_paths)
        alpha = (1.0 - float(ci_level)) / 2.0
        q_low = np.nanquantile(M, alpha, axis=0)
        q_high = np.nanquantile(M, 1.0 - alpha, axis=0)

        low = pd.Series(q_low, index=effect_index, name="low")
        high = pd.Series(q_high, index=effect_index, name="high")

        return {"treated": df_true, "low": low, "high": high}

    # --------------------------------------------------
    # Method 2: Jackknife
    # --------------------------------------------------
    elif method.lower() == "jackknife":
        n = len(base_donors)

        def _one_jackknife(d: str):
            donors_jk = [u for u in base_donors if u != d]
            try:
                out_jk = run_scm(
                    df=df,
                    date_col=date_col,
                    unit_col=unit_col,
                    outcome_col=outcome_col,
                    treated_unit=treated_unit,
                    cutoff_date=cutoff_date,
                    donors=donors_jk,
                    scm_backend=scm_backend,
                    **kwargs,
                )
                return out_jk["effect"].reindex(effect_index)
            except Exception as e:
                log.debug("jackknife fit (dropping donor %s) failed: %s", d, e)
                return None

        results = Parallel(n_jobs=n_jobs)(delayed(_one_jackknife)(d) for d in base_donors)
        jackknife_paths = [r for r in results if r is not None]
        n_fail = n - len(jackknife_paths)
        if n_fail:
            emit = log.warning if n_fail > n // 2 else log.debug
            emit("%d/%d jackknife fits failed and were skipped.", n_fail, n)

        if not jackknife_paths:
            return {"treated": df_true, "low": None, "high": None}

        jack_df = pd.concat(jackknife_paths, axis=1)
        theta_dot = jack_df.mean(axis=1)

        # jackknife SE
        se = np.sqrt(((n - 1) / n) * ((jack_df.sub(theta_dot, axis=0) ** 2).sum(axis=1)))

        z = norm.ppf(0.5 + ci_level / 2.0)
        low = df_true["effect"] - z * se
        high = df_true["effect"] + z * se

        return {
            "treated": df_true,
            "low": pd.Series(low, index=effect_index, name="low"),
            "high": pd.Series(high, index=effect_index, name="high"),
            "jackknife_effects": jack_df,
        }

    else:
        raise ValueError("`method` must be either 'bootstrap' or 'jackknife'.")


def plot_effect_with_bands(
    bands_df: pd.DataFrame,
    cutoff_date: object | None = None,
    title: str = "Effect with Placebo Bands",
    ax=None,
):
    """
    Plot treated effect with placebo-based uncertainty bands.

    Supports outputs from both effect_bands (placebo-in-space; date index)
    and event-time bands (placebo-in-time; integer index).

    Parameters
    ----------
    bands_df : pandas.DataFrame
        Must contain column 'effect' and band columns named either
        ('lower','upper') or ('low','high'). Index can be datetime-like
        (dates) or numeric (event time).
    cutoff_date : str | pandas.Timestamp | int | None, optional
        Vertical reference line. If the index is datetime-like, pass a date;
        if the index is event time, pass an integer (e.g., 0).
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes | None, optional
        Axes to draw on. Creates new if None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    if "effect" not in bands_df.columns:
        raise ValueError("bands_df must have an 'effect' column.")

    # Accept either naming convention for bands
    lower_col = (
        "lower" if "lower" in bands_df.columns else ("low" if "low" in bands_df.columns else None)
    )
    upper_col = (
        "upper" if "upper" in bands_df.columns else ("high" if "high" in bands_df.columns else None)
    )

    idx = bands_df.index

    # Shade band if available
    if lower_col and upper_col:
        ax.fill_between(
            idx, bands_df[lower_col], bands_df[upper_col], alpha=0.25, label="placebo band"
        )

    # Plot effect
    ax.plot(idx, bands_df["effect"], lw=1.8, label="treated effect")

    # Optional vertical reference line
    if cutoff_date is not None:
        try:
            # If index is datetime-like, convert cutoff to Timestamp
            if pd.api.types.is_datetime64_any_dtype(idx):
                cd = pd.to_datetime(cutoff_date)  # type: ignore[call-overload]
            else:
                # Event-time axis (numeric); try to cast to int/float
                cd = int(cutoff_date)  # type: ignore[call-overload]
            ax.axvline(cd, ls="--", lw=1, color="k", alpha=0.6, label="cutoff")
        except Exception:
            # Swallow conversion issues silently; the plot is still useful
            pass

    ax.set_title(title)
    ax.set_xlabel("Date" if pd.api.types.is_datetime64_any_dtype(idx) else "Event time")
    ax.set_ylabel("Effect (Observed - Synthetic)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return ax


def plot_uncertainty_bands(
    out: dict,
    cutoff_date: object | None = None,
    title: str = "SCM Effect with Uncertainty Bands",
    ax=None,
):
    """
    Plot synthetic-control treatment effects with uncertainty bands.

    This function visualises the estimated treatment effect path from
    synthetic control along with its associated uncertainty bands.
    It works for both bootstrap-based and jackknife-based results.

    Parameters
    ----------
    out : dict
        Dictionary returned by :func:`uncertainty_bands`. Must contain:

        - ``"treated"`` : pandas.DataFrame with index=date and column "effect".
        - ``"low"`` : pandas.Series (lower bound), same index as treated.
        - ``"high"`` : pandas.Series (upper bound), same index as treated.

        Optionally:

        - ``"jackknife_effects"`` : pandas.DataFrame. If present, indicates that
          uncertainty bands were constructed via the jackknife method.
    cutoff_date : str or pandas.Timestamp, optional
        If provided, a vertical dashed line is drawn at this date to mark
        the intervention cutoff.
    title : str, default="SCM Effect with Uncertainty Bands"
        Title for the plot.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to draw on. If None, a new figure and axis are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axis with the plotted series and bands.

    Notes
    -----
    - If `out` contains "jackknife_effects", the band is labelled as
      "jackknife band".
    - If the results were generated via bootstrap, the band is labelled
      "bootstrap band".
    - If band information is missing (`low` or `high` is None),
      only the treated effect path is drawn.

    Examples
    --------
    >>> out_boot = uncertainty_bands(df, ..., method="bootstrap")
    >>> plot_uncertainty_bands(out_boot, cutoff_date="2015-10-23",
    ...                        title="SCM Effect (Bootstrap)")
    >>> plt.show()

    >>> out_jk = uncertainty_bands(df, ..., method="jackknife")
    >>> plot_uncertainty_bands(out_jk, cutoff_date="2015-10-23",
    ...                        title="SCM Effect (Jackknife)")
    >>> plt.show()
    """
    if out is None or "treated" not in out:
        raise ValueError("`out` must include key 'treated'.")

    df_true = out["treated"]
    if not isinstance(df_true, pd.DataFrame) or "effect" not in df_true.columns:
        raise ValueError("`out['treated']` must be a DataFrame with an 'effect' column.")

    low = out.get("low")
    high = out.get("high")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    # Plot bands if available
    if low is not None and high is not None:
        low = low.reindex(df_true.index)
        high = high.reindex(df_true.index)

        label_band = "uncertainty band"
        if "jackknife_effects" in out:
            label_band = "jackknife band"
        elif "bootstrap" in title.lower():
            label_band = "bootstrap band"

        ax.fill_between(df_true.index, low, high, alpha=0.25, label=label_band)

    # Plot treated effect
    ax.plot(df_true.index, df_true["effect"], lw=1.8, label="treated effect")

    # Mark cutoff
    if cutoff_date is not None:
        cd = pd.to_datetime(cutoff_date)  # type: ignore[call-overload]
        ax.axvline(cd, ls="--", lw=1, color="k", alpha=0.6, label="cutoff")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Effect (Observed - Synthetic)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    return ax
