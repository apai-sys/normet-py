# src/normet/causal/panel.py
"""Build a clean, dense long-format panel for synthetic-control estimators."""

from __future__ import annotations

import pandas as pd

from ..exceptions import DataError
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["prepare_panel"]


def prepare_panel(
    df: pd.DataFrame,
    *,
    date_col: str,
    unit_col: str,
    outcome_col: str,
    cutoff_date: str,
    treated_unit: str | None = None,
    min_coverage: float = 0.65,
    date_from: str | None = None,
    date_to: str | None = None,
    freq: str = "D",
    max_donor_ratio: float = 0.3,
) -> pd.DataFrame:
    """
    Screen, densify, and reshape a ragged long panel for :func:`normet.run_scm`
    and friends (:func:`normet.causal.batch.scm_all`,
    :func:`normet.causal.placebo.placebo_in_time`/``placebo_in_space``).

    Real-world panels (sensor networks especially) are rarely complete: units
    come online at different times, drop out, or report at irregular
    intervals. The ridge fit in :func:`normet.causal.scm.scm` requires a fully
    dense pre-period (it drops any date where *any* unit is missing), so a
    handful of sparse donors can silently collapse the usable sample to zero
    rows. This function:

    1. Pivots ``df`` to one row per timestamp, one column per unit.
    2. Drops donor units that don't meet ``min_coverage`` in *both* the
       pre- and post-cutoff windows (the treated unit is always kept).
    3. Linearly interpolates remaining small gaps (and fills edges) so the
       panel has no missing values; units that are still incomplete
       afterwards are dropped with a warning.
    4. Warns if the surviving donor count is large relative to the
       pre-period length — donor count approaching or exceeding the number
       of pre-period observations destabilises the ridge fit (symptoms
       include negative "concentrations" in the synthetic series and
       placebo-in-time p-values that swing wildly with small donor changes).

    Parameters
    ----------
    df : pandas.DataFrame
        Ragged long panel with at least ``[date_col, unit_col, outcome_col]``.
    date_col, unit_col, outcome_col : str
        Column names in ``df``.
    cutoff_date : str
        Treatment cutoff (``"YYYY-MM-DD"``); only used to split the pre/post
        coverage screen, not to fit anything here.
    treated_unit : str, optional
        Unit ID to always keep regardless of its coverage. Raises
        :class:`~normet.exceptions.DataError` if it's still incomplete after
        interpolation (silently dropping the treated unit would be worse
        than failing loudly).
    min_coverage : float, default 0.65
        Minimum fraction of non-missing observations a donor must have in
        *both* the pre- and post-cutoff windows to be kept.
    date_from, date_to : str, optional
        Bounds for the dense date grid. Defaults to ``df[date_col].min()/max()``.
    freq : str, default "D"
        Resampling frequency for the dense grid (pandas offset alias).
    max_donor_ratio : float, default 0.3
        Warn when ``n_donors / n_pre_period_rows`` exceeds this.

    Returns
    -------
    pandas.DataFrame
        Long-format panel ``[date_col, unit_col, outcome_col]`` with no
        missing values, ready for :func:`normet.run_scm`. Coverage and
        donor-ratio diagnostics are attached to ``result.attrs``.
    """
    d = df[[date_col, unit_col, outcome_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    if d[date_col].isna().any():
        raise DataError(f"Some rows have invalid `{date_col}` values after coercion.")
    # Normalize to tz-naive throughout: mixing a tz-aware data column with
    # tz-naive `date_from`/`date_to`/`cutoff_date` strings silently produces
    # an all-NaN reindex (no error) rather than a comparison failure.
    if d[date_col].dt.tz is not None:
        d[date_col] = d[date_col].dt.tz_convert("UTC").dt.tz_localize(None)

    wide = d.pivot_table(index=date_col, columns=unit_col, values=outcome_col, aggfunc="mean")
    # Aggregate to the target frequency *before* reindexing onto the dense
    # grid below. Without this, higher-frequency input (e.g. hourly readings
    # with freq="D") only lines up with the grid at exact midnight
    # timestamps, and every other reading is misclassified as missing.
    wide = wide.resample(freq).mean()

    full_range = pd.date_range(
        pd.to_datetime(date_from) if date_from else wide.index.min(),
        pd.to_datetime(date_to) if date_to else wide.index.max(),
        freq=freq,
    )
    wide = wide.reindex(full_range)
    wide.index.name = date_col

    cutoff_ts = pd.to_datetime(cutoff_date)
    pre_mask = full_range < cutoff_ts
    post_mask = full_range >= cutoff_ts
    if not pre_mask.any() or not post_mask.any():
        raise DataError(
            f"`cutoff_date`={cutoff_date} leaves an empty pre- or post-period "
            f"within [{full_range.min().date()}, {full_range.max().date()}]."
        )

    pre_cov = wide[pre_mask].notna().mean()
    post_cov = wide[post_mask].notna().mean()
    coverage = pd.DataFrame({"pre_coverage": pre_cov, "post_coverage": post_cov})

    keep = (pre_cov >= min_coverage) & (post_cov >= min_coverage)
    if treated_unit is not None:
        if treated_unit not in wide.columns:
            raise DataError(f"`treated_unit`='{treated_unit}' not found in `{unit_col}`.")
        keep[treated_unit] = True

    n_dropped_coverage = int((~keep).sum())
    if n_dropped_coverage:
        log.info(
            "prepare_panel: dropping %d/%d units below min_coverage=%.0f%%.",
            n_dropped_coverage,
            len(keep),
            min_coverage * 100,
        )
    wide = wide.loc[:, keep[keep].index]

    wide = wide.interpolate(method="linear", limit_direction="both")
    still_missing = wide.columns[wide.isna().any()]
    if treated_unit is not None and treated_unit in still_missing:
        raise DataError(
            f"`treated_unit`='{treated_unit}' still has missing values after interpolation; "
            "it has no usable data across the requested date range."
        )
    if len(still_missing):
        log.warning(
            "prepare_panel: dropping %d unit(s) still incomplete after interpolation: %s",
            len(still_missing),
            list(still_missing),
        )
        wide = wide.drop(columns=still_missing)

    n_pre = int(pre_mask.sum())
    n_donors = wide.shape[1] - (
        1 if treated_unit is not None and treated_unit in wide.columns else 0
    )
    donor_ratio = n_donors / n_pre if n_pre else float("inf")
    if donor_ratio > max_donor_ratio:
        log.warning(
            "prepare_panel: %d donors vs %d pre-period rows (ratio=%.2f > max_donor_ratio=%.2f). "
            "A donor pool this large relative to the pre-period length can destabilise the ridge "
            "fit in scm() — watch for synthetic values outside the plausible range of the outcome, "
            "and validate with placebo_in_time before trusting the result. Consider raising "
            "min_coverage or capping the donor pool (e.g., one or a few per region/group).",
            n_donors,
            n_pre,
            donor_ratio,
            max_donor_ratio,
        )

    long_panel = wide.reset_index().melt(
        id_vars=date_col, var_name=unit_col, value_name=outcome_col
    )
    long_panel.attrs["coverage"] = coverage
    long_panel.attrs["n_donors"] = n_donors
    long_panel.attrs["n_pre_period_rows"] = n_pre
    long_panel.attrs["donor_ratio"] = donor_ratio
    log.info(
        "prepare_panel: %d units (%d donors%s), %d rows/unit, donor_ratio=%.2f.",
        wide.shape[1],
        n_donors,
        f" + treated '{treated_unit}'" if treated_unit else "",
        wide.shape[0],
        donor_ratio,
    )
    return long_panel
