# src/normet/analysis/chemistry.py
"""Photochemical conservation and chemical-regime-conditioned deweathering.

Single-species meteorological normalisation breaks down on coupled secondary
pollutants (F3 nonlinear titration / regime switching): when :math:`\\mathrm{NO_x}`
emissions drop, reduced NO titration raises :math:`\\mathrm{O_3}` even at fixed
meteorology, and a single-species model attributes part of that photochemical
repartitioning to weather. This module enforces stoichiometric conservation and
chemical-regime conditioning:

- :func:`normalise_ox_invariant`: transforms :math:`(\\mathrm{O_3}, \\mathrm{NO_2},
  \\mathrm{NO})` into conserved odd-oxygen (:math:`\\mathrm{O_x = O_3 + NO_2}`) and
  :math:`\\mathrm{NO_x = NO + NO_2}` coordinates, deweathers the conserved totals,
  and repartitions via a meteorology-marginalised photochemical fraction conditioned
  on the deweathered :math:`\\mathrm{NO_x}` regime.
- :func:`normalise_regime_conditioned`: deweathers a secondary pollutant
  (e.g. :math:`\\mathrm{O_3}` or :math:`\\mathrm{PM_{2.5}}`) conditioned on
  deweathered primary emission tracers (:math:`\\mathrm{NO_x}`, :math:`\\mathrm{CO}`,
  etc.), separating meteorological effects from precursor-driven chemical shifts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.exceptions import ConfigError, DataError
from ..model.train import build_model, train_model
from ..utils._config import DEFAULT_SEED
from ..utils.prepare import prepare_data, process_date
from .normalise import (
    _CROSSFIT_DEFAULT_EMBARGO,
    _is_time_feature,
    crossfit_normalise,
    dml_normalise,
    normalise,
)

__all__ = [
    "ConservedOxResult",
    "RegimeConditionedResult",
    "normalise_ox_invariant",
    "normalise_regime_conditioned",
    "ox_conservation_error",
]

# IUPAC standard molar masses (g/mol) for mass <-> molar-equivalent conversion.
_MOLAR_MASS: dict[str, float] = {
    "O3": 48.0000,
    "NO2": 46.0055,
    "NO": 30.0061,
}


@dataclass
class ConservedOxResult:
    """Output of :func:`normalise_ox_invariant`.

    ``species`` reports every species in the input's concentration unit. The
    totals are *constituent sums*, not a single species' equivalent: the odd
    oxygen and NOx columns are

    ``Ox_observed = O3_observed + NO2_observed`` and
    ``NOx_observed = NO2_observed + NO_observed`` (and the same for the
    ``*_normalised`` block), and ``O3_titration_effect`` stays O3-equivalent
    mass. So the returned mass columns add up exactly, and the molar laws
    ``Ox = O3 + NO2`` / ``NOx = NO + NO2`` are recovered from them with each
    species' own IUPAC molar mass (:func:`ox_conservation_error`).

    ``conservation_max_err`` is the largest absolute residual of those molar
    identities over the record -- ``mol m^-3`` when ``unit='ug_m3'`` (a ug
    m^-3 column divided by its species' molar mass in ug per mol) and
    ppb-equivalent moles when ``unit='ppb'``; it is absolute, not relative.
    A consistent fit leaves it at round-off (~1e-22 mol m^-3 at tens of
    ug m^-3); it rises as soon as a reported column stops summing to its
    total. See :func:`ox_conservation_error` for the exact definition.
    """

    species: pd.DataFrame
    ox_result: pd.DataFrame
    nox_result: pd.DataFrame
    titration_reallocation: pd.Series
    unit: str
    method: str
    conservation_max_err: float


def _molar_scale(unit: str) -> dict[str, float]:
    """Per-species divisor (in the frame's unit per mole) turning a column into ``mol``.

    With ``unit='ug_m3'`` the columns are mass concentrations in ug m^-3, so a
    species divides by its **own** IUPAC molar mass in ug per mol -- g/mol
    times 1e6 (O3 48.0000, NO2 46.0055, NO 30.0061 g/mol) -- and the quotient
    is mol m^-3. With ``unit='ppb'`` the columns are already molar (mole per
    mole) and every divisor is 1.0.
    """
    if unit == "ug_m3":
        return {name: mass * 1e6 for name, mass in _MOLAR_MASS.items()}
    if unit == "ppb":
        return {name: 1.0 for name in _MOLAR_MASS}
    raise ConfigError(f"`unit` must be 'ug_m3' or 'ppb', got {unit!r}.")


def ox_conservation_error(
    species: pd.DataFrame,
    *,
    unit: str = "ug_m3",
    ox_mol: np.ndarray | pd.Series | None = None,
    nox_mol: np.ndarray | pd.Series | None = None,
) -> float:
    """Largest absolute violation of the molar O_x / NO_x conservation laws.

    Both laws are written in **moles**, not in mass: odd oxygen is
    ``Ox = O3 + NO2`` and ``NOx = NO + NO2`` in ppb-equivalent moles, and each
    species enters with its *own* molar mass (1 umol of O3 is 48.00 ug, 1 umol
    of NO2 is 46.0055 ug). A reported mass column is turned back into moles by
    dividing it by that species' molar mass in ug per mol, so a fit whose
    columns add up in their own units also satisfies the molar law.

    Parameters
    ----------
    species : pandas.DataFrame
        The ``*_normalised`` columns of a :class:`ConservedOxResult` frame
        (the ones :func:`normalise_ox_invariant` reports). ``O3_normalised``
        and ``NO2_normalised`` are required; ``Ox_normalised``,
        ``NOx_normalised`` and ``NO_normalised`` are used when present.
    unit : {"ug_m3", "ppb"}, default "ug_m3"
        Unit of the columns; ``'ug_m3'`` divides by the species' molar mass
        in ug per mol, ``'ppb'`` leaves the (already molar) values alone.
    ox_mol, nox_mol : array-like, optional
        The deweathered *molar* total odd oxygen / NOx series. When given,
        the residual is also measured against them, which is the only part of
        the check that cannot be reproduced from the frame alone.

    Returns
    -------
    float
        Maximum over samples of

        - ``|O3_normalised + NO2_normalised - Ox_normalised| / M(O3)`` -- the
          arithmetic a caller repeats on the returned frame, expressed in
          O3-equivalent moles (``M`` in ug per mol, so the quotient is
          mol m^-3);
        - ``|NO2_normalised + NO_normalised - NOx_normalised| / M(NO2)`` --
          the same for NOx (only when the frame carries ``NO_normalised``,
          i.e. when the input had a NO column);
        - ``|o3_mol + no2_mol - ox_mol|`` and ``|no_mol + no2_mol - nox_mol|``
          when the molar totals are supplied.

        An absolute residual, in ``mol m^-3`` for ``unit='ug_m3'`` and in
        ppb-equivalent moles for ``unit='ppb'`` (not relative: divide by the
        ``Ox_*`` column for a relative error). It is round-off -- 0 or ~1e-22
        at concentrations of a few tens of ug m^-3 -- for a frame that adds up,
        and grows linearly with any perturbation of a constituent column.
        ``nan`` if ``species`` holds no usable rows.
    """
    mw = _molar_scale(unit)
    missing = [c for c in ("O3_normalised", "NO2_normalised") if c not in species.columns]
    if missing:
        raise DataError(
            f"`species` is missing the column(s) {missing}; expected the frame "
            "`normalise_ox_invariant` returns."
        )

    def _column(name: str) -> np.ndarray:
        return species[name].to_numpy(dtype=float)

    o3_mass = _column("O3_normalised")
    no2_mass = _column("NO2_normalised")
    o3_moles = o3_mass / mw["O3"]
    no2_moles = no2_mass / mw["NO2"]
    no_moles = _column("NO_normalised") / mw["NO"] if "NO_normalised" in species.columns else None

    residuals: list[np.ndarray] = []
    # The totals the frame reports must be the sums of their constituents in
    # the frame's own unit; divided by a molar mass they become O3- /
    # NO2-equivalent moles, the unit the comparison is made in.
    if "Ox_normalised" in species.columns:
        residuals.append((_column("Ox_normalised") - (o3_mass + no2_mass)) / mw["O3"])
    if "NOx_normalised" in species.columns and no_moles is not None:
        nox_parts_mass = no2_mass + _column("NO_normalised")
        residuals.append((_column("NOx_normalised") - nox_parts_mass) / mw["NO2"])
    # The model-side closure: the molar series the deweathering models
    # produced, which the repartition has to reproduce from the reported masses.
    if ox_mol is not None:
        residuals.append(o3_moles + no2_moles - np.asarray(ox_mol, dtype=float))
    if nox_mol is not None:
        parts = no2_moles if no_moles is None else no2_moles + no_moles
        residuals.append(parts - np.asarray(nox_mol, dtype=float))

    worst = np.nan
    for residual in residuals:
        finite = np.abs(residual[np.isfinite(residual)])
        if finite.size:
            worst = float(finite.max()) if np.isnan(worst) else max(worst, float(finite.max()))
    return float(worst)


@dataclass
class RegimeConditionedResult:
    """Output of :func:`normalise_regime_conditioned`."""

    normalised: pd.DataFrame
    precursors: pd.DataFrame
    precursor_regime_effect: pd.Series
    method: str
    target: str
    precursor_names: list[str]


def _deweather_series(
    df: pd.DataFrame,
    target_col: str,
    *,
    covariates: list[str],
    variables_resample: list[str] | None,
    method: str,
    backend: str,
    model_config: dict[str, Any] | None,
    cutoff_days: float,
    n_splits: int,
    embargo: int,
    n_samples: int,
    log_transform: bool,
    use_paired: bool,
    seed: int,
    n_cores: int | None,
    verbose: bool,
    icn_estimator: Any = None,
    icn_design: Any = None,
    icn_pipeline: Any = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Deweather one scalar column and return ``(result_df, aligned_estimate_series)``."""
    if method == "icn":
        from ..foundation import (
            Chronos2Estimator,
            ICNDesign,
            normalise_in_context,
            to_indexed_frame,
        )

        keep_cols = list(
            dict.fromkeys([c for c in ("date", target_col, *covariates) if c in df.columns])
        )
        indexed = to_indexed_frame(df[keep_cols])
        est = (
            icn_estimator
            if icn_estimator is not None
            else Chronos2Estimator(met_covariates=covariates)
        )
        if icn_design is not None:
            design = icn_design
        else:
            min_ctx = (
                min(336, max(48, len(indexed) // 4))
                if len(indexed) <= int(getattr(est, "context_length", 2048)) + 336
                else None
            )
            design = ICNDesign.v1(
                log_transform=bool(log_transform),
                seed=seed,
                min_context=min_ctx,
            )
        met_feats = list(variables_resample) if variables_resample is not None else None
        icn_res = normalise_in_context(
            est,
            indexed,
            target_col,
            design=design,
            pipeline=icn_pipeline,
            met_features=met_feats,
            schema="normet",
        )
        res = icn_res.to_normet_frame("level", include_factual=True)
        res["paired"] = icn_res.paired.reindex(res.index).to_numpy(dtype=float)
    elif method == "dml":
        res = dml_normalise(
            df,
            target=target_col,
            covariates=covariates,
            variables_resample=variables_resample,
            backend=backend,
            model_config=model_config,
            cutoff_days=cutoff_days,
            n_splits=n_splits,
            embargo=embargo,
            paired="log_ratio" if log_transform else "additive",
            n_samples=n_samples,
            log_transform=log_transform,
            sampling="balanced",
            seed=seed,
            n_cores=n_cores,
            verbose=verbose,
        ).normalised
    elif method == "crossfit":
        res = crossfit_normalise(
            df,
            target=target_col,
            covariates=covariates,
            variables_resample=variables_resample,
            backend=backend,
            model_config=model_config,
            n_splits=n_splits,
            embargo=embargo,
            paired="log_ratio" if log_transform else "additive",
            n_samples=n_samples,
            log_transform=log_transform,
            sampling="balanced",
            seed=seed,
            n_cores=n_cores,
            verbose=verbose,
        ).normalised
    else:
        df_prep, mdl = build_model(
            df,
            target=target_col,
            backend=backend,
            covariates=covariates,
            model_config=model_config,
            seed=seed,
            n_cores=n_cores,
            verbose=verbose,
            # Mirror the DML/cross-fit branches: the fitted and evaluated
            # frames must be on the same (log1p) scale, and `normalise`
            # expects `build_model` to have applied the transform already.
            log_transform=bool(log_transform),
        )
        res = normalise(
            df_prep,
            mdl,
            covariates=covariates,
            variables_resample=variables_resample,
            n_samples=n_samples,
            paired="log_ratio" if log_transform else True,
            sampling="balanced",
            seed=seed,
            n_cores=n_cores,
            verbose=verbose,
            log_transform=bool(log_transform),
        )
    col = "paired" if (use_paired and "paired" in res.columns) else "normalised"
    return res, res[col].astype(float)


def normalise_ox_invariant(
    df: pd.DataFrame,
    o3_col: str = "O3",
    no2_col: str = "NO2",
    *,
    no_col: str | None = "NO",
    covariates: Sequence[str],
    variables_resample: Sequence[str] | None = None,
    unit: str = "ug_m3",
    method: str = "dml",
    backend: str = "lightgbm",
    model_config: dict[str, Any] | None = None,
    cutoff_days: float = 30.0,
    n_splits: int = 5,
    embargo: int = _CROSSFIT_DEFAULT_EMBARGO,
    n_samples: int = 300,
    log_transform: bool = False,
    use_paired: bool = True,
    seed: int = DEFAULT_SEED,
    n_cores: int | None = None,
    verbose: bool = False,
    icn_estimator: Any = None,
    icn_design: Any = None,
    icn_pipeline: Any = None,
) -> ConservedOxResult:
    """Conserved :math:`\\mathrm{O_x\\text{–}NO_x}` photochemical coordinate deweathering.

    Solves F3 nonlinear titration leakage by deweathering total odd oxygen
    (:math:`\\mathrm{O_x = O_3 + NO_2}`) and :math:`\\mathrm{NO_x = NO + NO_2}`
    in molar-conserved coordinates, then repartitioning :math:`\\mathrm{O_x}`
    via a meteorology-marginalised photochemical fraction
    :math:`f_{\\mathrm{NO_2}} = \\mathrm{NO_2} / \\mathrm{O_x}` evaluated at the
    deweathered :math:`\\mathrm{NO_x}` emission regime.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame with ``date`` (or DatetimeIndex), ``o3_col``, ``no2_col``,
        optional ``no_col``, and meteorological/time covariates.
    o3_col, no2_col : str
        Column names for :math:`\\mathrm{O_3}` and :math:`\\mathrm{NO_2}`.
    no_col : str or None, default "NO"
        Optional column name for :math:`\\mathrm{NO}`. When absent or ``None``,
        :math:`\\mathrm{NO_x}` defaults to :math:`\\mathrm{NO_2}`.
    covariates : sequence of str
        Meteorological and calendar covariates.
    variables_resample : sequence of str, optional
        Meteorological variables to marginalise over; defaults to non-time
        covariates.
    unit : {"ug_m3", "ppb"}, default "ug_m3"
        Concentration units of ``o3_col``, ``no2_col``, and ``no_col``.
    method : {"dml", "crossfit", "standard", "icn"}, default "dml"
        Deweathering engine used for :math:`\\mathrm{O_x}` and :math:`\\mathrm{NO_x}`.
    backend : str, default "lightgbm"
        Model backend for regression and photochemical partitioning.
    use_paired : bool, default True
        Use the anomaly-preserving ``paired`` series when available.
    icn_estimator, icn_design, icn_pipeline : optional
        Optional Chronos-2 estimator, :class:`~normet.foundation.in_context.ICNDesign`,
        and :class:`~normet.foundation.in_context.ICLPipeline` forwarded when
        ``method="icn"``.

    Returns
    -------
    ConservedOxResult

    Notes
    -----
    **What is conserved.** Odd oxygen and NOx are conserved in *moles*,
    :math:`\\mathrm{O_x} = \\mathrm{O_3} + \\mathrm{NO_2}` and
    :math:`\\mathrm{NO_x} = \\mathrm{NO} + \\mathrm{NO_2}` in ppb-equivalent
    moles, with every species carrying its own IUPAC molar mass (O3 48.0000,
    NO2 46.0055, NO 30.0061 g/mol) so that 1 umol of O3 (48.00 ug) and 1 umol
    of NO2 (46.0055 ug) each count as one mole of Ox. With ``unit='ppb'`` the
    columns are already molar and the molar masses are 1.0.

    **How the columns report it.** ``species`` keeps every column in the input
    unit. The two legs are reported in each species' own mass, and the totals
    are their **constituent sums** --
    ``Ox_* = O3_* + NO2_*`` and ``NOx_* = NO2_* + NO_*`` -- so the mass
    columns add up exactly. (Reporting Ox as O3-*equivalent* mass, as earlier
    versions did, made ``Ox_normalised - O3_normalised - NO2_normalised``
    reach ~4 % of the Ox level, about 1.7 ug m^-3 at 40 ug m^-3 of O3.)
    ``O3_titration_effect`` remains an O3-equivalent mass, as named.

    **How the error is normalised.**
    :attr:`ConservedOxResult.conservation_max_err` is an **absolute** molar
    residual -- ``mol m^-3`` for ``unit='ug_m3'`` (mass columns divided by
    g/mol) and ppb-equivalent moles for ``unit='ppb'`` -- of the identities
    above, measured on the returned frame (:func:`ox_conservation_error`).
    It is round-off (0 or ~1e-22 at tens of ug m^-3) for a consistent fit and
    grows as soon as a reported column stops summing to its total; divide by
    the ``Ox_*`` column if a relative error is wanted. When the input has no
    NO column there is no NO leg, so only the Ox identity is checked there.
    """
    if unit not in ("ug_m3", "ppb"):
        raise ConfigError(f"`unit` must be 'ug_m3' or 'ppb', got {unit!r}.")
    if method not in ("dml", "crossfit", "standard", "icn"):
        raise ConfigError(
            f"`method` must be 'dml', 'crossfit', 'standard', or 'icn', got {method!r}."
        )
    if not covariates:
        raise ConfigError("`covariates` must be non-empty.")

    # A `date` column that is a string is not a datetime axis: grouping and
    # label-based reindexing then compare raw labels, and the `observed`
    # columns come back all-NaN. Coerce whenever the column exists but is not
    # already datetime64; a frame without the column is handled by
    # `process_date` from its index.
    work = df.copy()
    if "date" not in work.columns or not pd.api.types.is_datetime64_any_dtype(work["date"]):
        work = process_date(work)
    if "date" not in work.columns:
        raise DataError("`df` must contain a 'date' column or DatetimeIndex.")
    for req in (o3_col, no2_col):
        if req not in work.columns:
            raise DataError(f"Required species column {req!r} not found in `df`.")
    has_no = bool(no_col is not None and no_col in work.columns)

    cov_list = [str(c) for c in covariates]
    res_list = (
        [str(c) for c in variables_resample]
        if variables_resample is not None
        else [c for c in cov_list if not _is_time_feature(c)]
    )

    mw_o3 = _MOLAR_MASS["O3"] if unit == "ug_m3" else 1.0
    mw_no2 = _MOLAR_MASS["NO2"] if unit == "ug_m3" else 1.0
    mw_no = _MOLAR_MASS["NO"] if unit == "ug_m3" else 1.0

    o3_obs = np.maximum(pd.to_numeric(work[o3_col], errors="coerce").to_numpy(dtype=float), 0.0)
    no2_obs = np.maximum(pd.to_numeric(work[no2_col], errors="coerce").to_numpy(dtype=float), 0.0)
    o3_mol = o3_obs / mw_o3
    no2_mol = no2_obs / mw_no2
    ox_mol = o3_mol + no2_mol
    if has_no:
        assert no_col is not None
        no_obs = np.maximum(pd.to_numeric(work[no_col], errors="coerce").to_numpy(dtype=float), 0.0)
        no_mol = no_obs / mw_no
        nox_mol = no2_mol + no_mol
    else:
        no_obs = np.zeros_like(o3_obs)
        no_mol = np.zeros_like(o3_mol)
        nox_mol = no2_mol.copy()

    work_chem = work.copy()
    work_chem["_Ox_mol"] = ox_mol
    work_chem["_NOx_mol"] = nox_mol

    ox_df, ox_dew_s = _deweather_series(
        work_chem,
        "_Ox_mol",
        covariates=cov_list,
        variables_resample=res_list,
        method=method,
        backend=backend,
        model_config=model_config,
        cutoff_days=cutoff_days,
        n_splits=n_splits,
        embargo=embargo,
        n_samples=n_samples,
        log_transform=log_transform,
        use_paired=use_paired,
        seed=seed,
        n_cores=n_cores,
        verbose=verbose,
        icn_estimator=icn_estimator,
        icn_design=icn_design,
        icn_pipeline=icn_pipeline,
    )
    nox_df, nox_dew_s = _deweather_series(
        work_chem,
        "_NOx_mol",
        covariates=cov_list,
        variables_resample=res_list,
        method=method,
        backend=backend,
        model_config=model_config,
        cutoff_days=cutoff_days,
        n_splits=n_splits,
        embargo=embargo,
        n_samples=n_samples,
        log_transform=log_transform,
        use_paired=use_paired,
        seed=seed + 101,
        n_cores=n_cores,
        verbose=verbose,
        icn_estimator=icn_estimator,
        icn_design=icn_design,
        icn_pipeline=icn_pipeline,
    )

    common_idx = ox_dew_s.index.intersection(nox_dew_s.index)
    if method == "icn":
        fin_mask = ox_dew_s.reindex(common_idx).notna() & nox_dew_s.reindex(common_idx).notna()
        common_idx = common_idx[fin_mask.to_numpy(dtype=bool)]
        if len(common_idx) == 0:
            raise DataError(
                "ICN produced no finite Ox/NOx estimates; supply a longer series or set min_context."
            )
    ox_dew = np.maximum(ox_dew_s.reindex(common_idx).to_numpy(dtype=float), 0.0)
    nox_dew = np.maximum(nox_dew_s.reindex(common_idx).to_numpy(dtype=float), 0.0)

    # Fit photochemical odd-oxygen partitioning model: f_NO2 = NO2_mol / Ox_mol.
    f_no2_obs = np.clip(no2_mol / np.maximum(ox_mol, 1e-6), 0.0, 1.0)
    work_part = work_chem.copy()
    work_part["_f_no2"] = f_no2_obs
    part_covs_base = [c for c in cov_list if c not in ("date_unix", "year")]
    part_covs = list(dict.fromkeys([*part_covs_base, "_Ox_mol", "_NOx_mol"]))
    df_part = prepare_data(
        work_part,
        target="_f_no2",
        covariates=part_covs,
        dropna=True,
        split_method="ts",
        train_fraction=1.0,
        seed=seed,
        impute="train_median",
    )
    part_model = train_model(
        df=df_part,
        target="value",
        backend=backend,
        covariates=part_covs,
        model_config=model_config,
        seed=seed + 202,
        n_cores=n_cores,
        verbose=False,
    )

    df_part_indexed = df_part.set_index("date").reindex(common_idx)
    df_part_dew = df_part_indexed.copy()
    df_part_dew["_Ox_mol"] = ox_dew
    df_part_dew["_NOx_mol"] = nox_dew
    part_res_vars = [c for c in res_list if c in part_covs and c not in ("_Ox_mol", "_NOx_mol")]

    f_dew_df = normalise(
        df_part_dew,
        part_model,
        covariates=part_covs,
        variables_resample=part_res_vars,
        n_samples=n_samples,
        sampling="balanced",
        seed=seed + 303,
        n_cores=n_cores,
        verbose=False,
    )
    f_dew = np.clip(f_dew_df["normalised"].reindex(common_idx).to_numpy(dtype=float), 0.0, 1.0)

    df_part_clim = df_part_dew.copy()
    df_part_clim["_NOx_mol"] = float(np.nanmean(nox_dew))
    f_clim_df = normalise(
        df_part_clim,
        part_model,
        covariates=part_covs,
        variables_resample=part_res_vars,
        n_samples=n_samples,
        sampling="balanced",
        seed=seed + 303,
        n_cores=n_cores,
        verbose=False,
    )
    f_clim = np.clip(f_clim_df["normalised"].reindex(common_idx).to_numpy(dtype=float), 0.0, 1.0)

    no2_mol_dew = np.clip(f_dew * ox_dew, 0.0, np.minimum(ox_dew, nox_dew))
    o3_mol_dew = np.maximum(0.0, ox_dew - no2_mol_dew)
    no_mol_dew = np.maximum(0.0, nox_dew - no2_mol_dew)
    # Each leg is converted to mass with its own molar mass. The totals are
    # the *constituent sums* of those masses, so the reported columns add up
    # in the frame's own unit and the molar law is recoverable from them
    # (see `ox_conservation_error`); the previous `Ox = ox_mol * M(O3)`
    # reported O3-equivalent mass and was ~4 % away from O3 + NO2.
    o3_mass_dew = o3_mol_dew * mw_o3
    no2_mass_dew = no2_mol_dew * mw_no2
    no_mass_dew = no_mol_dew * mw_no

    titration_mol = (f_clim - f_dew) * ox_dew
    titration_series = pd.Series(
        titration_mol * mw_o3, index=common_idx, name="O3_titration_effect"
    )

    obs_frame = (
        pd.DataFrame(
            {
                "date": work["date"].to_numpy(),
                "O3_observed": o3_obs,
                "NO2_observed": no2_obs,
                "NO_observed": no_obs,
            }
        )
        .groupby("date", as_index=True)
        .mean()
        .reindex(common_idx)
    )
    o3_obs_g = obs_frame["O3_observed"].to_numpy(dtype=float)
    no2_obs_g = obs_frame["NO2_observed"].to_numpy(dtype=float)
    no_obs_g = obs_frame["NO_observed"].to_numpy(dtype=float)

    species_dict: dict[str, Any] = {
        "O3_observed": o3_obs_g,
        "O3_normalised": o3_mass_dew,
        "NO2_observed": no2_obs_g,
        "NO2_normalised": no2_mass_dew,
        "Ox_observed": o3_obs_g + no2_obs_g,
        "Ox_normalised": o3_mass_dew + no2_mass_dew,
        # Without a NO leg there is nothing to add to NO2, so NOx is the
        # reported NO2 (the deweathered NOx series itself stays in
        # `nox_result`).
        "NOx_observed": no2_obs_g + no_obs_g if has_no else no2_obs_g,
        "NOx_normalised": no2_mass_dew + no_mass_dew if has_no else no2_mass_dew,
        "O3_titration_effect": titration_series.to_numpy(dtype=float),
        "f_no2_normalised": f_dew,
    }
    if has_no:
        species_dict["NO_observed"] = no_obs_g
        species_dict["NO_normalised"] = no_mass_dew

    species_df = pd.DataFrame(species_dict, index=pd.Index(common_idx, name="date"))
    # Measured on the frame that is returned, so it is the same number a
    # caller recomputing O3 + NO2 against Ox sees. The molar totals are the
    # model-side reference and are passed only when the frame can express
    # them: with no NO column, NOx = NO2 is a *repartition* choice, not a
    # conservation law the frame can violate.
    #
    # `ox_dew`/`nox_dew` are the deweathered `_Ox_mol`/`_NOx_mol` series,
    # i.e. a ug m^-3 column over g/mol -- umol m^-3, not mol m^-3 -- so they
    # are scaled to the mol m^-3 the diagnostic is reported in. (With
    # `unit='ppb'` the series are already molar and the factor is 1.)
    micro_to_mol = 1e-6 if unit == "ug_m3" else 1.0
    cons_err = ox_conservation_error(
        species_df,
        unit=unit,
        ox_mol=ox_dew * micro_to_mol,
        nox_mol=nox_dew * micro_to_mol if has_no else None,
    )
    return ConservedOxResult(
        species=species_df,
        ox_result=ox_df,
        nox_result=nox_df,
        titration_reallocation=titration_series,
        unit=unit,
        method=method,
        conservation_max_err=cons_err,
    )


def normalise_regime_conditioned(
    df: pd.DataFrame,
    target: str,
    precursors: Sequence[str],
    *,
    covariates: Sequence[str],
    variables_resample: Sequence[str] | None = None,
    method: str = "dml",
    backend: str = "lightgbm",
    model_config: dict[str, Any] | None = None,
    cutoff_days: float = 30.0,
    n_splits: int = 5,
    embargo: int = _CROSSFIT_DEFAULT_EMBARGO,
    n_samples: int = 300,
    log_transform: bool = False,
    use_paired: bool = True,
    seed: int = DEFAULT_SEED,
    n_cores: int | None = None,
    verbose: bool = False,
) -> RegimeConditionedResult:
    """Deweather a secondary pollutant conditioned on deweathered primary precursors.

    First deweathers each primary tracer in ``precursors`` (e.g. ``["NOx", "CO"]``)
    independently of the secondary target, then fits the secondary pollutant on
    meteorology + precursors and marginalises over meteorology while holding the
    precursors fixed at their deweathered emission-driven trajectories.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame carrying ``date`` (or DatetimeIndex), ``target``,
        ``precursors``, and ``covariates``.
    target : str
        Secondary pollutant column (e.g. ``"O3"`` or ``"PM25"``).
    precursors : sequence of str
        One or more primary tracer column names defining the chemical regime.
    covariates : sequence of str
        Meteorological and calendar covariates.
    variables_resample : sequence of str, optional
        Meteorological variables to marginalise over; defaults to non-time
        covariates excluding ``precursors``.
    method : {"dml", "crossfit", "standard"}, default "dml"
        Deweathering engine used for precursors and secondary response.
    backend : str, default "lightgbm"
        Registered regression backend.
    use_paired : bool, default True
        Use the ``paired`` series when available.

    Returns
    -------
    RegimeConditionedResult
    """
    if method not in ("dml", "crossfit", "standard"):
        raise ConfigError(f"`method` must be 'dml', 'crossfit', or 'standard', got {method!r}.")
    prec_list = list(dict.fromkeys(str(p) for p in precursors))
    if not prec_list:
        raise ConfigError("`precursors` must contain at least one column name.")
    if not covariates:
        raise ConfigError("`covariates` must be non-empty.")

    work = process_date(df.copy()) if "date" not in df.columns else df.copy()
    if "date" not in work.columns or target not in work.columns:
        raise DataError(f"`df` must contain 'date' and target column {target!r}.")
    missing_p = [p for p in prec_list if p not in work.columns]
    if missing_p:
        raise DataError(f"Precursor column(s) absent from `df`: {missing_p}")

    cov_list = [str(c) for c in covariates]
    prec_covs = [c for c in cov_list if c not in prec_list]
    if not prec_covs:
        raise ConfigError("`covariates` must contain predictors outside `precursors`.")

    met_res = (
        [str(c) for c in variables_resample if c not in prec_list]
        if variables_resample is not None
        else [c for c in prec_covs if not _is_time_feature(c)]
    )

    prec_series: dict[str, pd.Series] = {}
    for idx_p, p_col in enumerate(prec_list):
        _, p_dew = _deweather_series(
            work,
            p_col,
            covariates=prec_covs,
            variables_resample=met_res,
            method=method,
            backend=backend,
            model_config=model_config,
            cutoff_days=cutoff_days,
            n_splits=n_splits,
            embargo=embargo,
            n_samples=n_samples,
            log_transform=log_transform,
            use_paired=use_paired,
            seed=seed + 17 * (idx_p + 1),
            n_cores=n_cores,
            verbose=verbose,
        )
        prec_series[p_col] = p_dew

    prec_df = pd.DataFrame(prec_series)
    prec_df.index.name = "date"

    sec_covs = list(dict.fromkeys([*cov_list, *prec_list]))
    # Replace precursor inputs with their deweathered trajectories before
    # marginalising meteorology so meteorology changes do not feed through precursors.
    work_sec = work.copy()
    work_dates = pd.DatetimeIndex(work_sec["date"])
    for p_col in prec_list:
        mapped = prec_df[p_col].reindex(work_dates).to_numpy(dtype=float)
        orig_p = pd.to_numeric(work_sec[p_col], errors="coerce").to_numpy(dtype=float)
        work_sec[p_col] = np.where(np.isfinite(mapped), mapped, orig_p)

    cond_df, cond_s = _deweather_series(
        work_sec,
        target,
        covariates=sec_covs,
        variables_resample=met_res,
        method=method,
        backend=backend,
        model_config=model_config,
        cutoff_days=cutoff_days,
        n_splits=n_splits,
        embargo=embargo,
        n_samples=n_samples,
        log_transform=log_transform,
        use_paired=use_paired,
        seed=seed + 500,
        n_cores=n_cores,
        verbose=verbose,
    )
    _, uncond_s = _deweather_series(
        work,
        target,
        covariates=prec_covs,
        variables_resample=met_res,
        method=method,
        backend=backend,
        model_config=model_config,
        cutoff_days=cutoff_days,
        n_splits=n_splits,
        embargo=embargo,
        n_samples=n_samples,
        log_transform=log_transform,
        use_paired=use_paired,
        seed=seed + 600,
        n_cores=n_cores,
        verbose=verbose,
    )

    common = cond_df.index.intersection(uncond_s.index)
    regime_eff = (cond_s.reindex(common) - uncond_s.reindex(common)).rename(
        "precursor_regime_effect"
    )
    out_df = cond_df.loc[common].copy()
    out_df["normalised_unconditioned"] = uncond_s.reindex(common).to_numpy(dtype=float)
    out_df["precursor_regime_effect"] = regime_eff.to_numpy(dtype=float)

    return RegimeConditionedResult(
        normalised=out_df,
        precursors=prec_df.reindex(common),
        precursor_regime_effect=regime_eff,
        method=method,
        target=target,
        precursor_names=prec_list,
    )
