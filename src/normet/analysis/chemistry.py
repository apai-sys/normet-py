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

from ..exceptions import ConfigError, DataError
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
]

# IUPAC standard molar masses (g/mol) for mass <-> molar-equivalent conversion.
_MOLAR_MASS: dict[str, float] = {
    "O3": 48.0000,
    "NO2": 46.0055,
    "NO": 30.0061,
}


@dataclass
class ConservedOxResult:
    """Output of :func:`normalise_ox_invariant`."""

    species: pd.DataFrame
    ox_result: pd.DataFrame
    nox_result: pd.DataFrame
    titration_reallocation: pd.Series
    unit: str
    method: str
    conservation_max_err: float


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
) -> tuple[pd.DataFrame, pd.Series]:
    """Deweather one scalar column and return ``(result_df, aligned_estimate_series)``."""
    if method == "dml":
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
        )
        res = normalise(
            df_prep,
            mdl,
            covariates=covariates,
            variables_resample=variables_resample,
            n_samples=n_samples,
            paired=True,
            sampling="balanced",
            seed=seed,
            n_cores=n_cores,
            verbose=verbose,
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
    method : {"dml", "crossfit", "standard"}, default "dml"
        Deweathering engine used for :math:`\\mathrm{O_x}` and :math:`\\mathrm{NO_x}`.
    backend : str, default "lightgbm"
        Model backend for regression and photochemical partitioning.
    use_paired : bool, default True
        Use the anomaly-preserving ``paired`` series when available.

    Returns
    -------
    ConservedOxResult
    """
    if unit not in ("ug_m3", "ppb"):
        raise ConfigError(f"`unit` must be 'ug_m3' or 'ppb', got {unit!r}.")
    if method not in ("dml", "crossfit", "standard"):
        raise ConfigError(f"`method` must be 'dml', 'crossfit', or 'standard', got {method!r}.")
    if not covariates:
        raise ConfigError("`covariates` must be non-empty.")

    work = process_date(df.copy()) if "date" not in df.columns else df.copy()
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
    )

    common_idx = ox_dew_s.index.intersection(nox_dew_s.index)
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
    cons_err = float(np.nanmax(np.abs((o3_mol_dew + no2_mol_dew) - ox_dew)))

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
                "Ox_observed": ox_mol * mw_o3,
                "NOx_observed": nox_mol * mw_no2,
            }
        )
        .groupby("date", as_index=True)
        .mean()
        .reindex(common_idx)
    )

    species_dict: dict[str, Any] = {
        "O3_observed": obs_frame["O3_observed"].to_numpy(dtype=float),
        "O3_normalised": o3_mol_dew * mw_o3,
        "NO2_observed": obs_frame["NO2_observed"].to_numpy(dtype=float),
        "NO2_normalised": no2_mol_dew * mw_no2,
        "Ox_observed": obs_frame["Ox_observed"].to_numpy(dtype=float),
        "Ox_normalised": ox_dew * mw_o3,
        "NOx_observed": obs_frame["NOx_observed"].to_numpy(dtype=float),
        "NOx_normalised": nox_dew * mw_no2,
        "O3_titration_effect": titration_series.to_numpy(dtype=float),
        "f_no2_normalised": f_dew,
    }
    if has_no:
        species_dict["NO_observed"] = obs_frame["NO_observed"].to_numpy(dtype=float)
        species_dict["NO_normalised"] = no_mol_dew * mw_no

    species_df = pd.DataFrame(species_dict, index=pd.Index(common_idx, name="date"))
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
