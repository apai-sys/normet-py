# src/normet/pipeline/do_all.py
"""Single-call pipelines that prepare data, train a model, and normalise.

Provides :func:`do_all` (one model) and :func:`do_all_unc` (an ensemble of
models for uncertainty quantification), plus their config dataclasses
:class:`SingleConfig` and :class:`UncConfig`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ..analysis.normalise import (
    _CROSSFIT_DEFAULT_EMBARGO,
    CrossFitNormaliseResult,
    _back_transform_values,
    crossfit_normalise,
    dml_normalise,
    normalise,
)
from ..core.exceptions import ConfigError
from ..foundation.estimator import (
    CHRONOS_BACKEND,
    resolve_n_samples,
)
from ..model.train import build_model, train_model
from ..utils._config import resolve_config
from ..utils.logging import _progress_str, get_logger
from ..utils.metrics import modStats
from ..utils.prepare import (
    compute_duan_smearing,
    expm1_columns,
    impute_values,
    log1p_target,
    prepare_data,
)

log = get_logger(__name__)

__all__ = ["CHRONOS_BACKEND", "SingleConfig", "UncConfig", "do_all", "do_all_unc"]


@dataclass
class SingleConfig:
    """Configuration for :func:`do_all`."""

    target: str = "value"
    backend: str = "flaml"
    covariates: list[str] | None = None
    variables_resample: list[str] | None = None
    split_method: str = "random"
    train_fraction: float = 0.75
    model_config: dict[str, Any] | None = None
    #: ``None`` resolves per backend: 300 for the AutoML backends, 8 for
    #: Chronos-2, where each sample is a full transformer forward pass.
    n_samples: int | None = None
    aggregate: bool = True
    seed: int = 7_654_321
    n_cores: int | None = None
    resample_df: pd.DataFrame | None = None
    memory_save: bool = False
    verbose: bool = False
    log_transform: bool = False
    """Fit on ``log1p(value)`` instead of raw ``value``.

    A pollutant that spans sub-unit to hundreds of µg/m³ spends most of its
    raw-scale variance on high hours; fitting on the log scale makes errors
    count in relative terms, so a fixed relative miss maps to a bounded error
    at every level while its absolute error grows without bound. With
    ``back_transform="geometric"`` (the default, legacy behaviour), resampled
    log predictions are averaged on the log scale before a single ``expm1``;
    ``"arithmetic"`` back-transforms each draw before averaging; ``"smearing"``
    adds Duan's residual correction to that arithmetic mean.
    """
    back_transform: str = "geometric"
    smearing_factor: float | None = None
    paired: bool | str = False
    paired_floor: float = 1e-6
    return_quantiles: Sequence[float] | None = None
    return_mc_se: bool = False
    stratify_by: str | Sequence[str] | None = None
    conditional_k: int | None = None
    conditional_on: Mapping[str, Any] | None = None
    resample_pools: Mapping[str, pd.DataFrame] | None = None
    sampling: str = "iid"
    block_hours: int | None = None
    cache: str | Path | None = None
    impute: str = "median"
    """Missing-covariate strategy forwarded to :func:`prepare_data`.

    ``"median"`` (the default) fills from the full-frame median, matching the
    behaviour before the strategy was configurable; ``"train_median"`` opts
    into training-set-only statistics (leak-free for the test block).
    """
    max_gap: int | None = 6
    embargo: int | None = None
    """Rows purged around a testing block.

    ``None`` (the default) means 0 for the ordinary split and
    :func:`crossfit_normalise`'s own 24-row default when ``cross_fit=True``;
    an explicit value (including ``0``) is used as given.
    """
    cross_fit: bool = False
    cv_folds: int = 5
    chronos_mode: str = "deweather"
    normalise_method: str = "standard"
    """Deweathering engine: ``"standard"`` (fitted model + Monte-Carlo
    resampling) or ``"dml"`` (orthogonal partialling-out with a smooth
    low-frequency nuisance basis). ``"dml"`` does its own blocked
    out-of-fold pass, so it is incompatible with ``cross_fit=True`` and with
    the Chronos-2 backend; the returned model slot then holds the
    :class:`~normet.analysis.normalise.DMLNormaliseResult`."""
    control_variates: bool = False
    """Use Monte-Carlo control variates in :func:`normalise` (variance
    reduction; requires resampled variables). Forwarded only on the
    ``"standard"`` and ``"dml"`` paths."""
    block_mode: str = "independent"
    """Block-resampling mode for the meteorological draws: ``"independent"``
    or ``"markov"``. ``"markov"`` needs ``block_hours > 1`` to differ from
    independent sampling."""
    dml_cutoff_days: float = 30.0
    """Smooth-basis cutoff (days) for ``normalise_method="dml"``."""
    dml_n_harmonics: int | None = None
    """Number of low-frequency cosine terms for ``normalise_method="dml"``;
    ``None`` chooses it from the record span and ``dml_cutoff_days``."""


@dataclass
class UncConfig(SingleConfig):
    """Configuration for :func:`do_all_unc`; adds ensemble settings."""

    n_models: int = 10
    confidence_level: float = 0.95
    unc_method: str = "legacy"
    ci_method: str = "quantile"
    block_length: int | None = None


_ICN_DESIGN_KEYS = frozenset(
    {
        "intervene",
        "pool",
        "gap_h",
        "block_h",
        "stride",
        "direction",
        "aggregate",
        "lead_tau",
        "min_context",
        "support_min_frac",
    }
)


def _do_all_zero_shot(
    df: pd.DataFrame, cfg: SingleConfig
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    """The Chronos-2 branch of :func:`do_all`: prepare -> load -> de-weather / ICN.

    The AutoML path is prepare -> train -> normalise, and the middle step has no
    counterpart here: nothing is fitted, so there is no held-out set, no model
    to persist and no feature importance. What is returned in the model slot is
    the loaded :class:`~normet.foundation.Chronos2Estimator`, which is what the
    caller needs to run further projections or the covariate-sensitivity check.

    ``chronos_mode="deweather"`` preserves the legacy past+future deweathering
    path; ``chronos_mode="icn"`` runs future-only In-Context Normalisation via
    :func:`~normet.foundation.in_context.normalise_in_context`.

    Two ICN spellings remain accepted so neither caller changes behaviour.
    ``model_config`` may carry ``method`` / ``estimate`` / ``icn_design`` /
    ``pool`` / ``model_path`` / ``estimator`` (the chronos-2 branch's spelling;
    it returns the compact ``observed/normalised/qNNN`` frame from
    :meth:`~normet.foundation.in_context.ICNResult.to_normet_frame`), or the
    design fields directly (``block_h``, ``stride``, ``min_context``, ... with
    ``chronos_mode="icn"``, the stack spelling; it returns the richer frame
    with the per-hour uncertainty decomposition). An explicit
    ``model_config['method']`` wins over ``chronos_mode``.
    """
    from ..foundation import (
        Chronos2Estimator,
        ICNDesign,
        MetPool,
        add_calendar_covariates,
        to_indexed_frame,
    )

    raw_mcfg = dict(cfg.model_config or {})
    explicit_method = raw_mcfg.pop("method", None)
    method = str(
        explicit_method if explicit_method is not None else (cfg.chronos_mode or "deweather")
    ).lower()
    estimate = str(raw_mcfg.pop("estimate", "level")).lower()
    model_path = raw_mcfg.pop("model_path", None)
    estimator_arg = raw_mcfg.pop("estimator", None)
    icn_design = raw_mcfg.pop("icn_design", None)
    ext_pool = raw_mcfg.pop("pool", None)
    custom_pipeline = raw_mcfg.pop("pipeline", None)
    if method not in ("deweather", "icn"):
        raise ValueError(f"model_config['method'] must be 'deweather' or 'icn', got {method!r}")
    if estimate not in ("level", "paired"):
        raise ValueError(f"model_config['estimate'] must be 'level' or 'paired', got {estimate!r}")
    mode = str(cfg.chronos_mode or "deweather").lower()
    if mode not in ("deweather", "icn"):
        raise ConfigError(f"`chronos_mode` must be 'deweather' or 'icn', got {cfg.chronos_mode!r}.")
    # Without an explicit ``method`` the stack's spelling applies: the design
    # axes live in ``model_config`` and ICN runs on the caller's frame scale
    # (log1p applied here) rather than through ``design.log_transform``.
    stack_dialect = explicit_method is None
    design_kw: dict[str, Any] = {}
    if method == "icn":
        design_kw = {
            k: raw_mcfg.pop(k) for k in list(raw_mcfg) if k in _ICN_DESIGN_KEYS and k != "pool"
        }
        if isinstance(ext_pool, str):
            # ``pool`` is an ICNDesign field name; an object reserves the
            # external-weather meaning used by the chronos spelling.
            design_kw["pool"] = ext_pool
            ext_pool = None

    covariates = list(cfg.covariates or [])
    met = list(cfg.variables_resample) if cfg.variables_resample else covariates
    if not met:
        raise ValueError(
            "the chronos-2 backend needs meteorological covariates to condition on: "
            "pass variables_resample (or covariates), otherwise the run is a plain "
            "forecast rather than a de-weathering"
        )
    # prepare keeps the columns its covariates list names: a call that names the
    # meteorology through variables_resample alone would otherwise leave none of
    # it in the frame the de-weathering conditions on. The union leaves the
    # covariates-only and covariates-plus-identical-resample calls unchanged.
    kept = covariates + [c for c in met if c not in covariates]

    df_prep = prepare_data(
        df,
        target=cfg.target,
        covariates=kept,
        split_method=cfg.split_method,
        train_fraction=cfg.train_fraction,
        seed=cfg.seed,
        impute=cfg.impute,
        max_gap=cfg.max_gap,
        embargo=_prepare_embargo(cfg.embargo),
    )
    if cfg.log_transform and (method == "deweather" or stack_dialect):
        df_prep = log1p_target(df_prep)
    indexed = to_indexed_frame(df_prep)

    if estimator_arg is not None:
        estimator = estimator_arg
        if estimator.met_covariates is None:
            estimator.met_covariates = list(met)
        if custom_pipeline is None and getattr(estimator, "_pipeline", None) is None:
            estimator._load_pipeline()
    elif model_path is not None:
        estimator = Chronos2Estimator.load(model_path, device=raw_mcfg.get("device"))
        if estimator.met_covariates is None:
            estimator.met_covariates = list(met)
    else:
        estimator = Chronos2Estimator(met_covariates=met, **raw_mcfg)
        if custom_pipeline is None:
            estimator._load_pipeline()
    log.info(
        "Chronos-2 ready on %s | context=%d | horizon=%d",
        estimator.device,
        estimator.context_length,
        estimator.prediction_length,
    )
    estimator._check_index(indexed.index)

    if method == "icn" and not stack_dialect:
        # Chronos-2 spelling: future-only ICNDesign.v1 on a frame carrying
        # the calendar covariates, converted to the compact normet frame.
        indexed_icn = add_calendar_covariates(indexed)
        if isinstance(icn_design, ICNDesign):
            design = icn_design
        else:
            overrides: dict[str, Any] = {
                "context_length": estimator.context_length,
                "n_resamples": cast(int, cfg.n_samples),
                "seed": cfg.seed,
                "log_transform": bool(cfg.log_transform),
            }
            if "prediction_length" in raw_mcfg:
                bh = int(raw_mcfg["prediction_length"])
                overrides["block_h"] = bh
                overrides["stride"] = max(1, bh // 2)
            overrides.update(design_kw)
            if isinstance(icn_design, Mapping):
                overrides.update(icn_design)
            design = ICNDesign.v1(**overrides)
        pool_arg: Any = ext_pool
        if pool_arg is None and cfg.resample_df is not None:
            pool_arg = MetPool.from_frame(to_indexed_frame(cfg.resample_df), met)
        icn_res = estimator.normalise_in_context(
            indexed_icn,
            "value",
            design=design,
            pipeline=custom_pipeline,
            met_features=met,
            pool=pool_arg,
            quantiles=(0.1, 0.5, 0.9),
        )
        out = icn_res.to_normet_frame(estimate=estimate)
        uncovered = int(icn_res.diagnostics["uncovered_hours"])
        log.info(
            "do_all (ICN-v1, %s) finished: %d timestamps, %d uncovered hours",
            estimate,
            len(out),
            uncovered,
        )
        return out, estimator, df_prep

    if method == "icn":
        # Stack spelling: the design is built from ``model_config``'s design
        # fields and the result keeps the uncertainty decomposition.
        if isinstance(icn_design, ICNDesign):
            design = icn_design
        else:
            if isinstance(icn_design, Mapping):
                design_kw.update(icn_design)
            design = ICNDesign(
                n_resamples=cast(int, cfg.n_samples),
                seed=cfg.seed,
                context_length=estimator.context_length,
                **design_kw,
            )
        quantiles = tuple(cfg.return_quantiles) if cfg.return_quantiles is not None else None
        icn_res = estimator.normalise_in_context(
            indexed,
            "value",
            design=design,
            pipeline=custom_pipeline,
            met_features=met,
            schema="normet",
            pool=ext_pool,
            quantiles=quantiles,
        )
        obs = indexed["value"].reindex(icn_res.level.index).to_numpy(dtype=float)
        norm = icn_res.level.to_numpy(dtype=float)
        pair = icn_res.paired.to_numpy(dtype=float)
        anchor_spread = icn_res.spread_by_source["anchor"].to_numpy(dtype=float)
        mc_spread = icn_res.spread_by_source["mc"].to_numpy(dtype=float)
        if cfg.log_transform:
            obs = np.expm1(obs)
            norm = np.expm1(norm)
            pair = np.expm1(pair)
            # ICN reports its spreads on the input frame's scale, which here
            # is log1p. Convert with the delta method for y = expm1(z),
            # dy/dz = exp(z) = 1 + y, so the columns stay in the same units
            # as `normalised`; leaving log-scale SDs beside expm1'd levels
            # made a +/- 2 SD band invisible.
            scale = 1.0 + norm
            anchor_spread = anchor_spread * scale
            mc_spread = mc_spread * scale
        out = pd.DataFrame(
            {
                "observed": obs,
                "normalised": norm,
                "paired": pair,
                "weather_effect": obs - norm,
                "anchor_spread": anchor_spread,
                "mc_spread": mc_spread,
                "hour_contrib": icn_res.hour_contrib.to_numpy(dtype=int),
            },
            index=pd.Index(icn_res.level.index, name="date"),
        )
        if icn_res.quantiles is not None:
            for col in icn_res.quantiles.columns:
                vals = icn_res.quantiles[col].reindex(icn_res.level.index).to_numpy(dtype=float)
                out[col] = np.expm1(vals) if cfg.log_transform else vals
        uncovered = int(icn_res.diagnostics["uncovered_hours"])
        log.info(
            "do_all (ICN, %s) finished: %d timestamps, %d uncovered hours",
            estimate,
            len(out),
            uncovered,
        )
        return out, estimator, df_prep

    out = estimator.deweather(
        indexed,
        "value",
        met_features=met,
        n_samples=cast(int, cfg.n_samples),
        random_state=cfg.seed,
        schema="normet",
    )
    out.index.name = "date"
    if cfg.log_transform:
        out = expm1_columns(out)

    seeded = min(estimator.context_length, len(out))
    log.info(
        "do_all finished: %d timestamps, of which the first %d are seeded with the "
        "observed values (Chronos-2 has no history to condition on until then)",
        len(out),
        seeded,
    )
    return out, estimator, df_prep


def _do_all_dml(df: pd.DataFrame, cfg: SingleConfig) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    """The ``normalise_method="dml"`` branch of :func:`do_all`.

    ``dml_normalise`` owns its own preparation, blocked folds and nuisance
    basis, so the trained-model slot is the :class:`DMLNormaliseResult`
    instead of an estimator. ``df_prep`` is still prepared with the same
    split settings as the standard path so run archives and reports keep the
    same contract.
    """
    paired_opt: bool | str = (
        ("log_ratio" if cfg.log_transform else "additive") if cfg.paired is False else cfg.paired
    )
    res = dml_normalise(
        df,
        target=cfg.target,
        covariates=list(cfg.covariates or []),
        variables_resample=cfg.variables_resample,
        backend=cfg.backend,
        model_config=cfg.model_config,
        cutoff_days=float(cfg.dml_cutoff_days),
        n_harmonics=cfg.dml_n_harmonics,
        n_splits=int(cfg.cv_folds),
        embargo=(_CROSSFIT_DEFAULT_EMBARGO if cfg.embargo is None else int(cfg.embargo)),
        paired=paired_opt,
        paired_floor=float(cfg.paired_floor),
        n_samples=int(cast(int, cfg.n_samples)),
        log_transform=bool(cfg.log_transform),
        back_transform=cfg.back_transform,
        stratify_by=cfg.stratify_by,
        conditional_k=cfg.conditional_k,
        conditional_on=cfg.conditional_on,
        resample_pools=cfg.resample_pools,
        resample_df=cfg.resample_df,
        return_quantiles=cfg.return_quantiles,
        return_mc_se=bool(cfg.return_mc_se),
        sampling=cfg.sampling,
        control_variates=bool(cfg.control_variates),
        block_hours=cfg.block_hours,
        block_mode=cfg.block_mode,
        seed=int(cfg.seed),
        n_cores=cfg.n_cores,
        memory_save=bool(cfg.memory_save),
        cache=cfg.cache,
        impute=cfg.impute,
        max_gap=cfg.max_gap,
        verbose=bool(cfg.verbose),
    )
    df_prep = prepare_data(
        df,
        target=cfg.target,
        covariates=list(cfg.covariates or []),
        split_method=cfg.split_method,
        train_fraction=cfg.train_fraction,
        seed=cfg.seed,
        embargo=_prepare_embargo(cfg.embargo),
    )
    log.info("do_all finished (dml): %d timestamps", len(res.normalised))
    return res.normalised, res, df_prep


def _prepare_embargo(embargo: int | None) -> int:
    """Map :attr:`SingleConfig.embargo`'s ``None`` sentinel to 0 for ``prepare_data``.

    ``None`` only means "the caller did not choose"; the ordinary split has
    never purged rows by default, whereas cross-fitting uses
    :data:`~normet.analysis.normalise._CROSSFIT_DEFAULT_EMBARGO`.
    """
    return 0 if embargo is None else int(embargo)


def _resolve_single_config(config: SingleConfig | None = None, **kwargs: Any) -> SingleConfig:
    return resolve_config(SingleConfig, config, **kwargs)


def _resolve_unc_config(config: UncConfig | None = None, **kwargs: Any) -> UncConfig:
    return resolve_config(UncConfig, config, **kwargs)


def _stats_for_member(
    df_eval: pd.DataFrame,
    model: object,
    *,
    log_transform: bool,
    back_transform: str,
    seed_val: int,
) -> pd.DataFrame | None:
    """Compute member statistics, keeping legacy signature when log_transform is False."""
    if not log_transform:
        stats_i = modStats(df=df_eval, model=model, subset=None, statistic=None)
    else:
        from ..model.predict import ml_predict

        df_orig = df_eval.copy()
        df_orig["value"] = np.expm1(
            pd.to_numeric(df_orig["value"], errors="coerce").to_numpy(dtype=float)
        )
        phi_i = 1.0
        if back_transform == "smearing":
            sub = df_eval[df_eval["set"] == "training"] if "set" in df_eval.columns else df_eval
            if sub.empty:
                sub = df_eval
            preds_tr = ml_predict(model, sub, num_threads=1)
            phi_i = compute_duan_smearing(sub["value"], preds_tr)

        def _orig_pred(m: object, x: pd.DataFrame) -> np.ndarray:
            return _back_transform_values(ml_predict(m, x, num_threads=1), smearing_factor=phi_i)

        stats_i = modStats(
            df=df_orig,
            model=model,
            subset=None,
            statistic=None,
            predictor=_orig_pred,
        )
    if isinstance(stats_i, pd.DataFrame) and not stats_i.empty:
        stats_i = stats_i.copy()
        stats_i["seed"] = int(seed_val)
        return stats_i
    return None


def do_all(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    config: SingleConfig | None = None,
    backend: str | None = None,
    covariates: list[str] | None = None,
    variables_resample: list[str] | None = None,
    split_method: str | None = None,
    train_fraction: float | None = None,
    model_config: dict[str, Any] | None = None,
    n_samples: int | None = None,
    seed: int | None = None,
    n_cores: int | None = None,
    resample_df: pd.DataFrame | None = None,
    memory_save: bool | None = None,
    verbose: bool | None = None,
    aggregate: bool | None = None,
    log_transform: bool | None = None,
    back_transform: str | None = None,
    smearing_factor: float | None = None,
    paired: bool | str | None = None,
    paired_floor: float | None = None,
    return_quantiles: Sequence[float] | None = None,
    return_mc_se: bool | None = None,
    stratify_by: str | Sequence[str] | None = None,
    conditional_k: int | None = None,
    conditional_on: Mapping[str, Any] | None = None,
    resample_pools: Mapping[str, pd.DataFrame] | None = None,
    sampling: str | None = None,
    block_hours: int | None = None,
    cache: str | Path | None = None,
    impute: str | None = None,
    max_gap: int | None = None,
    embargo: int | None = None,
    cross_fit: bool | None = None,
    cv_folds: int | None = None,
    chronos_mode: str | None = None,
    normalise_method: str | None = None,
    control_variates: bool | None = None,
    block_mode: str | None = None,
    dml_cutoff_days: float | None = None,
    dml_n_harmonics: int | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    r"""Run the standard pipeline: prepare → build_model → normalise.

    With ``backend="chronos-2"`` the middle step is skipped: nothing is trained,
    the Chronos-2 checkpoint is loaded instead and the de-weathering runs through
    :meth:`~normet.foundation.Chronos2Estimator.deweather` (``chronos_mode="deweather"``)
    or :func:`~normet.foundation.in_context.normalise_in_context` (``chronos_mode="icn"``).
    With ``cross_fit=True`` on AutoML backends, runs blocked embargoed out-of-fold
    normalisation via :func:`crossfit_normalise` and returns
    ``(result_df, CrossFitNormaliseResult, df_prep)``. Leaving ``embargo`` at
    its (``None``) default there uses ``crossfit_normalise``'s 24-row purge;
    passing ``embargo=0`` explicitly requests no purge.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    target : str or None, default=None
        Target column in *df*. Can be ``None`` when *config* is provided
        with a non-None ``target`` field.
    config : SingleConfig or None, default=None
        Optional :class:`SingleConfig` holding all parameters.
        Individual keyword arguments override fields on *config*\ -- but only
        when they are given: a keyword left at ``None`` (every one of the
        function's own defaults) leaves the config's value, or the
        dataclass's default, in place.
    log_transform : bool or None, default=None
        Fit on ``log1p(value)`` instead of the raw target. With it on, the model
        trains on the log scale and results return in original units according to
        ``back_transform`` (``"geometric"``, the default and legacy behaviour /
        ``"arithmetic"`` / ``"smearing"``);
        the prepared frame returned as the third tuple item carries ``log1p(value)``.
    normalise_method : {"standard", "dml"} or None, default=None
        De-weathering engine. ``"standard"`` runs Monte-Carlo resampling of the
        meteorological covariates (the historical path). ``"dml"`` runs
        :func:`~normet.analysis.normalise.dml_normalise` -- orthogonal Double-ML
        with a smooth low-frequency nuisance basis -- which owns its own blocked
        out-of-fold pass, so it cannot be combined with ``cross_fit=True``; it is
        also unavailable on the Chronos-2 backend, whose ``chronos_mode`` already
        selects the engine.
    control_variates, block_mode : bool / str or None, default=None
        Variance reduction for the standard engine: ``control_variates=True``
        uses the meteorological covariates as control variates, and
        ``block_mode="markov"`` resamples trajectory blocks with a Markov bridge
        instead of independent hours (``"independent"``, the default).
    dml_cutoff_days, dml_n_harmonics : float / int or None, default=None
        The Double-ML nuisance basis: harmonics up to the cutoff period
        (30 d in :func:`dml_normalise`) and an optional explicit harmonic count.
    **kwargs
        Any additional keyword arguments are forwarded to
        :func:`_resolve_single_config`.

    Returns
    -------
    (pandas.DataFrame, object, pandas.DataFrame)
        Normalised contributions, trained model (or :class:`CrossFitNormaliseResult`),
        prepared dataframe.
    """
    if target is None and config is None:
        # SingleConfig's target default is "value"; a caller who gave no
        # target at all gets the old, explicit refusal rather than a silent
        # attempt to find a column called "value".
        raise ValueError("`target` must be provided either directly or via config.")
    _cfg = _resolve_single_config(
        # An instance (empty when none was given) keeps ``None`` meaning
        # "not given": resolve_config drops None-valued overrides only when
        # it has a config to override.
        config=config or SingleConfig(),
        target=target,
        backend=backend,
        covariates=covariates,
        variables_resample=variables_resample,
        split_method=split_method,
        train_fraction=train_fraction,
        model_config=model_config,
        n_samples=n_samples,
        seed=seed,
        n_cores=n_cores,
        resample_df=resample_df,
        memory_save=memory_save,
        verbose=verbose,
        aggregate=aggregate,
        log_transform=log_transform,
        back_transform=back_transform,
        smearing_factor=smearing_factor,
        paired=paired,
        paired_floor=paired_floor,
        return_quantiles=return_quantiles,
        return_mc_se=return_mc_se,
        stratify_by=stratify_by,
        conditional_k=conditional_k,
        conditional_on=conditional_on,
        resample_pools=resample_pools,
        sampling=sampling,
        block_hours=block_hours,
        cache=cache,
        impute=impute,
        max_gap=max_gap,
        embargo=embargo,
        cross_fit=cross_fit,
        cv_folds=cv_folds,
        chronos_mode=chronos_mode,
        normalise_method=normalise_method,
        control_variates=control_variates,
        block_mode=block_mode,
        dml_cutoff_days=dml_cutoff_days,
        dml_n_harmonics=dml_n_harmonics,
        **kwargs,
    )

    if _cfg.target is None:
        raise ValueError("`target` must be provided either directly or via config.")

    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)
    if _cfg.normalise_method not in ("standard", "dml"):
        raise ConfigError(
            f"`normalise_method` must be 'standard' or 'dml', got {_cfg.normalise_method!r}."
        )
    if _cfg.block_mode not in ("independent", "markov"):
        raise ConfigError(
            f"`block_mode` must be 'independent' or 'markov', got {_cfg.block_mode!r}."
        )
    if _cfg.normalise_method == "dml" and _cfg.cross_fit:
        raise ConfigError(
            "`normalise_method='dml'` runs its own blocked out-of-fold pass; set cross_fit=False."
        )
    if _cfg.backend == CHRONOS_BACKEND and _cfg.normalise_method != "standard":
        raise ConfigError(
            "`normalise_method` selects an AutoML de-weathering engine; the "
            "Chronos-2 backend has its own (chronos_mode/ICN)."
        )
    log.info(
        "Starting do_all | backend=%s | target=%s | n_samples=%d",
        _cfg.backend,
        _cfg.target,
        _cfg.n_samples,
    )

    if _cfg.backend == CHRONOS_BACKEND:
        return _do_all_zero_shot(df, _cfg)

    if _cfg.covariates is None:
        raise ValueError("covariates must be provided")

    if _cfg.normalise_method == "dml":
        return _do_all_dml(df, _cfg)

    if _cfg.cross_fit:
        paired_opt: bool | str = (
            ("log_ratio" if _cfg.log_transform else "additive")
            if _cfg.paired is False
            else _cfg.paired
        )
        cf_res = crossfit_normalise(
            df,
            target=_cfg.target,
            covariates=list(_cfg.covariates),
            variables_resample=_cfg.variables_resample,
            backend=_cfg.backend,
            model_config=_cfg.model_config,
            n_splits=int(_cfg.cv_folds),
            embargo=(_CROSSFIT_DEFAULT_EMBARGO if _cfg.embargo is None else int(_cfg.embargo)),
            paired=paired_opt,
            paired_floor=float(_cfg.paired_floor),
            n_samples=int(_cfg.n_samples),
            log_transform=bool(_cfg.log_transform),
            back_transform=_cfg.back_transform,
            stratify_by=_cfg.stratify_by,
            conditional_k=_cfg.conditional_k,
            conditional_on=_cfg.conditional_on,
            resample_pools=_cfg.resample_pools,
            resample_df=_cfg.resample_df,
            return_quantiles=_cfg.return_quantiles,
            return_mc_se=bool(_cfg.return_mc_se),
            sampling=_cfg.sampling,
            block_hours=_cfg.block_hours,
            control_variates=_cfg.control_variates,
            block_mode=_cfg.block_mode,
            seed=int(_cfg.seed),
            n_cores=_cfg.n_cores,
            memory_save=bool(_cfg.memory_save),
            cache=_cfg.cache,
            impute=_cfg.impute,
            max_gap=_cfg.max_gap,
            verbose=bool(_cfg.verbose),
        )
        df_prep_cf = prepare_data(
            df,
            target=_cfg.target,
            covariates=_cfg.covariates,
            split_method="ts",
            train_fraction=1.0,
            seed=_cfg.seed,
            impute=_cfg.impute,
            max_gap=_cfg.max_gap,
        )
        if _cfg.log_transform:
            df_prep_cf = log1p_target(df_prep_cf)
        return cf_res.normalised, cf_res, df_prep_cf

    df_prep = prepare_data(
        df,
        target=_cfg.target,
        covariates=_cfg.covariates,
        split_method=_cfg.split_method,
        train_fraction=_cfg.train_fraction,
        seed=_cfg.seed,
        impute=_cfg.impute,
        max_gap=_cfg.max_gap,
        embargo=_prepare_embargo(_cfg.embargo),
    )
    if _cfg.log_transform:
        df_prep = log1p_target(df_prep)
    log.info(
        "Data prepared: %d rows (%d training, %d testing)",
        len(df_prep),
        int((df_prep["set"] == "training").sum()) if "set" in df_prep.columns else len(df_prep),
        int((df_prep["set"] == "testing").sum()) if "set" in df_prep.columns else 0,
    )

    df_prep, model = build_model(
        df=df_prep,
        target="value",
        backend=_cfg.backend,
        covariates=_cfg.covariates,
        split_method=_cfg.split_method,
        train_fraction=_cfg.train_fraction,
        model_config=_cfg.model_config,
        seed=_cfg.seed,
        n_cores=_cfg.n_cores,
        verbose=_cfg.verbose,
        cache=_cfg.cache,
    )
    log.info("Model trained with backend=%s", _cfg.backend)

    out = normalise(
        df=df_prep,
        model=model,
        covariates=_cfg.covariates or [c for c in df_prep.columns if c not in {"value"}],
        variables_resample=_cfg.variables_resample,
        n_samples=_cfg.n_samples,
        aggregate=_cfg.aggregate,
        seed=_cfg.seed,
        n_cores=_cfg.n_cores,
        resample_df=_cfg.resample_df,
        memory_save=_cfg.memory_save,
        verbose=_cfg.verbose,
        log_transform=_cfg.log_transform,
        back_transform=_cfg.back_transform,
        smearing_factor=_cfg.smearing_factor,
        paired=_cfg.paired,
        paired_floor=_cfg.paired_floor,
        return_quantiles=_cfg.return_quantiles,
        return_mc_se=_cfg.return_mc_se,
        stratify_by=_cfg.stratify_by,
        conditional_k=_cfg.conditional_k,
        conditional_on=_cfg.conditional_on,
        resample_pools=_cfg.resample_pools,
        sampling=_cfg.sampling,
        block_hours=_cfg.block_hours,
        control_variates=_cfg.control_variates,
        block_mode=_cfg.block_mode,
        cache=_cfg.cache,
    )

    log.info("do_all finished: %d timestamps", len(out))
    return out, model, df_prep


def do_all_unc(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    config: UncConfig | None = None,
    backend: str | None = None,
    covariates: list[str] | None = None,
    variables_resample: list[str] | None = None,
    split_method: str | None = None,
    train_fraction: float | None = None,
    model_config: dict[str, Any] | None = None,
    n_samples: int | None = None,
    n_models: int | None = None,
    confidence_level: float | None = None,
    seed: int | None = None,
    n_cores: int | None = None,
    resample_df: pd.DataFrame | None = None,
    memory_save: bool | None = None,
    verbose: bool | None = None,
    weighted_method: str | None = None,
    log_transform: bool | None = None,
    back_transform: str | None = None,
    smearing_factor: float | None = None,
    paired: bool | str | None = None,
    paired_floor: float | None = None,
    stratify_by: str | Sequence[str] | None = None,
    conditional_k: int | None = None,
    conditional_on: Mapping[str, Any] | None = None,
    resample_pools: Mapping[str, pd.DataFrame] | None = None,
    sampling: str | None = None,
    block_hours: int | None = None,
    cache: str | Path | None = None,
    impute: str | None = None,
    max_gap: int | None = None,
    embargo: int | None = None,
    cross_fit: bool | None = None,
    cv_folds: int | None = None,
    normalise_method: str | None = None,
    control_variates: bool | None = None,
    block_mode: str | None = None,
    dml_cutoff_days: float | None = None,
    dml_n_harmonics: int | None = None,
    unc_method: str | None = None,
    ci_method: str | None = None,
    block_length: int | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""Run the uncertainty pipeline: :func:`do_all` repeated *n_models* times.

    Supports three ensemble uncertainty strategies via ``unc_method``:
      - ``"legacy"``: repeats :func:`do_all` across ``n_models`` seeds.
      - ``"block_bootstrap"``: circular moving-block bootstrap over the full
        record with leak-free per-bootstrap imputation, out-of-bag skill metrics,
        and common random weather resamples across replicates.
      - ``"crossfit_ensemble"``: repeated blocked out-of-fold ensembles with
        pooled OOF skill metrics.

    Confidence intervals use empirical order statistics (``ci_method="quantile"``)
    or small-sample Student-*t* intervals (``ci_method="student_t"``), and model
    weights support ``"r2"``, ``"rmse"``, ``"inv_mse"``, or ``"uniform"``.
    """
    if target is None and config is None:
        raise ValueError("`target` must be provided either directly or via config.")
    _cfg = _resolve_unc_config(
        config=config or UncConfig(),
        target=target,
        backend=backend,
        covariates=covariates,
        variables_resample=variables_resample,
        split_method=split_method,
        train_fraction=train_fraction,
        model_config=model_config,
        n_samples=n_samples,
        n_models=n_models,
        confidence_level=confidence_level,
        seed=seed,
        n_cores=n_cores,
        resample_df=resample_df,
        memory_save=memory_save,
        verbose=verbose,
        log_transform=log_transform,
        back_transform=back_transform,
        smearing_factor=smearing_factor,
        paired=paired,
        paired_floor=paired_floor,
        stratify_by=stratify_by,
        conditional_k=conditional_k,
        conditional_on=conditional_on,
        resample_pools=resample_pools,
        sampling=sampling,
        block_hours=block_hours,
        cache=cache,
        impute=impute,
        max_gap=max_gap,
        embargo=embargo,
        cross_fit=cross_fit,
        cv_folds=cv_folds,
        normalise_method=normalise_method,
        control_variates=control_variates,
        block_mode=block_mode,
        dml_cutoff_days=dml_cutoff_days,
        dml_n_harmonics=dml_n_harmonics,
        unc_method=unc_method,
        ci_method=ci_method,
        block_length=block_length,
        **kwargs,
    )

    weighted_method = (weighted_method or "r2").lower()
    if weighted_method not in {"r2", "rmse", "inv_mse", "uniform"}:
        raise ValueError("`weighted_method` must be 'r2', 'rmse', 'inv_mse', or 'uniform'.")
    unc_mode = str(_cfg.unc_method or "legacy").lower()
    if unc_mode not in {"legacy", "block_bootstrap", "crossfit_ensemble"}:
        raise ValueError(
            "`unc_method` must be 'legacy', 'block_bootstrap', or 'crossfit_ensemble'."
        )
    ci_mode = str(_cfg.ci_method or "quantile").lower()
    if ci_mode not in {"quantile", "student_t"}:
        raise ValueError("`ci_method` must be 'quantile' or 'student_t'.")
    if _cfg.normalise_method not in ("standard", "dml"):
        raise ConfigError(
            f"`normalise_method` must be 'standard' or 'dml', got {_cfg.normalise_method!r}."
        )
    if _cfg.block_mode not in ("independent", "markov"):
        raise ConfigError(
            f"`block_mode` must be 'independent' or 'markov', got {_cfg.block_mode!r}."
        )
    if _cfg.normalise_method == "dml" and unc_mode != "legacy":
        raise ConfigError(
            "`normalise_method='dml'` is available through the 'legacy' ensemble "
            "(which repeats do_all) only; block_bootstrap and crossfit_ensemble "
            "use the standard normalise path."
        )
    if not (0.0 < float(_cfg.confidence_level) < 1.0):
        raise ValueError("`confidence_level` must be strictly between 0 and 1.")

    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)
    rng = np.random.default_rng(_cfg.seed)
    seeds = rng.choice(1_000_001, size=_cfg.n_models, replace=False).tolist()

    series_list: list[pd.DataFrame] = []
    stats_list: list[pd.DataFrame] = []
    observed_ref: pd.Series | None = None
    series_col = "paired" if _cfg.paired else "normalised"

    t0 = time.time()
    if unc_mode == "block_bootstrap":
        if not _cfg.covariates:
            raise ValueError("covariates must be provided for `unc_method='block_bootstrap'`.")
        df_base = prepare_data(
            df,
            target=_cfg.target,
            covariates=_cfg.covariates,
            dropna=True,
            split_method="ts",
            train_fraction=1.0,
            seed=_cfg.seed,
            impute="none",
        ).drop(columns=["set"])
        n = len(df_base)
        b_len = (
            int(_cfg.block_length)
            if _cfg.block_length is not None
            else max(24, min(168, max(1, n // 10)))
        )
        b_len = max(1, min(b_len, n))
        cov_list = list(_cfg.covariates)

        for i, s in enumerate(seeds, start=1):
            (log.info if _cfg.verbose else log.debug)(
                "do_all_unc [block_bootstrap]: member %d/%d (seed=%d) %s",
                i,
                _cfg.n_models,
                s,
                _progress_str(i, _cfg.n_models, t0),
            )
            boot_rng = np.random.default_rng(int(s))
            n_blocks = int(np.ceil(n / b_len))
            starts = boot_rng.integers(0, n, size=n_blocks)
            boot_idx = np.concatenate([(st + np.arange(b_len)) % n for st in starts])[:n]
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[np.unique(boot_idx)] = False

            df_boot_raw = df_base.iloc[boot_idx].copy().reset_index(drop=True)
            df_boot_raw["set"] = "training"
            df_eval_raw = df_base.copy().reset_index(drop=True)
            df_eval_raw["set"] = "testing"
            combined = pd.concat([df_boot_raw, df_eval_raw], ignore_index=True)
            # The combined frame labels the bootstrap sample "training" and
            # the out-of-bag rows "testing"; the fill statistics must come
            # from the former, so the full-frame "median" strategy maps to
            # the training-only one here.
            imputed = impute_values(
                combined,
                dropna=False,
                strategy="train_median" if _cfg.impute == "median" else _cfg.impute,
                max_gap=_cfg.max_gap,
            )
            df_boot_tr = imputed.iloc[:n].copy().reset_index(drop=True)
            df_eval_i = imputed.iloc[n:].copy().reset_index(drop=True)
            df_eval_i["set"] = np.where(oob_mask, "testing", "in_bag")
            if _cfg.log_transform:
                df_boot_tr = log1p_target(df_boot_tr)
                df_eval_i = log1p_target(df_eval_i)

            model_i = train_model(
                df=df_boot_tr,
                target="value",
                backend=_cfg.backend,
                covariates=cov_list,
                model_config=_cfg.model_config,
                seed=int(s),
                verbose=_cfg.verbose and i == 1,
                n_cores=_cfg.n_cores,
                cache=_cfg.cache,
            )
            # Common random numbers across replicates isolate model & data uncertainty.
            out_i = normalise(
                df=df_eval_i,
                model=model_i,
                covariates=cov_list,
                variables_resample=_cfg.variables_resample,
                n_samples=_cfg.n_samples,
                aggregate=True,
                seed=int(_cfg.seed),
                n_cores=_cfg.n_cores,
                resample_df=_cfg.resample_df,
                memory_save=_cfg.memory_save,
                verbose=False,
                log_transform=_cfg.log_transform,
                back_transform=_cfg.back_transform,
                smearing_factor=_cfg.smearing_factor,
                paired=_cfg.paired,
                paired_floor=_cfg.paired_floor,
                stratify_by=_cfg.stratify_by,
                conditional_k=_cfg.conditional_k,
                conditional_on=_cfg.conditional_on,
                resample_pools=_cfg.resample_pools,
                sampling=_cfg.sampling,
                block_hours=_cfg.block_hours,
                control_variates=_cfg.control_variates,
                block_mode=_cfg.block_mode,
                cache=_cfg.cache,
            )
            if observed_ref is None:
                observed_ref = out_i["observed"].copy()
            pick = series_col if series_col in out_i.columns else "normalised"
            series_list.append(out_i[[pick]].rename(columns={pick: f"normalised_{s}"}))
            try:
                eval_for_stats = (
                    df_eval_i if bool(oob_mask.any()) else df_eval_i.assign(set="testing")
                )
                st = _stats_for_member(
                    eval_for_stats,
                    model_i,
                    log_transform=bool(_cfg.log_transform),
                    back_transform=_cfg.back_transform,
                    seed_val=int(s),
                )
                if st is not None:
                    stats_list.append(st)
            except Exception as e:
                log.warning("Failed to compute OOB metrics for seed %d: %s", s, e)

    elif unc_mode == "crossfit_ensemble":
        if not _cfg.covariates:
            raise ValueError("covariates must be provided for `unc_method='crossfit_ensemble'`.")
        paired_opt_cf: bool | str = (
            ("log_ratio" if _cfg.log_transform else "additive")
            if _cfg.paired is False
            else _cfg.paired
        )
        for i, s in enumerate(seeds, start=1):
            (log.info if _cfg.verbose else log.debug)(
                "do_all_unc [crossfit_ensemble]: member %d/%d (seed=%d) %s",
                i,
                _cfg.n_models,
                s,
                _progress_str(i, _cfg.n_models, t0),
            )
            cf_i = crossfit_normalise(
                df,
                target=_cfg.target,
                covariates=list(_cfg.covariates),
                variables_resample=_cfg.variables_resample,
                backend=_cfg.backend,
                model_config=_cfg.model_config,
                n_splits=int(_cfg.cv_folds),
                embargo=(_CROSSFIT_DEFAULT_EMBARGO if _cfg.embargo is None else int(_cfg.embargo)),
                paired=paired_opt_cf,
                paired_floor=float(_cfg.paired_floor),
                n_samples=int(_cfg.n_samples),
                log_transform=bool(_cfg.log_transform),
                back_transform=_cfg.back_transform,
                stratify_by=_cfg.stratify_by,
                conditional_k=_cfg.conditional_k,
                conditional_on=_cfg.conditional_on,
                resample_pools=_cfg.resample_pools,
                resample_df=_cfg.resample_df,
                sampling=_cfg.sampling,
                block_hours=_cfg.block_hours,
                control_variates=_cfg.control_variates,
                block_mode=_cfg.block_mode,
                seed=int(s),
                n_cores=_cfg.n_cores,
                memory_save=bool(_cfg.memory_save),
                cache=_cfg.cache,
                impute=_cfg.impute,
                max_gap=_cfg.max_gap,
                verbose=_cfg.verbose and i == 1,
            )
            out_i = cf_i.normalised
            if observed_ref is None:
                observed_ref = out_i["observed"].copy()
            pick = series_col if series_col in out_i.columns else "normalised"
            series_list.append(out_i[[pick]].rename(columns={pick: f"normalised_{s}"}))
            if not cf_i.oof_metrics.empty:
                st = cf_i.oof_metrics.copy()
                st["set"] = np.where(
                    st["fold"].astype(str) == "all", "testing", st["fold"].astype(str)
                )
                st["seed"] = int(s)
                stats_list.append(st)

    else:
        is_chronos = _cfg.backend == CHRONOS_BACKEND
        shared_chronos: object | None = None
        for i, s in enumerate(seeds, start=1):
            (log.info if _cfg.verbose else log.debug)(
                "do_all_unc: running model %d/%d (seed=%d) %s",
                i,
                _cfg.n_models,
                s,
                _progress_str(i, _cfg.n_models, t0),
            )

            mc_i = _cfg.model_config
            if is_chronos and shared_chronos is not None:
                # Chronos-2 is zero-shot: one checkpoint serves every member,
                # and only the weather draws differ across seeds.
                mc_i = {**(mc_i or {}), "estimator": shared_chronos}

            out_i, model_i, df_prep_i = do_all(
                df=df,
                target=_cfg.target,
                backend=_cfg.backend,
                covariates=_cfg.covariates,
                variables_resample=_cfg.variables_resample,
                split_method=_cfg.split_method,
                train_fraction=_cfg.train_fraction,
                model_config=mc_i,
                n_samples=_cfg.n_samples,
                seed=int(s),
                n_cores=_cfg.n_cores,
                resample_df=_cfg.resample_df,
                memory_save=_cfg.memory_save,
                aggregate=True,
                log_transform=_cfg.log_transform,
                back_transform=_cfg.back_transform,
                smearing_factor=_cfg.smearing_factor,
                paired=_cfg.paired,
                paired_floor=_cfg.paired_floor,
                stratify_by=_cfg.stratify_by,
                conditional_k=_cfg.conditional_k,
                conditional_on=_cfg.conditional_on,
                resample_pools=_cfg.resample_pools,
                sampling=_cfg.sampling,
                block_hours=_cfg.block_hours,
                cache=_cfg.cache,
                impute=_cfg.impute,
                max_gap=_cfg.max_gap,
                embargo=_cfg.embargo,
                cross_fit=_cfg.cross_fit,
                cv_folds=_cfg.cv_folds,
                chronos_mode=_cfg.chronos_mode,
                normalise_method=_cfg.normalise_method,
                control_variates=_cfg.control_variates,
                block_mode=_cfg.block_mode,
                dml_cutoff_days=_cfg.dml_cutoff_days,
                dml_n_harmonics=_cfg.dml_n_harmonics,
            )

            if is_chronos and shared_chronos is None:
                shared_chronos = model_i

            if observed_ref is None:
                observed_ref = out_i["observed"].copy()

            pick = series_col if series_col in out_i.columns else "normalised"
            series_list.append(out_i[[pick]].rename(columns={pick: f"normalised_{s}"}))

            if is_chronos:
                stats_list.append(
                    pd.DataFrame(
                        [
                            {
                                "seed": int(s),
                                "backend": CHRONOS_BACKEND,
                                "n_samples": _cfg.n_samples,
                            }
                        ]
                    )
                )
                continue

            try:
                if isinstance(model_i, CrossFitNormaliseResult):
                    if not model_i.oof_metrics.empty:
                        st = model_i.oof_metrics.copy()
                        st["set"] = np.where(
                            st["fold"].astype(str) == "all", "testing", st["fold"].astype(str)
                        )
                        st["seed"] = int(s)
                        stats_list.append(st)
                else:
                    st = _stats_for_member(
                        df_prep_i,
                        model_i,
                        log_transform=bool(_cfg.log_transform),
                        back_transform=_cfg.back_transform,
                        seed_val=int(s),
                    )
                    if st is not None:
                        stats_list.append(st)
            except Exception as e:
                log.warning("Failed to compute metrics for seed %d: %s", s, e)

    if observed_ref is None:
        raise RuntimeError("do_all_unc produced no outputs — verify inputs and seeds.")

    out = observed_ref.to_frame(name="observed")
    for s in series_list:
        out = out.join(s, how="outer")

    pred_cols = [c for c in out.columns if c.startswith("normalised_")]
    P = out[pred_cols]

    out["mean"] = P.mean(axis=1)
    out["std"] = P.std(axis=1)
    out["median"] = P.median(axis=1)

    alpha = (1.0 - float(_cfg.confidence_level)) / 2.0
    if ci_mode == "student_t":
        from scipy.stats import t as student_t

        dof = max(len(pred_cols) - 1, 1)
        t_crit = float(student_t.ppf(1.0 - alpha, df=dof))
        half = t_crit * out["std"].to_numpy(dtype=float)
        mean_arr = out["mean"].to_numpy(dtype=float)
        lower_arr = mean_arr - half
        upper_arr = mean_arr + half
        if _cfg.log_transform:
            lower_arr = np.maximum(lower_arr, 0.0)
        out["lower_bound"] = lower_arr
        out["upper_bound"] = upper_arr
    else:
        out["lower_bound"] = P.quantile(alpha, axis=1)
        out["upper_bound"] = P.quantile(1.0 - alpha, axis=1)

    def _pick_metric(df_in: pd.DataFrame, names: list[str]) -> float | None:
        frame = df_in
        if "set" in frame.columns:
            for wanted in ("testing", "all"):
                sel = frame[frame["set"] == wanted]
                if not sel.empty:
                    frame = sel
                    break
        for n in names:
            if n in frame.columns:
                try:
                    return float(frame[n].iloc[0])
                except Exception:
                    continue
        return None

    perf_rows = []
    for si in stats_list:
        sd = int(si["seed"].iloc[0])
        r2_val = _pick_metric(si, ["r2", "R2", "r_squared", "R2_score"])
        rmse_val = _pick_metric(si, ["rmse", "RMSE", "root_mean_squared_error"])
        perf_rows.append({"seed": sd, "r2": r2_val, "rmse": rmse_val})
    perf_df = (
        pd.DataFrame(perf_rows).set_index("seed")
        if perf_rows
        else pd.DataFrame(columns=["r2", "rmse"])
    )

    def _parse_seed(col: str) -> int | None:
        try:
            return int(col.split("_", 1)[1])
        except Exception:
            return None

    seeds_in_P = [s for s in map(_parse_seed, pred_cols)]

    scores = np.zeros(len(seeds_in_P), dtype=float)
    if weighted_method == "uniform":
        scores[:] = 1.0
    elif not perf_df.empty:
        if weighted_method == "r2":
            for i, s in enumerate(seeds_in_P):
                if s is None or s not in perf_df.index:
                    continue
                r2 = perf_df.loc[s, "r2"]
                if pd.notna(r2):
                    scores[i] = max(float(cast(Any, r2)), 0.0)
        else:
            eps = 1e-9
            for i, s in enumerate(seeds_in_P):
                if s is None or s not in perf_df.index:
                    continue
                rmse = perf_df.loc[s, "rmse"]
                if pd.notna(rmse):
                    rm = float(cast(Any, rmse))
                    scores[i] = (
                        1.0 / (rm * rm + eps) if weighted_method == "inv_mse" else 1.0 / (rm + eps)
                    )

    if np.all(~np.isfinite(scores)) or np.all(scores <= 0):
        w = np.full(len(pred_cols), 1.0 / len(pred_cols)) if pred_cols else np.array([])
    else:
        scores[:] = np.where(np.isfinite(scores) & (scores > 0), scores, 0.0)
        ssum = scores.sum()
        w = scores / ssum if ssum > 0 else np.full(len(pred_cols), 1.0 / len(pred_cols))

    if pred_cols:
        out["weighted"] = (P.values * w[np.newaxis, :]).sum(axis=1)
    else:
        out["weighted"] = np.nan

    w_by_seed = {s: float(w[i]) for i, s in enumerate(seeds_in_P) if i < len(w)}
    mod_stats = pd.concat(stats_list, ignore_index=True) if stats_list else pd.DataFrame()
    if not mod_stats.empty and "seed" in mod_stats.columns:
        mod_stats["weight"] = mod_stats["seed"].map(w_by_seed).astype(float)

    return out, mod_stats
