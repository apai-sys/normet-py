# src/normet/analysis/uncertainty.py
"""Uncertainty decomposition: how much of the total spread is weather or model.

De-weathering hands the analyst a deweathered estimate with a Monte-Carlo
spread, and an ensemble of models (``do_all_unc``) a wider, model-form
spread. The question neither of those numbers answers is the one a policy
evaluation must settle: *of the total band around a policy's estimated
effect, how much is the met resampling and how much is the model-form
uncertainty?* If a ULEZ effect is smaller than the band, whether it is
"significant" depends on which error dominates, and the honest answer comes
from attributing the variance to its source.

:func:`decompose_uncertainty` attributes the total per-date variance to its
source by an orthogonal (two-way) decomposition of the full model-by-weather
prediction matrix, exploiting the package's common-random-number discipline
rather than adding a new source of randomness:

- Every ensemble member is given the *same* ``n_samples`` met resamples.
  The ``normalise`` draws are seeded once -- the user's ``seed`` -- and are
  therefore identical across models (``_draw_rows`` is a pure function of
  the seed), which is the common-random-number anchor the package already
  relies on for ``decompose``'s differences to be comparable. The models
  differ only by *form*: member ``i`` is trained on its own seed.
- For one timestamp the response is a matrix ``Z`` of shape ``n_models x
  n_samples``: the prediction of member ``i`` on met draw ``j``. Its total
  variance (over both factors, mean-centred) splits orthogonally into the
  *between-model* main effect (the model-form variance), the *between-draw*
  main effect (the weather variance), and the interaction (what is not
  additive -- a model that reacts differently to a met draw than the
  ensemble does). The total is exactly the three summands, with every term
  non-negative. The identity is pinned by a test, so a future change to the
  decomposition cannot quietly break the additivity the attribution rests on.

The per-date result is one row of ``var_weather`` / ``var_model`` /
``var_interaction`` / ``var_total`` and their ``pct_*`` shares; the
:class:`UncDecomposition` result carries the median share across the record
in ``summary()`` and a stacked-variance-band figure in ``plot()``. The cost
is one prediction per ``(model, draw, timestamp)`` -- the same total count
as one ``do_all_unc`` run -- so the attribution costs no extra fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .. import style as S
from ..exceptions import ConfigError
from ..model.train import train_model
from ..style import styled
from ..utils._config import DEFAULT_SEED
from ..utils.logging import get_logger
from ..utils.prepare import prepare_data
from .normalise import normalise

log = get_logger(__name__)

__all__ = ["UncDecomposition", "decompose_uncertainty"]


@dataclass
class UncDecomposition:
    """The per-date attribution of a run's total variance to its sources.

    ``var_weather``, ``var_model`` and ``var_interaction`` are the per-date (in
    the target's units squared) two-way main effects and their interaction;
    ``var_total`` is their sum exactly. ``pct_*`` are the shares of ``var_total``.
    ``summary()`` reports the median share across the record; ``plot()`` draws
    the stacked bands over time. ``notes`` carry the caveats a report must
    state (the interaction is a residual, the shares are medians).
    """

    var_weather: np.ndarray
    var_model: np.ndarray
    var_interaction: np.ndarray
    var_total: np.ndarray
    index: pd.Index
    target: str
    n_models: int
    n_weather_draws: int
    seed: int
    notes: list[str] = field(default_factory=list)

    @property
    def pct_weather(self) -> np.ndarray:
        return self._pct(self.var_weather)

    @property
    def pct_model(self) -> np.ndarray:
        return self._pct(self.var_model)

    @property
    def pct_interaction(self) -> np.ndarray:
        return self._pct(self.var_interaction)

    def _pct(self, part: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(
                self.var_total > 0, part / np.where(self.var_total > 0, self.var_total, 1.0), np.nan
            )
        return p

    def frame(self) -> pd.DataFrame:
        """The per-date rows as a DataFrame, report-ready and directly plottable."""
        df = pd.DataFrame(
            {
                "var_weather": self.var_weather,
                "var_model": self.var_model,
                "var_interaction": self.var_interaction,
                "var_total": self.var_total,
                "pct_weather": self.pct_weather,
                "pct_model": self.pct_model,
                "pct_interaction": self.pct_interaction,
            },
            index=self.index,
        )
        df.index.name = "date"
        return df

    def summary(self) -> pd.DataFrame:
        """The per-source median variance and its median share of the total.

        The shares are the headline of a policy evaluation ("of the band, this
        fraction is weather, that one is model form"), so they are medians
        across the record rather than means: a single extreme day does not
        re-weight the attribution. ``interaction`` is the residual the other
        two do not explain, reported like the others so its size is visible.
        """

        def med(a: np.ndarray) -> float:
            return float(np.nanmedian(a))

        rows = [
            ("weather", med(self.var_weather), med(self.pct_weather)),
            ("model", med(self.var_model), med(self.pct_model)),
            ("interaction", med(self.var_interaction), med(self.pct_interaction)),
            ("total", med(self.var_total), 1.0),
        ]
        return pd.DataFrame(rows, columns=["source", "var_median", "pct_median"])

    @styled
    def plot(
        self,
        *,
        title: str | None = None,
        ylabel: str = "Variance (target units**2)",
        ax: Any | None = None,
    ) -> Any:
        """The stacked per-date variance bands (weather, model, interaction).

        The three sources stack to ``var_total`` on every date, so the height
        of the stack is the total spread the model and the met resampling
        together carry, and each band's height is that source's share. The
        figure is the policy-evaluation deliverable: which band dominates the
        total, date by date and in the median.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3), layout="constrained")
        idx = self.index
        Y = np.vstack([self.var_weather, self.var_model, self.var_interaction])
        labels = ["weather", "model", "interaction"]
        colours = [S.PALETTE[0], S.PALETTE[1], S.PALETTE[3]]
        ax.stackplot(idx, Y, labels=labels, colors=colours, alpha=0.85, lw=0)
        S.line(ax, idx, self.var_total, color=S.INK, emphasis=False, lw=0.8, label="total")
        ax.set_title(title or "Uncertainty: where the spread comes from")
        ax.set_ylabel(ylabel)
        if isinstance(idx, pd.DatetimeIndex):
            S.format_time_axis(ax)
        ax.margins(x=0)
        S.grid(ax)
        S.legend(ax, loc="upper right", ncols=2)
        return ax

    def __str__(self) -> str:
        n = len(self.index)
        med_w = float(np.nanmedian(self.pct_weather))
        med_m = float(np.nanmedian(self.pct_model))
        med_i = float(np.nanmedian(self.pct_interaction))
        return (
            f"{self.target}: {n} dates, {self.n_models} models x "
            f"{self.n_weather_draws} met draws | median spread share: "
            f"weather {100.0 * med_w:.0f}%, model {100.0 * med_m:.0f}%, "
            f"interaction {100.0 * med_i:.0f}%"
        )


def _met_matrix(
    df: pd.DataFrame,
    model: object,
    covariates: list[str],
    variables_resample: list[str] | None,
    resample_df: pd.DataFrame | None,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """One timestamp-independent set of ``n_samples`` met draws for the record.

    Runs :func:`normalise` with ``aggregate=False`` anchored on ``seed``: the
    per-draw columns are the model's predictions on each met resample of the
    same record. Every ensemble member is handed this *same* draw (same seed),
    and the same resample pool, which is what makes the per-draw variance
    across members a common-random number rather than an extra, uncorrelated
    Monte-Carlo draw.
    """
    res = normalise(
        df,
        model,
        covariates=covariates,
        variables_resample=variables_resample,
        resample_df=resample_df,
        n_samples=n_samples,
        aggregate=False,
        seed=seed,
        n_cores=1,
        verbose=False,
    )
    draw_cols = [c for c in res.columns if c != "observed"]
    return res[draw_cols].to_numpy(dtype=float).T  # (n_samples, n_dates)


def _decompose(
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The two-way orthogonal split of the model-by-draw matrix, all timestamps at once.

    ``z`` is ``(n_models, n_samples, n_dates)``: the prediction of member ``i``
    on met draw ``j`` at each timestamp. At one timestamp the mean is removed
    once, then the variance is partitioned by the two factors: the model mean
    (over draws) and the draw mean (over models) are the two main effects, and
    what the two do not account for jointly is the interaction. The partition
    is orthogonal, so the total variance is exactly the three summands and
    each is a sum of squares (hence non-negative). See the module docstring.
    """
    grand = z.mean(axis=(0, 1))  # (n_dates,)
    per_model = z.mean(axis=1)  # (I, n_dates)
    per_draw = z.mean(axis=0)  # (J, n_dates)
    var_model = np.mean((per_model - grand[None, :]) ** 2, axis=0)
    var_weather = np.mean((per_draw - grand[None, :]) ** 2, axis=0)
    interaction = z - (per_model[:, None, :] + per_draw[None, :, :] - grand[None, None, :])
    var_interaction = np.mean(interaction**2, axis=(0, 1))
    var_total = np.mean((z - grand[None, None, :]) ** 2, axis=(0, 1))
    return var_weather, var_model, var_interaction, var_total


def decompose_uncertainty(
    df: pd.DataFrame,
    target: str,
    *,
    backend: str = "lightgbm",
    covariates: list[str] | None = None,
    variables_resample: list[str] | None = None,
    split_method: str = "ts",
    train_fraction: float = 0.75,
    model_config: dict[str, Any] | None = None,
    n_samples: int = 300,
    n_models: int = 10,
    seed: int = DEFAULT_SEED,
    n_cores: int | None = None,
    resample_df: pd.DataFrame | None = None,
    verbose: bool = False,
) -> UncDecomposition:
    """Attribute a de-weathered estimate's total spread to weather and model form.

    Trains ``n_models`` members (each on its own seed, so they differ in form)
    and, for each, predicts on the *same* ``n_samples`` met resamples shared
    across the whole ensemble. The per-date variance of that
    ``(n_models, n_samples)`` response is split orthogonally into the
    weather, model and interaction terms; see the module docstring for the
    construction and why the three summands sum to the total exactly.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with a ``date`` column (datetime) and the target.
    target : str
        The target column name.
    backend : str, default "lightgbm"
        The AutoML backend the ensemble members are trained with. "lightgbm"
        is the default because ``n_models`` fits must each be fast; pass
        "flaml" for AutoML-searched members at higher cost.
    covariates : list of str, optional
        The model's feature set. Required (or passed via the model's own
        trained features when ``None``); it must be a superset of the model's
        actual features, as :func:`normet.normalise` enforces.
    variables_resample : list of str, optional
        Which covariates the met resampling draws. Defaults to all the
        non-time-feature covariates, as :func:`normalise` does.
    split_method : str, default "ts"
        Train/test split. ``"ts"`` (the default) so the training sets of the
        members are the same rows under different seeds -- only the model
        form varies, which is the quantity the decomposition isolates.
    train_fraction : float, default 0.75
    model_config : dict, optional
        Backend-specific; with the lightgbm default, keep it small (the
        ensemble is ``n_models`` fits). See the backend's docstring.
    n_samples : int, default 300
        Met resamples per member -- the weather-draw factor.
    n_models : int, default 10
        Ensemble members -- the model-form factor.
    seed : int, default ``DEFAULT_SEED``
        The *weather* seed: every member shares it, so the met draws are the
        common-random-number anchor and are reproducible. Member seeds are
        derived from it (``np.random.default_rng([seed, 1])``), so the whole
        decomposition is reproducible under ``seed``.
    n_cores, resample_df, verbose :
        Forwarded to :func:`normalise` / :func:`train_model`.

    Returns
    -------
    UncDecomposition
        The per-date attribution; see :class:`UncDecomposition` for
        ``frame()``, ``summary()`` and ``plot()``.

    Raises
    ------
    ConfigError
        If ``covariates`` is missing or empty, or ``n_models``/``n_samples`` < 2
        (a factor with one level carries no variance to attribute).
    ValueError
        If ``df`` has no usable ``date``/target, as :func:`prepare_data` reports.
    """
    if n_samples < 2:
        raise ConfigError(
            "`n_samples` must be at least 2: the weather factor needs a draw to vary."
        )
    if n_models < 2:
        raise ConfigError(
            "`n_models` must be at least 2: with one model there is no model-form "
            "variance to attribute (the spread is then all weather)."
        )
    if covariates is None or not covariates:
        raise ConfigError("`covariates` must be provided and non-empty.")
    covariates = list(covariates)

    df_prep = prepare_data(
        df=df,
        target=target,
        covariates=covariates,
        split_method=split_method,
        train_fraction=train_fraction,
        seed=seed,
    )
    member_rng = np.random.default_rng([seed, 1])
    member_seeds = [int(s) for s in member_rng.choice(1_000_000, size=n_models, replace=False)]
    log.info(
        "decompose_uncertainty: %d members x %d met draws on %d timestamps (seed=%d)",
        n_models,
        n_samples,
        len(df_prep),
        seed,
    )

    Z = np.empty((n_models, n_samples, len(df_prep)), dtype=float)
    for i, ms in enumerate(member_seeds):
        model_i = train_model(
            df=df_prep,
            target="value",
            backend=backend,
            covariates=covariates,
            model_config=model_config,
            seed=ms,
            n_cores=n_cores,
            verbose=verbose and i == 0,
        )
        # All members see the same met draws: anchor the resampling on the
        # user's seed, not the member seed, so draw j is identical across i.
        Z[i] = _met_matrix(
            df_prep,
            model_i,
            covariates,
            variables_resample,
            resample_df,
            n_samples,
            seed,
        )
        (log.info if verbose else log.debug)("member %d/%d done", i + 1, n_models)

    var_weather, var_model, var_interaction, var_total = _decompose(Z)

    notes = [
        "Each source's variance is a two-way main effect; 'interaction' is what the "
        "models do not share about a met draw (a residual, not an independent draw).",
        "Shares are medians across the record; a single extreme day cannot re-weight them.",
        "All members share the same n_samples met resamples (common random numbers), "
        "so the total per-date variance equals the three summands exactly, to floating "
        "point.",
    ]
    return UncDecomposition(
        var_weather=var_weather,
        var_model=var_model,
        var_interaction=var_interaction,
        var_total=var_total,
        index=pd.DatetimeIndex(df_prep["date"]),
        target=target,
        n_models=n_models,
        n_weather_draws=n_samples,
        seed=seed,
        notes=notes,
    )
