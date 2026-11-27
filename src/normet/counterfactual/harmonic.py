"""Econometric Harmonic & Calendar Counterfactual Inference Engine.

Implements high-precision 168-D Day-of-Week x Hour interaction modeling with
Fourier annual harmonics, vehicle fleet modernization trends, optional
meteorological covariates, time-blocked Ridge cross-validation, and
moving-block bootstrap uncertainty quantification for policy evaluation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..causal._common import newey_west_se_mean

if TYPE_CHECKING:
    from sklearn.linear_model import Ridge

log = logging.getLogger(__name__)

#: Every day-of-week x hour-of-day cell, in a fixed order.
_DOW_HOUR_CATEGORIES: list[str] = [f"dow_{d}_h_{h:02d}" for d in range(7) for h in range(24)]

#: The standard normal's 90th percentile: a two-sided 80 % band, which is
#: what ``counterfactual_p10`` / ``counterfactual_p90`` name.
_Z90 = 1.2815515655446004


def _on_index_clock(value: str | pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """``value`` as a timestamp comparable with ``index``'s clock."""
    ts = pd.Timestamp(value)
    tz = index.tz
    if tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


@dataclass
class HarmonicCounterfactualResult:
    """Container for Harmonic Counterfactual outputs."""

    observed: pd.Series
    counterfactual_bau: pd.Series
    counterfactual_p10: pd.Series
    counterfactual_p90: pd.Series
    absolute_impact: pd.Series
    relative_impact_pct: pd.Series
    validation_bias_pct: float
    coefficients: dict[str, float]
    summary_metrics: dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert counterfactual evaluation into a unified DataFrame."""
        return pd.DataFrame(
            {
                "observed": self.observed,
                "counterfactual_bau": self.counterfactual_bau,
                "counterfactual_p10": self.counterfactual_p10,
                "counterfactual_p90": self.counterfactual_p90,
                "impact": self.absolute_impact,
                "impact_pct": self.relative_impact_pct,
            },
            index=self.observed.index,
        )


class HarmonicCounterfactual:
    """Econometric-grade Harmonic and Calendar Interaction Counterfactual Model.

    Combines:
      - 168-D Diurnal x Day-of-Week Interaction Matrix (D_dow (x) H_hour)
      - Linear Fleet Modernisation Trend (beta_trend * t)
      - Order-``n_harmonics`` Annual and Semi-Annual Fourier Harmonics
      - UK/International Bank Holiday Calendar Adjustments
      - Optional standardised meteorological covariates (and optional squares)
      - Optional time-blocked cross-validated Ridge penalty (``alpha_ridge="auto"``)
      - Parametric or Moving-Block Bootstrap predictive intervals (p10 / p90)
    """

    def __init__(
        self,
        n_harmonics: int = 4,
        alpha_ridge: float | str = 1.0,
        include_trend: bool = True,
        bank_holidays: list[str] | None = None,
        *,
        alpha_grid: Sequence[float] | None = None,
        cv_folds: int = 5,
        standardize_covariates: bool = True,
        quadratic_covariates: bool = False,
    ) -> None:
        if isinstance(alpha_ridge, str):
            if alpha_ridge.lower() != "auto":
                raise ValueError(f"alpha_ridge must be numeric >= 0 or 'auto', got {alpha_ridge!r}")
        elif float(alpha_ridge) < 0.0:
            raise ValueError(f"alpha_ridge must be non-negative, got {alpha_ridge}")
        if int(cv_folds) < 2:
            raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")

        self.n_harmonics = n_harmonics
        self.alpha_ridge: float | str = alpha_ridge
        self.include_trend = include_trend
        self.bank_holidays = set(pd.to_datetime(bank_holidays).date) if bank_holidays else set()
        self.alpha_grid = list(alpha_grid) if alpha_grid is not None else None
        self.cv_folds = int(cv_folds)
        self.standardize_covariates = bool(standardize_covariates)
        self.quadratic_covariates = bool(quadratic_covariates)

        self.model: Ridge | None = None
        self.feature_names: list[str] = []
        self.res_std: float = 1.0
        self.alpha_selected_: float = 1.0 if isinstance(alpha_ridge, str) else float(alpha_ridge)
        self.cv_scores_: dict[float, float] | None = None
        self.validation_bias_pct_: float | None = None
        self.validation_rmse_: float | None = None
        self.covariate_names_: list[str] = []
        self.cov_mean_: pd.Series | None = None
        self.cov_std_: pd.Series | None = None
        self._t0: pd.Timestamp | None = None
        self._train_X: np.ndarray | None = None
        self._train_y: np.ndarray | None = None
        self._train_residuals: np.ndarray | None = None

    def _prepare_covariate_block(
        self,
        times: pd.DatetimeIndex,
        covariates: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align, clean, and (optionally) standardise external covariates."""
        missing = [c for c in self.covariate_names_ if c not in covariates.columns]
        if missing:
            raise ValueError(f"covariates missing required column(s): {missing}")
        block = covariates[self.covariate_names_].reindex(times).copy()
        for col in self.covariate_names_:
            block[col] = pd.to_numeric(block[col], errors="coerce")
        block = block.interpolate().bfill().ffill().fillna(0.0)
        if self.standardize_covariates and self.cov_mean_ is not None and self.cov_std_ is not None:
            block = (block - self.cov_mean_) / self.cov_std_
        out = pd.DataFrame(index=times)
        for col in self.covariate_names_:
            vals = block[col].to_numpy(dtype=np.float64)
            out[f"cov_{col}"] = vals
            if self.quadratic_covariates:
                out[f"cov_sq_{col}"] = vals**2
        return out

    def _build_feature_matrix(
        self,
        times: pd.DatetimeIndex,
        t0_timestamp: pd.Timestamp,
        covariates: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Construct the 168-D interaction + Fourier + Trend (+ covariate) feature matrix."""
        df_feat = pd.DataFrame(index=times)

        # 1. 168-D Day-of-Week x Hour Dummy Matrix
        dow = times.dayofweek
        hour = times.hour
        dow_hour_cat = pd.Categorical(
            [f"dow_{d}_h_{h:02d}" for d, h in zip(dow, hour, strict=True)],
            categories=_DOW_HOUR_CATEGORIES,
        )
        dummies = pd.get_dummies(dow_hour_cat, prefix="dh", drop_first=True, dtype=np.float64)
        dummies.index = times
        df_feat = pd.concat([df_feat, dummies], axis=1)

        # 2. Linear Fleet Modernization Trend
        if self.include_trend:
            hours_since_t0 = (times - t0_timestamp).total_seconds() / 3600.0
            df_feat["linear_trend"] = hours_since_t0 / 8766.0

        # 3. Fourier Annual & Semi-Annual Harmonics
        day_of_year = times.dayofyear.to_numpy(dtype=np.float64) + (hour / 24.0)
        for k in range(1, self.n_harmonics + 1):
            df_feat[f"fourier_sin_{k}"] = np.sin(2.0 * np.pi * k * day_of_year / 365.25)
            df_feat[f"fourier_cos_{k}"] = np.cos(2.0 * np.pi * k * day_of_year / 365.25)

        # 4. Bank Holiday Adjustment
        if self.bank_holidays:
            is_holiday = [t.date() in self.bank_holidays for t in times]
            df_feat["is_bank_holiday"] = np.asarray(is_holiday, dtype=np.float64)

        # 5. External Meteorological / Transport Covariates
        if covariates is not None and self.covariate_names_:
            cov_block = self._prepare_covariate_block(times, covariates)
            df_feat = pd.concat([df_feat, cov_block], axis=1)

        return df_feat.fillna(0.0)

    def _fit_cov_scaler(
        self,
        index: pd.DatetimeIndex,
        covariates: pd.DataFrame,
    ) -> None:
        raw = covariates[self.covariate_names_].reindex(index).copy()
        for col in self.covariate_names_:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw = raw.interpolate().bfill().ffill().fillna(0.0)
        self.cov_mean_ = raw.mean()
        sd = raw.std(ddof=0)
        self.cov_std_ = sd.where(sd > 1e-8, 1.0).fillna(1.0)

    def fit(
        self,
        series: pd.Series,
        validation_split_date: str | pd.Timestamp | None = None,
        *,
        covariates: pd.DataFrame | None = None,
    ) -> HarmonicCounterfactual:
        """Fit harmonic counterfactual on pre-intervention baseline data.

        Parameters
        ----------
        series : pandas.Series
            Time-indexed Series of historical pre-intervention concentrations.
        validation_split_date : str or Timestamp, optional
            Pre-intervention split date. When provided, holds out
            ``index >= validation_split_date`` to measure out-of-sample
            ``validation_bias_pct_`` and ``validation_rmse_``, then refits the
            final model on the full ``series``.
        covariates : pandas.DataFrame, optional
            Time-indexed meteorological or transport covariates aligned with
            ``series`` so weather anomalies are controlled rather than
            attributed to policy impacts.
        """
        index = series.index
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("Input series must have a pd.DatetimeIndex.")

        clean_s = series.interpolate().bfill().ffill()
        t0 = index[0]
        self._t0 = t0

        split_ts = (
            _on_index_clock(validation_split_date, index)
            if validation_split_date is not None
            else None
        )
        if (
            split_ts is not None
            and (index < split_ts).sum() >= 3
            and (index >= split_ts).sum() >= 1
        ):
            stat_mask = np.asarray(index < split_ts, dtype=bool)
        else:
            stat_mask = np.ones(len(index), dtype=bool)

        if covariates is not None:
            if not isinstance(covariates, pd.DataFrame):
                raise ValueError("`covariates` must be a pandas.DataFrame indexed by datetime.")
            self.covariate_names_ = list(covariates.columns)
            self._fit_cov_scaler(index[stat_mask], covariates)
        else:
            self.covariate_names_ = []
            self.cov_mean_ = None
            self.cov_std_ = None

        X_init = self._build_feature_matrix(index, t0_timestamp=t0, covariates=covariates)
        self.feature_names = list(X_init.columns)

        from sklearn.linear_model import Ridge

        y_all = clean_s.to_numpy(dtype=float)
        X_cv_arr = X_init.loc[stat_mask].to_numpy(dtype=float)
        y_cv_arr = y_all[stat_mask]

        if isinstance(self.alpha_ridge, str) and self.alpha_ridge.lower() == "auto":
            grid = (
                np.asarray(self.alpha_grid, dtype=float)
                if self.alpha_grid is not None
                else np.geomspace(1e-2, 1e3, 15)
            )
            n_cv = len(y_cv_arr)
            n_folds = min(self.cv_folds, max(2, n_cv // 24)) if n_cv >= 8 else min(2, n_cv)
            blocks = [
                b
                for b in np.array_split(np.arange(n_cv), max(n_folds, 2))
                if len(b) > 0 and n_cv - len(b) >= 2
            ]
            if len(blocks) >= 2:
                all_pos = np.arange(n_cv)
                scores: dict[float, float] = {}
                for alpha_cand in grid:
                    fold_mse = []
                    for val_idx in blocks:
                        tr_mask = np.ones(n_cv, dtype=bool)
                        tr_mask[val_idx] = False
                        tr_idx = all_pos[tr_mask]
                        mdl_cv = Ridge(alpha=float(alpha_cand), fit_intercept=True)
                        mdl_cv.fit(X_cv_arr[tr_idx], y_cv_arr[tr_idx])
                        preds_cv = mdl_cv.predict(X_cv_arr[val_idx])
                        fold_mse.append(float(np.mean((y_cv_arr[val_idx] - preds_cv) ** 2)))
                    scores[float(alpha_cand)] = float(np.mean(fold_mse))
                self.cv_scores_ = scores
                self.alpha_selected_ = min(scores, key=lambda k: scores[k])
            else:
                self.cv_scores_ = None
                self.alpha_selected_ = 1.0
        else:
            self.cv_scores_ = None
            self.alpha_selected_ = float(self.alpha_ridge)

        if split_ts is not None and not np.all(stat_mask):
            val_mask = ~stat_mask
            mdl_v = Ridge(alpha=self.alpha_selected_, fit_intercept=True)
            mdl_v.fit(X_cv_arr, y_cv_arr)
            X_val_arr = X_init.loc[val_mask].to_numpy(dtype=float)
            val_preds = np.maximum(0.0, mdl_v.predict(X_val_arr))
            val_obs = y_all[val_mask]
            val_obs_m = float(np.mean(val_obs))
            val_pred_m = float(np.mean(val_preds))
            self.validation_bias_pct_ = float(
                ((val_pred_m - val_obs_m) / max(val_obs_m, 1e-3)) * 100.0
            )
            self.validation_rmse_ = float(np.sqrt(np.mean((val_obs - val_preds) ** 2)))
        else:
            self.validation_bias_pct_ = None
            self.validation_rmse_ = None

        # Refit final model on the full input baseline.
        if covariates is not None and self.covariate_names_ and not np.all(stat_mask):
            self._fit_cov_scaler(index, covariates)
            X_full = self._build_feature_matrix(index, t0_timestamp=t0, covariates=covariates)
        else:
            X_full = X_init

        self.feature_names = list(X_full.columns)
        X_arr = X_full.to_numpy(dtype=float)

        self.model = Ridge(alpha=self.alpha_selected_, fit_intercept=True)
        self.model.fit(X_arr, y_all)

        preds = self.model.predict(X_arr)
        residuals = y_all - preds
        self.res_std = float(np.std(residuals))
        self._train_X = X_arr.copy()
        self._train_y = y_all.copy()
        self._train_residuals = residuals.copy()

        log.info(
            "HarmonicCounterfactual fitted on %d timestamps (alpha=%.4g, res_std=%.2f)",
            len(clean_s),
            self.alpha_selected_,
            self.res_std,
        )
        return self

    def predict(
        self,
        future_times: pd.DatetimeIndex,
        t0_timestamp: pd.Timestamp,
        *,
        covariates: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Extrapolate Business-As-Usual (BAU) baseline over future time horizon."""
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if self.covariate_names_ and covariates is None:
            raise ValueError(
                f"Model was fitted with covariates {self.covariate_names_}; pass `covariates` to predict()."
            )

        X_future = self._build_feature_matrix(
            future_times, t0_timestamp=t0_timestamp, covariates=covariates
        )
        X_aligned = X_future.reindex(columns=self.feature_names, fill_value=0.0)

        preds = np.maximum(0.0, self.model.predict(X_aligned.to_numpy(dtype=float)))
        return pd.Series(preds, index=future_times, name="counterfactual_bau")

    def evaluate_intervention(
        self,
        full_series: pd.Series,
        train_end_date: str | pd.Timestamp,
        intervention_start_date: str | pd.Timestamp,
        val_start_date: str | pd.Timestamp | None = None,
        *,
        covariates: pd.DataFrame | None = None,
        uncertainty: str = "parametric",
        n_boot: int = 200,
        block_hours: int = 24,
        random_state: int = 7_654_321,
        residual_carryover: bool = False,
        carryover_min_phi: float = 0.2,
        carryover_window: int = 6,
    ) -> HarmonicCounterfactualResult:
        """Run complete counterfactual policy impact estimation with confidence bounds.

        Parameters
        ----------
        full_series : pandas.Series
            Continuous time-series spanning baseline and intervention.
        train_end_date : str or Timestamp
            End of pure pre-intervention training window.
        intervention_start_date : str or Timestamp
            Start of policy intervention.
        val_start_date : str or Timestamp, optional
            Start of pre-intervention validation window.
        covariates : pandas.DataFrame, optional
            Time-indexed meteorological / transport covariates covering the full
            horizon so meteorological extremes are held out of impact estimates.
        uncertainty : {"parametric", "block_bootstrap"}, default "parametric"
            ``"parametric"`` uses ``±1.2816 * res_std`` around BAU;
            ``"block_bootstrap"`` resamples training blocks of length
            ``block_hours``, refitting Ridge parameters and resampling residuals
            to capture both parameter uncertainty and temporal autocorrelation.
        n_boot : int, default 200
            Number of moving-block bootstrap replicates when
            ``uncertainty="block_bootstrap"``.
        block_hours : int, default 24
            Moving-block length (rows) for block-bootstrap uncertainty.
        random_state : int, default 7_654_321
        """
        unc_mode = (uncertainty or "parametric").lower()
        if unc_mode not in {"parametric", "block_bootstrap"}:
            raise ValueError(
                f"uncertainty must be 'parametric' or 'block_bootstrap', got {uncertainty!r}"
            )

        index = full_series.index
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("full_series must have a pd.DatetimeIndex.")

        train_ts = _on_index_clock(train_end_date, index)
        interv_ts = _on_index_clock(intervention_start_date, index)
        val_ts = _on_index_clock(val_start_date, index) if val_start_date else train_ts

        # 1. Fit on baseline
        train_mask = index <= train_ts
        cov_train = covariates.reindex(index[train_mask]) if covariates is not None else None
        self.fit(full_series[train_mask], covariates=cov_train)

        t0 = index[0]
        # 2. Predict full trajectory
        bau_full = self.predict(index, t0_timestamp=t0, covariates=covariates)
        bau_values = bau_full.to_numpy(dtype=float)

        # 3. Confidence Intervals (p10 / p90 = 80% prediction interval)
        interv_mask = np.asarray(index >= interv_ts, dtype=bool)
        boot_net_stats: dict[str, float] = {}
        if unc_mode == "parametric":
            p10_full = pd.Series(np.maximum(0.0, bau_values - _Z90 * self.res_std), index=index)
            p90_full = pd.Series(bau_values + _Z90 * self.res_std, index=index)
        else:
            assert (
                self._train_X is not None
                and self._train_y is not None
                and self._train_residuals is not None
            )
            X_tr = self._train_X
            y_tr = self._train_y
            r_tr = self._train_residuals
            X_eval = (
                self._build_feature_matrix(index, t0_timestamp=t0, covariates=covariates)
                .reindex(columns=self.feature_names, fill_value=0.0)
                .to_numpy(dtype=float)
            )
            n_tr, n_p = X_tr.shape
            n_tot = len(index)
            B = min(max(1, int(block_hours)), max(1, n_tr // 3))
            n_b_tr = int(np.ceil(n_tr / B))
            n_b_tot = int(np.ceil(n_tot / B))
            rng = np.random.default_rng(int(random_state))
            ridge_reg = max(float(self.alpha_selected_), 1e-8)
            eye_p = np.eye(n_p, dtype=float) * ridge_reg
            n_rep = max(10, int(n_boot))
            boot_draws = np.empty((n_rep, n_tot), dtype=float)
            r_centered = r_tr - float(np.mean(r_tr))

            max_start = max(1, n_tr - B + 1)
            for bi in range(n_rep):
                starts_tr = rng.integers(0, max_start, size=n_b_tr)
                idx_tr = np.concatenate([np.arange(s, s + B) for s in starts_tr])[:n_tr]
                Xb = X_tr[idx_tr]
                yb = y_tr[idx_tr]
                xb_mu = Xb.mean(axis=0)
                yb_mu = float(yb.mean())
                Xbc = Xb - xb_mu
                try:
                    coef_b = np.linalg.solve(Xbc.T @ Xbc + eye_p, Xbc.T @ (yb - yb_mu))
                except np.linalg.LinAlgError:
                    coef_b, *_ = np.linalg.lstsq(
                        Xbc.T @ Xbc + eye_p, Xbc.T @ (yb - yb_mu), rcond=None
                    )
                int_b = yb_mu - float(np.dot(xb_mu, coef_b))
                mu_b = int_b + X_eval @ coef_b

                starts_res = rng.integers(0, max_start, size=n_b_tot)
                idx_res = np.concatenate([np.arange(s, s + B) for s in starts_res])[:n_tot]
                eps_b = r_centered[idx_res]
                boot_draws[bi] = np.maximum(0.0, mu_b + eps_b)

            p10_full = pd.Series(np.quantile(boot_draws, 0.10, axis=0), index=index)
            p90_full = pd.Series(np.quantile(boot_draws, 0.90, axis=0), index=index)

        # Optional persistence-gated AR(1) residual carryover across the intervention boundary
        pre_mask = np.asarray(index < interv_ts, dtype=bool)
        y_pre_vals = full_series.to_numpy(dtype=float)[pre_mask]
        bau_pre_vals = bau_values[pre_mask]
        fin_pre = np.isfinite(y_pre_vals) & np.isfinite(bau_pre_vals)
        r_pre = y_pre_vals[fin_pre] - bau_pre_vals[fin_pre]
        phi1 = 0.0
        if len(r_pre) >= 12:
            a, b = r_pre[:-1], r_pre[1:]
            if float(np.std(a)) > 1e-12 and float(np.std(b)) > 1e-12:
                phi1 = float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 0.99))

        carryover_applied = False
        carryover_r0 = 0.0
        if (
            residual_carryover
            and interv_mask.any()
            and phi1 >= float(carryover_min_phi)
            and len(r_pre) > 0
        ):
            w_len = max(1, min(int(carryover_window), len(r_pre)))
            carryover_r0 = float(np.mean(r_pre[-w_len:]))
            n_post = int(interv_mask.sum())
            steps = np.arange(1, n_post + 1, dtype=float)
            carryover_vec = carryover_r0 * (phi1**steps)
            bau_arr = bau_full.to_numpy(dtype=float).copy()
            p10_arr = p10_full.to_numpy(dtype=float).copy()
            p90_arr = p90_full.to_numpy(dtype=float).copy()
            bau_arr[interv_mask] = np.maximum(0.0, bau_arr[interv_mask] + carryover_vec)
            p10_arr[interv_mask] = np.maximum(0.0, p10_arr[interv_mask] + carryover_vec)
            p90_arr[interv_mask] = np.maximum(0.0, p90_arr[interv_mask] + carryover_vec)
            bau_full = pd.Series(bau_arr, index=index)
            p10_full = pd.Series(p10_arr, index=index)
            p90_full = pd.Series(p90_arr, index=index)
            if unc_mode == "block_bootstrap":
                boot_draws[:, interv_mask] = np.maximum(
                    0.0, boot_draws[:, interv_mask] + carryover_vec[None, :]
                )
            carryover_applied = True

        if unc_mode == "block_bootstrap" and interv_mask.any():
            y_obs_interv = full_series.to_numpy(dtype=float)[interv_mask]
            boot_imp = np.nanmean(y_obs_interv[None, :] - boot_draws[:, interv_mask], axis=1)
            boot_net_stats = {
                "net_impact_se": round(float(np.std(boot_imp, ddof=1)), 3),
                "net_impact_p10": round(float(np.quantile(boot_imp, 0.10)), 2),
                "net_impact_p90": round(float(np.quantile(boot_imp, 0.90)), 2),
            }

        pre_audit_metrics: dict[str, Any] = {}
        if covariates is not None and int(fin_pre.sum()) >= 15:
            cov_pre_df = covariates.reindex(index[pre_mask]).iloc[fin_pre]
            num_cov_cols = [
                c for c in cov_pre_df.columns if pd.api.types.is_numeric_dtype(cov_pre_df[c])
            ]
            if num_cov_cols:
                from ..analysis.diagnostics import counterfactual_separability_audit

                X_pre = (
                    cov_pre_df[num_cov_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                pre_audit = counterfactual_separability_audit(
                    y_pre_vals[fin_pre],
                    bau_pre_vals[fin_pre],
                    X_pre,
                    nonlinear=False,
                    met_names=num_cov_cols,
                    n_boot=0,
                    seed=int(random_state),
                )
                pre_audit_metrics = {
                    "pre_rc_score_linear": round(float(pre_audit["rc_score_linear"]), 4),
                    "pre_rho_lb_raw": round(float(pre_audit["interaction_rho_lb_raw"]), 4),
                }

        # 4. Impacts
        impact_abs = full_series - bau_full
        impact_pct = (impact_abs / np.maximum(bau_full, 1.0)) * 100.0

        # 5. Pre-intervention Validation Bias (between train_ts and interv_ts)
        val_mask = (index >= val_ts) & (index < interv_ts)
        if val_mask.sum() > 0:
            val_obs_mean = float(full_series[val_mask].mean())
            val_bau_mean = float(bau_full[val_mask].mean())
            val_bias_pct = float(((val_bau_mean - val_obs_mean) / max(val_obs_mean, 1e-3)) * 100.0)
        else:
            val_bias_pct = 0.0

        # 6. Intervention Impact Summary
        mean_obs_interv = (
            float(full_series[interv_mask].mean()) if interv_mask.any() else float("nan")
        )
        mean_bau_interv = float(bau_full[interv_mask].mean()) if interv_mask.any() else float("nan")
        net_impact_ugm3 = float(mean_obs_interv - mean_bau_interv)
        net_impact_pct = float((net_impact_ugm3 / max(mean_bau_interv, 1e-3)) * 100.0)
        hac_se, _ = (
            newey_west_se_mean(impact_abs.to_numpy(dtype=float)[interv_mask])
            if int(interv_mask.sum()) >= 2
            else (float("nan"), 0)
        )

        summary: dict[str, Any] = {
            "train_end": str(train_ts),
            "intervention_start": str(interv_ts),
            "alpha_ridge": round(float(self.alpha_selected_), 6),
            "uncertainty_method": unc_mode,
            "validation_bias_pct": round(val_bias_pct, 2),
            "mean_observed_during_intervention": round(mean_obs_interv, 2),
            "mean_counterfactual_bau": round(mean_bau_interv, 2),
            "net_impact_ugm3": round(net_impact_ugm3, 2),
            "net_impact_pct": round(net_impact_pct, 2),
            "net_impact_hac_se": round(float(hac_se), 3) if np.isfinite(hac_se) else float("nan"),
            **boot_net_stats,
            **pre_audit_metrics,
        }
        if residual_carryover:
            summary["residual_carryover_applied"] = bool(carryover_applied)
            summary["pre_residual_phi1"] = round(float(phi1), 4)
            summary["carryover_initial_residual"] = round(float(carryover_r0), 3)

        coef_dict: dict[str, float] = {}
        if self.model is not None:
            for name, coef in zip(self.feature_names, self.model.coef_, strict=True):
                if abs(coef) > 0.01:
                    coef_dict[name] = float(coef)

        return HarmonicCounterfactualResult(
            observed=full_series,
            counterfactual_bau=bau_full,
            counterfactual_p10=p10_full,
            counterfactual_p90=p90_full,
            absolute_impact=impact_abs,
            relative_impact_pct=impact_pct,
            validation_bias_pct=val_bias_pct,
            coefficients=coef_dict,
            summary_metrics=summary,
        )
