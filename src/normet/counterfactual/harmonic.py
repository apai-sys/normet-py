"""Econometric Harmonic & Calendar Counterfactual Inference Engine.

Implements high-precision 168-D Day-of-Week x Hour interaction modeling with
Fourier annual harmonics and vehicle fleet modernization trends for rigorous
policy evaluation (e.g. COVID-19 lockdowns, Low Emission Zones).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

log = logging.getLogger(__name__)

#: Every day-of-week x hour-of-day cell, in a fixed order.
#:
#: ``pd.get_dummies(..., drop_first=True)`` drops whichever category comes first
#: *in the data it was handed*, so deriving the categories from each window made
#: the reference level depend on where the window happened to start. A frame
#: built for a short horizon then dropped a different column than ``fit`` had,
#: and ``predict``'s ``reindex`` silently filled it with zeros -- putting every
#: row of that cell on the reference level. Pinning the categories here keeps the
#: design matrix identical across windows; cells absent from a window simply come
#: out all-zero. ``test_predict_is_invariant_to_horizon_length`` pins the fix.
_DOW_HOUR_CATEGORIES: list[str] = [f"dow_{d}_h_{h:02d}" for d in range(7) for h in range(24)]


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
      - 4th-order Annual and Semi-Annual Fourier Harmonics
      - UK/International Bank Holiday Calendar Adjustments
      - Parametric Residual Quantiles (p10, p50, p90)
    """

    def __init__(
        self,
        n_harmonics: int = 4,
        alpha_ridge: float = 1.0,
        include_trend: bool = True,
        bank_holidays: list[str] | None = None,
    ) -> None:
        self.n_harmonics = n_harmonics
        self.alpha_ridge = alpha_ridge
        self.include_trend = include_trend
        self.bank_holidays = set(pd.to_datetime(bank_holidays).date) if bank_holidays else set()
        self.model: Ridge | None = None
        self.feature_names: list[str] = []
        self.res_std: float = 1.0

    def _build_feature_matrix(
        self, times: pd.DatetimeIndex, t0_timestamp: pd.Timestamp
    ) -> pd.DataFrame:
        """Construct the 168-D interaction + Fourier + Trend feature matrix."""
        df_feat = pd.DataFrame(index=times)

        # 1. 168-D Day-of-Week x Hour Dummy Matrix (eliminates leap year and weekend shift errors)
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
            df_feat["linear_trend"] = hours_since_t0 / 8766.0  # normalized per year

        # 3. Fourier Annual & Semi-Annual Harmonics
        day_of_year = times.dayofyear.to_numpy(dtype=np.float64) + (hour / 24.0)
        for k in range(1, self.n_harmonics + 1):
            df_feat[f"fourier_sin_{k}"] = np.sin(2.0 * np.pi * k * day_of_year / 365.25)
            df_feat[f"fourier_cos_{k}"] = np.cos(2.0 * np.pi * k * day_of_year / 365.25)

        # 4. Bank Holiday Adjustment
        if self.bank_holidays:
            is_holiday = [t.date() in self.bank_holidays for t in times]
            df_feat["is_bank_holiday"] = np.asarray(is_holiday, dtype=np.float64)

        return df_feat.fillna(0.0)

    def fit(
        self,
        series: pd.Series,
        validation_split_date: str | pd.Timestamp | None = None,
    ) -> HarmonicCounterfactual:
        """Fit harmonic counterfactual on pre-intervention baseline data.

        Args:
            series: Time-indexed Series of historical pre-intervention concentrations.
            validation_split_date: Optional cutoff for pre-intervention validation bias testing.
        """
        index = series.index
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("Input series must have a pd.DatetimeIndex.")

        clean_s = series.interpolate().bfill().ffill()
        t0 = index[0]
        X = self._build_feature_matrix(index, t0_timestamp=t0)
        self.feature_names = list(X.columns)

        y = clean_s.values
        self.model = Ridge(alpha=self.alpha_ridge, fit_intercept=True)
        self.model.fit(X.values, y)

        preds = self.model.predict(X.values)
        residuals = y - preds
        self.res_std = float(np.std(residuals))

        log.info(
            "HarmonicCounterfactual fitted on %d timestamps (Residual std: %.2f)",
            len(clean_s),
            self.res_std,
        )
        return self

    def predict(
        self,
        future_times: pd.DatetimeIndex,
        t0_timestamp: pd.Timestamp,
    ) -> pd.Series:
        """Extrapolate Business-As-Usual (BAU) baseline over future time horizon."""
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_future = self._build_feature_matrix(future_times, t0_timestamp=t0_timestamp)
        # Ensure exact columns match
        X_aligned = X_future.reindex(columns=self.feature_names, fill_value=0.0)

        preds = np.maximum(0.0, self.model.predict(X_aligned.values))
        return pd.Series(preds, index=future_times, name="counterfactual_bau")

    def evaluate_intervention(
        self,
        full_series: pd.Series,
        train_end_date: str | pd.Timestamp,
        intervention_start_date: str | pd.Timestamp,
        val_start_date: str | pd.Timestamp | None = None,
    ) -> HarmonicCounterfactualResult:
        """Run complete counterfactual policy impact estimation with confidence bounds.

        Args:
            full_series: Continuous hourly time-series spanning baseline and intervention.
            train_end_date: End of pure pre-intervention training window (e.g. '2019-12-31').
            intervention_start_date: Start of policy intervention (e.g. '2020-03-23').
            val_start_date: Start of pre-intervention validation window (e.g. '2020-01-01').
        """
        index = full_series.index
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("full_series must have a pd.DatetimeIndex.")

        train_ts = pd.to_datetime(train_end_date)
        interv_ts = pd.to_datetime(intervention_start_date)
        val_ts = pd.to_datetime(val_start_date) if val_start_date else train_ts

        # 1. Fit on baseline
        train_mask = index <= train_ts
        self.fit(full_series[train_mask])

        t0 = index[0]
        # 2. Predict full trajectory
        bau_full = self.predict(index, t0_timestamp=t0)

        # 3. Calculate Confidence Intervals
        bau_values = bau_full.to_numpy(dtype=float)
        p10_full = pd.Series(np.maximum(0.0, bau_values - 1.645 * self.res_std), index=index)
        p90_full = pd.Series(bau_values + 1.645 * self.res_std, index=index)

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
        interv_mask = index >= interv_ts
        mean_obs_interv = float(full_series[interv_mask].mean())
        mean_bau_interv = float(bau_full[interv_mask].mean())
        net_impact_ugm3 = float(mean_obs_interv - mean_bau_interv)
        net_impact_pct = float((net_impact_ugm3 / max(mean_bau_interv, 1e-3)) * 100.0)

        summary = {
            "train_end": str(train_ts),
            "intervention_start": str(interv_ts),
            "validation_bias_pct": round(val_bias_pct, 2),
            "mean_observed_during_intervention": round(mean_obs_interv, 2),
            "mean_counterfactual_bau": round(mean_bau_interv, 2),
            "net_impact_ugm3": round(net_impact_ugm3, 2),
            "net_impact_pct": round(net_impact_pct, 2),
        }

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
