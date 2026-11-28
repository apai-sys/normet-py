"""Hierarchical multi-site World Model decomposing regional plumes and local sources.

Fits site-specific :class:`SeparableWorldModel` encoders against local weather at
every monitoring station, then factorizes the aligned deweathered anomaly network
into shared low-rank regional transport factors and station-specific local
emission dynamics for regional plume abatement What-Ifs and false-alarm-free
local leak detection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .latent_wm import SeparableWorldModel, WorldModelConfig

__all__ = [
    "MultisiteDecompositionResult",
    "MultisiteWorldModel",
    "fit_multisite_world_model",
]


@dataclass
class MultisiteDecompositionResult:
    """Outputs of hierarchical regional + local decomposition across a station network."""

    observed: pd.DataFrame
    deweathered: pd.DataFrame
    regional_component: pd.DataFrame
    local_component: pd.DataFrame
    regional_factors: pd.DataFrame
    site_loadings: pd.DataFrame
    variance_explained: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


class MultisiteWorldModel:
    """Hierarchical multi-station atmospheric World Model."""

    def __init__(
        self,
        *,
        n_regional_factors: int = 1,
        config: WorldModelConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_regional_factors = max(1, int(n_regional_factors))
        self.config = config if config is not None else WorldModelConfig(**kwargs)
        self._sites: list[str] = []
        self._site_models: dict[str, SeparableWorldModel] = {}
        self._decomp: MultisiteDecompositionResult | None = None
        self._local_scales: dict[str, float] = {}
        self._local_medians: dict[str, float] = {}

    def _require_fitted(self) -> MultisiteDecompositionResult:
        if self._decomp is None:
            raise RuntimeError("MultisiteWorldModel must be fitted via .fit() first.")
        return self._decomp

    def fit(
        self,
        df: pd.DataFrame,
        *,
        date_col: str = "date",
        unit_col: str = "ID",
        outcome_col: str = "value",
        met_cols: Sequence[str],
        n_samples: int = 60,
    ) -> MultisiteWorldModel:
        """Fit per-site world models and extract shared regional transport factors."""
        for col in (date_col, unit_col, outcome_col):
            if col not in df.columns:
                raise KeyError(f"Column {col!r} not found in DataFrame.")

        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col])
        sites = [str(u) for u in work[unit_col].unique()]
        if len(sites) < 2:
            raise ValueError("MultisiteWorldModel requires at least 2 stations.")

        mcols = [c for c in met_cols if c in work.columns]
        obs_map: dict[str, pd.Series] = {}
        dew_map: dict[str, pd.Series] = {}
        models: dict[str, SeparableWorldModel] = {}

        common_idx: pd.DatetimeIndex | None = None
        for s in sites:
            sub = (
                work.loc[work[unit_col] == s]
                .sort_values(date_col)
                .drop_duplicates(subset=[date_col])
                .set_index(date_col)
            )
            if len(sub) < 20:
                continue
            wm = SeparableWorldModel(self.config).fit(sub, outcome_col, mcols)
            norm = wm.normalise(sub, n_samples=n_samples, paired=True, seed=self.config.seed)
            obs_ser = pd.Series(
                pd.to_numeric(sub[outcome_col], errors="coerce").to_numpy(dtype=float),
                index=pd.DatetimeIndex(sub.index),
                name=s,
            )
            dew_ser = pd.Series(
                norm.df["normalised"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(norm.df.index),
                name=s,
            )
            obs_map[s] = obs_ser
            dew_map[s] = dew_ser
            models[s] = wm
            cur_dti = pd.DatetimeIndex(obs_ser.index)
            common_idx = (
                cur_dti
                if common_idx is None
                else pd.DatetimeIndex(common_idx.intersection(cur_dti))
            )

        valid_sites = list(models.keys())
        if len(valid_sites) < 2 or common_idx is None or len(common_idx) < 20:
            raise ValueError("Insufficient overlapping stations/timestamps for multisite fit.")

        common_idx = common_idx.sort_values()
        obs_df = pd.DataFrame(
            {s: obs_map[s].reindex(common_idx) for s in valid_sites}, index=common_idx
        )
        dew_df = pd.DataFrame(
            {s: dew_map[s].reindex(common_idx) for s in valid_sites}, index=common_idx
        )

        D = dew_df.to_numpy(dtype=float)
        fin_rows = np.all(np.isfinite(D), axis=1)
        D_clean = D[fin_rows]
        if len(D_clean) < 10:
            raise ValueError("Fewer than 10 finite aligned rows across sites.")

        site_means = np.mean(D_clean, axis=0, keepdims=True)
        D_anom = np.where(fin_rows[:, None], D - site_means, 0.0)
        D_anom_clean = D_clean - site_means

        k = min(self.n_regional_factors, len(valid_sites) - 1, len(D_anom_clean) - 1)
        U, s_vals, Vt = np.linalg.svd(D_anom_clean, full_matrices=False)
        tot_var = float(np.sum(s_vals**2))
        ev_rows = []
        for idx_f in range(k):
            v_ratio = float(s_vals[idx_f] ** 2 / tot_var) if tot_var > 1e-12 else 0.0
            ev_rows.append(
                {
                    "factor": f"regional_{idx_f + 1}",
                    "singular_value": float(s_vals[idx_f]),
                    "variance_ratio": v_ratio,
                }
            )

        # Robust Huber-IRLS factor projection prevents single-site emission spikes
        # from bleeding onto clean background stations.
        V_k = Vt[:k, :]  # (K, N_sites)
        F_all = D_anom @ V_k.T  # (T, K) warm start
        eye_k = 1e-6 * np.eye(k, dtype=float)
        for _ in range(6):
            E_mat = D_anom - F_all @ V_k
            med_e = np.median(E_mat, axis=1, keepdims=True)
            mad_e = np.median(np.abs(E_mat - med_e), axis=1, keepdims=True)
            scale_t = np.maximum(1.4826 * mad_e, 1e-3)
            abs_e = np.abs(E_mat)
            huber_w = np.where(
                abs_e <= 1.345 * scale_t, 1.0, (1.345 * scale_t) / np.maximum(abs_e, 1e-8)
            )
            for t in range(len(D_anom)):
                wt = huber_w[t]
                Vw = V_k * wt[None, :]
                gram = Vw @ V_k.T + eye_k
                rhs = Vw @ D_anom[t]
                F_all[t] = np.linalg.solve(gram, rhs)
        R_comp = F_all @ V_k  # (T, N_sites)
        L_comp = D_anom - R_comp

        factor_cols = [f"regional_{i + 1}" for i in range(k)]
        reg_factors_df = pd.DataFrame(F_all, index=common_idx, columns=factor_cols)
        loadings_df = pd.DataFrame(V_k.T, index=valid_sites, columns=factor_cols)
        reg_comp_df = pd.DataFrame(R_comp, index=common_idx, columns=valid_sites)
        loc_comp_df = pd.DataFrame(L_comp, index=common_idx, columns=valid_sites)
        ev_df = pd.DataFrame(ev_rows)

        scales: dict[str, float] = {}
        medians: dict[str, float] = {}
        for j, s in enumerate(valid_sites):
            col_l = L_comp[fin_rows, j]
            med = float(np.median(col_l))
            mad = float(np.median(np.abs(col_l - med)))
            scales[s] = max(1.4826 * mad, float(np.std(col_l)), 1e-6)
            medians[s] = med

        self._sites = valid_sites
        self._site_models = models
        self._local_scales = scales
        self._local_medians = medians
        self._decomp = MultisiteDecompositionResult(
            observed=obs_df,
            deweathered=dew_df,
            regional_component=reg_comp_df,
            local_component=loc_comp_df,
            regional_factors=reg_factors_df,
            site_loadings=loadings_df,
            variance_explained=ev_df,
            diagnostics={
                "n_sites": len(valid_sites),
                "n_timestamps": len(common_idx),
                "n_regional_factors": int(k),
                "total_regional_variance_ratio": float(ev_df["variance_ratio"].sum()),
            },
        )
        return self

    def decompose(self) -> MultisiteDecompositionResult:
        """Return regional and local network decomposition."""
        return self._require_fitted()

    def local_anomaly_scores(self) -> pd.DataFrame:
        """Return standardized z-scores of site-local residuals (shielded from weather & regional plumes)."""
        dec = self._require_fitted()
        z_cols: dict[str, np.ndarray] = {}
        for s in self._sites:
            arr = dec.local_component[s].to_numpy(dtype=float)
            z_cols[s] = np.abs(arr - self._local_medians[s]) / self._local_scales[s]
        return pd.DataFrame(z_cols, index=dec.local_component.index)

    def regional_what_if(
        self,
        site: str,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        regional_scale: float = 0.0,
    ) -> pd.DataFrame:
        """Reconstruct a station's trajectory under a scaled/ablated regional transport plume."""
        dec = self._require_fitted()
        if site not in self._sites:
            raise KeyError(f"Site {site!r} not in fitted network {self._sites}.")
        idx = pd.DatetimeIndex(dec.observed.index)

        def _align(val: str | pd.Timestamp) -> pd.Timestamp:
            ts = pd.Timestamp(val)
            if idx.tz is None:
                return ts.tz_localize(None) if ts.tzinfo is not None else ts
            return ts.tz_localize(idx.tz) if ts.tzinfo is None else ts.tz_convert(idx.tz)

        mask = np.ones(len(idx), dtype=bool)
        if start is not None:
            mask &= np.asarray(idx >= _align(start), dtype=bool)
        if end is not None:
            mask &= np.asarray(idx <= _align(end), dtype=bool)
        if not np.any(mask):
            raise ValueError("No timestamps inside requested regional_what_if window.")

        act = dec.observed.loc[mask, site].to_numpy(dtype=float)
        reg = dec.regional_component.loc[mask, site].to_numpy(dtype=float)
        removed = (1.0 - float(regional_scale)) * reg
        cf = np.maximum(0.0, act - removed)
        return pd.DataFrame(
            {
                "actual": act,
                "counterfactual": cf,
                "difference": act - cf,
                "regional_component": reg,
                "local_component": dec.local_component.loc[mask, site].to_numpy(dtype=float),
            },
            index=idx[mask],
        )


def fit_multisite_world_model(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    unit_col: str = "ID",
    outcome_col: str = "value",
    met_cols: Sequence[str],
    n_regional_factors: int = 1,
    config: WorldModelConfig | None = None,
    **kwargs: Any,
) -> MultisiteWorldModel:
    """Convenience constructor fitting :class:`MultisiteWorldModel`."""
    return MultisiteWorldModel(n_regional_factors=n_regional_factors, config=config, **kwargs).fit(
        df, date_col=date_col, unit_col=unit_col, outcome_col=outcome_col, met_cols=met_cols
    )
