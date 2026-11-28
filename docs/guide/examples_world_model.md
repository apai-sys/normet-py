# The world model: JEPA latents, separable de-weathering, station networks

`normet.world_model` carries three models that sit above `normalise`:

| Entry point | What it adds |
|---|---|
| `nm.LatentJEPA` / `nm.fit_latent_jepa` | An action-conditioned latent dynamics model: patches of a target series are encoded, the meteorological patch predicts the next latent, and the innovation (the part the weather does not explain) is the anomaly score. `nm.detect_jepa_events` turns a score threshold into episodes. |
| `nm.SeparableWorldModel` / `nm.fit_world_model` | A de-weathering model whose latent state is split into a slowly varying emission coordinate and a meteorological action, with a monotone quantile-spline GAM mapping meteorology into that coordinate. `normalise` resamples weather in that space; `counterfactual_rollout` runs a trajectory-level what-if. |
| `nm.MultisiteWorldModel` / `nm.fit_multisite_world_model` | Fits one `SeparableWorldModel` per station, then splits the de-weathered panel into a shared regional component and station-local residuals (`decompose`, `local_anomaly_scores`, `regional_what_if`). |
| `nm.jepa_scm` | Synthetic control on JEPA latents: donors are ranked in weather-invariant latent space strictly before the cutoff, so a post-cutoff weather difference cannot pick the donor set. |

All three take a DatetimeIndex-ed frame and named meteorology columns; none of
them trains on the post-intervention period.

```python
import numpy as np
import pandas as pd
import normet as nm
from normet.world_model import detect_jepa_events

rng = np.random.default_rng(7)
n = 24 * 90                                    # 90 days hourly
idx = pd.date_range("2023-01-01", periods=n, freq="h")
t = np.arange(n, dtype=float)
ws = np.clip(3.5 + 1.5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.6, n), 0.3, 12.0)
blh = np.clip(600 + 300 * np.sin(2 * np.pi * (t - 6) / 24) + rng.normal(0, 60, n), 100, 2000)
t2m = 10 + 7 * np.sin(2 * np.pi * t / (24 * 365)) + rng.normal(0, 1.5, n)
dilution = 40 * np.exp(-ws / 3.0) + 25 * (400 / blh)
no2 = np.maximum(1, 60 + 12 * np.sin(2 * np.pi * (t - 8) / 24) + dilution + rng.normal(0, 0.8, n))
pm25 = np.maximum(1, 25 + 6 * np.sin(2 * np.pi * (t - 10) / 24) + 0.6 * dilution + rng.normal(0, 0.6, n))
df = pd.DataFrame({"no2": no2, "pm25": pm25, "ws": ws, "blh": blh, "t2m": t2m}, index=idx)

# 1. Fit and de-weather. Two species are enough for the model to test whether
#    the mixing step commutes with the meteorological warp (warp_then_mix vs
#    mix_then_warp is chosen from the data).
wm = nm.fit_world_model(df, ["no2", "pm25"], ["ws", "blh", "t2m"])
norm = wm.normalise(df, n_samples=25, paired=True, seed=1)
frame = norm.to_normet_frame("no2")            # observed / normalised / factual
print(norm.separability_gain)                   # rho_lb / RC / W2, raw vs latent

# 2. A what-if over the last 14 days: boundary layer 30% higher, 20 bootstrap
#    resamples of the rollout for p10/p90 bands.
roll = wm.counterfactual_rollout(
    df, start=df.index[-24 * 14], end=df.index[-1],
    met_shifts={"blh": ("*", 1.3)}, n_boot=20, seed=2,
)
print(roll.summary_metrics)                     # mean/pct change, peak
print(roll.counterfactual_p10.head())

# 3. Weather-invariant anomaly episodes.
episodes = detect_jepa_events(df, ["no2", "pm25"], ["ws", "blh", "t2m"], threshold=3.0)
print(episodes[["start", "end", "n", "max_score"]])

# 4. A station network with a treated site: JEPA-dewathered synthetic control
#    on the pre-cutoff window only.
sites = ["A", "B", "C", "D"]
rows = []
for i, s in enumerate(sites):
    y = np.maximum(1, 20 + 3 * i + 0.5 * dilution + rng.normal(0, 0.7, n))
    if s == "A":
        y[-24 * 20:] *= 0.75                   # the intervention
    rows.append(pd.DataFrame({"date": idx, "ID": s, "value": y, "ws": ws, "blh": blh, "t2m": t2m}))
panel = pd.concat(rows, ignore_index=True)

scm = nm.jepa_scm(
    panel, met_cols=["ws", "blh", "t2m"],
    treated_unit="A", cutoff_date=idx[-24 * 20],
)
print(scm.att, scm.att_pct, scm.att_hac_se, scm.pre_rmspe)

net = nm.MultisiteWorldModel(n_regional_factors=1)
net.fit(panel, met_cols=["ws", "blh", "t2m"], n_samples=8)
dec = net.decompose()
print(dec.variance_explained.head())            # singular value per regional factor
print(dec.site_loadings)                        # station weights on the plume
print(net.local_anomaly_scores().head())
```

## Reading the numbers

- `normalise()` returns, per species, `*_observed`, `*_normalised` and
  `*_factual`; `to_normet_frame(species)` renames them to the
  `observed` / `normalised` / `factual` columns the rest of `normet` expects.
- `separability_gain` is the diagnostic, not the fit: `mean_rho_lb_latent`
  is the interaction lower bound in latent space and `rho_lb_reduction` its
  improvement over the raw series. A **negative** reduction means the latent
  split did not separate emission from weather better than raw chemistry on
  that record — check it before quoting a trend, exactly as
  `nm.counterfactual_separability_audit` is used for the LightGBM/ICN paths.
- `counterfactual_rollout(...).summary_metrics` reports `mean_actual`,
  `mean_counterfactual`, `pct_change` and the peaks over the window; the
  `p10`/`p90` frames are populated only when `n_boot > 0`. The bands resample
  moving blocks (`block_hours`, default 24 h), so they widen with the
  persistence the model measured.
- `jepa_scm` fits each station's meteorological response **before the
  cutoff only**, then builds the counterfactual from the latent-selected
  donors. `pre_rmspe` is the pre-period fit; a large value means the donor
  pool does not track the treated site and the ATT should not be read.
- `MultisiteWorldModel.decompose()` returns frames in the wide
  station-by-time layout, plus `regional_factors` (the shared plume series),
  `site_loadings`, and `variance_explained` (the singular value of each
  regional factor). With `n_regional_factors=1` the cut between "regional"
  and "local" is a single rank-1 plume; raise it when the network spans more
  than one airshed.

The patch encoder is a choice: `"spectral_lag"` (the default, pure NumPy) or
any object exposing `embed_patches(Y, context_length=...)` — for example a
`nm.ChronosEmbedder` when `normet[foundation]` is installed:

```python
jepa = nm.fit_latent_jepa(
    df, ["no2", "pm25"], ["ws", "blh", "t2m"],
    encoder=nm.ChronosEmbedder(),          # loads amazon/chronos-2 on first use
)
```

The rest of the JEPA code path (scores, episodes, donor selection) is the same
either way.
