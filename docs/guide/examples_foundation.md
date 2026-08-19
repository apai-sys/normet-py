# Zero-shot de-weathering with Chronos-2

`normet.foundation` wraps Amazon's [Chronos-2](https://github.com/amazon-science/chronos-forecasting)
time-series foundation model. Nothing is trained: the checkpoint conditions on
the meteorology through its covariate channel, and the weather is marginalised
out by resampling exactly as in `normalise`.

```bash
pip install "normet[foundation]"
```

## When this is the right tool — and when it is not

Use it when there is not enough history to train on, when a site has just come
online, or when you want a second estimate from a model that never saw the
target's own record. It also gives calibrated quantiles for free: Chronos-2
emits 21 native levels rather than bootstrapping them.

Do not reach for it as a drop-in replacement for the AutoML path. Three
properties decide most cases:

- **The first `context_length` rows are not a result.** The model has no
  history to condition on until then, so `deweather` seeds them with the
  observed values and the two series coincide there *by construction*. On a
  2400-hour record with the default 2048-hour context, 85% of the output is a
  copy of the input. Give it years, not months.
- **Every Monte-Carlo sample is a full forward pass.** The AutoML path happily
  runs 300; here 8 is the default and 300 would be days of CPU. An accelerator
  changes the arithmetic, nothing else does.
- **Position is time.** Chronos-2 counts rows, not timestamps, so a frame with
  dropped hours slides against its own calendar covariates. `to_regular_index`
  rebuilds the grid and leaves the holes as NaN for the model to mask;
  `IrregularIndexError` is raised rather than letting a silently shifted series
  through.

## Where it runs

CPU works everywhere and is the fallback; it is also slow, because de-weathering
spends one full forward pass per Monte-Carlo sample. `device=None` (the default)
picks CUDA if there is a GPU, then Apple Silicon's Metal backend (`mps`), then
the CPU — so a Mac laptop gets the accelerator without being asked. Name one
explicitly to override:

```python
est = Chronos2Estimator(met_covariates=met, device="mps")   # or "cpu", "cuda", "cuda:1"
```

An auto-selected accelerator that fails to load falls back to the CPU with a
warning; a device you named explicitly does not, so you see torch's own error
rather than a silent downgrade. The estimator's `.device` attribute always
reports where it actually landed. The CLI exposes this as `--device` and the
desktop GUI as a **Device** selector next to the backend.

## De-weathering

```python
import pandas as pd
from normet.foundation import Chronos2Estimator, to_regular_index

df = pd.read_csv("my_site.csv", parse_dates=["date"]).set_index("date").sort_index()
df = to_regular_index(df)          # gaps become NaN rows; the model masks them

met = ["t2m", "blh", "u10", "v10"]
est = Chronos2Estimator(met_covariates=met)

out = est.deweather(
    df, "PM2.5",
    met_features=met,
    n_samples=8,
    quantiles=(0.1, 0.5, 0.9),
    schema="normet",
)
# columns: observed, normalised, q100, q500, q900
```

`schema="normet"` renames the result onto the `observed`/`normalised`/`qNNN`
shape `normalise` emits, so it drops straight into the existing plot and report
path:

```python
import normet as nm

nm.normalise_plot(out, ci_low="q100", ci_high="q900")
```

## Check that the model reacts to weather at all

There is no parity plot and no feature importance here — nothing was fitted and
there is no held-out set. The question those plots exist to answer is whether
the output is driven by meteorology or is merely autoregressing, and
`covariate_sensitivity` answers it directly by shuffling the future weather and
measuring how far the forecast moves:

```python
shift = est.covariate_sensitivity(
    df, "PM2.5",
    anchor=df.index[-168], horizon=168,
    met_features=met, random_state=0,
)
shift["pct_of_prediction"]   # e.g. 9.0
```

Read it as a gate, not a metric. Below roughly 1% the model is ignoring the
covariate channel and a "de-weathered" series from it is meaningless — check
the met columns before trusting anything downstream.

## Counterfactuals

`counterfactual` projects a business-as-usual series across an intervention
without ever seeing the post-intervention observations, and reports its own
pre-intervention hold-out bias so a projection that cannot reproduce the months
*before* the intervention says so:

```python
res = est.counterfactual(df, "NO2", intervention_date="2020-03-23", features=met)
res.relative_impact_pct_p50.mean()   # the effect
res.pre_intervention_bias_pct        # the placebo check on the months before it
res.to_normet_frame()                # same observed/normalised/qNNN shape
```

Any effect smaller than `pre_intervention_bias_pct` is not separable from the
counterfactual's own error, so read the two numbers together or not at all.

## From the command line

```bash
normet deweather my_site.csv \
  --target PM2.5 --met-vars t2m,blh,u10,v10 \
  --n-samples 8 --out normalised.csv
```

The command rebuilds the time grid, reports the covariate sensitivity before it
commits to anything, and says how many leading rows were seeded rather than
projected.

## Station embeddings

`ChronosEmbedder` turns each station's series into a 768-D vector from the same
encoder, for clustering sites by dynamics rather than by geography:

```python
import numpy as np

from normet.foundation import ChronosEmbedder

emb = ChronosEmbedder()
vectors = emb.embed_stations({"LN1": s1, "MAN3": s2, "BIR2": s3})
coords, labels = ChronosEmbedder.cluster_embeddings(
    np.vstack(list(vectors.values())), n_clusters=3
)
```

Every series is cut or NaN-padded to the same context length before batching.
That is not cosmetic: Chronos-2 left-pads a ragged batch, which changes the
patch count and shifts a station's pooled embedding depending on which other
stations happened to share its batch.

## Limits

`decompose`, `rolling` and `pdp` have no Chronos-2 equivalent — they are defined
in terms of a fitted model's response surface, and there is no fitted model
here. Use the AutoML backends for those.
