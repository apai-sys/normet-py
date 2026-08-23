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

The Monte-Carlo draws go to the model in batches of `batch_size` (default 32)
rather than one at a time. Measured on an L40S at a 512 h context and a 48 h
horizon, that is worth about 3x from eight draws upward; on a single-threaded
CPU it is worth nothing, because the samples are compute-bound rather than
dispatch-bound. The numbers are the same either way — `predict` is
deterministic, and batching only reorders float32 accumulation (~1e-5). Lower
`batch_size` if a long context and many covariates exhaust the device's memory.

Because the default is fixed rather than device-dependent, `n_samples` is worth
raising by hand on a GPU: the same L40S runs 128 batched draws in 0.55 s against
0.10 s for eight, so more of the Monte-Carlo error is cheap to buy. It is left
to you rather than chosen automatically so that the same script and seed give
the same answer on a laptop and on a cluster.

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

## The one-call pipeline

`do_all` takes the same backend switch, and keeps its three-tuple shape:

```python
out, estimator, df_prep = nm.do_all(
    df, target="PM2.5", backend="chronos-2",
    covariates=["t2m", "blh", "u10", "v10"],
    variables_resample=["t2m", "blh", "u10", "v10"],
)
```

Nothing is trained, so the model slot holds the loaded estimator rather than a
fitted model -- which is what you need for further projections or the
diagnostic below. `split_method` and `train_fraction` have nothing to act on;
`model_config` is forwarded to the estimator's constructor instead, which is
where `device`, `context_length` and `prediction_length` belong.

`n_samples` defaults to 8 here rather than the AutoML path's 300, since each
sample is a full forward pass. An explicit value always wins.

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
encoder, describing how a site behaves rather than where it is:

```python
from normet.foundation import ChronosEmbedder

emb = ChronosEmbedder()
vectors = emb.embed_stations({"LN1": s1, "MAN3": s2, "BIR2": s3})
# {"LN1": 768-D array, ...} -- cluster, correlate or project them as you like
```

Every series is cut or NaN-padded to the same context length before batching.
That is not cosmetic: Chronos-2 left-pads a ragged batch, which changes the
patch count and shifts a station's pooled embedding depending on which other
stations happened to share its batch.

## Decomposition

`decompose(method="meteorology", backend="chronos-2")` works: it fixes one
meteorological feature at a time and takes successive differences, exactly as
the AutoML path does, with `deweather` in place of `normalise`. Without fitted
importances the feature order comes from each one's individual covariate
sensitivity; pass `variable_order` to pin it.

```python
out = nm.decompose(
    df, target="PM2.5", method="meteorology", backend="chronos-2",
    covariates=["t2m", "blh", "u10", "v10"],
)
# observed, emi_total, t2m, blh, u10, v10, met_total, met_base, met_noise
```

`method="emission"` is refused, and the reason is worth stating because it is
not a missing feature. That decomposition isolates trend, seasonal, weekly and
diurnal components by resampling `date_unix` / `day_julian` / `weekday` /
`hour`. Chronos-2 conditions on the target's own history, and `deweather` never
resamples history, so a repeating calendar signal survives every draw -- the
model reads it off the past and takes nothing from the covariate.

Measured on synthetic series, with the calendar encoders supplied as ordinary
covariates (the most favourable setting), by covariate sensitivity:

| Covariate | Sensitivity | Meteorology, same run |
|---|---|---|
| Time index (trend) | 0.53% | 10.72% |
| Weekly (`dow_sin/cos`) | 0.75% | 4.78% |
| Diurnal (`hour_sin/cos`) | 1.50% | 4.78% |
| All six calendar encoders | 2.47% | 4.78% |

The diurnal signal was the *largest* injected component in that run -- amplitude
15 against a meteorological scale of 10 and a series standard deviation of 18.5
-- and still drew the weaker response. Attribution runs opposite to signal size,
because meteorology is irregular and must be read from the covariate channel
while calendar cycles repeat and can be read from history.

Two consequences. A trend component recovered this way would come back near zero
and read as "no trend" rather than "not separable", which is why the call is
refused rather than served. And no attribution method fixes it: integrated
gradients over the covariate channel would faithfully report near-zero for the
calendar inputs, because near-zero is the truth about how the model uses them.

## Grouping sites by how they behave

`embed_multisite` reads each site's series through the encoder and returns the
768-D vector describing its dynamics. It exists because the shapes did not
meet: `ChronosEmbedder.embed_stations` wants one column per site, while
multi-site frames in this package are long-format.

```python
vectors = nm.embed_multisite(df, site_col="site", target="PM2.5")
# {site: 768-D array}, keyed by your own site values
```

Sites are embedded in a single batched pass, and every series is cut or
NaN-padded to the same context length first -- otherwise a site's vector shifts
depending on which other sites shared its batch.

Grouping sites from those vectors is left to you. KMeans, hierarchical
clustering or a plain correlation matrix are each a few lines of scikit-learn,
and the choices that matter -- the metric, the number of groups, whether to
normalise first -- are yours rather than this package's to make.

## Several species at one site

Chronos-2 takes a 2-D target and attends across the variates, so the species
measured at one site can be forecast as one task rather than several:

```python
out = est.predict_quantiles_multivariate(
    df, ["no2", "nox", "pm25"], anchor="2023-06-01", horizon=48
)
out["no2"]  # q0.1 / q0.5 / q0.9, indexed by forecast timestamp
```

What this buys over looping is the cross-variate structure -- NO2 and NOx at
the same kerbside rise and fall together, and a joint call can use one to
inform the other. The variates must share an index and a horizon. Covariates
are shared across them, because meteorology is a property of the site rather
than of the species.

Neighbouring stations can be variates too, once they are columns of one frame.
That is the point where `normet.physics`'s graphs and the foundation model meet:
the graph says which stations are neighbours, and the variate stack is how you
tell the model to read them together.

## Several sites at once

Sites that do not share an index -- different records, different lengths -- are
separate tasks rather than variates, and go through a different call:

```python
frames = {"MY1": my1_df, "KC1": kc1_df, "HORS": hors_df}
out = est.predict_quantiles_multisite(frames, "no2", anchor="2023-06-01")
out["MY1"]
```

`cross_learning=True` (the default here) lets the model treat the batch as one
group and carry structure between the sites. Upstream reports it helps most
where an individual series has little history, which is exactly a newly
commissioned station sitting next to twenty established ones.

It is not free. Upstream is explicit that cross-learning does not always help
and must be tested per use case, and because the sharing happens *within a
batch*, the answer for a site depends on which other sites were in the call and
on `batch_size`. Pass `cross_learning=False` to get the per-site loop back --
it agrees with calling `predict_quantiles` on each frame to within float32
rounding, which the test suite pins.

## Categorical covariates

Site type, wind sector, a road-class label: Chronos-2 encodes these itself, so
hand them over as strings rather than one-hotting first.

```python
df["site_type"] = "kerbside"     # object dtype, not one-hot
df["wind_sector"] = pd.cut(df["wd"], bins=8).astype(str)

est.predict_quantiles(df, "no2", anchor="2023-06-01",
                      covariates=["ws", "blh", "site_type", "wind_sector"])
```

Anything pandas does not call numeric is treated as a category. Upstream
target-encodes it against the observed target and maps the forecast window onto
the categories seen in the past, so a level that appears only in the future is
handled as unseen rather than silently renumbering the rest. Missing values get
their own category -- a station with no recorded site type is a fact about the
station, not a value to impute.

One caveat: target encoding is only defined against a single target, so on the
multivariate path above upstream falls back to ordinal encoding. The covariate
still reaches the model, just less informatively.

## Fine-tuning

Everything above is zero-shot. `finetune` is the one call that trains, and it
is deliberately not `fit` -- `fit` sits on the sklearn-shaped path that
`do_all` walks, where a call costs nothing and is made freely, and silently
turning that into a thousand optimiser steps would be a trap.

```python
tuned = est.finetune(df, "no2", mode="lora", num_steps=1000)
tuned.predict_quantiles(df, "no2", anchor="2023-06-01")
est.predict_quantiles(df, "no2", anchor="2023-06-01")   # still pretrained
```

A *new* estimator comes back; the original keeps its pretrained weights, so a
fine-tune can be compared against the baseline it came from without reloading
half a gigabyte.

`mode="lora"` (the default) trains low-rank adapters and needs `peft`:
`pip install "normet[finetune]"`. If `peft` is missing, `finetune` raises rather
than proceeding -- upstream's own behaviour is to warn and silently do full
fine-tuning instead, which updates every one of 119M parameters and is a
different and far more expensive thing than what was asked for.

`mode="full"` updates every parameter. LoRA is the default because a single
station's record is small next to the model, which is the setting full
fine-tuning overfits. Whether either helps on your data is a question for your
data; the package supplies the path, not the promise.

## Limits

`decompose`, `rolling` and `pdp` have no Chronos-2 equivalent — they are defined
in terms of a fitted model's response surface, and there is no fitted model
here. Use the AutoML backends for those.
