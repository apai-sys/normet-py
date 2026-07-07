"""Manual smoke test for the new series-level convergence criterion in
normalise_auto (convergence_metric="series"/"global"). Run standalone:
python smoke_normalise_auto_series.py"""

import time
import warnings

import numpy as np
import pandas as pd

t0 = time.time()
import normet as nm  # noqa: E402

rng = np.random.default_rng(42)
n = 500
dates = pd.date_range("2021-01-01", periods=n, freq="h")
met1 = rng.normal(10, 3, n)
met2 = rng.normal(0, 1, n)
y = 5 + 2 * met1 - 3 * met2 + rng.normal(0, 2, n)
df = pd.DataFrame({"date": dates, "y": y, "met1": met1, "met2": met2})
FEATS = ["met1", "met2", "date_unix", "day_julian", "weekday", "hour"]

prep, model = nm.build_model(
    df,
    value="y",
    backend="lightgbm",
    feature_names=FEATS,
    split_method="random",
    fraction=0.75,
    seed=1,
    model_config={"n_trials": 1, "cv_folds": 2, "nrounds": 20},
    verbose=False,
)
print(f"model built {time.time() - t0:.1f}s", flush=True)

kw = dict(
    feature_names=FEATS,
    variables_resample=["met1", "met2"],
    batch_size=10,
    seed=1,
    verbose=False,
    n_cores=1,
)

# 1. series metric, loose tol -> stops at floor batch*(streak(3)+1) = 40
r = nm.normalise_auto(
    prep, model, convergence_tol="50%", max_samples=200, return_history=True, **kw
)
print(f"series loose: best_n={r['best_n']} (expect 40)", flush=True)
assert r["best_n"] == 40
assert set(r["res"].columns) == {"date", "observed", "normalised"}
assert r["res"]["normalised"].notna().all() and len(r["res"]) == n
assert list(r["history"].columns) == ["n", "metric", "global_mean", "stable_count"]
print(r["history"].to_string(index=False), flush=True)

# 2. RSE declines ~1/sqrt(n_batches): metric at n=200 ≈ metric at n=50 * sqrt(4/19)
r_long = nm.normalise_auto(
    prep, model, convergence_tol="0.0001%", max_samples=200, return_history=True, **kw
)
h = r_long["history"]
m50, m200 = h.loc[h.n == 50, "metric"].iloc[0], h.loc[h.n == 200, "metric"].iloc[0]
ratio = m50 / m200
print(
    f"RSE scaling: metric(n=50)/metric(n=200) = {ratio:.2f} (CLT predicts ~{np.sqrt(19 / 4):.2f})",
    flush=True,
)
assert 1.3 < ratio < 4.0, "RSE not declining like 1/sqrt(n)"

# 3. strict tol hits max_samples with warning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r2 = nm.normalise_auto(prep, model, convergence_tol="0.0001%", max_samples=60, **kw)
    assert r2["best_n"] == 60
    assert any("without strict convergence" in str(x.message) for x in w)
print("strict-tol max_samples + warning OK", flush=True)

# 4. legacy global metric: loose tol -> floor 10*(5+1)=60
r3 = nm.normalise_auto(
    prep, model, convergence_metric="global", convergence_tol="50%", max_samples=200, **kw
)
print(f"global loose: best_n={r3['best_n']} (expect 60)", flush=True)
assert r3["best_n"] == 60

# 5. same-n results identical across metrics (aggregation math is metric-independent)
r4 = nm.normalise_auto(
    prep, model, convergence_metric="global", convergence_tol="0.0001%", max_samples=40, **kw
)
merged = r["res"].merge(r4["res"], on="date", suffixes=("_s", "_g"))
assert np.allclose(merged["normalised_s"], merged["normalised_g"])
print("series/global res identical at same n OK", flush=True)

# 6. bad metric raises ConfigError
try:
    nm.normalise_auto(prep, model, convergence_metric="bogus", **kw)
    raise SystemExit("should have raised")
except Exception as e:
    assert type(e).__name__ == "ConfigError"
    print("bad metric raises ConfigError OK", flush=True)

print(f"ALL SMOKE TESTS PASSED ({time.time() - t0:.1f}s)")
