"""Regression invariants for time-series decomposition (FLAML-gated).

These avoid brittle golden numbers: they assert structural identities that the
decomposition must satisfy by construction, plus determinism under a fixed model
and seed. A fixed pre-trained model is reused so results don't depend on FLAML's
wall-clock-bounded search being reproducible across runs.
"""

import numpy as np
import pandas as pd
import pytest


def _flaml_importable() -> bool:
    try:
        from flaml.automl import AutoML  # noqa: F401

        return True
    except Exception:
        return False


needs_flaml = pytest.mark.skipif(not _flaml_importable(), reason="flaml not importable")

FEATS = ["t2m", "blh", "u10", "v10", "date_unix", "day_julian", "weekday", "hour"]
MODELCFG = {"time_budget": 5, "metric": "r2", "estimator_list": ["lgbm"]}


@pytest.fixture(scope="module")
def trained(synthetic_aq):
    """A single FLAML model + prepared frame reused across the invariant tests."""
    from normet.model.train import train_model
    from normet.utils.prepare import prepare_data

    df_prep = prepare_data(
        synthetic_aq.copy(),
        value="PM2.5",
        feature_names=FEATS,
        split_method="ts",
        fraction=0.8,
    )
    model = train_model(
        df_prep,
        value="value",
        backend="flaml",
        feature_names=FEATS,
        model_config=MODELCFG,
        verbose=False,
    )
    return df_prep, model


@needs_flaml
def test_meteorology_decomposition_closure(trained):
    import normet as nm

    df_prep, model = trained
    res = nm.decompose(
        df=df_prep,
        model=model,
        method="meteorology",
        value="value",
        feature_names=FEATS,
        n_samples=8,
        seed=7654321,
        n_cores=1,
    )
    # By construction: met_total = observed - emi_total, so the two parts must
    # add back to the observed series exactly.
    assert {"observed", "emi_total", "met_total"} <= set(res.columns)
    recon = res["emi_total"].to_numpy() + res["met_total"].to_numpy()
    assert np.allclose(recon, res["observed"].to_numpy(), atol=1e-8)


@needs_flaml
def test_decomposition_is_deterministic(trained):
    import normet as nm

    df_prep, model = trained
    kw = dict(
        df=df_prep,
        model=model,
        method="meteorology",
        value="value",
        feature_names=FEATS,
        n_samples=8,
        seed=7654321,
        n_cores=1,
    )
    r1 = nm.decompose(**kw)
    r2 = nm.decompose(**kw)
    # Fixed model + fixed seed → bit-identical resampling/decomposition.
    pd.testing.assert_frame_equal(r1, r2)
