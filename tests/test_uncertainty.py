"""Uncertainty decomposition: the per-date variance attribution to weather,
model form and their interaction, and the two-way additive identity it rests on.

The additive identity (weather + model + interaction == total, every term a sum
of squares) is the correctness contract of the attribution, so it is tested on
raw matrices independent of any model. The *dominance* scenarios (weather-driven
vs model-driven) are driven through the public API with a deterministic stub
backend, so the weather effect (``c * t2m``) and the model offset (``seed % 7``)
are exactly known, and the attribution is checked against the closed form rather
than against a trained model's behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet import UncDecomposition, decompose_uncertainty
from normet.analysis.uncertainty import _decompose
from normet.backends import backend_registry
from normet.exceptions import ConfigError

# ---------------------------------------------------------------------------
# _decompose: the two-way identity, checked on raw matrices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2, 3, 4), (5, 7, 2), (3, 3, 30)])
def test_the_additive_identity_holds_for_any_matrix(shape):
    def check(z):
        vw, vm, vi, vt = _decompose(z)
        assert np.allclose(vw + vm + vi, vt, rtol=0, atol=1e-9), (vw, vm, vi, vt)
        assert (vw >= -1e-12).all() and (vm >= -1e-12).all() and (vi >= -1e-12).all()

    rng = np.random.default_rng(0)
    check(rng.normal(scale=rng.uniform(0.1, 9, size=shape)))
    check(rng.exponential(size=shape))
    check(rng.integers(-3, 3, size=shape).astype(float))


def test_an_additive_split_has_zero_interaction_and_knowable_main_effects():
    # Z[i, j] = c * t2m_{j, n} + a_i is additive: the models differ by a_i, the
    # draws by the resampled t2m, and there is no model-by-draw interaction. The
    # attribution must read the model term as var_i(a_i) and the weather term as
    # c^2 * the per-date variance of the draw values, with the interaction at 0.
    rng = np.random.default_rng(1)
    a = rng.normal(size=4)
    c = 2.5
    n_dates = 12
    W = rng.normal(size=(6, n_dates))  # the (resampled) t2m values, by draw and date
    Z3 = a[:, None, None] + c * W[None, :, :]
    vw, vm, vi, vt = _decompose(Z3)
    assert np.allclose(vm, np.var(a), atol=1e-9)  # constant across dates
    assert np.allclose(vw, c**2 * np.var(W, axis=0), atol=1e-9)
    assert np.allclose(vi, 0.0, atol=1e-9)
    assert np.allclose(vt, vm + vw, atol=1e-9)


def test_a_constant_response_carryns_no_variance():
    z = np.full((2, 3, 4), 5.0)
    vw, vm, vi, vt = _decompose(z)
    assert np.allclose([vw, vm, vi, vt], 0.0)


# ---------------------------------------------------------------------------
# the result object: shares, summary, frame
# ---------------------------------------------------------------------------


def _mk(vw, vm, vi, idx=None) -> UncDecomposition:
    idx = idx if idx is not None else pd.date_range("2024-01-01", periods=len(vw))
    vw, vm, vi = (np.asarray(v, float) for v in (vw, vm, vi))
    return UncDecomposition(
        var_weather=vw,
        var_model=vm,
        var_interaction=vi,
        var_total=vw + vm + vi,
        index=idx,
        target="X",
        n_models=3,
        n_weather_draws=5,
        seed=1,
    )


def test_the_shares_sum_to_one_and_the_summary_reports_medians():
    res = _mk([1.0, 4.0], [3.0, 2.0], [4.0, 4.0])
    assert np.allclose(res.pct_weather + res.pct_model + res.pct_interaction, 1.0)
    s = res.summary()
    assert list(s["source"]) == ["weather", "model", "interaction", "total"]
    # median of pct_weather = median(1/8, 4/10)
    assert s.loc[s.source == "weather", "pct_median"].iloc[0] == pytest.approx(0.2625)
    assert s.loc[s.source == "total", "pct_median"].iloc[0] == pytest.approx(1.0)
    f = res.frame()
    assert list(f.index) == list(res.index)
    assert (f["var_total"] == f["var_weather"] + f["var_model"] + f["var_interaction"]).all()


def test_a_zero_total_row_has_nan_shares_not_a_division_error():
    res = _mk([0.0], [0.0], [0.0])
    # nothing varies: every share is undefined rather than 0/0 (no div-by-zero)
    assert np.isnan(res.pct_weather[0])
    assert np.isnan(res.pct_model[0])
    assert np.isnan(res.pct_interaction[0])


def test_the_summary_and_string_summarise_the_median_shares():
    res = _mk([4.0, 4.0], [0.0, 0.0], [0.0, 0.0])
    assert res.summary().loc[res.summary().source == "weather", "pct_median"].iloc[
        0
    ] == pytest.approx(1.0)
    txt = str(res)
    assert "100%" in txt and "0%" in txt and "X:" in txt


# ---------------------------------------------------------------------------
# a deterministic stub backend for the dominance scenarios through the API
# ---------------------------------------------------------------------------

_STUB_REGISTERED = False


class _StubModel:
    """predicts ``a + c * t2m`` -- the model offset ``a`` and the weather scale ``c``."""

    def __init__(self, a: float, c: float, features: tuple[str, ...]):
        self.backend = "uncstub"  # the registered name; ml_predict dispatches on it
        self.a = a
        self.c = c
        self.feature_names_in_ = features

    def predict(self, X):
        w = X["t2m"].to_numpy(dtype=float)
        return self.a + self.c * w


class _StubBackend:
    """A registered backend whose models are fully controlled; a_i = seed % 7."""

    name = "uncstub"

    def train(
        self,
        df,
        target="value",
        covariates=None,
        variables=None,
        model_config=None,
        seed=0,
        verbose=False,
        n_cores=None,
        cache=None,
    ):
        c = (model_config or {}).get("c", 1.0)
        return _StubModel(int(seed) % 7, float(c), tuple(covariates or ()))

    def save(self, model, path=".", filename="stub.joblib"):
        return path

    def load(self, path=".", filename=None):
        raise NotImplementedError


@pytest.fixture(scope="module", autouse=True)
def _stub_registered():
    global _STUB_REGISTERED
    if not _STUB_REGISTERED:
        backend_registry.register(_StubBackend())
        _STUB_REGISTERED = True
    yield


def _aq_data(n: int = 24 * 14, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    t2m = 10 + 8 * np.sin(2 * np.pi * np.arange(n) / (24 * 30)) + rng.normal(0, 1, n)
    blh = 800 + 400 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 50, n)
    u10 = rng.normal(0, 3, n)
    pm = 20 - 0.01 * blh + 0.3 * (t2m - t2m.mean()) ** 2 + 0.5 * u10 + rng.normal(0, 2, n)
    return pd.DataFrame(
        {"date": dates, "PM2.5": np.clip(pm, 0, None), "t2m": t2m, "blh": blh, "u10": u10}
    )


@pytest.fixture
def aq():
    return _aq_data()


COMMON = dict(
    backend="uncstub",
    covariates=["t2m", "blh", "u10"],
    variables_resample=["t2m", "blh", "u10"],
    split_method="ts",
    train_fraction=0.8,
    n_samples=8,
    n_models=3,
    seed=999,
    n_cores=1,
    verbose=False,
)


def _member_offsets(n_models: int, seed: int) -> np.ndarray:
    """The model offsets a_i = member_seed % 7, in the order the run derives them."""
    rs = np.random.default_rng([seed, 1])
    return np.asarray([int(s) % 7 for s in rs.choice(1_000_000, size=n_models, replace=False)])


def test_the_identity_holds_end_to_end_and_weather_can_dominate(aq):
    res = decompose_uncertainty(aq, "PM2.5", **COMMON, model_config={"c": 30.0})
    d = res.frame()
    # the variance-identity contract, on every date, to floating point
    assert np.allclose(
        d["var_weather"] + d["var_model"] + d["var_interaction"], d["var_total"], rtol=0, atol=1e-9
    )
    assert (d[["var_weather", "var_model", "var_interaction"]] >= -1e-9).all().all()
    # a big weather scale c makes the between-draw term dominate the whole band
    s = res.summary()
    assert s.loc[s.source == "weather", "pct_median"].iloc[0] > 0.85
    # the model main effect equals the closed form for a_i = member_seed % 7
    a = _member_offsets(COMMON["n_models"], COMMON["seed"])
    assert s.loc[s.source == "model", "var_median"].iloc[0] == pytest.approx(np.var(a), rel=1e-6)
    # the additive stub has no model-by-draw interaction to attribute
    assert s.loc[s.source == "interaction", "var_median"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    # the deterministic attribution reproduces the same numbers on a re-run
    res2 = decompose_uncertainty(aq, "PM2.5", **COMMON, model_config={"c": 30.0})
    assert np.allclose(res2.var_weather, res.var_weather)
    assert np.allclose(res2.var_model, res.var_model)


def test_a_zero_weather_scale_puts_the_whole_spread_on_model_form(aq):
    res = decompose_uncertainty(aq, "PM2.5", **COMMON, model_config={"c": 0.0})
    s = res.summary()
    # c = 0: the between-draw and interaction terms vanish, the model term is 100 %
    assert s.loc[s.source == "weather", "var_median"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert s.loc[s.source == "interaction", "var_median"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert s.loc[s.source == "model", "pct_median"].iloc[0] == pytest.approx(1.0)
    a = _member_offsets(COMMON["n_models"], COMMON["seed"])
    assert s.loc[s.source == "model", "var_median"].iloc[0] == pytest.approx(np.var(a), rel=1e-6)


def test_the_result_carries_the_config_and_plots_a_stacked_band(aq):
    res = decompose_uncertainty(aq, "PM2.5", **COMMON, model_config={"c": 5.0})
    assert res.n_models == COMMON["n_models"] and res.n_weather_draws == COMMON["n_samples"]
    assert "weather" in str(res) and "model" in str(res) and "interaction" in str(res)
    ax = res.plot()
    # the stacked figure has one patch collection per band (weather, model, interaction)
    assert len(ax.collections) == 3
    assert len(res.index) == len(aq)


# ---------------------------------------------------------------------------
# argument guards
# ---------------------------------------------------------------------------


def test_one_level_in_a_factor_is_refused(aq):
    for key in ("n_models", "n_samples"):
        base = {k: v for k, v in COMMON.items() if k != key}
        with pytest.raises(ConfigError, match=key):
            decompose_uncertainty(aq, "PM2.5", **base, **{key: 1})


def test_covariates_are_required(aq):
    base = {k: v for k, v in COMMON.items() if k != "covariates"}
    with pytest.raises(ConfigError, match="covariates"):
        decompose_uncertainty(aq, "PM2.5", **base)
