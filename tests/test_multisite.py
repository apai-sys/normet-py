"""Multisite driver — exercise multisite_apply without a real model."""

import pandas as pd
import pytest

from normet.pipeline.multisite import (
    decompose_multisite,
    do_all_multisite,
    multisite_apply,
)


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"mean_pm": [df["PM2.5"].mean()]})


def _summarise_with_kwarg(df: pd.DataFrame, site: str = "") -> pd.DataFrame:
    return pd.DataFrame({"mean_pm": [df["PM2.5"].mean()], "site_arg": [site]})


def test_multisite_apply_loops_over_sites(synthetic_aq):
    aq = synthetic_aq.copy()
    aq["site"] = "A"
    aq2 = synthetic_aq.copy().assign(site="B")
    df = pd.concat([aq, aq2], ignore_index=True)
    out = multisite_apply(df, site_col="site", func=_summarise, n_cores=1)
    assert {"site", "mean_pm"} <= set(out.columns)
    assert set(out["site"].unique()) == {"A", "B"}


def test_multisite_apply_raises_on_missing_col(synthetic_aq):
    with pytest.raises(ValueError):
        multisite_apply(synthetic_aq, site_col="not_a_col", func=_summarise)


def test_multisite_apply_empty_sites_returns_empty(synthetic_aq):
    df = synthetic_aq.copy()
    df["site"] = ""
    # Filter so there are zero sites with data
    out = multisite_apply(df.iloc[:0], site_col="site", func=_summarise)
    assert isinstance(out, pd.DataFrame) and out.empty


def test_multisite_apply_site_kwarg(synthetic_aq):
    aq = synthetic_aq.copy()
    aq["site"] = "A"
    out = multisite_apply(
        aq,
        site_col="site",
        func=_summarise_with_kwarg,
        n_cores=1,
        site_kwarg="site",
    )
    assert "site_arg" in out.columns
    assert out["site_arg"].iloc[0] == "A"


def test_multisite_apply_all_fail_raises(synthetic_aq):
    def _failing(_df):
        raise RuntimeError("boom")

    aq = synthetic_aq.copy()
    aq["site"] = "X"
    with pytest.raises(RuntimeError, match="All per-site runs failed"):
        multisite_apply(aq, site_col="site", func=_failing, n_cores=1)


def test_multisite_apply_returns_empty_df(synthetic_aq):
    def _empty(_df):
        return pd.DataFrame()

    aq = synthetic_aq.copy()
    aq["site"] = "A"
    with pytest.raises(RuntimeError, match="All per-site"):
        multisite_apply(aq, site_col="site", func=_empty, n_cores=1)


def test_multisite_apply_func_returns_tuple(synthetic_aq):
    def _tuple(df):
        return (df[["PM2.5"]].head(1),)

    aq = synthetic_aq.copy()
    aq["site"] = "A"
    out = multisite_apply(aq, site_col="site", func=_tuple, n_cores=1)
    assert "site" in out.columns


def test_do_all_multisite_with_sklearn(synthetic_aq):
    from normet.backends import backend_registry

    if not backend_registry.has("sklearn"):
        pytest.skip("sklearn backend not registered")
    aq = synthetic_aq.copy()
    aq["site"] = "X"
    result = do_all_multisite(
        aq,
        site_col="site",
        target="PM2.5",
        covariates=["t2m", "blh"],
        backend="sklearn",
        n_cores=1,
        n_samples=3,
    )
    assert "date" in result.columns
    assert "observed" in result.columns


def test_do_all_multisite_raises_on_missing_col(synthetic_aq):
    with pytest.raises((ValueError, KeyError)):
        do_all_multisite(
            synthetic_aq,
            site_col="site",
            target="PM2.5",
            covariates=["t2m", "blh"],
            backend="flaml",
            n_cores=1,
        )


def test_decompose_multisite_raises_on_missing_col(synthetic_aq):
    with pytest.raises(ValueError, match="not in df"):
        decompose_multisite(
            synthetic_aq,
            site_col="not_a_col",
            target="PM2.5",
            covariates=["t2m", "blh"],
            backend="flaml",
            n_cores=1,
        )


def test_decompose_multisite_raises_on_missing_col(synthetic_aq):
    with pytest.raises(ValueError, match="not in df"):
        decompose_multisite(
            synthetic_aq,
            site_col="site",
            target="PM2.5",
            covariates=["t2m", "blh"],
            backend="flaml",
            n_cores=1,
        )
