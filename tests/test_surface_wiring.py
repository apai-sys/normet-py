"""The v3 methods reach the CLI, MCP tools and pipeline config surfaces.

The first-principles methods were library-only when they landed: the pipeline
config ignored their keys (``resolve_config`` filters unknown fields), the CLI
forwarded nine keywords whatever the YAML said, and the MCP tools wrap the CLI
by argv. These tests pin the wiring at each layer so a future option cannot
silently sink back into "Python only".
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from normet.agent.tools import Tools
from normet.core.exceptions import ConfigError
from normet.pipeline import do_all


def _frame(n: int = 48) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "value": [10.0] * n,
            "t2m": [5.0] * n,
            "blh": [500.0] * n,
        }
    )


@pytest.fixture
def patched_do_all(monkeypatch: pytest.MonkeyPatch) -> dict:
    """do_all with prepare/build/normalise replaced; returns the captured normalise kwargs."""
    import importlib

    da = importlib.import_module("normet.pipeline.do_all")

    calls: dict = {}

    def fake_prepare(df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        return df.assign(set="training")

    def fake_build(**kwargs: object) -> tuple[pd.DataFrame, object]:
        return kwargs["df"], object()  # type: ignore[return-value]

    def fake_normalise(**kwargs: object) -> pd.DataFrame:
        calls.update(kwargs)
        return pd.DataFrame(
            {"observed": [1.0], "normalised": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"])
        )

    monkeypatch.setattr(da, "prepare_data", fake_prepare)
    monkeypatch.setattr(da, "build_model", fake_build)
    monkeypatch.setattr(da, "normalise", fake_normalise)
    return calls


def test_do_all_forwards_control_variates_and_block_mode(patched_do_all: dict) -> None:
    out, _model, _prep = do_all(
        _frame(),
        target="value",
        covariates=["t2m", "blh"],
        backend="lightgbm",
        n_samples=2,
        control_variates=True,
        block_mode="markov",
        block_hours=3,
    )
    assert "normalised" in out.columns
    assert patched_do_all["control_variates"] is True
    assert patched_do_all["block_mode"] == "markov"
    assert patched_do_all["block_hours"] == 3


def test_do_all_dml_routes_to_dml_normalise(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    da = importlib.import_module("normet.pipeline.do_all")

    captured: dict = {}
    expected = pd.DataFrame(
        {"observed": [2.0], "normalised": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"])
    )

    def fake_dml(df: pd.DataFrame, target: str = "value", **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        captured["target"] = target
        return SimpleNamespace(normalised=expected)

    monkeypatch.setattr(da, "dml_normalise", fake_dml)
    monkeypatch.setattr(da, "prepare_data", lambda df, **kw: df.assign(set="training"))

    out, model, _prep = do_all(
        _frame(),
        target="value",
        covariates=["t2m", "blh"],
        backend="lightgbm",
        n_samples=2,
        normalise_method="dml",
        dml_cutoff_days=45.0,
        control_variates=True,
        block_mode="markov",
    )
    assert out is expected
    assert model.normalised is expected
    assert captured["cutoff_days"] == 45.0
    assert captured["control_variates"] is True
    assert captured["block_mode"] == "markov"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"normalise_method": "magic"}, "normalise_method"),
        ({"normalise_method": "dml", "cross_fit": True}, "cross_fit"),
        ({"normalise_method": "dml", "backend": "chronos-2"}, "Chronos-2"),
        ({"block_mode": "bridge"}, "block_mode"),
    ],
)
def test_do_all_rejects_incompatible_engine_settings(kwargs: dict, message: str) -> None:
    base = {
        "target": "value",
        "covariates": ["t2m", "blh"],
        "backend": "lightgbm",
        "n_samples": 2,
    }
    with pytest.raises(ConfigError, match=message):
        do_all(_frame(), **{**base, **kwargs})


def test_cli_do_all_forwards_yaml_config_fields(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import normet
    from normet.app.cli import _build_cli

    captured: dict = {}

    def fake_do_all(
        df: pd.DataFrame, **kwargs: object
    ) -> tuple[pd.DataFrame, object, pd.DataFrame]:
        captured.update(kwargs)
        out = pd.DataFrame(
            {"observed": [1.0], "normalised": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"])
        )
        return out, object(), df

    monkeypatch.setattr(normet, "do_all", fake_do_all)
    data = tmp_path / "in.csv"
    _frame().to_csv(data, index=False)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "control_variates: true\n"
        "block_mode: markov\n"
        "block_hours: 3\n"
        "normalise_method: dml\n"
        "dml_cutoff_days: 21\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "out.csv"
    from click.testing import CliRunner

    result = CliRunner().invoke(
        _build_cli(),
        [
            "do-all",
            str(data),
            "--target",
            "value",
            "--covariates",
            "t2m,blh",
            "--backend",
            "lightgbm",
            "--n-samples",
            "2",
            "--config",
            str(cfg),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["control_variates"] is True
    assert captured["block_mode"] == "markov"
    assert captured["block_hours"] == 3
    assert captured["normalise_method"] == "dml"
    assert captured["dml_cutoff_days"] == 21.0
    assert out_path.exists()


def test_tools_start_do_all_argv_carries_new_options(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = Tools()
    captured: dict = {}
    monkeypatch.setattr(tools, "_source", lambda data, example: "in.csv")
    monkeypatch.setattr(
        tools, "_start", lambda kind, argv, label: captured.update(kind=kind, argv=argv) or "ok"
    )
    tools.start_do_all(
        "value",
        ["t2m", "blh"],
        normalise_method="dml",
        control_variates=True,
        block_mode="markov",
        block_hours=3,
        dml_cutoff_days=21.0,
    )
    argv = captured["argv"]
    assert "--normalise-method" in argv and argv[argv.index("--normalise-method") + 1] == "dml"
    assert "--control-variates" in argv
    assert argv[argv.index("--block-mode") + 1] == "markov"
    assert argv[argv.index("--block-hours") + 1] == "3"
    assert argv[argv.index("--dml-cutoff-days") + 1] == "21.0"


def test_tools_start_deweather_argv_carries_surprise_and_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools = Tools()
    captured: dict = {}
    monkeypatch.setattr(tools, "_source", lambda data, example: "in.csv")
    monkeypatch.setattr(tools, "path", lambda p, must_exist=True: p)
    monkeypatch.setattr(
        tools, "_start", lambda kind, argv, label: captured.update(kind=kind, argv=argv) or "ok"
    )
    tools.start_deweather("value", ["t2m"], surprise_tau=1.5, save_bank=True)
    argv = captured["argv"]
    assert argv[argv.index("--surprise-tau") + 1] == "1.5"
    assert argv[argv.index("--save-bank") + 1] == "bank"

    tools.start_deweather("value", ["t2m"], bank_path="/tmp/bank.joblib")
    argv = captured["argv"]
    assert argv[argv.index("--bank") + 1] == "/tmp/bank.joblib"


def test_tools_start_scm_argv_carries_deweathered_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = Tools()
    captured: dict = {}
    monkeypatch.setattr(tools, "_source", lambda data, example: "in.csv")
    monkeypatch.setattr(
        tools, "_start", lambda kind, argv, label: captured.update(kind=kind, argv=argv) or "ok"
    )
    tools.start_scm(
        "A",
        "2020-01-01",
        backend="deweathered",
        covariates=["t2m", "blh"],
        resample_vars=["t2m"],
        deweather_method="dml",
    )
    argv = captured["argv"]
    assert argv[argv.index("--backend") + 1] == "deweathered"
    assert argv[argv.index("--covariates") + 1] == "t2m,blh"
    assert argv[argv.index("--resample-vars") + 1] == "t2m"
    assert argv[argv.index("--deweather-method") + 1] == "dml"


def test_run_scm_routes_deweathered_and_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    import normet.causal.deweathered as dew
    from normet.causal import run_scm

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "ID": ["A"] * 3 + ["B"] * 3,
            "value": [1.0] * 6,
        }
    )
    synthetic = pd.DataFrame({"observed": [1.0], "synthetic": [1.0], "effect": [0.0]})
    seen: dict = {}

    def fake_deweathered(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(synthetic=synthetic)

    def fake_transport(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(synthetic=synthetic)

    monkeypatch.setattr(dew, "deweathered_scm", fake_deweathered)
    monkeypatch.setattr(dew, "transport_adjusted_scm", fake_transport)

    out = run_scm(
        df=frame,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="A",
        cutoff_date="2020-01-02",
        donors=["B"],
        scm_backend="deweathered",
        covariates=["t2m"],
    )
    assert out is synthetic
    assert seen["covariates"] == ["t2m"]

    out = run_scm(
        df=frame,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="A",
        cutoff_date="2020-01-02",
        donors=["B"],
        scm_backend="transport",
        spillover_exposure="F",
    )
    assert out is synthetic
    assert seen["spillover_exposure"] == "F"


def test_cli_met_features_adds_physics_columns(tmp_path) -> None:
    from click.testing import CliRunner

    from normet.app.cli import _build_cli

    src = tmp_path / "in.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="h"),
            "ws": [1.0] * 6,
            "wd": [180.0] * 6,
            "blh": [500.0] * 6,
            "t2m": [280.0] * 6,
            "rh": [50.0] * 6,
        }
    ).to_csv(src, index=False)
    out = tmp_path / "out.csv"
    result = CliRunner().invoke(_build_cli(), ["met-features", str(src), "--out", str(out)])
    assert result.exit_code == 0, result.output
    got = pd.read_csv(out)
    assert {"u_wind", "v_wind", "ventilation_coef", "vpd"} <= set(got.columns)


def test_tools_make_met_features_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = Tools()
    captured: dict = {}
    monkeypatch.setattr(tools, "path", lambda p: str(p))
    monkeypatch.setattr(
        tools, "_start", lambda kind, argv, label: captured.update(kind=kind, argv=argv) or "ok"
    )
    tools.make_met_features("in.csv", include_memory=True, group_col="site")
    argv = captured["argv"]
    assert captured["kind"] == "met_features"
    assert argv[:2] == ["met-features", "in.csv"]
    assert "--include-memory" in argv
    assert argv[argv.index("--group-col") + 1] == "site"
    assert argv[argv.index("--out") + 1] == "met_features.csv"
