"""Smoke tests for the click-based CLI (gated on click availability)."""

import importlib.util
import json

import pytest

needs_click = pytest.mark.skipif(
    importlib.util.find_spec("click") is None, reason="click not installed"
)


@needs_click
def test_cli_info_runs():
    from click.testing import CliRunner

    from normet.cli import _build_cli

    runner = CliRunner()
    result = runner.invoke(_build_cli(), ["info"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "normet" in data
    assert "python" in data


@needs_click
def test_cli_help():
    from click.testing import CliRunner

    from normet.cli import _build_cli

    runner = CliRunner()
    result = runner.invoke(_build_cli(), ["--help"])
    assert result.exit_code == 0
    for cmd in ("do-all", "decompose", "scm", "cv", "info"):
        assert cmd in result.output


@needs_click
def test_cli_backend_choices_track_the_registry():
    """The --backend literal had gone stale, offering flaml long after lightgbm registered.

    Sourcing the choices from the registry is only worth anything if it stays
    sourced from the registry, so assert the two agree rather than pinning a list
    here that can drift the same way.
    """
    from click.testing import CliRunner

    from normet.backends import backend_registry
    from normet.cli import _build_cli

    runner = CliRunner()
    for cmd in ("do-all", "decompose", "cv"):
        out = runner.invoke(_build_cli(), [cmd, "--help"]).output
        line = next(ln for ln in out.splitlines() if "--backend" in ln)
        for name in backend_registry.available:
            assert name in line, f"{cmd}: {name} missing from {line!r}"


@needs_click
def test_only_the_commands_with_a_zero_shot_path_offer_that_backend():
    """chronos-2 is not in the registry, and only some commands have a path for it.

    The registry's contract is train/save/load and Chronos-2 does none of the
    three, so it is added to one command's choice list rather than registered.
    decompose and cv have no zero-shot counterpart and must not advertise one.
    """
    from click.testing import CliRunner

    from normet.cli import _build_cli
    from normet.pipeline import CHRONOS_BACKEND

    runner = CliRunner()

    def backend_line(cmd: str) -> str:
        out = runner.invoke(_build_cli(), [cmd, "--help"]).output
        return next(ln for ln in out.splitlines() if "--backend" in ln)

    for cmd in ("do-all", "decompose"):
        assert CHRONOS_BACKEND in backend_line(cmd), cmd
    # cv trains and scores folds; there is nothing to train zero-shot.
    assert CHRONOS_BACKEND not in backend_line("cv")


@needs_click
def test_cli_deweather_is_advertised_and_documented():
    from click.testing import CliRunner

    from normet.cli import _build_cli

    runner = CliRunner()
    assert "deweather" in runner.invoke(_build_cli(), ["--help"]).output

    out = runner.invoke(_build_cli(), ["deweather", "--help"]).output
    for flag in ("--target", "--met-vars", "--n-samples", "--quantiles", "--device", "--out"):
        assert flag in out, flag


@needs_click
def test_cli_deweather_refuses_without_met_vars(tmp_path):
    """Without covariates the run is a plain forecast, not a de-weathering.

    Failing here costs a second; failing after the checkpoint download costs
    minutes and produces a column named `normalised` that normalises nothing.
    """
    import pandas as pd
    from click.testing import CliRunner

    from normet.cli import _build_cli

    src = tmp_path / "site.csv"
    pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=48, freq="h"), "PM2.5": range(48)}
    ).to_csv(src, index=False)

    result = CliRunner().invoke(
        _build_cli(),
        ["deweather", str(src), "--target", "PM2.5", "--out", str(tmp_path / "o.csv")],
    )
    assert result.exit_code != 0
    assert "--met-vars is required" in result.output


@needs_click
def test_cli_info_reports_the_foundation_dependencies():
    """`normet info` is where someone checks why the chronos-2 path is unavailable."""
    import json

    from click.testing import CliRunner

    from normet.backends import backend_registry
    from normet.cli import _build_cli

    data = json.loads(CliRunner().invoke(_build_cli(), ["info"]).output)
    for pkg in ("chronos-forecasting", "torch"):
        assert pkg in data["optional"], pkg
    assert data["backends"] == backend_registry.available


@needs_click
def test_cli_split_csv_accepts_a_yaml_list():
    """`covariates: [t2m, blh]` in a config file arrives as a list, not a string."""
    from normet.cli import _split_csv

    assert _split_csv("t2m, blh ,u10") == ["t2m", "blh", "u10"]
    assert _split_csv(["t2m", "blh"]) == ["t2m", "blh"]
    assert _split_csv([0.1, 0.9]) == ["0.1", "0.9"]
    assert _split_csv(None) is None
    assert _split_csv("") is None


@needs_click
def test_cli_do_all_and_decompose_preserve_backend_default_n_samples(monkeypatch, tmp_path):
    """Omitting --n-samples must pass None so Chronos-2 keeps its safe default of 8."""
    import pandas as pd
    from click.testing import CliRunner

    import normet
    from normet.cli import _build_cli

    src = tmp_path / "site.csv"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=48, freq="h"),
            "PM2.5": range(48),
            "t2m": 10.0,
        }
    )
    df.to_csv(src, index=False)

    seen_do_all = []
    seen_decompose = []

    def _fake_do_all(*args, **kwargs):
        seen_do_all.append(kwargs.get("n_samples"))
        out = pd.DataFrame({"observed": [1.0], "normalised": [1.0]}, index=df["date"][:1])
        out.index.name = "date"
        return out, object(), df

    def _fake_decompose(*args, **kwargs):
        seen_decompose.append(kwargs.get("n_samples"))
        out = pd.DataFrame({"observed": [1.0], "emi_total": [1.0]}, index=df["date"][:1])
        out.index.name = "date"
        return out

    monkeypatch.setattr(normet, "do_all", _fake_do_all)
    monkeypatch.setattr(normet, "decompose", _fake_decompose)

    runner = CliRunner()
    r1 = runner.invoke(
        _build_cli(),
        [
            "do-all",
            str(src),
            "--target",
            "PM2.5",
            "--covariates",
            "t2m",
            "--backend",
            "chronos-2",
            "--out",
            str(tmp_path / "do_all.csv"),
        ],
    )
    assert r1.exit_code == 0, r1.output
    r2 = runner.invoke(
        _build_cli(),
        [
            "decompose",
            str(src),
            "--target",
            "PM2.5",
            "--covariates",
            "t2m",
            "--method",
            "meteorology",
            "--backend",
            "chronos-2",
            "--out",
            str(tmp_path / "decomp.csv"),
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert seen_do_all == [None]
    assert seen_decompose == [None]


@needs_click
def test_cli_deweather_icn_method_and_help(monkeypatch, tmp_path):
    """`normet deweather --method icn` routes through ICN-v1 and saves normet schema."""
    import numpy as np
    import pandas as pd
    from click.testing import CliRunner

    from normet.cli import _build_cli
    from normet.foundation import Chronos2Estimator, ICNResult

    runner = CliRunner()
    help_out = runner.invoke(_build_cli(), ["deweather", "--help"]).output
    assert "--method" in help_out
    assert "--estimate" in help_out

    src = tmp_path / "site.csv"
    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({"date": idx, "PM2.5": np.linspace(10.0, 20.0, 100), "t2m": 5.0})
    df.to_csv(src, index=False)

    monkeypatch.setattr(Chronos2Estimator, "_load_pipeline", lambda self: None)
    monkeypatch.setattr(
        Chronos2Estimator,
        "covariate_sensitivity",
        lambda self, *a, **kw: {
            "mean_abs_shift": 1.2,
            "max_abs_shift": 2.0,
            "pct_of_prediction": 8.0,
        },
    )

    seen_designs = []

    def _fake_nic(self, frame, target, *, design=None, met_features=None, quantiles=None, **kw):
        seen_designs.append((design, list(frame.columns), quantiles))
        return ICNResult(
            level=pd.Series(np.full(len(frame), 12.0), index=frame.index, name="level"),
            paired=pd.Series(np.full(len(frame), 15.0), index=frame.index, name="paired"),
            weather_effect=pd.Series(
                np.zeros(len(frame)), index=frame.index, name="weather_effect"
            ),
            spread_by_source=pd.DataFrame({"anchor": 0.0, "mc": 0.0}, index=frame.index),
            hour_contrib=pd.Series(np.ones(len(frame), dtype=int), index=frame.index),
            diagnostics={"n_anchors": 2, "uncovered_hours": 0},
            observed=frame[target].rename("observed"),
            quantiles=pd.DataFrame({"q100": 10.0, "q500": 12.0, "q900": 14.0}, index=frame.index),
            paired_quantiles=pd.DataFrame(
                {"q100": 13.0, "q500": 15.0, "q900": 17.0}, index=frame.index
            ),
        )

    monkeypatch.setattr(Chronos2Estimator, "normalise_in_context", _fake_nic)

    out_csv = tmp_path / "icn.csv"
    res = runner.invoke(
        _build_cli(),
        [
            "deweather",
            str(src),
            "--target",
            "PM2.5",
            "--met-vars",
            "t2m",
            "--context-length",
            "48",
            "--prediction-length",
            "24",
            "--n-samples",
            "3",
            "--method",
            "icn",
            "--estimate",
            "paired",
            "--log-transform",
            "--out",
            str(out_csv),
        ],
    )
    assert res.exit_code == 0, res.output
    assert len(seen_designs) == 1
    design, cols, qs = seen_designs[0]
    assert design.intervene == "future_only"
    assert design.direction == "bidirectional"
    assert design.context_length == 48
    assert design.block_h == 24
    assert design.stride == 12
    assert design.log_transform is True
    assert design.n_resamples == 3
    assert "hour_sin" in cols and "doy_cos" in cols
    assert qs == (0.1, 0.5, 0.9)
    saved = pd.read_csv(out_csv)
    assert list(saved.columns) == ["date", "observed", "normalised", "q100", "q500", "q900"]
    assert (saved["normalised"] == 15.0).all()

    bad = runner.invoke(
        _build_cli(),
        [
            "deweather",
            str(src),
            "--target",
            "PM2.5",
            "--met-vars",
            "t2m",
            "--method",
            "deweather",
            "--log-transform",
            "--out",
            str(tmp_path / "bad.csv"),
        ],
    )
    assert bad.exit_code != 0


@needs_click
def test_cli_scm_chronos_backend(monkeypatch, tmp_path):
    """`normet scm --backend chronos` exposes and dispatches chronos backend."""
    import pandas as pd
    from click.testing import CliRunner

    import normet
    from normet.cli import _build_cli

    runner = CliRunner()
    assert "chronos" in runner.invoke(_build_cli(), ["scm", "--help"]).output

    panel = tmp_path / "panel.csv"
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ID": ["T"] * 10 + ["D1"] * 10,
            "value": list(range(10)) + list(range(10)),
        }
    ).to_csv(panel, index=False)

    seen: list[str] = []

    def _fake_run_scm(**kw):
        seen.append(str(kw.get("scm_backend")))
        out = pd.DataFrame(
            {"observed": [1.0], "synthetic": [1.0], "effect": [0.0]},
            index=dates[:1],
        )
        out.index.name = "date"
        return out

    monkeypatch.setattr(normet, "run_scm", _fake_run_scm)
    out_csv = tmp_path / "scm_chronos.csv"
    r = runner.invoke(
        _build_cli(),
        [
            "scm",
            str(panel),
            "--treated",
            "T",
            "--cutoff",
            "2024-01-06",
            "--backend",
            "chronos",
            "--out",
            str(out_csv),
        ],
    )
    assert r.exit_code == 0, r.output
    assert seen == ["chronos"]


@needs_click
def test_cli_impute_backtest_events_and_model_path(monkeypatch, tmp_path):
    """Test new CLI commands (`impute`, `backtest`, `events`) and `--model-path`."""
    import numpy as np
    import pandas as pd
    from click.testing import CliRunner

    import normet
    import normet.foundation.operators as ops_mod
    from normet.cli import _build_cli
    from normet.foundation import Chronos2Estimator, QuantileCalibration

    runner = CliRunner()
    top_help = runner.invoke(_build_cli(), ["--help"]).output
    for cmd in ("impute", "backtest", "events"):
        assert cmd in top_help

    idx = pd.date_range("2024-01-01", periods=60, freq="h")
    vals = np.linspace(10.0, 20.0, 60)
    vals_gap = vals.copy()
    vals_gap[20:23] = np.nan
    src = tmp_path / "series.csv"
    pd.DataFrame({"date": idx, "PM2.5": vals_gap, "t2m": 8.0}).to_csv(src, index=False)

    ckpt_dir = tmp_path / "saved_ckpt"
    ckpt_dir.mkdir()
    loaded_paths: list[str] = []

    def _fake_load(cls, path, device=None):
        loaded_paths.append(str(path))
        return cls(device=device or "cpu", context_length=12, prediction_length=6)

    monkeypatch.setattr(Chronos2Estimator, "load", classmethod(_fake_load))
    monkeypatch.setattr(Chronos2Estimator, "_load_pipeline", lambda self: None)

    seen_impute: dict[str, object] = {}

    def _fake_cal(est, frame, target, anchor, *, quantiles=None, covariates=None, **kw):
        return QuantileCalibration(
            levels=(0.9,),
            threshold=[1.0],
            half_width_factor=[1.0],
            coverage=pd.DataFrame(),
            warnings=(),
            method="constant",
            n_points=10,
            n_blocks=1,
        )

    def _fake_impute(
        self,
        frame,
        target,
        *,
        covariates=None,
        direction="bidirectional",
        quantiles=None,
        max_gap=None,
        calibration=None,
        **kw,
    ):
        seen_impute.update(
            direction=direction,
            max_gap=max_gap,
            calibrated=calibration is not None,
            covariates=list(covariates or []),
        )
        return pd.DataFrame(
            {
                "observed": frame[target].to_numpy(dtype=float),
                "imputed": np.linspace(10.0, 20.0, len(frame)),
                "is_imputed": np.isnan(frame[target].to_numpy(dtype=float)),
                "lower": np.linspace(9.0, 19.0, len(frame)),
                "upper": np.linspace(11.0, 21.0, len(frame)),
            },
            index=frame.index,
        )

    monkeypatch.setattr(normet.foundation, "calibrate_quantiles", _fake_cal)
    monkeypatch.setattr(Chronos2Estimator, "impute", _fake_impute)

    imp_out = tmp_path / "imputed.csv"
    r_imp = runner.invoke(
        _build_cli(),
        [
            "impute",
            str(src),
            "--target",
            "PM2.5",
            "--met-vars",
            "t2m",
            "--direction",
            "forward",
            "--max-gap",
            "6",
            "--calibrate",
            "--model-path",
            str(ckpt_dir),
            "--out",
            str(imp_out),
        ],
    )
    assert r_imp.exit_code == 0, r_imp.output
    assert loaded_paths[-1] == str(ckpt_dir)
    assert seen_impute == {
        "direction": "forward",
        "max_gap": 6,
        "calibrated": True,
        "covariates": ["t2m"],
    }
    assert pd.read_csv(imp_out)["is_imputed"].sum() == 3

    # Backtest
    seen_bt: dict[str, object] = {}

    def _fake_backtest(
        self,
        frame,
        target,
        *,
        covariates=None,
        horizon=None,
        stride=None,
        calibrate=False,
        calibration_frac=0.5,
        **kw,
    ):
        seen_bt.update(
            horizon=horizon, stride=stride, calibrate=calibrate, calibration_frac=calibration_frac
        )
        return {
            "predictions": pd.DataFrame(
                {
                    "anchor_idx": [0],
                    "anchor": [frame.index[12]],
                    "date": [frame.index[12]],
                    "lead": [1],
                    "observed": [12.0],
                    "q10": [11.0],
                    "q50": [12.0],
                    "q90": [13.0],
                    "pit": [0.5],
                }
            ),
            "by_lead": pd.DataFrame(
                {
                    "lead": [1, 2],
                    "crps": [0.4, 0.5],
                    "crps_skill": [0.2, 0.15],
                    "mae": [0.3, 0.4],
                    "rmse": [0.4, 0.5],
                    "bias": [0.0, 0.0],
                    "pit_ks_stat": [0.08, 0.09],
                    "coverage_80": [0.80, 0.82],
                }
            ),
            "summary": {
                "crps": 0.45,
                "crps_skill": 0.18,
                "mae": 0.35,
                "pit_ks_stat": 0.085,
                "coverage_80": 0.81,
            },
            "calibration": None,
            "calibrated_summary": {"coverage_80": 0.83} if calibrate else None,
        }

    monkeypatch.setattr(Chronos2Estimator, "backtest", _fake_backtest)
    bt_out = tmp_path / "by_lead.csv"
    bt_preds = tmp_path / "preds.csv"
    r_bt = runner.invoke(
        _build_cli(),
        [
            "backtest",
            str(src),
            "--target",
            "PM2.5",
            "--met-vars",
            "t2m",
            "--horizon",
            "12",
            "--stride",
            "6",
            "--calibrate",
            "--calibration-frac",
            "0.4",
            "--predictions-out",
            str(bt_preds),
            "--out",
            str(bt_out),
        ],
    )
    assert r_bt.exit_code == 0, r_bt.output
    assert seen_bt == {"horizon": 12, "stride": 6, "calibrate": True, "calibration_frac": 0.4}
    assert bt_out.exists() and bt_preds.exists()
    assert "calibrated cov80=" in r_bt.output

    # Events: classical IQR + chronos adaptive
    ev_src = tmp_path / "events_in.csv"
    ev_vals = np.full(60, 10.0)
    ev_vals[30:33] = 80.0
    pd.DataFrame({"date": idx, "PM2.5": ev_vals, "t2m": 5.0}).to_csv(ev_src, index=False)

    ev_iqr = tmp_path / "events_iqr.csv"
    r_ev_iqr = runner.invoke(
        _build_cli(),
        ["events", str(ev_src), "--value-col", "PM2.5", "--method", "iqr", "--out", str(ev_iqr)],
    )
    assert r_ev_iqr.exit_code == 0, r_ev_iqr.output
    assert ev_iqr.exists()

    bad_ev = runner.invoke(
        _build_cli(),
        [
            "events",
            str(ev_src),
            "--value-col",
            "PM2.5",
            "--method",
            "iqr",
            "--adaptive",
            "--out",
            str(tmp_path / "bad_ev.csv"),
        ],
    )
    assert bad_ev.exit_code != 0

    seen_ev: dict[str, object] = {}

    def _fake_chronos_events(
        est, frame, col, *, alpha=0.05, min_length=1, direction="both", adaptive=False, **kw
    ):
        seen_ev.update(
            col=col, alpha=alpha, min_length=min_length, direction=direction, adaptive=adaptive
        )
        return pd.DataFrame(
            {
                "start": [frame.index[30]],
                "end": [frame.index[32]],
                "n": [3],
                "direction": ["high"],
                "max_surprise": [9.5],
                "mean_surprise": [8.1],
                "mean_residual": [50.0],
            }
        )

    monkeypatch.setattr(ops_mod, "detect_chronos_events", _fake_chronos_events)
    monkeypatch.setattr(normet.foundation, "detect_chronos_events", _fake_chronos_events)
    ev_ch = tmp_path / "events_chronos.csv"
    r_ev_ch = runner.invoke(
        _build_cli(),
        [
            "events",
            str(ev_src),
            "--value-col",
            "PM2.5",
            "--method",
            "chronos",
            "--covariates",
            "t2m",
            "--alpha",
            "0.01",
            "--min-length",
            "2",
            "--direction",
            "high",
            "--adaptive",
            "--model-path",
            str(ckpt_dir),
            "--out",
            str(ev_ch),
        ],
    )
    assert r_ev_ch.exit_code == 0, r_ev_ch.output
    assert seen_ev == {
        "col": "PM2.5",
        "alpha": 0.01,
        "min_length": 2,
        "direction": "high",
        "adaptive": True,
    }

    # `--model-path` on `scm` (valid on chronos, rejected otherwise)
    panel = tmp_path / "panel.csv"
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    pd.DataFrame(
        {"date": list(dates) * 2, "ID": ["T"] * 10 + ["D1"] * 10, "value": list(range(10)) * 2}
    ).to_csv(panel, index=False)
    seen_scm_cfg: list[object] = []

    def _fake_scm_cfg(**kw):
        seen_scm_cfg.append(kw.get("model_config"))
        out = pd.DataFrame(
            {"observed": [1.0], "synthetic": [1.0], "effect": [0.0]}, index=dates[:1]
        )
        out.index.name = "date"
        return out

    monkeypatch.setattr(normet, "run_scm", _fake_scm_cfg)
    r_scm = runner.invoke(
        _build_cli(),
        [
            "scm",
            str(panel),
            "--treated",
            "T",
            "--cutoff",
            "2024-01-06",
            "--backend",
            "chronos",
            "--model-path",
            str(ckpt_dir),
            "--out",
            str(tmp_path / "scm_ckpt.csv"),
        ],
    )
    assert r_scm.exit_code == 0, r_scm.output
    assert seen_scm_cfg == [{"model_path": str(ckpt_dir)}]

    bad_scm = runner.invoke(
        _build_cli(),
        [
            "scm",
            str(panel),
            "--treated",
            "T",
            "--cutoff",
            "2024-01-06",
            "--backend",
            "scm",
            "--model-path",
            str(ckpt_dir),
            "--out",
            str(tmp_path / "bad_scm.csv"),
        ],
    )
    assert bad_scm.exit_code != 0
