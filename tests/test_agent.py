"""normet.agent: the readings, the job store and the tools an assistant drives normet with."""

from __future__ import annotations

import asyncio
import json
import os

import numpy as np
import pandas as pd
import pytest

from normet.agent import readings
from normet.agent.jobs import JobStore, command_line
from normet.agent.tools import Tools


@pytest.fixture
def tools(tmp_path):
    return Tools(store=JobStore(home=str(tmp_path / "home")), roots=[str(tmp_path)])


# -- readings -----------------------------------------------------------------------


def test_num_and_table():
    assert readings.num(0.000123456) == "1.235e-04"
    assert readings.num(12.3456) == "12.35"
    assert readings.num(float("nan")) == "—"
    assert readings.num("x") == "x"
    text = readings.table(["a", "bb"], [["x", 1.0], ["yy", 22.5]])
    assert text.splitlines()[1] == "--  ----"
    assert text.splitlines()[3].startswith("yy  22.5")


def test_human_step():
    assert readings.human_step(pd.Timedelta("1h")) == "1h"
    assert readings.human_step(pd.Timedelta("30min")) == "30min"
    assert readings.human_step(pd.Timedelta("1D")) == "1 day"
    assert readings.human_step(pd.Timedelta("7D")) == "7 days"
    assert readings.human_step(pd.Timedelta("90s")) == "90s"


def test_data_reading_counts_gaps_and_correlations(synthetic_aq):
    df = synthetic_aq.drop(index=range(10, 20)).reset_index(drop=True)
    df.loc[0:4, "t2m"] = np.nan
    df["site"] = "A"
    text = readings.data_reading(df, target="PM2.5", name="aq.csv")
    assert text.startswith(f"aq.csv: {len(df):,} rows")
    assert "Usual step 1h" in text
    assert "(10 missing)" in text
    assert "5 (0.3%)" in text  # t2m's missing values
    assert "r with PM2.5" in text and "pollutant (target)" in text
    assert "Not numeric: site." in text
    assert "No column 'nope'." in readings.data_reading(df, target="nope")


def test_data_reading_without_dates():
    text = readings.data_reading(pd.DataFrame({"x": [1.0, 2.0]}))
    assert "No 'date' column" in text


def test_panel_is_recognised(scm_panel):
    assert readings.panel_units(scm_panel) == ("ID", 7)
    assert "A panel: 7 units in column 'ID'" in readings.data_reading(scm_panel)
    assert readings.panel_units(scm_panel[scm_panel.ID == "T"]) is None


def _normalised(n=24 * 400):
    idx = pd.date_range("2021-01-01", periods=n, freq="h")
    norm = np.full(n, 10.0)
    return pd.DataFrame({"observed": norm + np.sin(np.arange(n)), "normalised": norm}, index=idx)


def test_normalised_reading_by_month_and_year():
    short = readings.normalised_reading(_normalised(24 * 60))
    assert "Means by month:" in short and "2021-02" in short
    long = _normalised(24 * 365 * 3 + 24)
    text = readings.normalised_reading(long.reset_index(names="date"))
    assert "Means by year:" in text and "2023" in text
    assert "mean normalised 10" in text
    with pytest.raises(ValueError, match="observed"):
        readings.normalised_reading(long.rename(columns={"observed": "obs"}))
    with pytest.raises(ValueError, match="date"):
        readings.normalised_reading(long.reset_index(drop=True))


def test_decomposition_and_cv_readings():
    df = pd.DataFrame({"date": pd.date_range("2021", periods=3), "trend": [1.0, 2.0, 3.0]})
    text = readings.decomposition_reading(df)
    assert "1 components" in text and "trend" in text
    cv = pd.DataFrame({"fold": [0, 1], "RMSE": [1.0, 3.0]})
    assert "Means over the folds: RMSE 2." in readings.cv_reading(cv)


def test_scm_reading(scm_panel):
    from normet import run_scm

    out = run_scm(
        df=scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=None,
    )
    text = readings.scm_reading(out.reset_index(), "2023-05-01")
    assert "pre-period RMSE" in text and "ATT (mean effect after the cutoff)" in text


def test_model_reading_and_importance(synthetic_aq):
    pytest.importorskip("lightgbm")
    from normet import do_all

    out, model, df_prep = do_all(
        synthetic_aq.iloc[: 24 * 30],
        target="PM2.5",
        covariates=["t2m", "blh", "u10", "v10"],
        backend="lightgbm",
        n_samples=5,
    )
    text = readings.model_reading(df_prep, model)
    assert text.index("testing") < text.index("training")
    assert "Feature importance" in text
    assert readings.feature_importance(object()) is None


# -- the job store ------------------------------------------------------------------


def test_command_line_quotes():
    assert command_line(["scm", "a b.csv", "--treated", "T"]) == "normet scm 'a b.csv' --treated T"


def test_jobs_run_queue_and_record(tmp_path, scm_panel):
    pytest.importorskip("click")
    store = JobStore(home=str(tmp_path))
    data = tmp_path / "panel.csv"
    scm_panel.to_csv(data, index=False)
    argv = ["scm", str(data), "--treated", "T", "--cutoff", "2023-05-01", "--out", "scm.csv"]
    first = store.submit("scm", argv, label="one")
    second = store.submit(
        "scm", ["scm", str(data), "--treated", "nope", "--cutoff", "2023-05-01", "--out", "x.csv"]
    )
    assert second["state"] == "queued" and second["ahead"] == 1
    done = store.wait(first["id"], timeout=120)
    assert done["state"] == "done" and done["exit_code"] == 0
    assert any(f.endswith("scm.csv") for f in done["files"])
    with open(os.path.join(done["folder"], "job.json")) as fh:
        assert json.load(fh)["command"].startswith("normet scm ")
    failed = store.wait(second["id"], timeout=120)
    assert failed["state"] == "failed"
    assert store.text(second["id"], "stderr.txt")
    with pytest.raises(KeyError):
        store.status("../escape")


def test_cancel_a_queued_job(tmp_path):
    store = JobStore(home=str(tmp_path))
    store.pump = lambda: None  # nothing starts: the job stays queued
    job = store.submit("cv", ["cv", "x.csv", "--out", "cv.csv"])
    assert job["state"] == "queued"
    assert store.cancel(job["id"])["state"] == "cancelled"


# -- the tools ----------------------------------------------------------------------


def test_paths_stay_inside_the_roots(tools, tmp_path):
    inside = tmp_path / "a.csv"
    inside.write_text("date,x\n2021-01-01,1\n")
    assert tools.path(str(inside)) == str(inside)
    with pytest.raises(PermissionError):
        tools.path("/etc/hosts")
    with pytest.raises(FileNotFoundError):
        tools.path(str(tmp_path / "missing.csv"))
    with pytest.raises(ValueError, match="one of"):
        tools.inspect_data()
    with pytest.raises(ValueError, match="no example"):
        tools.example_path("nope")


def test_examples_and_inspect(tools):
    assert "my1:" in tools.list_examples()
    path = tools.example_path("my1")
    assert os.path.exists(path) and tools.example_path("my1") == path
    text = tools.inspect_data(example="my1", target="NO2")
    assert text.startswith("my1.csv:") and "r with NO2" in text


def test_options_skip_empty_values():
    assert Tools._options(target="NO2", covariates=["a", "b"], seed=None, label="") == [
        "--target",
        "NO2",
        "--covariates",
        "a,b",
    ]


def test_scm_job_through_the_tools(tools, scm_panel, tmp_path):
    pytest.importorskip("click")
    data = tmp_path / "panel.csv"
    scm_panel.to_csv(data, index=False)
    said = tools.start_scm("T", "2023-05-01", data=str(data), label="ulez")
    assert said.startswith("Started job ")
    job_id = tools.store.ids()[-1]
    tools.store.wait(job_id, timeout=120)
    status = tools.job_status(job_id)
    assert "done" in status and "pre-period RMSE" in status
    assert "ulez" in tools.list_jobs()
    out = os.path.join(tools.store.root, job_id, "scm.csv")
    caption, png = tools.figure(out, "scm", cutoff="2023-05-01")
    assert png[:4] == b"\x89PNG" and "2023-05-01" in caption
    with pytest.raises(ValueError, match="cutoff"):
        tools.read_result(out, "scm")
    with pytest.raises(ValueError, match="kind"):
        tools.read_result(out, "other")
    with pytest.raises(ValueError, match="kind is one of"):
        tools.figure(out, "other")


def test_start_tools_check_their_arguments(tools):
    with pytest.raises(ValueError, match="covariates"):
        tools.start_do_all("NO2", [], example="my1")
    with pytest.raises(ValueError, match="method"):
        tools.start_decompose("NO2", ["ws"], method="other", example="my1")


def test_a_run_read_drawn_and_reported(tools, synthetic_aq, tmp_path):
    pytest.importorskip("lightgbm")
    from normet import do_all, make_run, save_run

    out, model, df_prep = do_all(
        synthetic_aq.iloc[: 24 * 30],
        target="PM2.5",
        covariates=["t2m", "blh", "u10", "v10"],
        backend="lightgbm",
        n_samples=5,
    )
    run = make_run(
        out,
        model=model,
        df_prep=df_prep,
        kind="do_all",
        config={"target": "PM2.5", "backend": "lightgbm"},
    )
    save_run(run, tmp_path / "run")
    text = tools.read_run(str(tmp_path / "run"))
    assert '"target": "PM2.5"' in text and "Model fit" in text and "Means by month" in text
    for kind in ("normalised", "model", "importance"):
        caption, png = tools.figure(str(tmp_path / "run.joblib"), kind)
        assert png[:4] == b"\x89PNG" and caption
    assert tools.write_report(str(tmp_path / "run.joblib")).startswith("Wrote ")
    assert (tmp_path / "run_report.html").exists()
    out.reset_index().to_csv(tmp_path / "normalised.csv", index=False)
    assert "Means by month" in tools.read_result(str(tmp_path / "normalised.csv"), "normalised")


# -- the MCP server -----------------------------------------------------------------


def test_mcp_server_lists_and_calls_tools(tools):
    pytest.importorskip("mcp")
    from normet.mcp_server import build_server

    server = build_server(tools)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {
        "normet_inspect_data",
        "normet_start_do_all",
        "normet_job_status",
        "normet_figure",
    } <= names
    said = asyncio.run(server.call_tool("normet_list_examples", {}))
    assert "my1" in str(said)
    with pytest.raises(Exception, match="no example"):
        asyncio.run(server.call_tool("normet_inspect_data", {"example": "nope"}))


# -- the trend ------------------------------------------------------------------------


def test_trend_reading_says_which_p_to_quote():
    from normet.analysis.trend import TrendResult, TrendSlope, TrendTest

    def mk(ar: bool, p_boot: float | None) -> TrendResult:
        return TrendResult(
            name="demo",
            n=400,
            n_dropped=0,
            test=TrendTest(S=120, var_S=1.0, z=4.0, p=0.001, n=400),
            slope=TrendSlope(slope=0.02, ci_low=0.01, ci_high=0.03, n=400, time_unit="day"),
            ar1_acf=0.5,
            ar1_p=1e-6,
            ar_autocorrelated=ar,
            boot_slope_se=(0.002 if p_boot is not None else None),
            boot_ci_low=(0.012 if p_boot is not None else None),
            boot_ci_high=(0.028 if p_boot is not None else None),
            p_wald_boot=p_boot,
            block_length=(16 if p_boot is not None else None),
            n_bootstrap=(500 if p_boot is not None else None),
            notes=[],
        )

    text = readings.trend_reading(mk(True, 0.002), name="MY1's emission part")
    assert text.startswith("Trend test on MY1's emission part (normet.trend):")
    assert "Mann-Kendall S" in text and "slope (per day)" in text and "increasing" in text
    assert "p (Wald on block bootstrap)" in text
    assert "0.002" in text
    plain = readings.trend_reading(mk(False, None))
    assert "p (Wald on block bootstrap)" not in plain
    none_boot = readings.trend_reading(mk(True, None))
    assert "bootstrap was unavailable" in none_boot


def test_headless_trend_runs_on_a_result_table(tools, tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 24 * 45
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    frame = pd.DataFrame({"date": idx, "emi_total": 0.01 * np.arange(n) + rng.normal(0, 0.3, n)})
    path = str(tmp_path / "decomposition.csv")
    frame.to_csv(path, index=False)
    said = tools.trend(path)
    assert said.startswith("Trend test on emi_total (normet.trend):")
    assert "direction (5%)" in said.replace("  ", " ") and "increasing" in said
    with pytest.raises(ValueError, match="no 'nope' column"):
        tools.trend(path, column="nope")


def test_headless_trend_breaks_runs_on_a_result_table(tools, tmp_path):
    """Where the headless segment fit reads a table: the long-term as its separated
    component (the default), and the columns it names, with no long-term there a
    no-nonsense error."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 24 * 45
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    b = n // 2
    emi = np.empty(n)
    emi[:b] = 0.02 * np.arange(b) + rng.normal(0, 0.3, b) + 10
    emi[b:] = 0.005 * np.arange(n - b) + rng.normal(0, 0.3, n - b)
    dux = np.empty(n)
    dux[:b] = 0.01 * np.arange(b) + rng.normal(0, 0.2, b)
    dux[b:] = 0.01 * np.arange(n - b) + rng.normal(0, 0.2, n - b) + 4.0  # a step: the break
    frame = pd.DataFrame(
        {
            "date": idx,
            "emi_total": emi,
            "date_unix": dux,
            "emi_base": 8.0,
        }
    )
    path = str(tmp_path / "decomp2.csv")
    frame.to_csv(path, index=False)
    said = tools.trend_breaks(path)
    assert said.startswith("Segmented trend test on long-term (normet.detect_trend_breaks):")
    assert "p (block bootstrap of the search)" in said
    on_emission = tools.trend_breaks(path, column="emission")
    assert "break" in on_emission or "one regime" in on_emission
    plain = pd.DataFrame({"date": idx, "a": np.arange(n, dtype=float)})
    p2 = str(tmp_path / "plain.csv")
    plain.to_csv(p2, index=False)
    with pytest.raises(ValueError, match="no 'date_unix' column"):
        tools.trend_breaks(p2)
    with pytest.raises(ValueError, match="no 'nope' column"):
        tools.trend_breaks(path, column="nope")


def test_trend_breaks_reading_says_where_and_whose_p():
    import pandas as pd

    from normet.analysis.trend import TrendBreaks, TrendSlope

    def mk(n_segments: int, p: float) -> TrendBreaks:
        idx = pd.date_range("2024-01-01", periods=400, freq="h")
        return TrendBreaks(
            name="demo",
            n=400,
            n_dropped=0,
            time_unit="day",
            starts=[idx[0]] + [idx[200]] * (n_segments - 1),
            ends=[idx[199]] * (n_segments - 1) + [idx[-1]],
            segments=[
                TrendSlope(
                    slope=0.05,
                    ci_low=0.04,
                    ci_high=0.06,
                    n=200,
                    time_unit="day",
                    slope_per_year=18.0,
                    ci_low_per_year=14.0,
                    ci_high_per_year=22.0,
                )
                for _ in range(n_segments)
            ],
            n_segments=n_segments,
            best_break=idx[200],
            best_split_pct=40.0,
            sse_single=100.0,
            sse_segmented=60.0,
            improvement_pct=40.0,
            bic_single=500.0,
            bic_segmented=480.0,
            p_value=p,
            significant=p < 0.05,
            alpha=0.05,
            max_segments=2,
            min_segment=28,
            block_length=16,
            n_bootstrap=500,
            notes=[],
        )

    two = readings.trend_breaks_reading(mk(2, 0.004), name="MY1's emission part")
    assert two.startswith(
        "Segmented trend test on MY1's emission part (normet.detect_trend_breaks):"
    )
    assert "2024-01-09" in two and "SSE improvement" in two
    assert "p (block bootstrap of the search)" in two and "0.004" in two
    one = readings.trend_breaks_reading(mk(1, 0.42), name="flat")
    assert "one regime" not in one
    assert "best candidate cut" in one and "0.42" in one
