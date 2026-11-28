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


def test_start_tools_check_their_arguments(tools, tmp_path):
    with pytest.raises(ValueError, match="covariates"):
        tools.start_do_all("NO2", [], example="my1")
    with pytest.raises(ValueError, match="method"):
        tools.start_decompose("NO2", ["ws"], method="other", example="my1")
    with pytest.raises(ValueError, match="met_vars"):
        tools.start_deweather("NO2", [], example="my1")
    with pytest.raises(ValueError, match="method"):
        tools.start_deweather("NO2", ["ws"], example="my1", method="bogus")
    with pytest.raises(ValueError, match="estimate"):
        tools.start_deweather("NO2", ["ws"], example="my1", estimate="bogus")

    # Command assembly for start_deweather without actually running chronos-2 in CI
    tools.store.submit = lambda kind, argv, label="": {
        "id": "job-dw",
        "state": "queued",
        "command": command_line(argv),
    }
    msg = tools.start_deweather(
        "NO2",
        ["ws", "temp"],
        example="my1",
        method="icn",
        estimate="paired",
        log_transform=True,
        n_samples=3,
    )
    assert "job-dw" in msg and "deweather" in msg
    assert "--method icn" in msg and "--estimate paired" in msg
    assert "--log-transform" in msg and "--out normalised.csv" in msg

    df = _normalised(24 * 40).reset_index(names="date")
    dw_csv = tmp_path / "dw.csv"
    df.to_csv(dw_csv, index=False)
    assert "Means by month" in tools.read_result(str(dw_csv), "deweather")


def test_world_model_start_and_reading(tools, tmp_path):
    """The world-model job assembles its flags, and its output reads back."""
    with pytest.raises(ValueError, match="target"):
        tools.start_world_model([], ["ws"], example="my1")
    with pytest.raises(ValueError, match="meteorological"):
        tools.start_world_model(["NO2"], [], example="my1")

    tools.store.submit = lambda kind, argv, label="": {
        "id": "job-wm",
        "state": "queued",
        "command": command_line(argv),
    }
    msg = tools.start_world_model(
        ["NO2", "PM2.5"],
        ["ws", "blh"],
        example="my1",
        met_head="linear",
        model_backend="separable",
        n_samples=7,
        paired=False,
    )
    assert "job-wm" in msg and "world_model" in msg
    assert "--target NO2,PM2.5" in msg and "--met ws,blh" in msg
    assert "--met-head linear" in msg and "--n-samples 7" in msg
    assert "--unpaired" in msg and "--summary world_model_summary.json" in msg

    n = 24 * 3
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    frame = pd.DataFrame(
        {
            "date": idx,
            "no2_observed": np.arange(n, dtype=float) + 10.0,
            "no2_normalised": np.arange(n, dtype=float) + 8.0,
        }
    )
    csv = tmp_path / "world_model.csv"
    frame.to_csv(csv, index=False)
    (tmp_path / "world_model_summary.json").write_text(
        json.dumps(
            {
                "separability_gain": {
                    "mean_rho_lb_raw": 0.10,
                    "mean_rho_lb_latent": 0.02,
                    "rho_lb_reduction": -0.08,
                }
            }
        )
    )
    text = tools.read_result(str(csv), "world_model")
    assert "World-model de-weathering" in text
    assert "no2" in text and "Separability certificate" in text
    # A negative reduction is called out: the latent split did not beat raw chemistry.
    assert "negative" in text


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
    from normet.app.mcp_server import build_server

    server = build_server(tools)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {
        "normet_inspect_data",
        "normet_start_do_all",
        "normet_start_deweather",
        "normet_start_impute",
        "normet_start_backtest",
        "normet_start_events",
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
    assert text.startswith("Trend analysis on MY1's emission part (normet.trend):")
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
    assert said.startswith("Trend analysis on emi_total (normet.trend):")
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
    assert said.startswith("Segmented trend analysis on long-term (normet.detect_trend_breaks):")
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
        "Segmented trend analysis on MY1's emission part (normet.detect_trend_breaks):"
    )
    assert "2024-01-09" in two and "SSE improvement" in two
    assert "p (block bootstrap of the search)" in two and "0.004" in two
    one = readings.trend_breaks_reading(mk(1, 0.42), name="flat")
    assert "one regime" not in one
    assert "best candidate cut" in one and "0.42" in one


# -- llm: context policy, key lookup, certificates -----------------------------------


def _rounds(n, big=400):
    """A chat of n tool rounds: inspect_data first, then `n-1` calls of `results`."""
    from normet.agent import llm

    msgs = [{"role": "user", "content": "go"}]
    for i in range(n):
        name = "inspect_data" if i == 0 else "results"
        msgs.append(
            {
                "role": "assistant",
                "text": "",
                "calls": [llm.Call(f"c{i}", name, {})],
                "reasoning": "thinking",
            }
        )
        msgs.append({"role": "tool", "results": [{"id": f"c{i}", "text": f"R{i} " + "x" * big}]})
    return msgs


def test_context_policy_folds_old_rounds_and_pins_readings():
    from normet.agent import llm

    msgs = _rounds(6)
    out = llm.ContextPolicy(keep=2, pinned=("inspect_data", "results")).prepare(msgs)
    texts = {r["id"]: r["text"] for m in out if m["role"] == "tool" for r in m["results"]}
    # the latest `results` and the one `inspect_data` stay whole; other old ones fold to 160
    assert texts["c0"] == msgs[2]["results"][0]["text"]
    assert texts["c5"] == msgs[-1]["results"][0]["text"]
    for cid in ("c1", "c2"):
        assert texts[cid].endswith(llm.ContextPolicy.FOLDED)
        assert len(texts[cid]) == len(llm.ContextPolicy.FOLDED) + len(texts[cid][:160].rstrip())
        assert len(texts[cid]) < 160 + len(llm.ContextPolicy.FOLDED) + 1
    # reasoning of old rounds is left out, the recent keeps it
    assert "reasoning" not in out[1] and out[-2]["reasoning"] == "thinking"
    # the messages themselves are untouched
    assert msgs[4]["results"][0]["text"].endswith("x" * 400)


def test_context_policy_unpinned_and_failed_results():
    from normet.agent import llm

    msgs = _rounds(5)
    out = llm.ContextPolicy(keep=2).prepare(msgs)
    assert out[2]["results"][0]["text"].endswith(llm.ContextPolicy.FOLDED)
    short = _rounds(2)
    assert llm.ContextPolicy(keep=2).prepare(short) is short
    # a failed pinned result is not the pinned one: the earlier whole one is
    msgs = _rounds(6)
    msgs[-1]["results"][0]["error"] = True
    out = llm.ContextPolicy(keep=1, pinned=("results",)).prepare(msgs)
    texts = {r["id"]: r["text"] for m in out if m["role"] == "tool" for r in m["results"]}
    assert texts["c4"] == msgs[-3]["results"][0]["text"]


def test_converse_sends_the_folded_payload_but_keeps_the_messages():
    from normet.agent import llm

    seen = []
    n = {"i": 0}

    def policy(messages, tools):
        seen.append(list(messages))
        n["i"] += 1
        if n["i"] > 6:
            return llm.Reply("done", [], {})
        return llm.Reply("", [llm.Call(f"k{n['i']}", "big", {})], {"input": 1, "output": 1})

    big = llm.as_tool(lambda: "y" * 500, name="big", description="a long reading")
    messages = [{"role": "user", "content": "go"}]
    turn = llm.converse(llm.FakeModel(policy), [big], "sys", messages)
    assert turn.stopped == "answered"
    last = [r["text"] for m in seen[-1] if m["role"] == "tool" for r in m["results"]]
    assert last[0].endswith(llm.ContextPolicy.FOLDED) and last[-1] == "y" * 500
    full = [r["text"] for m in messages if m["role"] == "tool" for r in m["results"]]
    assert all(t == "y" * 500 for t in full)


def test_default_max_tokens_and_zshrc_key(tmp_path, monkeypatch):
    from normet.agent import llm

    rc = tmp_path / "zshrc"
    rc.write_text(
        "export OTHER=1\nexport ANTHROPIC_API_KEY=old\n"
        "export ANTHROPIC_API_KEY='sk-new'  # latest\n"
    )
    assert llm._zshrc_key("ANTHROPIC_API_KEY", str(rc)) == "sk-new"
    assert llm._zshrc_key("MISSING_KEY", str(rc)) is None
    assert llm._zshrc_key("X", str(tmp_path / "nope")) is None
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_zshrc_key", lambda env, path="": "from-zshrc")
    m = llm.Model("anthropic:claude-sonnet-5-5")
    assert m.key == "from-zshrc" and m.max_tokens == 16384
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert llm.Model("anthropic:claude-sonnet-5-5").key == "from-env"


def test_anthropic_round_caches_its_prefix_and_counts_the_cache(monkeypatch):
    """The system prompt, the tools and the history's end carry cache_control.

    Without it every round of a tool loop buys the whole prefix again; with it a later
    round reads it at 0.1x. The usage count adds the write and the read back to input,
    or the panel's count falls as the cache starts to work.
    """
    from normet.agent import llm

    seen = {}

    def post(url, body, headers, **kwargs):
        seen["body"] = body
        return {
            "content": [{"type": "text", "text": "hi"}],
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 1000,
                "output_tokens": 5,
            },
        }

    monkeypatch.setattr(llm, "_post", post)
    tool = llm.Tool("t", "does t", {"type": "object"}, lambda: None)
    model = llm.Model("anthropic:claude-sonnet-5-5", api_key="k", effort="high")
    reply = model.chat(
        "SYS", [{"role": "user", "content": "go"}, {"role": "user", "content": "x"}], [tool]
    )

    body = seen["body"]
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"] == {"effort": "high"}
    assert reply.usage["input"] == 1110


def test_openai_compatible_round_sends_the_reasoning_effort(monkeypatch):
    from normet.agent import llm

    seen = {}

    def post(url, body, headers, **kwargs):
        seen["body"] = body
        return {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }

    monkeypatch.setattr(llm, "_post", post)
    model = llm.Model("deepseek:deepseek-chat", api_key="k", effort="xhigh")
    model.chat("SYS", [{"role": "user", "content": "go"}], [])
    assert seen["body"]["reasoning_effort"] == "xhigh"
    # auto sends nothing
    model = llm.Model("deepseek:deepseek-chat", api_key="k")
    model.chat("SYS", [{"role": "user", "content": "go"}], [])
    assert "reasoning_effort" not in seen["body"]


def test_reasoning_efforts_follow_the_provider_and_the_model_family():
    from normet.agent import llm

    assert llm.reasoning_efforts("anthropic", "claude-sonnet-5-5")[-1] == "max"
    # DeepSeek's template refuses "medium"
    assert llm.reasoning_efforts("deepseek", "deepseek-chat") == ("low", "high", "xhigh", "max")
    # a server of the analyst's own reads the family off the model's name
    llm.register_provider("lab", "http://127.0.0.1:8080/v1")
    try:
        assert llm.reasoning_efforts("lab", "deepseek-chat") == llm.REASONING_EFFORTS["deepseek"]
        assert llm.reasoning_efforts("lab", "qwen3") == ("low", "medium", "high")
    finally:
        llm.forget_provider("lab")


def test_ca_bundle_falls_back_to_certifi(monkeypatch):
    import ssl
    import types

    from normet.agent import llm

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(
        ssl, "get_default_verify_paths", lambda: types.SimpleNamespace(cafile="/no/such/file")
    )
    certifi = pytest.importorskip("certifi")
    llm.ca_bundle()
    assert os.environ["SSL_CERT_FILE"] == certifi.where()
    monkeypatch.setenv("SSL_CERT_FILE", "/mine.pem")
    llm.ca_bundle()
    assert os.environ["SSL_CERT_FILE"] == "/mine.pem"


def test_chronos_impute_backtest_events_tools_and_readings(tools, tmp_path):
    """New start_impute / start_backtest / start_events, readings, and imputation figure."""
    import pandas as pd

    ckpt = tmp_path / "ckpt_dir"
    ckpt.mkdir()

    # Sandbox enforcement on model_path
    with pytest.raises(PermissionError):
        tools.start_impute("NO2", example="my1", model_path="/etc")

    msg_imp = tools.start_impute(
        "NO2",
        met_vars=["ws", "temp"],
        example="my1",
        direction="forward",
        max_gap=12,
        calibrate=True,
        model_path=str(ckpt),
        label="gap-fill",
    )
    assert "impute" in msg_imp and "--direction forward" in msg_imp
    assert "--calibrate" in msg_imp and "--out imputed.csv" in msg_imp

    msg_bt = tools.start_backtest(
        "NO2",
        met_vars=["ws"],
        example="my1",
        horizon=24,
        stride=12,
        calibrate=True,
        calibration_frac=0.4,
    )
    assert "backtest" in msg_bt and "--horizon 24" in msg_bt and "--out backtest.csv" in msg_bt

    msg_ev = tools.start_events(
        "NO2",
        method="chronos",
        covariates=["ws"],
        example="my1",
        alpha=0.01,
        min_length=3,
        direction="high",
        adaptive=True,
    )
    assert "events" in msg_ev and "--method chronos" in msg_ev
    assert "--adaptive" in msg_ev and "--out events.csv" in msg_ev

    with pytest.raises(ValueError, match="adaptive"):
        tools.start_events("NO2", method="iqr", example="my1", adaptive=True)

    # Readings + figure
    dates = pd.date_range("2024-01-01", periods=12, freq="h")
    imp_df = pd.DataFrame(
        {
            "date": dates,
            "observed": [float(i) if i not in (4, 5) else float("nan") for i in range(12)],
            "imputed": [float(i) for i in range(12)],
            "is_imputed": [i in (4, 5) for i in range(12)],
            "lower": [float(i) - 1.0 for i in range(12)],
            "upper": [float(i) + 1.0 for i in range(12)],
        }
    )
    imp_csv = tmp_path / "imputed.csv"
    imp_df.to_csv(imp_csv, index=False)
    imp_txt = tools.read_result(str(imp_csv), "impute")
    assert "imputed spans" in imp_txt and "mean interval width" in imp_txt
    caption, png = tools.figure(str(imp_csv), "imputation")
    assert png[:4] == b"\x89PNG" and "Chronos-2" in caption

    bt_df = pd.DataFrame(
        {
            "lead": [1, 2, 3],
            "crps": [0.5, 0.6, 0.7],
            "crps_skill": [0.3, 0.25, 0.2],
            "mae": [0.4, 0.5, 0.6],
            "coverage_80": [0.80, 0.81, 0.79],
        }
    )
    bt_csv = tmp_path / "backtest.csv"
    bt_df.to_csv(bt_csv, index=False)
    bt_txt = tools.read_result(str(bt_csv), "backtest")
    assert "mean CRPS" in bt_txt and "mean cov80" in bt_txt

    ev_df = pd.DataFrame(
        {
            "start": ["2024-01-02 01:00"],
            "end": ["2024-01-02 04:00"],
            "n": [4],
            "direction": ["high"],
            "max_surprise": [8.5],
            "mean_residual": [22.0],
        }
    )
    ev_csv = tmp_path / "events.csv"
    ev_df.to_csv(ev_csv, index=False)
    ev_txt = tools.read_result(str(ev_csv), "events")
    assert "1 episode(s)" in ev_txt and "max_surprise" in ev_txt


def test_job_store_launched_without_started_is_running_and_not_pruned(tmp_path):
    """A newly launched job without started.json must look running to another store and survive prune()."""
    import subprocess
    import sys
    import time

    home = str(tmp_path / "home")
    server = JobStore(home=home)
    other = JobStore(home=home)

    job_id = "20260101-000000.000001-abcd"
    folder = tmp_path / "home" / "jobs" / job_id
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(
        json.dumps(
            {"id": job_id, "kind": "do_all", "argv": [], "label": "", "folder": str(folder)}
        ),
        encoding="utf-8",
    )

    # 1. Un-PID'd fresh `launched` marker inside LAUNCH_GRACE
    (folder / "launched").write_text(json.dumps({"launched": time.time()}), encoding="utf-8")
    assert other.state(job_id) == "running"
    assert other.prune(dry_run=False)["removed"] == []
    assert folder.is_dir()

    # 2. Live child PID in `launched` even without `started.json` and with old mtime
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        (folder / "launched").write_text(
            json.dumps({"pid": proc.pid, "launched": time.time() - 3600}),
            encoding="utf-8",
        )
        old = time.time() - 3600
        os.utime(folder / "launched", (old, old))
        assert other.state(job_id) == "running"
        assert other.prune(dry_run=False)["removed"] == []
        assert folder.is_dir()
    finally:
        proc.terminate()
        proc.wait(timeout=5)

    # Once the child is gone and exit.json was never written, it is failed and prunable.
    assert other.state(job_id) == "failed"
    assert server.state(job_id) == "failed"
    assert other.prune(dry_run=False)["removed"] == [job_id]
    assert not folder.exists()


def test_read_result_and_job_status_handle_all_seven_extended_job_kinds(tools, tmp_path):
    """Every job kind in _OUTPUTS must produce a real reading when done (never ValueError)."""
    from normet.agent.tools import _OUTPUTS

    dates = pd.date_range("2024-01-01", periods=6, freq="h")

    # 1. met_features
    mf_csv = tmp_path / "met_features.csv"
    pd.DataFrame(
        {
            "date": dates,
            "ws": [2.0] * 6,
            "u_wind": [1.0] * 6,
            "v_wind": [-1.0] * 6,
            "ventilation_coef": [1000.0] * 6,
        }
    ).to_csv(mf_csv, index=False)
    assert "Derived features (3)" in tools.read_result(str(mf_csv), "met_features")

    # 2. normalise_ox
    ox_csv = tmp_path / "ox_species.csv"
    pd.DataFrame(
        {
            "date": dates,
            "O3_observed": [40.0] * 6,
            "O3_normalised": [42.0] * 6,
            "NO2_observed": [20.0] * 6,
            "NO2_normalised": [18.0] * 6,
            "Ox_observed": [60.0] * 6,
            "Ox_normalised": [60.0] * 6,
            "NOx_observed": [30.0] * 6,
            "NOx_normalised": [27.0] * 6,
            "O3_titration_effect": [2.0] * 6,
            "f_no2_normalised": [0.3] * 6,
        }
    ).to_csv(ox_csv, index=False)
    ox_txt = tools.read_result(str(ox_csv), "normalise_ox")
    assert "Conserved Ox–NOx deweathering" in ox_txt and "titration effect" in ox_txt

    # 3. normalise_regime
    reg_csv = tmp_path / "regime_normalised.csv"
    pd.DataFrame(
        {
            "date": dates,
            "observed": [30.0] * 6,
            "normalised": [28.0] * 6,
            "normalised_unconditioned": [26.0] * 6,
            "precursor_regime_effect": [2.0] * 6,
        }
    ).to_csv(reg_csv, index=False)
    reg_txt = tools.read_result(str(reg_csv), "normalise_regime")
    assert "Means by month:" in reg_txt and "precursor regime effect" in reg_txt

    # 4. icn_decompose
    icn_csv = tmp_path / "icn_decomposition.csv"
    pd.DataFrame(
        {"date": dates, "observed": [10.0] * 6, "emi_total": [8.0] * 6, "ws": [2.0] * 6}
    ).to_csv(icn_csv, index=False)
    assert "3 components" in tools.read_result(str(icn_csv), "icn_decompose")

    # 5. shapley_interactions (non-empty and empty)
    si_csv = tmp_path / "interactions.csv"
    pd.DataFrame(
        {
            "term": ["ws", "ws:t2m"],
            "kind": ["main", "interaction"],
            "mean": [1.5, -0.3],
            "std": [0.4, 0.2],
            "mean_abs": [1.5, 0.3],
            "max_abs": [2.1, 0.6],
        }
    ).to_csv(si_csv, index=False)
    si_txt = tools.read_result(str(si_csv), "shapley_interactions")
    assert "1 main effect(s), 1 pairwise interaction(s)" in si_txt and "ws:t2m" in si_txt

    si_empty = tmp_path / "interactions_empty.csv"
    pd.DataFrame(columns=["term", "kind", "mean", "std", "mean_abs", "max_abs"]).to_csv(
        si_empty, index=False
    )
    assert "0 terms" in tools.read_result(str(si_empty), "shapley_interactions")

    # 6. benchmark (with sibling metadata.json and scores.json)
    bdir = tmp_path / "benchmark"
    bdir.mkdir()
    pd.DataFrame({"date": dates, "value": [12.0] * 6, "ws": [3.0] * 6}).to_csv(
        bdir / "benchmark.csv", index=False
    )
    (bdir / "metadata.json").write_text(json.dumps({"family": "F1_additive", "n_days": 7}))
    (bdir / "scores.json").write_text(json.dumps({"rmse": 0.42}))
    b_txt = tools.read_result(str(bdir / "benchmark.csv"), "benchmark")
    assert "F1_additive" in b_txt and "0.42" in b_txt and "benchmark.csv: 6 rows" in b_txt

    # 7. invert_multisite (JSON summary + sibling site_metrics.csv)
    idir = tmp_path / "inversion"
    idir.mkdir()
    (idir / "summary.json").write_text(
        json.dumps(
            {
                "alpha": 1.2,
                "spatial_alpha": 0.5,
                "r2": 0.87,
                "rmse": 2.1,
                "n_obs": 120,
                "intercepts": {"S1": 1.5, "S2": 2.0},
            }
        )
    )
    pd.DataFrame({"site": ["S1", "S2"], "r2": [0.88, 0.85]}).to_csv(
        idir / "site_metrics.csv", index=False
    )
    inv_txt = tools.read_result(str(idir / "summary.json"), "invert_multisite")
    assert (
        "Multi-site footprint inversion" in inv_txt
        and "0.87" in inv_txt
        and "Site metrics:" in inv_txt
    )

    # End-to-end job_status() check across all 7 kinds
    files_by_kind = {
        "met_features": (mf_csv, "Derived features"),
        "normalise_ox": (ox_csv, "Conserved Ox–NOx"),
        "normalise_regime": (reg_csv, "precursor regime effect"),
        "icn_decompose": (icn_csv, "3 components"),
        "shapley_interactions": (si_csv, "1 main effect(s)"),
        "benchmark": (bdir / "benchmark.csv", "F1_additive"),
        "invert_multisite": (idir / "summary.json", "Multi-site footprint inversion"),
    }
    for idx, (kind, (src_file, expected_phrase)) in enumerate(files_by_kind.items(), start=1):
        jid = f"20260101-000001.{idx:06d}-{kind[:4]}"
        jdir = tmp_path / "home" / "jobs" / jid
        jdir.mkdir(parents=True)
        (jdir / "job.json").write_text(
            json.dumps(
                {
                    "id": jid,
                    "kind": kind,
                    "argv": [kind],
                    "command": f"normet {kind}",
                    "folder": str(jdir),
                }
            ),
            encoding="utf-8",
        )
        (jdir / "exit.json").write_text(json.dumps({"state": "done", "code": 0}), encoding="utf-8")
        dest = jdir / _OUTPUTS[kind]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")
        if kind == "benchmark":
            (dest.parent / "metadata.json").write_text(
                (src_file.parent / "metadata.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        status_txt = tools.job_status(jid)
        assert "could not read its output" not in status_txt
        assert expected_phrase in status_txt


def test_data_reading_on_header_only_csv(tools, tmp_path):
    """A 0-row CSV with headers must not claim columns are 'Not numeric'."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("date,NO2,ws\n", encoding="utf-8")
    text = tools.inspect_data(data=str(empty_csv))
    assert "empty.csv: 0 rows × 3 columns." in text
    assert "Not numeric" not in text
    assert "Dates NaT" not in text
    assert "No rows to type or summarise (columns: date, NO2, ws)." in text
