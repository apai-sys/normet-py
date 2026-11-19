"""NormetRun + save_run/load_run round-trips."""

import json
from pathlib import Path

import pandas as pd

from normet.utils.provenance import NormetRun, load_run, make_run, save_run


def test_make_run_attaches_metadata(synthetic_aq):
    run = make_run(
        result=synthetic_aq.head(),
        kind="do_all",
        config={"n_samples": 100, "backend": "flaml"},
        df=synthetic_aq,
        seed=42,
    )
    assert isinstance(run, NormetRun)
    m = run.metadata
    assert m["kind"] == "do_all"
    assert m["seed"] == 42
    assert "normet_version" in m
    assert "data_hash" in m and len(m["data_hash"]) == 40  # sha1 hex length
    assert "config_hash" in m and len(m["config_hash"]) == 40


def test_save_load_round_trip(tmp_path: Path, synthetic_aq):
    run = make_run(
        result=synthetic_aq.head(10),
        kind="normalise",
        config={"n_samples": 50},
        df=synthetic_aq,
        seed=7,
    )
    paths = save_run(run, tmp_path / "rundir" / "run1")
    assert Path(paths["artifact"]).exists()
    assert Path(paths["metadata"]).exists()

    side = json.loads(Path(paths["metadata"]).read_text())
    assert side["seed"] == 7
    assert side["kind"] == "normalise"

    back = load_run(tmp_path / "rundir" / "run1")
    assert isinstance(back, NormetRun)
    pd.testing.assert_frame_equal(back.result, run.result)
    assert back.metadata["seed"] == 7
