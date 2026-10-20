"""A session script read back (File -> Load Session Script): what its calls set.

The script the command log writes (:mod:`.commandlog`) is Python, run as it is to
repeat the session. Before it runs, its calls are read as written -- without running
anything -- for what the panel should show afterwards: the target and predictors the
model was trained on, each step's settings. Each known call's literal arguments are
kept by the function's name, as the window keeps what it ran (the Methods paragraph is
written from them); an argument that is a name (``df``, ``model``) is kept as None.
"""

from __future__ import annotations

import ast
from typing import Any

#: The functions a session script calls, as the command log writes them.
KNOWN = (
    "build_model",
    "load_estimator",
    "normalise",
    "deweather",
    "decompose",
    "rolling",
    "compute_multiscale",
    "pdp",
    "finetune",
)

#: What the window's tasks return, by the name the script assigns it to.
RESULTS = {
    "normalised": "normalise",
    "decomposition": "decompose",
    "rolled": "rolling",
    "pd_values": "pdp",
}

CALENDAR = ("date_unix", "day_julian", "weekday", "hour")


def _value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None


def read_calls(text: str) -> dict[str, dict[str, Any]]:
    """Each known call in ``text``, the last of each name: its literal keyword arguments
    and, as ``args``, its positional ones (a name, not a literal, as None)."""
    tree = ast.parse(text)
    out: dict[str, dict[str, Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in KNOWN:
            continue
        call: dict[str, Any] = {kw.arg: _value(kw.value) for kw in node.keywords if kw.arg}
        call["args"] = [_value(a) for a in node.args]
        out[name] = call
    return out


def config_from_calls(calls: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The panel's settings (as get_config writes them) the calls ran with."""
    c: dict[str, Any] = {}
    train = calls.get("build_model")
    if train:
        args = train.get("args") or []
        if len(args) > 1 and isinstance(args[1], str):
            c["target"] = args[1]
        covs = train.get("covariates") or []
        c["features"] = [v for v in covs if v not in CALENDAR]
        c["backend"] = train.get("backend", "flaml")
        for key in ("split_method", "train_fraction", "seed"):
            if train.get(key) is not None:
                c[key] = train[key]
        cfg = train.get("model_config") or {}
        if "time_budget" in cfg:
            c["time_budget"] = cfg["time_budget"]
        if cfg.get("estimator_list"):
            c["flaml_estimators"] = list(cfg["estimator_list"])
        if "n_trials" in cfg:
            c["lgb_trials"] = cfg["n_trials"]
    load = calls.get("load_estimator")
    if load:
        args = load.get("args") or []
        if len(args) > 1 and isinstance(args[1], str):
            c["target"] = args[1]
        if len(args) > 2 and isinstance(args[2], list):
            c["features"] = [v for v in args[2] if v not in CALENDAR]
        c["backend"] = "chronos-2"
    norm = calls.get("normalise")
    if norm:
        if norm.get("variables_resample"):
            c["norm_vars"] = list(norm["variables_resample"])
        if norm.get("n_samples") is not None:
            c["norm_samples"] = norm["n_samples"]
        c["norm_quantiles"] = bool(norm.get("return_quantiles"))
    dec = calls.get("decompose")
    if dec:
        c["decom_method"] = dec.get("method", "emission")
        if dec.get("n_samples") is not None:
            c["decom_samples"] = dec["n_samples"]
        if dec.get("attribution"):
            c["decom_attribution"] = dec["attribution"]
        groups = dec.get("groups")
        if isinstance(groups, dict):
            c["decom_groups"] = "; ".join(f"{k}: {', '.join(v)}" for k, v in groups.items())
    roll = calls.get("rolling")
    if roll:
        for src, dst in (
            ("window_days", "roll_window"),
            ("rolling_every", "roll_step"),
            ("n_samples", "roll_samples"),
        ):
            if roll.get(src) is not None:
                c[dst] = roll[src]
    ms = calls.get("compute_multiscale")
    if ms:
        for src, dst in (
            ("fast_days", "ms_fast"),
            ("meso_days", "ms_meso"),
            ("slow_days", "ms_slow"),
        ):
            if ms.get(src) is not None:
                c[dst] = ms[src]
    pdp = calls.get("pdp")
    if pdp and pdp.get("grid_points") is not None:
        c["pdp_grid"] = pdp["grid_points"]
    return c


def run_script(text: str, workdir: str | None = None) -> dict[str, Any]:
    """Run a session script and return what it made, by the names the log gives them
    (``df``, ``df_prep``, ``model``, ``normalised``, ...). It runs as the analyst's own
    Python does -- the window asks first -- in the folder it was saved in, so a CSV it
    reads by a relative path is found."""
    import os

    ns: dict[str, Any] = {"__name__": "__normet_session__"}
    here = os.getcwd()
    try:
        if workdir:
            os.chdir(workdir)
        exec(compile(text, "<session script>", "exec"), ns)  # noqa: S102 - the analyst's own script
    finally:
        os.chdir(here)
    keep = ("df", "df_prep", "model", "multiscale", *RESULTS)
    return {k: ns[k] for k in keep if k in ns}
