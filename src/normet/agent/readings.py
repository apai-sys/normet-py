"""The evidence a normet analysis is judged by, written out as text.

An analyst chooses the target and its predictors from the table, judges a model
by how well it predicts the rows it was not trained on, and reads a normalised
series against the observed one. The window shows these as figures; a language
model needs them as numbers. These functions say what is there and stop: no
predictor is chosen and no result is judged here (:mod:`normet.agent`).

Every number an assistant quotes should come from one of these readings.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

#: Column names read as meteorological predictors (as the GUI preselects them).
MET_NAMES = frozenset(
    {
        "u10",
        "v10",
        "d2m",
        "t2m",
        "blh",
        "sp",
        "ssrd",
        "tcc",
        "tp",
        "rh2m",
        "ws",
        "wd",
        "temp",
        "rh",
        "pressure",
        "atmos_pres",
    }
)

#: Column names read as pollutants.
POLLUTANTS = frozenset(
    {"PM2.5", "PM25", "PM10", "NO2", "NO", "NOX", "O3", "SO2", "CO", "NH3", "BC", "OX"}
)

#: Time features normet adds to every model.
TIME_FEATURES = ("date_unix", "day_julian", "weekday", "hour")


def num(x: Any, digits: int = 4) -> str:
    """A number as text: *digits* significant figures, ``—`` for missing."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "—"
    if v == 0:
        return "0"
    if abs(v) >= 10 ** (digits + 2) or abs(v) < 10 ** -(digits - 1):
        return f"{v:.{digits - 1}e}"
    return f"{v:.{digits}g}"


def table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Rows as an aligned plain-text table."""
    cells = [[str(h) for h in header]] + [
        [c if isinstance(c, str) else num(c) for c in row] for row in rows
    ]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def role(column: str) -> str:
    """What a column's name says it is: ``pollutant``, ``met`` or ``other``."""
    if column.upper() in POLLUTANTS:
        return "pollutant"
    if column in MET_NAMES or column.lower() in MET_NAMES:
        return "met"
    return "other"


def _dates(df: pd.DataFrame) -> pd.Series | None:
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce")
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    return None


def _step(dates: pd.Series) -> pd.Timedelta | None:
    stamps = dates.dropna().drop_duplicates().sort_values().to_numpy()
    if len(stamps) < 2:
        return None
    return pd.Timedelta(int(np.median(np.diff(stamps).astype("timedelta64[ns]").astype("int64"))))


def human_step(step: pd.Timedelta) -> str:
    """A time step as one says it: ``1h``, ``30min``, ``1 day``, ``7 days``."""
    secs = int(step.total_seconds())
    if secs % 86400 == 0:
        days = secs // 86400
        return f"{days} day" + ("s" if days != 1 else "")
    if secs % 3600 == 0:
        return f"{secs // 3600}h"
    if secs % 60 == 0:
        return f"{secs // 60}min"
    return f"{secs}s"


def panel_units(df: pd.DataFrame) -> tuple[str, int] | None:
    """The column naming units if *df* is a panel (dates repeated, one row per
    unit and date), with how many units; None for a single series."""
    dates = _dates(df)
    if dates is None or not dates.duplicated().any():
        return None
    for col in df.columns:
        if col == "date" or pd.api.types.is_numeric_dtype(df[col]):
            continue
        n = int(df[col].nunique())
        if 1 < n < len(df) and not df.assign(_d=dates.values).duplicated(["_d", col]).any():
            return str(col), n
    return None


def data_reading(df: pd.DataFrame, target: str | None = None, name: str = "the table") -> str:
    """A table before anything is fitted: its size and span, the time step and
    gaps, and each numeric column's role, missing values and range -- with
    *target*, each column's correlation with it."""
    lines = [f"{name}: {len(df):,} rows × {df.shape[1]} columns."]
    dates = _dates(df)
    panel = panel_units(df)
    if dates is None:
        lines.append("No 'date' column: normet needs one, with the timestamp of each row.")
    else:
        bad = int(dates.isna().sum())
        lines.append(
            f"Dates {dates.min()} to {dates.max()}" + (f"; {bad} unreadable." if bad else ".")
        )
        if panel is not None:
            col, n = panel
            lines.append(
                f"A panel: {n} units in column '{col}', one row per unit and date "
                "(a synthetic-control layout: the SCM tools, not do-all)."
            )
        else:
            step = _step(dates)
            if step is not None and step > pd.Timedelta(0):
                expected = int((dates.max() - dates.min()) / step) + 1
                present = int(dates.dropna().nunique())
                dup = int(dates.duplicated().sum())
                lines.append(
                    f"Usual step {human_step(step)}; {present:,} of {expected:,} timestamps at that step "
                    f"present ({expected - present:,} missing)"
                    + (f"; {dup} duplicated." if dup else ".")
                )
    numeric = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    if target is not None and target not in df.columns:
        lines.append(f"No column {target!r}.")
        target = None
    rows = []
    for c in numeric:
        s = df[c]
        miss = int(s.isna().sum())
        row: list[Any] = [
            c,
            role(c) + (" (target)" if c == target else ""),
            f"{miss:,} ({100 * miss / max(len(s), 1):.1f}%)",
            s.min(),
            s.median(),
            s.max(),
        ]
        if target is not None:
            row.append("" if c == target else s.corr(df[target]))
        rows.append(row)
    header = ["column", "role", "missing", "min", "median", "max"]
    if target is not None:
        header.append(f"r with {target}")
    if rows:
        lines += ["", table(header, rows)]
    other = [c for c in df.columns if c != "date" and c not in numeric]
    if other:
        lines.append(f"\nNot numeric: {', '.join(map(str, other))}.")
    lines.append(
        "\nRoles are read from the column names only. normet adds the time features "
        f"({', '.join(TIME_FEATURES)}) to every model itself."
    )
    return "\n".join(lines)


def feature_importance(model: Any) -> tuple[list[str], list[float]] | None:
    """Best-effort ``(names, values)`` from a trained model, else None."""
    try:
        if hasattr(model, "feature_importance"):  # LgbModel
            vals = [float(v) for v in model.feature_importance()]
            names = [str(n) for n in getattr(model, "feature_names", [])]
            if len(vals) == len(names):
                return names, vals
        automl = getattr(model, "model", model)
        est = getattr(automl, "model", automl)  # flaml AutoML.model -> sklearn wrapper
        fi = getattr(est, "feature_importances_", None)
        fn = getattr(automl, "feature_names_in_", None)
        if fn is None:
            fn = getattr(est, "feature_names_in_", None)
        if fi is not None and fn is not None and len(fn) == len(fi):
            return [str(n) for n in fn], [float(v) for v in fi]
    except Exception:
        pass
    return None


#: modStats' columns an assistant is shown, in this order.
_STATS = ("n", "R2", "r", "RMSE", "MB", "NMB", "FAC2", "COE", "IOA")


def model_reading(df_prep: pd.DataFrame, model: Any) -> str:
    """How well a trained model predicts: :func:`normet.modStats` on the rows it
    was trained on, the rows held out, and all -- and its features' importance."""
    from normet import modStats

    stats = modStats(df_prep, model)
    cols = [c for c in _STATS if c in stats.columns]
    order = {"testing": 0, "training": 1, "all": 2}
    stats = stats.sort_values("set", key=lambda s: s.map(order).fillna(9))
    lines = [
        "Model fit (normet.modStats; 'testing' is the rows held out of training):",
        "",
        table(["set", *cols], [[r["set"], *(r[c] for c in cols)] for _, r in stats.iterrows()]),
    ]
    fi = feature_importance(model)
    if fi is not None:
        names, vals = fi
        total = sum(vals) or 1.0
        top = sorted(zip(names, vals, strict=True), key=lambda p: -p[1])[:12]
        lines += [
            "",
            "Feature importance (the model's own, as a share of its total):",
            "",
            table(["feature", "share"], [[n, f"{100 * v / total:.1f}%"] for n, v in top]),
        ]
    return "\n".join(lines)


def _periods(index: pd.DatetimeIndex) -> tuple[str, str]:
    years = index.year.nunique()
    return ("YE", "year") if years >= 3 else ("ME", "month")


def normalised_reading(result: pd.DataFrame) -> str:
    """A normalised series against the observed one: their means overall and by
    year (or month, for a short record), and the meteorological part
    (observed − normalised)."""
    df = result.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" not in df.columns:
            raise ValueError("a normalised result needs a 'date' column or a datetime index")
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns="date")
    if not {"observed", "normalised"} <= set(df.columns):
        raise ValueError(
            f"expected 'observed' and 'normalised' columns, found {', '.join(map(str, df.columns))}"
        )
    obs, norm = df["observed"], df["normalised"]
    met = obs - norm
    lines = [
        f"{len(df):,} rows, {df.index.min()} to {df.index.max()}.",
        f"Mean observed {num(obs.mean())}, mean normalised {num(norm.mean())}; "
        f"meteorological part (observed − normalised): mean {num(met.mean())}, "
        f"standard deviation {num(met.std())}.",
        f"Standard deviation: observed {num(obs.std())}, normalised {num(norm.std())}.",
    ]
    rule, word = _periods(pd.DatetimeIndex(df.index))
    by = df[["observed", "normalised"]].resample(rule).mean()
    by["met part"] = by["observed"] - by["normalised"]
    label = "%Y" if word == "year" else "%Y-%m"
    lines += [
        "",
        f"Means by {word}:",
        "",
        table(
            [word, "observed", "normalised", "met part"],
            [
                [t.strftime(label), *r]
                for t, r in zip(pd.DatetimeIndex(by.index), by.itertuples(index=False), strict=True)
            ],
        ),
    ]
    other = [c for c in df.columns if c not in ("observed", "normalised")]
    if other:
        lines.append(f"\nAlso in the result: {', '.join(map(str, other))}.")
    return "\n".join(lines)


def decomposition_reading(result: pd.DataFrame) -> str:
    """Each component of a decomposition: its mean, spread and range."""
    df = result.drop(columns=["date"], errors="ignore")
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rows = [[c, df[c].mean(), df[c].std(), df[c].min(), df[c].max()] for c in cols]
    return "\n".join(
        [
            f"{len(df):,} rows; {len(cols)} components.",
            "",
            table(["component", "mean", "std", "min", "max"], rows),
        ]
    )


def rolling_reading(result: pd.DataFrame) -> str:
    """A rolling normalisation: how many windows, their mean series by year (or month)
    and how far the windows that overlap a time disagree there."""
    import re

    df = result.copy()
    if not isinstance(df.index, pd.DatetimeIndex) and "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns="date")
    cols = [
        c for c in df.columns if isinstance(c, str) and re.fullmatch(r"(window|rolling)_\d+", c)
    ]
    if not cols:
        raise ValueError("a rolling result has window_01, window_02, ... columns")
    windows = df[cols].apply(pd.to_numeric, errors="coerce")
    mean, spread = windows.mean(axis=1), windows.std(axis=1)
    lines = [
        f"{len(cols)} windows over {len(df):,} rows, {df.index.min()} to {df.index.max()}.",
        f"Mean of the windows' normalised series {num(mean.mean())}; where windows overlap "
        f"they differ by {num(spread.mean())} on average (standard deviation across them).",
    ]
    if isinstance(df.index, pd.DatetimeIndex):
        rule, word = _periods(df.index)
        by = pd.DataFrame({"windows' mean": mean, "spread": spread}).resample(rule).mean()
        label = "%Y" if word == "year" else "%Y-%m"
        lines += [
            "",
            f"By {word}:",
            "",
            table(
                [word, "windows' mean", "spread"],
                [
                    [t.strftime(label), *r]
                    for t, r in zip(by.index, by.itertuples(index=False), strict=True)
                ],
            ),
        ]
    return "\n".join(lines)


def pdp_reading(result: pd.DataFrame) -> str:
    """Partial dependence: for each variable, the prediction's range over its values and
    where along them it is lowest and highest."""
    need = {"variable", "value", "pdp_mean"}
    if not need <= set(result.columns):
        raise ValueError(f"a PDP result has columns {', '.join(sorted(need))}")
    rows: list[list] = []
    for var, g in result.groupby("variable", sort=False):
        g = g.dropna(subset=["pdp_mean"])
        if g.empty:
            continue
        lo, hi = g.loc[g["pdp_mean"].idxmin()], g.loc[g["pdp_mean"].idxmax()]
        rows.append(
            [
                var,
                hi["pdp_mean"] - lo["pdp_mean"],
                lo["value"],
                lo["pdp_mean"],
                hi["value"],
                hi["pdp_mean"],
            ]
        )
    rows.sort(key=lambda r: -float(r[1]))
    return "\n".join(
        [
            f"Partial dependence of {len(rows)} variables, widest range first: the predicted "
            "mean at the value where it is lowest and where it is highest.",
            "",
            table(
                ["variable", "range", "lowest at", "prediction", "highest at", "prediction"], rows
            ),
        ]
    )


def scm_reading(result: pd.DataFrame, cutoff: str) -> str:
    """A synthetic-control fit (:func:`normet.scm_diagnostics`): how closely the
    synthetic unit follows the treated one before *cutoff*, and the effect after."""
    from normet import scm_diagnostics

    df = result
    if "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns="date")
    d = scm_diagnostics(df, cutoff)
    rows = [
        ["pre-period points", d.get("pre_n")],
        ["pre-period RMSE", d.get("pre_rmse")],
        ["pre-period MAE", d.get("pre_mae")],
        ["pre-period R²", d.get("pre_r2")],
        ["post-period points", d.get("post_n")],
        ["ATT (mean effect after the cutoff)", d.get("att")],
        ["cumulative effect", d.get("att_cum")],
        ["post-period RMSE", d.get("post_rmse")],
    ]
    if d.get("n_donors"):
        rows += [
            ["donors with weight", d.get("n_donors")],
            ["effective number of donors", d.get("effective_n_donors")],
            ["largest donor share", d.get("top_donor_share")],
        ]
    pre_mean = df.loc[df.index < pd.Timestamp(cutoff), "observed"].mean()
    rows.append(["treated unit's pre-period mean", pre_mean])
    pt = d.get("pre_trend") or {}
    rows += [
        ["pre-trend p-value (slope)", pt.get("p_slope")],
        ["pre-trend drift (final third vs rest)", pt.get("rmpe_drift")],
    ]
    lines = [
        f"Synthetic control, cutoff {cutoff} (normet.scm_diagnostics):",
        "",
        table(["", "value"], rows),
    ]
    if pt.get("verdict"):
        lines += ["", f"pre-trend: {pt['verdict']}"]
    return "\n".join(lines)


def cv_reading(scores: pd.DataFrame) -> str:
    """A time-series cross-validation's scores, fold by fold, with their means."""
    df = scores.drop(columns=["date"], errors="ignore")
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rows = [[*(r[c] for c in df.columns)] for _, r in df.iterrows()]
    lines = [f"{len(df)} folds.", "", table([str(c) for c in df.columns], rows)]
    if numeric:
        means = ", ".join(f"{c} {num(df[c].mean())}" for c in numeric if c not in ("fold",))
        lines += ["", f"Means over the folds: {means}."]
    return "\n".join(lines)


def trend_reading(res: Any, name: str = "series") -> str:
    """A trend test on a series (normet.trend result): its direction and the
    Mann-Kendall test, the slope with its interval, and -- where the record is
    autocorrelated -- the block-bootstrap interval and Wald p, which is the one
    to quote (the classical MK p is anti-conservative there and says so)."""
    t = res.test
    s = res.slope
    unit = "per day" if s.time_unit == "day" else "per step"
    rows = [
        ["series", res.name],
        ["rows used", res.n],
    ]
    if res.n_dropped:
        rows.append(["missing dropped", res.n_dropped])
    rows += [
        ["direction (5%)", t.direction],
        ["Mann-Kendall S", res.test.S],
        ["MK z", num(t.z, 3)],
        ["MK p (two-sided, classical)", num(t.p, 4)],
        [f"slope ({unit})", num(s.slope, 4)],
        [f"slope 95% CI ({unit})", f"[{num(s.ci_low, 4)}, {num(s.ci_high, 4)}]"],
    ]
    if s.slope_per_year is not None:
        rows.append(["slope per year", num(s.slope_per_year, 4)])
    rows += [
        ["AR(1) residual acf", num(res.ar1_acf, 3)],
        ["AR(1) p", num(res.ar1_p, 4)],
    ]
    if res.ar_autocorrelated:
        if res.boot_slope_se is not None:
            rows += [
                ["bootstrap SE (block)", num(res.boot_slope_se, 4)],
                [
                    f"bootstrap 95% CI ({unit})",
                    f"[{num(res.boot_ci_low, 4)}, {num(res.boot_ci_high, 4)}]",
                ],
                ["p (Wald on block bootstrap)", num(res.p_wald_boot, 4)],
                ["block length", res.block_length],
                ["bootstrap replicates", res.n_bootstrap],
            ]
            rows.append(
                [
                    "quote",
                    f"{res.name}: {t.direction} at 5%, slope {num(s.slope, 4)} {unit} "
                    f"(bootstrap CI [{num(res.boot_ci_low, 4)}, {num(res.boot_ci_high, 4)}], "
                    f"p_wald {num(res.p_wald_boot, 4)})",
                ]
            )
        else:
            rows.append(
                ["quote", "AR(1) detected but the block bootstrap was unavailable — see the notes"]
            )
    lines = [f"Trend test on {name} (normet.trend):", ""]
    lines.append(table(["", "value"], rows))
    lines += ["", *res.notes]
    return "\n".join(lines)


def trend_breaks_reading(res: Any, name: str = "series") -> str:
    """A segmented trend test on a series (normet.detect_trend_breaks result): where
    the trend breaks, each segment with its own Theil-Sen slope and Sen (1968)
    interval, and the block-bootstrap p of the search over every candidate
    position -- the multiple-testing-corrected, autocorrelation-aware one to
    quote; the record's reason for staying one regime where the gates refused a
    split."""
    unit = "per day" if res.time_unit == "day" else "per step"
    rows = [
        ["series", res.name],
        ["rows used", res.n],
        ["segments", res.n_segments],
    ]
    if res.n_dropped:
        rows.append(["missing dropped", res.n_dropped])
    header = ["start", "end", "n", f"slope ({unit})", "slope 95% CI", "slope per year"]
    seg_rows = []
    for start, end, seg in zip(res.starts, res.ends, res.segments, strict=False):
        row = [
            str(start)[:16],
            str(end)[:16],
            seg.n,
            num(seg.slope, 4),
            f"[{num(seg.ci_low, 4)}, {num(seg.ci_high, 4)}]",
        ]
        row.append(num(seg.slope_per_year, 3) if seg.slope_per_year is not None else "n/a")
        seg_rows.append(row)
    lines = [f"Segmented trend test on {name} (normet.detect_trend_breaks):", ""]
    if res.n_segments > 1:
        from normet.analysis.trend import _fmt_index

        breaks = ", ".join(_fmt_index(v) for v in res.starts[1:])
        rows += [
            ["break(s)", breaks],
            ["single-slope SSE", num(res.sse_single, 6)],
            ["segmented SSE", num(res.sse_segmented, 6)],
            ["SSE improvement", f"{num(res.improvement_pct, 1)} %"],
            [
                "BIC (single / segmented)",
                f" {num(res.bic_single, 6)} / {num(res.bic_segmented, 6)}",
            ],
            ["p (block bootstrap of the search)", num(res.p_value, 4)],
            ["alpha", res.alpha],
        ]
        if res.n_bootstrap is not None:
            rows += [
                ["bootstrap replicates", res.n_bootstrap],
                ["block length", res.block_length],
            ]
        lines.append(table(["", "value"], rows))
        lines += ["", table(header, seg_rows)]
        lines += [
            "",
            f"Quote: {res.n_segments} segments, break(s) at {breaks}; "
            f"p={num(res.p_value, 4)} (block bootstrap over {res.n_bootstrap} resamples "
            f"is the multiple-testing-corrected, autocorrelation-aware test -- the one "
            f"to quote). Each segment's Theil-Sen slope above, with its interval; "
            "compare adjacent segments against their Sen intervals, do not re-test.",
        ]
    else:
        rows += [
            ["single-slope SSE", num(res.sse_single, 6)],
            ["BIC (single / searched)", f" {num(res.bic_single, 6)} / {num(res.bic_segmented, 6)}"],
        ]
        if res.p_value is not None:
            rows += [
                [
                    "best candidate cut",
                    str(res.best_break)[:16] if res.best_break is not None else "n/a",
                ],
                ["best candidate SSE improvement", f"{num(res.best_split_pct, 1)} %"],
                ["p (block bootstrap of the search)", num(res.p_value, 4)],
            ]
        lines.append(table(["", "value"], rows))
        # the notes say where the search wanted to cut and why it was refused
    lines += ["", *res.notes]
    return "\n".join(lines)
