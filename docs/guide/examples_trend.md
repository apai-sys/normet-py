# Trend analysis

De-weathering produces an emission-driven series. The next question every
such record invites is whether that series is *trending*, how fast, and
whether that statement is statistically honest. That is what
`normet.trend` answers.

## Why a dedicated module

Hourly air-quality records are strongly autocorrelated -- the diurnal and
weekly structure survives the model, and consecutive hours share the same
met-driven state. A Mann-Kendall test on such a record treats its 8,760
hourly values as independent draws, and the overconfidence is not a small
correction: on a two-year hourly AR(1) record with true trend, the
classical p-value can read 0 when the correct inference would be
"significant but far less decisively so." `normet.trend` therefore refuses
to hand over a naked p-value for a record it can see to be autocorrelated,
and reports both what was measured and what the correction changed.

## The entry point: `nm.trend`

```python
import normet as nm

# The emission component is the natural input; a deweathered series works too.
df_emi = nm.decompose(
    df=df_prep, model=model, covariates=predictors, method="emission"
)

res = nm.trend(
    df_emi["emi_total"],
    name="MY1 NO2 emission-driven",
    # seasonal_period=24 for hourly records with a diurnal cycle,
    # 7 for daily records with a weekly cycle.
)
print(res)
# MY1 NO2 emission-driven: n=17520 | MK p=1.2e-09 (increasing)
# | slope 0.0054 u/m3/day | 95% CI [0.0041, 0.0067]
# | AR(1) acf=0.52 -> block bootstrap: SE=0.00021, p=4.7e-06

res.summary()   # tidy metric/value DataFrame, report-ready
```

What each field means:

| Quantity | Source | Note |
|---|---|---|
| `test.p` | Mann-Kendall, classical | With tie correction (Hirsch et al. 1982) and continuity correction. *Not* autocorrelation-adjusted -- there is no equally standard corrected variant, and the result says so in `notes`. |
| `slope.slope` | Theil-Sen | Median pair-wise slope; robust to the occasional spike a pollution event leaves in the record. Units per **day** when the input carries timestamps, per **step** otherwise. |
| `slope.ci_low/ci_high` | Sen (1968) analytic 95% CI | From `scipy.stats.theilslopes`. Assumes independence -- see the bootstrap below. |
| `slope.slope_per_year` | conversion | `slope * 365.25`, when timestamps are present. |
| `ar1_acf`, `ar1_p` | OLS-residual ACF(1) | Triggers the correction when `p < test_alpha` (default 0.05). |
| `boot_slope_se` / `boot_ci_low` / `boot_ci_high` | moving-block bootstrap | 1,000 replicates by default; block length is the longer of `3 n^(1/3)` (Politis-Romano heuristic) and one calendar week for sub-daily records, so the weekly cycle stays inside a block. Deterministic under `seed`. |
| `p_wald_boot` | Wald from the bootstrap SE | The autocorrelation-corrected trend test; the one to quote when `ar_autocorrelated` is true. |

The rule of thumb the results implement: **when `ar_autocorrelated` is
true, quote `p_wald_boot` and the bootstrap interval; the classical
quantities stay in the record as what the naive test would have claimed.**

## `nm.mann_kendall` and `nm.theil_sen`

Either can be called on its own.

```python
t = nm.mann_kendall(series, seasonal_period=24)
t.S, t.z, t.p, t.direction    # "increasing" / "decreasing" / "none"

s = nm.theil_sen(series)
s.slope, s.ci_low, s.ci_high  # per day for a datetime-indexed series
```

The seasonal Mann-Kendall test (Sen 1968) groups the record into
calendar periods -- `seasonal_period=24` means *all hours recorded at
03:00, tested against their own within-group order*, not "each week as a
group". The cycle then contributes only through *which values each slot
holds*, not through its rise-then-fall shape. The record must contain a
whole number of periods after missing values are dropped; a trailing
partial week is refused rather than tested on uneven groups. Grouping is
by record position, so assume a gap-free record: missing hours shift what
lands in each slot.

## Choosing `seasonal_period`

| Record | `seasonal_period` |
|---|---|
| Hourly | `24` (diurnal cycle) |
| Daily | `7` (weekly cycle) |
| Monthly | `12` (annual cycle) |
| Demeaned / detrended first | leave as `None` |

The classical test remains available as `seasonal_period=None`; the
module does not decide which is right for a given record -- that is a
statement about the data's structure, and a report should say which was
used.

## In the window and the Assistant

The window runs the test as one of its steps. The **Trend test** box on the
Decompose page names the series (the decomposition's emission part, its
weather part, or the observed target), the seasonal period (`auto`, `none`,
`24`, `7`, `12`) and the bootstrap's replicates, and a *Run* button starts
it; with **Run after each emission decomposition** ticked (the default), a
finished emission decomposition starts the test of its emission part under
those same settings. Its result opens on the **Trend** tab: the series with
its Theil-Sen line, a verdict that says whether the trend is significant and
*which p stands for it* -- the block-bootstrap's where the record is
autocorrelated, the classical Mann-Kendall's elsewhere -- the test's notes,
and its two-column summary. The tab carries its own CSV export (the summary)
and a command-log entry, `trend_call(...)`, as a saved session's script.
The box's `auto` seasonal is the *stated* choice of the table above, made
from the record's step and spoken where it fell back to the classical test
because the record was not a whole number of periods.

The window's **Assistant** runs the step for you: ask `*is the emission
signal trending?*` and it calls the `trend` tool (series `emission`, `met`
or `observed`; seasonal `auto` or named), which runs the step as its button
does and returns the reading -- `readings.trend_reading` -- the same numbers
the tab shows, with its notes on which p to quote. The headless MCP server
does the same in place: `normet_trend(path, column=...)` on a result table
(a decomposition's `emi_total` or `met_total`, a normalised table's
`normalised`, a raw table's target).

## Performance note

The S statistic is O(n²) in pairs, so runtime is set by the *record length*
of the classic (non-seasonal) test: a year of hourly data is about 38 million
pairs (about a second), three years a second or two, and ten years about 3.8
billion pairs (~10 seconds, measured). For multi-year hourly records, the
`seasonal_period` form is both faster and the right test, since each group
stays at most one year long. The block bootstrap is O(n · n_bootstrap) and
negligible by comparison.

## Relation to the rest of normet

`nm.trend` is statistics on a series you already have. It does not
resample weather, train a model, or touch the `prepare_data` schema: pass
it a `pd.Series` (or a DataFrame plus `value_col`). `res.summary()` returns
a two-column frame you can fold into your own report (markdown, a notebook,
or `pd.DataFrame.to_html` to drop into `generate_html_report`'s output).
Nothing in normet auto-inserts a trend block into the HTML report yet; that
is a follow-up, so this module is explicitly *not* coupled to the report
pipeline.
