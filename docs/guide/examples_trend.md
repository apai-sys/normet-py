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

## When the trend changes: `nm.detect_trend_breaks`

A single slope cannot say what happened on a record whose level or slope
changes part-way through -- a policy year, a lockdown. Fit the whole record
with one line and the change is diluted into a muddier trend: the pre-event
segment drags the slope toward zero, the post-event segment drags it up,
and either significance statement or the slope itself is a statement about
*both* regimes mixed together. `detect_trend_breaks` asks the question
directly: *where does the regime change, and what are the slopes of the
regimes?*

```python
res = nm.detect_trend_breaks(
    df_emi["emi_total"],                       # or any Series/DataFrame + value_col
    name="MY1 NO2 emission",
    max_segments=2,                            # one split (the default)
    # min_segment=None  -> the longer of 28 and 5 % of the record
    # alpha=0.05, n_bootstrap=500, seed=7654321
)
print(res)
# MY1 NO2 emission: n=5793 | 2 segments | break at 2020-03-25 22:00 | SSE -39.9%
# | seg1 slope 0.0216 u/m3/day | seg2 slope 0.0229 u/m3/day | p=0.002

res.summary()   # one row per segment: start, end, n, slope, CI (+ per-year)
```

On MY1's emission part (January-August 2020, the record that holds the
UK lockdown) the detector puts the break at **2020-03-25 22:00**, two days
after the 23 March lockdown order, with the SSE falling 39.9 % when the
record is allowed one break and the block-bootstrap p at 0.002. The two
regimes split on exactly the question of interest: the pre-lockdown
segment's 95 % CI is `[-0.0019, 0.0446]` (straddling 0, so not
significantly trending), the post-lockdown segment's is `[0.0171,
0.0286]` (entirely positive). A single slope for the whole record could
not have made that distinction -- the point of the segmentation is that
the two regimes carry their own, separately honest, intervals.

### How the split is found, and why the gate exists

The search is a full O(n) grid: the two-segment OLS residual sum of
squares of *every* candidate position is O(1) from prefix sums, so the
record is scanned once, and the split that most reduces the SSE wins.
A grid searched over `n-1` positions, however, always finds *something* --
pure noise contains a split that cuts its SSE by some amount -- so the
candidate split passes two gates before it is reported:

1. **BIC, per round.** Each split must beat the BIC of the current fit, so
   the extra slope + intercept earn their keep. This is the *stopping*
   rule for `max_segments > 2` (greedy forward segmentation).
2. **Block-bootstrap p for the search.** The observed SSE reduction of the
   final fit over the single-segment fit is compared with the *same
   search re-run on every moved-block resample* of the series. The
   resulting p is the probability that a record without a break --
   searched exactly as hard, with its autocorrelation preserved by the
   blocks -- produces a reduction at least that large. It is the
   multiple-testing-corrected, autocorrelation-aware test; it is why a
   grid search can be honest.

If either gate fails, the series is reported as **one regime** and
`summary()` has a single row; the search is not hidden -- `best_break`,
`best_split_pct` and `notes` keep *where it wanted to cut* and *what the
p was*, so a report can say "a cut at X lowers the SSE by Y % but the
block-bootstrap p is Z, so it is not claimed as a break."

| Quantity | Note |
|---|---|
| `starts`, `ends` | Index value of each segment's first and last point (timestamps for a datetime record, positions otherwise). The break is `starts[1]` -- the first point of the new regime. |
| `segments[i].slope` | Theil-Sen per segment, in the same units as `theil_sen` (per day / per step, with `slope_per_year` when timestamps are present). Robust to the spikes a pollution event leaves inside a segment. |
| `best_break`, `best_split_pct` | The first round's best candidate and its SSE reduction, recorded *even when the gates rejected it*. |
| `sse_single`, `sse_segmented`, `improvement_pct` | The single- and segmented-fit residuals and their relative reduction. |
| `bic_single`, `bic_segmented` | The BICs the gate compared (3 parameters per segment: slope, intercept, noise variance). |
| `p_value`, `significant` | The block-bootstrap p and whether it fell below `alpha` (default 0.05). |
| `min_segment` | Minimum points per segment (default: the longer of 28 and 5 % of the record). Bars splits pinned to an edge and keeps each piece long enough for a Theil-Sen to mean something. |

The per-segment CIs are Sen intervals: compare adjacent segments against
*their* intervals (here: the pre-lockdown CI straddles 0, the
post-lockdown interval is entirely positive) rather than re-testing the
split -- the split itself is what `p_value` certifies.

### Performance

The search is O(n) and the bootstrap O(n · n_bootstrap) with an O(n)
search inside each replicate: MY1's 5,793-hour record runs in about a
second at the default 500 replicates, and three years of hourly
(26,280 points) takes about 20 seconds (measured). The block length uses
the same heuristic as `nm.trend` (longer of `3 n^(1/3)` and a calendar
week for sub-daily records). The per-segment Theil-Sen is the same
median-of-pairs as everywhere else, so a *very* long single segment costs
as much as `nm.theil_sen` would on the whole record; for multi-year
hourly records prefer `max_segments=2` and a smaller record window when
speed matters.

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
