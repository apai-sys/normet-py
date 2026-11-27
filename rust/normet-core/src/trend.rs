//! Exact Theil-Sen slope and Sen (1968) interval order statistics.
//!
//! `scipy.stats.theilslopes`, normet's `_theil_sen_large` and R's Fenwick
//! path all select the same order statistics from the multiset of pairwise
//! slopes `(y_j - y_i) / (x_j - x_i)` over pairs with `x_i < x_j`. This
//! module performs the same IEEE-754 arithmetic in Rust, so the slope and
//! the two interval endpoints are bit-identical to the hosts' values; only
//! the overhead differs. Keys are stored as an order-preserving `f64 -> u64`
//! map, so every selection below is exact integer selection on the same
//! multiset the hosts sort.
//!
//! Above [`FULL_BELOW_PAIRS`] pairs (1 << 22, about 2 900 points with
//! distinct x) the multiset is never materialised. A deterministic pilot of
//! 131 072 stratified pairs places a key window around the four target
//! ranks; one exact pass counts the keys below and inside the window, copies
//! the in-window keys (at most 128 MiB), and selects from that copy. When
//! the window misses a rank or overflows, a 16-bit radix histogram resolves
//! the ranks from exact prefix counts, refining sixteen bits at a time until
//! a bucket fits the bounded collection or every bit of the key is fixed.
//! The pilot only chooses where to look: both fallback stages count the true
//! multiset, so no estimate can change an answer, and the pilot itself is
//! deterministic (no RNG, no statics).
//!
//! Measured on an 8-core Apple M1 Pro (release, best of three): a three-year
//! hourly record (n = 26 280, m = 345 M) takes 0.54 s and 87 MB peak instead
//! of 2.06-3.39 s and 2.76 GB; one year (n = 8 700) takes 0.063 s against
//! 0.15-0.19 s and 15 MB against 310 MB.
//! The small path (m <= 1 << 22) keeps the materialise-and-select order
//! verbatim: it is unchanged within noise there and avoids the pilot copy on
//! short records. The tail is measured, not hidden: near-noiseless linear
//! records whose distinct slopes share one top-16 key bucket and alternating
//! +/-1e308 overflow records take the four-level radix fallback and cost up
//! to ~1.5x the old path; heavy-duplicate records 1.1-1.5x. The small path
//! still needs m * 8 bytes (32 MiB at the cutoff; 11 GB at 52 400 points,
//! which the window path handles in 5.4 s and 14 MB).
//!
//! `z_alpha` is the lower normal quantile (scipy `norm.ppf(0.025)`, R
//! `qnorm(0.025)`), supplied by the host: a ppf implementation here would
//! not reproduce either host's last bit, and the interval ranks are a
//! function of it.

use std::fmt;

/// The Theil-Sen slope and its Sen (1968) interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TheilSenInterval {
    /// Median of the pairwise slopes.
    pub slope: f64,
    /// Lower confidence bound (`alpha = 0.05`, as in the hosts).
    pub low: f64,
    /// Upper confidence bound.
    pub high: f64,
    /// Number of pairwise slopes selected from.
    pub n_pairs: u64,
    /// True when every `x` is identical: all three outputs are NaN, as the
    /// hosts warn and return.
    pub degenerate: bool,
}

/// Input problems; all map to Python `ValueError` / R errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TheilSenError {
    /// `x` and `y` differ in length.
    LengthMismatch,
    /// Fewer than two points.
    TooShort,
    /// A non-finite value in `x`, `y` or `z_alpha`; the hosts drop missing
    /// values before calling, so this is a programming error.
    NonFinite,
}

impl fmt::Display for TheilSenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch => write!(f, "x and y must have the same length"),
            Self::TooShort => write!(f, "at least two points are required"),
            Self::NonFinite => write!(f, "x, y and z_alpha must be finite"),
        }
    }
}

impl std::error::Error for TheilSenError {}

/// Selection knobs for [`theil_sen_interval_with_config`].
///
/// The defaults are the production configuration. The struct is exposed so
/// the PyO3 parity tests (and the core unit tests) can force the window and
/// radix-fallback branches on short records; every configuration must give
/// identical bits, because each path is exact selection on the same keys.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct TheilSenConfig {
    /// `m <= full_below_pairs` materialises all keys and selects in the
    /// reference order. Below this size that path is as fast as the window
    /// (it is the old code) and avoids the pilot copy.
    pub full_below_pairs: u64,
    /// Stratified pilot size when the window path is used.
    pub pilot_pairs: usize,
    /// Maximum number of in-window keys copied in the one-pass window path.
    pub window_cap: usize,
}

/// The production selection configuration: materialise up to 4 194 304
/// pairs (~2 900 points with distinct x), sample 131 072 pilot pairs and
/// copy at most 16 777 216 in-window keys (128 MiB).
#[doc(hidden)]
pub const THEIL_SEN_DEFAULT_CONFIG: TheilSenConfig = TheilSenConfig {
    full_below_pairs: FULL_BELOW_PAIRS,
    pilot_pairs: PILOT_PAIRS,
    window_cap: WINDOW_CAP,
};

const FULL_BELOW_PAIRS: u64 = 1 << 22;
const PILOT_PAIRS: usize = 1 << 17;
const WINDOW_CAP: usize = 1 << 24;
/// 16-bit radix digits: 65 536 counters per level, summed as exact u64.
const BUCKETS: usize = 1 << 16;
/// Interleaved histogram copies break the load-add-store chain on the few
/// hot counters near the median (slopes are highly concentrated).
const HIST_COPIES: usize = 4;
/// Pilot window half-width in rank units: eight binomial standard errors
/// (`8 * sqrt(0.25 / p)`). The measured prediction error is ~1 %, so the
/// window almost always contains every target rank; exactness does not
/// depend on it.
const WINDOW_DELTA_FACTOR: f64 = 8.0;
/// Fallback collections are bounded at 4 194 304 keys (32 MiB): selecting
/// from a 16 M-key bucket measured slower than two extra refinement levels.
const FALLBACK_COLLECT_CAP: usize = 1 << 22;

/// Why a run took a particular path. Counters only (no strings or
/// timings), read by the unit tests and the parity harness.
#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
struct PathStats {
    full_materialise: bool,
    window: bool,
    uniform: bool,
    fallback: bool,
    /// Every full scan of the pair multiset (materialise, uniform check,
    /// window pass, each histogram or collection pass).
    passes: u32,
    window_keys: usize,
    collected_keys: usize,
    predicted_inside: u64,
}

/// The four ranks the reference reads, after the clamps.
#[derive(Clone, Copy, Debug)]
struct Ranks {
    mid_hi: u64,
    mid_lo: u64,
    ru: u64,
    rl: u64,
}

/// Sum of Sen's tie term over the run lengths of a sorted array.
fn tie_term(sorted: &[f64]) -> i128 {
    let mut total = 0i128;
    let mut i = 0usize;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == sorted[i] {
            j += 1;
        }
        let k = (j - i) as i128;
        total += k * (k - 1) * (2 * k + 5);
        i = j;
    }
    total
}

/// The index Python would use for a possibly negative rank (negative
/// indices count from the end; the hosts never produce one, but mirroring
/// costs nothing). Widths are u64 so a 32-bit build cannot truncate a
/// pair count whose two years of hourly data already exceed 4.29e9.
fn py_index(rank: i64, len: u64) -> u64 {
    if rank < 0 {
        (len as i64 + rank) as u64
    } else {
        rank as u64
    }
}

/// Order-preserving map from an `f64` to a `u64`, so selection compares
/// native integers instead of running `total_cmp` through a comparator
/// (measured: about 4x faster selection at 300 M slopes). The map is a
/// bijection on bit patterns; `slope_from_key` inverts it exactly.
#[inline]
fn slope_key(f: f64) -> u64 {
    let bits = f.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits | (1 << 63)
    }
}

/// Inverse of [`slope_key`].
#[inline]
fn slope_from_key(key: u64) -> f64 {
    if key >> 63 == 1 {
        f64::from_bits(key & !(1u64 << 63))
    } else {
        f64::from_bits(!key)
    }
}

/// Select the `k`-th smallest key in place and return its slope.
fn kth(keys: &mut [u64], k: usize) -> f64 {
    let (_, value, _) = keys.select_nth_unstable(k);
    slope_from_key(*value)
}

/// Theil-Sen slope with the analytic Sen (1968) 95 % interval.
///
/// Parameters
/// ----------
/// - `x`, `y`: equal-length finite samples; `x` need not be sorted.
/// - `z_alpha`: the lower standard-normal quantile for the interval
///   (the hosts pass `ppf(0.025)` ~ -1.9599639845400545).
///
/// Pairs with equal `x` are excluded, and ties are accounted for exactly as
/// scipy's `1/18 *` tie-corrected variance does. The returned values equal
/// scipy's `theilslopes(y, x)` bit for bit on finite inputs.
pub fn theil_sen_interval(
    x: &[f64],
    y: &[f64],
    z_alpha: f64,
) -> Result<TheilSenInterval, TheilSenError> {
    theil_sen_interval_with_config(x, y, z_alpha, THEIL_SEN_DEFAULT_CONFIG)
}

/// [`theil_sen_interval`] with explicit selection knobs.
///
/// Hidden from the public documentation: the Python binding exposes it for
/// parity tests, which force the window and radix branches on records too
/// short to reach them naturally. The three-number result is identical for
/// every configuration.
#[doc(hidden)]
pub fn theil_sen_interval_with_config(
    x: &[f64],
    y: &[f64],
    z_alpha: f64,
    cfg: TheilSenConfig,
) -> Result<TheilSenInterval, TheilSenError> {
    interval_stats(x, y, z_alpha, cfg).map(|(interval, _)| interval)
}

/// The frozen prologue, the selection stage, and the reference assembly.
fn interval_stats(
    x: &[f64],
    y: &[f64],
    z_alpha: f64,
    cfg: TheilSenConfig,
) -> Result<(TheilSenInterval, PathStats), TheilSenError> {
    if x.len() != y.len() {
        return Err(TheilSenError::LengthMismatch);
    }
    let n = x.len();
    if n < 2 {
        return Err(TheilSenError::TooShort);
    }
    if !z_alpha.is_finite() || x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
        return Err(TheilSenError::NonFinite);
    }

    // Sort by x, carrying y. Stability does not matter: equal-x pairs are
    // excluded and the tie correction is order-free.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| x[a].total_cmp(&x[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = order.iter().map(|&i| y[i]).collect();

    // earlier[j] = number of points with a strictly smaller x, i.e. the
    // width of j's row in scipy's upper-triangle mask.
    let earlier: Vec<usize> = (0..n).map(|j| xs.partition_point(|&v| v < xs[j])).collect();
    let m: u64 = earlier.iter().map(|&k| k as u64).sum();

    if m == 0 {
        return Ok((
            TheilSenInterval {
                slope: f64::NAN,
                low: f64::NAN,
                high: f64::NAN,
                n_pairs: 0,
                degenerate: true,
            },
            PathStats::default(),
        ));
    }

    // Sen (1968) eq. 2.6, as scipy computes it: `1/18 *` the tie-corrected
    // integer expression. The Python large path divides by 18.0 instead; the
    // two differ by at most an ulp of sigsq and the selection ranks are far
    // from a rounding boundary for every measured record.
    let mut xsorted = xs.clone();
    xsorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let mut ysorted = ys.clone();
    ysorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let nn = n as i128;
    let base = nn * (nn - 1) * (2 * nn + 5);
    let sigsq = (1.0 / 18.0) * ((base - tie_term(&xsorted) - tie_term(&ysorted)) as f64);
    let sigma = sigsq.max(0.0).sqrt();

    let nt = m as f64;
    let ru_f = ((nt - z_alpha * sigma) / 2.0).round_ties_even();
    let rl_f = ((nt + z_alpha * sigma) / 2.0).round_ties_even();
    // scipy and _theil_sen_large both truncate a rounded float here.
    let ru = py_index((ru_f as i64).min(m as i64 - 1).max(0), m);
    let rl = py_index((rl_f as i64 - 1).max(0), m);
    let ranks = Ranks {
        mid_hi: m / 2,
        mid_lo: (m - 1) / 2,
        ru,
        rl,
    };
    let want = target_ranks(&ranks, m);

    let mut stats = PathStats::default();
    let values = select_values(&xs, &ys, &earlier, n, m, &ranks, &want, cfg, &mut stats);
    debug_assert!(stats.passes <= 19, "pass bound");

    // Assemble exactly as the reference does: same formulas, same order.
    let value = |rank: u64| -> f64 {
        let slot = want.binary_search(&rank).expect("target rank selected");
        values[slot].expect("target rank resolved")
    };
    let mid_hi_v = value(ranks.mid_hi);
    let mid_lo_v = if m % 2 == 0 {
        value(ranks.mid_lo)
    } else {
        mid_hi_v
    };
    let high = if ranks.ru > ranks.mid_hi {
        value(ranks.ru)
    } else {
        mid_hi_v
    };
    let low = if ranks.rl < ranks.mid_hi {
        value(ranks.rl)
    } else {
        mid_hi_v
    };
    let slope = if m % 2 == 0 {
        (mid_lo_v + mid_hi_v) / 2.0
    } else {
        mid_hi_v
    };

    Ok((
        TheilSenInterval {
            slope,
            low,
            high,
            n_pairs: m,
            degenerate: false,
        },
        stats,
    ))
}

/// The distinct full-array ranks the reference reads. `mid_lo` is only read
/// for even m, `ru` only when it lies above the median, `rl` only below it;
/// at most four values, sorted for deterministic grouping.
fn target_ranks(ranks: &Ranks, m: u64) -> Vec<u64> {
    let mut want = Vec::with_capacity(4);
    want.push(ranks.mid_hi);
    if m % 2 == 0 {
        want.push(ranks.mid_lo);
    }
    if ranks.ru > ranks.mid_hi {
        want.push(ranks.ru);
    }
    if ranks.rl < ranks.mid_hi {
        want.push(ranks.rl);
    }
    want.sort_unstable();
    want.dedup();
    want
}

/// Dispatch between the small materialised path and the window/radix path.
/// Every branch fills every slot of `values`; the slots are `Option` so a
/// legitimate NaN slope is not an "unresolved" sentinel.
fn select_values(
    xs: &[f64],
    ys: &[f64],
    earlier: &[usize],
    n: usize,
    m: u64,
    ranks: &Ranks,
    want: &[u64],
    cfg: TheilSenConfig,
    stats: &mut PathStats,
) -> Vec<Option<f64>> {
    let mut values = vec![None; want.len()];

    if m <= cfg.full_below_pairs && m <= usize::MAX as u64 {
        // Small input: one exhaustive pass and select, with exactly the
        // reference's selection order (median first, then the interval
        // ranks from half-sized slices). Four independent full-array
        // selects cost about twice as much at m ~ 1 M.
        stats.full_materialise = true;
        stats.passes += 1;
        let mut keys = build_all_keys(xs, ys, earlier, n);
        debug_assert_eq!(keys.len() as u64, m);
        let slot = |r: u64| want.binary_search(&r).expect("target rank selected");
        let mid_hi_us = ranks.mid_hi as usize;
        let mid_hi_v = kth(&mut keys, mid_hi_us);
        values[slot(ranks.mid_hi)] = Some(mid_hi_v);
        if m % 2 == 0 {
            values[slot(ranks.mid_lo)] = Some(kth(&mut keys[..mid_hi_us], ranks.mid_lo as usize));
        }
        if ranks.ru > ranks.mid_hi {
            values[slot(ranks.ru)] = Some(kth(
                &mut keys[mid_hi_us + 1..],
                (ranks.ru - ranks.mid_hi - 1) as usize,
            ));
        }
        if ranks.rl < ranks.mid_hi {
            values[slot(ranks.rl)] = Some(kth(&mut keys[..mid_hi_us], ranks.rl as usize));
        }
        return values;
    }

    // 1. Deterministic stratified pilot over the flat pair index.
    let pilot = pilot_keys(xs, ys, earlier, n, m, cfg.pilot_pairs.max(1));
    let p = pilot.len();
    let (lo_key, hi_key) = choose_window(&pilot, m, want);

    // Pilot estimate of the in-window population. The top-16 histogram is
    // cheap on its own (~0.04 s at n=8700) but fusing it into the branchy
    // window pass measured ~0.19 s, so the one-pass path omits it; the
    // fallback builds it in a separate pass.
    let pilot_inside = pilot
        .iter()
        .filter(|&&key| key >= lo_key && key <= hi_key)
        .count() as u128;
    let predicted = pilot_inside * m as u128 / p as u128;
    stats.predicted_inside = predicted.min(u64::MAX as u128) as u64;

    let mut resolved = false;
    if pilot[0] == pilot[p - 1] {
        // A constant pilot almost always means a constant multiset
        // (constant y, y = a*x, y = 0). Verify in one pass without copying:
        // the window path would otherwise push every one of the m keys,
        // which at m ~ 10 M is a 10x regression against the fallback it
        // replaces. A constant pilot does not imply uniform keys, so this
        // pass is a real verification, never a shortcut on the pilot alone.
        stats.passes += 1;
        let (uniform, first_key) = uniform_multiset(xs, ys, earlier, n);
        if uniform {
            stats.uniform = true;
            values.fill(Some(slope_from_key(first_key)));
            resolved = true;
        }
    }
    if !resolved && predicted * 5 <= (cfg.window_cap as u128) * 4 {
        // 2a. One full pass without the histogram: exact below/inside
        //     counts plus the bounded in-window copy. Only enter when the
        //     prediction leaves 20 % headroom under the cap.
        stats.window = true;
        stats.passes += 1;
        let reserve = predicted.min(cfg.window_cap as u128) as usize;
        let (below, inside, mut win) =
            window_pass(xs, ys, earlier, n, lo_key, hi_key, cfg.window_cap, reserve);
        stats.window_keys = win.len();
        let complete = inside <= cfg.window_cap as u64 && win.len() as u64 == inside;
        let mut ok = complete;
        if ok {
            for &r in want {
                if !(below <= r && r < below + inside) {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            for (slot, &r) in want.iter().enumerate() {
                values[slot] = Some(kth(&mut win, (r - below) as usize));
            }
            resolved = true;
        }
    }
    if !resolved {
        // 2b./3. Exact fallback: the top-16 histogram in its own pass,
        //       then radix selection from exact bucket counts. A uniform
        //       multiset is detected in the same pass and needs no
        //       refinement.
        stats.fallback = true;
        stats.passes += 1;
        let (hist, uniform, first_key) = histogram_pass(xs, ys, earlier, n);
        if uniform {
            stats.uniform = true;
            values.fill(Some(slope_from_key(first_key)));
        } else {
            values = fallback_select(xs, ys, earlier, n, want, &hist, cfg.window_cap, stats);
        }
    }
    values
}

/// One exhaustive pass over the pair multiset (small inputs only), built in
/// the reference's row order so quickselect sees the same array.
fn build_all_keys(xs: &[f64], ys: &[f64], earlier: &[usize], n: usize) -> Vec<u64> {
    let m: usize = earlier.iter().sum();
    let mut keys = Vec::with_capacity(m);
    for j in 1..n {
        let k = earlier[j];
        let xj = xs[j];
        let yj = ys[j];
        for i in 0..k {
            keys.push(slope_key((yj - ys[i]) / (xj - xs[i])));
        }
    }
    keys
}

/// Exact uniformity check in one pass: true when every pairwise slope key
/// equals the first one. Used only when the pilot is constant, to confirm a
/// uniform multiset without copying m keys into the window buffer.
fn uniform_multiset(xs: &[f64], ys: &[f64], earlier: &[usize], n: usize) -> (bool, u64) {
    debug_assert!(earlier.iter().any(|&k| k > 0));
    let mut j0 = 1;
    while earlier[j0] == 0 {
        j0 += 1;
    }
    let first = slope_key((ys[j0] - ys[0]) / (xs[j0] - xs[0]));
    let mut uniform = true;
    for j in 1..n {
        let k = earlier[j];
        let xj = xs[j];
        let yj = ys[j];
        for i in 0..k {
            let key = slope_key((yj - ys[i]) / (xj - xs[i]));
            uniform &= key == first;
        }
    }
    (uniform, first)
}

/// Deterministic pilot: `p` pairs stratified evenly over the flat index
/// space `0..m` of the multiset, mapped to `(j, i)` with the prefix sums of
/// `earlier`. One pair is one unit of the flat index, so the sample mirrors
/// the true pair distribution instead of over-weighting early `j`.
fn pilot_keys(xs: &[f64], ys: &[f64], earlier: &[usize], n: usize, m: u64, p: usize) -> Vec<u64> {
    let mut cum = vec![0u64; n + 1];
    for j in 0..n {
        cum[j + 1] = cum[j] + earlier[j] as u64;
    }
    let p = (p as u64).min(m).max(1) as usize;
    let mut pilot = Vec::with_capacity(p);
    for t in 0..p {
        let pos = ((t as u128) * (m as u128) / (p as u128)) as u64;
        let j = cum[1..].partition_point(|&c| c <= pos);
        let i = (pos - cum[j]) as usize;
        debug_assert!(j < n && i < earlier[j]);
        pilot.push(slope_key((ys[j] - ys[i]) / (xs[j] - xs[i])));
    }
    pilot.sort_unstable();
    pilot
}

/// Window rule: for each target rank `k`, take the pilot quantile at
/// `q = k/m`, index `q*(p-1)`, and widen by the rank fraction
/// `delta = max(8*sqrt(0.25/p), 1/p)` on both sides. The window is the
/// union over targets (minimum lower bound, maximum upper bound), so it
/// covers all of them; a target clamped at an end simply reaches the end of
/// the pilot order. Exactness does not depend on this rule: the window pass
/// counts the true ranks and the fallback is exact.
fn choose_window(pilot: &[u64], m: u64, want: &[u64]) -> (u64, u64) {
    let p = pilot.len();
    let pf = p as f64;
    let delta = (WINDOW_DELTA_FACTOR * (0.25 / pf).sqrt()).max(1.0 / pf);
    let pm1 = (p - 1) as f64;
    let mut lo = u64::MAX;
    let mut hi = 0u64;
    for &r in want {
        let q = r as f64 / m as f64;
        let lo_f = ((q - delta) * pm1).floor().max(0.0);
        let hi_f = ((q + delta) * pm1).ceil().min(pm1);
        let mut i0 = (lo_f as usize).min(p - 1);
        let mut i1 = (hi_f as usize).min(p - 1);
        if i0 == i1 {
            if i0 > 0 {
                i0 -= 1;
            } else if i1 + 1 < p {
                i1 += 1;
            }
        }
        lo = lo.min(pilot[i0]);
        hi = hi.max(pilot[i1]);
    }
    (lo, hi)
}

/// One full pass without the histogram (the common path). The below-window
/// test is branchless: the slope stream is unordered, so a 50/50
/// data-dependent branch mispredicts on about half of the 38 M keys at
/// n=8700 (measured 0.22 s branchy against 0.06 s branchless). The wrapping
/// range check `key - lo <= hi - lo` is exact for every u64 interval,
/// including the full range. The only remaining branch is the in-window
/// push, taken on a few percent of keys.
///
/// The buffer is pre-reserved to the pilot prediction and its growth is
/// clamped so capacity never exceeds `cap`: growth-to-16 M plus the realloc
/// copy peaked at 244 MB without the reservation, instead of the ~90 MB the
/// exact in-window slice needs.
fn window_pass(
    xs: &[f64],
    ys: &[f64],
    earlier: &[usize],
    n: usize,
    lo_key: u64,
    hi_key: u64,
    cap: usize,
    reserve: usize,
) -> (u64, u64, Vec<u64>) {
    let span = hi_key.wrapping_sub(lo_key);
    let mut below = 0u64;
    let mut inside = 0u64;
    let mut win: Vec<u64> = Vec::with_capacity(reserve);
    for j in 1..n {
        let k = earlier[j];
        let xj = xs[j];
        let yj = ys[j];
        for (&xi, &yi) in xs[..k].iter().zip(ys[..k].iter()) {
            let key = slope_key((yj - yi) / (xj - xi));
            below += (key < lo_key) as u64;
            let d = key.wrapping_sub(lo_key);
            if d <= span {
                inside += 1;
                if win.len() < cap {
                    if win.len() == win.capacity() {
                        let extra = win.capacity().max(1024).min(cap - win.capacity());
                        win.reserve_exact(extra);
                    }
                    win.push(key);
                }
            }
        }
    }
    (below, inside, win)
}

/// The top-16 key histogram, in its own pass (fallback only). Also reports
/// whether the whole multiset is one key (a constant series, or a series
/// whose every pair slope is the same representable value); in that case no
/// refinement or collection is needed.
fn histogram_pass(xs: &[f64], ys: &[f64], earlier: &[usize], n: usize) -> (Vec<u64>, bool, u64) {
    histogram_pass_multi::<HIST_COPIES>(xs, ys, earlier, n)
}

/// Histogram with `C` private copies, summed exactly at the end. Slopes near
/// the median are highly concentrated, so a single accumulator serialises on
/// the load-add-store latency of a few hot counters; round-robin copies
/// break that chain. The sum is the same exact count as a single histogram.
fn histogram_pass_multi<const C: usize>(
    xs: &[f64],
    ys: &[f64],
    earlier: &[usize],
    n: usize,
) -> (Vec<u64>, bool, u64) {
    let mut first = 0u64;
    let mut have_first = false;
    for j in 1..n {
        if earlier[j] > 0 {
            first = slope_key((ys[j] - ys[0]) / (xs[j] - xs[0]));
            have_first = true;
            break;
        }
    }
    debug_assert!(have_first, "m > 0 guarantees a pair");
    let mut hists: [Vec<u64>; C] = std::array::from_fn(|_| vec![0u64; BUCKETS]);
    let mut uniform = true;
    for j in 1..n {
        let k = earlier[j];
        let xj = xs[j];
        let yj = ys[j];
        for (i, (&xi, &yi)) in xs[..k].iter().zip(ys[..k].iter()).enumerate() {
            let key = slope_key((yj - yi) / (xj - xi));
            uniform &= key == first;
            hists[i & (C - 1)][(key >> 48) as usize] += 1;
        }
    }
    let mut out = hists[0].clone();
    for hist in hists.iter().skip(1) {
        for (o, &h) in out.iter_mut().zip(hist.iter()) {
            *o += h;
        }
    }
    (out, uniform, first)
}

/// Exact prefix sums of a histogram, length `hist.len() + 1`.
fn cumulate(hist: &[u64]) -> Vec<u64> {
    let mut cum = Vec::with_capacity(hist.len() + 1);
    cum.push(0u64);
    let mut acc = 0u64;
    for &h in hist {
        acc += h;
        cum.push(acc);
    }
    cum
}

/// Exact fallback driven by the top-16 histogram. Groups of target ranks
/// sharing a bucket are resolved either by one bounded collection pass
/// (bucket size <= cap) or by radix refinement on the next sixteen bits,
/// recursively, until a digit group fits or all 64 bits are resolved (a
/// single key value). All counting is over the full pair multiset, so the
/// result does not depend on the pilot or the window.
fn fallback_select(
    xs: &[f64],
    ys: &[f64],
    earlier: &[usize],
    n: usize,
    want: &[u64],
    hist0: &[u64],
    cap: usize,
    stats: &mut PathStats,
) -> Vec<Option<f64>> {
    let cap = cap.min(FALLBACK_COLLECT_CAP);
    let cum0 = cumulate(hist0);
    let mut out = vec![None; want.len()];

    // `want` is sorted, so equal buckets are adjacent and groups come out
    // in ascending bucket order.
    let mut groups: Vec<(usize, Vec<(u64, usize)>)> = Vec::new();
    for (slot, &r) in want.iter().enumerate() {
        let b = cum0[1..].partition_point(|&c| c <= r);
        let local = r - cum0[b];
        match groups.last_mut() {
            Some(last) if last.0 == b => last.1.push((local, slot)),
            _ => groups.push((b, vec![(local, slot)])),
        }
    }
    debug_assert!(groups.windows(2).all(|w| w[0].0 < w[1].0));

    let mut thin: Vec<(usize, usize, usize)> = Vec::new(); // (digit, count, group)
    let mut fat: Vec<usize> = Vec::new();
    for (gi, &(b, _)) in groups.iter().enumerate() {
        let c = hist0[b];
        if c <= cap as u64 {
            thin.push((b, c as usize, gi));
        } else {
            fat.push(gi);
        }
    }

    if !thin.is_empty() {
        let digits: Vec<(usize, usize)> = thin.iter().map(|&(b, c, _)| (b, c)).collect();
        let (mut buf, starts) = collect_digits(xs, ys, earlier, n, None, 64, &digits, stats);
        let refs: Vec<&[(u64, usize)]> = thin
            .iter()
            .map(|&(_, _, gi)| groups[gi].1.as_slice())
            .collect();
        select_slices(&mut buf, &starts, &digits, &refs, &mut out);
    }

    for gi in fat {
        let (b, _) = &groups[gi];
        refine(
            xs,
            ys,
            earlier,
            n,
            (*b as u64) << 48,
            48,
            hist0[*b],
            &groups[gi].1,
            &mut out,
            cap,
            stats,
        );
    }
    out
}

/// Radix refinement inside a fixed key prefix. `shift` is the number of
/// unresolved low bits (48 -> 32 -> 16 -> 0); the high `64 - shift` bits of
/// every key in this range equal those of `prefix`, and the range holds
/// exactly `parent_count` keys. `items` holds the rank (local to this
/// range) and output slot of each target.
fn refine(
    xs: &[f64],
    ys: &[f64],
    earlier: &[usize],
    n: usize,
    prefix: u64,
    shift: u32,
    parent_count: u64,
    items: &[(u64, usize)],
    out: &mut [Option<f64>],
    cap: usize,
    stats: &mut PathStats,
) {
    if shift == 0 {
        // Every bit of the key is fixed: the order statistic is the prefix.
        let v = slope_from_key(prefix);
        for &(_, slot) in items {
            out[slot] = Some(v);
        }
        return;
    }

    let dshift = shift - 16;
    let plo = prefix >> shift;
    stats.passes += 1;
    // Branchless parent filter: out-of-parent keys land in slot 65536 of a
    // 65537-entry counter (one dummy per copy) instead of taking a branch
    // that defeats vectorisation of the key computation.
    let mut hists: [Vec<u64>; HIST_COPIES] = std::array::from_fn(|_| vec![0u64; BUCKETS + 1]);
    for j in 1..n {
        let k = earlier[j];
        let xj = xs[j];
        let yj = ys[j];
        for (i, (&xi, &yi)) in xs[..k].iter().zip(ys[..k].iter()).enumerate() {
            let key = slope_key((yj - yi) / (xj - xi));
            let inpar = (key >> shift == plo) as usize;
            let d = ((key >> dshift) & 0xFFFF) as usize;
            let idx = (d & inpar.wrapping_neg()) | ((1 - inpar) << 16);
            hists[i & (HIST_COPIES - 1)][idx] += 1;
        }
    }
    let mut hist = hists[0][..BUCKETS].to_vec();
    for h in hists.iter().skip(1) {
        for (o, &c) in hist.iter_mut().zip(h[..BUCKETS].iter()) {
            *o += c;
        }
    }
    debug_assert_eq!(hist.iter().sum::<u64>(), parent_count);

    let cum = cumulate(&hist);
    let mut groups: Vec<(usize, Vec<(u64, usize)>)> = Vec::new();
    for &(local, slot) in items {
        let d = cum[1..].partition_point(|&c| c <= local);
        let local2 = local - cum[d];
        match groups.last_mut() {
            Some(last) if last.0 == d => last.1.push((local2, slot)),
            _ => groups.push((d, vec![(local2, slot)])),
        }
    }
    debug_assert!(groups.windows(2).all(|w| w[0].0 < w[1].0));

    let mut thin: Vec<(usize, usize, usize)> = Vec::new();
    let mut fat: Vec<usize> = Vec::new();
    for (gi, &(d, _)) in groups.iter().enumerate() {
        let c = hist[d];
        if c <= cap as u64 {
            thin.push((d, c as usize, gi));
        } else {
            fat.push(gi);
        }
    }

    if !thin.is_empty() {
        let digits: Vec<(usize, usize)> = thin.iter().map(|&(d, c, _)| (d, c)).collect();
        let (mut buf, starts) =
            collect_digits(xs, ys, earlier, n, Some(prefix), shift, &digits, stats);
        let refs: Vec<&[(u64, usize)]> = thin
            .iter()
            .map(|&(_, _, gi)| groups[gi].1.as_slice())
            .collect();
        select_slices(&mut buf, &starts, &digits, &refs, out);
    }

    for gi in fat {
        let (d, _) = &groups[gi];
        let new_prefix = prefix | ((*d as u64) << dshift);
        refine(
            xs,
            ys,
            earlier,
            n,
            new_prefix,
            dshift,
            hist[*d],
            &groups[gi].1,
            out,
            cap,
            stats,
        );
    }
}

/// Collect every key matching one of the requested digit values within the
/// parent range, in one full pass. Keys are placed at their digit's
/// pre-computed slice, so after the pass each slice holds exactly that
/// digit's keys (unordered; selection is permutation-invariant).
fn collect_digits(
    xs: &[f64],
    ys: &[f64],
    earlier: &[usize],
    n: usize,
    parent: Option<u64>,
    shift: u32,
    digits: &[(usize, usize)],
    stats: &mut PathStats,
) -> (Vec<u64>, Vec<usize>) {
    let total: usize = digits.iter().map(|&(_, c)| c).sum();
    let mut buf = vec![0u64; total];
    let mut cursor: Vec<usize> = Vec::with_capacity(digits.len());
    let mut starts: Vec<usize> = Vec::with_capacity(digits.len());
    let mut acc = 0usize;
    for &(_, c) in digits {
        starts.push(acc);
        cursor.push(acc);
        acc += c;
    }
    // Slot 65536 acts as "not in the parent/not a wanted digit"; keeping
    // the parent test branchless lets the key computation vectorise. Only
    // the rare final store is conditional.
    let mut map = vec![0u16; BUCKETS + 1];
    for (pos, &(d, _)) in digits.iter().enumerate() {
        map[d] = pos as u16 + 1;
    }
    let dshift = shift - 16;
    let plo = parent.map(|p| p >> shift);
    stats.passes += 1;
    for j in 1..n {
        let k = earlier[j];
        let xj = xs[j];
        let yj = ys[j];
        for (&xi, &yi) in xs[..k].iter().zip(ys[..k].iter()) {
            let key = slope_key((yj - yi) / (xj - xi));
            let inpar = match plo {
                Some(pl) => (key >> shift == pl) as usize,
                None => 1,
            };
            let d = ((key >> dshift) & 0xFFFF) as usize;
            let idx = (d & inpar.wrapping_neg()) | ((1 - inpar) << 16);
            let g = map[idx];
            if g != 0 {
                let pos = g as usize - 1;
                let at = cursor[pos];
                buf[at] = key;
                cursor[pos] = at + 1;
            }
        }
    }
    for (pos, &(_, c)) in digits.iter().enumerate() {
        debug_assert_eq!(cursor[pos], starts[pos] + c);
    }
    stats.collected_keys += total;
    (buf, starts)
}

/// Select the requested local ranks from each collected digit slice.
fn select_slices(
    buf: &mut [u64],
    starts: &[usize],
    digits: &[(usize, usize)],
    items: &[&[(u64, usize)]],
    out: &mut [Option<f64>],
) {
    for (i, group) in items.iter().enumerate() {
        let start = starts[i];
        let end = start + digits[i].1;
        let slice = &mut buf[start..end];
        for &(local, slot) in group.iter() {
            debug_assert!((local as usize) < slice.len());
            let (_, value, _) = slice.select_nth_unstable(local as usize);
            out[slot] = Some(slope_from_key(*value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // scipy.stats.norm.ppf(0.025) and its positive counterpart.
    const Z: f64 = -1.959_963_984_540_054_5;
    const Z_POS: f64 = 1.959_963_984_540_054_5;

    fn plain_cfg() -> TheilSenConfig {
        TheilSenConfig {
            full_below_pairs: u64::MAX,
            pilot_pairs: 1,
            window_cap: 1,
        }
    }

    fn cfg(full_below_pairs: u64, pilot_pairs: usize, window_cap: usize) -> TheilSenConfig {
        TheilSenConfig {
            full_below_pairs,
            pilot_pairs,
            window_cap,
        }
    }

    fn interval(x: &[f64], y: &[f64], z: f64, c: TheilSenConfig) -> TheilSenInterval {
        interval_stats(x, y, z, c).expect("finite inputs").0
    }

    fn assert_bits(a: &TheilSenInterval, b: &TheilSenInterval, what: &str) {
        assert_eq!(a.slope.to_bits(), b.slope.to_bits(), "slope: {what}");
        assert_eq!(a.low.to_bits(), b.low.to_bits(), "low: {what}");
        assert_eq!(a.high.to_bits(), b.high.to_bits(), "high: {what}");
        assert_eq!(a.n_pairs, b.n_pairs, "n_pairs: {what}");
        assert_eq!(a.degenerate, b.degenerate, "degenerate: {what}");
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(state: &mut u64) -> f64 {
        ((splitmix64(state) >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// The canonical benchmark record: hourly step plus jitter, a linear
    /// trend and gaussian noise. The goldens below were captured from this
    /// exact sequence; it uses libm (ln, cos), hence the macOS-aarch64 gate.
    fn canonical(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut s = 12345u64;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let u = unit(&mut s).max(f64::MIN_POSITIVE);
            let v = unit(&mut s);
            let noise = (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos();
            let xi = i as f64 + 0.25 * (unit(&mut s) - 0.5);
            x.push(xi);
            y.push(0.5 + 0.0002 * xi + 0.1 * noise);
        }
        (x, y)
    }

    fn arange(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64).collect()
    }

    /// Generators spanning the branches: distinct and duplicate x, tied and
    /// constant y, exact slopes, both zeros, overflow-infinite slopes and an
    /// unsorted record.
    fn named_cases(n: usize) -> Vec<(&'static str, Vec<f64>, Vec<f64>)> {
        let x = arange(n);
        let duplicate_x: Vec<f64> = (0..n).map(|i| (i / 2) as f64).collect();
        let duplicate_y: Vec<f64> = (0..n).map(|i| ((i * 37) % 101) as f64 * 0.1).collect();
        let rounded_y: Vec<f64> = (0..n).map(|i| ((i * 13) % 97) as f64 * 0.1).collect();
        let (canon_x, canon_y) = canonical(n);
        let (mut rev_x, mut rev_y) = (canon_x.clone(), canon_y.clone());
        rev_x.reverse();
        rev_y.reverse();
        vec![
            ("canonical", canon_x, canon_y),
            ("duplicate-x", duplicate_x, duplicate_y),
            ("rounded-y", x.clone(), rounded_y),
            ("constant-y", x.clone(), vec![5.0; n]),
            ("linear2", x.clone(), x.iter().map(|&v| 2.0 * v).collect()),
            ("linear0", x.clone(), vec![0.0; n]),
            (
                "block-zeros",
                x.clone(),
                (0..n).map(|i| if i < n / 2 { 0.0 } else { -0.0 }).collect(),
            ),
            (
                "mixed-zeros",
                x.clone(),
                (0..n)
                    .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
                    .collect(),
            ),
            (
                "overflow",
                x.clone(),
                (0..n)
                    .map(|i| if i % 2 == 0 { 1e308 } else { -1e308 })
                    .collect(),
            ),
            ("unsorted", rev_x, rev_y),
        ]
    }

    fn prepare(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<usize>, u64) {
        let n = x.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| x[a].total_cmp(&x[b]));
        let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
        let ys: Vec<f64> = order.iter().map(|&i| y[i]).collect();
        let earlier: Vec<usize> = (0..n).map(|j| xs.partition_point(|&v| v < xs[j])).collect();
        let m: u64 = earlier.iter().map(|&k| k as u64).sum();
        (xs, ys, earlier, m)
    }

    /// Independent oracle: sort every key, decode the four ranks and apply
    /// the reference's assembly. Isolates selection from quickselect order.
    fn oracle(x: &[f64], y: &[f64], z: f64) -> TheilSenInterval {
        let (xs, ys, earlier, m) = prepare(x, y);
        let n = xs.len();
        if m == 0 {
            return TheilSenInterval {
                slope: f64::NAN,
                low: f64::NAN,
                high: f64::NAN,
                n_pairs: 0,
                degenerate: true,
            };
        }
        let mut keys = build_all_keys(&xs, &ys, &earlier, n);
        keys.sort_unstable();
        let mut xsorted = xs.clone();
        xsorted.sort_unstable_by(|a, b| a.total_cmp(b));
        let mut ysorted = ys.clone();
        ysorted.sort_unstable_by(|a, b| a.total_cmp(b));
        let nn = n as i128;
        let base = nn * (nn - 1) * (2 * nn + 5);
        let sigsq = (1.0 / 18.0) * ((base - tie_term(&xsorted) - tie_term(&ysorted)) as f64);
        let sigma = sigsq.max(0.0).sqrt();
        let nt = m as f64;
        let ru_f = ((nt - z * sigma) / 2.0).round_ties_even();
        let rl_f = ((nt + z * sigma) / 2.0).round_ties_even();
        let ru = py_index((ru_f as i64).min(m as i64 - 1).max(0), m) as usize;
        let rl = py_index((rl_f as i64 - 1).max(0), m) as usize;
        let mid_hi = (m / 2) as usize;
        let mid_lo = ((m - 1) / 2) as usize;
        let get = |k: usize| slope_from_key(keys[k]);
        let mid_hi_v = get(mid_hi);
        let mid_lo_v = if m % 2 == 0 { get(mid_lo) } else { mid_hi_v };
        let high = if ru > mid_hi { get(ru) } else { mid_hi_v };
        let low = if rl < mid_hi { get(rl) } else { mid_hi_v };
        let slope = if m % 2 == 0 {
            (mid_lo_v + mid_hi_v) / 2.0
        } else {
            mid_hi_v
        };
        TheilSenInterval {
            slope,
            low,
            high,
            n_pairs: m,
            degenerate: false,
        }
    }

    #[test]
    fn configs_match_the_materialised_path() {
        for &n in &[2usize, 3, 4, 5, 17, 100] {
            for (name, x, y) in named_cases(n) {
                for &z in &[Z, -3.0, 0.0, Z_POS, -100.0] {
                    let want = interval(&x, &y, z, plain_cfg());
                    for c in [
                        THEIL_SEN_DEFAULT_CONFIG,
                        cfg(0, 8, 4),
                        cfg(0, 64, 32),
                        cfg(0, 1, 1),
                        cfg(0, 1024, 1 << 20),
                    ] {
                        let (got, st) = interval_stats(&x, &y, z, c).expect("finite");
                        assert_bits(&got, &want, &format!("{name} n={n} z={z} cfg={c:?}"));
                        assert!(st.passes <= 19, "pass bound: {name} n={n} {c:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn sort_oracle_matches_every_path() {
        for n in 2..=60 {
            for (name, x, y) in named_cases(n) {
                for &z in &[Z, -3.0, Z_POS] {
                    let want = oracle(&x, &y, z);
                    for c in [
                        THEIL_SEN_DEFAULT_CONFIG,
                        plain_cfg(),
                        cfg(0, 8, 4),
                        cfg(0, 1, 1),
                    ] {
                        let got = interval(&x, &y, z, c);
                        assert_bits(&got, &want, &format!("{name} n={n} z={z} cfg={c:?}"));
                    }
                }
            }
        }
    }

    #[test]
    fn pair_cutoff_is_on_m_not_n() {
        let (x, y) = canonical(2896);
        let (got, st) = interval_stats(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG).unwrap();
        assert!(st.full_materialise);
        assert_eq!(got.n_pairs, 4_191_960);
        assert_bits(&got, &interval(&x, &y, Z, plain_cfg()), "2896");

        let (x, y) = canonical(2897);
        let (got, st) = interval_stats(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG).unwrap();
        assert!(!st.full_materialise);
        assert_eq!(got.n_pairs, 4_194_856);
        assert_bits(&got, &interval(&x, &y, Z, plain_cfg()), "2897");

        // Tie-heavy x: the cutoff counts pairs, not points.
        let mut x = vec![0.0; 2000];
        x.extend(vec![1.0; 2000]);
        let y: Vec<f64> = (0..4000).map(|i| 0.1 * i as f64).collect();
        let (got, st) = interval_stats(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG).unwrap();
        assert!(st.full_materialise);
        assert_eq!(got.n_pairs, 4_000_000);

        let mut x = vec![0.0; 2050];
        x.extend(vec![1.0; 2050]);
        let y: Vec<f64> = (0..4100).map(|i| 0.1 * i as f64).collect();
        let (got, st) = interval_stats(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG).unwrap();
        assert!(!st.full_materialise);
        assert_eq!(got.n_pairs, 4_202_500);
        assert_bits(&got, &interval(&x, &y, Z, plain_cfg()), "4100");
    }

    #[test]
    fn window_path_handles_tied_days_and_tied_values() {
        let n = 3000;
        let mut x: Vec<f64> = (0..n).map(|i| ((i * 37) % 1500) as f64).collect();
        x.sort_unstable_by(|a, b| a.total_cmp(b));
        let y: Vec<f64> = (0..n).map(|i| ((i * 13) % 97) as f64 * 0.1).collect();
        let want = interval(&x, &y, Z, plain_cfg());
        let (got, st) = interval_stats(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG).unwrap();
        assert!(!st.full_materialise);
        assert_bits(&got, &want, "tied days");
    }

    #[test]
    fn window_path_is_permutation_invariant() {
        let n = 4200;
        let (x, y) = canonical(n);
        let base = interval(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG);
        let (mut rx, mut ry) = (x.clone(), y.clone());
        rx.reverse();
        ry.reverse();
        let mut sx = Vec::with_capacity(n);
        let mut sy = Vec::with_capacity(n);
        for k in 0..n {
            let i = (k * 13 + 777) % n;
            sx.push(x[i]);
            sy.push(y[i]);
        }
        for (vx, vy) in [(rx, ry), (sx, sy)] {
            assert_bits(
                &interval(&vx, &vy, Z, THEIL_SEN_DEFAULT_CONFIG),
                &base,
                "permuted",
            );
        }
        assert_bits(
            &base,
            &interval(&x, &y, Z, plain_cfg()),
            "4200 materialised",
        );
    }

    #[test]
    fn uniform_slopes_short_circuit() {
        let n = 3000;
        let x = arange(n);
        for (a, expected) in [(2.0f64, 2.0f64), (-3.5, -3.5), (0.0, 0.0)] {
            let y: Vec<f64> = x.iter().map(|&v| a * v).collect();
            let (got, st) = interval_stats(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG).unwrap();
            assert!(st.uniform, "a={a}");
            assert_eq!(st.passes, 1, "a={a}");
            assert_eq!(got.slope.to_bits(), expected.to_bits(), "a={a}");
            assert_eq!(got.low.to_bits(), expected.to_bits(), "a={a} low");
            assert_eq!(got.high.to_bits(), expected.to_bits(), "a={a} high");
            assert_eq!(got.n_pairs, 4_498_500);
        }
        let y = vec![5.0; n];
        let (got, st) = interval_stats(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG).unwrap();
        assert!(st.uniform);
        assert_eq!(st.passes, 1);
        for v in [got.slope, got.low, got.high] {
            assert_eq!(v.to_bits(), 0.0f64.to_bits());
        }
    }

    #[test]
    fn signed_zero_keys_keep_bits() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let neg_zero = (-0.0f64).to_bits();
        for c in [THEIL_SEN_DEFAULT_CONFIG, cfg(0, 2, 1 << 20), cfg(0, 1, 1)] {
            let y = [0.0, 0.0, -0.0, -0.0];
            let r = interval(&x, &y, Z, c);
            assert_eq!(r.slope.to_bits(), neg_zero, "cfg={c:?} slope");
            assert_eq!(r.low.to_bits(), neg_zero, "cfg={c:?} low");
            assert_eq!(r.high.to_bits(), neg_zero, "cfg={c:?} high");
        }
        let y = [0.0, -0.0, 0.0, -0.0];
        let r = interval(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG);
        assert_eq!(r.low.to_bits(), neg_zero);
        assert_eq!(r.slope.to_bits(), 0.0f64.to_bits());
        assert_eq!(r.high.to_bits(), 0.0f64.to_bits());

        let x3 = [0.0, 1.0, 2.0];
        let y3 = [-0.0, 0.0, -0.0];
        let r = interval(&x3, &y3, Z, THEIL_SEN_DEFAULT_CONFIG);
        for v in [r.slope, r.low, r.high] {
            assert_eq!(v.to_bits(), 0.0f64.to_bits());
        }
    }

    #[test]
    fn even_m_averaging_uses_one_add_then_div() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 1.0, 2.0, 10.0];
        let r = interval(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG);
        let expected = (1.0f64 + 10.0f64 / 3.0) / 2.0;
        assert_eq!(r.slope.to_bits(), expected.to_bits());
        assert_bits(&r, &oracle(&x, &y, Z), "even m");

        let (x, y) = (arange(6), vec![0.0, 1.0, 4.0, 9.0, 16.0, 25.0]);
        assert_bits(
            &interval(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG),
            &oracle(&x, &y, Z),
            "quadratic",
        );
    }

    #[test]
    fn interval_ranks_and_positive_z_follow_the_reference() {
        for n in [5usize, 300, 1200] {
            let x = arange(n);
            let y: Vec<f64> = (0..n).map(|i| ((i * 13) % 97) as f64 * 0.1).collect();
            for &z in &[Z, -3.0, 0.0, Z_POS, -100.0] {
                let want = oracle(&x, &y, z);
                for c in [THEIL_SEN_DEFAULT_CONFIG, plain_cfg(), cfg(0, 64, 32)] {
                    let got = interval(&x, &y, z, c);
                    assert_bits(&got, &want, &format!("n={n} z={z} cfg={c:?}"));
                }
            }
        }
    }

    #[test]
    fn deep_radix_fallback_terminates_at_full_key() {
        for (name, x, y) in named_cases(300) {
            if !["canonical", "duplicate-x", "block-zeros", "overflow"].contains(&name) {
                continue;
            }
            let want = interval(&x, &y, Z, plain_cfg());
            let (got, st) = interval_stats(&x, &y, Z, cfg(0, 1, 1)).unwrap();
            assert!(st.fallback, "{name}");
            assert!(
                st.passes >= 2 && st.passes <= 19,
                "{name} passes={}",
                st.passes
            );
            assert_bits(&got, &want, name);
        }
    }

    #[test]
    fn window_cap_and_capacity_are_bounded() {
        let (x, y) = canonical(300);
        let (xs, ys, earlier, m) = prepare(&x, &y);
        let (below, inside, win) = window_pass(&xs, &ys, &earlier, 300, 0, u64::MAX, 64, 10);
        assert_eq!(below, 0);
        assert_eq!(inside, m);
        assert_eq!(win.len(), 64);
        assert!(win.capacity() <= 64);
        let want = interval(&x, &y, Z, plain_cfg());
        assert_bits(&interval(&x, &y, Z, cfg(0, 1024, 64)), &want, "cap 64");
    }

    #[test]
    fn pilot_is_deterministic_and_permutation_invariant() {
        let n = 3000;
        let (x, y) = canonical(n);
        let (base, st) = interval_stats(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG).unwrap();
        let (mut rx, mut ry) = (x.clone(), y.clone());
        rx.reverse();
        ry.reverse();
        let mut sx = Vec::with_capacity(n);
        let mut sy = Vec::with_capacity(n);
        for k in 0..n {
            let i = (k * 7 + 123) % n;
            sx.push(x[i]);
            sy.push(y[i]);
        }
        for (vx, vy) in [(rx, ry), (sx, sy)] {
            let (got, s) = interval_stats(&vx, &vy, Z, THEIL_SEN_DEFAULT_CONFIG).unwrap();
            assert_bits(&got, &base, "permutation");
            assert_eq!(s.passes, st.passes);
            assert_eq!(s.window_keys, st.window_keys);
            assert_eq!(s.predicted_inside, st.predicted_inside);
            assert_eq!(s.window, st.window);
        }
    }

    #[test]
    fn degenerate_and_tiny_m_unchanged() {
        let r = interval(
            &[2.0; 4],
            &[1.0, 2.0, 3.0, 4.0],
            Z,
            THEIL_SEN_DEFAULT_CONFIG,
        );
        assert!(r.degenerate);
        assert_eq!(r.n_pairs, 0);
        assert!(r.slope.is_nan() && r.low.is_nan() && r.high.is_nan());

        // -0.0 and +0.0 x compare equal under `<`, so this is degenerate.
        let r = interval(
            &[-0.0, 0.0, 0.0],
            &[1.0, 2.0, 3.0],
            Z,
            THEIL_SEN_DEFAULT_CONFIG,
        );
        assert!(r.degenerate);

        let r = interval(&[0.0, 1.0], &[2.0, 3.0], Z, THEIL_SEN_DEFAULT_CONFIG);
        assert_eq!(r.n_pairs, 1);
        for v in [r.slope, r.low, r.high] {
            assert_eq!(v.to_bits(), 1.0f64.to_bits());
        }

        let mut x = vec![0.0; 99];
        x.push(1.0);
        let y = arange(100);
        let r = interval(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG);
        assert_eq!(r.n_pairs, 99);
        assert_bits(&r, &oracle(&x, &y, Z), "one row");
    }

    #[test]
    fn infinity_and_nan_slopes_are_preserved() {
        for n in [1500usize, 3000] {
            let x = arange(n);
            let y: Vec<f64> = (0..n)
                .map(|i| if i % 2 == 0 { 1e308 } else { -1e308 })
                .collect();
            let want = interval(&x, &y, Z, plain_cfg());
            for c in [THEIL_SEN_DEFAULT_CONFIG, cfg(0, 1, 1)] {
                let got = interval(&x, &y, Z, c);
                assert_bits(&got, &want, &format!("overflow n={n} cfg={c:?}"));
            }
            assert_eq!(want.slope.to_bits(), 0.0f64.to_bits());
        }

        // inf / inf yields a NaN slope; only in-process bit equality is
        // pinned, not a NaN payload across hosts.
        let x = [-1e308, 0.0, 1e308];
        let y = x;
        let want = interval(&x, &y, Z, plain_cfg());
        for c in [THEIL_SEN_DEFAULT_CONFIG, cfg(0, 1, 1)] {
            assert_bits(&interval(&x, &y, Z, c), &want, &format!("inf/inf {c:?}"));
        }
    }

    #[test]
    fn error_precedence_unchanged() {
        assert_eq!(
            theil_sen_interval(&[1.0, 2.0], &[1.0], Z),
            Err(TheilSenError::LengthMismatch)
        );
        assert_eq!(
            theil_sen_interval(&[1.0, f64::NAN], &[1.0], Z),
            Err(TheilSenError::LengthMismatch)
        );
        assert_eq!(
            theil_sen_interval(&[], &[], Z),
            Err(TheilSenError::TooShort)
        );
        assert_eq!(
            theil_sen_interval(&[1.0], &[1.0], Z),
            Err(TheilSenError::TooShort)
        );
        assert_eq!(
            theil_sen_interval(&[1.0, f64::NAN], &[1.0, 2.0], Z),
            Err(TheilSenError::NonFinite)
        );
        assert_eq!(
            theil_sen_interval(&[1.0, f64::INFINITY], &[1.0, 2.0], Z),
            Err(TheilSenError::NonFinite)
        );
        assert_eq!(
            theil_sen_interval(&[1.0, 2.0], &[1.0, 2.0], f64::NAN),
            Err(TheilSenError::NonFinite)
        );
        assert_eq!(
            TheilSenError::LengthMismatch.to_string(),
            "x and y must have the same length"
        );
        assert_eq!(
            TheilSenError::TooShort.to_string(),
            "at least two points are required"
        );
        assert_eq!(
            TheilSenError::NonFinite.to_string(),
            "x, y and z_alpha must be finite"
        );
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn canonical_bits_match_the_measured_goldens() {
        let cases = [
            (
                1200usize,
                "3f2b16b0e88f7a66",
                "3f28e97580f3f45a",
                "3f2d40ac5a180ba3",
                719_400u64,
            ),
            (
                2896,
                "3f2a4ca508c61c56",
                "3f29b88f0a719b70",
                "3f2ae1517eb52667",
                4_191_960,
            ),
            (
                2897,
                "3f2a4a3c06343909",
                "3f29b61895273c3b",
                "3f2adeb58a1ae9f4",
                4_194_856,
            ),
            (
                3000,
                "3f2a303977e8bf7a",
                "3f29a445a65aede6",
                "3f2abd1356c1d8a0",
                4_498_500,
            ),
        ];
        for (n, slope, low, high, m) in cases {
            let (x, y) = canonical(n);
            let r = interval(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG);
            let bits = |s: &str| u64::from_str_radix(s, 16).unwrap();
            assert_eq!(r.slope.to_bits(), bits(slope), "slope n={n}");
            assert_eq!(r.low.to_bits(), bits(low), "low n={n}");
            assert_eq!(r.high.to_bits(), bits(high), "high n={n}");
            assert_eq!(r.n_pairs, m, "m n={n}");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    #[ignore = "release-only: 8.7k-52.4k points, 0.06-5.4 s each"]
    fn large_canonical_goldens() {
        let cases = [
            (
                8700usize,
                "3f2a24a683abaa71",
                "3f2a07f31ea7d185",
                "3f2a415bc7361955",
                37_840_650u64,
            ),
            (
                26280,
                "3f2a385e501ace62",
                "3f2a32e0c392d26e",
                "3f2a3dd97a4ad6fa",
                345_306_060,
            ),
            (
                52400,
                "3f2a369aa361c42f",
                "3f2a34a6e4b8b4d2",
                "3f2a388ef56951e6",
                1_372_853_800,
            ),
        ];
        for (n, slope, low, high, m) in cases {
            let (x, y) = canonical(n);
            let r = interval(&x, &y, Z, THEIL_SEN_DEFAULT_CONFIG);
            let bits = |s: &str| u64::from_str_radix(s, 16).unwrap();
            assert_eq!(r.slope.to_bits(), bits(slope), "slope n={n}");
            assert_eq!(r.low.to_bits(), bits(low), "low n={n}");
            assert_eq!(r.high.to_bits(), bits(high), "high n={n}");
            assert_eq!(r.n_pairs, m, "m n={n}");
        }
    }
}
