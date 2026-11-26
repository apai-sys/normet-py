//! Exact Theil-Sen slope and Sen (1968) interval order statistics.
//!
//! `scipy.stats.theilslopes`, normet's `_theil_sen_large` and R's Fenwick
//! path all select the same order statistics from the multiset of pairwise
//! slopes `(y_j - y_i) / (x_j - x_i)` over pairs with `x_i < x_j`. This
//! module performs the same IEEE-754 arithmetic in Rust, so the slope and
//! the two interval endpoints are bit-identical to the hosts' values; only
//! the overhead differs. The multiset is materialised (8 bytes per pair, as
//! in the Python path) and selected in place -- an order-statistic-only
//! variant that avoids the `O(n^2)` allocation is the planned follow-up.
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
/// costs nothing).
fn py_index(rank: i64, len: usize) -> usize {
    if rank < 0 {
        (len as i64 + rank) as usize
    } else {
        rank as usize
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
///   (the hosts pass `ppf(0.025)` ≈ -1.9599639845400545). The interval is
///   the 95 % one scipy and the R wrapper also produce.
///
/// Pairs with equal `x` are excluded, and ties are accounted for exactly as
/// scipy's `1/18 *` tie-corrected variance does. The returned values equal
/// scipy's `theilslopes(y, x)` bit for bit on finite inputs.
pub fn theil_sen_interval(
    x: &[f64],
    y: &[f64],
    z_alpha: f64,
) -> Result<TheilSenInterval, TheilSenError> {
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
    let m: usize = earlier.iter().sum();

    if m == 0 {
        return Ok(TheilSenInterval {
            slope: f64::NAN,
            low: f64::NAN,
            high: f64::NAN,
            n_pairs: 0,
            degenerate: true,
        });
    }

    // Slopes are stored as order-preserving keys, not floats: the values are
    // only ever compared and then read back, and integer selection is much
    // faster than a float comparator at hundreds of millions of elements.
    let mut keys: Vec<u64> = Vec::with_capacity(m);
    for j in 1..n {
        let k = earlier[j];
        let xj = xs[j];
        let yj = ys[j];
        for i in 0..k {
            keys.push(slope_key((yj - ys[i]) / (xj - xs[i])));
        }
    }
    debug_assert_eq!(keys.len(), m);

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
    let mid_hi = m / 2;
    let mid_lo = (m - 1) / 2;

    // Locate the median first: that selection partitions the array, so the
    // interval ranks (both on the median's far side) are selected from
    // half-sized slices. `ru >= mid_hi` and `rl <= mid_hi` follow from the
    // negative quantile before the clamps; equality is the tiny-record case.
    let mid_hi_v = kth(&mut keys, mid_hi);
    let mid_lo_v = if m % 2 == 0 {
        kth(&mut keys[..mid_hi], mid_lo)
    } else {
        mid_hi_v
    };
    let high = if ru > mid_hi {
        kth(&mut keys[mid_hi + 1..], ru - mid_hi - 1)
    } else {
        mid_hi_v
    };
    let low = if rl < mid_hi {
        kth(&mut keys[..mid_hi], rl)
    } else {
        mid_hi_v
    };
    let slope = if m % 2 == 0 {
        (mid_lo_v + mid_hi_v) / 2.0
    } else {
        mid_hi_v
    };

    Ok(TheilSenInterval {
        slope,
        low,
        high,
        n_pairs: m as u64,
        degenerate: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // scipy.stats.norm.ppf(0.025)
    const Z: f64 = -1.959_963_984_540_054_5;

    fn run(x: &[f64], y: &[f64]) -> TheilSenInterval {
        theil_sen_interval(x, y, Z).expect("finite inputs")
    }

    #[test]
    fn quadratic_matches_scipy() {
        let x: Vec<f64> = (0..10).map(|v| v as f64).collect();
        let y = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0];
        let r = run(&x, &y);
        assert_eq!(r.slope, 9.0);
        assert_eq!(r.low, 6.0);
        assert_eq!(r.high, 12.0);
        assert_eq!(r.n_pairs, 45);
        assert!(!r.degenerate);
    }

    #[test]
    fn ties_match_scipy() {
        let x = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let y = [5.0, 7.0, 4.0, 6.0, 7.0, 8.0, 9.0, 7.0, 10.0, 12.0];
        let r = run(&x, &y);
        assert_eq!(r.slope, 1.291_666_666_666_666_5);
        assert_eq!(r.low, 0.5);
        assert_eq!(r.high, 2.0);
        assert_eq!(r.n_pairs, 40);
    }

    #[test]
    fn irregular_spacing_match_scipy() {
        let x = [
            0.3, 1.7, 2.2, 4.9, 5.1, 7.7, 8.0, 10.4, 11.1, 13.6, 15.0,
        ];
        let y = [
            2.1, 3.3, 2.9, 5.4, 4.8, 6.1, 7.0, 6.2, 7.4, 7.9, 9.1,
        ];
        let r = run(&x, &y);
        assert_eq!(r.slope, 0.436_090_225_563_909_86);
        assert_eq!(r.low, 0.333_333_333_333_333_3);
        assert_eq!(r.high, 0.5625);
    }

    #[test]
    fn unsorted_x_is_sorted_internally() {
        let x = [5.0, 1.0, 9.0, 3.0, 7.0, 0.0, 8.0, 2.0, 6.0, 4.0];
        let y = [50.0, 10.0, 90.0, 30.0, 70.0, 0.0, 80.0, 20.0, 60.0, 40.0];
        let r = run(&x, &y);
        assert_eq!(r.slope, 10.0);
        assert_eq!(r.low, 10.0);
        assert_eq!(r.high, 10.0);
    }

    #[test]
    fn identical_x_is_degenerate() {
        let r = run(&[2.0, 2.0, 2.0, 2.0], &[1.0, 2.0, 3.0, 4.0]);
        assert!(r.degenerate);
        assert_eq!(r.n_pairs, 0);
        assert!(r.slope.is_nan() && r.low.is_nan() && r.high.is_nan());
    }

    #[test]
    fn errors() {
        assert_eq!(
            theil_sen_interval(&[1.0, 2.0], &[1.0], Z),
            Err(TheilSenError::LengthMismatch)
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
            theil_sen_interval(&[1.0, 2.0], &[1.0, 2.0], f64::NAN),
            Err(TheilSenError::NonFinite)
        );
    }
}
