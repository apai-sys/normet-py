//! Exact CRPS of quantile and ensemble rows.
//!
//! `normet` scores a forecast by integrating `(F - 1{x >= y})^2` over the
//! piecewise-linear CDF its quantile points define (`_row_crps` in
//! `normet.analysis.verification`). That scorer walks the rows in a Python
//! loop and builds about twenty temporary arrays per row; a normalise run
//! scores tens of thousands of rows, so the loop overhead, not the
//! arithmetic, dominates. This module repeats the same arithmetic in native
//! code with scratch buffers reused across rows, and reproduces the host's
//! numbers bit for bit: each row's segment integral is summed in numpy's
//! pairwise order, and the row scores are accumulated in row order before
//! the final division by the row count.
//!
//! The kernel is model-free and free of RNG, pandas and R types, so the R
//! wrapper can call it with the same contract: finite values, finite levels
//! and observations in `(0, 1)`-style order, and the same numbers back.

use std::fmt;

/// Input problems; all map to Python `ValueError` / R errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrpsError {
    /// The flattened matrix, level vector and observation vector disagree.
    LengthMismatch,
    /// No rows to average.
    Empty,
    /// A non-finite value; the hosts drop missing rows before scoring.
    NonFinite,
}

impl fmt::Display for CrpsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch => {
                write!(f, "values, levels and observations disagree in shape")
            }
            Self::Empty => write!(f, "at least one forecast row is required"),
            Self::NonFinite => write!(f, "values, levels and observations must be finite"),
        }
    }
}

impl std::error::Error for CrpsError {}

/// numpy's summation block size (`PW_BLOCKSIZE` in numpy's loops).
const PW_BLOCKSIZE: usize = 128;

/// numpy's pairwise summation, reproduced exactly.
///
/// `np.sum` on a contiguous float64 array is not a sequential sum: for
/// fewer than eight values it adds in order; for up to 128 it keeps eight
/// independent accumulators and folds them in a fixed tree; above that it
/// splits at half, rounded down to a multiple of eight. The score is a mean
/// of such sums, so a sequential sum in Rust would differ in the last few
/// ulps. This mirrors the host's algorithm so the two backends return the
/// same bits.
pub(crate) fn pairwise_sum(a: &[f64]) -> f64 {
    // numpy reduces a contiguous array through its ufunc buffer, so arrays
    // longer than NPY_BUFSIZE (8192 elements) are summed in consecutive
    // chunks and the chunk totals are added sequentially; the tree alone
    // differs in the last ulp above that length (measured: 16/40 random
    // arrays). Both paths are identical at or below 8192 terms.
    if a.len() > PW_BUFFER {
        return a.chunks(PW_BUFFER).map(pairwise_sum_tree).sum();
    }
    pairwise_sum_tree(a)
}

/// numpy's `NPY_BUFSIZE`, in elements: the ufunc reduction buffer.
const PW_BUFFER: usize = 8192;

/// One buffer's worth of numpy's pairwise summation tree.
fn pairwise_sum_tree(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        return res;
    }
    if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        let end = n - (n % 8);
        while i < end {
            r[0] += a[i];
            r[1] += a[i + 1];
            r[2] += a[i + 2];
            r[3] += a[i + 3];
            r[4] += a[i + 4];
            r[5] += a[i + 5];
            r[6] += a[i + 6];
            r[7] += a[i + 7];
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum_tree(&a[..n2]) + pairwise_sum_tree(&a[n2..])
}

/// Scratch buffers reused for every row, keeping the row loop allocation-free.
#[derive(Default)]
pub struct CrpsScratch {
    pairs: Vec<(f64, f64)>,
    values: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    grid: Vec<f64>,
    segments: Vec<f64>,
}

/// One-sided CDF limits `(left, right)` at a break point `g`.
///
/// `values`/`lo`/`hi` are the distinct forecast values and their group's
/// lowest and highest quantile levels. At a data value the CDF jumps from
/// `lo` to `hi` (from 0 at the smallest value, to 1 at the largest); at a
/// point between two values it is the straight line the quantile function
/// interpolates there.
fn limits(values: &[f64], lo: &[f64], hi: &[f64], g: f64) -> (f64, f64) {
    let m = values.len();
    if g < values[0] {
        return (0.0, 0.0);
    }
    if g > values[m - 1] {
        return (1.0, 1.0);
    }
    if g == values[0] {
        return if m == 1 { (0.0, 1.0) } else { (0.0, hi[0]) };
    }
    if g == values[m - 1] {
        return (lo[m - 1], 1.0);
    }
    let j = values.partition_point(|&v| v <= g) - 1;
    if g == values[j] {
        return (lo[j], hi[j]);
    }
    let t = (g - values[j]) / (values[j + 1] - values[j]);
    let line = hi[j] + t * (lo[j + 1] - hi[j]);
    (line, line)
}

impl CrpsScratch {
    /// One row's exact CRPS against the observation `y`.
    fn row(&mut self, row: &[f64], levels: &[f64], y: f64) -> f64 {
        self.pairs.clear();
        self.pairs
            .extend(row.iter().copied().zip(levels.iter().copied()));
        // Stable order does not matter: a repeated value merges its levels
        // into one jump, which is a min/max of the group, not an order.
        self.pairs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

        self.values.clear();
        self.lo.clear();
        self.hi.clear();
        for &(value, level) in &self.pairs {
            if let Some(&last) = self.values.last() {
                if value == last {
                    let i = self.values.len() - 1;
                    if level < self.lo[i] {
                        self.lo[i] = level;
                    }
                    if level > self.hi[i] {
                        self.hi[i] = level;
                    }
                    continue;
                }
            }
            self.values.push(value);
            self.lo.push(level);
            self.hi.push(level);
        }

        // The break points: every distinct forecast value, plus the
        // observation when it is not one of them. The indicator in the
        // integrand is constant on each interval between them.
        self.grid.clear();
        self.grid.extend_from_slice(&self.values);
        if !self.grid.iter().any(|&g| g == y) {
            let at = self.grid.partition_point(|&g| g < y);
            self.grid.insert(at, y);
        }

        self.segments.clear();
        for i in 0..self.grid.len() - 1 {
            let right = limits(&self.values, &self.lo, &self.hi, self.grid[i]).1;
            let left_next = limits(&self.values, &self.lo, &self.hi, self.grid[i + 1]).0;
            let d = self.grid[i + 1] - self.grid[i];
            let ind = if self.grid[i] >= y { 1.0 } else { 0.0 };
            let c = right - ind;
            let slope = (left_next - right) / d;
            self.segments
                .push(c * c * d + c * slope * d * d + (slope * slope * d * d * d) / 3.0);
        }
        pairwise_sum(&self.segments)
    }
}

/// The mean exact CRPS over `n_rows` forecast rows of `n_levels` values.
///
/// `values` is row-major `(n_rows, n_levels)`; `levels` names the columns
/// (for ensemble samples, the sorted ranks' `i / (K + 1)`); `obs` carries
/// one observation per row. The result is `_crps_rows`'s number: the row
/// scores accumulated in row order, divided by the row count.
pub fn crps_rows(
    values: &[f64],
    n_rows: usize,
    n_levels: usize,
    levels: &[f64],
    obs: &[f64],
) -> Result<f64, CrpsError> {
    let Some(total_len) = n_rows.checked_mul(n_levels) else {
        return Err(CrpsError::LengthMismatch);
    };
    if n_levels == 0 || values.len() != total_len || levels.len() != n_levels || obs.len() != n_rows
    {
        return Err(CrpsError::LengthMismatch);
    }
    if n_rows == 0 {
        return Err(CrpsError::Empty);
    }
    if values
        .iter()
        .chain(levels.iter())
        .chain(obs.iter())
        .any(|v| !v.is_finite())
    {
        return Err(CrpsError::NonFinite);
    }

    let mut scratch = CrpsScratch::default();
    let mut total = 0.0;
    for (i, &y) in obs.iter().enumerate() {
        let start = i * n_levels;
        total += scratch.row(&values[start..start + n_levels], levels, y);
    }
    Ok(total / n_rows as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Values generated by normet's Python `_row_crps` (scipy not involved);
    // the last is `_crps_rows` over three rows sharing one level vector.
    fn row(values: &[f64], levels: &[f64], y: f64) -> f64 {
        let mut s = CrpsScratch::default();
        s.row(values, levels, y)
    }

    #[test]
    fn ties_merge_into_one_jump() {
        assert_eq!(
            row(&[2.0, 2.0, 5.0], &[0.25, 0.75, 0.5], 3.0),
            0.854_166_666_666_666_6
        );
    }

    #[test]
    fn crossing_levels_use_the_group_limits() {
        assert_eq!(
            row(&[0.0, 1.0, 2.0], &[0.9, 0.1, 0.5], 1.0),
            0.806_666_666_666_666_8
        );
    }

    #[test]
    fn observation_at_a_data_value() {
        assert_eq!(
            row(&[1.0, 2.0, 4.0], &[0.1, 0.5, 0.9], 2.0),
            0.310_000_000_000_000_05
        );
    }

    #[test]
    fn observation_above_the_forecast_range() {
        assert_eq!(
            row(&[-3.5, 0.25, 0.25, 7.0], &[0.05, 0.4, 0.6, 0.95], 100.0),
            97.351_25
        );
    }

    #[test]
    fn matrix_matches_the_python_row_loop() {
        let levels = [0.2, 0.5, 0.8];
        let values = [1.0, 2.0, 3.0, 2.0, 2.0, 5.0, 0.0, 2.0, 2.0];
        let obs = [2.0, 3.0, -1.0];
        assert_eq!(
            crps_rows(&values, 3, 3, &levels, &obs).expect("finite"),
            0.870_000_000_000_000_1
        );
    }

    #[test]
    fn pairwise_sum_matches_the_block_rule() {
        // Below eight values: sequential.
        let a = [1.0e16, 1.0, -1.0e16];
        assert_eq!(pairwise_sum(&a), 0.0);
        // Eight or more: the fixed eight-accumulator tree.
        let b: Vec<f64> = (0..200).map(|i| (i as f64) + 0.5).collect();
        let expected: f64 = b.iter().copied().sum();
        assert_eq!(pairwise_sum(&b), expected); // exact in binary for this set
    }

    #[test]
    fn pairwise_sum_matches_numpy_above_the_ufunc_buffer() {
        // numpy 1.26.4 reduces in 8192-element ufunc-buffer chunks; without
        // the chunk loop this array's sum differs in the last bit.
        let mut state: u64 = 7;
        let a: Vec<f64> = (0..20_000)
            .map(|_| {
                state = (16807 * state) % 2_147_483_647;
                state as f64 / 2_147_483_647.0
            })
            .collect();
        assert_eq!(pairwise_sum(&a), 9989.011415038265);
    }

    #[test]
    fn errors() {
        let ok = [0.1, 0.9];
        assert_eq!(
            crps_rows(&[1.0, 2.0], 1, 2, &ok, &[1.0, 1.0]),
            Err(CrpsError::LengthMismatch)
        );
        assert_eq!(crps_rows(&[], 0, 2, &ok, &[]), Err(CrpsError::Empty));
        assert_eq!(
            crps_rows(&[1.0, f64::NAN], 1, 2, &ok, &[1.0]),
            Err(CrpsError::NonFinite)
        );
    }
}
