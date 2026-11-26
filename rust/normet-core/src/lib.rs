//! Pure-Rust numerical kernels shared by the normet Python and R packages.
//!
//! The crate deliberately knows nothing about pandas, numpy, R or strings:
//! kernels take `&[f64]`/`&[i64]` plus shape arguments and write plain
//! numbers, so the Python binding (PyO3) and the R binding (extendr) share
//! one implementation. Random streams stay in the host language, where the
//! package's seeds and provenance live; kernels receive index matrices.
//!
//! Started with the highest-ROI call of the 2026-11-26 audit: Theil-Sen
//! order statistics (Python ~4-5x, R >100x, memory 2.76 GB -> O(pairs)).
//! The second is the exact CRPS of quantile/ensemble rows, where the Python
//! scorer's per-row loop dominates a normalise run's scoring pass.

pub mod crps;
pub mod trend;

pub use crps::{crps_rows, CrpsError, CrpsScratch};
pub use trend::{theil_sen_interval, TheilSenError, TheilSenInterval};
