//! PyO3 bindings for `normet-core`, imported by Python as `normet_rust`.
//!
//! The module is optional and opt-in: `import normet` never imports it, and
//! `normet.theil_sen(..., backend="rust")` asks for it explicitly. Arrays
//! must be contiguous float64; the Python wrapper does the conversion and
//! keeps the scipy quantile constant in the host, where the package's other
//! constants live.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use normet_core::theil_sen_interval as core_theil_sen;

/// Theil-Sen slope and Sen (1968) interval from `(x, y, z_alpha)`.
///
/// Returns `(slope, low, high, n_pairs, degenerate)`; `z_alpha` is the
/// lower standard-normal quantile (scipy `norm.ppf(0.025)`). `degenerate`
/// is true when every `x` is identical, in which case the three numbers
/// are NaN (the Python layer warns, as scipy does).
#[pyfunction]
#[pyo3(signature = (x, y, z_alpha))]
fn theil_sen_interval(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    z_alpha: f64,
) -> PyResult<(f64, f64, f64, u64, bool)> {
    let xs = x
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("x must be a contiguous float64 array: {e}")))?;
    let ys = y
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("y must be a contiguous float64 array: {e}")))?;
    let res = py
        .detach(|| core_theil_sen(xs, ys, z_alpha))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((res.slope, res.low, res.high, res.n_pairs, res.degenerate))
}

#[pymodule]
fn normet_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(theil_sen_interval, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
