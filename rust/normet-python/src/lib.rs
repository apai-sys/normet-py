//! PyO3 bindings for `normet-core`, imported by Python as `normet_rust`.
//!
//! The module is optional and opt-in: `import normet` never imports it, and
//! `theil_sen(..., backend="rust")` / `crps_from_*(..., backend="rust")`
//! ask for it explicitly. Arrays must be contiguous float64; the Python
//! wrapper does the conversion and keeps the scipy quantile constant in the
//! host, where the package's other constants live.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use normet_core::angle_distance_matrix as core_angle_matrix;
use normet_core::crps_rows as core_crps_rows;
use normet_core::theil_sen_interval as core_theil_sen;
use normet_core::theil_sen_interval_with_config as core_theil_sen_cfg;
use normet_core::TheilSenConfig;

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

/// Theil-Sen interval with explicit selection knobs (hidden; test use).
///
/// The public binding is the three-argument function above; this entry lets
/// the parity tests force the window and radix-fallback branches on records
/// too short to reach them with the production configuration. Every
/// configuration must return the same bits.
#[pyfunction]
#[pyo3(signature = (x, y, z_alpha, full_below_pairs, pilot_pairs, window_cap))]
fn theil_sen_interval_cfg(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    z_alpha: f64,
    full_below_pairs: u64,
    pilot_pairs: usize,
    window_cap: usize,
) -> PyResult<(f64, f64, f64, u64, bool)> {
    let xs = x
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("x must be a contiguous float64 array: {e}")))?;
    let ys = y
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("y must be a contiguous float64 array: {e}")))?;
    let cfg = TheilSenConfig {
        full_below_pairs,
        pilot_pairs,
        window_cap,
    };
    let res = py
        .detach(|| core_theil_sen_cfg(xs, ys, z_alpha, cfg))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((res.slope, res.low, res.high, res.n_pairs, res.degenerate))
}

/// Mean exact CRPS over a `(rows, levels)` block of quantile or ensemble
/// forecasts, one observation per row.
///
/// The Python wrapper masks non-finite rows and converts to contiguous
/// float64 before calling; `levels` are the host's own floats, so the
/// interval weights match the pure-Python scorer bit for bit.
#[pyfunction]
#[pyo3(signature = (values, levels, obs))]
fn crps_rows(
    py: Python<'_>,
    values: PyReadonlyArray2<'_, f64>,
    levels: PyReadonlyArray1<'_, f64>,
    obs: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let shape = values.shape();
    let (n_rows, n_levels) = (shape[0], shape[1]);
    let vals = values
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("values must be contiguous: {e}")))?;
    let lev = levels
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("levels must be contiguous: {e}")))?;
    let ob = obs
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("obs must be contiguous: {e}")))?;
    py.detach(|| core_crps_rows(vals, n_rows, n_levels, lev, ob))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Sirois-Bottenheim mean directional-angle distances for trajectory pairs.
///
/// `vx`, `vy` and `norms` are `(n_traj, n_steps)` contiguous float64
/// blocks; the host computes the norms with `np.hypot`, so both backends
/// share the same bits. Returns the flat row-major `(n_traj, n_traj)`
/// symmetric matrix with a zero diagonal; the Python wrapper reshapes it.
#[pyfunction]
#[pyo3(signature = (vx, vy, norms))]
fn angle_distances<'py>(
    py: Python<'py>,
    vx: PyReadonlyArray2<'py, f64>,
    vy: PyReadonlyArray2<'py, f64>,
    norms: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let shape = vx.shape();
    let (n_traj, n_steps) = (shape[0], shape[1]);
    if vy.shape() != shape || norms.shape() != shape {
        return Err(PyValueError::new_err(
            "vx, vy and norms must have the same (n_traj, n_steps) shape",
        ));
    }
    let vx_s = vx
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("vx must be contiguous: {e}")))?;
    let vy_s = vy
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("vy must be contiguous: {e}")))?;
    let norms_s = norms
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("norms must be contiguous: {e}")))?;
    let flat = py
        .detach(|| {
            let mut out = vec![0.0_f64; n_traj * n_traj];
            core_angle_matrix(vx_s, vy_s, norms_s, n_traj, n_steps, &mut out).map(|()| out)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, flat))
}

#[pymodule]
fn normet_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(theil_sen_interval, m)?)?;
    m.add_function(wrap_pyfunction!(theil_sen_interval_cfg, m)?)?;
    m.add_function(wrap_pyfunction!(crps_rows, m)?)?;
    m.add_function(wrap_pyfunction!(angle_distances, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
