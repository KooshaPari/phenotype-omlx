//! PyO3 bindings for the evaluation harness.
//!
//! Exposes the core evaluation types and scoring functions to Python.
//! Enabled via the `python` feature flag.

use pyo3::prelude::*;

mod functions;
mod reports;
mod types;

use reports::*;
use types::*;

#[pymodule]
fn _eval_harness(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySuite>()?;
    m.add_class::<PyTaskSpec>()?;
    m.add_class::<PyTaskResult>()?;
    m.add_class::<PyEvaluationReport>()?;
    m.add_class::<PyDatasetProvenance>()?;
    m.add_class::<PySuiteReportEntry>()?;
    m.add_class::<PyMultiSuiteReport>()?;
    m.add_class::<PyCriteria>()?;

    functions::register(m)?;

    Ok(())
}
