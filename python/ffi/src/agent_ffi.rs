//! pyo3 entry points for concurrent-exec agent backends.
//!
//! `concurrent-exec` currently offers deterministic scheduling stubs rather
//! than model-backed execution.  Returning their placeholder text as a Python
//! completion would fabricate execution evidence, so each public runner fails
//! closed until a real backend is available through an explicit adapter.

use pyo3::prelude::*;
use pyo3::types::PyDict;

fn unavailable(runner: &str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "{runner} is unavailable: perf-core only provides a deterministic scheduling stub; \\
         configure a real backend before requesting execution"
    ))
}

#[pyfunction]
#[pyo3(signature = (n_agents, req, device=None))]
pub fn run_latentmas(
    n_agents: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<Py<PyAny>> {
    let _ = (n_agents, req, device);
    Err(unavailable("run_latentmas"))
}

#[pyfunction]
#[pyo3(signature = (draft_len, diff_steps, req, device=None))]
pub fn run_tidar(
    draft_len: usize,
    diff_steps: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<Py<PyAny>> {
    let _ = (draft_len, diff_steps, req, device);
    Err(unavailable("run_tidar"))
}

#[pyfunction]
#[pyo3(signature = (tree_width, tree_depth, req, device=None))]
pub fn run_jetspec(
    tree_width: usize,
    tree_depth: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<Py<PyAny>> {
    let _ = (tree_width, tree_depth, req, device);
    Err(unavailable("run_jetspec"))
}

#[pyfunction]
#[pyo3(signature = (gamma, req, device=None))]
pub fn run_ssd(
    gamma: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<Py<PyAny>> {
    let _ = (gamma, req, device);
    Err(unavailable("run_ssd"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exported_runners_reject_synthetic_execution() {
        Python::attach(|py| {
            let request = PyDict::new(py);
            request
                .set_item("prompt", "must not be fabricated")
                .expect("request accepts prompt");
            request
                .set_item("max_tokens", 4)
                .expect("request accepts max_tokens");
            request
                .set_item("temperature", 0.0_f32)
                .expect("request accepts temperature");

            let results = [
                run_latentmas(2, &request, None),
                run_tidar(2, 2, &request, None),
                run_jetspec(2, 2, &request, None),
                run_ssd(2, &request, None),
            ];
            for result in results {
                let error = result.expect_err("synthetic runner must fail closed");
                assert!(error.to_string().contains("unavailable"));
            }
        });
    }
}
