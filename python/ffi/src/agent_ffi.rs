//! pyo3 entry points for concurrent-exec agent backends.

use concurrent_exec::{
    jetspec::JetSpecBackend, latentmas::LatentMasBackend, plan::AgentId, ssd::SsdBackend,
    tidar::TidarAgent, ExecBackend, ExecRequest, ExecResult,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn py_to_exec_request(req: &Bound<'_, PyDict>) -> PyResult<ExecRequest> {
    let stop = req
        .get_item("stop")?
        .map_or_else(|| Ok(Vec::new()), |value| value.extract())?;
    Ok(ExecRequest {
        prompt: req
            .get_item("prompt")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("prompt"))?
            .extract()?,
        max_tokens: req
            .get_item("max_tokens")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("max_tokens"))?
            .extract()?,
        temperature: req
            .get_item("temperature")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("temperature"))?
            .extract()?,
        stop,
    })
}

fn exec_result_to_py(py: Python<'_>, result: ExecResult) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("text", result.text)?;
    dict.set_item("tokens", result.tokens)?;
    dict.set_item("elapsed_ms", result.elapsed_ms)?;
    Ok(dict.into_any().unbind())
}

fn runtime() -> PyResult<Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

fn device_arc(device: Option<String>) -> Arc<str> {
    Arc::from(device.unwrap_or_else(|| "cpu".into()).into_boxed_str())
}

fn run_backend(
    py: Python<'_>,
    backend: impl ExecBackend,
    agent_id: &'static str,
    req: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    let request = py_to_exec_request(req)?;
    let result = runtime()?
        .block_on(backend.run(AgentId::new(agent_id), request))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    exec_result_to_py(py, result)
}

#[pyfunction]
#[pyo3(signature = (n_agents, req, device=None))]
pub fn run_latentmas(
    py: Python<'_>,
    n_agents: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<Py<PyAny>> {
    run_backend(
        py,
        LatentMasBackend::new(n_agents, device_arc(device)),
        "latentmas",
        req,
    )
}

#[pyfunction]
#[pyo3(signature = (draft_len, diff_steps, req, device=None))]
pub fn run_tidar(
    py: Python<'_>,
    draft_len: usize,
    diff_steps: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<Py<PyAny>> {
    run_backend(
        py,
        TidarAgent::drafter(draft_len, diff_steps, device_arc(device)),
        "tidar",
        req,
    )
}

#[pyfunction]
#[pyo3(signature = (tree_width, tree_depth, req, device=None))]
pub fn run_jetspec(
    py: Python<'_>,
    tree_width: usize,
    tree_depth: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<Py<PyAny>> {
    run_backend(
        py,
        JetSpecBackend::new(tree_width, tree_depth, device_arc(device)),
        "jetspec",
        req,
    )
}

#[pyfunction]
#[pyo3(signature = (gamma, req, device=None))]
pub fn run_ssd(
    py: Python<'_>,
    gamma: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<Py<PyAny>> {
    run_backend(py, SsdBackend::new(gamma, device_arc(device)), "ssd", req)
}
