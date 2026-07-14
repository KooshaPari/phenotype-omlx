// phenotype-omlx FFI — pyo3 bridge exposing the Rust perf-core to Python.
//
// Build:
//   cd /Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-omlx/python/ffi
//   maturin develop --release
//
// Ground truth (read from perf-core/ crate source):
//   turbo_quant::QuantizedTensor::encode_uniform(&[f32], bits, group_size) -> Self
//   turbo_quant::QuantizedTensor::decode_uniform(&self, &mut [f32])
//   turbo_quant::TurboMode::label(&self) -> &'static str
//   spec_decode::{DraftMode, SpecDecodeConfig, build_engine(cfg, target, draft)}
//   spec_decode::backend::{TargetBackend (async_trait), DraftBackend,
//                          NullDraftBackend, TargetOutput, BackendInfo}
//   tree_attention::{TreePlan::new(w,d), .total_nodes(), tree_causal_mask(...)}
//   concurrent_exec::{ExecBackend, ExecRequest, ExecResult, JobError}
//   concurrent_exec::plan::AgentId(impl From<&str>)
//   concurrent_exec::{latentmas::LatentMasBackend, jetspec::JetSpecBackend,
//                     ssd::SsdBackend, tidar::{TidarAgent, TidarRole}}

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use tokio::runtime::Runtime;

use concurrent_exec::{
    ExecBackend, ExecRequest as RustExecRequest, ExecResult as RustExecResult,
    jetspec::JetSpecBackend, latentmas::LatentMasBackend, plan::AgentId,
    ssd::SsdBackend, tidar::TidarAgent,
};
use spec_decode::{
    backend::{BackendInfo, NullDraftBackend, TargetBackend, TargetOutput},
    DraftMode as RustDraftMode, SpecDecodeConfig as RustSpecDecodeConfig,
    SpecDecodeEngine, build_engine,
};
use tree_attention::{tree_causal_mask, TreePlan};
use turbo_quant::{QuantizedTensor, TurboMode};

// ── TurboQuant ──────────────────────────────────────────────────────────────

#[pyfunction]
fn turbo_quant_label_for_bits(bits: u8) -> PyResult<String> {
    let m = match bits {
        4 => TurboMode::Asymmetric4,
        3 => TurboMode::Symmetric3,
        2 => TurboMode::Symmetric2,
        _ => TurboMode::Symmetric4,
    };
    Ok(m.label().to_string())
}

#[pyfunction]
fn turbo_quant_encode(
    py: Python<'_>,
    data: Vec<f32>,
    group_size: usize,
    bits: u8,
) -> PyResult<PyObject> {
    let q = QuantizedTensor::encode_uniform(&data, bits, group_size);
    let dict = PyDict::new_bound(py);
    dict.set_item("shape", q.shape)?;
    dict.set_item("packed", q.packed)?;
    dict.set_item("scales", q.scales)?;
    dict.set_item("zeros", q.zeros)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (packed, scales, zeros, n))]
fn turbo_quant_decode(
    py: Python<'_>,
    packed: Vec<u8>,
    scales: Vec<f32>,
    zeros: Vec<f32>,
    n: usize,
) -> PyResult<PyObject> {
    let mut buf = vec![0f32; n];
    let q = QuantizedTensor { shape: vec![n], packed, scales, zeros };
    q.decode_uniform(&mut buf);
    let lst = pyo3::types::PyList::new_bound(py, buf.iter().copied());
    Ok(lst.into())
}

// ── Spec-decode ─────────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone)]
struct PyDraftMode {
    inner: RustDraftMode,
}

#[pymethods]
impl PyDraftMode {
    #[staticmethod]
    fn same_model() -> Self { Self { inner: RustDraftMode::SameModel } }
    #[staticmethod]
    fn draft_model() -> Self { Self { inner: RustDraftMode::DraftModel } }
    #[staticmethod]
    fn medusa() -> Self { Self { inner: RustDraftMode::Medusa } }
    fn __repr__(&self) -> String { format!("{:?}", self.inner) }
}

#[pyclass]
struct PySpecDecodeConfig {
    inner: RustSpecDecodeConfig,
}

#[pymethods]
impl PySpecDecodeConfig {
    #[new]
    #[pyo3(signature = (
        mode=None, max_draft_tokens=None, tree_width=None, tree_depth=None,
        temperature=None, fallback_on_reject=None,
    ))]
    fn new(
        mode: Option<PyDraftMode>,
        max_draft_tokens: Option<usize>,
        tree_width: Option<usize>,
        tree_depth: Option<usize>,
        temperature: Option<f32>,
        fallback_on_reject: Option<bool>,
    ) -> Self {
        let mut cfg = RustSpecDecodeConfig::default();
        if let Some(m) = mode { cfg.mode = m.inner; }
        if let Some(v) = max_draft_tokens { cfg.max_draft_tokens = v; }
        if let Some(v) = tree_width { cfg.tree_width = v; }
        if let Some(v) = tree_depth { cfg.tree_depth = v; }
        if let Some(v) = temperature { cfg.temperature = v; }
        if let Some(v) = fallback_on_reject { cfg.fallback_on_reject = v; }
        Self { inner: cfg }
    }
}

/// Null target backend — no model loaded; used for FFI plumbing tests.
struct NullTargetBackend;

#[async_trait]
impl TargetBackend for NullTargetBackend {
    async fn forward(&self, _token_ids: &[u32]) -> Result<TargetOutput, String> {
        Ok(TargetOutput {
            logits: vec![0.0; 4],
            hidden: None,
            finished: false,
        })
    }
    fn info(&self) -> BackendInfo {
        BackendInfo {
            engine: "null".into(),
            model_id: "none".into(),
            device: "cpu".into(),
            dtype: "f32".into(),
            kv_cache_type: None,
        }
    }
}

#[pyclass]
struct PySpecDecodeEngine {
    inner: Arc<tokio::sync::Mutex<SpecDecodeEngine>>,
}

#[pymethods]
impl PySpecDecodeEngine {
    #[new]
    fn new(cfg: &PySpecDecodeConfig) -> Self {
        let engine = build_engine(
            cfg.inner.clone(),
            Box::new(NullTargetBackend),
            Some(Box::new(NullDraftBackend)),
        );
        Self { inner: engine }
    }
    fn config_summary(&self, py: Python<'_>) -> PyResult<String> {
        // Hold the GIL while calling .blocking_lock() — that's the tokio Mutex
        // API that lets us avoid deadlocking against the runtime.
        let g = self.inner.blocking_lock();
        let _ = py;
        Ok(format!(
            "SpecDecodeEngine {{ mode={:?}, max_draft_tokens={} }}",
            g.config.mode, g.config.max_draft_tokens
        ))
    }
}

// ── Tree-attention ──────────────────────────────────────────────────────────

#[pyclass]
struct PyTreePlan {
    inner: std::sync::Mutex<TreePlan>,
}

#[pymethods]
impl PyTreePlan {
    #[new]
    fn new(width: usize, depth: usize) -> Self {
        Self { inner: std::sync::Mutex::new(TreePlan::new(width, depth)) }
    }
    fn total_nodes(&self) -> PyResult<usize> {
        Ok(self.inner.lock().unwrap().total_nodes())
    }
}

#[pyfunction]
fn tree_attn_causal_mask(
    py: Python<'_>,
    seq_len: usize,
    tree_width: usize,
    tree_depth: usize,
    offset: usize,
) -> PyResult<PyObject> {
    let m = tree_causal_mask(seq_len, tree_width, tree_depth, offset);
    let outer = pyo3::types::PyList::empty_bound(py);
    for row in m {
        let inner = pyo3::types::PyList::new_bound(py, row.iter().copied());
        outer.append(inner)?;
    }
    Ok(outer.into())
}

// ── Concurrent-exec helpers ────────────────────────────────────────────────

fn py_to_exec_request(req: &Bound<'_, PyDict>) -> PyResult<RustExecRequest> {
    let stop: Vec<String> = match req.get_item("stop").ok().flatten() {
        Some(v) => v.extract()?,
        None => Vec::new(),
    };
    Ok(RustExecRequest {
        prompt: req.get_item("prompt")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("prompt"))?
            .extract()?,
        max_tokens: req.get_item("max_tokens")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("max_tokens"))?
            .extract()?,
        temperature: req.get_item("temperature")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("temperature"))?
            .extract()?,
        stop,
    })
}

fn exec_result_to_py(py: Python, r: RustExecResult) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("text", r.text)?;
    d.set_item("tokens", r.tokens)?;
    d.set_item("elapsed_ms", r.elapsed_ms)?;
    Ok(d.into())
}

fn runtime() -> PyResult<Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

fn device_arc(d: Option<String>) -> Arc<str> {
    Arc::from(d.unwrap_or_else(|| "cpu".into()).into_boxed_str())
}

// LatentMAS
#[pyfunction]
#[pyo3(signature = (n_agents, req, device=None))]
fn run_latentmas(
    py: Python<'_>,
    n_agents: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<PyObject> {
    let backend = LatentMasBackend::new(n_agents, device_arc(device));
    let exec_req = py_to_exec_request(req)?;
    let rt = runtime()?;
    let res = rt.block_on(backend.run(AgentId::new("latentmas"), exec_req))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    exec_result_to_py(py, res)
}

// TiDAR
#[pyfunction]
#[pyo3(signature = (draft_len, diff_steps, req, device=None))]
fn run_tidar(
    py: Python<'_>,
    draft_len: usize,
    diff_steps: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<PyObject> {
    let backend = TidarAgent::drafter(draft_len, diff_steps, device_arc(device));
    let exec_req = py_to_exec_request(req)?;
    let rt = runtime()?;
    let res = rt.block_on(backend.run(AgentId::new("tidar"), exec_req))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    exec_result_to_py(py, res)
}

// JetSpec
#[pyfunction]
#[pyo3(signature = (tree_width, tree_depth, req, device=None))]
fn run_jetspec(
    py: Python<'_>,
    tree_width: usize,
    tree_depth: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<PyObject> {
    let backend = JetSpecBackend::new(tree_width, tree_depth, device_arc(device));
    let exec_req = py_to_exec_request(req)?;
    let rt = runtime()?;
    let res = rt.block_on(backend.run(AgentId::new("jetspec"), exec_req))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    exec_result_to_py(py, res)
}

// SSD
#[pyfunction]
#[pyo3(signature = (gamma, req, device=None))]
fn run_ssd(
    py: Python<'_>,
    gamma: usize,
    req: &Bound<'_, PyDict>,
    device: Option<String>,
) -> PyResult<PyObject> {
    let backend = SsdBackend::new(gamma, device_arc(device));
    let exec_req = py_to_exec_request(req)?;
    let rt = runtime()?;
    let res = rt.block_on(backend.run(AgentId::new("ssd"), exec_req))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    exec_result_to_py(py, res)
}

// ── Module registration ────────────────────────────────────────────────────

#[pymodule]
fn _perf(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(turbo_quant_label_for_bits, m)?)?;
    m.add_function(wrap_pyfunction!(turbo_quant_encode, m)?)?;
    m.add_function(wrap_pyfunction!(turbo_quant_decode, m)?)?;
    m.add_function(wrap_pyfunction!(tree_attn_causal_mask, m)?)?;
    m.add_function(wrap_pyfunction!(run_latentmas, m)?)?;
    m.add_function(wrap_pyfunction!(run_tidar, m)?)?;
    m.add_function(wrap_pyfunction!(run_jetspec, m)?)?;
    m.add_function(wrap_pyfunction!(run_ssd, m)?)?;
    m.add_class::<PyDraftMode>()?;
    m.add_class::<PySpecDecodeConfig>()?;
    m.add_class::<PySpecDecodeEngine>()?;
    m.add_class::<PyTreePlan>()?;
    Ok(())
}
