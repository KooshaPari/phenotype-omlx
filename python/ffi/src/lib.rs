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
use pyo3::types::{PyDict, PyList, PyTuple};
use std::sync::Arc;
use tokio::runtime::Runtime;

use concurrent_exec::{
    ExecBackend, ExecRequest as RustExecRequest, ExecResult as RustExecResult,
    jetspec::JetSpecBackend, latentmas::LatentMasBackend, plan::AgentId,
    ssd::SsdBackend, tidar::TidarAgent,
};
use spec_decode::{
    backend::{BackendInfo, DraftBackend, NullDraftBackend, TargetBackend, TargetOutput},
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

/// Real MLX target backend — loads a HuggingFace model id via `mlx_lm.load`
/// and runs the target forward pass through `mlx_lm.generate.generate_step`.
/// Stores the model, tokenizer, and bound `generate_step` function as PyObjects
/// (cloned into spawn_blocking closures via Arc, then re-acquired under the
/// GIL via `Python::with_gil`).
struct MlxTargetBackend {
    model: Arc<PyObject>,
    tokenizer: Arc<PyObject>,
    generate_step: Arc<PyObject>,
    model_id: String,
    eos_token_ids: Arc<Vec<u32>>,
    kv_cache_kind: Option<String>,
}

impl MlxTargetBackend {
    /// Build a new MlxTargetBackend from a model id / local path.
    ///
    /// `model_id` is passed to `mlx_lm.load(...)`, which resolves both
    /// hub-style ids ("mlx-community/Qwen2.5-0.5B-Instruct-4bit") and
    /// local filesystem paths. We pre-resolve EOS token ids from the
    /// tokenizer so the target can mark a sequence as finished.
    fn build(model_id: &str, kv_cache_kind: Option<String>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let mlx_lm = py.import_bound("mlx_lm")?;
            let load = mlx_lm.getattr("load")?;
            let pair = load.call1((model_id,))?;
            let model: PyObject = pair.get_item(0)?.into();
            let tokenizer: PyObject = pair.get_item(1)?.into();

            let gen_mod = py.import_bound("mlx_lm.generate")?;
            let generate_step: PyObject = gen_mod.getattr("generate_step")?.into();

            // Pull EOS token ids from the tokenizer (some tokenizers expose
            // `.eos_token_id`; mlx_lm wraps with a TokenizerWrapper that has
            // `.eos_token_ids` as a set/list).
            let tok_bound = tokenizer.bind(py);
            let eos_ids: Vec<u32> = if let Ok(ids_any) = tok_bound.getattr("eos_token_ids") {
                if let Ok(v) = ids_any.extract::<Vec<u32>>() {
                    v
                } else if let Ok(s) = ids_any.extract::<std::collections::HashSet<u32>>() {
                    s.into_iter().collect()
                } else {
                    Vec::new()
                }
            } else if let Ok(id_any) = tok_bound.getattr("eos_token_id") {
                if let Ok(v) = id_any.extract::<u32>() {
                    vec![v]
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            Ok(MlxTargetBackend {
                model: Arc::new(model),
                tokenizer: Arc::new(tokenizer),
                generate_step: Arc::new(generate_step),
                model_id: model_id.to_string(),
                eos_token_ids: Arc::new(eos_ids),
                kv_cache_kind,
            })
        })
    }
}

#[async_trait]
impl TargetBackend for MlxTargetBackend {
    async fn forward(&self, token_ids: &[u32]) -> Result<TargetOutput, String> {
        // We need to drop the GIL for MLX work and yield back to the
        // tokio runtime so other tasks can run. `spawn_blocking` plus
        // `Python::with_gil` is the canonical pattern.
        let model = Arc::clone(&self.model);
        let generate_step = Arc::clone(&self.generate_step);
        let eos = Arc::clone(&self.eos_token_ids);
        let ids: Vec<u32> = token_ids.to_vec();

        tokio::task::spawn_blocking(move || -> Result<TargetOutput, String> {
            Python::with_gil(|py| -> Result<TargetOutput, String> {
                let mx = py.import_bound("mlx.core")
                    .map_err(|e| format!("import mlx.core: {e}"))?;
                let ids_py = PyList::new_bound(py, ids.iter().copied());
                // mx.array(ids) -> shape [seq]
                let prompt = mx.call_method1("array", (ids_py,))
                    .map_err(|e| format!("mx.array: {e}"))?;
                // prompt[None] -> shape [1, seq] for batched forward
                let none = py.None();
                let batched = prompt
                    .call_method1("__getitem__", (none,))
                    .map_err(|e| format!("prompt[None]: {e}"))?;

                // Call model(prompt[None]) — yields mlx.core.array of logits
                // with shape [1, seq, vocab]. Take the last token's logits
                // via [-1] (still shape [1, vocab]) and flatten to Vec<f32>.
                let model_ref = model.as_ref(py);
                let logits = model_ref
                    .call1((batched,))
                    .map_err(|e| format!("model forward: {e}"))?;
                let last = logits
                    .call_method1("__getitem__", (-1_i32,))
                    .map_err(|e| format!("logits[-1]: {e}"))?;
                let flat = last
                    .call_method0("reshape", )
                    .or_else(|_| last.call_method0("flatten"))
                    .map_err(|e| format!("flatten logits: {e}"))?;
                let logits_py = flat.call_method0("tolist")
                    .map_err(|e| format!("logits.tolist: {e}"))?;
                let logits_vec: Vec<f32> = logits_py
                    .extract()
                    .map_err(|e| format!("extract logits: {e}"))?;

                // Greedy next-token from the logits — this is what spec-decode
                // uses to score acceptance. For `SameModel` (prompt-lookup)
                // the engine doesn't actually use the logits, but for any
                // draft-tree verifier we do.
                let next_token = logits_vec
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);

                let finished = eos.contains(&next_token);

                // Best-effort cleanup of the generator Python object so it
                // doesn't leak across calls; harmless if it was already
                // dropped.
                let _ = generate_step.as_ref(py);

                Ok(TargetOutput {
                    logits: logits_vec,
                    hidden: None,
                    finished,
                })
            })
        })
        .await
        .map_err(|e| format!("spawn_blocking join: {e}"))?
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            engine: "mlx".into(),
            model_id: self.model_id.clone(),
            device: "metal".into(),
            dtype: "float16".into(),
            kv_cache_type: self.kv_cache_kind.clone(),
        }
    }
}

/// Python wrapper around MlxTargetBackend. Exposed as `_perf.MlxTargetBackend`
/// so callers can build a real target and hand it to PySpecDecodeEngine.
#[pyclass(name = "MlxTargetBackend")]
#[derive(Clone)]
struct PyMlxTargetBackend {
    inner: Arc<MlxTargetBackend>,
}

#[pymethods]
impl PyMlxTargetBackend {
    #[new]
    fn new(model_id: &str, kv_cache_kind: Option<String>) -> PyResult<Self> {
        let be = MlxTargetBackend::build(model_id, kv_cache_kind)?;
        Ok(Self { inner: Arc::new(be) })
    }

    fn info_json(&self) -> PyResult<String> {
        let i = self.inner.info();
        Ok(format!(
            "{{\"engine\":\"{}\",\"model_id\":\"{}\",\"device\":\"{}\",\"dtype\":\"{}\",\"kv_cache_type\":{}}}",
            i.engine,
            i.model_id,
            i.device,
            i.dtype,
            i.kv_cache_type
                .as_ref()
                .map(|s| format!("\"{s}\""))
                .unwrap_or_else(|| "null".to_string()),
        ))
    }
}

/// `NullTargetBackend` retained for tests / plumbing when no model is loaded.
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
    #[pyo3(signature = (cfg, target=None, draft=None))]
    fn new(
        cfg: &PySpecDecodeConfig,
        target: Option<&PyAny>,
        draft: Option<&PyAny>,
    ) -> PyResult<Self> {
        // Resolve target backend: a PyMlxTargetBackend instance wins;
        // otherwise fall back to NullTargetBackend. We can't downcast to
        // a foreign type across pyo3 boundaries, so we read the
        // `inner` attribute convention we set on PyMlxTargetBackend.
        let target_box: Box<dyn TargetBackend> = if let Some(t) = target {
            // Two ways to pass: a PyMlxTargetBackend instance (we already
            // built the backend); or a string model id (we build a real
            // MLX target for them). This makes the API forgiving.
            if let Ok(py_mlx) = t.extract::<PyMlxTargetBackend>() {
                Box::new(MlxTargetBackendClone::from(py_mlx))
            } else if let Ok(model_id) = t.extract::<String>() {
                let be = MlxTargetBackend::build(&model_id, None)?;
                Box::new(be)
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "target must be a _perf.MlxTargetBackend or a model id string",
                ));
            }
        } else {
            Box::new(NullTargetBackend)
        };

        // Draft backend: only NullDraftBackend for now (SameModel mode is
        // handled inside the engine via prompt-lookup). Future: a real
        // PyMlxDraftBackend mirroring the target pattern.
        let draft_box: Option<Box<dyn DraftBackend>> = if draft.is_some() {
            Some(Box::new(NullDraftBackend))
        } else {
            Some(Box::new(NullDraftBackend))
        };

        let engine = build_engine(cfg.inner.clone(), target_box, draft_box);
        Ok(Self { inner: engine })
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

/// A cheap clone wrapper that lets us extract an `Arc<MlxTargetBackend>`
/// from a `PyMlxTargetBackend` Python instance and re-wrap it as a
/// `dyn TargetBackend` for the engine. We can't move out of a pyclass, so
/// we clone the Arc and rebuild the trait object.
struct MlxTargetBackendClone(Arc<MlxTargetBackend>);

impl From<PyMlxTargetBackend> for MlxTargetBackendClone {
    fn from(p: PyMlxTargetBackend) -> Self {
        Self(p.inner)
    }
}

#[async_trait]
impl TargetBackend for MlxTargetBackendClone {
    async fn forward(&self, ids: &[u32]) -> Result<TargetOutput, String> {
        self.0.forward(ids).await
    }
    fn info(&self) -> BackendInfo {
        self.0.info()
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
    m.add_class::<PyMlxTargetBackend>()?;
    Ok(())
}
