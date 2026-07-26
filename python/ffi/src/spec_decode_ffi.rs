//! Speculative-decoding configuration bindings.
//!
//! MLX model execution deliberately stays in the Python-owned backend layer.
//! Passing an MLX object into a Tokio blocking worker crosses the Metal/MLX
//! ownership boundary and is not a supported FFI contract.

use pyo3::prelude::*;
use spec_decode::{DraftMode as RustDraftMode, SpecDecodeConfig as RustSpecDecodeConfig};

#[pyclass(from_py_object)]
#[derive(Clone)]
pub(crate) struct PyDraftMode {
    inner: RustDraftMode,
}

#[pymethods]
impl PyDraftMode {
    #[staticmethod]
    fn same_model() -> Self {
        Self {
            inner: RustDraftMode::SameModel,
        }
    }

    #[staticmethod]
    fn draft_model() -> Self {
        Self {
            inner: RustDraftMode::DraftModel,
        }
    }

    #[staticmethod]
    fn medusa() -> Self {
        Self {
            inner: RustDraftMode::Medusa,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass]
pub(crate) struct PySpecDecodeConfig {
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
        if let Some(mode) = mode {
            cfg.mode = mode.inner;
        }
        if let Some(value) = max_draft_tokens {
            cfg.max_draft_tokens = value;
        }
        if let Some(value) = tree_width {
            cfg.tree_width = value;
        }
        if let Some(value) = tree_depth {
            cfg.tree_depth = value;
        }
        if let Some(value) = temperature {
            cfg.temperature = value;
        }
        if let Some(value) = fallback_on_reject {
            cfg.fallback_on_reject = value;
        }
        Self { inner: cfg }
    }

    fn __repr__(&self) -> String {
        format!(
            "SpecDecodeConfig {{ mode={:?}, max_draft_tokens={}, tree_width={}, tree_depth={}, temperature={}, fallback_on_reject={} }}",
            self.inner.mode,
            self.inner.max_draft_tokens,
            self.inner.tree_width,
            self.inner.tree_depth,
            self.inner.temperature,
            self.inner.fallback_on_reject,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::PySpecDecodeConfig;

    #[test]
    fn config_repr_exposes_the_effective_configuration() {
        let config = PySpecDecodeConfig::new(None, Some(7), Some(3), Some(2), Some(0.5), None);
        let repr = config.__repr__();
        assert!(repr.contains("max_draft_tokens=7"));
        assert!(repr.contains("tree_width=3"));
        assert!(repr.contains("tree_depth=2"));
        assert!(repr.contains("temperature=0.5"));
    }
}
