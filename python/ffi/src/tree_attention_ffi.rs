//! Tree-attention pyo3 types and payload conversion.

use pyo3::prelude::*;
use pyo3::types::PyList;
use tree_attention::{tree_causal_mask, TreePlan};

#[pyclass]
pub struct PyTreePlan {
    inner: std::sync::Mutex<TreePlan>,
}

#[pymethods]
impl PyTreePlan {
    #[new]
    fn new(width: usize, depth: usize) -> Self {
        Self {
            inner: std::sync::Mutex::new(TreePlan::new(width, depth)),
        }
    }

    fn total_nodes(&self) -> PyResult<usize> {
        Ok(self.inner.lock().unwrap().total_nodes())
    }
}

#[pyfunction]
pub fn tree_attn_causal_mask(
    py: Python<'_>,
    seq_len: usize,
    tree_width: usize,
    tree_depth: usize,
    offset: usize,
) -> PyResult<Py<PyAny>> {
    let mask = tree_causal_mask(seq_len, tree_width, tree_depth, offset);
    let outer = PyList::empty(py);
    for row in mask {
        outer.append(PyList::new(py, row.iter().copied())?)?;
    }
    Ok(outer.into_any().unbind())
}
