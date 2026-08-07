// phenotype-omlx FFI -- pyo3 bridge exposing Rust perf-cores to Python.
//
// The Python module registration is intentionally kept small.  Each public
// binding surface lives in a cohesive module so no FFI module grows beyond the
// repository's 500-line limit.

use pyo3::prelude::*;

mod agent_ffi;
mod spec_decode_ffi;
mod tree_attention_ffi;
mod turbo_quant_ffi;

use agent_ffi::{run_jetspec, run_latentmas, run_ssd, run_tidar};
use spec_decode_ffi::{PyDraftMode, PySpecDecodeConfig};
use tree_attention_ffi::{tree_attn_causal_mask, PyTreePlan};
use turbo_quant_ffi::{turbo_quant_decode, turbo_quant_encode, turbo_quant_label_for_bits};

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
    m.add_class::<PyTreePlan>()?;
    Ok(())
}
