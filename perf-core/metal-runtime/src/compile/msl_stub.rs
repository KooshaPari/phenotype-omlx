//! Internal helpers used by [`super::compiler::BoundedCompiler`]:
//!
//! - [`emit_msl_stub`]: produces a deterministic MSL stub string from a
//!   [`ModelPlan`] + [`DeviceFingerprint`]. Each per-op line is emitted
//!   through [`crate::dispatch::emit_per_op_stub`] so the kernel-registry
//!   dispatch tag travels with the operator into the shader template.
//! - [`synthesize_compile_work`]: sha256 over the plan + fingerprint so
//!   different plans produce different outputs and the optimizer cannot
//!   fold it away.
//! - [`synthetic_compile_time_ms`]: deterministic wall-clock estimate
//!   that scales with operator count + a fingerprint-derived term.
//! - [`plan_revision`]: content-derived hash used as the cache-dimension
//!   source revision. `pub(crate)` so [`crate::pipeline`] can re-use it.

use model_plan::ModelPlan;
use sha2::{Digest, Sha256};

use crate::fingerprint::DeviceFingerprint;

use super::shader_catalog::source_for_tag;

/// Emit a deterministic MSL stub string for `plan`. The stub is currently
/// informational — it includes the plan id, family, and a one-line kernel
/// per operator. A future task will replace this with real codegen driven
/// by the operator kinds.
///
/// Each per-op line is emitted through [`crate::dispatch::emit_per_op_stub`]
/// so the kernel-registry dispatch tag (or `no-kernel-tag`) travels with
/// the operator into the shader template. Selector audits can then
/// grep the shader source to confirm the bridge produced the expected
/// routing for every operator.
pub(crate) fn emit_msl_stub(plan: &ModelPlan, fp: &DeviceFingerprint) -> String {
    use crate::dispatch::emit_per_op_stub;
    let mut out = String::with_capacity(256 + plan.operators.len() * 120);
    out.push_str("// metal-runtime MSL stub (real codegen in a future task)\n");
    out.push_str(&format!("// plan_id      = {}\n", plan.id.0));
    out.push_str(&format!("// family       = {}\n", plan.model_family));
    out.push_str(&format!("// gpu_family   = {}\n", fp.gpu_family.tag()));
    out.push_str(&format!("// op_count     = {}\n", plan.operators.len()));
    out.push_str(&format!("// max_seq_len  = {}\n", plan.max_seq_len));
    out.push_str("#include <metal_stdlib>\n");
    out.push_str("using namespace metal;\n\n");
    for op in &plan.operators {
        out.push_str(&emit_per_op_stub(op));
    }
    out
}

/// Emit the deterministic plan envelope plus checked-in kernel source for every
/// mapped operator. This remains reference-mode source assembly: the caller must
/// perform a real Metal compilation before treating it as executable evidence.
pub(crate) fn emit_msl_bundle(plan: &ModelPlan, fp: &DeviceFingerprint) -> String {
    use crate::dispatch::plan_kernel_tag;
    use std::collections::BTreeSet;

    let mut out = emit_msl_stub(plan, fp);
    let mut tags = BTreeSet::new();
    for op in &plan.operators {
        if let Some(tag) = plan_kernel_tag(op) {
            tags.insert(tag);
        }
    }
    out.push_str("// checked-in kernel sources below; native compilation required\n");
    for tag in tags {
        if let Some(source) = source_for_tag(tag) {
            out.push_str("\n// [kernel-source=");
            out.push_str(tag);
            out.push_str("]\n");
            out.push_str(source);
            out.push('\n');
        }
    }
    out
}

/// Simulate the compile-side work — sha256 over the plan + fingerprint so
/// different plans produce different outputs and the optimizer cannot fold
/// it away.
pub(crate) fn synthesize_compile_work(plan: &ModelPlan, fp: &DeviceFingerprint) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(plan.id.0.to_le_bytes());
    h.update(plan.name.as_bytes());
    h.update(plan.model_family.as_bytes());
    h.update(plan.max_seq_len.to_le_bytes());
    h.update(plan.vocab_size.to_le_bytes());
    h.update(fp.device_name.as_bytes());
    h.update(fp.os.as_bytes());
    h.update(fp.arch.as_bytes());
    h.update(fp.simd_bit_width.to_le_bytes());
    h.update(fp.total_memory_bytes.to_le_bytes());
    h.update((fp.gpu_family as u8).to_le_bytes());
    for op in &plan.operators {
        h.update(op.id.0.to_le_bytes());
        h.update(op.kind.tag().as_bytes());
    }
    h.finalize().into()
}

/// Estimate compile wall-clock time in milliseconds. Scales with operator
/// count so a tiny plan is fast and a large plan is slow. Combined with
/// `Instant` elapsed, this gives a deterministic-ish budget signal.
pub(crate) fn synthetic_compile_time_ms(plan: &ModelPlan, fp: &DeviceFingerprint) -> u64 {
    // 1 ms per operator + 5 ms base + an extra 1 ms per byte of
    // simd_bit_width * 8 (purely deterministic and stable).
    let base = 5u64;
    let per_op = plan.operators.len() as u64;
    let fp_overhead = (fp.simd_bit_width as u64) / 8;
    base + per_op + fp_overhead
}

/// Derive a `source_revision` u64 from the plan contents. The current
/// policy is a content-derived hash of the operator set so any change
/// (added op, new dep, swapped kind) bumps the revision and invalidates
/// cached entries. Plan-level metadata (id, name, family) is deliberately
/// excluded so renaming a plan does not invalidate the cache.
pub(crate) fn plan_revision(plan: &ModelPlan) -> u64 {
    let mut h = Sha256::new();
    h.update((plan.operators.len() as u64).to_le_bytes());
    h.update(plan.max_seq_len.to_le_bytes());
    h.update(plan.vocab_size.to_le_bytes());
    for op in &plan.operators {
        h.update(op.id.0.to_le_bytes());
        h.update(op.kind.tag().as_bytes());
        h.update((op.inputs.len() as u64).to_le_bytes());
        h.update((op.outputs.len() as u64).to_le_bytes());
        h.update((op.deps.len() as u64).to_le_bytes());
        for d in &op.deps {
            h.update(d.0.to_le_bytes());
        }
    }
    let bytes = h.finalize();
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(buf)
}
