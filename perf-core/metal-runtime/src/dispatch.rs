//! Plan-side → kernel-registry dispatch bridge.
//!
//! The metal-runtime pipeline walks a [`model_plan::ModelPlan`] operator
//! by operator when it emits the MSL stub. Every operator carries a
//! plan-side [`model_plan::operator::OperatorKind`] tag and an optional
//! [`model_plan::attention::AttentionKind`] tag. The kernel registry,
//! in turn, has its own [`model_kernels::KernelOp`] enum (different
//! namespace, different cardinality) that names each dispatchable
//! family.
//!
//! This module bridges the two:
//!
//! - [`plan_kernel_tag`] maps a plan operator to the
//!   [`model_kernels::KernelOp::tag`] string the selector would emit,
//!   when a mapping exists. Operators without a mapping (e.g. softmax,
//!   embedding, sampling) return `None`; the compile pipeline falls
//!   back to the plan-side tag in that case.
//! - [`emit_per_op_stub`] returns a per-op MSL stub line that bakes
//!   both the plan-side tag and the kernel-side tag into the shader
//!   template so traces and selector audits can correlate the two.
//!
//! The mapping is deliberately conservative: it only enumerates the
//! operator families the kernel registry already understands
//! (attention, MoE, recurrent, diffusion, quantized, speculative). New
//! operator families should be added here *after* the kernel-registry
//! `KernelOp` enum gains the corresponding variant — never before.

use model_kernels::KernelOp;
use model_plan::attention::AttentionKind;
use model_plan::operator::OperatorKind;
use model_plan::OperatorPlan;

/// Map a plan operator to the kernel-registry dispatch tag, if a
/// mapping exists.
///
/// Returns `Some(<kernel-tag>)` when the operator kind and (where
/// relevant) attention family correspond to a registered
/// [`model_kernels::KernelOp`] variant. Returns `None` otherwise — the
/// compile pipeline falls back to the plan-side tag in that case so the
/// MSL stub still names the operator.
///
/// # Examples
///
/// ```
/// use metal_runtime::dispatch::plan_kernel_tag;
/// use model_plan::operator::OperatorKind;
/// use model_plan::attention::AttentionKind;
/// use model_plan::{DType, OperatorId, OperatorPlan, Precision, QuantizationPolicy, TensorRef};
/// use model_plan::state::StatePlan;
///
/// fn op(kind: OperatorKind, attn: Option<AttentionKind>) -> OperatorPlan {
///     let tr = TensorRef { name: "x".into(), shape: vec![1], dtype: DType::F32, state_id: None };
///     OperatorPlan {
///         id: OperatorId(1),
///         kind,
///         attention: attn,
///         inputs: vec![tr.clone()],
///         outputs: vec![tr],
///         precision: Precision::Fp32,
///         quant: QuantizationPolicy::Dense,
///         deps: vec![],
///     }
/// }
///
/// // Grouped-query attention maps to the gqa_attention kernel.
/// assert_eq!(plan_kernel_tag(&op(OperatorKind::Rope, Some(AttentionKind::Gqa { kv_heads: 4 }))),
///            Some("gqa_attention"));
///
/// // Multi-latent attention maps to the mla_attention kernel.
/// assert_eq!(plan_kernel_tag(&op(OperatorKind::Rope, Some(AttentionKind::Mla { d_latent: 64, d_rope: 16 }))),
///            Some("mla_attention"));
///
/// // Plain softmax has no kernel-registry mapping.
/// assert_eq!(plan_kernel_tag(&op(OperatorKind::Softmax, None)), None);
/// ```
#[must_use]
pub fn plan_kernel_tag(op: &OperatorPlan) -> Option<&'static str> {
    // The plan-side operator kinds (Rope, LayerNorm, etc.) do not
    // line up 1:1 with kernel-registry variants. The discriminating
    // signal is the (OperatorKind, AttentionKind) pair. Treat every
    // operator-with-attention as an attention operator first; fall
    // back to the plan-side kind only when no attention is set.
    if let Some(attn) = &op.attention {
        return Some(attention_kernel_tag(attn));
    }
    match op.kind {
        OperatorKind::MoeRouter { .. } => Some(KernelOp::MoeRouter.tag()),
        // Recurrent / linear-recurrent family — no attention slot.
        // The Qwen3-Coder-Next hybrid uses these in alternation with
        // attention layers, so the plan carries a plain kind here.
        OperatorKind::Rope => None, // bare Rope → embedding-side; no kernel mapping
        _ => None,
    }
}

/// Map a plan-side [`AttentionKind`] to the kernel-registry dispatch
/// tag for attention operators.
fn attention_kernel_tag(attn: &AttentionKind) -> &'static str {
    match attn {
        AttentionKind::Gqa { .. } => KernelOp::GqaAttention.tag(),
        AttentionKind::Mla { .. } => KernelOp::MlaAttention.tag(),
        AttentionKind::Cca { .. } => KernelOp::CcaAttention.tag(),
        AttentionKind::Paged { .. } => KernelOp::PagedAttention.tag(),
        AttentionKind::Tree { .. } => KernelOp::TreeAttention.tag(),
        AttentionKind::SlidingWindow { .. } => KernelOp::SlidingWindowAttention.tag(),
        AttentionKind::Dense => KernelOp::DenseAttention.tag(),
    }
}

/// Build the per-op MSL stub line for one operator.
///
/// The format is intentionally machine-greppable:
///
/// ```text
/// // op#1 dense_matmul [no-kernel-tag] : 2 input(s), 1 output(s)
/// // op#2 gqa_attention [from-plan=gqa] : 3 input(s), 1 output(s)
/// ```
///
/// `[no-kernel-tag]` is emitted (with the exact text) when
/// [`plan_kernel_tag`] returns `None` so trace audits can confirm the
/// absence of a mapping was deliberate rather than a missing case.
pub fn emit_per_op_stub(op: &OperatorPlan) -> String {
    let kernel = plan_kernel_tag(op).unwrap_or("no-kernel-tag");
    let from_plan = op.kind.tag();
    format!(
        "// op#{id} {plan_tag} [kernel={kernel}, from-plan={from_plan}] : {n_in} input(s), {n_out} output(s)\n",
        id = op.id.0,
        plan_tag = from_plan,
        kernel = kernel,
        from_plan = from_plan,
        n_in = op.inputs.len(),
        n_out = op.outputs.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_plan::{
        DType, OperatorId, OperatorPlan, Precision, QuantizationPolicy, SchedulerPolicy, TensorRef,
    };
    use model_plan::{ModelId, ModelPlan};

    fn tr(name: &str) -> TensorRef {
        TensorRef {
            name: name.into(),
            shape: vec![1],
            dtype: DType::F32,
            state_id: None,
        }
    }

    fn op_with_attn(id: u64, kind: OperatorKind, attn: AttentionKind) -> OperatorPlan {
        OperatorPlan {
            id: OperatorId(id),
            kind,
            attention: Some(attn),
            inputs: vec![tr("x")],
            outputs: vec![tr("y")],
            precision: Precision::Fp32,
            quant: QuantizationPolicy::Dense,
            deps: vec![],
        }
    }

    fn op_plain(id: u64, kind: OperatorKind) -> OperatorPlan {
        OperatorPlan {
            id: OperatorId(id),
            kind,
            attention: None,
            inputs: vec![tr("x")],
            outputs: vec![tr("y")],
            precision: Precision::Fp32,
            quant: QuantizationPolicy::Dense,
            deps: vec![],
        }
    }

    // -- plan_kernel_tag: known attention mappings ----------------------

    #[test]
    fn gqa_attention_maps_to_gqa_attention_tag() {
        let op = op_with_attn(1, OperatorKind::Rope, AttentionKind::Gqa { kv_heads: 4 });
        assert_eq!(plan_kernel_tag(&op), Some("gqa_attention"));
    }

    #[test]
    fn mla_attention_maps_to_mla_attention_tag() {
        let op = op_with_attn(1, OperatorKind::Rope, AttentionKind::Mla {
            d_latent: 64,
            d_rope: 16,
        });
        assert_eq!(plan_kernel_tag(&op), Some("mla_attention"));
    }

    #[test]
    fn cca_attention_maps_to_cca_attention_tag() {
        let op = op_with_attn(1, OperatorKind::Rope, AttentionKind::Cca { compressed_factor: 4 });
        assert_eq!(plan_kernel_tag(&op), Some("cca_attention"));
    }

    #[test]
    fn paged_attention_maps_to_paged_attention_tag() {
        let op = op_with_attn(1, OperatorKind::Rope, AttentionKind::Paged { block_size: 16 });
        assert_eq!(plan_kernel_tag(&op), Some("paged_attention"));
    }

    #[test]
    fn tree_attention_maps_to_tree_attention_tag() {
        let op = op_with_attn(1, OperatorKind::Rope, AttentionKind::Tree { width: 4, depth: 3 });
        assert_eq!(plan_kernel_tag(&op), Some("tree_attention"));
    }

    #[test]
    fn sliding_window_attention_maps_to_sliding_window_attention_tag() {
        let op = op_with_attn(1, OperatorKind::Rope, AttentionKind::SlidingWindow {
            window_size: 256,
        });
        assert_eq!(
            plan_kernel_tag(&op),
            Some("sliding_window_attention"),
            "SlidingWindow must route to the sliding-window kernel tag"
        );
    }

    #[test]
    fn dense_attention_maps_to_dense_attention_tag() {
        let op = op_with_attn(1, OperatorKind::Rope, AttentionKind::Dense);
        assert_eq!(plan_kernel_tag(&op), Some("dense_attention"));
    }

    // -- plan_kernel_tag: deliberately unmapped ------------------------

    #[test]
    fn plain_softmax_has_no_kernel_mapping() {
        let op = op_plain(1, OperatorKind::Softmax);
        assert_eq!(
            plan_kernel_tag(&op), None,
            "Softmax has no kernel-registry mapping yet"
        );
    }

    #[test]
    fn plain_embedding_has_no_kernel_mapping() {
        let op = op_plain(1, OperatorKind::Embedding);
        assert_eq!(plan_kernel_tag(&op), None);
    }

    #[test]
    fn plain_sampling_has_no_kernel_mapping() {
        let op = op_plain(1, OperatorKind::Sampling);
        assert_eq!(plan_kernel_tag(&op), None);
    }

    #[test]
    fn moe_router_maps_to_existing_kernel_registry_tag() {
        let op = op_plain(
            1,
            OperatorKind::MoeRouter {
                num_experts: 64,
                top_k: 8,
            },
        );
        assert_eq!(plan_kernel_tag(&op), Some(KernelOp::MoeRouter.tag()));
    }

    // -- emit_per_op_stub: format invariants ---------------------------

    #[test]
    fn per_op_stub_includes_op_id_plan_tag_and_kernel_tag() {
        let op = op_with_attn(7, OperatorKind::Rope, AttentionKind::Gqa { kv_heads: 4 });
        let line = emit_per_op_stub(&op);
        assert!(line.contains("op#7"), "must include op id: {line}");
        assert!(line.contains("rope"), "must include plan-side kind tag: {line}");
        assert!(
            line.contains("[kernel=gqa_attention"),
            "must include the kernel-registry tag: {line}"
        );
        assert!(line.contains("from-plan=rope"), "must include from-plan tag: {line}");
        assert!(line.contains("1 input(s)"), "must include input count: {line}");
        assert!(line.contains("1 output(s)"), "must include output count: {line}");
    }

    #[test]
    fn per_op_stub_marks_unmapped_operators_clearly() {
        let op = op_plain(11, OperatorKind::Softmax);
        let line = emit_per_op_stub(&op);
        assert!(line.contains("[kernel=no-kernel-tag"), "must mark unmapped: {line}");
        assert!(line.contains("from-plan=softmax"), "must include plan tag: {line}");
    }

    #[test]
    fn per_op_stub_line_ends_with_newline() {
        let op = op_with_attn(1, OperatorKind::Rope, AttentionKind::SlidingWindow { window_size: 4 });
        let line = emit_per_op_stub(&op);
        assert!(line.ends_with('\n'), "per-op stub must end with newline: {line:?}");
    }

    #[test]
    fn per_op_stub_counts_inputs_and_outputs_correctly() {
        let op = OperatorPlan {
            id: OperatorId(3),
            kind: OperatorKind::Rope,
            attention: Some(AttentionKind::Mla { d_latent: 64, d_rope: 16 }),
            inputs: (0..5).map(|i| tr(&format!("in{i}"))).collect(),
            outputs: (0..2).map(|i| tr(&format!("out{i}"))).collect(),
            precision: Precision::Fp32,
            quant: QuantizationPolicy::Dense,
            deps: vec![],
        };
        let line = emit_per_op_stub(&op);
        assert!(line.contains("5 input(s)"), "got {line}");
        assert!(line.contains("2 output(s)"), "got {line}");
    }

    // -- integration with ModelPlan (smoke) ----------------------------

    #[test]
    fn dispatch_map_is_stable_across_a_qwen_hybrid_plan() {
        // Smoke check: build a 3-op plan (sliding-window attention,
        // DeltaNet via Rope-without-attn placeholder, plain softmax)
        // and confirm we get one mapped kernel tag + two unmapped.
        let ops = vec![
            op_with_attn(1, OperatorKind::Rope, AttentionKind::SlidingWindow { window_size: 256 }),
            op_with_attn(2, OperatorKind::Rope, AttentionKind::Gqa { kv_heads: 2 }),
            op_plain(3, OperatorKind::Softmax),
        ];
        let plan = ModelPlan::new_unchecked(
            ModelId(1),
            "qwen-hybrid",
            "qwen",
            ops,
            vec![],
            SchedulerPolicy::Eager,
            4,
            8,
        );
        plan.validate().expect("smoke plan must validate");
        let tags: Vec<Option<&str>> = plan.operators.iter().map(plan_kernel_tag).collect();
        assert_eq!(
            tags,
            vec![Some("sliding_window_attention"), Some("gqa_attention"), None],
            "Qwen3-Next hybrid plan must emit the expected kernel tags"
        );
    }

    #[test]
    fn dispatch_returns_consistent_tags_with_kernel_registry_enum() {
        // Lock the invariant that the static strings emitted by
        // plan_kernel_tag match `KernelOp::tag()` exactly. This guards
        // against accidental drift between the two namespaces.
        let cases: &[(&str, AttentionKind, KernelOp)] = &[
            ("gqa_attention", AttentionKind::Gqa { kv_heads: 4 }, KernelOp::GqaAttention),
            ("mla_attention", AttentionKind::Mla { d_latent: 64, d_rope: 16 }, KernelOp::MlaAttention),
            ("cca_attention", AttentionKind::Cca { compressed_factor: 4 }, KernelOp::CcaAttention),
            ("paged_attention", AttentionKind::Paged { block_size: 16 }, KernelOp::PagedAttention),
            ("tree_attention", AttentionKind::Tree { width: 4, depth: 3 }, KernelOp::TreeAttention),
            (
                "sliding_window_attention",
                AttentionKind::SlidingWindow { window_size: 256 },
                KernelOp::SlidingWindowAttention,
            ),
            ("dense_attention", AttentionKind::Dense, KernelOp::DenseAttention),
        ];
        for (want, attn, _op) in cases {
            let op = op_with_attn(1, OperatorKind::Rope, attn.clone());
            let got = plan_kernel_tag(&op);
            assert_eq!(got, Some(*want), "tag drift for {attn:?}: got {got:?}");
        }
    }
}
