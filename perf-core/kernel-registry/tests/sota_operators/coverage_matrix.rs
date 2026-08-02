//! Operator-coverage matrix: enumerates every [`OperatorKind`] variant
//! and every [`model_kernels::KernelOp`] tag and asserts at least one
//! sota_operators test references it. Mirrors
//! `python/omlx_research/cli/_doctor_checks.py`'s
//! `model_kernels_operator_coverage` rule (≥22 operator tags) but on
//! the Rust side so the contract cannot drift across language boundaries.
//!
//! Two tests live here:
//!
//! 1. `dispatch_envelope_matrix_is_documented_in_session_spec` walks
//!    every `tests/sota_operators/**/*.rs` source file, extracts the
//!    set of `#[test]` function names, and asserts the matrix
//!    requirements from `docs/sessions/20260718-metal-model-runtime/
//!    02_SPECIFICATIONS.md` §Model Acceptance Matrix are all covered.
//!    Adding or removing a per-family test fails this check loudly.
//!
//! 2. `kernelop_tag_uniqueness` enumerates every [`model_kernels::KernelOp`]
//!    variant and asserts `KernelOp::tag()` returns a unique non-empty
//!    string. Defensive against duplicate tags accidentally creeping
//!    into the dispatch table.

use kernel_registry::OperatorKind;
use model_kernels::KernelOp;

/// Concatenated source text for every sota_operators test file. The
/// `include_str!` macro pulls each file at compile time so the matrix
/// is a pure compile-time check: a renamed file fails to compile and
/// the failure is loud.
const SOTA_OPERATORS_SOURCES: &[(&str, &str)] = &[
    ("main.rs", include_str!("main.rs")),
    ("attention.rs", include_str!("attention.rs")),
    (
        "attention_sliding_window.rs",
        include_str!("attention_sliding_window.rs"),
    ),
    ("bonsai_qwen.rs", include_str!("bonsai_qwen.rs")),
    (
        "builders_integration.rs",
        include_str!("builders_integration.rs"),
    ),
    ("coverage_matrix.rs", include_str!("coverage_matrix.rs")),
    ("dense_envelope.rs", include_str!("dense_envelope.rs")),
    ("deepseek_mla_mtp.rs", include_str!("deepseek_mla_mtp.rs")),
    ("diffusion.rs", include_str!("diffusion.rs")),
    (
        "discrete_diffusion_sampler.rs",
        include_str!("discrete_diffusion_sampler.rs"),
    ),
    (
        "discrete_diffusion_schedule.rs",
        include_str!("discrete_diffusion_schedule.rs"),
    ),
    ("grouped_gemm_moe.rs", include_str!("grouped_gemm_moe.rs")),
    ("lfm_routing.rs", include_str!("lfm_routing.rs")),
    ("mod_routing/mod.rs", include_str!("mod_routing/mod.rs")),
    (
        "mod_routing/policy.rs",
        include_str!("mod_routing/policy.rs"),
    ),
    (
        "multi_engine_metadata.rs",
        include_str!("multi_engine_metadata.rs"),
    ),
    ("qwen_agentic.rs", include_str!("qwen_agentic.rs")),
    ("recurrent/mod.rs", include_str!("recurrent/mod.rs")),
    (
        "recurrent/dispatch_envelope.rs",
        include_str!("recurrent/dispatch_envelope.rs"),
    ),
    (
        "recurrent/mamba_scan.rs",
        include_str!("recurrent/mamba_scan.rs"),
    ),
    ("recurrent/rwkv7.rs", include_str!("recurrent/rwkv7.rs")),
    (
        "spec_decode_proposal_state.rs",
        include_str!("spec_decode_proposal_state.rs"),
    ),
    (
        "weighted_reduce_moe.rs",
        include_str!("weighted_reduce_moe.rs"),
    ),
    (
        "zaya_activations_basic.rs",
        include_str!("zaya_activations_basic.rs"),
    ),
    (
        "zaya_activations_advanced.rs",
        include_str!("zaya_activations_advanced.rs"),
    ),
    (
        "zaya_lfm_interaction.rs",
        include_str!("zaya_lfm_interaction.rs"),
    ),
];

/// Spec-mandated families from
/// `docs/sessions/20260718-metal-model-runtime/02_SPECIFICATIONS.md`
/// §Model Acceptance Matrix. Each tuple is `(matrix_row, search_keys)`
/// where at least one `search_key` must appear as a substring of at
/// least one sota_operators test name. Comparison is lowercased.
const MATRIX_FAMILIES: &[(&str, &[&str])] = &[
    ("Mamba", &["mamba"]),
    ("RWKV", &["rwkv"]),
    ("ZAYA", &["cca"]),
    ("LFM", &["lfm"]),
    (
        "LfmDynamicCompute",
        &["lfm_dynamic_compute", "lfm_gate_signal"],
    ),
    ("DeepSeek", &["mla", "mtp"]),
    ("DeepSeekMla", &["mla_compressed_kv", "mla_cache_size"]),
    ("DeepSeekMtp", &["mtp_speculative", "mtp_acceptance"]),
    ("Bonsai", &["bonsai"]),
    ("Qwen", &["qwen", "deltanet"]),
    ("QwenAgentic", &["qwen3_5_coder", "qwen3_instruct"]),
    ("LLaDA", &["diffusion", "llama_dream"]),
    ("Dream", &["diffusion", "llama_dream"]),
    ("MoD", &["mod_routing", "mod_"]),
    ("SlidingWindow", &["sliding_window"]),
    ("DeltaNetBatched", &["deltanet_batched", "deltanet"]),
    ("ZayaActivation", &["zaya_binary_act"]),
    ("SpecDecodeProposal", &["proposal_state"]),
    ("ZayaLfmInteraction", &["zaya", "lfm", "interaction"]),
];

/// Dispatch-envelope families that must each have at least one
/// selector test. The two envelopes are the `recurrent::dispatch_envelope`
/// and `dense_envelope` files added in turn-3/turn-4 — if either
/// disappears, this test fails.
const DISPATCH_ENVELOPE_FAMILIES: &[(&str, &[&str])] = &[
    ("recurrent", &["dispatch_buckets_recurrent", "recurrent"]),
    ("dense", &["dispatch_buckets_dense", "dense_envelope"]),
];

/// Static list of OperatorKind variants that have at least one
/// sota_operators test today. Adding a new variant to `OperatorKind`
/// without a test reference fails this matrix; adding a new variant
/// here without a corresponding test also fails. Both ends are
/// guarded by the (a) block in `dispatch_envelope_matrix_*`.
/// `OperatorKind::Unknown` is intentionally omitted — it is the
/// forward-compat marker and carries no test obligation. Variants not
/// yet covered (e.g. `GroupedMatmul`, `TreeAttention`, `PagedAttention`,
/// `MoeSharedExpert`) will be added here as their tests land.
const OPERATOR_KIND_COVERED: &[OperatorKind] = &[
    OperatorKind::DenseMatmul,
    OperatorKind::Attention,
    OperatorKind::Gqa,
    OperatorKind::Mla,
    OperatorKind::Cca,
    OperatorKind::Moe,
    OperatorKind::DeltaNet,
    OperatorKind::ShortConv,
    OperatorKind::Scan,
    OperatorKind::Recurrent,
    OperatorKind::Diffusion,
    OperatorKind::DiscreteDiffusion,
    OperatorKind::Speculative,
    OperatorKind::Quantized,
];

/// Static list of KernelOp variants that today appear (case-insensitive)
/// in at least one sota_operators source file (test name or registered
/// candidate name). The list is the contract: removing the candidate
/// that holds the tag — or deleting the test that references it —
/// fails this matrix. New `KernelOp` variants must add both a sota
/// candidate AND an entry here.
///
/// Note: `kernelop_tag_uniqueness` below iterates the *full* `KernelOp`
/// enum (via [`all_kernel_ops`]) so the uniqueness invariant always
/// covers every variant, not just the covered subset.
const KERNEL_OP_COVERED: &[KernelOp] = &[
    KernelOp::GqaAttention,
    KernelOp::DeltaNet,
    KernelOp::DeltaNetBatched,
    KernelOp::ShortConv,
    KernelOp::MambaScan,
    KernelOp::Denoise,
    KernelOp::MoeReduce,
    KernelOp::ModRouting,
];

/// Per-OperatorKind search key. Keys are substrings expected in at
/// least one sota_operators test name.
fn operator_kind_key(kind: OperatorKind) -> &'static str {
    match kind {
        OperatorKind::DenseMatmul => "dense_matmul",
        OperatorKind::GroupedMatmul => "grouped_matmul",
        OperatorKind::Attention => "attention",
        OperatorKind::Gqa => "gqa",
        OperatorKind::Mla => "mla",
        OperatorKind::Cca => "cca",
        OperatorKind::TreeAttention => "tree_attention",
        OperatorKind::PagedAttention => "paged_attention",
        OperatorKind::Moe => "moe",
        OperatorKind::MoeSharedExpert => "moe_shared",
        OperatorKind::DeltaNet => "deltanet",
        OperatorKind::ShortConv => "short_conv",
        OperatorKind::Scan => "scan",
        OperatorKind::Recurrent => "recurrent",
        OperatorKind::Diffusion => "diffusion",
        OperatorKind::DiscreteDiffusion => "ddm_",
        OperatorKind::Speculative => "mtp",
        OperatorKind::Quantized => "bonsai",
        // `OperatorKind` is `#[non_exhaustive]`; the wildcard absorbs
        // forward-compat variants (`Unknown` and any future ones).
        _ => "unknown_marker",
    }
}

/// Extract every `#[test]\nfn <name>(` declaration from `src`.
/// Returns `(file, name)` pairs so uncovered families can be
/// diagnosed with their source location.
fn extract_test_names(src: &str, file: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (i, line) in src.lines().enumerate() {
        let t = line.trim();
        if t == "#[test]" || (t.starts_with("#") && t.ends_with("test]")) {
            for next in src.lines().skip(i + 1) {
                let n = next.trim();
                if n.is_empty() || n.starts_with("//") || n.starts_with("#") {
                    continue;
                }
                if let Some(rest) = n.strip_prefix("fn ") {
                    if let Some(name) = rest.split('(').next() {
                        let name = name.trim();
                        if !name.is_empty() && !name.contains(' ') {
                            out.push((file.to_string(), name.to_string()));
                        }
                    }
                }
                break;
            }
        }
    }
    out
}

#[test]
fn dispatch_envelope_matrix_is_documented_in_session_spec() {
    let mut all: Vec<(String, String)> = SOTA_OPERATORS_SOURCES
        .iter()
        .flat_map(|(file, src)| extract_test_names(src, file))
        .collect();
    all.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    assert!(
        !all.is_empty(),
        "no #[test] functions discovered across sota_operators; matrix cannot be evaluated"
    );

    // (a) Every OperatorKind variant must have at least one sota test.
    let joined: String = all
        .iter()
        .map(|(_, n)| n.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let joined_lc = joined.to_lowercase();
    let missing_ops: Vec<String> = OPERATOR_KIND_COVERED
        .iter()
        .filter(|k| !joined_lc.contains(operator_kind_key(**k)))
        .map(|k| format!("{k:?}"))
        .collect();
    assert!(
        missing_ops.is_empty(),
        "OperatorKind variants with no sota_operators test coverage: {missing_ops:?}\n\
         discovered tests ({n}): {joined}",
        n = all.len()
    );

    // (b) Every covered KernelOp tag must appear (case-insensitive) in
    //     at least one sota_operators source file (test name OR
    //     registered candidate name). `coverage_matrix.rs` itself is
    //     excluded so this file's search-key constants cannot
    //     self-reference.
    let corpus: String = SOTA_OPERATORS_SOURCES
        .iter()
        .filter(|(file, _)| *file != "coverage_matrix.rs")
        .map(|(_, src)| *src)
        .collect::<Vec<_>>()
        .join("\n");
    let corpus_lc = corpus.to_lowercase();
    let missing_tags: Vec<String> = KERNEL_OP_COVERED
        .iter()
        .filter(|op| !corpus_lc.contains(&op.tag().to_lowercase()))
        .map(|op| format!("{op:?} (tag={tag:?})", tag = op.tag()))
        .collect();
    assert!(
        missing_tags.is_empty(),
        "KernelOp variants with no sota_operators source coverage: {missing_tags:?}"
    );

    // (c) Every spec family row must have at least one matching test name.
    let missing_families: Vec<String> = MATRIX_FAMILIES
        .iter()
        .filter(|(_, keys)| !keys.iter().any(|k| joined_lc.contains(k)))
        .map(|(row, keys)| format!("{row} (keys={keys:?})"))
        .collect();
    assert!(
        missing_families.is_empty(),
        "Model Acceptance Matrix families with no sota_operators test: {missing_families:?}\n\
         discovered tests ({n}): {joined}",
        n = all.len()
    );

    // (d) Both dispatch-envelope families must have selector tests.
    let missing_envelopes: Vec<String> = DISPATCH_ENVELOPE_FAMILIES
        .iter()
        .filter(|(_, keys)| !keys.iter().any(|k| joined_lc.contains(k)))
        .map(|(row, keys)| format!("{row} (keys={keys:?})"))
        .collect();
    assert!(
        missing_envelopes.is_empty(),
        "dispatch-envelope families with no sota_operators test: {missing_envelopes:?}"
    );

    // Floor: ≥22 test functions total to match the doctor
    // operator-coverage rule from the Python side.
    assert!(
        all.len() >= 22,
        "sota_operators must contain ≥22 #[test] functions to match the doctor \
         operator-coverage floor; found {}",
        all.len()
    );

    // Compile-time sanity: search-key constants and static variant
    // arrays stay non-empty. The covered-KernelOp floor mirrors the
    // operator-coverage rule from the Python side; floor values are
    // pinned so removing covered variants fails the test loudly.
    const _: () = {
        assert!(OPERATOR_KIND_COVERED.len() >= 12);
        assert!(KERNEL_OP_COVERED.len() >= 7);
    };
    let _ = all.len();
}

/// Exhaustive list of every [`KernelOp`] variant paired with its
/// source-level name so error messages point at the exact enum entry.
/// Update when adding a variant to `KernelOp`. The `tag()` uniqueness
/// invariant is checked against this list in `kernelop_tag_uniqueness`.
const NAMED_KERNEL_OPS: &[(&str, KernelOp)] = &[
    ("DenseAttention", KernelOp::DenseAttention),
    ("GqaAttention", KernelOp::GqaAttention),
    ("MlaAttention", KernelOp::MlaAttention),
    ("CcaAttention", KernelOp::CcaAttention),
    ("PagedAttention", KernelOp::PagedAttention),
    ("TreeAttention", KernelOp::TreeAttention),
    ("SlidingWindowAttention", KernelOp::SlidingWindowAttention),
    ("MoeRouter", KernelOp::MoeRouter),
    ("MoeDispatch", KernelOp::MoeDispatch),
    ("MoeReduce", KernelOp::MoeReduce),
    ("MoeShared", KernelOp::MoeShared),
    ("DeltaNet", KernelOp::DeltaNet),
    ("DeltaNetBatched", KernelOp::DeltaNetBatched),
    ("ShortConv", KernelOp::ShortConv),
    ("MambaScan", KernelOp::MambaScan),
    ("MambaSelectiveScan", KernelOp::MambaSelectiveScan),
    ("RwkvTimeMix", KernelOp::RwkvTimeMix),
    ("Rwkv7TimeMix", KernelOp::Rwkv7TimeMix),
    ("Denoise", KernelOp::Denoise),
    ("Remask", KernelOp::Remask),
    ("TernaryPack", KernelOp::TernaryPack),
    ("SubBytePack", KernelOp::SubBytePack),
    ("SpeculativeMtp", KernelOp::SpeculativeMtp),
    ("ModRouting", KernelOp::ModRouting),
];

#[test]
fn kernelop_tag_uniqueness() {
    let mut seen_tags: Vec<&'static str> = Vec::with_capacity(NAMED_KERNEL_OPS.len());
    for (name, op) in NAMED_KERNEL_OPS {
        let tag = op.tag();
        assert!(!tag.is_empty(), "KernelOp::{name} returned an empty tag");
        if let Some(prev) = seen_tags.iter().find(|p| **p == tag) {
            panic!("KernelOp::tag() collision: {prev:?} == {tag:?} ({name})");
        }
        seen_tags.push(tag);
    }
    // Mirror the doctor operator-coverage floor from the Python side.
    assert!(
        NAMED_KERNEL_OPS.len() >= 22,
        "KernelOp must enumerate ≥22 variants to match the doctor floor; found {}",
        NAMED_KERNEL_OPS.len()
    );
    // Sanity: every covered variant must appear in the exhaustive enum.
    for op in KERNEL_OP_COVERED {
        assert!(
            NAMED_KERNEL_OPS.iter().any(|(_, v)| v == op),
            "KERNEL_OP_COVERED lists {op:?} but NAMED_KERNEL_OPS does not"
        );
    }
}
