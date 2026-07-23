//! Multi-engine candidate metadata parity tests (SGLang / vLLM / TRT-LLM /
//! llama.cpp).
//!
//! These tests pin the contract described in `AGENTS.md` §2 ("Multi-engine
//! is the default") against the kernel-registry's selector and trace
//! machinery. An *external-engine* candidate is one that the runtime has
//! mapped onto a [`BackendKind`] substrate (Metal, Cuda, Cpu, ...) but
//! tagged with a metadata string naming the inference engine it serves
//! (e.g. `SGLang`, `vLLM`, `TRT-LLM`, `llama.cpp`). The engine name is a
//! `Candidate::engine_name: Option<String>` — **not** a `BackendKind`
//! variant — and is surfaced in three places:
//!
//! 1. The trace's `human_explanation` so the audit trail records which
//!    engine was selected.
//! 2. The candidate's `source_hash` (folded via `[engine:<name>]` so two
//!    candidates with identical source bytes but different engines are
//!    distinguishable on disk).
//! 3. `Candidate::engine_name` itself, so registry consumers can filter
//!    or report by engine.
//!
//! Selector determinism is preserved: the multi-engine candidate still
//! competes on `(metric, candidate_id)` exactly like any other tuned
//! candidate, and the lowest p95 wins under
//! `SelectionPolicy::Deterministic { prefer_lower_p95: true }`.
//!
//! Conventions match the rest of `sota_operators/`:
//! - `NOW_UNIX_MS` and `TEST_FINGERPRINT` come from the shared `main.rs`.
//! - `make_candidate` is the in-tree helper; `make_engine_candidate` is
//!   the multi-engine helper that mirrors it but tags the candidate with
//!   an external engine name.

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{
    BackendKind, Candidate, CandidateId, Capability, ExecutionTrace, KernelKey, KernelRegistry,
    SelectionPolicy,
};

use super::{
    build_record, fresh_capabilities, samples_with_p95, shape, NOW_UNIX_MS, TEST_FINGERPRINT,
};

/// Canonical set of external inference engines the runtime must be able to
/// route against. Kept here (not in `src/`) because they are a project /
/// deployment surface, not a kernel-registry implementation detail.
const SGLANG: &str = "SGLang";
const VLLM: &str = "vLLM";
const TRT_LLM: &str = "TRT-LLM";
const LLAMA_CPP: &str = "llama.cpp";

/// One row in the multi-engine matrix: an in-tree substrate (Metal,
/// Cuda, Cpu, ...) and the external engine it represents.
struct MultiEngineCandidate {
    candidate: Candidate,
}

/// Build a candidate tagged with an external engine name. The
/// `source_hash` is folded deterministically (`<hash>[engine:<name>]`) so
/// audit logs and on-disk artifacts can tell two engine-targeted
/// candidates apart even when the kernel source bytes are identical.
fn make_engine_candidate(
    name: &str,
    backend: BackendKind,
    engine: Option<&str>,
    requires: Vec<Capability>,
    min_shape: kernel_registry::ShapeSignature,
    max_shape: kernel_registry::ShapeSignature,
    supports_dtypes: Vec<DType>,
    tunable: bool,
) -> MultiEngineCandidate {
    MultiEngineCandidate {
        candidate: Candidate::with_engine(
            name,
            backend,
            format!("sha256:{name}"),
            engine,
            requires,
            min_shape,
            max_shape,
            supports_dtypes,
            tunable,
        ),
    }
}

/// Single shared key: `OperatorKind::DenseMatmul` at a small 16x16x16
/// shape on Fp16 + MetalGpu. Every test in this file uses the same key
/// so the considered-list assertions remain comparable.
fn multi_engine_key() -> KernelKey {
    KernelKey {
        operator_kind: OperatorKind::DenseMatmul,
        attention_kind: None,
        shape_signature: shape(16, 16, 16, 1, 1, 1),
        dtype: DType::Fp16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

/// Build a registry holding one Metal-only candidate plus one
/// multi-engine candidate for the same key. Each engine has a
/// deterministic p95 attached so we can assert the deterministic policy
/// picks the right winner.
fn registry_with_pair(
    engine: &'static str,
    metal_p95: u64,
    engine_p95: u64,
) -> (KernelRegistry, CandidateId, CandidateId, KernelKey) {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 1, 1, 1);
    let metal = make_engine_candidate(
        "DenseMatmulMetal",
        BackendKind::Metal,
        None,
        vec![Capability::MetalGpu],
        min,
        max,
        vec![DType::Fp16],
        true,
    );
    let external = make_engine_candidate(
        "DenseMatmulExternal",
        BackendKind::Metal,
        Some(engine),
        vec![Capability::MetalGpu],
        min,
        max,
        vec![DType::Fp16],
        true,
    );
    let id_metal = metal.candidate.id;
    let id_external = external.candidate.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(metal.candidate);
    reg.register_candidate(external.candidate);
    let key = multi_engine_key();
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_metal,
            key.clone(),
            &samples_with_p95(metal_p95),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    reg.attach_tuning_record(
        key.clone(),
        build_record(
            id_external,
            key.clone(),
            &samples_with_p95(engine_p95),
            Some(NOW_UNIX_MS + 86_400_000),
        ),
    );
    (reg, id_metal, id_external, key)
}

#[test]
fn sglang_candidate_and_mlx_metal_both_appear_in_considered_list() {
    let (reg, id_metal, id_sglang, key) = registry_with_pair(SGLANG, 2000, 1500);
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_sglang,
                "SGLang candidate p95=1500 must beat MLX/Metal p95=2000"
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
    // Both candidates must have been considered: assert via the rejected
    // list (which is empty for a successful choice) and via the registry
    // `list_candidates_for_key` snapshot.
    let visible = reg.list_candidates_for_key(&key);
    let visible_ids: Vec<CandidateId> = visible.iter().map(|c| c.id).collect();
    assert!(
        visible_ids.contains(&id_metal),
        "MLX/Metal candidate must appear in considered list; got {visible_ids:?}"
    );
    assert!(
        visible_ids.contains(&id_sglang),
        "SGLang candidate must appear in considered list; got {visible_ids:?}"
    );
}

#[test]
fn sglang_trace_human_explanation_includes_engine_name() {
    let (reg, _id_metal, id_sglang, key) = registry_with_pair(SGLANG, 2000, 1500);
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match &decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_sglang,
                "SGLang (lower p95) must be the chosen candidate"
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
    let trace: ExecutionTrace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_sglang));
    assert!(
        trace.human_explanation.contains(SGLANG),
        "trace must surface the external engine name in human_explanation; got {:?}",
        trace.human_explanation
    );
}

#[test]
fn sglang_source_hash_folds_engine_name_deterministically() {
    let base = "sha256:DenseMatmulExternal".to_string();
    let with_engine = Candidate::with_engine(
        "DenseMatmulExternal",
        BackendKind::Metal,
        base.clone(),
        Some(SGLANG),
        vec![Capability::MetalGpu],
        shape(1, 1, 1, 1, 1, 1),
        shape(64, 64, 64, 1, 1, 1),
        vec![DType::Fp16],
        true,
    );
    assert_eq!(
        with_engine.source_hash,
        format!("{base}[engine:{SGLANG}]"),
        "source_hash must include the engine tag deterministically"
    );
    assert_eq!(with_engine.engine_name.as_deref(), Some(SGLANG));

    // No engine -> hash is the bare base, no suffix.
    let without_engine = Candidate::with_engine(
        "DenseMatmulExternal",
        BackendKind::Metal,
        base.clone(),
        None::<&str>,
        vec![Capability::MetalGpu],
        shape(1, 1, 1, 1, 1, 1),
        shape(64, 64, 64, 1, 1, 1),
        vec![DType::Fp16],
        true,
    );
    assert_eq!(
        without_engine.source_hash, base,
        "source_hash must NOT be modified when engine_name is None"
    );
    assert!(without_engine.engine_name.is_none());
}

#[test]
fn vllm_deterministic_policy_picks_lower_p95() {
    let (reg, id_metal, id_vllm, key) = registry_with_pair(VLLM, 1800, 1200);
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match &decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_vllm,
                "vLLM candidate p95=1200 must beat MLX/Metal p95=1800"
            );
            assert_ne!(candidate.id, id_metal);
            assert_eq!(candidate.engine_name.as_deref(), Some(VLLM));
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
    let trace = reg.explain(&decision);
    assert!(
        trace.human_explanation.contains(VLLM),
        "trace must surface vLLM in human_explanation; got {:?}",
        trace.human_explanation
    );
}

#[test]
fn trtllm_deterministic_policy_picks_lower_p95_and_records_engine() {
    let (reg, id_metal, id_trt, key) = registry_with_pair(TRT_LLM, 1700, 1100);
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match &decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_trt,
                "TRT-LLM candidate p95=1100 must beat MLX/Metal p95=1700"
            );
            assert_ne!(candidate.id, id_metal);
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
    let trace = reg.explain(&decision);
    assert!(
        trace.human_explanation.contains(TRT_LLM),
        "trace must surface TRT-LLM in human_explanation; got {:?}",
        trace.human_explanation
    );
}

#[test]
fn llama_cpp_deterministic_policy_picks_lower_p95_and_records_engine() {
    let (reg, id_metal, id_lcpp, key) = registry_with_pair(LLAMA_CPP, 1600, 1000);
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match &decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_lcpp,
                "llama.cpp candidate p95=1000 must beat MLX/Metal p95=1600"
            );
            assert_ne!(candidate.id, id_metal);
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
    let trace = reg.explain(&decision);
    assert!(
        trace.human_explanation.contains(LLAMA_CPP),
        "trace must surface llama.cpp in human_explanation; got {:?}",
        trace.human_explanation
    );
}

#[test]
fn metal_candidate_still_wins_when_its_p95_is_lower_than_external_engine() {
    // Symmetry check: when the in-tree MLX/Metal path is the fastest,
    // it must still win even though an external-engine candidate exists.
    let (reg, id_metal, id_external, key) = registry_with_pair(SGLANG, 800, 1500);
    let decision = reg.select_with_caps(
        &key,
        SelectionPolicy::Deterministic {
            prefer_lower_p95: true,
        },
        &fresh_capabilities(),
        NOW_UNIX_MS,
    );
    match &decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(
                candidate.id, id_metal,
                "MLX/Metal p95=800 must beat SGLang p95=1500"
            );
            assert_ne!(candidate.id, id_external);
            assert!(
                candidate.engine_name.is_none(),
                "the winning in-tree candidate must NOT carry an engine tag; got {:?}",
                candidate.engine_name
            );
        }
        other => panic!("expected Chosen, got {other:?}"),
    }
    let trace = reg.explain(&decision);
    assert!(
        !trace.human_explanation.contains(SGLANG),
        "trace must NOT surface SGLang when MLX/Metal won; got {:?}",
        trace.human_explanation
    );
}
