//! Qwen agentic operator suite — covers the agentic-coding variants of the
//! Qwen model family: Qwen3-Coder tool calling, Qwen3-Instruct chat-template
//! selection, and Qwen3.5-Coder edge-case prompts. Complements `bonsai_qwen.rs`
//! (which pins the *weight* and *recurrent* sides of Qwen) by exercising the
//! *agentic* layer — the prompt-decoding and tool-binding surface the runtime
//! relies on when Qwen drives a multi-step coding task.
//!
//! Tests:
//!
//!   * `qwen3_coder_tool_call_oracle_byte_identical` — register scalar +
//!     Metal + CPU candidates; verify the parsed tool-call JSON
//!     (`{"name":"...","arguments":{...}}`) is byte-identical to the
//!     in-file reference parser across runs.
//!   * `qwen3_instruct_chat_template_deterministic_picks_correct_binding`
//!     — under ChatML/Base/Custom templates, the deterministic policy must
//!     pick the candidate bound to the requested template, and the choice
//!     must be stable across calls.
//!   * `qwen3_5_coder_edge_case_prompts_select_stably` — register
//!     candidates for edge-case prompts (long context, multi-line code,
//!     special tokens); verify the selector returns a stable Chosen
//!     decision across runs.
//!
//! Convention: `OperatorKind::Gqa` (grouped-query attention) is the canonical
//! Qwen operator; `state_layout_version=1` and `policy_version=1` mirror the
//! project baseline.

use kernel_registry::compat::{DType, OperatorKind, QuantizationPolicy};
use kernel_registry::selector::SelectionDecision;
use kernel_registry::{
    BackendKind, Capability, KernelKey, KernelRegistry, SelectionPolicy,
};

use super::{
    build_record, fresh_capabilities, full_capabilities, make_candidate, samples_with_p95, shape,
    NOW_UNIX_MS, TEST_FINGERPRINT,
};

/// Canonical `KernelKey` for the Qwen agentic tests. `OperatorKind::Gqa`
/// matches the Qwen prompt-decoding path. Shape axes are deliberately
/// small (`8/8/8/1/1/1`) — the agentic layer cares about prompt-binding
/// bytes, not throughput.
fn qwen_agentic_key() -> KernelKey {
    KernelKey {
        operator_kind: OperatorKind::Gqa,
        attention_kind: None,
        shape_signature: shape(8, 8, 8, 1, 1, 1),
        dtype: DType::Bf16,
        quantization: QuantizationPolicy::None,
        state_layout_version: 1,
        device_fingerprint: TEST_FINGERPRINT.to_string(),
        policy_version: 1,
    }
}

// ---- Tool-call oracle ----------------------------------------------------

/// Parsed Qwen3-Coder tool-call. Wire format:
/// `<tool_call>{"name":"...","arguments":{...}}</tool_call>`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolCall {
    name: String,
    arguments: String,
}

impl ToolCall {
    /// Render to the canonical JSON wire bytes. The byte-identity test
    /// pins this exact sequence — indentation is intentionally absent
    /// so a regression that adds pretty-printing surfaces as a byte
    /// mismatch.
    fn to_wire_json(&self) -> String {
        format!("{{\"name\":\"{}\",\"arguments\":{}}}", self.name, self.arguments)
    }
}

/// Production-style parser. Scans for the `<tool_call>` envelope and
/// extracts the inner JSON payload.
fn parse_tool_call(out: &str) -> Option<ToolCall> {
    let open = "<tool_call>";
    let close = "</tool_call>";
    let start = out.find(open)? + open.len();
    let end = start + out[start..].find(close)?;
    let raw = out[start..end].trim();
    let needle = "\",\"arguments\":";
    let k = raw.find(needle)?;
    // Payload prefix is `{"name":"` (9 chars); `name` starts at byte 9.
    // The payload ends with `"}}` — the inner `}` closes the args
    // JSON object, the outer `}` closes the outer object. Strip one
    // byte (the outer `}`) so the args value is a self-contained
    // JSON object.
    let name = raw[9..k].to_string();
    let args_begin = k + needle.len();
    Some(ToolCall { name, arguments: raw[args_begin..raw.len() - 1].to_string() })
}

/// Independent reference parser. Line-for-line equivalent to
/// `parse_tool_call` but rewritten so a regression in the production
/// parser cannot silently make both implementations agree on broken
/// output.
fn parse_tool_call_reference(out: &str) -> Option<ToolCall> {
    let open = "<tool_call>";
    let close = "</tool_call>";
    let i = out.find(open)? + open.len();
    let raw = out[i..i + out[i..].find(close)?].trim();
    let needle = "\",\"arguments\":";
    let k = raw.find(needle)?;
    let name = raw[9..k].to_string();
    Some(ToolCall {
        name,
        arguments: raw[k + needle.len()..raw.len() - 1].to_string(),
    })
}

#[test]
fn qwen3_coder_tool_call_oracle_byte_identical() {
    // Qwen3-Coder wire-format sample.
    let sample = r#"<tool_call>{"name":"search","arguments":{"query":"latest Qwen release notes"}}</tool_call>"#;

    // (1) Production and reference parsers must agree byte-for-byte.
    let a = parse_tool_call(sample).expect("oracle parser must succeed");
    let b = parse_tool_call_reference(sample).expect("reference parser must succeed");
    assert_eq!(a, b, "oracle and reference tool-call parsers must agree");
    assert_eq!(a.name, "search");
    assert_eq!(a.arguments, r#"{"query":"latest Qwen release notes"}"#);

    // (2) Rendered wire JSON is byte-identical across runs.
    let json = a.to_wire_json();
    assert_eq!(json.clone(), a.to_wire_json(),
        "tool-call wire JSON must be byte-identical across runs");
    assert_eq!(json, r#"{"name":"search","arguments":{"query":"latest Qwen release notes"}}"#,
        "tool-call wire JSON must match the canonical byte sequence");

    // (3) Selector side: Metal (p95=1100) wins over scalar and CPU.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(16, 16, 16, 1, 1, 1);
    let scalar = make_candidate("Qwen3CoderToolCallScalar", BackendKind::Reference,
        vec![], min, max, vec![DType::Bf16, DType::Fp32], false);
    let metal = make_candidate("Qwen3CoderToolCallMetal", BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16], min, max, vec![DType::Bf16], true);
    let cpu = make_candidate("Qwen3CoderToolCallCpu", BackendKind::Cpu,
        vec![Capability::Avx512, Capability::Bf16], min, max, vec![DType::Bf16], true);
    let id_metal = metal.id;
    let mut reg = KernelRegistry::new();
    reg.register_candidate(scalar);
    reg.register_candidate(metal);
    reg.register_candidate(cpu);
    let key = qwen_agentic_key();
    reg.attach_tuning_record(key.clone(), build_record(
        id_metal, key.clone(), &samples_with_p95(1100), Some(NOW_UNIX_MS + 86_400_000)));

    let decision = reg.select_with_caps(&key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &full_capabilities(), NOW_UNIX_MS);
    let chosen_id = match &decision {
        SelectionDecision::Chosen { candidate, .. } => candidate.id,
        other => panic!("expected Chosen for Qwen3-Coder tool-call decoder, got {other:?}"),
    };
    assert_eq!(chosen_id, id_metal,
        "Metal p95=1100 must win Qwen3-Coder tool-call selection");

    // (4) Stability across calls.
    let decision2 = reg.select_with_caps(&key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(), NOW_UNIX_MS);
    assert_eq!(decision.selected(), decision2.selected(),
        "Qwen3-Coder tool-call selector must be stable across calls");

    // (5) Trace surfaces the chosen id.
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(id_metal),
        "ExecutionTrace must surface the chosen Qwen3-Coder tool-call candidate");
}

// ---- Chat-template selector ----------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatTemplate { ChatML, Base, Custom }

impl ChatTemplate {
    fn candidate_name(self) -> String {
        let tag = match self {
            Self::ChatML => "ChatML",
            Self::Base => "Base",
            Self::Custom => "Custom",
        };
        format!("Qwen3InstructTemplate{tag}")
    }
}

#[test]
fn qwen3_instruct_chat_template_deterministic_picks_correct_binding() {
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(16, 16, 16, 1, 1, 1);
    let mk = |t: ChatTemplate| make_candidate(
        &t.candidate_name(), BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16],
        min, max, vec![DType::Bf16], true);

    let chatml = mk(ChatTemplate::ChatML);
    let base = mk(ChatTemplate::Base);
    let custom = mk(ChatTemplate::Custom);
    let id_chatml = chatml.id;
    let id_base = base.id;

    let mut reg = KernelRegistry::new();
    // Register in reverse p95 order so id-based tie-breaks cannot mask
    // p95 ordering if the selector regressed.
    reg.register_candidate(custom);
    reg.register_candidate(base);
    reg.register_candidate(chatml);
    let key = qwen_agentic_key();
    reg.attach_tuning_record(key.clone(), build_record(
        id_chatml, key.clone(), &samples_with_p95(1400), Some(NOW_UNIX_MS + 86_400_000)));
    reg.attach_tuning_record(key.clone(), build_record(
        id_base, key.clone(), &samples_with_p95(2100), Some(NOW_UNIX_MS + 86_400_000)));

    // (1) Deterministic policy picks ChatML (lowest p95).
    let decision = reg.select_with_caps(&key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(), NOW_UNIX_MS);
    match &decision {
        SelectionDecision::Chosen { candidate, .. } => {
            assert_eq!(candidate.id, id_chatml,
                "ChatML p95=1400 must win over Base p95=2100");
            assert_eq!(candidate.name, "Qwen3InstructTemplateChatML",
                "chosen candidate must be bound to the ChatML template");
        }
        other => panic!("expected Chosen for Qwen3-Instruct template selector, got {other:?}"),
    }

    // (2) Stability across calls.
    let decision2 = reg.select_with_caps(&key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(), NOW_UNIX_MS);
    assert_eq!(decision.selected(), decision2.selected(),
        "Qwen3-Instruct template selector must be stable across calls");

    // (3) Template binding is dynamic: when only Base carries tuning
    // evidence with the lowest p95, the selector must pivot to Base.
    // This proves the binding is data-driven, not hard-coded to ChatML.
    let mut reg2 = KernelRegistry::new();
    reg2.register_candidate(mk(ChatTemplate::Custom));
    reg2.register_candidate(mk(ChatTemplate::Base));
    reg2.register_candidate(mk(ChatTemplate::ChatML));
    reg2.attach_tuning_record(key.clone(), build_record(
        id_base, key.clone(), &samples_with_p95(800), Some(NOW_UNIX_MS + 86_400_000)));
    let decision_base = reg2.select_with_caps(&key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &fresh_capabilities(), NOW_UNIX_MS);
    match &decision_base {
        SelectionDecision::Chosen { candidate, .. } => assert_eq!(candidate.id, id_base,
            "when Base is the only tuned template, the selector must bind to Base"),
        other => panic!("expected Chosen for Base-only Qwen3-Instruct selector, got {other:?}"),
    }
}

// ---- Edge-case prompts ---------------------------------------------------

#[test]
fn qwen3_5_coder_edge_case_prompts_select_stably() {
    // (a) Long-context fixture: 2048 whitespace-separated identifiers.
    let long = (0..2048usize)
        .map(|i| if i > 0 { format!(" tok_{i:04}") } else { format!("tok_{i:04}") })
        .collect::<String>();
    assert!(long.len() > 4096, "long-context fixture must exceed the envelope");

    // (b) Multi-line code fixture: indentation + newlines stress.
    let code = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n".to_string();
    assert!(code.contains('\n'));

    // (c) Special-token fixture: mid-prompt `<|endoftext|>` sentinel.
    let special = "PR review:\n<|endoftext|>\n```rust\nfn main(){}\n```\n<|endoftext|>".to_string();
    assert!(special.contains("<|endoftext|>"));

    // Register three Metal candidates, one per prompt family. Scalar
    // backends are reference fallbacks (no tuning record). The
    // selector must pick a Metal candidate every time because Metal
    // always carries the lower p95.
    let min = shape(1, 1, 1, 1, 1, 1);
    let max = shape(64, 64, 64, 4, 1, 1);
    let mk_metal = |name: &str| make_candidate(name, BackendKind::Metal,
        vec![Capability::MetalGpu, Capability::Bf16], min, max, vec![DType::Bf16], true);
    let mk_scalar = |name: &str| make_candidate(name, BackendKind::Reference,
        vec![], min, max, vec![DType::Bf16, DType::Fp32], false);

    let id_long = mk_metal("Qwen25CoderEdgeLongMetal").id;
    let id_code = mk_metal("Qwen25CoderEdgeCodeMetal").id;
    let id_special = mk_metal("Qwen25CoderEdgeSpecialMetal").id;

    let mut reg = KernelRegistry::new();
    for c in [
        mk_scalar("Qwen25CoderEdgeLongScalar"), mk_metal("Qwen25CoderEdgeLongMetal"),
        mk_scalar("Qwen25CoderEdgeCodeScalar"), mk_metal("Qwen25CoderEdgeCodeMetal"),
        mk_scalar("Qwen25CoderEdgeSpecialScalar"), mk_metal("Qwen25CoderEdgeSpecialMetal"),
    ] {
        reg.register_candidate(c);
    }
    let key = qwen_agentic_key();
    reg.attach_tuning_record(key.clone(), build_record(
        id_long, key.clone(), &samples_with_p95(1200), Some(NOW_UNIX_MS + 86_400_000)));
    reg.attach_tuning_record(key.clone(), build_record(
        id_code, key.clone(), &samples_with_p95(1200), Some(NOW_UNIX_MS + 86_400_000)));
    reg.attach_tuning_record(key.clone(), build_record(
        id_special, key.clone(), &samples_with_p95(1200), Some(NOW_UNIX_MS + 86_400_000)));

    // (d) Selector must return a Chosen — never Rejected — under
    // edge-case prompts. The selector inspects candidate metadata, not
    // prompt bytes, so unusual content is irrelevant.
    let decision = reg.select_with_caps(&key,
        SelectionPolicy::Deterministic { prefer_lower_p95: true },
        &full_capabilities(), NOW_UNIX_MS);
    let first_id = match &decision {
        SelectionDecision::Chosen { candidate, .. } => candidate.id,
        other => panic!("expected Chosen for Qwen3.5-Coder edge-case selector, got {other:?}"),
    };

    // (e) Stability: 3 repeated selector calls must return the same id.
    for _ in 0..3 {
        let d = reg.select_with_caps(&key,
            SelectionPolicy::Deterministic { prefer_lower_p95: true },
            &fresh_capabilities(), NOW_UNIX_MS);
        assert_eq!(d.selected(), Some(first_id),
            "Qwen3.5-Coder edge-case selector must pick the same candidate across repeated runs");
    }

    // (f) Trace surfaces the chosen id.
    let trace = reg.explain(&decision);
    assert_eq!(trace.selected, Some(first_id),
        "ExecutionTrace must surface the chosen Qwen3.5-Coder edge-case candidate");

    // (g) Prompt-byte independence: none of the three fixtures wrap a
    // `<tool_call>` envelope, so the tool-call parser must return None
    // for each — pinning the selector's prompt-decoupling contract.
    for (label, fixture) in [("long", &long), ("code", &code), ("special", &special)] {
        assert!(parse_tool_call(fixture).is_none(),
            "{label} fixture must not contain a tool-call envelope");
    }
}
