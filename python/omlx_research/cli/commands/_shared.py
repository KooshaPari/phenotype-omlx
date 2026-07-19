"""Shared schema constants for CLI commands.

Keeping the model-plan and trace schemas in one place makes it easy for
``inspect``, ``replay``, ``compare``, and ``evidence`` to agree on the
document shape without importing each other.

These are intentionally minimal — this is a *research* CLI, not a production
validator. The schema is enforced by hand-rolled checks rather than
``jsonschema`` to avoid a new top-level dependency.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# OperatorKind registry (single source of truth for inspect / explain / tune).
# ---------------------------------------------------------------------------

#: Set of operator kinds recognized by ``explain`` / ``tune``. Anything outside
#: this set exits with code 4 ("unknown op-kind").
OPERATOR_KINDS: tuple[str, ...] = (
    "DenseMatmul",
    "GroupedMatmul",
    "RoPE",
    "RMSNorm",
    "Softmax",
    "SiLU",
    "Embedding",
    "Attention",
)

#: Per-operator canonical contract consumed by ``explain``. Each entry lists
#: the input/output dtypes, shape rule, and dependency rules in plain prose.
OPERATOR_CONTRACTS: dict[str, dict[str, object]] = {
    "DenseMatmul": {
        "inputs": [("A", "f16"), ("B", "f16"), ("bias", "f16")],
        "outputs": [("Y", "f16")],
        "shape_rule": "Y[m, n] = A[m, k] @ B[k, n] + bias[n]",
        "deps": "Requires A, B, bias produced upstream; Y feeds downstream matmul/attention.",
        "kernels": ("gemm_metal_fp16", "gemm_mlx_fp16", "gemm_tiled_fp16"),
    },
    "GroupedMatmul": {
        "inputs": [("A", "f16"), ("B", "f16"), ("group_ids", "i32")],
        "outputs": [("Y", "f16")],
        "shape_rule": "Y[g, m, n] = sum_g A_g[m, k] @ B_g[k, n] over group_ids",
        "deps": "group_ids must be produced before this op; Y is consumed per-group.",
        "kernels": ("gemm_grouped_metal", "gemm_grouped_mlx", "gemm_grouped_cpu"),
    },
    "RoPE": {
        "inputs": [("x", "f16"), ("freqs", "f16")],
        "outputs": [("y", "f16")],
        "shape_rule": "y[b, s, d] = rotate_half(x[b, s, d]) * freqs[s, d/2]",
        "deps": "freqs table must be available; y preserves shape of x.",
        "kernels": ("rope_metal", "rope_mlx", "rope_neon"),
    },
    "RMSNorm": {
        "inputs": [("x", "f16"), ("weight", "f16"), ("eps", "f32")],
        "outputs": [("y", "f16")],
        "shape_rule": "y[b, s, d] = x[b, s, d] / sqrt(mean(x^2) + eps) * weight[d]",
        "deps": "weight is a parameter; eps is a constant scalar.",
        "kernels": ("rmsnorm_metal", "rmsnorm_mlx", "rmsnorm_fallback"),
    },
    "Softmax": {
        "inputs": [("x", "f16")],
        "outputs": [("y", "f16")],
        "shape_rule": "y[i, j] = exp(x[i, j]) / sum_k exp(x[i, k])  (last dim)",
        "deps": "None beyond input; numerically safe with row-max subtract.",
        "kernels": ("softmax_metal", "softmax_mlx", "softmax_cpu"),
    },
    "SiLU": {
        "inputs": [("x", "f16")],
        "outputs": [("y", "f16")],
        "shape_rule": "y = x * sigmoid(x)  (elementwise, shape preserved)",
        "deps": "None beyond input.",
        "kernels": ("silu_metal", "silu_mlx", "silu_neon"),
    },
    "Embedding": {
        "inputs": [("table", "f16"), ("ids", "i32")],
        "outputs": [("y", "f16")],
        "shape_rule": "y[b, s, d] = table[ids[b, s]]",
        "deps": "table is a parameter; ids produced by tokenizer/token-routing op.",
        "kernels": ("embedding_metal", "embedding_mlx", "embedding_gather"),
    },
    "Attention": {
        "inputs": [("Q", "f16"), ("K", "f16"), ("V", "f16"), ("mask", "i8")],
        "outputs": [("Y", "f16")],
        "shape_rule": "Y[b, h, s, d] = softmax(QK^T / sqrt(d) + mask) @ V",
        "deps": "Q/K/V projections upstream; mask built from tree/padding state.",
        "kernels": ("attn_metal", "attn_mlx", "attn_fallback"),
    },
}

# ---------------------------------------------------------------------------
# Model-plan JSON Schema (kept inline; hand-rolled checks, no jsonschema dep).
# ---------------------------------------------------------------------------

#: Required top-level keys for a model plan.
PLAN_REQUIRED_KEYS: tuple[str, ...] = (
    "plan_id",
    "name",
    "family",
    "scheduler_policy",
    "operators",
)

#: Required keys per-operator entry.
OPERATOR_REQUIRED_KEYS: tuple[str, ...] = ("op_id", "kind", "inputs", "outputs")

#: Required keys per-state entry (states are optional in the plan).
STATE_REQUIRED_KEYS: tuple[str, ...] = ("state_id", "kind", "persistence", "dtype", "owning_op")

#: Required keys per-edge entry (edges are optional).
EDGE_REQUIRED_KEYS: tuple[str, ...] = ("from_id", "to_id")

#: Supported scheduler policies for the plan header.
SCHEDULER_POLICIES: tuple[str, ...] = (
    "fifo",
    "priority",
    "critical_path",
    "dataflow",
)


def validate_plan(plan: object) -> list[str]:
    """Return a list of validation errors (empty list means plan is valid).

    Performs hand-rolled structural checks. Unknown / wrong-type fields are
    reported as human-readable strings so the CLI can print them.
    """
    errors: list[str] = []
    if not isinstance(plan, dict):
        return [f"plan must be a JSON object, got {type(plan).__name__}"]

    for key in PLAN_REQUIRED_KEYS:
        if key not in plan:
            errors.append(f"missing required key: {key!r}")

    if "scheduler_policy" in plan and plan["scheduler_policy"] not in SCHEDULER_POLICIES:
        errors.append(
            f"scheduler_policy {plan['scheduler_policy']!r} not in {SCHEDULER_POLICIES}"
        )

    ops = plan.get("operators")
    if ops is not None and not isinstance(ops, list):
        errors.append("'operators' must be a list")
    elif isinstance(ops, list):
        seen_ids: set[str] = set()
        for i, op in enumerate(ops):
            if not isinstance(op, dict):
                errors.append(f"operators[{i}] must be an object")
                continue
            for key in OPERATOR_REQUIRED_KEYS:
                if key not in op:
                    errors.append(f"operators[{i}] missing key {key!r}")
            op_id = op.get("op_id")
            if isinstance(op_id, str):
                if op_id in seen_ids:
                    errors.append(f"duplicate op_id {op_id!r}")
                seen_ids.add(op_id)
            kind = op.get("kind")
            if isinstance(kind, str) and kind not in OPERATOR_KINDS:
                errors.append(f"operators[{i}].kind {kind!r} unknown")

    states = plan.get("states")
    if states is not None and not isinstance(states, list):
        errors.append("'states' must be a list")
    elif isinstance(states, list):
        seen: set[str] = set()
        for i, st in enumerate(states):
            if not isinstance(st, dict):
                errors.append(f"states[{i}] must be an object")
                continue
            for key in STATE_REQUIRED_KEYS:
                if key not in st:
                    errors.append(f"states[{i}] missing key {key!r}")
            sid = st.get("state_id")
            if isinstance(sid, str):
                if sid in seen:
                    errors.append(f"duplicate state_id {sid!r}")
                seen.add(sid)

    edges = plan.get("edges")
    if edges is not None and not isinstance(edges, list):
        errors.append("'edges' must be a list")
    elif isinstance(edges, list):
        for i, e in enumerate(edges):
            if not isinstance(e, dict):
                errors.append(f"edges[{i}] must be an object")
                continue
            for key in EDGE_REQUIRED_KEYS:
                if key not in e:
                    errors.append(f"edges[{i}] missing key {key!r}")

    return errors


# ---------------------------------------------------------------------------
# Trace JSON shape consumed by replay / compare / evidence.
# ---------------------------------------------------------------------------

#: Required top-level keys for an execution trace.
TRACE_REQUIRED_KEYS: tuple[str, ...] = (
    "plan_id",
    "op_id",
    "selected",
    "rejected",
)


def validate_trace(trace: object) -> list[str]:
    """Return a list of validation errors (empty list means trace is valid)."""
    errors: list[str] = []
    if not isinstance(trace, dict):
        return [f"trace must be a JSON object, got {type(trace).__name__}"]
    for key in TRACE_REQUIRED_KEYS:
        if key not in trace:
            errors.append(f"missing required key: {key!r}")
    sel = trace.get("selected")
    if sel is not None and not isinstance(sel, dict):
        errors.append("'selected' must be an object")
    rej = trace.get("rejected")
    if rej is not None and not isinstance(rej, list):
        errors.append("'rejected' must be a list")
    return errors


def parse_shape(spec: str | None) -> list[int] | None:
    """Parse a comma-separated shape spec like ``"m,n,k,batch,seq"``.

    Returns ``None`` when ``spec`` is empty (i.e. ``--shape`` not provided).
    Raises ``ValueError`` on malformed input so the caller can map to the
    appropriate exit code.
    """
    if spec is None or spec == "":
        return None
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            raise ValueError(f"empty dimension in shape {spec!r}")
        try:
            out.append(int(tok))
        except ValueError as e:
            raise ValueError(f"non-integer dimension {tok!r} in shape {spec!r}") from e
    return out


def shape_hash(shape: list[int] | None, op_kind: str) -> str:
    """Stable short hash of (op_kind, shape) used as a cache key."""
    import hashlib
    payload = f"{op_kind}|{','.join(str(d) for d in (shape or []))}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
