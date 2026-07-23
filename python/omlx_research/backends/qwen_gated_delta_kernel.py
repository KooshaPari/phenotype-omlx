"""Opt-in Qwen3.5 gated-delta Metal replacement.

The kernel mirrors MLX-LM's vector-gated delta update.  It is deliberately
opt-in and falls back to the upstream implementation for unsupported shapes.
"""

from __future__ import annotations

from typing import Any


_SOURCE = r"""
auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;
auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;
auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;
auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;
float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  state[i] = static_cast<float>(i_state[s_idx]);
}
auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
auto beta_ = beta + b_idx * T * Hv;
for (int t = 0; t < T; ++t) {
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] * g_[s_idx];
    kv_mem += state[i] * k_[s_idx];
  }
  kv_mem = simd_sum(kv_mem);
  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] + k_[s_idx] * delta;
    out += state[i] * q_[s_idx];
  }
  out = simd_sum(out);
  if (thread_index_in_simdgroup == 0) y[dv_idx] = static_cast<InT>(out);
  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
  g_ += Hv * Dk;
  beta_ += Hv;
}
for (int i = 0; i < n_per_t; ++i) {
  auto s_idx = n_per_t * dk_idx + i;
  o_state[s_idx] = static_cast<StT>(state[i]);
}
"""

_SOURCE_SCALAR = (
    _SOURCE.replace(
        "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;",
        "auto g_ = g + b_idx * T * Hv;",
    )
    .replace("state[i] = state[i] * g_[s_idx];", "state[i] = state[i] * g_[hv_idx];")
    .replace("g_ += Hv * Dk;", "g_ += Hv;")
)


def install(model: Any, mx: Any) -> dict[str, Any]:
    """Install the replacement into Qwen3.5's gated-delta call site."""
    import mlx_lm.models.gated_delta as gd
    import mlx_lm.models.qwen3_5 as qwen35

    original = qwen35.gated_delta_update
    stats = {"dispatches": 0, "fallbacks": 0, "installed": True}
    vector_kernel = mx.fast.metal_kernel(
        name="phenotype_qwen35_gated_delta_vec",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=_SOURCE,
    )
    scalar_kernel = mx.fast.metal_kernel(
        name="phenotype_qwen35_gated_delta_scalar",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"],
        source=_SOURCE_SCALAR,
    )

    def replacement(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        if (
            not use_kernel
            or mask is not None
            or mx.default_device() != mx.gpu
            or not mx.metal.is_available()
            or q.ndim != 4
            or q.shape[-1] % 32
        ):
            stats["fallbacks"] += 1
            return original(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
        g = gd.compute_g(A_log, a, dt_bias)
        beta = mx.sigmoid(b)
        B, T, Hk, Dk = q.shape
        Hv, Dv = v.shape[2:]
        if state is None:
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        kernel = vector_kernel if g.ndim == 4 else scalar_kernel
        kernel_kind = "vector" if g.ndim == 4 else "scalar"
        result = kernel(
            inputs=[q, k, v, g, beta, state, T],
            template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
            grid=(32, Dv, B * Hv),
            threadgroup=(32, 4, 1),
            output_shapes=[(B, T, Hv, Dv), state.shape],
            output_dtypes=[q.dtype, state.dtype],
        )
        stats["dispatches"] += 1
        stats["kernel_kinds"] = stats.get("kernel_kinds", {})
        stats["kernel_kinds"][kernel_kind] = stats["kernel_kinds"].get(kernel_kind, 0) + 1
        return result

    qwen35.gated_delta_update = replacement
    return stats
