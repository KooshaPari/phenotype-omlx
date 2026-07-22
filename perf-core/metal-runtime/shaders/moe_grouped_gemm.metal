#include <metal_stdlib>
using namespace metal;

// Assignment-list grouped GEMM for MoE decode/prefill.
//
// A:              [tokens, k] row-major
// expert_weights: [experts, k, n] row-major, expert-major
// assignment_tokens/expert_ids: [assignments]
// out:            [assignments, n] row-major
//
// One thread computes one output element. The assignment-list ABI keeps
// capacity padding and dropped-token policy in the dispatch layer, while
// preserving contiguous expert-major weights for cache-friendly loads.
kernel void moe_grouped_gemm_f32(
    device const float* A [[buffer(0)]],
    device const float* expert_weights [[buffer(1)]],
    device const uint* assignment_tokens [[buffer(2)]],
    device const uint* assignment_experts [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& assignments [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint assignment = gid.x;
    const uint column = gid.y;
    if (assignment >= assignments || column >= n || k == 0 || n == 0) return;

    const uint token = assignment_tokens[assignment];
    const uint expert = assignment_experts[assignment];
    const device float* a_row = A + token * k;
    const device float* b = expert_weights + expert * k * n;

    float acc = 0.0f;
    for (uint kk = 0; kk < k; ++kk) {
        acc = fma(a_row[kk], b[kk * n + column], acc);
    }
    out[assignment * n + column] = acc;
}
