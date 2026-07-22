#include <metal_stdlib>
using namespace metal;

// Small-shape correctness kernel for Flux/SD3-style joint attention.
// q is the image/query stream; k and v contain the concatenated image+text
// context stream. Layout for all tensors is [tokens, heads, head_dim].
kernel void joint_attention_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& q_tokens [[buffer(4)]],
    constant uint& kv_tokens [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]) {
    const uint token = gid.x;
    const uint head = gid.y;
    const uint lane = gid.z;
    if (token >= q_tokens || head >= heads || lane >= head_dim) return;
    const uint out_base = (token * heads + head) * head_dim;
    float max_score = -INFINITY;
    for (uint key = 0; key < kv_tokens; ++key) {
        float score = 0.0f;
        const uint q_base = out_base;
        const uint k_base = (key * heads + head) * head_dim;
        for (uint d = 0; d < head_dim; ++d) score += q[q_base + d] * k[k_base + d];
        max_score = max(max_score, score * scale);
    }
    float denom = 0.0f;
    float numer = 0.0f;
    for (uint key = 0; key < kv_tokens; ++key) {
        float score = 0.0f;
        const uint q_base = out_base;
        const uint k_base = (key * heads + head) * head_dim;
        for (uint d = 0; d < head_dim; ++d) score += q[q_base + d] * k[k_base + d];
        const float weight = exp(score * scale - max_score);
        denom += weight;
        numer += weight * v[k_base + lane];
    }
    out[out_base + lane] = numer / denom;
}
