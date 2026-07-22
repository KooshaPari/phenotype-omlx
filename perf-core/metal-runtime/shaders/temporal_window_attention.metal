#include <metal_stdlib>
using namespace metal;

// Windowed causal attention over [tokens, heads, head_dim]. The window is
// aligned to the query token and excludes future keys, bounding memory and
// score work for long video sequences.
kernel void temporal_window_attention_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& window [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]) {
    const uint token = gid.x;
    const uint head = gid.y;
    const uint lane = gid.z;
    if (token >= tokens || head >= heads || lane >= head_dim || window == 0u) return;
    const uint base = (token * heads + head) * head_dim;
    const uint first = token + 1u > window ? token + 1u - window : 0u;
    float max_score = -INFINITY;
    for (uint key = first; key <= token; ++key) {
        const uint key_base = (key * heads + head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += q[base + d] * k[key_base + d];
        max_score = max(max_score, score * scale);
    }
    float denom = 0.0f;
    float numer = 0.0f;
    for (uint key = first; key <= token; ++key) {
        const uint key_base = (key * heads + head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += q[base + d] * k[key_base + d];
        const float weight = exp(score * scale - max_score);
        denom += weight;
        numer += weight * v[key_base + lane];
    }
    out[base + lane] = numer / denom;
}
