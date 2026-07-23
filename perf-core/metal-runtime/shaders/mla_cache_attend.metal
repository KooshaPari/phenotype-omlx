#include <metal_stdlib>
using namespace metal;

// DeepSeek-style MLA cache attention. One thread handles one decode query.
// compressed_kv is [entries, d_latent], k_rope is [entries, d_rope].
kernel void mla_cache_attend_f32(
    device const float* q_latent [[buffer(0)]],
    device const float* q_rope [[buffer(1)]],
    device const float* compressed_kv [[buffer(2)]],
    device const float* k_rope [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& entries [[buffer(5)]],
    constant uint& d_latent [[buffer(6)]],
    constant uint& d_rope [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0 || d_latent == 0) return;
    for (uint d = 0; d < d_latent; ++d) out[d] = 0.0f;
    if (entries == 0) return;
    float max_score = -INFINITY;
    for (uint t = 0; t < entries; ++t) {
        float score = 0.0f;
        for (uint d = 0; d < d_latent; ++d) score += q_latent[d] * compressed_kv[t * d_latent + d];
        for (uint d = 0; d < d_rope; ++d) score += q_rope[d] * k_rope[t * d_rope + d];
        max_score = max(max_score, score);
    }
    float denom = 0.0f;
    for (uint t = 0; t < entries; ++t) {
        float score = 0.0f;
        for (uint d = 0; d < d_latent; ++d) score += q_latent[d] * compressed_kv[t * d_latent + d];
        for (uint d = 0; d < d_rope; ++d) score += q_rope[d] * k_rope[t * d_rope + d];
        denom += exp(score - max_score);
    }
    if (!(denom > 0.0f)) return;
    for (uint t = 0; t < entries; ++t) {
        float score = 0.0f;
        for (uint d = 0; d < d_latent; ++d) score += q_latent[d] * compressed_kv[t * d_latent + d];
        for (uint d = 0; d < d_rope; ++d) score += q_rope[d] * k_rope[t * d_rope + d];
        float weight = exp(score - max_score) / denom;
        for (uint d = 0; d < d_latent; ++d) out[d] += weight * compressed_kv[t * d_latent + d];
    }
}
