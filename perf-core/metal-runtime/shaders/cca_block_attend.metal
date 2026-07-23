#include <metal_stdlib>
using namespace metal;

// ZAYA CCA: one thread computes a bounded block-parallel attention row.
kernel void cca_block_attend_f32(
    device const float* q [[buffer(0)]],
    device const float* summaries [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const uint* sizes [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0 || dim == 0) return;
    if (blocks == 0) { for (uint d = 0; d < dim; ++d) out[d] = 0.0f; return; }
    float max_score = -INFINITY;
    for (uint b = 0; b < blocks; ++b) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d) dot += q[d] * summaries[b * dim + d];
        max_score = max(max_score, dot * scales[b]);
    }
    float denom = 0.0f;
    for (uint b = 0; b < blocks; ++b) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d) dot += q[d] * summaries[b * dim + d];
        denom += exp(dot * scales[b] - max_score);
    }
    for (uint d = 0; d < dim; ++d) out[d] = 0.0f;
    if (!(denom > 0.0f)) return;
    for (uint b = 0; b < blocks; ++b) {
        float dot = 0.0f;
        for (uint d = 0; d < dim; ++d) dot += q[d] * summaries[b * dim + d];
        float weight = exp(dot * scales[b] - max_score) / denom * float(sizes[b]);
        for (uint d = 0; d < dim; ++d) out[d] += weight * summaries[b * dim + d];
    }
}
