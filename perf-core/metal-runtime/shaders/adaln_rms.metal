#include <metal_stdlib>
using namespace metal;

// Fused adaptive RMS normalization. x, scale, shift, and out are [tokens,d].
// The conditioning stream supplies per-token affine parameters.
kernel void adaln_rms_f32(
    device const float* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* shift [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
    const uint token = gid.x;
    const uint lane = gid.y;
    if (token >= tokens || lane >= dim) return;
    const uint base = token * dim;
    float sum_sq = 0.0f;
    for (uint d = 0; d < dim; ++d) sum_sq += x[base + d] * x[base + d];
    const float inv_rms = rsqrt(sum_sq / float(dim) + epsilon);
    out[base + lane] = (x[base + lane] * inv_rms) * (1.0f + scale[base + lane]) + shift[base + lane];
}
