#include <metal_stdlib>
using namespace metal;

// Bonsai/BitNet-style 2-bit sign-magnitude weights. Codes are 00=zero,
// 01=+1, 10=-1; the unused 11 code is treated as zero. Packed weights are
// output-major [n, ceil(k/4)] and activations are [m,k].
kernel void ternary_gemm_f32(
    device const float* activations [[buffer(0)]],
    device const uchar* packed_weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
    const uint row = gid.x;
    const uint col = gid.y;
    if (row >= m || col >= n) return;
    float sum = 0.0f;
    const uint packed_stride = (k + 3u) / 4u;
    for (uint d = 0; d < k; ++d) {
        const uchar code = (packed_weights[col * packed_stride + d / 4u] >> ((d & 3u) * 2u)) & 3u;
        const float weight = code == 1u ? 1.0f : code == 2u ? -1.0f : 0.0f;
        sum += activations[row * k + d] * weight;
    }
    out[row * n + col] = sum * scales[col];
}
