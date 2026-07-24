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
    const uint weight_base = col * packed_stride;
    const uint activation_base = row * k;
    // Decode four 2-bit ternary weights per byte. Keeping the packed byte in
    // a register avoids a division and global-memory address calculation for
    // every weight; the compiler can fully unroll this fixed four-lane loop.
    for (uint byte_idx = 0; byte_idx < packed_stride; ++byte_idx) {
        const uchar packed = packed_weights[weight_base + byte_idx];
        const uint d0 = byte_idx * 4u;
        for (uint lane = 0; lane < 4u; ++lane) {
            const uint d = d0 + lane;
            if (d >= k) break;
            const uchar code = (packed >> (lane * 2u)) & 3u;
            const float weight = code == 1u ? 1.0f : code == 2u ? -1.0f : 0.0f;
            sum += activations[activation_base + d] * weight;
        }
    }
    out[row * n + col] = sum * scales[col];
}
