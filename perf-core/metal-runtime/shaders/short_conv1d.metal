#include <metal_stdlib>
using namespace metal;

// One LFM2 short-convolution step. History is the previous k-1 inputs.
kernel void short_conv1d_step_f32(
    device const float* x [[buffer(0)]],
    device const float* kernel_weights [[buffer(1)]],
    device const float* history [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& taps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    float acc = 0.0f;
    for (uint i = 0; i < taps; ++i) {
        const float sample = (i + 1u == taps) ? x[0] : history[i];
        acc += sample * kernel_weights[i];
    }
    out[0] = acc;
}
