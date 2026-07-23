#include <metal_stdlib>
using namespace metal;

// RetNet recurrent retention (single head, one token).
// R_t = gamma * R_{t-1} + k_t outer v_t; y_t = q_t^T R_t.
kernel void retnet_retention_step_f32(
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device const float *state [[buffer(3)]],
    device const float *decay [[buffer(4)]], device float *out [[buffer(5)]],
    device float *next [[buffer(6)]], constant uint &n [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    float gamma = decay[0];
    for (uint i = 0; i < n; ++i) {
        for (uint j = 0; j < n; ++j) {
            next[i * n + j] = gamma * state[i * n + j] + k[i] * v[j];
        }
    }
    for (uint j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (uint i = 0; i < n; ++i) acc += q[i] * next[i * n + j];
        out[j] = acc;
    }
}
