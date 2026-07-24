#include <metal_stdlib>
using namespace metal;

kernel void rwkv7_time_mix_f32(
    device const float* x [[buffer(0)]],
    device float* state [[buffer(1)]],
    device const float* params [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    float nk = params[0] * x[0] + (1.0f - params[0]) * state[0];
    float nv = params[1] * x[1] + (1.0f - params[1]) * state[1];
    float nr = params[2] * x[2] + (1.0f - params[2]) * state[2];
    float nw = params[3] * x[3] + (1.0f - params[3]) * state[3];
    out[0] = nv * tanh(nw * params[4]);
    state[0] = nk; state[1] = nv; state[2] = nr; state[3] = nw;
}
