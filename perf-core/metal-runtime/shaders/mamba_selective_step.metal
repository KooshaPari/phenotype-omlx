#include <metal_stdlib>
using namespace metal;
kernel void mamba_selective_step_f32(
    device const float* a_log [[buffer(0)]], device float* state [[buffer(1)]],
    device const float* p [[buffer(2)]], device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    const float u=p[0], dt=p[1], b=p[2], c=p[3], d=p[4];
    const float dbu=dt*b*u; float acc=d*u;
    for (uint i=0; i<n; ++i) { state[i]=exp(dt*exp(a_log[i]))*state[i]+dbu; acc += c*state[i]; }
    out[0]=acc;
}
