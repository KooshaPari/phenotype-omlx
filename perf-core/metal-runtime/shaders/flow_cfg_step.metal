#include <metal_stdlib>
using namespace metal;

// Fused classifier-free guidance + Euler flow step. All vectors are [n].
kernel void flow_cfg_step_f32(
    device const float* x [[buffer(0)]],
    device const float* velocity_uncond [[buffer(1)]],
    device const float* velocity_cond [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& guidance_scale [[buffer(5)]],
    constant float& dt [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    const float guided = velocity_uncond[gid] + guidance_scale *
        (velocity_cond[gid] - velocity_uncond[gid]);
    out[gid] = x[gid] + dt * guided;
}
