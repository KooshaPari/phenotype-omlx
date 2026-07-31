#include <metal_stdlib>
using namespace metal;

// Update confidence trajectory metadata after a denoise proposal.
kernel void diffusion_trajectory_update_f32(
    device const float* previous_confidence [[buffer(0)]],
    device const float* confidence [[buffer(1)]], device const float* entropy [[buffer(2)]],
    device float* momentum [[buffer(3)]], device uchar* converged [[buffer(4)]],
    constant float& confidence_threshold [[buffer(5)]],
    constant float& momentum_threshold [[buffer(6)]], constant uint& tokens [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= tokens) return;
    const float delta = fabs(confidence[gid] - previous_confidence[gid]);
    momentum[gid] = delta;
    converged[gid] = (confidence[gid] >= confidence_threshold &&
                      delta <= momentum_threshold && entropy[gid] >= 0.0f) ? 1u : 0u;
}
