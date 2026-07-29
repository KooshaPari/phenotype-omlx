#include <metal_stdlib>
using namespace metal;

// Apply the confidence floor after a denoise proposal. `candidate_mask` is
// the mask produced by the proposal stage; low-confidence positions become
// active again for the next pass.
kernel void diffusion_remask_confidence_f32(
    device const uchar* candidate_mask [[buffer(0)]],
    device const float* confidence [[buffer(1)]], device uchar* next_mask [[buffer(2)]],
    constant float& threshold [[buffer(3)]], constant uint& tokens [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= tokens) return;
    next_mask[gid] = (candidate_mask[gid] != 0 || confidence[gid] < threshold) ? 1u : 0u;
}
