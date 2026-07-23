#include <metal_stdlib>
using namespace metal;

kernel void diffusion_argmax_confidence_f32(
    device const float* logits [[buffer(0)]], device uint* token_ids [[buffer(1)]],
    device float* confidence [[buffer(2)]], constant uint& tokens [[buffer(3)]],
    constant uint& vocab [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= tokens) return;
    const uint base = gid * vocab; float max_logit = -INFINITY; uint argmax = 0;
    for (uint j = 0; j < vocab; ++j) { float value = logits[base + j]; if (value > max_logit) { max_logit = value; argmax = j; } }
    float denominator = 0.0f;
    for (uint j = 0; j < vocab; ++j) denominator += exp(logits[base + j] - max_logit);
    token_ids[gid] = argmax; confidence[gid] = 1.0f / denominator;
}
