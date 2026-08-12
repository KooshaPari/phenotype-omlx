#include <metal_stdlib>
using namespace metal;

kernel void diffusion_argmax_confidence_f32(
    device const float* logits [[buffer(0)]], device uint* token_ids [[buffer(1)]],
    device float* confidence [[buffer(2)]], constant uint& tokens [[buffer(3)]],
    constant uint& vocab [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= tokens) return;
    const uint base = gid * vocab;
    float max_logit = -INFINITY;
    uint argmax = 0;
    uint positive_infinity_count = 0;
    bool has_positive_infinity = false;

    // MDLM/LLaDA can intentionally produce an all-masked row (-inf). Ignore
    // NaNs rather than allowing one invalid lane to poison the row's score.
    // Positive infinities are handled as a tied max below.
    for (uint j = 0; j < vocab; ++j) {
        const float value = logits[base + j];
        if (isnan(value)) continue;
        if (value == INFINITY) {
            has_positive_infinity = true;
            positive_infinity_count += 1;
            if (max_logit != INFINITY) {
                max_logit = INFINITY;
                argmax = j;
            }
        } else if (!has_positive_infinity && value > max_logit) {
            max_logit = value;
            argmax = j;
        }
    }

    token_ids[gid] = argmax;
    if (has_positive_infinity) {
        confidence[gid] = 1.0f / float(positive_infinity_count);
        return;
    }
    if (!isfinite(max_logit)) {
        // No finite logits: this is a fully masked/invalid row. Returning zero
        // makes the remask policy deterministic and, critically, not NaN.
        confidence[gid] = 0.0f;
        return;
    }
    float denominator = 0.0f;
    for (uint j = 0; j < vocab; ++j) {
        const float value = logits[base + j];
        if (!isnan(value) && value != INFINITY) {
            denominator += exp(value - max_logit);
        }
    }
    confidence[gid] = denominator > 0.0f ? 1.0f / denominator : 0.0f;
}
