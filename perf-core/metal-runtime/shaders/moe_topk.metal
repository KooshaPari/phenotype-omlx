#include <metal_stdlib>
using namespace metal;

// One thread routes one token. Fixed-size private arrays bound top_k <= 8;
// the runtime expert count is supported through 256 for current Qwen and
// DeepSeek-style sparse-MoE contracts.
kernel void moe_topk_f32(
    device const float* logits [[buffer(0)]],
    device uint* expert_ids [[buffer(1)]],
    device float* weights [[buffer(2)]],
    constant uint& experts [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    uint token [[thread_position_in_grid]]) {
    float selected[8];
    uint selected_ids[8];
    for (uint rank = 0; rank < top_k; ++rank) {
        selected[rank] = -INFINITY;
        selected_ids[rank] = UINT_MAX;
    }

    const uint row = token * experts;
    for (uint expert = 0; expert < experts; ++expert) {
        const float score = logits[row + expert];
        for (uint rank = 0; rank < top_k; ++rank) {
            if (score > selected[rank] ||
                (score == selected[rank] && expert < selected_ids[rank])) {
                for (uint shift = top_k - 1; shift > rank; --shift) {
                    selected[shift] = selected[shift - 1];
                    selected_ids[shift] = selected_ids[shift - 1];
                }
                selected[rank] = score;
                selected_ids[rank] = expert;
                break;
            }
        }
    }

    const float maximum = selected[0];
    float denominator = 0.0f;
    for (uint rank = 0; rank < top_k; ++rank) {
        selected[rank] = exp(selected[rank] - maximum);
        denominator += selected[rank];
    }
    const uint output = token * top_k;
    for (uint rank = 0; rank < top_k; ++rank) {
        expert_ids[output + rank] = selected_ids[rank];
        weights[output + rank] = selected[rank] / denominator;
    }
}
