#include <metal_stdlib>
using namespace metal;

// Fused 3-axis RoPE for image/video diffusion transformers.
//
// q/k: [tokens, heads, head_dim], contiguous float buffers
// positions: [tokens, 3] = (time, height, width)
// inv_freq_*: axis-specific inverse frequencies, length head_dim/6
// Each axis owns one third of the rotary pairs. Non-rotary tail lanes are
// copied unchanged, allowing head dimensions larger than the rotary region.
kernel void rope_3d_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* q_out [[buffer(2)]],
    device float* k_out [[buffer(3)]],
    device const uint3* positions [[buffer(4)]],
    device const float* inv_freq_t [[buffer(5)]],
    device const float* inv_freq_h [[buffer(6)]],
    device const float* inv_freq_w [[buffer(7)]],
    constant uint& tokens [[buffer(8)]],
    constant uint& heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& rotary_pairs_per_axis [[buffer(11)]],
    uint3 gid [[thread_position_in_grid]]) {
    const uint token = gid.x;
    const uint head = gid.y;
    const uint lane = gid.z;
    if (token >= tokens || head >= heads || lane >= head_dim) return;

    const uint base = (token * heads + head) * head_dim;
    const uint rotary_dim = rotary_pairs_per_axis * 6;
    if (lane >= rotary_dim) {
        q_out[base + lane] = q[base + lane];
        k_out[base + lane] = k[base + lane];
        return;
    }

    const uint pair = lane / 2;
    const uint axis_pair = pair % rotary_pairs_per_axis;
    const uint axis = pair / rotary_pairs_per_axis;
    const uint mate = (lane & 1u) ? lane - 1u : lane + 1u;
    const float position = axis == 0 ? float(positions[token].x)
                         : axis == 1 ? float(positions[token].y)
                                     : float(positions[token].z);
    const float inv_freq = axis == 0 ? inv_freq_t[axis_pair]
                           : axis == 1 ? inv_freq_h[axis_pair]
                                       : inv_freq_w[axis_pair];
    const float angle = position * inv_freq;
    const float c = cos(angle);
    const float s = sin(angle);
    const float q_even = q[base + (lane & ~1u)];
    const float q_odd = q[base + (lane | 1u)];
    const float k_even = k[base + (lane & ~1u)];
    const float k_odd = k[base + (lane | 1u)];
    const bool odd = (lane & 1u) != 0;
    q_out[base + lane] = odd ? q_even * s + q_odd * c : q_even * c - q_odd * s;
    k_out[base + lane] = odd ? k_even * s + k_odd * c : k_even * c - k_odd * s;
    (void)mate;
}
