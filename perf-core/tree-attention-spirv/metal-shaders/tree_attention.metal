// Tree-attention kernel — Metal Shading Language.
// Wired up via Swift→MSL→Runtime; Rust crate calls via the Swift bridge.
//
// Layout:
//   Q, K, V : (B, H, T, D) float16 (or float32)
//   Tree mask : (T, W) where W = tree_width (number of candidates per token)
//
// The kernel performs: out = softmax(Q @ K[mask].T / sqrt(D)) @ V[mask]
// for a tree-structured draft (JetSpec / EAGLE / Medusa style).

#include <metal_stdlib>
using namespace metal;

constant uint32_t HEAD_DIM = 128;
constant float SCALE = 0.08838834764831845f; // 1/sqrt(128)

struct TreeAttnParams {
    uint32_t B;       // batch
    uint32_t H;       // heads
    uint32_t T;       // sequence length
    uint32_t W;       // tree width
    uint32_t D;       // head dim (must be HEAD_DIM)
};

kernel void tree_attention_fwd(
    device const half*  Q            [[buffer(0)]],
    device const half*  K            [[buffer(1)]],
    device const half*  V            [[buffer(2)]],
    device const int*   tree_mask    [[buffer(3)]],   // (T, W) parent indices, -1 = root
    device       half*  Out          [[buffer(4)]],
    constant    TreeAttnParams& p    [[buffer(5)]],
    uint3                tid         [[thread_position_in_grid]])
{
    const uint b = tid.x;
    const uint h = tid.y;
    const uint t = tid.z;
    if (b >= p.B || h >= p.H || t >= p.T) return;

    // Gather K/V positions for this tree branch
    int pos[64]; // W <= 64 supported
    int depth = 0;
    int cur = t;
    while (cur >= 0 && depth < (int)p.W) {
        pos[depth++] = cur;
        cur = tree_mask[cur * p.W + (depth - 1)];
    }
    if (depth == 0) return;

    // Compute Q.K^T
    float scores[64] = {0};
    for (uint d = 0; d < p.D; ++d) {
        const float q = (float)Q[((b * p.H + h) * p.T + t) * p.D + d];
        for (int k = 0; k < depth; ++k) {
            const float kv = (float)K[((b * p.H + h) * p.T + pos[k]) * p.D + d];
            scores[k] += q * kv;
        }
    }
    for (int k = 0; k < depth; ++k) scores[k] *= SCALE;

    // Softmax
    float max_s = -INFINITY;
    for (int k = 0; k < depth; ++k) max_s = max(max_s, scores[k]);
    float sum = 0;
    for (int k = 0; k < depth; ++k) { scores[k] = exp(scores[k] - max_s); sum += scores[k]; }
    for (int k = 0; k < depth; ++k) scores[k] /= sum;

    // Weighted V
    for (uint d = 0; d < p.D; ++d) {
        float acc = 0;
        for (int k = 0; k < depth; ++k) {
            const float v = (float)V[((b * p.H + h) * p.T + pos[k]) * p.D + d];
            acc += scores[k] * v;
        }
        Out[((b * p.H + h) * p.T + t) * p.D + d] = (half)acc;
    }
}
