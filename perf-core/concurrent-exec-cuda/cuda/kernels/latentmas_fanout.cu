// ============================================================================
//  latentmas_fanout_kernel
// ----------------------------------------------------------------------------
//  Role in the perf-core pipeline
// ----------------------------------------------------------------------------
//  This kernel implements the per-agent top-k fan-out for the LatentMAS
//  strategy in perf-core/concurrent-exec/. While the Rust side owns the
//  orchestration (which agents run, how their outputs merge, scheduling
//  strategy), this kernel owns the math:
//
//      For each agent i in [0, batch_size):
//          Read logits[i, 0..vocab_size] (contiguous row-major)
//          Compute top-k indices + values along axis 1
//          Write to out_topk[i, 0..k] (row i holds the i-th agent's top-k)
//
//  Why CUDA:
//    - The fan-out is embarrassingly parallel: one CUDA block per agent,
//      threads inside the block cooperate on a single row's top-k.
//    - For typical agent counts (8-32) and vocab sizes (32k-128k) this turns
//      what was a CPU-side scan into a single device launch.
//
//  Memory layout:
//    logits:     [batch_size, vocab_size]   row-major, float32
//    out_topk:   [batch_size, k * 2]       values then indices, float32/int32
//
//  The host side allocates these tensors with cudaMalloc and passes the raw
//  device pointers into the launch. See loader::resolve in src/lib.rs for
//  the dlopen / dlsym wiring that calls into this kernel.
//
//  TODO: replace the stub body with a real per-block top-k (e.g. bitonic
//  sort within shared memory, then a final warp-level reduction).
// ============================================================================

#include <cuda_runtime.h>

extern "C" __global__ void latentmas_fanout_kernel(
    const float* __restrict__ logits,
    int batch_size,
    int vocab_size,
    float* __restrict__ out_topk_values,
    int* __restrict__ out_topk_indices,
    int k
) {
    // Stub: real implementation will use shared-memory bitonic sort.
    // Each block = one agent (one row of `logits`).
    int agent = blockIdx.x;
    if (agent >= batch_size) return;

    // Touch inputs so nvcc doesn't strip them during the stub phase.
    if (logits == nullptr || out_topk_values == nullptr || out_topk_indices == nullptr) {
        return;
    }
    if (vocab_size <= 0 || k <= 0) {
        return;
    }

    // First thread of each block writes a placeholder so the buffer is
    // observably touched by the device. Replace with top-k.
    if (threadIdx.x == 0) {
        out_topk_values[agent * k + 0] = logits[agent * vocab_size + 0];
        out_topk_indices[agent * k + 0] = 0;
    }
}