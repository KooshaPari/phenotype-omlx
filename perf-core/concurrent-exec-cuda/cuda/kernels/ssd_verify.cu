// ============================================================================
//  ssd_verify_kernel
// ----------------------------------------------------------------------------
//  Role in the perf-core pipeline
// ----------------------------------------------------------------------------
//  This kernel implements the draft-verification pass for the self-
//  speculative decoding (SSD) backend in perf-core/concurrent-exec/.
//
//  Given a draft token sequence of length gamma produced by the cheap
//  prompt-lookup path, and the target model's full logits for each step,
//  this kernel checks how many leading draft tokens are accepted:
//
//      For each draft position d in [0, gamma):
//          Read draft_token[d] and target_logits[step_d]
//          Compare against target argmax (or sampled token)
//          Increment accept_count if match
//
//  Why CUDA:
//    - gamma is small (4-16) but called once per generated token, so a single
//      device launch with one block and gamma threads is enough to keep
//      verification off the host's critical path.
//
//  Memory layout:
//    draft_tokens:    [gamma]              int32
//    target_logits:   [gamma, vocab_size]  float32, row-major
//    accepted_out:    [1]                  int32, count of leading accepts
//
//  The kernel writes only the number of accepted leading tokens; the host
//  uses that count to truncate the draft and emit the new tokens.
// ============================================================================

#include <cuda_runtime.h>

extern "C" __global__ void ssd_verify_kernel(
    const int*   __restrict__ draft_tokens,
    const float* __restrict__ target_logits,
    int gamma,
    int vocab_size,
    int* __restrict__ accepted_out
) {
    // Stub: real impl walks draft_tokens[0..gamma] and compares to
    // target_argmax(target_logits[step, :]).
    if (draft_tokens == nullptr || target_logits == nullptr || accepted_out == nullptr) {
        return;
    }
    if (gamma <= 0 || vocab_size <= 0) {
        return;
    }

    // Single-thread stub: write 0 accepted. Real impl will use a
    // warp vote to short-circuit on first rejection.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        accepted_out[0] = 0;
    }
}