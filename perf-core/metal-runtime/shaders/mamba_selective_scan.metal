#include <metal_stdlib>
using namespace metal;

// Chunked Mamba selective scan. One thread owns the recurrence so state continuity
// is exact across chunks; the inner channel loop is fused with output production.
kernel void mamba_selective_scan_f32(
    device const float* u [[buffer(0)]], device const float* dt [[buffer(1)]],
    device const float* b [[buffer(2)]], device const float* c [[buffer(3)]],
    device const float* d [[buffer(4)]], device const float* a_log [[buffer(5)]],
    device float* state [[buffer(6)]], device float* out [[buffer(7)]],
    constant uint& steps [[buffer(8)]], constant uint& state_dim [[buffer(9)]],
    uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[256];
    for (uint t = 0; t < steps; ++t) {
        const float dbu = dt[t] * b[t] * u[t];
        float local = 0.0f;
        for (uint channel = tid; channel < state_dim; channel += 256) {
            const float decay = exp(dt[t] * exp(a_log[channel]));
            state[channel] = decay * state[channel] + dbu;
            local += c[t] * state[channel];
        }
        partial[tid] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float acc = d[t] * u[t];
            for (uint i = 0; i < 256; ++i) acc += partial[i];
            out[t] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
