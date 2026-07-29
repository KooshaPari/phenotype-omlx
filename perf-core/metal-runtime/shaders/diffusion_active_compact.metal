#include <metal_stdlib>
using namespace metal;

// Compact unresolved token values and their original sequence positions.
// The caller clears `count` before dispatch and reads the first `count`
// entries after completion. Ordering is not used for correctness; the Rust
// oracle preserves ascending order for deterministic CPU/reference paths.
kernel void diffusion_active_compact_u32(
    device const uint* values [[buffer(0)]], device const uchar* active [[buffer(1)]],
    device uint* compacted [[buffer(2)]], device uint* positions [[buffer(3)]],
    device atomic_uint* count [[buffer(4)]], constant uint& tokens [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= tokens || active[gid] == 0) return;
    const uint slot = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    compacted[slot] = values[gid];
    positions[slot] = gid;
}
