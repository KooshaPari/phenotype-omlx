const tq = @import("turbo_quant.zig");
const std = @import("std");

export fn tq_zig_encode(
    data_ptr: [*]const f32, n: usize, bits: u8, group_size: usize,
    out_shape: [*]*usize, out_shape_len: *usize,
    out_packed: [*]*u8,   out_packed_len: *usize,
    out_scales: [*]*f32,  out_scales_len: *usize,
    out_zeros:  [*]*f32,  out_zeros_len:  *usize,
) bool {
    return tq.tq_zig_encode(data_ptr, n, bits, group_size,
        out_shape, out_shape_len, out_packed, out_packed_len,
        out_scales, out_scales_len, out_zeros, out_zeros_len);
}

export fn tq_zig_decode(
    packed_ptr: [*]const u8,  packed_len: usize,
    scales_ptr: [*]const f32, zeros_ptr: [*]const f32,
    n: usize, group_size: usize, bits: u8,
    out_ptr: [*]f32,
) void {
    tq.tq_zig_decode(packed_ptr, packed_len, scales_ptr, zeros_ptr, n, group_size, bits, out_ptr);
}

export fn tq_zig_free(ptr: ?*anyopaque, size: usize) void {
    tq.tq_zig_free(ptr, size);
}

// Removed entrypoint main to prevent duplicate symbol linker collision with Rust test framework

