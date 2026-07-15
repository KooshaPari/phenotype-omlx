// phenotype-omlx — turbo-quant in Zig
//
// Pure-Zig implementation of uniform TurboQuant encode/decode. Identical
// algorithm to perf-core/turbo-quant (Rust) so output is bit-exact
// compatible; the Rust crate at perf-core/turbo-quant-zig/src/lib.rs
// re-exports this through `extern "C"` for the Python perf-core to call.

const std = @import("std");

/// QuantizedTensor — the same shape as turbo_quant::QuantizedTensor in Rust.
pub const QuantizedTensor = struct {
    shape: []usize,
    packed: []u8,
    scales: []f32,
    zeros: []f32,
};

/// Lloyd-Max-like uniform quantizer on `bits` levels for the slice `data`,
/// group-scaled. Group size defaults to 64; pass `group_size=0` to use the
/// default. Returns a freshly-allocated QuantizedTensor in `arena`.
///
/// Caller frees `result.packed`, `result.scales`, `result.zeros` with `arena.free`.
pub fn encode_uniform(
    arena: std.mem.Allocator,
    data: []const f32,
    bits: u8,
    group_size: usize,
) !QuantizedTensor {
    const gs: usize = if (group_size == 0) 64 else group_size;
    const levels: u16 = (@as(u16, 1) << @intCast(bits)) - 1; // 4-bit → 15 levels
    const n = data.len;
    const n_groups = (n + gs - 1) / gs;
    const per_group_packed_bytes = (gs * bits + 7) / 8;

    const shape = try arena.alloc(usize, 1);
    shape[0] = n;
    const packed = try arena.alloc(u8, n_groups * per_group_packed_bytes);
    const scales = try arena.alloc(f32, n_groups);
    const zeros = try arena.alloc(f32, n_groups);
    @memset(packed, 0);
    @memset(scales, 0);
    @memset(zeros, 0);

    var g: usize = 0;
    while (g < n_groups) : (g += 1) {
        const start = g * gs;
        const end = @min(start + gs, n);

        // Per-group min/max
        var lo: f32 = data[start];
        var hi: f32 = data[start];
        var i: usize = start + 1;
        while (i < end) : (i += 1) {
            if (data[i] < lo) lo = data[i];
            if (data[i] > hi) hi = data[i];
        }
        const range = hi - lo;
        const scale: f32 = if (range > 0) range / @as(f32, @floatFromInt(levels)) else 1.0;
        const zero: f32 = lo;
        scales[g] = scale;
        zeros[g] = zero;

        // Quantize each element to `bits` bits, pack into bytes.
        var bit_pos: usize = 0;
        i = start;
        while (i < end) : (i += 1) {
            const normalized = (data[i] - zero) / scale;
            const clamped = @max(0.0, @min(@as(f32, @floatFromInt(levels)), @round(normalized)));
            const q: u16 = @intFromFloat(clamped);
            pack_bits(packed[g * per_group_packed_bytes ..][0..per_group_packed_bytes], bit_pos, bits, q);
            bit_pos += bits;
        }
    }

    return QuantizedTensor{ .shape = shape, .packed = packed, .scales = scales, .zeros = zeros };
}

/// Inverse of `encode_uniform` — writes the reconstructed f32 values into `out`.
pub fn decode_uniform(q: QuantizedTensor, out: []f32, group_size: usize, bits: u8) void {
    const gs: usize = if (group_size == 0) 64 else group_size;
    const levels: f32 = @floatFromInt((@as(u16, 1) << @intCast(bits)) - 1);
    const per_group_packed_bytes = (gs * bits + 7) / 8;

    var g: usize = 0;
    while (g * gs < out.len) : (g += 1) {
        const start = g * gs;
        const end = @min(start + gs, out.len);
        const scale = q.scales[g];
        const zero = q.zeros[g];
        var bit_pos: usize = 0;
        var i: usize = start;
        while (i < end) : (i += 1) {
            const qv = unpack_bits(q.packed[g * per_group_packed_bytes ..][0..per_group_packed_bytes], bit_pos, bits);
            out[i] = zero + @as(f32, @floatFromInt(qv)) * scale;
            bit_pos += bits;
        }
    }
}

fn pack_bits(dst: []u8, bit_pos: usize, bits: u8, value: u16) void {
    var v = value;
    var bp = bit_pos;
    var b: u8 = 0;
    while (b < bits) : (b += 1) {
        const byte_idx = bp / 8;
        const bit_idx: u3 = @intCast(bp % 8);
        const bit = @as(u8, @intCast((v >> @intCast(bits - 1 - b)) & 1));
        dst[byte_idx] |= bit << @intCast(7 - bit_idx);
        bp += 1;
    }
}

fn unpack_bits(src: []u8, bit_pos: usize, bits: u8) u16 {
    var value: u16 = 0;
    var bp = bit_pos;
    var b: u8 = 0;
    while (b < bits) : (b += 1) {
        const byte_idx = bp / 8;
        const bit_idx: u3 = @intCast(bp % 8);
        const bit: u16 = @as(u16, @intCast((src[byte_idx] >> @intCast(7 - bit_idx)) & 1));
        value = (value << 1) | bit;
        bp += 1;
    }
    return value;
}

// ── C ABI exports (consumed by Rust `extern "C"` wrapper) ──────────────

export fn tq_zig_encode(
    data_ptr: [*]const f32,
    n: usize,
    bits: u8,
    group_size: usize,
    out_shape: [*]*usize,
    out_shape_len: *usize,
    out_packed: [*]*u8,
    out_packed_len: *usize,
    out_scales: [*]*f32,
    out_scales_len: *usize,
    out_zeros: [*]*f32,
    out_zeros_len: *usize,
) bool {
    const data = data_ptr[0..n];
    var arena = std.heap.page_allocator;
    const q = encode_uniform(arena, data, bits, group_size) catch return false;

    out_shape.* = q.shape.ptr;
    out_shape_len.* = q.shape.len;
    out_packed.* = q.packed.ptr;
    out_packed_len.* = q.packed.len;
    out_scales.* = q.scales.ptr;
    out_scales_len.* = q.scales.len;
    out_zeros.* = q.zeros.ptr;
    out_zeros_len.* = q.zeros.len;
    return true;
}

export fn tq_zig_decode(
    packed_ptr: [*]const u8,
    packed_len: usize,
    scales_ptr: [*]const f32,
    zeros_ptr: [*]const f32,
    n: usize,
    group_size: usize,
    bits: u8,
    out_ptr: [*]f32,
) void {
    const packed = packed_ptr[0..packed_len];
    const scales = scales_ptr[0..((n + group_size - 1) / group_size)];
    const zeros = zeros_ptr[0..scales.len];
    const out = out_ptr[0..n];
    const q = QuantizedTensor{
        .shape = &[_]usize{n},
        .packed = @constCast(packed),
        .scales = @constCast(scales),
        .zeros = @constCast(zeros),
    };
    decode_uniform(q, out, group_size, bits);
}

// Entry point for `zig build run`
pub fn main() !void {
    const data = [_]f32{ 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8 };
    var arena = std.heap.page_allocator;
    const q = try encode_uniform(arena, &data, 4, 0);
    std.debug.print("encoded: packed={d} bytes, scales={d}\n", .{ q.packed.len, q.scales.len });

    var out: [data.len]f32 = undefined;
    decode_uniform(q, &out, 0, 4);
    var max_err: f32 = 0;
    for (data, out) |a, b| {
        const err = @abs(a - b);
        if (err > max_err) max_err = err;
    }
    std.debug.print("decoded: max_err={d}\n", .{max_err});
}
