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
    packed_data: []u8,
    scales: []f32,
    zeros: []f32,
};

/// Lloyd-Max-like uniform quantizer on `bits` levels for the slice `data`,
/// group-scaled. Group size defaults to 64; pass `group_size=0` to use the
/// default. Returns a freshly-allocated QuantizedTensor in `arena`.
///
/// Caller frees `result.packed_data`, `result.scales`, `result.zeros` with `arena.free`.
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
    const packed_data = try arena.alloc(u8, n_groups * per_group_packed_bytes);
    const scales = try arena.alloc(f32, n_groups);
    const zeros = try arena.alloc(f32, n_groups);
    @memset(packed_data, 0);
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
            pack_bits(packed_data[g * per_group_packed_bytes ..][0..per_group_packed_bytes], bit_pos, bits, q);
            bit_pos += bits;
        }
    }

    return QuantizedTensor{ .shape = shape, .packed_data = packed_data, .scales = scales, .zeros = zeros };
}

/// Inverse of `encode_uniform` — writes the reconstructed f32 values into `out`.
pub fn decode_uniform(q: QuantizedTensor, out: []f32, group_size: usize, bits: u8) void {
    const gs: usize = if (group_size == 0) 64 else group_size;
    // const levels: f32 = @floatFromInt((@as(u16, 1) << @intCast(bits)) - 1);
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
            const qv = unpack_bits(q.packed_data[g * per_group_packed_bytes ..][0..per_group_packed_bytes], bit_pos, bits);
            out[i] = zero + @as(f32, @floatFromInt(qv)) * scale;
            bit_pos += bits;
        }
    }
}

fn pack_bits(dst: []u8, bit_pos: usize, bits: u8, value: u16) void {
    const v = value;
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

// ── Native ABI v1 ───────────────────────────────────────────────────────
//
// Mirror of the versioned C ABI in `perf-core/native-abi/include/abi_v1.h`.
// Zig exports the same symbols (`tq_abi_encode`, `tq_abi_decode`,
// `tq_abi_release`) so polyglot callers can dispatch to either backend
// behind one contract. Caller-owned buffers, identical layout, identical
// status codes. See the Rust reference implementation in
// `perf-core/native-abi/src/dispatch.rs` for the canonical semantics.

const TQ_ABI_VERSION_MAJOR: u16 = 1;
const TQ_ABI_BITS_MIN: u8 = 2;
const TQ_ABI_BITS_MAX: u8 = 4;

const TqAbiStatus = enum(c_int) {
    Ok = 0,
    ErrNullArg = 1,
    ErrInvalidBits = 2,
    ErrInvalidGroupSize = 3,
    ErrNonFiniteInput = 4,
    ErrOverflow = 5,
    ErrAllocation = 6,
    ErrVersionMismatch = 7,
    ErrBackend = 8,
};

const TqAbiVersion = extern struct {
    major: u16,
    minor: u16,
};

pub const TqAbiEncodeRequest = extern struct {
    abi: TqAbiVersion,
    data_ptr: ?[*]const f32,
    n: usize,
    bits: u8,
    group_size: usize,
    out_shape: [*]?[*]usize,
    out_shape_capacity: usize,
    out_packed: [*]?[*]u8,
    out_packed_capacity: usize,
    out_scales: [*]?[*]f32,
    out_scales_capacity: usize,
    out_zeros: [*]?[*]f32,
    out_zeros_capacity: usize,
};

pub const TqAbiDecodeRequest = extern struct {
    abi: TqAbiVersion,
    packed_ptr: ?[*]const u8,
    packed_len: usize,
    scales_ptr: ?[*]const f32,
    zeros_ptr: ?[*]const f32,
    n: usize,
    group_size: usize,
    bits: u8,
    out_ptr: ?[*]f32,
};

pub const TqAbiEncodeResult = extern struct {
    status: c_int,
    written_packed_len: usize,
    written_shape_len: usize,
    written_scales_len: usize,
    written_zeros_len: usize,
};

const TqAbiReleaseKind = enum(c_int) {
    Shape = 0,
    Packed = 1,
    Scales = 2,
    Zeros = 3,
};

fn bits_valid(bits: u8) bool {
    return bits >= TQ_ABI_BITS_MIN and bits <= TQ_ABI_BITS_MAX;
}

/// Mirrors the C ABI: validates the request, returns the matching status on
/// failure, otherwise returns packed_len and n_groups via the out-params.
fn validate_encode(req: *const TqAbiEncodeRequest, out_packed_len: *usize, out_n_groups: *usize) TqAbiStatus {
    if (req.abi.major != TQ_ABI_VERSION_MAJOR) return .ErrVersionMismatch;
    if (req.n == 0) return .ErrNullArg;
    if (!bits_valid(req.bits)) return .ErrInvalidBits;
    if (req.group_size == 0) return .ErrInvalidGroupSize;
    if (req.out_shape_capacity == 0 or
        req.out_packed_capacity == 0 or
        req.out_scales_capacity == 0 or
        req.out_zeros_capacity == 0)
    {
        return .ErrNullArg;
    }
    if (req.n > std.math.maxInt(usize) - req.group_size) return .ErrOverflow;
    if (req.n > std.math.maxInt(usize) / @as(usize, @intCast(req.bits))) return .ErrOverflow;

    const packed_len: usize = (req.n * @as(usize, @intCast(req.bits)) + 7) / 8;
    const n_groups: usize = (req.n + req.group_size - 1) / req.group_size;

    if (req.out_packed_capacity < packed_len or
        req.out_shape_capacity < 1 or
        req.out_scales_capacity < n_groups or
        req.out_zeros_capacity < n_groups)
    {
        return .ErrOverflow;
    }

    const data = req.data_ptr.?;
    var i: usize = 0;
    while (i < req.n) : (i += 1) {
        if (!std.math.isFinite(data[i])) return .ErrNonFiniteInput;
    }

    out_packed_len.* = packed_len;
    out_n_groups.* = n_groups;
    return .Ok;
}

/// Native ABI v1 entry. Returns the encode result.
pub fn tq_abi_encode(req: *const TqAbiEncodeRequest) TqAbiEncodeResult {
    var res = TqAbiEncodeResult{
        .status = @intFromEnum(TqAbiStatus.ErrNullArg),
        .written_packed_len = 0,
        .written_shape_len = 0,
        .written_scales_len = 0,
        .written_zeros_len = 0,
    };

    // Cast away const to write into caller-owned output slots.
    const mreq: *TqAbiEncodeRequest = @constCast(req);

    var packed_len: usize = 0;
    var n_groups: usize = 0;
    const vstatus = validate_encode(req, &packed_len, &n_groups);
    if (vstatus != .Ok) {
        res.status = @intFromEnum(vstatus);
        return res;
    }

    const data = req.data_ptr.?;
    const levels_f: f32 = @as(f32, @floatFromInt((@as(u32, 1) << @intCast(req.bits)) - 1));
    @as([*]usize, @ptrCast(mreq.out_shape[0]))[0] = req.n;

    var g: usize = 0;
    while (g < n_groups) : (g += 1) {
        const start = g * req.group_size;
        const end = @min(start + req.group_size, req.n);

        var lo: f32 = data[start];
        var hi: f32 = data[start];
        var i: usize = start + 1;
        while (i < end) : (i += 1) {
            if (data[i] < lo) lo = data[i];
            if (data[i] > hi) hi = data[i];
        }
        const span = hi - lo;
        var scale: f32 = span / levels_f;
        if (!(scale > 0.0)) scale = 1e-30;

        @as([*]f32, @ptrCast(mreq.out_scales[0]))[g] = scale;
        @as([*]f32, @ptrCast(mreq.out_zeros[0]))[g] = lo;

        // The ABI uses one contiguous bitstream across groups, matching the
        // C/Rust native-abi implementations. Start at this group's global
        // element offset rather than resetting to byte zero per group.
        var bit_off: usize = g * req.group_size * @as(usize, @intCast(req.bits));
        i = start;
        while (i < end) : (i += 1) {
            const qf = (data[i] - lo) / scale;
            const clamped = @max(0.0, @min(levels_f, @round(qf)));
            const q: u32 = @intFromFloat(clamped);
            const slot = @as([*]u8, @ptrCast(mreq.out_packed[0]));
            write_bits_zig(slot, bit_off, @intCast(q), req.bits);
            bit_off += req.bits;
        }
    }

    res.status = @intFromEnum(TqAbiStatus.Ok);
    res.written_packed_len = packed_len;
    res.written_shape_len = 1;
    res.written_scales_len = n_groups;
    res.written_zeros_len = n_groups;
    return res;
}

/// Bit-packed write used by `tq_abi_encode`. LSB-first within each byte,
/// byte-straddles when the value doesn't fit. Mirrors the C `tq_write_bits`.
fn write_bits_zig(buf: [*]u8, bit_offset: usize, value: u32, bits: u8) void {
    const mask: u8 = @intCast((@as(u32, 1) << @intCast(bits)) - 1);
    const v: u8 = @intCast(value & @as(u32, mask));
    const byte_idx = bit_offset >> 3;
    const bit_in_byte: u8 = @intCast(bit_offset & 7);
    const room: u8 = 8 - bit_in_byte;

    if (room >= bits) {
        const shift_mask: u8 = @as(u8, @intCast(mask << @intCast(bit_in_byte)));
        buf[byte_idx] = (buf[byte_idx] & ~shift_mask) | (@as(u8, @intCast(v << @intCast(bit_in_byte))));
    } else {
        const lo_mask: u8 = @intCast((@as(u32, 1) << @intCast(room)) - 1);
        buf[byte_idx] = (buf[byte_idx] & ~(@as(u8, @intCast(lo_mask << @intCast(bit_in_byte))))) |
            (@as(u8, @intCast((v & lo_mask) << @intCast(bit_in_byte))));
        const hi_bits: u8 = bits - room;
        const hi_mask: u8 = @intCast((@as(u32, 1) << @intCast(hi_bits)) - 1);
        buf[byte_idx + 1] = (buf[byte_idx + 1] & ~hi_mask) |
            (@as(u8, @intCast((v >> @intCast(room)) & hi_mask)));
    }
}

pub fn tq_abi_decode(req: *const TqAbiDecodeRequest) c_int {
    if (req.abi.major != TQ_ABI_VERSION_MAJOR) return @intFromEnum(TqAbiStatus.ErrVersionMismatch);
    if (req.n == 0) return @intFromEnum(TqAbiStatus.ErrNullArg);
    if (!bits_valid(req.bits)) return @intFromEnum(TqAbiStatus.ErrInvalidBits);
    if (req.group_size == 0) return @intFromEnum(TqAbiStatus.ErrInvalidGroupSize);
    if (req.n > std.math.maxInt(usize) - req.group_size) return @intFromEnum(TqAbiStatus.ErrOverflow);
    if (req.n > std.math.maxInt(usize) / @as(usize, @intCast(req.bits))) return @intFromEnum(TqAbiStatus.ErrOverflow);

    const expected: usize = (req.n * @as(usize, @intCast(req.bits)) + 7) / 8;
    if (req.packed_len != expected) return @intFromEnum(TqAbiStatus.ErrInvalidBits);

    const n_groups: usize = (req.n + req.group_size - 1) / req.group_size;
    const out: [*]f32 = req.out_ptr.?;
    const packed_ptr = req.packed_ptr.?;
    const scales_ptr = req.scales_ptr.?;
    const zeros_ptr = req.zeros_ptr.?;

    var g: usize = 0;
    while (g < n_groups) : (g += 1) {
        const scale = scales_ptr[g];
        const zero = zeros_ptr[g];
        const start = g * req.group_size;
        const end = @min(start + req.group_size, req.n);

        // Decode from the same contiguous stream offset used by encode.
        var bit_off: usize = g * req.group_size * @as(usize, @intCast(req.bits));
        var i: usize = start;
        while (i < end) : (i += 1) {
            const q = read_bits_zig(packed_ptr, bit_off, req.bits);
            out[i] = zero + @as(f32, @floatFromInt(q)) * scale;
            bit_off += req.bits;
        }
    }
    return @intFromEnum(TqAbiStatus.Ok);
}

fn read_bits_zig(buf: [*]const u8, bit_offset: usize, bits: u8) u32 {
    const mask: u32 = (@as(u32, 1) << @intCast(bits)) - 1;
    const byte_idx = bit_offset >> 3;
    const bit_in_byte: u8 = @intCast(bit_offset & 7);
    const room: u8 = 8 - bit_in_byte;

    if (room >= bits) {
        return @as(u32, buf[byte_idx] >> @intCast(bit_in_byte)) & mask;
    }
    const lo_mask: u8 = @intCast((@as(u32, 1) << @intCast(room)) - 1);
    const lo: u32 = @as(u32, (buf[byte_idx] >> @intCast(bit_in_byte)) & lo_mask);
    const hi_bits: u8 = bits - room;
    const hi_mask: u8 = @intCast((@as(u32, 1) << @intCast(hi_bits)) - 1);
    const hi: u32 = @as(u32, buf[byte_idx + 1] & hi_mask);
    return lo | (hi << @intCast(room));
}

pub fn tq_abi_release(kind: c_int, ptr: ?*anyopaque, count: usize) void {
    if (ptr == null or count == 0) return;
    const allocator = std.heap.c_allocator;
    const k: TqAbiReleaseKind = @enumFromInt(kind);
    switch (k) {
        .Shape, .Packed, .Scales, .Zeros => {
            const slice = @as([*]u8, @ptrCast(ptr.?))[0..count];
            allocator.free(slice);
        },
    }
}

// ── Native ABI v1 result / status aliased types exposed to Rust ────────
//
// These are here purely so the Rust wrapper can `extern "C"` them; they
// match `abi_v1.h` byte-for-byte.

pub fn tq_abi_status_to_int(s: c_int) c_int {
    return s;
}

pub fn tq_zig_encode(
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
    const allocator = std.heap.c_allocator;
    const q = encode_uniform(allocator, data, bits, group_size) catch return false;

    out_shape[0] = @ptrCast(q.shape.ptr);
    out_shape_len.* = q.shape.len;
    out_packed[0] = @as(*u8, @ptrCast(q.packed_data.ptr));
    out_packed_len.* = q.packed_data.len;
    out_scales[0] = @as(*f32, @ptrCast(q.scales.ptr));
    out_scales_len.* = q.scales.len;
    out_zeros[0] = @as(*f32, @ptrCast(q.zeros.ptr));
    out_zeros_len.* = q.zeros.len;
    return true;
}

pub fn tq_zig_decode(
    packed_ptr: [*]const u8,
    packed_len: usize,
    scales_ptr: [*]const f32,
    zeros_ptr: [*]const f32,
    n: usize,
    group_size: usize,
    bits: u8,
    out_ptr: [*]f32,
) void {
    if (n == 0) return;
    const gs = if (group_size == 0) 64 else group_size;
    const num_groups = (n + gs - 1) / gs;

    const packed_data = if (packed_len == 0) &[_]u8{} else packed_ptr[0..packed_len];
    const scales = if (num_groups == 0) &[_]f32{} else scales_ptr[0..num_groups];
    const zeros = if (num_groups == 0) &[_]f32{} else zeros_ptr[0..num_groups];
    const out = out_ptr[0..n];
    const q = QuantizedTensor{
        .shape = @constCast(&[_]usize{n})[0..],
        .packed_data = @constCast(packed_data),
        .scales = @constCast(scales),
        .zeros = @constCast(zeros),
    };
    decode_uniform(q, out, group_size, bits);
}

pub fn tq_zig_free(ptr: ?*anyopaque, size: usize) void {
    if (ptr) |p| {
        // Use std.heap.c_allocator so std.os.raw/C-ABI compatible free can be safely done if needed,
        // or let's free via c_allocator.
        const allocator = std.heap.c_allocator;
        const slice = @as([*]u8, @ptrCast(p))[0..size];
        allocator.free(slice);
    }
}

// Optional entry point for standalone testing
pub fn tq_main() !void {
    const data = [_]f32{ 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8 };
    const arena = std.heap.page_allocator;
    const q = try encode_uniform(arena, &data, 4, 0);
    std.debug.print("encoded: packed_data={d} bytes, scales={d}\n", .{ q.packed_data.len, q.scales.len });

    var out: [data.len]f32 = undefined;
    decode_uniform(q, &out, 0, 4);
    var max_err: f32 = 0;
    for (data, out) |a, b| {
        const err = @abs(a - b);
        if (err > max_err) max_err = err;
    }
    std.debug.print("decoded: max_err={d}\n", .{max_err});
}
