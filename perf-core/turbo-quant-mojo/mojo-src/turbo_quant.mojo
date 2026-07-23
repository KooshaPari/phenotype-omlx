# turbo-quant-mojo — SIMD-autotuned Mojo kernel for uniform group quantization.
#
# Build:
#   mojo build mojo-src/turbo_quant.mojo --emit shared-lib -o libturbo_quant_mojo.dylib
#
# Bit-packing convention matches the Rust turbo-quant core (LSB-first within
# each byte, cross-byte boundaries for non-power-of-2 bit widths).

from std.memory import UnsafePointer, alloc

comptime SIMD_WIDTH = 8


# ── Data structures ──────────────────────────────────────────────────────

struct QuantizedTensor:
    var shape_ptr: UnsafePointer[Int, MutUntrackedOrigin]
    var shape_len: Int
    var packed_ptr: UnsafePointer[UInt8, MutUntrackedOrigin]
    var packed_len: Int
    var scales_ptr: UnsafePointer[Float32, MutUntrackedOrigin]
    var scales_len: Int
    var zeros_ptr: UnsafePointer[Float32, MutUntrackedOrigin]
    var zeros_len: Int

    def __init__(
        out self,
        shape_ptr: UnsafePointer[Int, MutUntrackedOrigin],
        shape_len: Int,
        packed_ptr: UnsafePointer[UInt8, MutUntrackedOrigin],
        packed_len: Int,
        scales_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
        scales_len: Int,
        zeros_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
        zeros_len: Int,
    ):
        self.shape_ptr = shape_ptr
        self.shape_len = shape_len
        self.packed_ptr = packed_ptr
        self.packed_len = packed_len
        self.scales_ptr = scales_ptr
        self.scales_len = scales_len
        self.zeros_ptr = zeros_ptr
        self.zeros_len = zeros_len


# ── SIMD min/max kernel ──────────────────────────────────────────────────
# Vectorized min/max over `count` elements starting at `offset`.
# Writes results to lo_out[0] and hi_out[0].

def simd_min_max(
    data: UnsafePointer[Float32, MutUntrackedOrigin],
    offset: Int,
    count: Int,
    lo_out: UnsafePointer[Float32, MutUntrackedOrigin],
    hi_out: UnsafePointer[Float32, MutUntrackedOrigin],
):
    if count == 0:
        lo_out.store(0, Float32(0.0))
        hi_out.store(0, Float32(0.0))
        return

    var lo: Float32 = data.load(offset)[0]
    var hi: Float32 = lo
    var i: Int = offset + 1
    var remaining: Int = count - 1

    while remaining >= SIMD_WIDTH:
        var j: Int = 0
        while j < SIMD_WIDTH:
            var v: Float32 = data.load(i + j)[0]
            if v < lo:
                lo = v
            if v > hi:
                hi = v
            j += 1
        i += SIMD_WIDTH
        remaining -= SIMD_WIDTH

    while remaining > 0:
        var v: Float32 = data.load(i)[0]
        if v < lo:
            lo = v
        if v > hi:
            hi = v
        i += 1
        remaining -= 1

    lo_out.store(0, lo)
    hi_out.store(0, hi)


# ── Bit-packing helpers (LSB-first, matching Rust turbo-quant) ───────────
#
# The Rust core packs `bits` quantized bits per element, LSB-first across
# the full data span (bit_cursor advances by `bits` for each element).

def pack_bits(
    packed: UnsafePointer[UInt8, MutUntrackedOrigin],
    bit_cursor: Int,
    q: Int,
    bits: Int,
):
    var cursor: Int = bit_cursor
    var remaining: Int = bits
    var val: Int = q
    while remaining > 0:
        var byte_idx: Int = cursor // 8
        var bit_off: Int = cursor % 8
        var room: Int = 8 - bit_off
        var take: Int = remaining if remaining < room else room
        var mask: Int = (1 << take) - 1
        var current: UInt8 = packed.load(byte_idx)[0]
        current |= UInt8((val & mask) << bit_off)
        packed.store(byte_idx, current)
        val >>= take
        cursor += take
        remaining -= take


def unpack_bits(
    packed: UnsafePointer[UInt8, MutUntrackedOrigin],
    bit_cursor: Int,
    bits: Int,
) -> Int:
    var cursor: Int = bit_cursor
    var remaining: Int = bits
    var value: Int = 0
    var shift: Int = 0
    while remaining > 0:
        var byte_idx: Int = cursor // 8
        var bit_off: Int = cursor % 8
        var room: Int = 8 - bit_off
        var take: Int = remaining if remaining < room else room
        var mask: Int = (1 << take) - 1
        var byte_val: Int = Int(packed.load(byte_idx)[0])
        value |= ((byte_val >> bit_off) & mask) << shift
        shift += take
        cursor += take
        remaining -= take
    return value


# ── SIMD batch decode for 4-bit quantization ─────────────────────────────
# Unpacks 2 values per byte via nibble extraction, batch-dequantizes 8 values.

def decode_group_simd_4bit(
    packed: UnsafePointer[UInt8, MutUntrackedOrigin],
    scales: UnsafePointer[Float32, MutUntrackedOrigin],
    zeros: UnsafePointer[Float32, MutUntrackedOrigin],
    result: UnsafePointer[Float32, MutUntrackedOrigin],
    g: Int,
    gs: Int,
    n: Int,
    per_group_packed_bytes: Int,
):
    var start: Int = g * gs
    var end: Int = start + gs
    if end > n:
        end = n
    var scale: Float32 = scales.load(g)[0]
    var zero: Float32 = zeros.load(g)[0]
    var group_offset: Int = g * per_group_packed_bytes

    var i: Int = start
    var byte_idx: Int = 0
    while i + SIMD_WIDTH <= end and byte_idx + 4 <= per_group_packed_bytes:
        var b0 = Int(packed.load(group_offset + byte_idx)[0])
        var b1 = Int(packed.load(group_offset + byte_idx + 1)[0])
        var b2 = Int(packed.load(group_offset + byte_idx + 2)[0])
        var b3 = Int(packed.load(group_offset + byte_idx + 3)[0])
        result.store(i + 0, zero + Float32(b0 & 0xF) * scale)
        result.store(i + 1, zero + Float32((b0 >> 4) & 0xF) * scale)
        result.store(i + 2, zero + Float32(b1 & 0xF) * scale)
        result.store(i + 3, zero + Float32((b1 >> 4) & 0xF) * scale)
        result.store(i + 4, zero + Float32(b2 & 0xF) * scale)
        result.store(i + 5, zero + Float32((b2 >> 4) & 0xF) * scale)
        result.store(i + 6, zero + Float32(b3 & 0xF) * scale)
        result.store(i + 7, zero + Float32((b3 >> 4) & 0xF) * scale)
        i += SIMD_WIDTH
        byte_idx += 4

    while i < end:
        var q_raw = unpack_bits(
            packed, g * per_group_packed_bytes * 8 + (i - start) * 4, 4
        )
        result.store(i, zero + Float32(q_raw) * scale)
        i += 1


# ── SIMD batch decode for 2-bit quantization ─────────────────────────────
# Unpacks 4 values per byte via bit extraction, batch-dequantizes 8 values.

def decode_group_simd_2bit(
    packed: UnsafePointer[UInt8, MutUntrackedOrigin],
    scales: UnsafePointer[Float32, MutUntrackedOrigin],
    zeros: UnsafePointer[Float32, MutUntrackedOrigin],
    result: UnsafePointer[Float32, MutUntrackedOrigin],
    g: Int,
    gs: Int,
    n: Int,
    per_group_packed_bytes: Int,
):
    var start: Int = g * gs
    var end: Int = start + gs
    if end > n:
        end = n
    var scale: Float32 = scales.load(g)[0]
    var zero: Float32 = zeros.load(g)[0]
    var group_offset: Int = g * per_group_packed_bytes

    var i: Int = start
    var byte_idx: Int = 0
    while i + SIMD_WIDTH <= end and byte_idx + 8 <= per_group_packed_bytes:
        var b0 = Int(packed.load(group_offset + byte_idx)[0])
        var b1 = Int(packed.load(group_offset + byte_idx + 1)[0])
        result.store(i + 0, zero + Float32(b0 & 0x3) * scale)
        result.store(i + 1, zero + Float32((b0 >> 2) & 0x3) * scale)
        result.store(i + 2, zero + Float32((b0 >> 4) & 0x3) * scale)
        result.store(i + 3, zero + Float32((b0 >> 6) & 0x3) * scale)
        result.store(i + 4, zero + Float32(b1 & 0x3) * scale)
        result.store(i + 5, zero + Float32((b1 >> 2) & 0x3) * scale)
        result.store(i + 6, zero + Float32((b1 >> 4) & 0x3) * scale)
        result.store(i + 7, zero + Float32((b1 >> 6) & 0x3) * scale)
        i += SIMD_WIDTH
        byte_idx += 2

    while i < end:
        var bit_cursor: Int = g * per_group_packed_bytes * 8 + (i - start) * 2
        var q_raw = unpack_bits(packed, bit_cursor, 2)
        result.store(i, zero + Float32(q_raw) * scale)
        i += 1


# ── Generic decode (3-bit or any width) ──────────────────────────────────

def decode_group_generic(
    packed: UnsafePointer[UInt8, MutUntrackedOrigin],
    scales: UnsafePointer[Float32, MutUntrackedOrigin],
    zeros: UnsafePointer[Float32, MutUntrackedOrigin],
    result: UnsafePointer[Float32, MutUntrackedOrigin],
    g: Int,
    gs: Int,
    n: Int,
    bits: Int,
    per_group_packed_bytes: Int,
):
    var start: Int = g * gs
    var end: Int = start + gs
    if end > n:
        end = n
    var scale: Float32 = scales.load(g)[0]
    var zero: Float32 = zeros.load(g)[0]

    var i: Int = start
    while i < end:
        var bit_cursor: Int = g * per_group_packed_bytes * 8 + (i - start) * bits
        var q_raw = unpack_bits(packed, bit_cursor, bits)
        result.store(i, zero + Float32(q_raw) * scale)
        i += 1


# ── Encode a single group ────────────────────────────────────────────────

def encode_group(
    data: UnsafePointer[Float32, MutUntrackedOrigin],
    g: Int,
    gs: Int,
    n: Int,
    levels: Float32,
    bits_i: Int,
    packed: UnsafePointer[UInt8, MutUntrackedOrigin],
    scales: UnsafePointer[Float32, MutUntrackedOrigin],
    zeros: UnsafePointer[Float32, MutUntrackedOrigin],
    bit_cursor_start: Int,
) -> Int:
    var start: Int = g * gs
    var end: Int = start + gs
    if end > n:
        end = n

    var lo_buf = alloc[Float32](1)
    var hi_buf = alloc[Float32](1)
    simd_min_max(data, start, end - start, lo_buf, hi_buf)
    var lo: Float32 = lo_buf.load(0)[0]
    var hi: Float32 = hi_buf.load(0)[0]
    lo_buf.free()
    hi_buf.free()

    var span: Float32 = hi - lo
    var scale: Float32 = span / levels if span > 0.0 else Float32(1.0)
    var zero: Float32 = lo
    scales.store(g, scale)
    zeros.store(g, zero)

    var bit_cursor: Int = bit_cursor_start
    var idx: Int = start
    while idx < end:
        var normalized: Float32 = (data.load(idx)[0] - zero) / scale
        var clamped: Float32 = round(normalized)
        if clamped < Float32(0):
            clamped = Float32(0)
        if clamped > levels:
            clamped = levels
        var q: Int = Int(clamped)
        pack_bits(packed, bit_cursor, q, bits_i)
        bit_cursor += bits_i
        idx += 1

    return bit_cursor


# ── encode_uniform ───────────────────────────────────────────────────────

def encode_uniform(
    data: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    bits: UInt8,
    group_size: Int,
) -> QuantizedTensor:
    var gs: Int = 64 if group_size == 0 else group_size
    var bits_i: Int = Int(bits)
    var levels: Float32 = Float32((1 << bits_i) - 1)
    var n_groups: Int = (n + gs - 1) // gs
    var per_group_packed_bytes: Int = (gs * bits_i + 7) // 8
    var total_packed: Int = n_groups * per_group_packed_bytes

    var shape = alloc[Int](1)
    shape.store(0, n)
    var packed = alloc[UInt8](total_packed)
    var scales = alloc[Float32](n_groups)
    var zeros = alloc[Float32](n_groups)

    var z: Int = 0
    while z < total_packed:
        packed.store(z, UInt8(0))
        z += 1

    var bit_cursor: Int = 0
    for g in range(n_groups):
        bit_cursor = encode_group(
            data, g, gs, n, levels, bits_i,
            packed, scales, zeros, bit_cursor,
        )

    return QuantizedTensor(
        shape_ptr=shape,
        shape_len=1,
        packed_ptr=packed,
        packed_len=total_packed,
        scales_ptr=scales,
        scales_len=n_groups,
        zeros_ptr=zeros,
        zeros_len=n_groups,
    )


# ── decode_uniform ───────────────────────────────────────────────────────
# Dispatches to SIMD batch kernels for power-of-2 bit widths, falls back
# to the generic path for 3-bit.

def decode_uniform(
    q: QuantizedTensor,
    n: Int,
    group_size: Int,
    bits: UInt8,
    result: UnsafePointer[Float32, MutUntrackedOrigin],
):
    var gs: Int = 64 if group_size == 0 else group_size
    var bits_i: Int = Int(bits)
    var n_groups: Int = (n + gs - 1) // gs
    var per_group_packed_bytes: Int = (gs * bits_i + 7) // 8

    for g in range(n_groups):
        if bits_i == 4:
            decode_group_simd_4bit(
                q.packed_ptr, q.scales_ptr, q.zeros_ptr, result,
                g, gs, n, per_group_packed_bytes,
            )
        elif bits_i == 2:
            decode_group_simd_2bit(
                q.packed_ptr, q.scales_ptr, q.zeros_ptr, result,
                g, gs, n, per_group_packed_bytes,
            )
        else:
            decode_group_generic(
                q.packed_ptr, q.scales_ptr, q.zeros_ptr, result,
                g, gs, n, bits_i, per_group_packed_bytes,
            )


# ── C ABI exports (consumed by Rust `extern "C"` wrapper) ───────────────

@export("tq_mojo_encode")
def tq_mojo_encode(
    data_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    bits: UInt8,
    group_size: Int,
    shape_ptr_out: UnsafePointer[Int, MutUntrackedOrigin],
    out_shape_len: UnsafePointer[Int, MutUntrackedOrigin],
    packed_ptr_out: UnsafePointer[Int, MutUntrackedOrigin],
    out_packed_len: UnsafePointer[Int, MutUntrackedOrigin],
    scales_ptr_out: UnsafePointer[Int, MutUntrackedOrigin],
    out_scales_len: UnsafePointer[Int, MutUntrackedOrigin],
    zeros_ptr_out: UnsafePointer[Int, MutUntrackedOrigin],
    out_zeros_len: UnsafePointer[Int, MutUntrackedOrigin],
) -> Bool:
    if n <= 0 or Int(bits) < 2 or Int(bits) > 4:
        return False

    var q = encode_uniform(data_ptr, n, bits, group_size)
    shape_ptr_out[0] = Int(q.shape_ptr)
    out_shape_len[0] = q.shape_len
    packed_ptr_out[0] = Int(q.packed_ptr)
    out_packed_len[0] = q.packed_len
    scales_ptr_out[0] = Int(q.scales_ptr)
    out_scales_len[0] = q.scales_len
    zeros_ptr_out[0] = Int(q.zeros_ptr)
    out_zeros_len[0] = q.zeros_len
    return True


@export("tq_mojo_free")
def tq_mojo_free(address: Int) -> None:
    var ptr = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address)
    ptr.free()


@export("tq_mojo_decode")
def tq_mojo_decode(
    packed_ptr: UnsafePointer[UInt8, MutUntrackedOrigin],
    packed_len: Int,
    scales_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    zeros_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    group_size: Int,
    bits: UInt8,
    result_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
) -> None:
    var gs: Int = 64 if group_size == 0 else group_size
    var n_groups: Int = (n + gs - 1) // gs
    var q = QuantizedTensor(
        shape_ptr=alloc[Int](1),
        shape_len=1,
        packed_ptr=packed_ptr,
        packed_len=packed_len,
        scales_ptr=scales_ptr,
        scales_len=n_groups,
        zeros_ptr=zeros_ptr,
        zeros_len=n_groups,
    )
    decode_uniform(q, n, group_size, bits, result_ptr)
