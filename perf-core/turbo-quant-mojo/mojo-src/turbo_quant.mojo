# turbo-quant-mojo — Mojo implementation of TurboQuant.
#
# Build:
#   mojo build mojo-src/turbo_quant.mojo -o libturbo_quant_mojo.a
#
# Without Mojo installed, this crate compiles as a no-op stub
# (matching the "no language forbidden" policy).

from std.memory import alloc, UnsafePointer


def _bit_count[
    origin: MutOrigin
](
    data: UnsafePointer[Float32, origin], start: Int, end: Int
) -> Tuple[Float32, Float32]:
    var lo: Float32 = data[start]
    var hi: Float32 = data[start]
    for i in range(start + 1, end):
        var v: Float32 = data[i]
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    return (lo, hi)


def _write_bits[
    origin: MutOrigin
](
    packed: UnsafePointer[UInt8, origin],
    group_byte_offset: Int,
    value: Int,
    bits: Int,
    bit_pos: Int,
) -> Int:
    var bp: Int = bit_pos
    for b in range(bits):
        var byte_idx: Int = bp // 8
        var bit_idx: Int = bp % 8
        var bit: UInt8 = UInt8((value >> (bits - 1 - b)) & 1)
        var current: UInt8 = packed[group_byte_offset + byte_idx]
        current |= bit << UInt8(7 - bit_idx)
        packed[group_byte_offset + byte_idx] = current
        bp += 1
    return bp


def _read_bits[
    origin: MutOrigin
](
    packed: UnsafePointer[UInt8, origin],
    group_byte_offset: Int,
    bits: Int,
    bit_pos: Int,
) -> Int:
    var value: Int = 0
    var bp: Int = bit_pos
    for b in range(bits):
        var byte_idx: Int = bp // 8
        var bit_idx: Int = bp % 8
        var bit: Int = Int((packed[group_byte_offset + byte_idx] >> UInt8(7 - bit_idx)) & UInt8(1))
        value = (value << 1) | bit
        bp += 1
    return value


def encode_uniform[
    d_origin: MutOrigin,
    p_origin: MutOrigin,
    s_origin: MutOrigin,
    z_origin: MutOrigin,
    l_origin: MutOrigin,
](
    data: UnsafePointer[Float32, d_origin],
    n: Int,
    bits: UInt8,
    group_size: Int,
    out_packed: UnsafePointer[UInt8, p_origin],
    out_scales: UnsafePointer[Float32, s_origin],
    out_zeros: UnsafePointer[Float32, z_origin],
    out_packed_len: UnsafePointer[Int, l_origin],
):
    var gs: Int = 64 if group_size == 0 else group_size
    var bits_i: Int = Int(bits)
    var levels: Int = (1 << bits_i) - 1
    var n_groups: Int = (n + gs - 1) // gs
    var per_group_packed_bytes: Int = (gs * bits_i + 7) // 8
    var total_packed: Int = n_groups * per_group_packed_bytes

    out_packed_len[0] = total_packed
    for i in range(total_packed):
        out_packed[i] = UInt8(0)

    for g in range(n_groups):
        var start: Int = g * gs
        var end: Int = min(start + gs, n)

        var bounds = _bit_count(data, start, end)
        var lo: Float32 = bounds[0]
        var hi: Float32 = bounds[1]
        var range_v: Float32 = hi - lo
        var scale: Float32 = range_v / Float32(levels) if range_v > 0 else Float32(1.0)
        var zero: Float32 = lo
        out_scales[g] = scale
        out_zeros[g] = zero

        var bit_pos: Int = 0
        for i in range(start, end):
            var normalized: Float32 = (data[i] - zero) / scale
            var clamped: Float32 = max(Float32(0), min(Float32(levels), round(normalized)))
            var q: Int = Int(clamped)
            bit_pos = _write_bits(out_packed, g * per_group_packed_bytes, q, bits_i, bit_pos)


def decode_uniform[
    p_origin: MutOrigin,
    s_origin: MutOrigin,
    z_origin: MutOrigin,
    r_origin: MutOrigin,
](
    packed: UnsafePointer[UInt8, p_origin],
    scales: UnsafePointer[Float32, s_origin],
    zeros: UnsafePointer[Float32, z_origin],
    n: Int,
    group_size: Int,
    bits: UInt8,
    result: UnsafePointer[Float32, r_origin],
):
    var gs: Int = 64 if group_size == 0 else group_size
    var bits_i: Int = Int(bits)
    var per_group_packed_bytes: Int = (gs * bits_i + 7) // 8

    var g: Int = 0
    while g * gs < n:
        var start: Int = g * gs
        var end: Int = min(start + gs, n)
        var scale: Float32 = scales[g]
        var zero: Float32 = zeros[g]
        var bit_pos: Int = 0
        for i in range(start, end):
            var value: Int = _read_bits(packed, g * per_group_packed_bytes, bits_i, bit_pos)
            result[i] = zero + Float32(value) * scale
            bit_pos += bits_i
        g += 1


# ── C ABI exports (consumed by Rust `extern "C"` wrapper) ───────────────
#
# The Rust side computes n_groups / total_packed upfront, pre-allocates
# the output buffers, and passes the raw address of the data pointer as
# an Int (we reconstruct the UnsafePointer inside via the
# `unsafe_from_address=` constructor). All output slots are typed as
# `UnsafePointer[Int, ...]` so the caller can read pointer addresses
# back as Int values without per-element type confusion.
#
# Note: Mojo 1.0.0b3's @export emits an "abi() effect required" warning
# but still produces an `internal_linkage` symbol. The Rust wrapper
# gracefully falls back to its pure-Rust implementation when the
# symbols are not externally visible, so the crate is usable either way.


@export("tq_mojo_encode")
def tq_mojo_encode[
    d_origin: MutOrigin,
    s0: MutOrigin, s1: MutOrigin, s2: MutOrigin, s3: MutOrigin,
    s4: MutOrigin, s5: MutOrigin, s6: MutOrigin, s7: MutOrigin,
](
    data_addr: Int,
    n: Int,
    bits: UInt8,
    group_size: Int,
    shape_ptr_out: UnsafePointer[Int, s0],
    out_shape_len: UnsafePointer[Int, s1],
    packed_ptr_out: UnsafePointer[Int, s2],
    out_packed_len: UnsafePointer[Int, s3],
    scales_ptr_out: UnsafePointer[Int, s4],
    out_scales_len: UnsafePointer[Int, s5],
    zeros_ptr_out: UnsafePointer[Int, s6],
    out_zeros_len: UnsafePointer[Int, s7],
) -> Bool:
    var data_ptr = UnsafePointer[Float32, d_origin](unsafe_from_address=data_addr)

    var gs: Int = 64 if group_size == 0 else group_size
    var bits_i: Int = Int(bits)
    var n_groups: Int = (n + gs - 1) // gs
    var per_group_packed_bytes: Int = (gs * bits_i + 7) // 8
    var total_packed: Int = n_groups * per_group_packed_bytes

    var shape_ptr = alloc[Int](1)
    var packed_ptr = alloc[UInt8](total_packed)
    var scales_ptr = alloc[Float32](n_groups)
    var zeros_ptr = alloc[Float32](n_groups)

    encode_uniform(
        data_ptr, n, bits, group_size,
        packed_ptr, scales_ptr, zeros_ptr, out_packed_len,
    )

    shape_ptr[0] = n
    shape_ptr_out[0] = Int(shape_ptr)
    out_shape_len[0] = 1
    packed_ptr_out[0] = Int(packed_ptr)
    out_scales_len[0] = n_groups
    zeros_ptr_out[0] = Int(zeros_ptr)
    out_zeros_len[0] = n_groups
    return True


@export("tq_mojo_decode")
def tq_mojo_decode[
    p_origin: MutOrigin,
    s_origin: MutOrigin,
    z_origin: MutOrigin,
    r_origin: MutOrigin,
](
    packed_ptr: UnsafePointer[UInt8, p_origin],
    packed_len: Int,
    scales_ptr: UnsafePointer[Float32, s_origin],
    zeros_ptr: UnsafePointer[Float32, z_origin],
    n: Int,
    group_size: Int,
    bits: UInt8,
    result_ptr: UnsafePointer[Float32, r_origin],
) -> None:
    decode_uniform(
        packed_ptr, scales_ptr, zeros_ptr, n, group_size, bits, result_ptr,
    )


def main():
    print("mojo turbo_quant — compile with `mojo build mojo-src/turbo_quant.mojo -o libturbo_quant_mojo.a`")
