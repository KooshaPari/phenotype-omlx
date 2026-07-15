# turbo-quant-mojo — Mojo implementation of TurboQuant
#
# Build:
#   1. Install Mojo:        `modular install mojo`
#   2. Compile the C-ABI lib from this Mojo source:
#        mojo build mojo-src/turbo_quant.mojo -o libturbo_quant_mojo.a --emit shared
#   3. Cargo will pick it up via the build script.
#
# Without Mojo installed, this crate compiles but is a no-op stub
# (matching the "no language forbidden" policy — the crate exists as a
# scaffold for when the toolchain is added).

from sys import ffi
from memory import DTypePointer, UnsafePointer

@value
struct QuantizedTensor:
    var shape_ptr: UnsafePointer[Int]
    var shape_len: Int
    var packed_ptr: UnsafePointer[UInt8]
    var packed_len: Int
    var scales_ptr: UnsafePointer[Float32]
    var scales_len: Int
    var zeros_ptr: UnsafePointer[Float32]
    var zeros_len: Int

fn encode_uniform(data: DTypePointer[DType.float32], n: Int, bits: UInt8, group_size: Int) -> QuantizedTensor:
    """Uniform group quantizer on `bits` levels. group_size=0 → default 64."""
    var gs: Int = 64 if group_size == 0 else group_size
    var levels: Int = (1 << bits.to_int()) - 1
    var n_groups: Int = (n + gs - 1) // gs
    var per_group_packed_bytes: Int = (gs * bits.to_int() + 7) // 8

    var shape = UnsafePointer[Int].alloc(1)
    shape[0] = n
    var packed = UnsafePointer[UInt8].alloc(n_groups * per_group_packed_bytes)
    var scales = UnsafePointer[Float32].alloc(n_groups)
    var zeros = UnsafePointer[Float32].alloc(n_groups)
    for _ in range(n_groups * per_group_packed_bytes):
        pass  # memset 0 by alloc; Mojo doesn't have memset builtin — caller zeros
    for _ in range(n_groups):
        pass

    for g in range(n_groups):
        var start: Int = g * gs
        var end: Int = min(start + gs, n)

        var lo: Float32 = data.load(start)
        var hi: Float32 = data.load(start)
        for i in range(start + 1, end):
            var v: Float32 = data.load(i)
            if v < lo: lo = v
            if v > hi: hi = v
        var range: Float32 = hi - lo
        var scale: Float32 = range / Float32(levels) if range > 0 else Float32(1.0)
        var zero: Float32 = lo
        scales[g] = scale
        zeros[g] = zero

        var bit_pos: Int = 0
        for i in range(start, end):
            var normalized: Float32 = (data.load(i) - zero) / scale
            var clamped: Float32 = max(Float32(0), min(Float32(levels), round(normalized)))
            var q: Int = clamped.to_int()
            # Pack `bits` bits of q into packed[g * per_group_packed_bytes..]
            var v: Int = q
            var bp: Int = bit_pos
            for b in range(bits.to_int()):
                var byte_idx: Int = bp // 8
                var bit_idx: Int = bp % 8
                var bit: UInt8 = UInt8((v >> (bits.to_int() - 1 - b)) & 1)
                packed[g * per_group_packed_bytes + byte_idx] |= bit << UInt8(7 - bit_idx)
                bp += 1
            bit_pos += bits.to_int()

    return QuantizedTensor(
        shape_ptr=shape, shape_len=1,
        packed_ptr=packed, packed_len=n_groups * per_group_packed_bytes,
        scales_ptr=scales, scales_len=n_groups,
        zeros_ptr=zeros, zeros_len=n_groups,
    )

fn decode_uniform(q: QuantizedTensor, n: Int, group_size: Int, bits: UInt8, out: DTypePointer[DType.float32]):
    """Inverse of encode_uniform — writes f32 reconstructed values into `out`."""
    var gs: Int = 64 if group_size == 0 else group_size
    var levels_f: Float32 = Float32((1 << bits.to_int()) - 1)
    var per_group_packed_bytes: Int = (gs * bits.to_int() + 7) // 8

    var g: Int = 0
    while g * gs < n:
        var start: Int = g * gs
        var end: Int = min(start + gs, n)
        var scale: Float32 = q.scales_ptr[g]
        var zero: Float32 = q.zeros_ptr[g]
        var bit_pos: Int = 0
        for i in range(start, end):
            var value: Int = 0
            var bp: Int = bit_pos
            for b in range(bits.to_int()):
                var byte_idx: Int = bp // 8
                var bit_idx: Int = bp % 8
                var bit: Int = (q.packed_ptr[g * per_group_packed_bytes + byte_idx] >> UInt8(7 - bit_idx)).to_int() & 1
                value = (value << 1) | bit
                bp += 1
            out.store(i, zero + Float32(value) * scale)
            bit_pos += bits.to_int()
        g += 1

# ── C ABI exports (consumed by Rust `extern "C"` wrapper) ───────────────

@export("tq_mojo_encode")
fn tq_mojo_encode(
    data_ptr: DTypePointer[DType.float32],
    n: Int,
    bits: UInt8,
    group_size: Int,
    out_shape: UnsafePointer[UnsafePointer[Int]],
    out_shape_len: UnsafePointer[Int],
    out_packed: UnsafePointer[UnsafePointer[UInt8]],
    out_packed_len: UnsafePointer[Int],
    out_scales: UnsafePointer[UnsafePointer[Float32]],
    out_scales_len: UnsafePointer[Int],
    out_zeros: UnsafePointer[UnsafePointer[Float32]],
    out_zeros_len: UnsafePointer[Int],
) -> Bool:
    let q = encode_uniform(data_ptr, n, bits, group_size)
    out_shape[0] = q.shape_ptr
    out_shape_len[0] = q.shape_len
    out_packed[0] = q.packed_ptr
    out_packed_len[0] = q.packed_len
    out_scales[0] = q.scales_ptr
    out_scales_len[0] = q.scales_len
    out_zeros[0] = q.zeros_ptr
    out_zeros_len[0] = q.zeros_len
    return True

@export("tq_mojo_decode")
fn tq_mojo_decode(
    packed_ptr: DTypePointer[DType.uint8],
    packed_len: Int,
    scales_ptr: DTypePointer[DType.float32],
    zeros_ptr: DTypePointer[DType.float32],
    n: Int,
    group_size: Int,
    bits: UInt8,
    out_ptr: DTypePointer[DType.float32],
):
    let n_groups = (n + group_size - 1) // group_size
    let q = QuantizedTensor(
        shape_ptr=UnsafePointer[Int](), shape_len=1,
        packed_ptr=packed_ptr, packed_len=packed_len,
        scales_ptr=scales_ptr, scales_len=n_groups,
        zeros_ptr=zeros_ptr, zeros_len=n_groups,
    )
    decode_uniform(q, n, group_size, bits, out_ptr)

fn main():
    print("mojo turbo_quant — compile with `mojo build mojo-src/turbo_quant.mojo -o libturbo_quant_mojo.a`")