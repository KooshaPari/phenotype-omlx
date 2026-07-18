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

from std.memory import UnsafePointer, alloc

@value
struct QuantizedTensor:
    var shape_ptr: UnsafePointer[Int, MutUntrackedOrigin]
    var shape_len: Int
    var packed_ptr: UnsafePointer[UInt8, MutUntrackedOrigin]
    var packed_len: Int
    var scales_ptr: UnsafePointer[Float32, MutUntrackedOrigin]
    var scales_len: Int
    var zeros_ptr: UnsafePointer[Float32, MutUntrackedOrigin]
    var zeros_len: Int

def encode_uniform(data: UnsafePointer[Float32, MutUntrackedOrigin], n: Int, bits: UInt8, group_size: Int) -> QuantizedTensor:
    """Uniform group quantizer on `bits` levels. group_size=0 → default 64."""
    var gs: Int = 64 if group_size == 0 else group_size
    var levels: Int = (1 << bits.to_int()) - 1
    var n_groups: Int = (n + gs - 1) // gs
    var per_group_packed_bytes: Int = (gs * bits.to_int() + 7) // 8

    var shape = alloc[Int](1)
    shape.store(0, n)
    var packed = alloc[UInt8](n_groups * per_group_packed_bytes)
    var scales = alloc[Float32](n_groups)
    var zeros = alloc[Float32](n_groups)
    for i in range(n_groups * per_group_packed_bytes):
        packed.store(i, UInt8(0))

    for g in range(n_groups):
        var start: Int = g * gs
        var end: Int = min(start + gs, n)

        var lo: Float32 = data.load(start)[0]
        var hi: Float32 = data.load(start)[0]
        for i in range(start + 1, end):
            var v: Float32 = data.load(i)[0]
            if v < lo: lo = v
            if v > hi: hi = v
        var range: Float32 = hi - lo
        var scale: Float32 = range / Float32(levels) if range > 0 else Float32(1.0)
        var zero: Float32 = lo
        scales.store(g, scale)
        zeros.store(g, zero)

        var bit_pos: Int = 0
        for i in range(start, end):
            var normalized: Float32 = (data.load(i)[0] - zero) / scale
            var clamped: Float32 = max(Float32(0), min(Float32(levels), round(normalized)))
            var q: Int = clamped.to_int()
            # Pack `bits` bits of q into packed[g * per_group_packed_bytes..]
            var v: Int = q
            var bp: Int = bit_pos
            for b in range(bits.to_int()):
                var byte_idx: Int = bp // 8
                var bit_idx: Int = bp % 8
                var bit: UInt8 = UInt8((v >> (bits.to_int() - 1 - b)) & 1)
                var current = packed.load(g * per_group_packed_bytes + byte_idx)[0]
                current |= bit << UInt8(7 - bit_idx)
                packed.store(g * per_group_packed_bytes + byte_idx, current)
                bp += 1
            bit_pos += bits.to_int()

    return QuantizedTensor(
        shape_ptr=shape, shape_len=1,
        packed_ptr=packed, packed_len=n_groups * per_group_packed_bytes,
        scales_ptr=scales, scales_len=n_groups,
        zeros_ptr=zeros, zeros_len=n_groups,
    )

def decode_uniform(q: QuantizedTensor, n: Int, group_size: Int, bits: UInt8, result: UnsafePointer[Float32, MutUntrackedOrigin]):
    """Inverse of encode_uniform — writes f32 reconstructed values into `out`."""
    var gs: Int = 64 if group_size == 0 else group_size
    var levels_f: Float32 = Float32((1 << bits.to_int()) - 1)
    var per_group_packed_bytes: Int = (gs * bits.to_int() + 7) // 8

    var g: Int = 0
    while g * gs < n:
        var start: Int = g * gs
        var end: Int = min(start + gs, n)
        var scale: Float32 = q.scales_ptr.load(g)[0]
        var zero: Float32 = q.zeros_ptr.load(g)[0]
        var bit_pos: Int = 0
        for i in range(start, end):
            var value: Int = 0
            var bp: Int = bit_pos
            for b in range(bits.to_int()):
                var byte_idx: Int = bp // 8
                var bit_idx: Int = bp % 8
                var bit: Int = (q.packed_ptr.load(g * per_group_packed_bytes + byte_idx)[0] >> UInt8(7 - bit_idx)).to_int() & 1
                value = (value << 1) | bit
                bp += 1
            result.store(i, zero + Float32(value) * scale)
            bit_pos += bits.to_int()
        g += 1

# ── C ABI exports (consumed by Rust `extern "C"` wrapper) ───────────────

@export("tq_mojo_encode")
def tq_mojo_encode(
    data_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    bits: UInt8,
    group_size: Int,
    shape_result: UnsafePointer[UnsafePointer[Int, MutUntrackedOrigin], MutUntrackedOrigin],
    out_shape_len: UnsafePointer[Int],
    packed_result: UnsafePointer[UnsafePointer[UInt8, MutUntrackedOrigin], MutUntrackedOrigin],
    out_packed_len: UnsafePointer[Int],
    scales_result: UnsafePointer[UnsafePointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
    out_scales_len: UnsafePointer[Int],
    zeros_result: UnsafePointer[UnsafePointer[Float32, MutUntrackedOrigin], MutUntrackedOrigin],
    out_zeros_len: UnsafePointer[Int],
) abi("C") -> Bool:
    let q = encode_uniform(data_ptr, n, bits, group_size)
    shape_result.store(0, q.shape_ptr)
    out_shape_len[0] = q.shape_len
    packed_result.store(0, q.packed_ptr)
    out_packed_len[0] = q.packed_len
    scales_result.store(0, q.scales_ptr)
    out_scales_len[0] = q.scales_len
    zeros_result.store(0, q.zeros_ptr)
    out_zeros_len[0] = q.zeros_len
    return True

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
) abi("C"):
    let n_groups = (n + group_size - 1) // group_size
    let q = QuantizedTensor(
        shape_ptr=alloc[Int](1), shape_len=1,
        packed_ptr=packed_ptr, packed_len=packed_len,
        scales_ptr=scales_ptr, scales_len=n_groups,
        zeros_ptr=zeros_ptr, zeros_len=n_groups,
    )
    decode_uniform(q, n, group_size, bits, result_ptr)

def main():
    print("mojo turbo_quant — compile with `mojo build mojo-src/turbo_quant.mojo -o libturbo_quant_mojo.a`")
