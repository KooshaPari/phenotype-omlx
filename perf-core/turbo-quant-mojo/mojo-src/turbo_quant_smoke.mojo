from std.memory import alloc

from turbo_quant import decode_uniform, encode_uniform


def main() raises:
    var n: Int = 8
    var group_size: Int = 4
    var bits = UInt8(4)
    var data = alloc[Float32](n)
    for i in range(n):
        data[i] = Float32(i) - Float32(3.5)

    var per_group = (group_size * Int(bits) + 7) // 8
    var n_groups = (n + group_size - 1) // group_size
    var packed = alloc[UInt8](n_groups * per_group)
    var scales = alloc[Float32](n_groups)
    var zeros = alloc[Float32](n_groups)
    var packed_len = alloc[Int](1)

    encode_uniform(data, n, bits, group_size, packed, scales, zeros, packed_len)
    var decoded = alloc[Float32](n)
    decode_uniform(packed, scales, zeros, n, group_size, bits, decoded)

    var max_error = Float32(0)
    for i in range(n):
        var error = abs(decoded[i] - data[i])
        if error > max_error:
            max_error = error
    debug_assert(max_error <= Float32(0.2))

    data.free()
    decoded.free()
    packed.free()
    scales.free()
    zeros.free()
    packed_len.free()
