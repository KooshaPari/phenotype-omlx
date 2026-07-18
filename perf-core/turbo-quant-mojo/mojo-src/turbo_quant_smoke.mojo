from std.memory import alloc

from turbo_quant import decode_uniform, encode_uniform


def main():
    var n: Int = 8
    var group_size: Int = 4
    var bits = UInt8(4)
    var data = alloc[Float32](n)
    for i in range(n):
        data.store(i, Float32(i) - Float32(3.5))

    var quantized = encode_uniform(data, n, bits, group_size)
    var decoded = alloc[Float32](n)
    decode_uniform(quantized, n, group_size, bits, decoded)

    var max_error = Float32(0)
    for i in range(n):
        var error = abs(decoded.load(i)[0] - data.load(i)[0])
        if error > max_error:
            max_error = error
    assert(max_error <= Float32(0.2), "quantized round-trip error exceeded bound")

    data.free()
    decoded.free()
    quantized.shape_ptr.free()
    quantized.packed_ptr.free()
    quantized.scales_ptr.free()
    quantized.zeros_ptr.free()
