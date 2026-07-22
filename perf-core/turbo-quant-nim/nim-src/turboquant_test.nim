import std/[math, unittest]
import turboquant

const Sample = [
  -3.0'f32, -2.25'f32, -1.5'f32, -0.75'f32,
  0.0'f32, 0.75'f32, 1.5'f32, 2.25'f32,
  3.0'f32, 3.75'f32, 4.5'f32, 5.25'f32,
]

suite "turboquant Nim ABI wrapper":
  test "round trips copied tensor data":
    var input = @Sample
    let tensor = encode(input, 4'u8, 4)
    input[0] = 999.0'f32
    let decoded = decode(tensor, input.len, 4, 4'u8)

    check tensor.shape == @[12]
    check tensor.scales.len == 3
    check tensor.zeros.len == 3
    check tensor.packed.len == 6
    check decoded.len == input.len
    for index, value in Sample:
      check abs(decoded[index] - value) <= 0.2'f32

  test "rejects invalid encode arguments":
    expect ValueError:
      discard encode(@[], 4'u8, 4)
    expect ValueError:
      discard encode(@Sample, 0'u8, 4)
    expect ValueError:
      discard encode(@Sample, 9'u8, 4)
    expect ValueError:
      discard encode(@Sample, 4'u8, 0)

  test "rejects invalid decode arguments":
    let tensor = encode(@Sample, 4'u8, 4)
    expect ValueError:
      discard decode(tensor, 0, 4, 4'u8)
    expect ValueError:
      discard decode(tensor, Sample.len, 0, 4'u8)
    expect ValueError:
      discard decode(tensor, Sample.len, 4, 0'u8)

    var malformed = tensor
    malformed.packed.setLen(0)
    expect ValueError:
      discard decode(malformed, Sample.len, 4, 4'u8)

    malformed = tensor
    malformed.scales.setLen(1)
    expect ValueError:
      discard decode(malformed, Sample.len, 4, 4'u8)
