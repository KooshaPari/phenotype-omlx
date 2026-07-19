## turboquant — Nim wrapper for the TurboQuant C ABI.
##
## Statically links against libturbo_quant_c.a produced by:
##   cargo build --release -p turbo-quant-c

import std/[os, strutils]

{.passC: "-I" & currentSourcePath.parentDir() / "../.." / "turbo-quant-c/c-src".}
{.passC: "-I" & currentSourcePath.parentDir() / "../.." / "native-abi/include".}
{.passL: currentSourcePath.parentDir() / ".." / ".." / "target/release/deps" / "libturbo_quant_c.a".}

type
  QuantizedTensor* = object
    shape*: seq[int]
    packed*: seq[byte]
    scales*: seq[float32]
    zeros*: seq[float32]

proc cEncode(
    data: ptr float32, n: int, bits: uint8, groupSize: int,
    outShape: ptr ptr int, outShapeLen: ptr int,
    outPacked: ptr ptr byte, outPackedLen: ptr int,
    outScales: ptr ptr float32, outScalesLen: ptr int,
    outZeros: ptr ptr float32, outZerosLen: ptr int,
): bool {.importc: "tq_c_encode", cdecl.}

proc cDecode(
    packed: ptr byte, packedLen: int,
    scales: ptr float32, zeros: ptr float32,
    n: int, groupSize: int, bits: uint8, outPtr: ptr float32,
): void {.importc: "tq_c_decode", cdecl.}

proc cFree(p: pointer): void {.importc: "tq_c_free", cdecl.}

proc encode*(data: openArray[float32], bits: uint8, groupSize: int): QuantizedTensor =
  if data.len == 0:
    raise newException(ValueError, "data must be non-empty")
  if bits < 2 or bits > 4:
    raise newException(ValueError, "bits must be 2, 3, or 4 (got " & $bits & ")")
  if groupSize <= 0:
    raise newException(ValueError, "groupSize must be > 0")
  if data.len mod groupSize != 0:
    raise newException(ValueError,
      "data length " & $data.len & " must be a multiple of groupSize " & $groupSize)

  var
    outShape: ptr int
    outShapeLen: int
    outPacked: ptr byte
    outPackedLen: int
    outScales: ptr float32
    outScalesLen: int
    outZeros: ptr float32
    outZerosLen: int

  let ok = cEncode(
    cast[ptr float32](unsafeAddr data[0]), data.len, bits, groupSize,
    addr outShape, addr outShapeLen,
    addr outPacked, addr outPackedLen,
    addr outScales, addr outScalesLen,
    addr outZeros, addr outZerosLen,
  )
  if not ok:
    raise newException(ValueError, "tq_c_encode returned false")

  result = QuantizedTensor()
  if outShapeLen > 0 and not outShape.isNil:
    result.shape.setLen(outShapeLen)
    copyMem(addr result.shape[0], outShape, outShapeLen * sizeof(int))
  if outPackedLen > 0 and not outPacked.isNil:
    result.packed.setLen(outPackedLen)
    copyMem(addr result.packed[0], outPacked, outPackedLen)
  if outScalesLen > 0 and not outScales.isNil:
    result.scales.setLen(outScalesLen)
    copyMem(addr result.scales[0], outScales, outScalesLen * sizeof(float32))
  if outZerosLen > 0 and not outZeros.isNil:
    result.zeros.setLen(outZerosLen)
    copyMem(addr result.zeros[0], outZeros, outZerosLen * sizeof(float32))

  if not outShape.isNil: cFree(cast[pointer](outShape))
  if not outPacked.isNil: cFree(cast[pointer](outPacked))
  if not outScales.isNil: cFree(cast[pointer](outScales))
  if not outZeros.isNil: cFree(cast[pointer](outZeros))


proc decode*(t: QuantizedTensor, n: int, groupSize: int, bits: uint8): seq[float32] =
  if n == 0:
    raise newException(ValueError, "n must be > 0")
  if groupSize <= 0:
    raise newException(ValueError, "groupSize must be > 0")
  if bits < 2 or bits > 4:
    raise newException(ValueError, "bits must be 2, 3, or 4 (got " & $bits & ")")
  let expectedGroups = n div groupSize
  if t.scales.len != expectedGroups or t.zeros.len != expectedGroups:
    raise newException(ValueError,
      "scales (" & $t.scales.len & ") and zeros (" & $t.zeros.len &
      ") must each have " & $expectedGroups & " entries")
  let expectedPacked = (n * int(bits) + 7) div 8
  if t.packed.len != expectedPacked:
    raise newException(ValueError,
      "packed length " & $t.packed.len & " does not match expected " & $expectedPacked)

  result.setLen(n)
  cDecode(
    cast[ptr byte](unsafeAddr t.packed[0]), t.packed.len,
    cast[ptr float32](unsafeAddr t.scales[0]),
    cast[ptr float32](unsafeAddr t.zeros[0]),
    n, groupSize, bits, cast[ptr float32](addr result[0]),
  )
