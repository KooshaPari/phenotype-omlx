{.push raises: [].}
import std/strutils

const LibPath =
  when defined(macos): "libturbo_quant_c.dylib"
  elif defined(windows): "turbo_quant_c.dll"
  else: "libturbo_quant_c.so"

type Tensor* = object
  shape*: seq[int]
  packed*: seq[byte]
  scales*: seq[float32]
  zeros*: seq[float32]

proc tq_c_encode(data: ptr float32; n: cint; bits: uint8; groupSize: cint;
    outShape: ptr ptr cint; outShapeLen: ptr cint;
    outPacked: ptr ptr uint8; outPackedLen: ptr cint;
    outScales: ptr ptr float32; outScalesLen: ptr cint;
    outZeros: ptr ptr float32; outZerosLen: ptr cint): bool {.cdecl, dynlib: LibPath, importc: "tq_c_encode".}

proc tq_c_decode(packed: ptr uint8; packedLen: cint;
    scales, zeros: ptr float32; n, groupSize: cint; bits: uint8;
    outBuf: ptr float32) {.cdecl, dynlib: LibPath, importc: "tq_c_decode".}

proc tq_c_free(p: pointer) {.cdecl, dynlib: LibPath, importc: "tq_c_free".}

proc encode*(data: seq[float32]; bits: uint8; groupSize = 64): Tensor =
  var shape, packed, scales, zeros: pointer = nil
  var shapeLen, packedLen, scalesLen, zerosLen: cint = 0
  let ok = tq_c_encode(addr data[0], cint(data.len), bits, cint(groupSize),
    addr shape, addr shapeLen, addr packed, addr packedLen,
    addr scales, addr scalesLen, addr zeros, addr zerosLen)
  if not ok: raise newException(ValueError, "encode failed")
  result.shape = @[]
  for i in 0..<shapeLen: result.shape.add(cast[ptr cint](shape)[])
  result.packed = @[]
  for i in 0..<packedLen: result.packed.add(cast[ptr uint8](packed)[])
  result.scales = @[]
  for i in 0..<scalesLen: result.scales.add(cast[ptr float32](scales)[])
  result.zeros = @[]
  for i in 0..<zerosLen: result.zeros.add(cast[ptr float32](zeros)[])
  tq_c_free(shape); tq_c_free(packed); tq_c_free(scales); tq_c_free(zeros)

proc decode*(t: Tensor; n, groupSize: int; bits: uint8): seq[float32] =
  result = newSeq[float32](n)
  tq_c_decode(addr t.packed[0], cint(t.packed.len),
    addr t.scales[0], addr t.zeros[0],
    cint(n), cint(groupSize), bits, addr result[0])
