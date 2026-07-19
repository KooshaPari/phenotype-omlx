package turboquant

/*
#cgo CFLAGS: -I${SRCDIR}/../../turbo-quant-c/c-src -I${SRCDIR}/../../native-abi/include
#cgo LDFLAGS: ${SRCDIR}/../../target/release/deps/libturbo_quant_c.a
#include "turbo_quant.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type Tensor struct {
	Shape  []int
	Packed []byte
	Scales []float32
	Zeros  []float32
}

func Encode(data []float32, bits uint8, groupSize int) (*Tensor, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty data")
	}
	if bits < 2 || bits > 4 {
		return nil, fmt.Errorf("bits must be 2, 3, or 4 (got %d)", bits)
	}
	if groupSize <= 0 {
		return nil, fmt.Errorf("groupSize must be > 0")
	}
	if len(data)%groupSize != 0 {
		return nil, fmt.Errorf("data length %d must be a multiple of groupSize %d",
			len(data), groupSize)
	}
	var outShape *C.size_t
	var outShapeLen C.size_t
	var outPacked *C.uint8_t
	var outPackedLen C.size_t
	var outScales *C.float
	var outScalesLen C.size_t
	var outZeros *C.float
	var outZerosLen C.size_t
	ok := C.tq_c_encode(
		(*C.float)(unsafe.Pointer(&data[0])), C.size_t(len(data)),
		C.uint8_t(bits), C.size_t(groupSize),
		&outShape, &outShapeLen,
		&outPacked, &outPackedLen,
		&outScales, &outScalesLen,
		&outZeros, &outZerosLen,
	)
	if !bool(ok) {
		return nil, fmt.Errorf("tq_c_encode failed")
	}

	t := &Tensor{
		Packed: C.GoBytes(unsafe.Pointer(outPacked), C.int(outPackedLen)),
		Scales: copyFloat32Slice(outScales, int(outScalesLen)),
		Zeros:  copyFloat32Slice(outZeros, int(outZerosLen)),
	}
	if outShapeLen > 0 && outShape != nil {
		t.Shape = make([]int, int(outShapeLen))
		for i := range t.Shape {
			t.Shape[i] = int(*(*C.size_t)(unsafe.Pointer(
				uintptr(unsafe.Pointer(outShape)) + uintptr(i)*unsafe.Sizeof(*outShape))))
		}
	}

	C.tq_c_free(unsafe.Pointer(outShape))
	C.tq_c_free(unsafe.Pointer(outPacked))
	C.tq_c_free(unsafe.Pointer(outScales))
	C.tq_c_free(unsafe.Pointer(outZeros))
	return t, nil
}

func copyFloat32Slice(src *C.float, n int) []float32 {
	if n <= 0 || src == nil {
		return nil
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = float32(*(*C.float)(unsafe.Pointer(
			uintptr(unsafe.Pointer(src)) + uintptr(i)*unsafe.Sizeof(*src))))
	}
	return out
}

func Decode(t *Tensor, n int, groupSize int, bits uint8) ([]float32, error) {
	if n == 0 {
		return nil, fmt.Errorf("n must be > 0")
	}
	if groupSize <= 0 {
		return nil, fmt.Errorf("groupSize must be > 0")
	}
	if bits < 2 || bits > 4 {
		return nil, fmt.Errorf("bits must be 2, 3, or 4 (got %d)", bits)
	}
	expectedGroups := n / groupSize
	if len(t.Scales) != expectedGroups || len(t.Zeros) != expectedGroups {
		return nil, fmt.Errorf("scales (%d) and zeros (%d) must each have %d entries",
			len(t.Scales), len(t.Zeros), expectedGroups)
	}
	expectedPacked := (n*int(bits) + 7) / 8
	if len(t.Packed) != expectedPacked {
		return nil, fmt.Errorf("packed length %d does not match expected %d",
			len(t.Packed), expectedPacked)
	}
	out := make([]float32, n)
	C.tq_c_decode(
		(*C.uint8_t)(unsafe.Pointer(&t.Packed[0])), C.size_t(len(t.Packed)),
		(*C.float)(unsafe.Pointer(&t.Scales[0])),
		(*C.float)(unsafe.Pointer(&t.Zeros[0])),
		C.size_t(n), C.size_t(groupSize), C.uint8_t(bits),
		(*C.float)(unsafe.Pointer(&out[0])),
	)
	return out, nil
}
