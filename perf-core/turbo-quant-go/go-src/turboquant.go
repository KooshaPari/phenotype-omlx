package turboquant

/*
#cgo CFLAGS: -I${SRCDIR}/../../turbo-quant-c/c-src -I${SRCDIR}/../../native-abi/include
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -lturbo_quant_c
#include "turbo_quant.h"
#include <stdlib.h>
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
	packed := C.GoBytes(unsafe.Pointer(outPacked), C.int(outPackedLen))
	scales := C.GoBytes(unsafe.Pointer(outScales), C.int(outScalesLen))
	zeros := C.GoBytes(unsafe.Pointer(outZeros), C.int(outZerosLen))
	C.tq_c_free(unsafe.Pointer(outShape))
	C.tq_c_free(unsafe.Pointer(outPacked))
	C.tq_c_free(unsafe.Pointer(outScales))
	C.tq_c_free(unsafe.Pointer(outZeros))
	return &Tensor{
		Packed: packed,
		Scales: unsafe.Slice((*float32)(unsafe.Pointer(&scales[0])), len(scales)/4),
		Zeros:  unsafe.Slice((*float32)(unsafe.Pointer(&zeros[0])), len(zeros)/4),
	}, nil
}

func Decode(t *Tensor, n int, groupSize int, bits uint8) ([]float32, error) {
	if len(t.Packed) == 0 || len(t.Scales) == 0 || len(t.Zeros) == 0 {
		return nil, fmt.Errorf("empty tensor data")
	}
	outBuf := (*C.float)(C.malloc(C.size_t(n * 4)))
	if outBuf == nil {
		return nil, fmt.Errorf("malloc failed")
	}
	defer C.free(unsafe.Pointer(outBuf))
	C.tq_c_decode(
		(*C.uint8_t)(unsafe.Pointer(&t.Packed[0])), C.size_t(len(t.Packed)),
		(*C.float)(unsafe.Pointer(&t.Scales[0])),
		(*C.float)(unsafe.Pointer(&t.Zeros[0])),
		C.size_t(n), C.size_t(groupSize), C.uint8_t(bits),
		outBuf,
	)
	outBytes := C.GoBytes(unsafe.Pointer(outBuf), C.int(n*4))
	return unsafe.Slice((*float32)(unsafe.Pointer(&outBytes[0])), n), nil
}
