package turboquant

import (
	"math"
	"testing"
)

func TestEncodeDecodeRoundtrip(t *testing.T) {
	data := []float32{0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8}
	tensor, err := Encode(data, 4, 8)
	if err != nil {
		t.Fatalf("Failed to encode: %v", err)
	}
	if len(tensor.Packed) == 0 {
		t.Error("Expected non-empty packed data")
	}

	decoded, err := Decode(tensor, len(data), 8, 4)
	if err != nil {
		t.Fatalf("Failed to decode: %v", err)
	}

	for i, v := range data {
		diff := float64(math.Abs(float64(v - decoded[i])))
		if diff > 0.15 {
			t.Errorf("Index %d: expected value close to %f, got %f (diff: %f)", i, v, decoded[i], diff)
		}
	}
}
