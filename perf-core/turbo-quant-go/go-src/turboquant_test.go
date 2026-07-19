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
		t.Fatal("Expected non-empty packed data")
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

func TestEncodeRejectsInvalidArguments(t *testing.T) {
	data := []float32{0.1, -0.2, 0.3, -0.4}
	if _, err := Encode([]float32{}, 4, 4); err == nil {
		t.Fatal("expected empty data rejection")
	}
	if _, err := Encode(data, 1, 4); err == nil {
		t.Fatal("expected invalid bits rejection")
	}
	if _, err := Encode(data, 4, 0); err == nil {
		t.Fatal("expected invalid groupSize rejection")
	}
}
